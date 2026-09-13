"""Hard-surface renderer for a level set: first hit by sphere tracing, then shading.

The body is {x : phi(x) <= 0} for a callable phi that must be a true signed distance
function, |grad phi| = 1. Sphere tracing relies on that: stepping by phi(x) cannot cross the
surface, because phi is the distance to it.

    t <- t + phi(o + t d)        until phi < eps

Bisection is not used. It requires a bracket in which phi changes sign, and a ray that enters
and exits the body has phi < 0 only on an interval, so bisecting on "is the midpoint inside"
diverges whenever the first midpoint misses the body.

Derivatives do not pass through the marching loop. The hit satisfies phi(o + t* d) = 0
identically, so for any parameter theta

    dt*/dtheta = - (dphi/dtheta) / (grad phi . d)

evaluated at the hit (implicit function theorem).

Scope: the silhouette is a step function of the shape, so the derivative of the lit area is
zero almost everywhere and undefined on the outline. This model gives no gradient through the
outline. It also models single-bounce shading only: no interreflection, no sensor chain.
"""
from __future__ import annotations

import torch

from .shared.common import pixel_rays, reduce_frame

__all__ = ["sphere_trace", "surface_normal", "render", "sphere_sdf", "pit_sdf"]


def sphere_trace(o: torch.Tensor, d: torch.Tensor, phi, tmax: float = 10.0,
                 iters: int = 200, eps: float = 1e-6):
    """First hit of {phi <= 0} along o + t d. Returns (t, hit)."""
    t = torch.zeros(o.shape[:-1], device=o.device, dtype=o.dtype)
    for _ in range(iters):
        s = phi(o + t[..., None] * d)
        t = torch.where((s > eps) & (t < tmax), t + s.clamp_min(eps * 0.5), t)
    return t, t < tmax


def surface_normal(x: torch.Tensor, phi, h: float = 1e-4) -> torch.Tensor:
    """Central differences on phi. For a true SDF the result is already unit length."""
    e = torch.eye(3, device=x.device, dtype=x.dtype) * h
    g = torch.stack([phi(x + e[i]) - phi(x - e[i]) for i in range(3)], dim=-1) / (2 * h)
    return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def render(phi, view: torch.Tensor, sun: torch.Tensor, res: int = 96,
           extent: float = 1.6, rho: float = 0.85, tau_i: float = 0.0,
           tau_b: float = 0.02, shadow: bool = True):
    """One frame: (intensity, lit area).

    Occlusion of the view comes from taking the first hit. Occlusion of the light is a second
    trace, from just above each hit point towards the source. A convex body blocks neither.
    """
    o, d, px = pixel_rays(view, extent, res)
    t, hit = sphere_trace(o, d, phi, tmax=4.0 * extent)
    x = o + t[..., None] * d
    n = surface_normal(x, phi)
    mu0 = (n * sun).sum(-1)
    lit = hit & (mu0 > 0)
    val = torch.zeros_like(mu0)
    if lit.any():
        v = mu0[lit] * (rho / torch.pi)
        if shadow:
            xs = x[lit] + n[lit] * 1e-3
            ts, blocked = sphere_trace(xs, sun.expand_as(xs), phi, tmax=4.0 * extent)
            v = torch.where(blocked, torch.zeros_like(v), v)
        val[lit] = v
    return reduce_frame(val, px, tau_i, tau_b)


# ----------------------------------------------------------------------------------
# Two analytic bodies. Both are exact SDFs.

def sphere_sdf(radius: float, centre=(0.0, 0.0, 0.0)):
    c = torch.tensor(centre, dtype=torch.float32)
    return lambda x: (x - c.to(x.device)).norm(dim=-1) - radius


def pit_sdf(radius: float, pit_r: float, depth: float, axis=(0.0, 0.0, 1.0)):
    """A sphere with a flat-floored cylindrical pit sunk into it along `axis`.

    Exact SDF, used as a non-convex reference body: the pit wall casts a shadow on the floor
    at oblique illumination and none when the source lies along the pit axis.
    """
    a = torch.tensor(axis, dtype=torch.float32)
    a = a / a.norm()

    def phi(x):
        ax = a.to(x.device)
        ball = x.norm(dim=-1) - radius
        z = (x * ax).sum(-1)                       # height along the pit axis
        r = (x - z[..., None] * ax).norm(dim=-1)   # radial distance from the axis
        # cylinder occupying z in [radius - depth, +inf), r <= pit_r
        d_r = r - pit_r
        d_z = (radius - depth) - z
        cyl = torch.maximum(d_r, d_z)
        return torch.maximum(ball, -cyl)           # sphere minus cylinder

    return phi
