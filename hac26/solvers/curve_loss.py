"""Differentiable side-view curve loss for the convex stage.

For a convex body K, the projection along v is convex and its support function within the
plane is h_K restricted to the great circle perpendicular to v:

    h_{proj_v K}(u) = h_K(u)      for unit u with u . v = 0

So the distance between two convex outlines is an integral along that circle,

    D(v) = (1/2pi) INT_{u perp v} | h_A(u) - h_B(u) | du

which needs no rasterising, no contour tracing and no nearest-neighbour search, and is
differentiable in h.

hac26.scoring.side_view measures the same quantity for arbitrary bodies by rasterising silhouettes,
which is necessary because a non-convex outline is not a support function, and is not
differentiable. This is its differentiable counterpart, exact for a convex body.

Scope: h_K = h_{conv K}, so this term is blind to concavity. It trains the convex stage to
place the hull; the concavity left in an outline is the non-convex stage's business.

The in-plane quadrature is uniform because it integrates a periodic function around a closed
circle, where the uniform rule converges faster than any power of the step. That differs from
the choice of viewing directions, which are sampled rather than integrated.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["great_circle_directions", "interpolate_support", "mesh_support_torch",
           "curve_loss"]


def great_circle_directions(view: np.ndarray, n_ang: int = 64) -> np.ndarray:
    """n_ang unit vectors spanning the great circle perpendicular to `view`."""
    v = np.asarray(view, dtype=float)
    v = v / np.linalg.norm(v)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(v @ up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(up, v); e1 /= np.linalg.norm(e1)
    e2 = np.cross(v, e1); e2 /= np.linalg.norm(e2)
    a = 2.0 * np.pi * np.arange(n_ang) / n_ang
    return np.cos(a)[:, None] * e1[None, :] + np.sin(a)[:, None] * e2[None, :]


def interpolate_support(u: torch.Tensor, normals: torch.Tensor, h: torch.Tensor,
                        kappa: float = 300.0) -> torch.Tensor:
    """h evaluated at arbitrary directions, by a von Mises-Fisher kernel over the design.

    softmax(kappa * u . n_i) weights the stored support values. Smooth and differentiable in
    h. This is not the polytope's own support function: between normals a vertex protrudes, by
    about 2/N of the support distance for a design of N normals, so the design must be dense
    enough that this sits below the resolution being measured. kappa should rise with N so the
    kernel stays narrower than the spacing between normals.
    """
    w = torch.softmax(kappa * (u @ normals.T), dim=-1)          # (M, N)
    return w @ h                                                # (M,)


def mesh_support_torch(verts: torch.Tensor, u: torch.Tensor,
                       chunk: int = 4096) -> torch.Tensor:
    """Exact support function of a mesh, h(u) = max_v (v . u). No interpolation."""
    outs = []
    for i in range(0, u.shape[0], chunk):
        outs.append((verts @ u[i:i + chunk].T).amax(dim=0))
    return torch.cat(outs)


def curve_loss(h_pred: torch.Tensor, normals: torch.Tensor, target_verts: torch.Tensor,
               views: np.ndarray, n_ang: int = 64, kappa: float = 300.0,
               reduction: str = "mean") -> torch.Tensor:
    """Mean over views of the integral difference between the two outlines.

    h_pred is the predicted support on `normals`; target_verts are the true body's vertices,
    posed in the same frame. `views` are the projection directions -- pass
    hac26.scoring.side_view.projection_directions(n, "side") so training and scoring look from the
    same places.

    The target support is exact from the mesh; only the prediction is interpolated, so the
    interpolation error enters once rather than twice.
    """
    dev, dt = h_pred.device, h_pred.dtype
    per_view = []
    for v in np.asarray(views):
        u = torch.tensor(great_circle_directions(v, n_ang), device=dev, dtype=dt)
        hp = interpolate_support(u, normals, h_pred, kappa)
        ht = mesh_support_torch(target_verts, u)
        per_view.append((hp - ht).abs().mean())                 # (1/2pi) INT |dh| du
    d = torch.stack(per_view)
    if reduction == "worst":
        return d.max()
    if reduction == "none":
        return d
    return d.mean()
