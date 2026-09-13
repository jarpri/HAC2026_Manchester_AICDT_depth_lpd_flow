"""Dice between convex bodies from their radial functions, differentiable in the support
function, so the challenge metric can serve as a training loss without a voxel grid or a
mesh.

A convex body with support values h_n on normals u_n is the half-space intersection
K = {x : <x, u_n> <= h_n}. Along the ray t*v (t >= 0) only the constraints with
<v, u_n> > 0 can bind, so the boundary is at

    rho(v) = min over n with <v,u_n> > 0 of  h_n / <v,u_n>                          (1)

the radial function of K. For two convex bodies containing the origin the intersection has
radial function min(rho_A, rho_B), and volume is (1/3) int rho^3 dw, so

    Dice = 2 |A n B| / (|A| + |B|)
         = 2 int min(rho_A,rho_B)^3 dw / ( int rho_A^3 dw + int rho_B^3 dw ).       (2)

Both are compositions of min, divide and power, so the gradient with respect to h reaches
only the constraint that bounds the body in each direction. The one approximation is the
quadrature over the sphere, a Fibonacci lattice with equal weights; the bodies must be
convex and contain the origin.
"""
from __future__ import annotations

import numpy as np


def fibonacci_sphere(n: int) -> np.ndarray:
    """`n` nearly equal-area directions on the sphere (Fibonacci lattice), shape (n, 3).
    Equal areas mean equal quadrature weights, so every integral below is a plain sum."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def support_ray_matrix(normals: np.ndarray, rays: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    """M[v, n] = max(<rays_v, normals_n>, 0): the coefficients of the constraints that can
    bind along each ray. Non-binding entries are 0 rather than masked, so the reciprocal
    form in radial_from_support needs no infinite sentinel: a zero never wins the max."""
    return np.maximum(rays @ normals.T, 0.0) * (np.abs(rays @ normals.T) > eps)


def radial_from_support(h: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Equation (1) in reciprocal form: rho(v) = 1 / max_n ( M[v,n] / h_n )."""
    return 1.0 / np.max(M * (1.0 / h)[None, :], axis=1)


def dice_from_radial(ra: np.ndarray, rb: np.ndarray) -> float:
    """Equation (2). Exact for convex bodies containing the origin."""
    inter = np.minimum(ra, rb) ** 3
    return float(2.0 * inter.sum() / ((ra ** 3).sum() + (rb ** 3).sum()))


def torch_radial(h, M):
    """Batched (1) in reciprocal form: h (B, N), M (V, N) -> (B, V). The gradient of the
    max reaches only the binding constraint of each ray."""
    return 1.0 / (M[None, :, :] * (1.0 / h)[:, None, :]).max(dim=2).values


def torch_dice(rho_a, rho_b, eps: float = 1e-8):
    """Batched (2) -> (B,). Differentiable in both arguments."""
    inter = torch.minimum(rho_a, rho_b) ** 3
    return 2.0 * inter.sum(1) / (rho_a.pow(3).sum(1) + rho_b.pow(3).sum(1) + eps)


def torch_dice_loss(h_pred, rho_true, M, chunk: int = 0):
    """1 - Dice(body(h_pred), true body), averaged over the batch. `rho_true` (B, V) is the
    true body's radial function on the rays of M, precomputed since it needs no gradient.
    Set `chunk` to split the ray axis when (B, V, N) does not fit in memory."""
    if not chunk:
        return (1.0 - torch_dice(torch_radial(h_pred, M), rho_true)).mean()
    num = 0.0
    den_a = 0.0
    den_b = 0.0
    for i in range(0, M.shape[0], chunk):
        ra = torch_radial(h_pred, M[i:i + chunk])
        rb = rho_true[:, i:i + chunk]
        num = num + torch.minimum(ra, rb).pow(3).sum(1)
        den_a = den_a + ra.pow(3).sum(1)
        den_b = den_b + rb.pow(3).sum(1)
    return (1.0 - 2.0 * num / (den_a + den_b + 1e-8)).mean()


try:  # torch is optional for the numpy-only paths above
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ---------------- radial function of a mesh --------------------------------------------
def mesh_radial(verts: np.ndarray, faces: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Radial function of a convex mesh containing the origin, on `rays`. Uses (1) with the
    mesh's own facet planes, so the result describes the mesh itself rather than its
    support values on a normal grid."""
    from .shapes import face_normals_areas

    n, a = face_normals_areas(verts, faces)
    keep = a > 1e-14
    n = n[keep]
    d = (n * verts[faces[keep, 0]]).sum(1)
    d = np.maximum(d, 1e-9)              # origin must be strictly inside
    return radial_from_support(d, support_ray_matrix(n, rays))
