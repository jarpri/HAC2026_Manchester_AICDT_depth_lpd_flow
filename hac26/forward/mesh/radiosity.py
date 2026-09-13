"""Radiosity on a triangle mesh: interreflection between facets, solved once per shape.

The solve is done in the body frame, where the geometry between facets never changes as the
turntable turns. So the form-factor matrix F and the factorisation of (I - rho F) are
computed once per shape, and every phase is one more right-hand side.

    F_ij = (1/A_i) int int V(x,y) cos(theta_x) cos(theta_y) / (pi r^2) dA_i dA_j
    A_i F_ij = A_j F_ji                          (reciprocity)
    (I - rho F) B = rho e(psi)
    e_i(psi) = irradiance of facet i from the source at phase psi, per unit area
    L_i = B_i / pi                               (radiance leaving facet i)

Only the emission e depends on the phase. The solver is a torch linear solve, so B is
differentiable in e and in rho; F is treated as a constant of the mesh.

Two checks: F is made reciprocal by symmetrising A_i F_ij, since the quadrature does not give
that exactly; and every row sum of F must be at most 1, since a facet cannot see more than
its whole hemisphere. A row sum above 1 means the quadrature has broken down on that mesh.
The solver either raises RadiosityError or, when asked, rescales the offending rows to sum to
1 and reports how many it touched.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["facet_geometry", "form_factors", "RadiositySolver", "RadiosityError"]


class RadiosityError(ValueError):
    """The form factors of this mesh are not usable."""


def facet_geometry(verts: np.ndarray, faces: np.ndarray):
    """Centroids, unit normals and areas of every facet, degenerate facets dropped."""
    tv = verts[faces]
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    a2 = np.linalg.norm(n, axis=1)
    keep = a2 > 1e-14
    tv, n, a2 = tv[keep], n[keep], a2[keep]
    return tv.mean(1), n / a2[:, None], 0.5 * a2


def _visibility_matrix(centroids: np.ndarray, normals: np.ndarray,
                       verts: np.ndarray, faces: np.ndarray,
                       backface_only: bool = False) -> np.ndarray:
    """V(i,j): 1 if facet centroids i and j see each other, else 0. Pairs that do not face
    each other are excluded first, and only the rest are ray-tested."""
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    n_f = len(centroids)
    d = centroids[None, :, :] - centroids[:, None, :]
    r = np.linalg.norm(d, axis=2)
    np.fill_diagonal(r, np.inf)
    u = d / np.maximum(r, 1e-12)[:, :, None]
    cos_i = (u * normals[:, None, :]).sum(2)
    cos_j = -(u * normals[None, :, :]).sum(2)
    facing = (cos_i > 1e-9) & (cos_j > 1e-9)
    if backface_only:
        return facing.astype(np.float64)
    ii, jj = np.nonzero(np.triu(facing, 1))
    V = np.zeros((n_f, n_f), dtype=bool)
    if len(ii):
        eps = 1e-4 * np.maximum(r[ii, jj], 1e-9)
        o = centroids[ii] + normals[ii] * eps[:, None]
        dirs = u[ii, jj]
        # a hit beyond the partner does not block, so the hit distance is tested
        loc, idx_ray, _ = m.ray.intersects_location(o, dirs, multiple_hits=False)
        blocked = np.zeros(len(ii), dtype=bool)
        if len(idx_ray):
            dist = np.linalg.norm(loc - o[idx_ray], axis=1)
            blocked[idx_ray] = dist < r[ii, jj][idx_ray] * (1 - 1e-3)
        ok = ~blocked
        V[ii[ok], jj[ok]] = True
        V[jj[ok], ii[ok]] = True
    return V.astype(np.float64)


def _barycentric_samples(verts: np.ndarray, faces: np.ndarray, n_samples: int):
    """n_samples points per facet, at fixed barycentric positions (centroid for n=1)."""
    tv = verts[faces]
    if n_samples == 1:
        return tv.mean(1)[:, None, :]
    bary = {
        3: [(2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)],
        4: [(1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)],
    }[n_samples]
    return np.stack([b[0] * tv[:, 0] + b[1] * tv[:, 1] + b[2] * tv[:, 2] for b in bary], 1)


def _kernel_mean(P: torch.Tensor, n: torch.Tensor, rows: slice) -> torch.Tensor:
    """Mean over the quadrature points of cos_i cos_j / (pi r^2) for facets i in `rows`
    against every facet j, from sample points P (F, S, 3) and unit normals n (F, 3).

    cos_i cos_j / r^2 = (d . n_i)(-d . n_j) / r^4 with d = P_j - P_i, so the two dot products
    factorise and r^2 comes from the expanded square; no (i, j, si, sj, 3) array is formed.
    """
    Pi, ni = P[rows], n[rows]                                  # (I, S, 3), (I, 3)
    I_, S = Pi.shape[0], Pi.shape[1]
    F_ = P.shape[0]
    flat_i = Pi.reshape(I_ * S, 3)
    flat_j = P.reshape(F_ * S, 3)
    rr2 = ((flat_i ** 2).sum(1)[:, None] + (flat_j ** 2).sum(1)[None, :]
           - 2.0 * flat_i @ flat_j.T).reshape(I_, S, F_, S).permute(0, 2, 1, 3).clamp_min(0.0)
    # d . n_i = P_j . n_i - P_i . n_i ;  d . n_j = P_j . n_j - P_i . n_j
    dni = (torch.einsum("jbk,ik->ijb", P, ni)[:, :, None, :]
           - torch.einsum("iak,ik->ia", Pi, ni)[:, None, :, None])
    dnj = (torch.einsum("jbk,jk->jb", P, n)[None, :, None, :]
           - torch.einsum("iak,jk->ija", Pi, n)[:, :, :, None])
    k = dni.clamp_min(0.0) * (-dnj).clamp_min(0.0) / (np.pi * rr2 ** 2)
    k = torch.where(torch.isfinite(k) & (rr2 > 0), k, torch.zeros_like(k))
    return k.mean(dim=(2, 3))                                  # (I, F)


def form_factors(verts: np.ndarray, faces: np.ndarray, occlusion: bool = True,
                 n_samples: int = 4, device=None, row_block: int = 256):
    """Form factors F with reciprocity imposed, as a torch tensor on `device`, and the facet
    areas, normals and centroids as numpy arrays.

    The double integral is done by quadrature with n_samples points per facet, in blocks of
    `row_block` facets to bound memory. One point per facet (the centroid) overestimates
    near-field pairs on a coarse mesh badly enough to push row sums above 1. Visibility is
    evaluated at the centroids only: it varies far more slowly than 1/r^2. Degenerate faces
    are dropped first.
    """
    tv = verts[faces]
    keep = np.linalg.norm(np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]), axis=1) > 1e-14
    faces = faces[keep]
    c, n, a = facet_geometry(verts, faces)
    V = torch.as_tensor(_visibility_matrix(c, n, verts, faces, backface_only=not occlusion),
                        dtype=torch.float64, device=device)
    P = torch.as_tensor(_barycentric_samples(verts, faces, n_samples), dtype=torch.float64,
                        device=device)
    nt = torch.as_tensor(n, dtype=torch.float64, device=device)
    nf = len(c)
    K = torch.empty(nf, nf, dtype=torch.float64, device=device)
    for i0 in range(0, nf, row_block):
        K[i0:i0 + row_block] = _kernel_mean(P, nt, slice(i0, min(i0 + row_block, nf)))
    at = torch.as_tensor(a, dtype=torch.float64, device=device)
    F = V * K * at[None, :]                                    # F_ij = V_ij K_ij A_j
    F.fill_diagonal_(0.0)
    # reciprocity: symmetrise G_ij = A_i F_ij and read F back off it
    G = at[:, None] * F
    G = 0.5 * (G + G.T)
    F = G / at[:, None].clamp_min(1e-300)
    return F, a, n, c


class RadiositySolver:
    """Factor (I - rho F) once; every phase is then a back-substitution.

    `F` is a torch tensor and is held constant. `rho` may be a tensor, in which case the
    solution is differentiable in it. Rows of F summing to more than 1 raise RadiosityError,
    or are rescaled to sum to 1 when `bad_rows="scale"`; `n_bad_rows` counts them.
    """

    def __init__(self, F: torch.Tensor, rho, bad_rows: str = "raise", row_tol: float = 1e-6):
        F = torch.as_tensor(F)
        if not torch.isfinite(F).all():
            raise RadiosityError("form factors are not finite")
        rs = F.sum(1)
        bad = rs > 1.0 + row_tol
        self.n_bad_rows = int(bad.sum())
        if self.n_bad_rows:
            if bad_rows == "raise":
                raise RadiosityError(
                    f"form-factor row sum {float(rs.max()):.6f} exceeds 1 on "
                    f"{self.n_bad_rows} facets: the quadrature has broken down on this mesh")
            if bad_rows != "scale":
                raise ValueError(f"bad_rows must be 'raise' or 'scale', not {bad_rows!r}")
            F = torch.where(bad[:, None], F / rs[:, None], F)
        self.F = F
        self.rho = torch.as_tensor(rho, dtype=F.dtype, device=F.device)
        eye = torch.eye(len(F), dtype=F.dtype, device=F.device)
        self.max_row_sum = float(F.sum(1).max())
        self._lu = torch.linalg.lu_factor(eye - self.rho * F)

    def solve(self, e: torch.Tensor) -> torch.Tensor:
        """B from the emission e (F,) or (F, n_rhs): (I - rho F) B = rho e."""
        e = torch.as_tensor(e, dtype=self.F.dtype, device=self.F.device)
        rhs = (self.rho * e).reshape(len(self.F), -1)
        if self.rho.requires_grad:
            # lu_factor has no gradient in the matrix; re-solve through a differentiable path
            eye = torch.eye(len(self.F), dtype=self.F.dtype, device=self.F.device)
            out = torch.linalg.solve(eye - self.rho * self.F, rhs)
        else:
            out = torch.linalg.lu_solve(*self._lu, rhs)
        return out.reshape(e.shape)

    @staticmethod
    def radiance(B: torch.Tensor) -> torch.Tensor:
        return B / np.pi
