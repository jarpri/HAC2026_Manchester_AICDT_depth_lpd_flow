"""MAP Gauss-Newton step, and the shape prior it needs.

    ( J^T Sigma^-1 J + Gamma^-1 ) delta = J^T Sigma^-1 r - Gamma^-1 (x - x0)

Directions where J^T Sigma^-1 J dominates Gamma^-1 are set by the data; the rest fall to the
prior. Because the fitted model error is inside Sigma, a direction determined only by model
error is demoted without special handling. There is no SVD and no threshold.

Gamma is a covariance estimated from a shape library projected into the solver's own
parameterisation, not a smoothness functional: it describes what shapes look like rather than
asserting that they are smooth.

The axisymmetric block is given a finite, small variance and decoupled,

    Gamma = (I - P) Gamma (I - P) + sigma0^2 P

because that direction is unidentifiable from these curves: an axisymmetric body gives
constant curves at every phase, so a spindle and an hourglass of equal volume, silhouette,
radius and z-extent produce identical mean-normalised data while differing in shape.
`m0_projector` builds P from Legendre polynomials in cos(theta) sampled at the
parameterisation's directions.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["m0_projector", "shape_prior", "map_step", "chi2_and_jacobian"]


def m0_projector(normals: np.ndarray, l_max: int = 8) -> np.ndarray:
    """Orthogonal projector onto the axisymmetric subspace of a support-like vector.

    A field on the sphere is axisymmetric about z exactly when it depends on theta alone, so
    the m = 0 subspace is spanned by the Legendre polynomials P_l(cos theta) sampled at the
    parameterisation's directions. The projector is the least-squares projector onto that
    span, which needs no quadrature weights and tolerates the directions being unequally
    spaced.
    """
    n = np.asarray(normals, dtype=float)
    ct = n[:, 2] / np.linalg.norm(n, axis=1).clip(1e-12)
    B = np.stack([np.polynomial.legendre.Legendre.basis(l)(ct) for l in range(l_max + 1)], 1)
    q, _ = np.linalg.qr(B)                      # orthonormal basis of the same span
    return q @ q.T


def shape_prior(library: np.ndarray, projector: np.ndarray | None = None,
                sigma0: float = 0.02, shrinkage: float | None = None,
                floor: float = 1e-8) -> dict:
    """Gamma from a shape library, with the axisymmetric block set to sigma0^2.

    library is (S, D): S shapes projected into the D-dimensional parameterisation.

    Shrinkage is required, not optional: a library of S shapes gives an empirical covariance
    of rank at most S - 1, which is singular whenever S <= D and cannot be inverted. The
    estimate is shrunk toward its own diagonal,

        Gamma = (1 - a) Gamma_emp + a diag(Gamma_emp)

    with a set by the Ledoit-Wolf ratio when not supplied. Shrinking toward the diagonal
    keeps the measured per-coordinate scales and discards only the under-determined
    correlations.

    If a projector onto the axisymmetric subspace is given, Gamma is replaced by

        Gamma = (I - P) Gamma (I - P) + sigma0^2 P

    so that block is exactly sigma0^2 and decoupled from the rest. Small sigma0 holds the
    estimator near x0 along the one direction the data cannot see at all.
    """
    X = np.asarray(library, dtype=np.float64)
    S, D = X.shape
    x0 = X.mean(0)
    Xc = X - x0
    emp = (Xc.T @ Xc) / max(S - 1, 1)
    if shrinkage is None:
        # Ledoit-Wolf: the dispersion of the per-sample covariances against the target
        num = np.mean([np.sum((np.outer(Xc[i], Xc[i]) - emp) ** 2) for i in range(S)]) / S
        den = np.sum((emp - np.diag(np.diag(emp))) ** 2)
        shrinkage = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 1.0
    G = (1.0 - shrinkage) * emp + shrinkage * np.diag(np.diag(emp))
    G = G + floor * np.eye(D)
    if projector is not None:
        P = np.asarray(projector, dtype=np.float64)
        if P.shape[0] != D:                      # prior over a block of the vector only
            Pf = np.zeros((D, D)); Pf[:P.shape[0], :P.shape[1]] = P; P = Pf
        Q = np.eye(D) - P
        G = Q @ G @ Q + (sigma0 ** 2) * P
        G = G + floor * np.eye(D)
    return {"x0": x0, "Gamma": G, "Gamma_inv": np.linalg.inv(G),
            "shrinkage": float(shrinkage), "sigma0": float(sigma0)}


def map_step(J: torch.Tensor, r: torch.Tensor, s2: torch.Tensor, x: torch.Tensor,
             x0: torch.Tensor, gamma_inv: torch.Tensor) -> torch.Tensor:
    """One MAP Gauss-Newton step.

    J is (n_resid, D), r is (n_resid,), s2 is (n_resid,) -- the diagonal of Sigma, which is
    diagonal because the whitening is per (curve, mode). x, x0 are (D,) and gamma_inv (D, D).
    """
    Sinv = (1.0 / s2.clamp_min(1e-18))
    A = J.T @ (Sinv[:, None] * J) + gamma_inv
    b = J.T @ (Sinv * r) - gamma_inv @ (x - x0)
    return torch.linalg.solve(A, b)


def chi2_and_jacobian(residual_fn, x: torch.Tensor, s2: torch.Tensor):
    """Whitened residual and its Jacobian by autodiff. residual_fn(x) -> (n_resid,)."""
    x = x.detach().requires_grad_(True)
    r = residual_fn(x)
    J = torch.autograd.functional.jacobian(lambda z: residual_fn(z), x.detach())
    chi2 = float((r.detach() ** 2 / s2.clamp_min(1e-18)).sum())
    return r.detach(), J, chi2
