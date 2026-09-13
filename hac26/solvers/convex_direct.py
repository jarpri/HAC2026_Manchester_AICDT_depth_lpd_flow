"""The convex stage as a direct solve rather than a learned one.

For a convex body nothing shadows, so the curve of a body is a sum over the facets of its
extended Gaussian image: with facet areas g on fixed normals u, the raw curve of camera c is
A[c] g, linear in g (hac26.forward.convex_egi). The organisers release every curve divided by
its own mean, which removes exactly one degree of freedom per curve, so what the data say is
that A[c] g is parallel to the released curve d[c], not equal to it. Writing that as a linear
constraint,

    P_c A[c] g = 0,     P_c = I - d[c] 1^T / m,

makes the whole problem one homogeneous least-squares system in g, subject to the two things
a facet-area vector must satisfy: g >= 0, and sum_i g_i u_i = 0, without which Minkowski's
theorem gives no polytope. The scale of g is not determined by normalised curves and is fixed
here by sum_i g_i = 1; the body is scaled to the published bounding radius downstream.

One thing is not linear. The organisers count pixels above Otsu's level of each video's first
frame, and that level depends on the body, so the binary rows of A depend on g through their
thresholds. The dependence is weak and is handled by alternation: derive the levels from the
current g, rebuild A, solve, repeat. Three passes move the levels by less than the width of
the histogram bin they are read from.

Nothing here is trained. The alternative in this repository is an unrolled primal-dual network
whose weights were fitted against a different photometric kernel, and whose operator is a
fixed tensor that cannot carry a threshold that depends on the body it is reconstructing.

It does not yet replace that network. Scored against the released shapes on the same measure,
this solve reaches 0.948, 0.832 and 0.710 on the three public bodies where the trained stage
reaches 0.983, 0.893 and 0.708, and a sweep of the smoothness over two decades does not close
the gap on the first two. What the trained network has and this does not is a prior over
bodies; the smoothness penalty here is the crudest possible stand-in for one. The residual of
this solve at its own answer is small on bodies 1 and 3, so what it is missing is not fit to
the curves but the part of the body the curves do not determine.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from hac26.conventions import TRANSFER_EXPONENT
from hac26.forward.convex_egi import build_A, curve_thresholds
from hac26.geometry import NormalGrid, build_cameras, project_closure
from hac26.solvers.minkowski import solve_minkowski

__all__ = ["solve_egi", "convex_body"]

SUM_WEIGHT = 1e3      # weight of the row that fixes sum(g) = 1. Large enough that the scale
                      # is pinned rather than traded against the curves, small enough that
                      # the system stays well conditioned.
SMOOTH = 3e-2         # weight of the smoothness penalty on the extended Gaussian image, as a
                      # fraction of the mean squared column norm of the design. The curves are
                      # integrals of the image against broad kernels and resolve it only to
                      # low order, so without a penalty the solve puts area on single normals
                      # no camera separates and Minkowski turns each of them into a facet.
                      # Measured on the public bodies.
RIDGE = 1e-6          # ridge on g, for conditioning alone.


def _laplacian(grid: NormalGrid) -> np.ndarray:
    """Graph Laplacian of the normal grid: each cell against the mean of its four neighbours.

    The rows are weighted by sin(theta), the area of the cell, so a penalty on the image is a
    penalty on the surface it stands for and not on the parametrisation, which would smooth
    the poles far harder than the equator.
    """
    nt, np_ = grid.n_theta, grid.n_phi
    idx = np.arange(nt * np_).reshape(nt, np_)
    L = np.zeros((nt * np_, nt * np_))
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        i = np.clip(np.arange(nt)[:, None] + di, 0, nt - 1)          # poles reflect
        j = (np.arange(np_)[None, :] + dj) % np_                     # azimuth wraps
        L[idx.ravel(), idx[i, np.broadcast_to(j, (nt, np_))].ravel()] -= 0.25
    L[np.arange(nt * np_), np.arange(nt * np_)] += 1.0
    w = np.sqrt(np.sin(grid.theta))[:, None] * np.ones((1, np_))
    return L * w.reshape(-1, 1)


def _design(A: np.ndarray, data: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Rows of the homogeneous system, one block per present curve.

    P_c A[c] with P_c = I - d 1^T/m removes from each curve exactly the direction the mean
    normalisation discarded, so a solution is a body whose curve is parallel to the measured
    one. Each block is divided by the norm of its own curve, so a bright curve and a faint
    one carry the same weight."""
    blocks = []
    for c in np.nonzero(present)[0]:
        d = data[c]
        y = A[c]                                          # (m, N)
        blocks.append((y - np.outer(d, y.mean(axis=0))) / max(np.linalg.norm(d), 1e-9))
    return np.vstack(blocks)


def solve_egi(data: np.ndarray, present: np.ndarray, grid: NormalGrid, cameras: list,
              m: int, sigma: float = -1.0, delta: float = 1.0,
              gamma: float = TRANSFER_EXPONENT, passes: int = 3,
              smooth: float = SMOOTH, ridge: float = RIDGE) -> dict:
    """Facet areas of the extended Gaussian image from one body's normalised curves.

    `data` is (2 C, m), every camera's intensity curve then every camera's binary curve, each
    divided by its own mean; `present` marks the rows that are measurements. Returns the
    areas and what the solve did.
    """
    cams2 = list(cameras) + list(cameras)
    types = ["intensity"] * len(cameras) + ["binary"] * len(cameras)
    n = grid.normals.shape[0]
    lap = _laplacian(grid)
    g = np.full(n, 1.0 / n)
    history = []
    thr = np.zeros(len(types))
    for _ in range(int(passes)):
        thr = curve_thresholds(grid.normals, g, cams2, m, types, gamma=gamma,
                               sigma=sigma, delta=delta)
        A = build_A(grid.normals, cams2, m, types, gamma=gamma, thresholds=thr,
                    sigma=sigma, delta=delta)
        M = _design(A, data, present)
        scale = float(np.sqrt((M ** 2).sum(axis=0).mean()))
        rows = [M, np.full((1, n), SUM_WEIGHT * scale),
                np.sqrt(smooth) * scale * lap, np.sqrt(ridge) * scale * np.eye(n)]
        rhs = np.concatenate([np.zeros(len(M)), [SUM_WEIGHT * scale], np.zeros(2 * n)])
        g = nnls(np.vstack(rows), rhs, maxiter=20 * n)[0]
        g = project_closure(g, grid.normals)
        s = g.sum()
        g = g / s if s > 0 else np.full(n, 1.0 / n)
        history.append({"residual": float(np.linalg.norm(M @ g)),
                        "roughness": float(np.linalg.norm(lap @ g)),
                        "closure": float(np.linalg.norm(grid.normals.T @ g)),
                        "support": int((g > 1e-4 * g.max()).sum())})
    return {"g": g, "thresholds": thr, "history": history, "smooth": float(smooth),
            "curves_used": int(present.sum())}


def convex_body(data: np.ndarray, present: np.ndarray, grid: NormalGrid | None = None,
                cameras: list | None = None, m: int | None = None, **kw) -> tuple:
    """(verts, faces, report) of the convex body one set of normalised curves implies."""
    from hac26.geometry import make_grid
    grid = make_grid() if grid is None else grid
    cameras = build_cameras() if cameras is None else cameras
    m = data.shape[-1] if m is None else m
    out = solve_egi(data, present, grid, cameras, m, **kw)
    poly = solve_minkowski(grid.normals, out["g"])
    return poly["verts"], poly["faces"], {**{k: v for k, v in out.items() if k != "g"},
                                          "egi_l1": poly["egi_l1"],
                                          "minkowski_success": bool(poly["success"])}
