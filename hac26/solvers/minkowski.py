"""Minkowski reconstruction: facet areas on fixed normals -> convex polytope mesh.

Given unit normals u_i and areas g_i with sum_i g_i u_i = 0, Minkowski's problem asks for
the convex polytope whose facet with outward normal u_i has area g_i. Its variational form
(Minkowski 1897; Schneider, Brunn-Minkowski Theory, sec. 8.2) is

    minimize   sum_i g_i h_i
    subject to vol(P(h)) >= 1,      P(h) = { x : <u_i, x> <= h_i  for all i }

and a minimiser has facet areas proportional to g. solve_minkowski minimises the equivalent
scale-free quotient written out below the imports with L-BFGS-B, using the exact gradient
d vol / d h_i = area of facet i, and rescales the result to unit volume. The absolute scale
does not matter here because the mesh is posed to z in [-1, 1] downstream.

Safeguards: normals with tiny weight are dropped from the objective and the half-space list
(their facets would have almost no area); six extra half-spaces forming a large box keep the
intersection bounded at every iterate and are inactive at the optimum; the polytope for a
given h comes from scipy's HalfspaceIntersection, started from the origin when every h_i is
positive and from a Chebyshev centre otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection

# Scale-free form used by solve_minkowski (equivalent to the constrained form because
# V scales as the cube of h and g.h linearly):
#     minimize  F(h) = (g . h) / V(h)^{1/3}   over  h > 0,
#     grad F    = g V^{-1/3} - (g . h)/3 * V^{-4/3} * areas(h),
# so grad F = 0 exactly when the facet areas are proportional to g.

CAGE_NORMALS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                         [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)


def chebyshev_center(U: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Centre of the largest ball inside {x : U x <= h}: max r subject to U x + r <= h."""
    n = U.shape[0]
    A_ub = np.hstack([U, np.ones((n, 1))])
    res = linprog(c=[0, 0, 0, -1.0], A_ub=A_ub, b_ub=h,
                  bounds=[(None, None)] * 3 + [(1e-9, None)], method="highs")
    if not res.success or res.x[3] <= 0:
        raise RuntimeError("no interior point: halfspaces infeasible or degenerate")
    return res.x[:3]


def polytope_geometry(U: np.ndarray, h: np.ndarray) -> dict:
    """Vertices, outward-oriented triangles, volume and per-half-space facet areas of P(h).

    Each hull triangle is assigned to the half-space whose normal is closest to its own."""
    # all h_i > 0  =>  the origin is interior (U x = 0 < h); the LP is only a fallback
    ip = np.zeros(3) if h.min() > 1e-9 else chebyshev_center(U, h)
    hs = HalfspaceIntersection(np.hstack([U, -h[:, None]]), ip)
    pts = hs.intersections
    hull = ConvexHull(pts)
    areas = np.zeros(U.shape[0])
    faces = hull.simplices.copy()
    eqs = hull.equations[:, :3]
    for i, f in enumerate(faces):
        n = np.cross(pts[f[1]] - pts[f[0]], pts[f[2]] - pts[f[0]])
        if n @ eqs[i] < 0:
            faces[i] = f[::-1]
        tri_area = 0.5 * np.linalg.norm(n)
        if tri_area > 1e-14:
            j = int(np.argmax(U @ eqs[i]))
            areas[j] += tri_area
    return {"verts": pts, "faces": faces, "volume": float(hull.volume), "areas": areas}


def solve_minkowski(normals: np.ndarray, g: np.ndarray, drop_tol: float = 1e-4,
                    cage: float = 50.0, maxiter: int = 300, verbose: bool = False) -> dict:
    """Polytope of unit volume whose facet areas on `normals` are proportional to g >= 0
    (with sum g_i u_i close to 0). Normals with g below drop_tol times the largest weight
    are left out.

    Returns a dict with 'verts' (shifted to zero mean), 'faces', 'volume', the kept
    normals, weights, areas and support values, 'egi_l1' (L1 distance between the
    normalised areas and the normalised weights), 'success' and the optimiser message."""
    g = np.asarray(g, dtype=float)
    keep = g > drop_tol * g.max()
    Uk, gk = normals[keep], g[keep]
    nk = len(gk)
    Uall = np.vstack([Uk, CAGE_NORMALS])

    cache: dict = {}

    def geom(hk: np.ndarray) -> dict:
        key = hk.tobytes()
        if key not in cache:
            h = np.concatenate([hk, np.full(6, cage)])
            cache.clear()
            cache[key] = polytope_geometry(Uall, h)
        return cache[key]

    h0 = np.full(nk, (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * 1.3)
    bounds = [(1e-3, 0.8 * cage)] * nk  # keeps the origin interior at every iterate

    def fun_grad(hk: np.ndarray):
        G = geom(hk)
        V = max(G["volume"], 1e-14)
        a = G["areas"][:nk]
        gh = float(gk @ hk)
        F = gh / V ** (1.0 / 3.0)
        dF = gk / V ** (1.0 / 3.0) - (gh / 3.0) * V ** (-4.0 / 3.0) * a
        return F, dF

    res = minimize(fun_grad, h0, jac=True, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10,
                            "disp": verbose})
    # rescale to unit volume (h -> lambda h scales P(h) by lambda)
    hk = res.x * (1.0 / max(geom(res.x)["volume"], 1e-12)) ** (1.0 / 3.0)
    G = geom(hk)
    verts = G["verts"] - G["verts"].mean(axis=0)
    a = G["areas"][:nk]
    egi_l1 = float(np.abs(a / max(a.sum(), 1e-14) - gk / gk.sum()).sum())
    return {"verts": verts, "faces": G["faces"], "volume": G["volume"],
            "areas_kept": a, "g_kept": gk, "normals_kept": Uk, "h": hk,
            "egi_l1": egi_l1, "success": bool(res.success) or egi_l1 < 0.05,
            "message": str(res.message)}
