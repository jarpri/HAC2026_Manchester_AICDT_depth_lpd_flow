"""How varied a shape library is, and whether each body is admissible.

Validity is per body and binary: closed, one component, correctly posed. `check_body`
returns each answer separately so a failure says which constraint broke, and measures the
convexity ratio alongside.

Diversity is a property of the set. The headline number is the participation ratio

    PR = (sum_i lambda_i)^2 / sum_i lambda_i^2

of the eigenvalues of the library's covariance: the number of directions that carry
variance, equal to D for an isotropic cloud in D dimensions and to 1 for a cloud on a line.
A library with a small PR is close to a few-parameter family.

PR depends on what is measured. Bodies that differ mostly in overall size put their largest
eigenvalue on a degree of freedom the challenge pose removes, so the descriptors here are
meant for posed bodies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .shape_library import (_parity_occupancy, convexity_ratio, decimate_mesh,
                            is_edge_manifold, n_components)

__all__ = ["participation_ratio", "spectrum", "descriptor_support", "descriptor_concavity",
           "library_descriptors", "occupancy", "principal_frame", "dice", "pairwise_dice",
           "check_body", "check_library", "design_normals"]


def design_normals(n: int = 256) -> np.ndarray:
    """`n` design normals for the descriptors below, from the cached `design{n}.npy` next to
    this module, or built with `hac26.field.spherical_design` if the cache is absent.

    Smaller than the solver's `DESIGN_N`, which is slow to build: these are library
    statistics, not the core. The participation ratio of the support descriptor is capped by
    this count, so it must be large enough to resolve the structure being measured.
    """
    p = Path(__file__).with_name(f"design{n}.npy")
    if p.exists():
        return np.load(p)
    from .field import spherical_design            # needs torch; only if the cache is absent
    return spherical_design(n)


# --------------------------------------------------------------------------- diversity

def spectrum(X: np.ndarray) -> np.ndarray:
    """Eigenvalues of the sample covariance of rows of X, descending, clipped at 0."""
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    lam = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))
    return np.clip(lam, 0.0, None)[::-1]


def participation_ratio(X: np.ndarray) -> float:
    """Effective number of dimensions spanned by the rows of X."""
    lam = spectrum(X)
    s2 = float((lam ** 2).sum())
    return float(lam.sum() ** 2 / s2) if s2 > 0 else 0.0


def descriptor_support(verts: np.ndarray, normals: np.ndarray | None = None) -> np.ndarray:
    """h(n) = max_v <v, n> on the design normals: the support function of the convex hull.

    `scripts/fit_shapes.py` pins each body's core support to this same quantity (on the
    solver's own normals), so this is the convex part of a code without running the fit.
    """
    n = design_normals() if normals is None else np.asarray(normals, float)
    return (np.asarray(verts, float) @ n.T).max(axis=0)


def descriptor_concavity(verts: np.ndarray, faces: np.ndarray,
                         probes: np.ndarray, normals: np.ndarray | None = None,
                         res: int = 64, extent: float = 1.35) -> np.ndarray:
    """f_body(y) - f_core(y) at fixed probe points, where f_core is the convex core at the
    hull support and f_body a signed distance from the body's own occupancy.

    This is the part of the field the depth correction (`hac26.field.DepthSphere`) has to
    carry when `scripts/fit_shapes.py` fits a body with the core pinned to its hull.
    Computed by a distance transform, so it needs neither torch nor trimesh.
    """
    n = design_normals() if normals is None else np.asarray(normals, float)
    h = descriptor_support(verts, n)
    occ = occupancy(verts, faces, res, extent)
    sp = 2.0 * extent / (res - 1)
    sdf = (ndimage.distance_transform_edt(~occ, sampling=sp)
           - ndimage.distance_transform_edt(occ, sampling=sp))
    idx = ((np.asarray(probes, float) + extent) / sp).T
    f_body = ndimage.map_coordinates(sdf, idx, order=1, mode="nearest")
    f_core = (np.asarray(probes, float) @ n.T - h).max(axis=1)
    return f_body - f_core


def _probe_points(n: int = 512, seed: int = 0, radius: float = 1.15) -> np.ndarray:
    """A fixed cloud of `n` points filling the cylinder a posed body sits in. The same points
    are used for every body, or the concavity descriptors would not be comparable."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([r * np.cos(t), r * np.sin(t), rng.uniform(-1.0, 1.0, n)], axis=1)


def library_descriptors(bodies: list, n_probes: int = 512, res: int = 64) -> dict:
    """Support, concavity and combined descriptors for a library of `Body`, one row each."""
    nrm = design_normals()
    probes = _probe_points(n_probes)
    H = np.stack([descriptor_support(b.verts, nrm) for b in bodies])
    C = np.stack([descriptor_concavity(b.verts, b.faces, probes, nrm, res=res)
                  for b in bodies])
    # Scale the two blocks to equal mean variance before concatenating, so the combined PR
    # is not dominated by whichever block carries larger numbers.
    hs = np.sqrt(max(H.var(0).mean(), 1e-18))
    cs = np.sqrt(max(C.var(0).mean(), 1e-18))
    return {"support": H, "concavity": C, "combined": np.hstack([H / hs, C / cs])}


# --------------------------------------------------------------------------- overlap

def occupancy(verts: np.ndarray, faces: np.ndarray, res: int = 64,
              extent: float = 1.35, decimate: bool = True) -> np.ndarray:
    """Boolean res^3 occupancy of a mesh over [-extent, extent]^3, by ray parity. Large
    meshes are decimated to the same grid first."""
    v, f = np.asarray(verts, float), np.asarray(faces, np.int64)
    if decimate and len(f) > 3000:
        v, f = decimate_mesh(v, f, extent, res)
    return _parity_occupancy(v, f, extent, res)


def principal_frame(verts: np.ndarray, faces: np.ndarray, res: int = 64,
                    extent: float = 1.35) -> np.ndarray:
    """Rotation matrix whose rows are the body's principal axes, largest moment first.

    The moments are computed from the occupancy, not from the vertices, so the frame depends
    on the body rather than on how finely each region is meshed. The four proper sign flips
    are left unresolved here and maximised over in `dice`, because a body with two near-equal
    moments has no stable sign.
    """
    occ = occupancy(verts, faces, res, extent)
    idx = np.argwhere(occ).astype(float)
    if len(idx) < 4:
        return np.eye(3)
    sp = 2.0 * extent / (res - 1)
    p = idx * sp - extent
    p -= p.mean(0)
    _, vecs = np.linalg.eigh(np.cov(p, rowvar=False))
    R = vecs[:, ::-1].T                       # rows = principal axes, largest first
    if np.linalg.det(R) < 0:
        R[2] *= -1.0
    return R


_SIGN_FLIPS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)


def occupancies_aligned(bodies: list, res: int = 48, extent: float = 1.35) -> list:
    """Occupancy grids of each body in its principal frame, one per proper sign flip.

    Computed once per body so `pairwise_dice` can reuse them across every pair.
    """
    out = []
    for b in bodies:
        R = principal_frame(b.verts, b.faces, res, extent)
        va = b.verts @ R.T
        out.append([occupancy(va * s, b.faces, res, extent) for s in _SIGN_FLIPS])
    return out


def _dice_from_occ(occs_a: list, occs_b: list) -> float:
    """Best Dice between the first grid of `occs_a` and any grid of `occs_b`."""
    A = occs_a[0]
    na = A.sum()
    best = 0.0
    for B in occs_b:
        nb = B.sum()
        if na + nb == 0:
            continue
        best = max(best, 2.0 * float((A & B).sum()) / float(na + nb))
    return best


def dice(a, b, res: int = 64, extent: float = 1.35, align: bool = True) -> float:
    """Voxel Dice between two bodies (`Body` or (verts, faces) pairs).

    With `align`, both are first carried into their principal frames, so the score measures
    shape rather than stored orientation, and the best of the four proper sign flips is
    taken. For many pairs from one library use `pairwise_dice`, which aligns each body once.
    """
    va, fa = (a.verts, a.faces) if hasattr(a, "verts") else a
    vb, fb = (b.verts, b.faces) if hasattr(b, "verts") else b
    if not align:
        A = occupancy(va, fa, res, extent)
        B = occupancy(vb, fb, res, extent)
        na, nb = A.sum(), B.sum()
        return 2.0 * float((A & B).sum()) / float(na + nb) if na + nb else 0.0
    Ra = principal_frame(va, fa, res, extent)
    Rb = principal_frame(vb, fb, res, extent)
    occs_a = [occupancy((va @ Ra.T) * _SIGN_FLIPS[0], fa, res, extent)]
    occs_b = [occupancy((vb @ Rb.T) * s, fb, res, extent) for s in _SIGN_FLIPS]
    return _dice_from_occ(occs_a, occs_b)


def pairwise_dice(bodies: list, res: int = 48, extent: float = 1.35,
                  max_pairs: int | None = 300, seed: int = 0) -> np.ndarray:
    """Aligned Dice over up to `max_pairs` randomly chosen distinct pairs of bodies.

    A varied library gives a broad distribution centred well below 1; a library of
    near-copies concentrates near 1.
    """
    n = len(bodies)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
    occs = occupancies_aligned(bodies, res, extent)
    return np.array([_dice_from_occ(occs[i], occs[j]) for i, j in pairs])


# --------------------------------------------------------------------------- validity

def check_body(body, radius: float = 1.0, pose_tol: float = 1e-6,
               radius_tol: float = 1e-6) -> dict:
    """Every per-body constraint as its own bool, plus the measured values behind them."""
    v, f = body.verts, body.faces
    z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
    rmax = float(np.hypot(v[:, 0], v[:, 1]).max())
    cx, cy = float(v[:, 0].mean()), float(v[:, 1].mean())
    c = convexity_ratio(v, f)
    return {
        "convexity": float(c),
        "closed": bool(is_edge_manifold(f)),
        "single_component": bool(n_components(v, f) == 1),
        "z_span": bool(abs(z0 + 1.0) <= pose_tol and abs(z1 - 1.0) <= pose_tol),
        "inside_cylinder": bool(rmax <= radius * (1.0 + radius_tol)),
        "on_axis": bool(max(abs(cx), abs(cy)) <= 0.25 * radius),
        "z_range": (z0, z1), "r_max": rmax, "xy_centroid": (cx, cy),
    }


def check_library(bodies: list, **kw) -> dict:
    """`check_body` over a library, plus the indices that failed each constraint."""
    rows = [check_body(b, **kw) for b in bodies]
    keys = ["closed", "single_component", "z_span", "inside_cylinder", "on_axis"]
    return {"n": len(bodies),
            "pass": {k: int(sum(r[k] for r in rows)) for k in keys},
            "failed": {k: [i for i, r in enumerate(rows) if not r[k]] for k in keys},
            "convexity": np.array([r["convexity"] for r in rows]),
            "rows": rows}
