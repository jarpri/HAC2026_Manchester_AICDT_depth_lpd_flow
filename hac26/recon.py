"""Mesh helpers for the convex stage's output and for voxel scoring.

body_from_support turns a support function on the normal grid into a polytope,
smooth_support low-passes a support function, fit_to_cylinder restores the published
bounding radius, save_submission_stl writes the posed mesh, and dice with mesh_occupancy
give the challenge's voxel overlap of two meshes on a common grid.

The body frame equals the world frame at frame 0 by construction of the operator, which is
the pose the submission requires, so no registration step is needed.
"""
from __future__ import annotations

import numpy as np



def body_from_support(normals: np.ndarray, h: np.ndarray, eps: float = 1e-3) -> tuple:
    """Convex body {x : <x, u_i> <= h_i} as (vertices, hull triangles), via scipy's
    half-space intersection. h is clamped to at least eps so the origin stays inside.
    Any positive h gives a valid body; constraints that h makes redundant drop out, so h
    need not itself be a support function."""
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    h = np.maximum(np.asarray(h, dtype=float), eps)
    halfspaces = np.hstack([normals, -h[:, None]])
    hs = HalfspaceIntersection(halfspaces, np.zeros(3))
    pts = hs.intersections
    hull = ConvexHull(pts)
    return pts, hull.simplices


def smooth_support(h: np.ndarray, n_theta: int, n_phi: int, k: int = 1) -> np.ndarray:
    """Box-average h over a (2k+1)^2 neighbourhood of the (theta, phi) grid, circular in
    phi and clamped in theta.

    body_from_support takes a minimum over constraints, so one spuriously low h_n cuts a
    slab off the whole body. Support functions vary smoothly over the sphere while such
    errors do not, so a mild low-pass removes them."""
    H = np.asarray(h, dtype=float).reshape(n_theta, n_phi)
    acc, cnt = np.zeros_like(H), 0
    for dt in range(-k, k + 1):
        rows = np.clip(np.arange(n_theta) + dt, 0, n_theta - 1)
        for dp in range(-k, k + 1):
            acc += H[rows][:, (np.arange(n_phi) + dp) % n_phi]
            cnt += 1
    return (acc / cnt).reshape(-1)


def fit_to_cylinder(verts: np.ndarray, radius: float) -> np.ndarray:
    """Scale x and y so the body's largest axis distance equals the published cylinder
    radius; z is left alone. Mean normalisation removes the overall scale and the pose fixes
    only z, so the width is the one number the curves do not supply and R does.
    A non-positive or None radius leaves the vertices unchanged."""
    v = verts.copy()
    r = float(np.sqrt((v[:, :2] ** 2).sum(1)).max())
    if r > 1e-12 and radius and radius > 0:
        v[:, :2] *= radius / r
    return v


def save_submission_stl(path: str, verts: np.ndarray, faces: np.ndarray,
                        cylinder_radius: float | None = None) -> dict:
    """Write the STL and return the z range and largest axis distance, plus whether the body
    lies inside the a-priori cylinder when a radius is given. The radius is not enforced; the
    mesh is repaired and refused if it is not one watertight body (solvers.output.export_stl).
    """
    info = {"zmin": float(verts[:, 2].min()), "zmax": float(verts[:, 2].max()),
            "max_axis_dist": float(np.sqrt((verts[:, :2] ** 2).sum(1)).max())}
    if cylinder_radius is not None:
        info["cylinder_radius_prior"] = cylinder_radius
        info["inside_prior_cylinder"] = bool(info["max_axis_dist"] <= cylinder_radius + 1e-9)
    # Through the same repair-and-check gate every solver's answer goes through, so a
    # submission file cannot be written non-watertight or inside out.
    from .solvers.output import export_stl
    info.update(export_stl(path, verts, faces))
    return info


# ---------- voxel evaluation ---------------------------------------------------------
def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two boolean voxel grids (the challenge's voxel measure); 1 when
    both are empty."""
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else 1.0


def mesh_occupancy(verts: np.ndarray, faces: np.ndarray, n: int, extent: float,
                   decimate: bool = True) -> np.ndarray:
    """Boolean n^3 grid: which cell centres of [-extent, extent]^3 lie inside the mesh.

    Inside is decided by a parity scan up each (x, y) column
    (shape_library._parity_occupancy). A mesh much finer than the grid is first
    vertex-clustered at half the cell pitch, which moves the occupancy by at most half a
    cell; coarser meshes are used as given.
    """
    from .shape_library import _parity_occupancy, decimate_mesh

    v = np.ascontiguousarray(verts, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    if decimate and len(f) > 4 * n * n:
        v, f = decimate_mesh(v, f, extent, 2 * n)
    axis = (np.arange(n) + 0.5) / n * 2.0 * extent - extent   # cell centres, the scoring grid
    return _parity_occupancy(v, f, extent, n, axis=axis)

