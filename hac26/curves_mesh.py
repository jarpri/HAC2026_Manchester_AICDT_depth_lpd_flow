"""Curves of a general, possibly non-convex, mesh by CPU ray casting on the challenge
conventions.

`hac26.shapes.mesh_curves_convex` counts every facet with mu > 0 and mu0 > 0 as seen and
lit, which is exact only for convex bodies. Bodies from hac26.shape_library can occlude and
shadow themselves, so they need a renderer that casts rays. This one takes the rotation
sense, camera directions and light direction from `hac26.conventions`; delta is a parameter,
so the value `hac26.data_io.fit_conventions` finds can be passed in.

Scope: Lambert radiance through the measured power-law transfer, orthographic projection,
hard cast shadows, no interreflection and no sensor chain. The mesh forward chain
(hac26.forward.mesh) adds those; this module runs without torch or a GPU.
"""
from __future__ import annotations

import numpy as np

from .conventions import (S_LAB, SENSE, TRANSFER_EXPONENT, cameras, psi_grid,
                          to_body)
from .shape_library import decimate_mesh

__all__ = ["render_curves_mesh", "convex_cross_check"]


def _ray_triangle_batch(o: np.ndarray, d: np.ndarray, v0, v1, v2, eps: float = 1e-9):
    """Moller-Trumbore ray-triangle test for R rays (o, d: (R, 3)) against T triangles
    (v0, v1, v2: (T, 3)). Returns the hit distance t of shape (R, T), np.inf where the ray
    misses or the hit lies behind the origin."""
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(d[:, None, :], e2[None, :, :])          # (R, T, 3)
    det = np.einsum("rtj,tj->rt", pvec, e1)
    inv = np.where(np.abs(det) > eps, 1.0 / np.where(np.abs(det) > eps, det, 1.0), 0.0)
    tvec = o[:, None, :] - v0[None, :, :]                    # (R, T, 3)
    u = np.einsum("rtj,rtj->rt", tvec, pvec) * inv
    qvec = np.cross(tvec, e1[None, :, :])
    v = np.einsum("rj,rtj->rt", d, qvec) * inv
    t = np.einsum("tj,rtj->rt", e2, qvec) * inv
    ok = (np.abs(det) > eps) & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > eps)
    return np.where(ok, t, np.inf)


def _first_hit(o, d, v0, v1, v2, chunk_t: int = 4000):
    """Nearest hit per ray: (distance, triangle index), -1 and inf where nothing is hit.
    Triangles are processed in chunks to bound memory."""
    best_t = np.full(len(o), np.inf)
    best_f = np.full(len(o), -1, dtype=np.int64)
    for s in range(0, len(v0), chunk_t):
        t = _ray_triangle_batch(o, d, v0[s:s + chunk_t], v1[s:s + chunk_t], v2[s:s + chunk_t])
        j = np.argmin(t, axis=1)
        tj = t[np.arange(len(o)), j]
        better = tj < best_t
        best_t = np.where(better, tj, best_t)
        best_f = np.where(better, j + s, best_f)
    return best_t, best_f


def _shadowed(hit, sun_dir, v0, v1, v2, eps: float = 1e-4, chunk_t: int = 4000):
    """True where a ray from `hit` toward `sun_dir` meets the mesh, i.e. the point is in
    cast shadow. The ray starts a small step off the surface to avoid hitting it."""
    o = hit + eps * sun_dir
    d = np.broadcast_to(sun_dir, o.shape)
    t, _ = _first_hit(o, d, v0, v1, v2, chunk_t)
    return np.isfinite(t)


def _otsu(val: np.ndarray, bins: int = 256) -> float:
    """Otsu's level of one frame, which is what the organisers threshold each video at and
    then hold for the rotation."""
    x = np.clip(np.asarray(val, float).ravel(), 0.0, 1.0)
    hist = np.bincount(np.round(x * (bins - 1)).astype(int), minlength=bins).astype(float)
    lev = np.arange(bins) / (bins - 1)
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * lev)
    a = m0 / np.maximum(w0, 1e-12)
    b = (m0[-1] - m0) / np.maximum(w1, 1e-12)
    return float(lev[int(np.argmax((w0 * w1 * (a - b) ** 2)[:-1]))])


def render_curves_mesh(verts: np.ndarray, faces: np.ndarray, m: int = 360,
                       curve_types: list | None = None, geoms: list | None = None,
                       psi0: float = 0.0, delta: float = 1.0,
                       gamma: float = TRANSFER_EXPONENT, tau_i: float = 0.0,
                       tau_b: float | None = None,
                       res: int = 96, extent: float | None = None, shadows: bool = True,
                       chunk_t: int = 4000, decimate_to: int | None = 4000) -> np.ndarray:
    """Raw (unnormalised) curves, one row per entry of `curve_types`; row i uses geometry
    i mod len(geoms). The defaults give every geometry as intensity, then as binary.

    `geoms` defaults to the released camera geometries in column order; pass a subset for a
    quick check. `delta` is the azimuth handedness; the rotation sense is always
    `hac26.conventions.SENSE`. Frames are `res` x `res` orthographic pixels over
    [-extent, extent]; the intensity curve sums pixel values above tau_i, the binary curve
    counts pixels above tau_b, both times the pixel area. `tau_b` defaults to Otsu's level of
    each geometry's first frame, which is how the organisers threshold each video, and is
    then held over the rotation.

    `decimate_to` caps the triangle count before ray casting, whose cost grows with
    triangles x pixels x frames (see `shape_library.decimate_mesh`); pass None to render
    the mesh as given.
    """
    geoms = cameras() if geoms is None else geoms
    curve_types = (["intensity"] * len(geoms) + ["binary"] * len(geoms)
                  if curve_types is None else curve_types)
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    if decimate_to is not None and len(f) > decimate_to:
        target_res = max(8, int(round((decimate_to / 2.0) ** 0.5)) * 2)
        v, f = decimate_mesh(v, f, float(np.max(np.linalg.norm(v, axis=1))) * 1.05, target_res)
    v0f, v1f, v2f = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(v1f - v0f, v2f - v0f)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True).clip(1e-12)

    extent = extent or 1.05 * float(np.max(np.linalg.norm(v, axis=1)))
    psi = psi_grid(m, sense=SENSE)
    a = np.linspace(-extent, extent, res)
    gx, gy = np.meshgrid(a, a, indexing="ij")
    px_area = (a[1] - a[0]) ** 2 if res > 1 else (2 * extent) ** 2

    sun_body = to_body(S_LAB, psi, psi0)                       # (m, 3), object -> source

    def _basis(view):
        up = np.array([0.0, 0.0, 1.0]) if abs(view[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
        ex = np.cross(up, view); ex /= np.linalg.norm(ex)
        ey = np.cross(view, ex)
        return ex, ey

    n_geom_types = len(geoms)
    inten = np.zeros((n_geom_types, m))
    binar = np.zeros((n_geom_types, m))
    for ci, cam in enumerate(geoms):
        # Each geometry is its own video, so each gets its own level from its own first
        # frame unless the caller names one.
        level = tau_b
        cam_lab = cam.v if hasattr(cam, "v") else camera_vector_compat(cam)
        cam_body = to_body(np.asarray(cam_lab, float), psi, psi0)   # (m, 3)
        for k in range(m):
            view = cam_body[k]
            ex, ey = _basis(view)
            o = (gx.reshape(-1, 1) * ex + gy.reshape(-1, 1) * ey) + 2.0 * extent * view
            d = np.broadcast_to(-view, o.shape).copy()
            t, fidx = _first_hit(o, d, v0f, v1f, v2f, chunk_t)
            hit_mask = np.isfinite(t)
            val = np.zeros(len(o))
            if hit_mask.any():
                idx = fidx[hit_mask]
                nrm = fn[idx]
                mu = nrm @ view
                mu0 = nrm @ sun_body[k]
                lit_geo = (mu > 0) & (mu0 > 0)
                if shadows and lit_geo.any():
                    hp = o[hit_mask][lit_geo] + t[hit_mask][lit_geo, None] * d[hit_mask][lit_geo]
                    sh = _shadowed(hp, sun_body[k], v0f, v1f, v2f, chunk_t=chunk_t)
                    lit_full = lit_geo.copy()
                    lit_full[lit_geo] = ~sh
                else:
                    lit_full = lit_geo
                # A Lambertian surface leaves a radiance proportional to the illumination
                # cosine and independent of the direction it is seen from; the stored pixel
                # value is that radiance through a transfer of exponent gamma.
                val[hit_mask] = np.where(lit_full, np.abs(mu0) ** gamma, 0.0)
            if level is None:
                level = _otsu(val)          # the level the organisers take from frame one
            inten[ci, k] = float((val * (val > tau_i)).sum() * px_area)
            binar[ci, k] = float((val > level).sum() * px_area)
    out = []
    for ci, ct in enumerate(curve_types[:n_geom_types]):
        out.append(inten[ci] if ct == "intensity" else binar[ci])
    for ci, ct in enumerate(curve_types[n_geom_types:]):
        out.append(inten[ci] if ct == "intensity" else binar[ci])
    return np.stack(out[:len(curve_types)], axis=0)


def camera_vector_compat(cam):
    """Accept a plain (az, el) tuple as well as `conventions.Camera`."""
    from .conventions import camera_vector
    return camera_vector(*cam)


def convex_cross_check(hull_verts: np.ndarray, hull_faces: np.ndarray, m: int = 36,
                       geoms: list | None = None, res: int = 64, **kw) -> dict:
    """Compare this renderer against the analytic convex operator on a convex hull.

    On a convex body every camera-facing, sun-facing point is visible, so
    `mesh_curves_convex` is exact there and the two must agree up to pixel discretisation.
    Returns both raw curve stacks and the largest and mean absolute difference after
    per-curve mean normalisation.
    """
    from .shapes import mesh_curves_convex
    from .geometry import build_cameras

    geoms = geoms or cameras()
    types = (["intensity"] * len(geoms) + ["binary"] * len(geoms))
    mine = render_curves_mesh(hull_verts, hull_faces, m=m, curve_types=types, geoms=geoms,
                              res=res, **kw)
    ref_cams = build_cameras()[:len(geoms)]
    ref = mesh_curves_convex(hull_verts, hull_faces, ref_cams + ref_cams, m, types,
                             gamma=kw.get("gamma", TRANSFER_EXPONENT),
                             sigma=SENSE, delta=kw.get("delta", 1.0))

    def _norm(y):
        mbar = y.mean(axis=1, keepdims=True)
        return y / np.where(np.abs(mbar) > 1e-12, mbar, 1.0)

    rel = np.abs(_norm(mine) - _norm(ref))
    return {"mine": mine, "ref": ref, "max_abs_diff_normalised": float(rel.max()),
            "mean_abs_diff_normalised": float(rel.mean())}
