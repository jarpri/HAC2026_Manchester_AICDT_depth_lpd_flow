#!/usr/bin/env python3
"""The organisers' two measures, as released in the challenge data.

`dataset/raw/Evaluation_measures/` holds the code the challenge is scored with, a Python
voxel measure and a MATLAB projection measure. This module carries the first over verbatim
and ports the second, so a reconstruction is scored here with the same arithmetic rather
than with an approximation of it.

Two properties of the released code decide how its numbers are read.

The projection measure as released looks down the rotation axis. `twoDmetric.m` rotates both
meshes about z by theta and then projects onto the xy plane, and a rotation about z followed
by a projection onto xy is an in-plane rotation of one and the same silhouette, so theta
changes the picture's orientation and nothing else. `projection_score` reproduces that, and
`projection_score_axis` offers the reading the challenge text implies instead, side views
with the camera swung around the body, so the difference can be measured. A recipe is scored
against the code as released, because that is what will be run, and is not allowed to
depend on the degeneracy.

Neither measure cares which way a face is wound. The voxel measure voxelises by subdividing
triangles and then flood-fills; the projection measure rasterises each triangle with
`poly2mask`. What the voxel measure is not blind to is a surface that does not close, since
`.fill()` leaks through a hole, which is why `scripts/check_submission.py` tests for both.

The challenge sums, per model, one voxel score and one projection score, each in [0, 1],
higher better. The released voxel code returns distances, lower better; `measure2` is
1 - Dice, so the published formula `1 - (#(A\\B) + #(B\\A))/(#A + #B)` is `1 - measure2`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IMG_SIZE = 500       # twoDmetric.m rasterisation resolution
N_RESAMPLE = 1000    # twoDmetric.m boundary resampling points
MAX_FACES_RASTER = 200_000   # decimation cap for the projection rasteriser; see _draw


def load_as_trimesh(path):
    """The organisers' loader: meshio, then trimesh over its triangle cells."""
    import meshio
    import trimesh
    m = meshio.read(str(path))
    return trimesh.Trimesh(vertices=m.points, faces=m.cells_dict["triangle"])


# --------------------------------------------------------------- voxel measure (verbatim)
def relative_volume_difference_voxelized(meshname1, meshname2, pitch=10):
    """Verbatim transcription of Evaluation_measures/Voxel measure - Python/metrics.py.

    Returns (measure1, measure2) = (1 - IoU, 1 - Dice); lower is better in both.
    """
    import trimesh
    A = load_as_trimesh(meshname1)
    B = load_as_trimesh(meshname2) if isinstance(meshname2, (str, Path)) else meshname2

    min_bound = np.minimum(A.bounds[0], B.bounds[0])
    max_bound = np.maximum(A.bounds[1], B.bounds[1])

    bbox1 = trimesh.creation.box(extents=np.array([pitch, pitch, pitch]))
    bbox1.apply_translation(min_bound)
    bbox2 = trimesh.creation.box(extents=np.array([pitch, pitch, pitch]))
    bbox2.apply_translation(max_bound)

    A_padded = trimesh.util.concatenate([A, bbox1, bbox2])
    B_padded = trimesh.util.concatenate([B, bbox1, bbox2])

    vA = A_padded.voxelized(pitch, method="subdivide").fill()
    vB = B_padded.voxelized(pitch, method="subdivide").fill()
    filled_A = vA.matrix.astype(bool)
    filled_B = vB.matrix.astype(bool)

    vol_voxel = pitch ** 3
    vol_D1 = np.sum(np.logical_and(filled_A, np.logical_not(filled_B))) * vol_voxel
    vol_D2 = np.sum(np.logical_and(filled_B, np.logical_not(filled_A))) * vol_voxel
    vol_AandB = np.sum(np.logical_and(filled_A, filled_B)) * vol_voxel
    vol_AorB = np.sum(np.logical_or(filled_A, filled_B)) * vol_voxel
    vol_A = np.sum(filled_A) * vol_voxel
    vol_B = np.sum(filled_B) * vol_voxel

    measure1 = 1 - vol_AandB / vol_AorB
    measure2 = (vol_D1 + vol_D2) / (vol_A + vol_B)
    return measure1, measure2


def voxel_score(truth_stl, recon_stl, pitch: float = 0.05) -> float:
    """The challenge's voxel score, `1 - measure2` = Dice. Higher is better, 1 is perfect."""
    _, m2 = relative_volume_difference_voxelized(truth_stl, recon_stl, pitch=pitch)
    return float(1.0 - m2)


# ----------------------------------------------------------- projection measure (ported)
def _draw(pts_xy, faces, img_size: int) -> np.ndarray:
    """`poly2mask` over every face, unioned: a boolean silhouette image.

    MATLAB ORs one `poly2mask` per triangle. Doing that literally over the released truth
    meshes (Vesta is 800k faces) is minutes per view, so a mesh above MAX_FACES_RASTER is
    decimated first. The union of projected triangles is the filled silhouette either way, and
    `imfill` + largest-component follow, so decimation moves the outline by well under a pixel
    at this resolution -- `projection_score(..., check_convergence=True)` measures that rather
    than assuming it.
    """
    from PIL import Image, ImageDraw
    img = Image.new("1", (img_size, img_size), 0)
    d = ImageDraw.Draw(img)
    for tri in pts_xy[faces]:
        d.polygon([(float(x), float(y)) for x, y in tri], fill=1)
    return np.asarray(img, dtype=bool)


def _largest_filled(mask: np.ndarray) -> np.ndarray:
    """`imfill(BW,'holes')` then `bwareafilt(BW,1)`."""
    from scipy import ndimage
    filled = ndimage.binary_fill_holes(mask)
    lab, n = ndimage.label(filled)
    if n <= 1:
        return filled
    sizes = ndimage.sum(filled, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def _boundary_resampled(mask: np.ndarray, n: int) -> np.ndarray:
    """`bwboundaries` then `resampleByArclength`, as (n, 2) in (col, row) pixel units."""
    from skimage import measure
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.zeros((0, 2))
    c = max(contours, key=len)                       # bwboundaries{1}: the outer boundary
    xy = np.column_stack([c[:, 1], c[:, 0]])         # (col, row), as MATLAB's [:,2],[:,1]
    d = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return np.repeat(xy[:1], n, axis=0)
    s_new = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(s_new, s, xy[:, 0]), np.interp(s_new, s, xy[:, 1])])


def _decimate_for_raster(v, f):
    from hac26.forward.mesh.exact import decimate
    if len(f) <= MAX_FACES_RASTER:
        return np.asarray(v, float), np.asarray(f, np.int64)
    dv, df = decimate(np.asarray(v, float), np.asarray(f, np.int64), MAX_FACES_RASTER)
    return np.asarray(dv, float), np.asarray(df, np.int64)


def _project(v, theta_deg: float, axis: str):
    """Rotate and project, returning the 2-D coordinates the measure rasterises.

    `axis="z"` is the released code: rotate about z, keep (x, y). `axis="side"` is the reading
    the challenge text implies: swing the camera around the body in the equatorial plane and
    keep (horizontal, z).
    """
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "z":
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return (v @ rz.T)[:, :2]
    if axis == "side":
        return np.column_stack([v[:, 0] * c + v[:, 1] * s, v[:, 2]])
    raise ValueError(f"unknown axis {axis!r}")


def projection_score(truth_stl, recon_stl, theta: float = 0.0, axis: str = "z",
                     img_size: int = IMG_SIZE, n: int = N_RESAMPLE) -> float:
    """`twoDmetric.m` in Python: 1 - RMS boundary deviation / the truth's bbox diagonal.

    Higher is better, 1 is perfect, floored at 0. Both meshes are centred on their vertex mean
    -- translation only, no scaling and no PCA -- exactly as the released code does.
    """
    tv_all, tf_all = _mesh_arrays(truth_stl)
    rv_all, rf_all = _mesh_arrays(recon_stl)
    # V - mean(V,1): the mean over VERTICES, which is what the released code uses
    tv_all = tv_all - tv_all.mean(0)
    rv_all = rv_all - rv_all.mean(0)
    tv, tf = _decimate_for_raster(tv_all, tf_all)
    rv, rf = _decimate_for_raster(rv_all, rf_all)

    pg = _project(tv, theta, axis)
    pr = _project(rv, theta, axis)

    xmin, ymin = pg.min(0)
    xmax, ymax = pg.max(0)
    gt_diag = float(np.hypot(xmax - xmin, ymax - ymin))
    margin = 0.05 * gt_diag
    xmin -= margin; xmax += margin; ymin -= margin; ymax += margin
    rng = max(xmax - xmin, ymax - ymin)

    def to_px(p):
        return np.column_stack([(p[:, 0] - xmin) / rng * (img_size - 1),
                                (p[:, 1] - ymin) / rng * (img_size - 1)])

    bw_gt = _largest_filled(_draw(to_px(pg), tf, img_size))
    bw_rc = _largest_filled(_draw(to_px(pr), rf, img_size))
    if not bw_gt.any() or not bw_rc.any():
        return 0.0

    gt_new = _boundary_resampled(bw_gt, n)
    rc_new = _boundary_resampled(bw_rc, n)
    if len(gt_new) == 0 or len(rc_new) == 0:
        return 0.0

    from scipy.spatial import cKDTree
    d1 = cKDTree(gt_new).query(rc_new)[0]
    d2 = cKDTree(rc_new).query(gt_new)[0]
    rms = float(np.sqrt(np.mean(np.concatenate([d1 ** 2, d2 ** 2]))))
    gt_diag_px = gt_diag / rng * (img_size - 1)
    return float(max(0.0, 1.0 - rms / gt_diag_px))


def projection_score_axis(truth_stl, recon_stl, n_dirs: int = 8, **kw) -> dict:
    """The projection score under both readings, averaged over `n_dirs` angles.

    Returns {'released': ..., 'released_spread': ..., 'side': ..., 'side_worst': ...}. The
    released reading should be flat in theta; the spread reports whether it is.
    """
    ang = np.linspace(0.0, 360.0, n_dirs, endpoint=False)
    rel = [projection_score(truth_stl, recon_stl, t, axis="z", **kw) for t in ang]
    side = [projection_score(truth_stl, recon_stl, t, axis="side", **kw) for t in ang]
    return {"released": float(np.mean(rel)), "released_spread": float(np.ptp(rel)),
            "side": float(np.mean(side)), "side_worst": float(np.min(side))}


def _mesh_arrays(stl):
    import trimesh
    if hasattr(stl, "vertices"):
        return np.asarray(stl.vertices, float), np.asarray(stl.faces, np.int64)
    m = trimesh.load(str(stl), process=False)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
