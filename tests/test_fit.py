"""The fit of the depths to a body: it has to reach whatever depth the body has, not whatever
depth a step budget allows, and it has to reproduce a body its own hull gets badly wrong.

The estimator is what is under test, and it does not depend on how many nodes the sphere has,
so these run on a small one. The production node set has thousands and the solve is a dense
system in that many unknowns, which needs a dozen sample points per unknown to be determined at
all; carrying that here would test the machine it runs on."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hac26.field import (EXTRACT_EXTENT, ImplicitBody, core_centre,          # noqa: E402
                         extract_mesh)
from hac26.recon import dice, mesh_occupancy                                # noqa: E402
from fit_shapes import DICE_EXTENT, DepthFit, POINTS_PER_NODE, sample_arrays  # noqa: E402

NODES = 256
N_PTS = POINTS_PER_NODE * NODES


def _dumbbell(res=96, extent=1.2, r=0.62, half=0.42):
    """Two overlapping spheres, as the level set of the smaller of their two distances: a
    body with a waist no convex body has."""
    from skimage import measure
    ax = np.linspace(-extent, extent, res)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    d = np.minimum(np.sqrt((x + half) ** 2 + y ** 2 + z ** 2),
                   np.sqrt((x - half) ** 2 + y ** 2 + z ** 2)) - r
    v, f, _, _ = measure.marching_cubes(d, level=0.0, spacing=(ax[1] - ax[0],) * 3)
    return v - extent, np.asarray(f, dtype=np.int64)


def _fit(verts, faces, n_pts=N_PTS, res=48):
    fit = DepthFit("cpu", n_nodes=NODES)
    normals = ImplicitBody().core.n.numpy()
    pts = sample_arrays(verts, faces, n_pts=n_pts, seed=0)
    h = np.maximum((verts @ normals.T).max(axis=0), 1e-3).astype(np.float32)
    a, before, after = fit.solve(torch.tensor(pts), h)
    body = ImplicitBody(n_nodes=NODES)
    body.set_support(torch.tensor(h), centre=core_centre(h, normals))
    with torch.no_grad():
        body.delta.a.copy_(a)
    v, f = extract_mesh(lambda y: body(y), EXTRACT_EXTENT, res=res, device="cpu")
    d = dice(mesh_occupancy(v, f, res, DICE_EXTENT),
             mesh_occupancy(verts, faces, res, DICE_EXTENT)) if len(f) >= 8 else 0.0
    return a, before, after, d


def test_the_fit_reproduces_a_body_with_a_waist():
    """The depths of a two-lobed body are solved, not descended to, so the waist comes back
    rather than being averaged away, and the residual of the field over the body's own surface
    falls by a large factor."""
    v, f = _dumbbell()
    a, before, after, d = _fit(v, f)
    assert d > 0.9, d
    assert after < 0.2 * before, (before, after)
    # the hull of this body bridges the waist, so carving it needs real depth
    assert float(a.max()) > 0.1


def test_a_convex_body_needs_almost_no_depth():
    """The core alone is already the answer for a convex body, so the fit leaves the depths
    near zero and does not carve something that is not there."""
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=2, radius=0.8)
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    a, before, after, d = _fit(v, f)
    assert d > 0.95, d
    assert float(a.abs().max()) < 0.05, float(a.abs().max())


def test_the_residual_before_the_fit_is_how_far_the_body_is_from_its_hull():
    """The fit's target is an identity and not an approximation: on the body's own surface the
    field has to vanish, so the depth wanted there is exactly minus the core's field. The
    residual the solve reports before fitting is therefore the distance from the body's surface
    out to its hull, which is zero for a convex body and large for a waisted one."""
    import trimesh
    ball = _fit(*_dumbbell())[1]
    m = trimesh.creation.icosphere(subdivisions=2, radius=0.8)
    convex = _fit(np.asarray(m.vertices), np.asarray(m.faces))[1]
    # a convex body is its own hull to the tolerance of its own triangulation; the waisted one
    # is an order of magnitude further from hers
    assert convex < 0.01 and ball > 10.0 * convex


def test_blocking_the_core_does_not_change_the_solve():
    """The core over the sample points is evaluated a block at a time, so the memory the solve
    needs is set by the block size and not by how many points the body is fitted from. It must
    give the same depths and the same residual however it is blocked."""
    import fit_shapes
    v, f = _dumbbell()
    fit = DepthFit("cpu", n_nodes=NODES)
    normals = ImplicitBody().core.n.numpy()
    pts = sample_arrays(v, f, n_pts=N_PTS, seed=0)
    h = np.maximum((v @ normals.T).max(axis=0), 1e-3).astype(np.float32)
    whole = fit_shapes.SOLVE_CHUNK
    try:
        fit_shapes.SOLVE_CHUNK = 10 ** 9
        a1, _, r1 = fit.solve(torch.tensor(pts), h)
        fit_shapes.SOLVE_CHUNK = 700
        a2, _, r2 = fit.solve(torch.tensor(pts), h)
    finally:
        fit_shapes.SOLVE_CHUNK = whole
    assert abs(r1 - r2) < 1e-4 * max(r1, 1e-9)
    assert float((a1 - a2).abs().max()) < 1e-4 * max(float(a1.abs().max()), 1e-9)
