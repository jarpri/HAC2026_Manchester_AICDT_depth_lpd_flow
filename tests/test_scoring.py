"""The two scoring measures, and the guards on the bodies they are given.

The measures decide which reconstruction is shipped, so a defect here is silent: it moves a
number rather than raising, and the number is the only thing anyone reads. These pin the
failure modes that produced a wrong score rather than an error.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.recon import dice, mesh_occupancy                      # noqa: E402
from hac26.scoring.side_view import (boundary_distance, outline_extent,   # noqa: E402
                                     outline_set, measure_outlines, side_view_measure)
from hac26.shapes import rescale_touch_z                          # noqa: E402
from hac26.solvers.output import export_stl                       # noqa: E402


def ball(n=24, r=0.8, ext=1.0, centre=(0.0, 0.0, 0.0)):
    ax = (np.arange(n) + 0.5) / n * 2 * ext - ext
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    return ((X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2) < r ** 2


def cube_mesh(half=1.0):
    v = np.array([[x, y, z] for x in (-half, half) for y in (-half, half)
                  for z in (-half, half)], dtype=float)
    f = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                  [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    return v, f


# --------------------------------------------------------------- the pose


def test_rescale_touch_z_refuses_a_body_with_no_z_extent():
    """A flat body used to come back as inf and nan, which makes every occupancy grid empty;
    dice() reads two empty grids as a perfect match, so the degenerate body scored 1.0."""
    flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="zero z extent"):
        rescale_touch_z(flat, centre_xy=False)


def test_rescale_touch_z_is_a_similarity_about_the_axis():
    """x and y are scaled by the same factor as z and are not translated, so the rotation
    axis stays where it is -- the property voxel.py and side_view.py rely on."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(200, 3)) * np.array([0.4, 0.7, 2.0]) + np.array([0.3, -0.2, 1.0])
    w = rescale_touch_z(v, centre_xy=False)
    assert w[:, 2].min() == pytest.approx(-1.0)
    assert w[:, 2].max() == pytest.approx(1.0)
    s = (w[:, 0] / v[:, 0])
    assert np.allclose(s, s[0])                       # one uniform factor on x
    assert np.allclose(w[:, 1] / v[:, 1], s[0])       # the same one on y
    assert np.allclose(rescale_touch_z(w, centre_xy=False), w)   # idempotent


# --------------------------------------------------------------- the voxel measure


def test_dice_of_a_body_with_itself_is_one_and_disjoint_is_zero():
    a = ball()
    assert dice(a, a) == pytest.approx(1.0)
    assert dice(a, np.zeros_like(a)) == pytest.approx(0.0)


def test_dice_matches_the_challenge_formula():
    """1 - (#(A\\B) + #(B\\A)) / (#A + #B), which is what docs/challenge_info.md defines."""
    a, b = ball(r=0.8), ball(r=0.8, centre=(0.15, 0.0, 0.0))
    spec = 1.0 - ((a & ~b).sum() + (b & ~a).sum()) / (a.sum() + b.sum())
    assert dice(a, b) == pytest.approx(spec)


def test_mesh_occupancy_of_a_cube_fills_the_expected_fraction():
    v, f = cube_mesh(0.5)
    occ = mesh_occupancy(v, f, 48, 1.0)
    assert occ.sum() / occ.size == pytest.approx(0.125, abs=0.01)   # (1/2)^3 of the grid


# --------------------------------------------------------------- the side-view measure


def test_boundary_distance_is_symmetric():
    rng = np.random.default_rng(1)
    ca, cb = rng.normal(size=(120, 2)), rng.normal(size=(90, 2))
    assert boundary_distance(ca, cb) == pytest.approx(boundary_distance(cb, ca))


def test_a_body_against_itself_has_almost_no_outline_distance():
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(20000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    r = side_view_measure(pts, pts.copy(), n_dirs=6, res=128)
    assert r["assd_mean"] < 0.02
    assert r["n_dirs"] == 6


def test_outline_extent_bounds_every_projection():
    """A silhouette that reaches the frame is clipped rather than measured, so the extent has
    to cover the largest xy radius and the largest |z|, not the largest single coordinate."""
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(4000, 3)) * np.array([1.0, 1.0, 0.3])
    ext = outline_extent([pts])
    for e1, e2 in [(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
                   (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))]:
        assert np.abs(pts @ e1).max() <= ext
        assert np.abs(pts @ e2).max() <= ext


def test_a_wider_body_reads_as_further_away():
    """The measure has to order bodies by how different their outlines are; without that it
    cannot rank candidates."""
    rng = np.random.default_rng(4)
    d = rng.normal(size=(30000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    near = d * np.array([1.05, 1.05, 1.0])
    far = d * np.array([1.30, 1.30, 1.0])
    ext = outline_extent([d, near, far])
    o, on, of = (outline_set(p, ext, n_dirs=6, res=256) for p in (d, near, far))
    assert (measure_outlines(o, on)["assd_mean"]
            < measure_outlines(o, of)["assd_mean"])


# --------------------------------------------------------------- what may be written


def test_export_stl_repairs_an_open_mesh_it_can_close(tmp_path):
    """A hole that can be filled is filled, and the report says so."""
    v, f = cube_mesh(0.5)
    rep = export_stl(str(tmp_path / "holed.stl"), v, f[:-2])   # a cube missing one side
    assert rep["filled_holes"] and rep["watertight"]
    assert rep["volume"] == pytest.approx(1.0, rel=1e-6)


def test_export_stl_refuses_a_body_it_cannot_close(tmp_path):
    """An open body inverts the parity scan that decides inside, so it scores wrongly rather
    than failing; a flat sheet encloses nothing at all. Three of the ten shipped
    reconstructions were written open, with a fifth of their faces degenerate."""
    v, f = cube_mesh()
    out = tmp_path / "sheet.stl"
    with pytest.raises(ValueError, match="not a closed solid"):
        export_stl(str(out), v, f[:1])             # one triangle: no solid to write
    assert not out.exists()


def test_export_stl_writes_a_positive_volume_solid(tmp_path):
    v, f = cube_mesh(0.5)
    rep = export_stl(str(tmp_path / "cube.stl"), v, f)
    assert rep["watertight"] and rep["volume"] > 0
    assert rep["volume"] == pytest.approx(1.0, rel=1e-6)


def test_export_stl_corrects_inverted_winding(tmp_path):
    """Marching cubes hands back the opposite winding from the draws; the volume in the
    report has to describe the body on disk, so it is measured after the repair."""
    v, f = cube_mesh(0.5)
    rep = export_stl(str(tmp_path / "flipped.stl"), v, f[:, ::-1])    # inside out
    assert rep["volume"] == pytest.approx(1.0, rel=1e-6)
