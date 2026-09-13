"""Which released channel a body is inverted from, and that the convex stage's recipe
produces a body in the submission pose from it."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hac26.conventions import CYLINDER_R  # noqa: E402
from hac26.data_io import (N_CAMS, load_inversion_curves, load_model_curves,  # noqa: E402
                           write_curves29)

DATA = "dataset/raw"
CKPT = "models/lpd_convex.pt"


def _write(root: Path, model: int, blender: bool, types=("intensity", "binary"), seed=0):
    """Curve files of one model under the released directory layout, with values that
    differ between the two channels so a test can tell which one was read."""
    rng = np.random.default_rng(seed + (100 if blender else 0))
    d = root / f"AsteroidModel{model:02d}_shape_secret" / f"Asteroid{model}_lightcurve_data"
    d.mkdir(parents=True, exist_ok=True)
    suffix = "_blender" if blender else ""
    for t in types:
        c = 1.0 + 0.1 * rng.standard_normal((N_CAMS, 36))
        write_curves29(str(d / f"Asteroid{model:02d}_lightcurve_{t}{suffix}.txt"),
                       np.arange(36.0), c)


def test_the_render_is_read_when_both_its_files_are_present(tmp_path):
    _write(tmp_path, 4, blender=False)
    _write(tmp_path, 4, blender=True)
    d = load_inversion_curves(str(tmp_path), 4, m=36)
    assert d["channel"] == "blender"
    assert d["mask"].sum() == 2 * N_CAMS
    ref = load_model_curves(str(tmp_path), 4, m=36, use_blender=True)
    assert np.allclose(d["curves"], ref["curves"])
    lab = load_model_curves(str(tmp_path), 4, m=36, use_blender=False)
    assert not np.allclose(d["curves"], lab["curves"])


def test_the_lab_curves_are_the_fallback_when_the_render_is_withheld(tmp_path):
    _write(tmp_path, 7, blender=False)
    d = load_inversion_curves(str(tmp_path), 7, m=36)
    assert d["channel"] == "real"
    assert d["mask"].sum() == 2 * N_CAMS
    with pytest.raises(FileNotFoundError):
        load_inversion_curves(str(tmp_path), 7, m=36, channel="blender")


def test_a_half_released_render_does_not_replace_the_lab_curves(tmp_path):
    _write(tmp_path, 8, blender=False)
    _write(tmp_path, 8, blender=True, types=("intensity",))
    d = load_inversion_curves(str(tmp_path), 8, m=36)
    assert d["channel"] == "real" and d["mask"].sum() == 2 * N_CAMS
    forced = load_inversion_curves(str(tmp_path), 8, m=36, channel="blender")
    assert forced["channel"] == "blender" and forced["mask"].sum() == N_CAMS


def test_the_channel_can_be_forced_and_must_be_named(tmp_path):
    _write(tmp_path, 9, blender=False)
    _write(tmp_path, 9, blender=True)
    d = load_inversion_curves(str(tmp_path), 9, m=36, channel="real")
    assert d["channel"] == "real"
    with pytest.raises(ValueError):
        load_inversion_curves(str(tmp_path), 9, m=36, channel="simulated")


@pytest.mark.skipif(not Path(DATA).exists() or not Path(CKPT).exists(),
                    reason="challenge data or the convex checkpoint not present")
def test_the_convex_recipe_poses_the_body_and_reads_the_render():
    from reconstruct import load_checkpoints, reconstruct_convex
    loaded = load_checkpoints([CKPT])
    v, f, info = reconstruct_convex(3, DATA, loaded)
    assert info["channel"] == "blender"
    assert abs(v[:, 2].min() + 1.0) < 1e-6 and abs(v[:, 2].max() - 1.0) < 1e-6
    r = np.sqrt((v[:, :2] ** 2).sum(1)).max()
    assert abs(r - CYLINDER_R[3]) < 1e-6
    v_lab, _, info_lab = reconstruct_convex(3, DATA, loaded, channel="real")
    assert info_lab["channel"] == "real"
    # the two channels are different recordings, so the two answers differ
    assert v_lab.shape != v.shape or not np.allclose(v_lab, v)


def test_the_two_horizontal_columns_of_a_render_are_one_curve():
    """Every azimuth is recorded by two columns of the same horizontal camera, and in the
    released renders they carry the same numbers. Summing a likelihood over all 28 would give
    those seven geometries twice the weight of the rest, so the repeat is masked out and the
    render's 42 distinct curves are what is fitted. In the laboratory files the two columns
    are two recordings and nothing is masked."""
    from hac26.data_io import distinct_geometries

    groups = distinct_geometries()
    assert len(groups) == 21
    assert sorted(c for g in groups for c in g) == list(range(N_CAMS))

    for model in (1, 3, 10):
        blender = load_inversion_curves(DATA, model, m=48, channel="blender")
        for g in groups:
            for first, other in ((g[0], c) for c in g[1:]):
                for block in (0, N_CAMS):
                    assert np.array_equal(blender["curves"][first + block],
                                          blender["curves"][other + block])
                    assert blender["mask"][other + block] == 0.0
                    assert blender["mask"][first + block] == 1.0
        assert len(blender["duplicate_columns"]) == 14
        lab = load_inversion_curves(DATA, model, m=48, channel="real")
        assert lab["duplicate_columns"] == []
        assert lab["mask"].sum() == 2 * N_CAMS


def test_a_count_curve_that_collapses_to_zero_is_refused():
    """Otsu's threshold is taken on the first frame and applied to the rest. On a body with
    large flat facets it can split the body's own brightness range instead of separating the
    body from the background, and the count then falls to nothing at the phases where no
    facet is bright enough. Model 2's released count curves do that; no other model's do, and
    the geometry keeps its intensity curve either way."""
    from hac26.data_io import count_curve_is_usable
    from reconstruct_lpd import curve_weight, geometry_mask, measured_geometries

    cube = load_inversion_curves(DATA, 2, m=48, channel="blender")
    assert len(cube["count_curves_refused"]) == 18
    assert all(c >= N_CAMS for c in cube["count_curves_refused"])
    w = curve_weight(cube["mask"])
    assert float(w[:, 0].sum()) == 21.0 and float(w[:, 1].sum()) == 8.0
    # The intensity curves survive, and a solver fits every geometry that has one: it selects
    # curves one at a time through curve_weight, so a geometry with one good curve is a
    # geometry with a measurement in it.
    assert len(measured_geometries(cube["mask"])) == 21
    # The flow's conditioning cannot say that: it has one flag per geometry, so a geometry
    # marked present with an absent curve would show the network a zero residual there and
    # read as a perfect fit. It therefore requires both curves, and on this body that is
    # eight geometries rather than twenty-one.
    assert int(geometry_mask(cube["mask"]).sum()) == 8

    for model in (1, 3, 4, 5, 6, 7, 8, 9, 10):
        d = load_inversion_curves(DATA, model, m=48, channel="blender")
        assert d["count_curves_refused"] == []

    assert count_curve_is_usable(np.array([0.9, 1.0, 1.1]))
    assert not count_curve_is_usable(np.array([0.0, 0.0, 3.0]))


def test_the_held_out_cameras_are_a_fixed_spread_over_the_camera_ordering():
    """Which cameras a fit is tested on is part of the measurement: two runs of one body have
    to be tested on the same ones or their numbers do not compare, and the held-out set has
    to span the viewing geometries or the test is concentrated in one corner of them."""
    from hac26.conventions import cameras
    from hac26.data_io import held_out_geoms
    cams = cameras()
    present = [i for i in range(len(cams)) if cams[i].kind != "hor_b"]
    held = held_out_geoms(present, 5)
    assert held == held_out_geoms(present, 5)                 # fixed
    assert len(held) == 5 and set(held) <= set(present)
    assert len({cams[i].azimuth_deg for i in held}) == 5      # five different azimuths
    assert len({cams[i].kind for i in held}) > 1              # and not one camera kind
    assert held_out_geoms(present, 0) == []
    # a prefix property is not claimed and is not needed; what is needed is that nothing is
    # held out twice and that something is left to fit on
    assert len(set(held_out_geoms(present, 8))) == 8
    with pytest.raises(ValueError):
        held_out_geoms(present, len(present))
