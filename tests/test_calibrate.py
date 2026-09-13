"""The calibration's convergence report: it has to notice when the fit stopped because the
step budget ran out rather than because the data was satisfied."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "calibrate", Path(__file__).resolve().parents[1] / "scripts" / "calibrate.py")
calibrate = importlib.util.module_from_spec(_spec)
sys.modules["calibrate"] = calibrate
_spec.loader.exec_module(calibrate)


def _report(start, now, budget=4.5):
    s = {k: torch.tensor(v) for k, v in start.items()}
    n = {k: torch.tensor(v) for k, v in now.items()}
    return calibrate.movement_report(s, n, {k: budget for k in start})


def test_movement_is_measured_in_raw_space_against_steps_times_lr():
    rep = _report({"raw_eye": 8.0}, {"raw_eye": 9.5}, budget=4.5)
    assert rep["raw_eye"]["moved"] == pytest.approx(1.5)
    assert rep["raw_eye"]["fraction"] == pytest.approx(1.5 / 4.5)


def test_a_parameter_that_spends_most_of_its_budget_is_flagged():
    """With a travel budget of steps times lr, a parameter that has moved most of it was
    still moving when the run ended and is named; one that moved little is not."""
    rep = _report({"raw_rho": -1.386, "raw_eye": 8.0, "raw_tau_i": -3.892},
                  {"raw_rho": 2.330, "raw_eye": 8.510, "raw_tau_i": -4.345})
    assert calibrate.print_movement(rep) == ["raw_rho"]
    assert rep["raw_rho"]["fraction"] > 0.8


def test_a_settled_fit_is_not_flagged():
    rep = _report({"raw_rho": -1.386, "raw_eye": 8.0}, {"raw_rho": -1.2, "raw_eye": 8.3})
    assert calibrate.print_movement(rep) == []


def test_a_vector_parameter_reports_its_largest_component():
    rep = _report({"raw_oetf": [0.0, 0.0, 0.0]}, {"raw_oetf": [0.1, -3.0, 0.4]})
    assert rep["raw_oetf"]["moved"] == pytest.approx(3.0)


# ---------------------------------------------------------------- alignment diagnostic

def _curves(P, shifts_deg):
    """(28, 2, P) curves, each azimuth group shifted by its own amount."""
    import numpy as np
    t = 2 * np.pi * np.arange(P) / P
    base = 1.0 + 0.3 * np.cos(2 * t + 0.4) + 0.15 * np.cos(5 * t)
    out = torch.zeros(28, 2, P)
    for i, az in enumerate((0, 45, 90, 135, 225, 270, 315)):
        f = shifts_deg[az] * P / 360.0
        idx = (np.arange(P) - f) % P
        y = np.interp(idx, np.arange(P + 1), np.r_[base, base[0]])
        for j in range(4):
            out[4 * i + j, 0] = torch.tensor(y, dtype=torch.float32)
            out[4 * i + j, 1] = torch.tensor(y, dtype=torch.float32)
    return out


def test_an_aligned_set_reports_no_offset():
    P = 96
    zero = dict.fromkeys((0, 45, 90, 135, 225, 270, 315), 0.0)
    c = _curves(P, zero)
    off = calibrate.phase_offset_report(c, c.clone(), torch.ones(28, dtype=torch.bool),
                                        torch.full((28, 2), 0.01))
    assert all(v == 0.0 for v in off.values()), off


def test_a_per_azimuth_misalignment_is_recovered():
    """What the organisers' 17/25 Aug realignment of model 1 looked like: a different
    whole-frame shift per azimuth, which no single psi0 can absorb."""
    P = 96
    truth = {0: -5.14, 45: 0.0, 90: 1.28, 135: 1.28, 225: -1.71, 270: -0.86, 315: 0.0}
    pred = _curves(P, dict.fromkeys(truth, 0.0))
    real = _curves(P, truth)
    off = calibrate.phase_offset_report(pred, real, torch.ones(28, dtype=torch.bool),
                                        torch.full((28, 2), 0.01))
    res = 360.0 * 0.125 / P                       # the search's own resolution
    for az, want in truth.items():
        got = off[str(float(az))]
        assert abs(got - want) <= res + 1e-9, f"az {az}: reported {got:+.2f}, want {want:+.2f}"


def test_a_common_offset_is_reported_on_every_azimuth():
    """A psi0 that is simply wrong shifts every group by the same amount, which is what
    distinguishes it from curves that disagree with each other."""
    P = 96
    common = dict.fromkeys((0, 45, 90, 135, 225, 270, 315), 3.75)
    off = calibrate.phase_offset_report(_curves(P, dict.fromkeys(common, 0.0)),
                                        _curves(P, common),
                                        torch.ones(28, dtype=torch.bool),
                                        torch.full((28, 2), 0.01))
    assert len(set(off.values())) == 1 and abs(list(off.values())[0] - 3.75) < 0.5, off


def test_the_worst_residual_masks_geometries_and_not_columns():
    """A curve carries a geometry, a column and a phase, and `present` is one flag per
    geometry, so the mask belongs on the first axis. Broadcast from the right it meets the
    column axis instead, which raises on this rig and would silently mask the wrong thing on
    one where the two counts happened to agree.

    The result has to be shaped like the model error it is compared against, one number per
    geometry and column, and a geometry that was never recorded has to contribute nothing.
    """
    from hac26.data_io import N_CAMS
    P = 48
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(N_CAMS, 2, P, generator=g)
    real = torch.randn(N_CAMS, 2, P, generator=g)
    present = torch.zeros(N_CAMS, dtype=torch.bool)
    present[:21] = True                    # what the Blender channel leaves on model 1

    out = calibrate.curve_residual(pred, real, present)
    assert out.shape == (N_CAMS, 2)
    assert bool((out[21:] == 0).all()), "an unrecorded geometry contributed a residual"
    assert bool((out[:21] > 0).all())

    ref = torch.zeros(N_CAMS, 2)
    for i in range(N_CAMS):
        if present[i]:
            ref[i] = (pred[i] - real[i]).pow(2).mean(-1).sqrt()
    assert torch.allclose(out, ref, atol=1e-6)
    # zeroing is what lets the caller take a maximum over bodies: a geometry with no data can
    # never win one, so it never raises the model error a curve is given
    assert float(out.max()) == float(out[:21].max())


def test_a_channel_with_no_independent_pair_still_calibrates():
    """The A/B mismatch is the disagreement between two independent recordings of the body in
    its two mountings. A deterministic render has no second recording -- the duplicated column
    comes back identical and the duplicate drop then leaves no pair with both columns -- so
    the quantity is undefined rather than small. It is a diagnostic, printed beside the noise
    and entering neither the fit nor the model error, so measuring it is what may fail and the
    calibration is what must not."""
    import numpy as np
    from hac26.data_io import N_CAMS
    from hac26.noise import ab_mismatch
    curves = np.random.default_rng(0).normal(size=(2 * N_CAMS, 48))
    with pytest.raises(ValueError):
        ab_mismatch(curves, np.zeros(2 * N_CAMS))
    assert np.isfinite(ab_mismatch(curves, np.ones(2 * N_CAMS))).all()


def test_a_measured_instrument_has_nothing_to_fit_through_the_render():
    """The rendered channel's instrument is measured against the release rather than fitted
    against it, so fitted_parameters() is empty there: a render has no penumbra, no lens
    falloff, no point spread and no spline transfer, and each of those would be a direction a
    fit could use to absorb an error of shape.

    The calibration therefore has no render-path parameter to move on that channel, which is
    what lets it hold the curves fixed and fit the model error alone. The laboratory channel
    does have them, and must."""
    from hac26.forward.mesh.instrument import Instrument
    rendered = Instrument.blender_start()
    assert rendered.fitted_parameters() == []
    assert not bool(rendered.interreflection) and bool(rendered.orthographic)

    lab = Instrument()
    assert [n for n, _ in lab.fitted_parameters()], "the laboratory chain has nothing to fit"
