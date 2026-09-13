"""The noise estimator: it must read the noise and not the signal, and it must not be
confused by the smooth A/B mismatch that the two columns of a geometry differ by."""
import numpy as np
import pytest

from hac26.noise import NOISE_PROFILE, ab_mismatch, apply_noise, sigma_from_highfreq


def _curve(m=841, amp=0.35, harmonics=(2, 4, 6)):
    """A smooth, mean-normalised lightcurve-shaped signal at the real data's frame rate."""
    t = 2 * np.pi * np.arange(m) / m
    y = sum(np.cos(k * t + 0.7 * k) / k for k in harmonics)
    y = 1.0 + amp * y / np.abs(y).max()
    return y


def test_recovers_a_known_noise_level_on_a_smooth_curve():
    rng = np.random.default_rng(0)
    y = _curve()
    for sigma in (0.001, 0.005, 0.02):
        noisy = y + sigma * rng.standard_normal(len(y))
        est = float(sigma_from_highfreq(noisy[None])[0])
        assert est == pytest.approx(sigma, rel=0.15), f"sigma={sigma} estimated {est}"


def test_the_signal_itself_is_not_counted_as_noise():
    """The whole point of the second difference: on a noiseless curve the estimate must sit
    far below any realistic noise level, even though the curve's own slope between frames is
    of the same order as that level."""
    y = _curve()
    first_diff = 1.4826 * np.median(np.abs(np.diff(y) - np.median(np.diff(y)))) / np.sqrt(2)
    clean = float(sigma_from_highfreq(y[None])[0])
    assert first_diff > 1e-3          # what a first-difference estimator would read here
    assert clean < first_diff / 10    # and what the second difference reads instead
    assert clean < 1e-4               # an order of magnitude below the quietest real curve


def test_a_smooth_offset_between_two_curves_is_not_read_as_noise():
    """The A/B mismatch is low-frequency, so it must not enter sigma. It is what the pair
    difference measures, which is why that difference is not the noise estimate."""
    rng = np.random.default_rng(1)
    m = 841
    t = 2 * np.pi * np.arange(m) / m
    sigma = 0.002
    a = _curve() + sigma * rng.standard_normal(m)
    b = _curve() + 0.05 * np.cos(t + 0.3) + sigma * rng.standard_normal(m)   # smooth offset
    both = np.stack([a, b])

    est = sigma_from_highfreq(both)
    assert est == pytest.approx(sigma, rel=0.15)

    pair = float(ab_mismatch(np.concatenate([both, np.zeros((26, m))]))[0])
    assert pair > 10 * float(est.mean())


def test_outliers_do_not_inflate_the_estimate():
    """A handful of frames where the curve really does jump (a facet appearing, a shadow edge)
    must not set the level for the whole curve; the MAD is there for that."""
    rng = np.random.default_rng(2)
    sigma = 0.002
    y = _curve() + sigma * rng.standard_normal(841)
    y[[100, 101, 400, 620]] += 0.3
    assert float(sigma_from_highfreq(y[None])[0]) == pytest.approx(sigma, rel=0.2)


def test_missing_curves_take_the_median_of_the_present_ones():
    rng = np.random.default_rng(3)
    both = np.stack([_curve() + 0.002 * rng.standard_normal(841), np.zeros(841)])
    est = sigma_from_highfreq(both, mask=np.array([1.0, 0.0]))
    assert est[1] == pytest.approx(est[0])


def test_profile_layout_and_shape():
    """56 entries, four cameras per azimuth, intensity block then binary block, mean one."""
    assert NOISE_PROFILE.shape == (56,)
    for block in (NOISE_PROFILE[:28], NOISE_PROFILE[28:]):
        assert block.mean() == pytest.approx(1.0, abs=0.02)
        per_az = block.reshape(7, 4)
        assert np.allclose(per_az, per_az[:, :1])            # four cameras share an azimuth
        # az 135 and 225 (indices 3, 4) are the alpha = 135 deg geometries and the noisiest
        assert per_az[:, 0].argmax() in (3, 4)


def test_apply_noise_respects_the_profile():
    rng = np.random.default_rng(4)
    curves = np.ones((56, 400))
    added = apply_noise(curves, rng, 0.01, 0.01) - curves
    got = added.std(axis=1) / 0.01
    assert np.corrcoef(got, NOISE_PROFILE)[0, 1] > 0.99
