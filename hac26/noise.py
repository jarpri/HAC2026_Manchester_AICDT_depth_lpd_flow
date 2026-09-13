"""Measurement noise of the lightcurves.

The noise is estimated from the high-frequency content of each curve on its own. A lightcurve
is smooth on the frame scale -- the body turns by less than half a degree between frames --
so the second difference

    d_i = x[i-1] - 2 x[i] + x[i+1],    sigma_c = 1.4826 * MAD(d) / sqrt(6)

is dominated by noise. The second difference rather than the first, because the first still
carries the slope of the signal: on a curve of amplitude 0.3 with a few harmonics that slope
is of the same order as the noise, and on the faceted bodies it is larger. The second
difference annihilates anything locally linear and leaves the curvature, which is smaller by
another factor of the frame step. The median absolute deviation rather than the RMS, so that
the few frames where the curve really does turn a corner (a facet coming into view, a shadow
edge crossing) do not set the level for the whole curve.

This must be done on the curves at their native frame rate. The released real curves have
~841 frames per revolution; resampling them to the operator's phase grid first would leave
the second difference measuring the curvature of the signal rather than the noise.

Why not the co-located pair. At each azimuth two columns hold the same nominal geometry, and
their difference looks like the obvious noise estimate. It is not one: there were only two
cameras, one horizontal and one looking down, and the four columns of an azimuth come from
two *separate* recordings of the body in two mountings (orientation A and B), aligned
afterwards by a time reversal and a shift. Each public model ships 28 real videos
(CAM1/CAM2 x 1A/1B x 7 angles) but only 21 simulated ones, which is the same statement.
So the pair difference carries the A/B mounting mismatch, the residual alignment error and
the stem. On the released curves it runs 1-277x the noise (median 12x), and 86-99% of its
power sits below the frame scale, which is what says it is structure and not noise. Using it
as sigma understates the misfit of anything compared against it and, through the likelihood,
silently discards the high-phase-angle geometries -- the ones with the longest shadows and
the most shape in them. It is kept here as `ab_mismatch`, which is what it measures; it
belongs in the model-error term eta, not in sigma.

NOISE_PROFILE is the per-azimuth shape of sigma, measured on the three public models with
`sigma_from_highfreq` and normalised to mean 1; training uses it to distribute synthetic
noise across the curves, at an overall level drawn per body from [NOISE_LO, NOISE_HI], a
range that covers the levels measured on the public models. It is a snapshot of the data and
can be recomputed from the public curves at native resolution with `sigma_from_highfreq`.
"""
from __future__ import annotations

import numpy as np

__all__ = ["AZIMUTHS_DEG", "NOISE_PROFILE", "NOISE_LO", "NOISE_HI", "sigma_from_highfreq",
           "ab_mismatch", "apply_noise"]

AZIMUTHS_DEG = (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0)
NOISE_LO, NOISE_HI = 0.0005, 0.004    # range of the mean noise level of a mean-normalised curve

# Per-azimuth median of sigma over the public models, normalised to mean 1. Curve layout:
# the intensity curves then the binary curves, four cameras per azimuth in the order
# (horizontal a, horizontal b, top, bottom). The profile rises with the solar phase angle
# (az 135 and 225 are both alpha = 135 deg), which is what a photon-limited measurement of a
# mostly-shadowed body should do, and the two curve types agree on that shape.
_AZ_INTENSITY = (0.849, 0.435, 0.762, 2.014, 1.720, 0.748, 0.472)
_AZ_BINARY = (0.457, 0.424, 0.818, 2.197, 1.841, 0.815, 0.448)
NOISE_PROFILE = np.array([v for v in _AZ_INTENSITY for _ in range(4)]
                         + [v for v in _AZ_BINARY for _ in range(4)], dtype=np.float32)


def sigma_from_highfreq(curves: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-curve noise sigma from the second difference along the curve.

    curves: (C, m) mean-normalised curves **at their native frame rate**; the curves are
    periodic, so the second difference is taken with wraparound and every frame is used.
    mask: (C,) non-zero where a curve is present; absent curves get the median of the present
    ones. Returns (C,).
    """
    c = np.asarray(curves, dtype=np.float64)
    if c.shape[-1] < 3:
        raise ValueError(f"need at least three frames to estimate noise, got {c.shape[-1]}")
    d = np.roll(c, 1, axis=-1) - 2.0 * c + np.roll(c, -1, axis=-1)
    mad = np.median(np.abs(d - np.median(d, axis=-1, keepdims=True)), axis=-1)
    out = 1.4826 * mad / np.sqrt(6.0)
    if mask is not None:
        m = np.asarray(mask) > 0
        if not m.any():
            raise ValueError("no curve is present; cannot estimate sigma")
        out = np.where(m, out, np.median(out[m]))
    return np.maximum(out, 1e-6)


def _pairs(n_curves: int):
    """Indices of the two columns holding the same geometry at each azimuth, per channel."""
    for offset in range(0, n_curves, 28):
        for i in range(len(AZIMUTHS_DEG)):
            a, b = offset + 4 * i, offset + 4 * i + 1
            if b < n_curves:
                yield i, a, b


def ab_mismatch(curves: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-curve RMS difference between the two columns of the same geometry, / sqrt(2).

    This is a diagnostic of how well the two mountings of the body agree after the
    organisers' matching, not a noise level -- see the module docstring. All four cameras at
    an azimuth inherit that azimuth's value, since only the horizontal pair is duplicated;
    curves with no usable pair get the median. curves: (C, m). Returns (C,).
    """
    C = curves.shape[0]
    mask = np.ones(C) if mask is None else np.asarray(mask)
    out = np.full(C, np.nan)
    for _, a, b in _pairs(C):
        if mask[a] > 0 and mask[b] > 0:
            s = float(np.sqrt(((curves[a] - curves[b]) ** 2).mean() / 2.0))
            out[(a // 4) * 4: (a // 4) * 4 + 4] = s
    if np.isnan(out).all():
        raise ValueError("no usable pair; cannot measure the A/B mismatch")
    out[np.isnan(out)] = np.nanmedian(out)
    return np.maximum(out, 1e-6)


def apply_noise(curves: np.ndarray, rng: np.random.Generator,
                scale_lo: float = NOISE_LO, scale_hi: float = NOISE_HI,
                profile: np.ndarray | None = None,
                relative: bool = True) -> np.ndarray:
    """Add Gaussian noise to generated curves. One overall scale is drawn per body from
    [scale_lo, scale_hi]; `profile` (default NOISE_PROFILE) distributes it across the curves
    without changing its mean level. relative=True scales the noise by each curve's own mean,
    which is right before the per-curve mean normalisation; pass False for curves that are
    already normalised."""
    curves = np.asarray(curves, dtype=np.float64)
    C = curves.shape[0]
    p = NOISE_PROFILE if profile is None else np.asarray(profile)
    if len(p) < C:
        raise ValueError(f"profile has {len(p)} entries, need at least {C}")
    p = p[:C, None]
    sig = rng.uniform(scale_lo, scale_hi)
    level = curves.mean(axis=1, keepdims=True) if relative else 1.0
    return curves + sig * p * level * rng.standard_normal(curves.shape)
