# Forward-model review, cross-referenced against Helsinki-Challenge-2026

Review of `hac26/forward/` and the calibration that fits it, done by cross-referencing an
independently developed forward model for the same challenge
(`Helsinki-Challenge-2026`, a Lambertian mesh renderer validated against the published
Blender curves). The two codebases share nothing but the challenge description, so
agreement between them is evidence and disagreement is a place to look.

Everything below is either a numerical comparison between the two models or a measurement
on the released data (`HAC_data_May_8`). Reproduction notes are at the end.

## Status

| | finding | status |
|---|---|---|
| 1 | the pair difference is not the noise | **fixed** -- `noise.sigma_from_highfreq` |
| 2 | the calibration is bounded by its step budget | **fixed** -- early stop, movement report |
| 3 | no BRDF freedom, and the data asks for one | **retracted** -- the evidence was an artefact of the pre-update Blender curves; see below |
| 4 | the xy centroid recentring | **fixed** -- `rescale_touch_z(..., centre_xy=False)` |
| 5 | the field of view is tied to the body | **open, and smaller than first stated** -- see the correction below |
| 6 | the data snapshot is stale for model 1 | **fixed** -- manifest refreshed, `scripts/check_data.py`, per-azimuth alignment report |

Findings 1, 2, 4 and 6 are commits on this branch, each with tests. Nothing here has been
rerun through `calibrate.py`: that needs nvdiffrast and a GPU, and every one of those four
changes what the calibration means, so `models/instrument_calibration.pt` is stale and
everything downstream of it should be regarded as provisional until it is refitted. Refresh
model 1's data first (§6) -- refitting against the superseded curves would bake the same
misalignment in again.

---

## What checks out

**The geometry conventions are exactly right.** `conventions.camera_vector`, `to_body` and
`psi_grid` (SENSE = −1, PSI0 = 0, δ = +1) were compared against the reference model's
independently derived camera and light directions over all 28 geometries × 360 phases:

```
lab camera directions    max |hac26 − reference| = 5.6e-16
body camera directions   max |hac26 − reference| = 1.6e-15
body sun directions      max |hac26 − reference| = 8.9e-16
conventions.to_body vs exact.rotate_z             2.2e-16
```

Light at `(-1, 0, 0)`, camera at lab azimuth `180 + az`, elevations per the published
table, frame *k* rotated by `R_z(+k·360/F)`, phase zero at the STL pose. The reference
model's best-fit phase offset against the Blender curves is 0, and `calibrate.py` fits
ψ₀ = 0.3°, 1.3°, 2.5° on the three public bodies — two independent confirmations that
`PSI0 = 0` is right.

**The transport chain is correct, and better than the reference model's in three places.**

- `LitCoverage`: `e_i = coverage / A_i = (n·s)₊` read off an orthographic sun raster is
  exact, and gets cast shadows, self-shadowing and penumbra for free. The clip-space
  *z* sign in `orthographic()` gives the correct depth test (nearer the source ⇒ smaller
  *z*).
- Radiosity: `B = (I − ρF)⁻¹ ρE`, `L = B/π`, direct light at mesh resolution and
  interreflection at patch resolution, is the right decomposition; the reference model has
  no interreflection at all.
- Radiance as a view-independent per-face attribute, with no 1/r² factor and foreshortening
  realised by pixel coverage, is right for a Lambertian surface.
- The coarea derivatives of the two thresholded reductions are exact where the reference
  model resorts to a soft threshold.

**Otsu computed from the render is the more faithful choice, and the obvious objection
against it does not hold.** The reference model fits a fixed binary threshold instead; the
worry with computing Otsu on a rendered frame is that the render has a black background and
a frame fill fixed by `fov_scale`, neither of which matches the video. Measured on
synthetic limb-darkened frames through `raster.otsu_threshold`:

```
frame fill 0.005 → 0.25       counted fraction of the body varies by  < 5%
background level/noise up to 0.10 ± 0.03   counted pixels vary by     < 0.5%
```

So the count is robust to both, and the argument in `Instrument.pedestal` for not fitting a
binary pedestal is sound. No change needed.

---

## 1. The replicate-pair difference is not measuring measurement noise

`hac26/noise.py` states: *"At each azimuth two cameras sit at the same place and see the
same body at the same instant, so their difference is measurement noise with no geometry in
it."*

That is not what the two horizontal columns are. Each public model ships **28 real videos**
— `CAM1/CAM2 × orientation 1A/1B × 7 angles` — and only **21 simulated** ones
(`{0,45,90,135,225,270,315} × {top, center, bottom}`). There were only ever two cameras, one
horizontal and one looking down. So `hor_a` and `hor_b` are the *same* nominal geometry
recorded in two different mountings of the body, aligned afterwards by time reversal and a
temporal shift (the challenge text describes exactly this). They are not simultaneous, and
their difference carries the A/B mounting mismatch, the residual alignment error and the
stem, not just noise.

Measured on the released curves — the pair difference against the noise estimated from each
curve's own high-frequency content, plus how much of the difference survives a 15-frame box
smooth:

```
                    pair difference / noise, per azimuth
model 1 intensity   az0  4x  az45  3x  az90  3x  az135 12x  az225 50x  az270 11x  az315  7x
model 2 intensity   az0 11x  az45 24x  az90 39x  az135 81x  az225 71x  az270 42x  az315 26x
model 3 intensity   az0  6x  az45  3x  az90  3x  az135  5x  az225  6x  az270  2x  az315  7x

over all 168 released curves:    min 1x, median 12x, max 277x
low-frequency share of (a − b): 0.86 to 0.99
```

σ is inflated by a median factor of 12, and 86–99% of the difference is low-frequency —
structure, not noise. Four consequences, all downstream of the same number:

- The `per_sigma` column of the residual report is divided by a model-error-inflated scale,
  so the headline "residual against the measurement noise alone" understates the real
  misfit by that factor. The README calls this "the number that says how well the forward
  model matches the organisers' processing".
- The NLL weights each curve by `1/(σ² + η²)`, so the az 135/225 geometries are
  down-weighted by up to two orders of magnitude. Those are the α = 135° geometries — the longest shadows and
  the most shape information in the whole dataset.
- `NOISE_PROFILE` is built from these σ, so training injects noise with the wrong overall
  level and the wrong per-azimuth shape, teaching the flow to distrust the same geometries.
- The polish step, "stopped at the noise level", stops far too early.

**Fixed.** σ now comes from the second difference along each curve,
`1.4826 · MAD(d) / √6`, taken at the files' native ~841 frames. Second differences rather
than first: the first difference still carries the signal's own slope, which on these curves
is of the same order as the noise and larger than it on the faceted bodies. The estimate has
to be made before the resampling to the operator's phase grid, so `load_model_curves` now
keeps the native-resolution curves. The pair difference survives as `noise.ab_mismatch`,
documented as the A/B consistency diagnostic it is and reported beside σ by `calibrate.py`;
`eta` is where it belongs, and the log-determinant term already prices it. `NOISE_PROFILE`,
`NOISE_LO` and `NOISE_HI` are re-measured with the new estimator — the profile now rises
monotonically with phase angle and the intensity and binary curves agree on its shape, which
the old one did not.

## 2. The calibration is bounded by the optimiser budget, not converged

`calibrate.py` runs 150 Adam steps at lr 0.03. Adam's per-step magnitude is ≈ lr, so the
total movement of any raw parameter is bounded by 150 × 0.03 = 4.5:

```
rho   raw0 = -1.39  →  fitted 0.911 needs raw = +2.33   (moved 3.71 of 4.50, 82%)
                       reachable range from init: [0.003, 0.957]
eye   raw0 =  8.00  →  fitted 8.51  (moved 0.51)
                       reachable range from init: [3.53, 12.50]
```

ρ used 82% of its budget and stopped just under the reachable ceiling — it was still moving
when the run ended. `eye_distance` **cannot exceed 12.5** from an init of 8.0 whatever the
data says, so 8.51 is not evidence that the fit prefers 8.51; it is close to where it
started.

That matters, because 8.5 is probably too close. Two independent estimates:

- A 100 mm lens on full frame is ≈ 13.7° vertical field of view. Framing a body of
  half-height 1 model unit to fill most of the frame puts the camera at ~20 model units.
- The reference model, fitting a pinhole camera distance to the re-rendered Blender curves,
  lands at ~24 model units (broad optimum over 20–32, consistent across asteroids 1, 2 and
  3). Its pre-re-render fit was ~8, which is suspiciously close to this init.

**Fixed.** The run now stops early on a plateau of the likelihood (`--tol` over
`--patience` steps), so the step cap can be raised without paying for it when it is not
needed, and the default cap goes 150 → 600. Afterwards it prints how far every parameter
travelled against its own budget and names the ones still moving, and the residual table is
explicitly downstream of that check: while anything is named, the residuals are those of a
truncated fit. The movement table goes into `instrument_calibration.json` as well.

Worth watching ρ specifically on the refit: `Instrument.__init__` argues at length for
starting at 0.20, and the data was pulling hard in the opposite direction when the run ended.

## 3. ~~There is no BRDF freedom, and the data asks for some~~ — retracted

**This finding does not survive the data update, and the section below is kept only so the
retraction is legible.** The evidence was measured against the *pre-update* Blender curves.
The organisers re-rendered them, and against the re-rendered ones the trend largely
disappears:

```
real/Blender intensity amplitude ratio     α=0    α=45   α=90   α=135
OLD Blender (what I first quoted), model 3   1.174  1.037  0.906  0.690
NEW Blender (re-rendered),         model 3   1.181  1.031  1.007  0.845
NEW Blender (re-rendered),         model 1   1.323  1.164  1.125  1.160
```

Model 1 shows no decline at all now; model 3's runs 1.18 → 0.85 rather than 1.17 → 0.70, and
the two bodies disagree in direction. "The real curves are flatter than a Lambertian render
by a factor that falls with the phase angle" was substantially a property of the old render,
not of the surface. Using a simulation with its own camera and tone mapping as the stand-in
for a Lambertian reference was the mistake.

The second forward model, run against both targets on the current data, says the same thing
from the other side: its amplitude ratio against the *re-rendered Blender* curves is flat at
0.91–0.97 across phase angle, and against the *real* curves it is 0.96 overall and 1.01 at
α = 135 on model 3. A plain Lambertian model with a distant camera and a gamma reproduces the
real amplitudes. There is no amplitude evidence for a BRDF term.

I did implement the Oren–Nayar term before retracting the finding, and it is worth recording
what it does, since it is the obvious thing for the next person to reach for:

```
                        amplitude ratio vs a Lambert render, by phase angle
Oren–Nayar, σ = 10°       α=0: 0.94   α=45: 0.98   α=90: 1.00   α=135: 1.00
Oren–Nayar, σ = 20°       α=0: 0.86   α=45: 0.96   α=90: 1.00   α=135: 1.00
Oren–Nayar, σ = 30°       α=0: 0.80   α=45: 0.94   α=90: 1.00   α=135: 1.00
```

It bites at α = 0 and does nothing at α = 135. Two implementation notes if anyone tries
again: the qualitative Oren–Nayar model diverges as both the incidence and emission angles go
to grazing, reaching a factor of ~3600 on a sphere at 20° of roughness, which saturates the
sensor and destroys the curve rather than shaping it — it needs a floor on
`max(n·s, n·v)`, around 0.25. And the α = 135 curve of the *default* instrument sits on
`tau_i` and collapses to identically zero, so any probe there needs a brighter surface and a
lower threshold or it measures nothing.

**What is left.** Both forward models leave a residual of roughly 5–10σ on the high
phase-angle real columns — this model at RMSE 0.051 against a measured noise near 0.005 on
asteroid 3, hac26 at `per_sigma` 6–9 on the same columns. That misfit is real and unexplained.
It is no longer evidence for a scattering term, and §6 accounts for part of it on model 1.

## 4. Recentring xy on the solid centroid is wrong and unnecessary

`shapes.rescale_touch_z(v, f)` and `CodeOperator.canonical` both translate the body so the
solid centroid sits on the rotation axis. The released STLs are *already* posed on that
axis, and recentring moves them off it. Posing each public STL to z ∈ [−1, 1] and measuring
the maximum xy radius:

```
              published R    STL origin           centroid-recentred
model 1          1.120       1.1198  (−0.0002)    1.1200  (−0.0000)
model 2          1.420       1.4142  (−0.0058)    1.4451  (+0.0251)
model 3          0.880       0.8782  (−0.0018)    0.8770  (−0.0030)
```

Left as they are, all three reproduce the published bounding-cylinder radius to within
0.02–0.6%, which is the check that the STL origin *is* the rotation axis. Recentring makes
model 2 four times worse and pushes it to 1.445 — larger than the published *minimal*
enclosing radius, which the true body cannot be. The displacement is up to 0.031 in xy.

This lands in two places that matter:

- `calibrate.py:load_truth` renders the displaced truth mesh, so every instrument parameter
  is fitted against a body that is slightly off-axis.
- `CodeOperator.physical` imposes it on every reconstruction iterate and then rescales xy to
  the published R, so the recovered body is both shifted and mis-scaled relative to a truth
  posed on the axis. For an asymmetric body the error grows with the asymmetry.

**Fixed.** `rescale_touch_z` takes a `centre_xy` flag, still true by default because a
procedural library body has no meaningful origin and has to be mounted somehow — the same
rule `shape_library.pose` already uses. The call sites handling bodies already in the
challenge frame pass `False`: the calibration's truth, both scorers, `reconstruct_lpd`'s dice
check, `measure_public_shapes`. `CodeOperator.canonical` no longer translates at all, so the
radius is `max|xy|` about the axis; a corpus body, which is centred when it is posed, is
unaffected. The published-radius test tightens from 3% to 1% and gains a companion showing
that centring makes the fit worse wherever the centroid really is off the axis.

The lightcurves are nearly blind to a lateral offset, so this is a prior rather than
something the data will correct — which is the reason to make it the right prior.

## 5. The field of view is tied to the body, not to the lens

`fov = 2·atan(fov_scale · extent / eye_distance)` with `fov_scale = 1.6` fixed means the
body always fills the same fraction of the frame, and the modelled field of view moves with
the body instead of staying at the lens's:

```
R ≈ 1.2  body   →  fov_y ≈ 33°
model 10 (R = 3.95) →  fov_y ≈ 75°
real 100 mm lens on full frame  ≈ 14°
```

**Correction to my first draft of this section.** I wrote that fixing `fov_scale` couples
the frame fill to the camera distance so that the two cannot be identified separately, and
that this was a plausible cause of §2. That is wrong. Perspective strength is
`extent / eye_distance`, which `eye_distance` sets on its own; `fov_scale` only fixes how
much of the frame the body fills. The two are separable and §2 stands on its own.

What is left is smaller and second-order: the cos⁴ falloff and the vignetting polynomial are
applied at body-relative radii rather than frame-relative ones, so a single fitted vignette
polynomial means something different for each body, and model 10 is rendered through a
75° lens it was never filmed with. Most of the absolute scale cancels in the per-curve mean
normalisation. Tying the framing to the body also has a real virtue — every body is sampled
by the same number of pixels, whatever its shape — so this is a trade, not a defect.

**Open, low priority.** If it is worth doing, make the field of view a fitted instrument
parameter and let the frame fill follow, rather than deriving it from each body's extent.

## 6. The pinned data snapshot is stale for model 1

`dataset/MANIFEST.sha256` pinned a mixed snapshot. Checked entry by entry against a current
download, 189 of its 198 entries were already right. The exceptions:

```
model 1  intensity, binary and both _blender     pinned to the PRE-update files
4 × *.stale29jul                                 renamed local backups, in no download
Readme.txt                                       matches neither the May nor the current file
```

The `.stale29jul` names say what happened: models 2 and 3 were refreshed by hand after the
organisers re-rendered the Blender curves on 29 July 2026, and model 1 was missed — its
curves changed later, in the 17/25 August update.

That update realigned model 1's real curves by **a different whole-frame shift per azimuth**,
identical across all four columns of a group, the shifted curves matching the old ones at
`corr = 1.0000`:

```
azimuth      0     45     90    135    225    270    315
shift      -12      0     +3     +3     -4     -2      0     frames of 841
```

`calibrate.py` fits one ψ₀ per body, so no value of it absorbs a shift that is −12 at az 0,
+3 at az 90/135 and 0 at az 45/315. What is left lands in the residual table as forward-model
error, and it costs most where the curves move fastest — the high phase angles. Model 1's
intensity `per_s` is 0.83–1.22 at az 135/225 against 0.15–0.26 at az 45 and 315, which are
the two azimuths that were never shifted. That is a data-staleness artefact, not a model one,
and it is part of what §3 was reading as physics.

**Fixed.** The four model-1 entries are refreshed and the four `.stale29jul` entries dropped
(`Readme.txt` is left alone — there is no copy in the repo to say which version is right).
`scripts/check_data.py` does the check the README always told people to do and nothing did,
with `--write` to regenerate after a refresh you meant to make. And `calibrate.py` now reports
the shift each azimuth still wants at the fitted ψ₀ — reported, not fitted, since seven more
free parameters per body would explain away real misfit just as readily. Validated against
this known realignment: at 96 phases it recovers +5.16° where the true shift is 5.14°, and
exact zeros on the two untouched azimuths.

Groups that all want the same shift mean ψ₀ is off; groups that disagree mean the curves are
not aligned with each other, and the fix is a fresh download rather than a wider fit.

## 7. Smaller things

- `data_io.resample_curves` decimates 841 → 48 (calibration) and 841 → 96 (reconstruction)
  with plain `np.interp` and no anti-aliasing. Features narrower than ~9 frames alias.
  Mostly harmless for smooth intensity curves; the binary curves of faceted bodies are the
  place it would show.
- The instrument is calibrated at 48 phases (`calibrate.py --phases`) and used at 96
  (`build_corpus.py`, `reconstruct_lpd.py`). Not a correctness bug — the chain is
  phase-count agnostic — but the fitted values have never been checked at the grid they run
  on.
- `forward/sdf_volumetric.py`, `forward/polytope_raycast.py` and `forward/sdf_surface.py`
  (660 lines) are not referenced anywhere in the package, the scripts or the tests.
- Every forward-model test is an internal-consistency test. Nothing compares the chain
  against the published curves except `calibrate.py` itself, which is also the thing being
  fitted. A regression test pinning the public-model residuals would catch a convention
  regression that the consistency tests cannot.
- `LitCoverage` relies on the mesh being closed: a face with `n·s < 0` is excluded only by
  losing the depth test to the front of the body, and there is no `(n·s) > 0` guard. True
  for everything FlexiCubes produces, but it would silently light back faces on an open mesh.

---

## One ambiguity both codebases share

The challenge text says the orientation pairs to match are 45↔225, 90↔270, 135↔315. That
pairing is geometrically impossible. The only rigid motion that flips a body upside down and
leaves the light along −x in the body frame is a 180° rotation about the x-axis; any
additional turntable rotation about z is absorbed into the "temporal shift" the text already
mentions. That motion maps camera azimuth β to −β, giving the pairs 45↔315, 90↔270,
135↔225 — agreeing with the published list on 90↔270 only.

Both codebases assume the bottom column of azimuth group θ is the geometry (θ, −e_θ), and
the reference model validated that against the Blender curves at corr 0.996 on asteroid 3,
so the *intended* geometry is almost certainly what both of us use, and the published
sentence is likely loose wording about which recordings were compared during curve matching.

The one thing still worth an experiment is the bottom camera's **elevation**. The elevation
table splits cleanly into 21° at az 0, 26° at az 45/90/135 and 24° at az 225/270/315, which
looks like two sessions with the tripod reset between them. If group 45's bottom column
comes from the session that was set at 24°, its elevation is −24° and not −26°. Both
codebases currently use −e_θ from the same group. Cheap to test: refit with the bottom
elevations taken from the paired azimuth instead and compare the per-geometry residuals.

---

## Reproducing the measurements

All measurements used the released `HAC_data_May_8` archive and needed only numpy plus this
package.

- Convention comparison: build `conventions.cameras()` vectors and `exact.rotate_z(w,
  −psi_grid())`, compare elementwise against the reference model's camera/light directions.
- §1: read `Asteroid0{m}_lightcurve_{intensity,binary}.txt`, take columns `4i` and `4i+1`
  per azimuth; compare `noise.ab_mismatch` with `noise.sigma_from_highfreq`, and the std of
  a 15-frame box smooth of `a−b` with the std of `a−b`. Both on the native-resolution
  curves — resample first and the noise estimate reads the signal's curvature instead.
- §2: `150 * 0.03 = 4.5`, then invert `softplus`/`sigmoid` at the init and fitted values
  quoted in `models/instrument_calibration.json`.
- §3: resample real and `_blender` curves to 360, mean-normalise, align each column by
  circular cross-correlation, and take the least-squares slope of `real − 1` on
  `blender − 1`.
- §4: `stl_io.load_stl` on each public STL, pose to z ∈ [−1, 1] with and without the xy
  centroid shift, compare `max|xy|` against `conventions.CYLINDER_R`.
- Otsu robustness: synthetic limb-darkened discs through `raster.otsu_threshold` at varying
  fill fraction, background level and background noise.
- §3: the amplitude ratios come from resampling both curve files to 360 frames,
  mean-normalising, aligning each column by circular cross-correlation and taking the
  least-squares slope of one against the other. Use the **current** download: the same
  measurement on the pre-update Blender curves is what produced the retracted trend.
  For the Oren–Nayar probe, render an ellipsoid hull at geometries 0, 4, 8, 12
  (α = 0, 45, 90, 135) with
  `Instrument(rho=0.85, tau_i=1e-4, quantise=False)` and the saturation pushed well clear, so
  no geometry sits on the intensity threshold; take the peak-to-peak of each mean-normalised
  intensity curve and divide by the same at zero roughness. Check the raw minima are non-zero
  first: with the default instrument the α = 135 curve sits on `tau_i` and collapses, which
  makes the ratio meaningless rather than small.
