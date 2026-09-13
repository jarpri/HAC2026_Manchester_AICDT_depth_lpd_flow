# hac26

Shape reconstruction from lightcurves for the
[Helsinki Asteroid Challenge 2026](https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/).

Ten 3D-printed asteroids were filmed on a turntable from 28 camera geometries. Each frame is
reduced to two numbers, summed intensity and lit-pixel count, giving 56 curves per body, and
the organisers release both the laboratory curves and a Blender rendering of the true shape
under the same geometries. Reconstructions are scored on voxel overlap with the true shape
plus the distance between the boundary curves of 2D projections. `docs/challenge_info.md` has
the rules.

To check or rebuild the result rather than read about it, start at
[REPRODUCE.md](REPRODUCE.md), or run `./reproduce.sh` for a summary of the three stages and
what each costs. The rest of this file is the account of how the method works and why it is
built the way it is.

## The submission

The submitted bodies are `results/submission/Asteroid04.stl` to `Asteroid10.stl`, and each is
the flow's reconstruction of that model: the end of the pipeline described under "The
non-convex path" below, trained on a library of 1500 bodies. Which run each body came from,
and that file's sha256, is recorded per model in `results/submission/provenance.json`.

They rest on a convex inversion, which is both the body the flow corrects and the fallback
where the flow produced nothing. A model can end without a flow body -- the reconstruction
refuses to answer when every draw comes out in several disconnected pieces or when none of
them render -- and the convex answer for that model then stands rather than leaving a gap.
The convex stage is

```
python scripts/make_submission.py
```

which reconstructs every model with the trained convex network `models/lpd_convex.pt`, poses
it as the challenge asks (rotation axis z, the body touching z = 1 and z = -1, the light at
minus infinity on x, frame 0 of the curves), sets its width from the published bounding
radius, writes the public models to `results/public/`, checks every file
(`scripts/check_submission.py`, which requires one watertight component of positive volume,
consistent winding, the pose and the cylinder) and scores the public ones against the released shapes
with the organisers' two measures (`hac26/scoring/official.py`) into
`results/public_scores.json`. It takes seconds per model on a CPU and needs neither a GPU
nor nvdiffrast. The recipe is fixed in the script, and `results/public_scores.json` records
the checkpoint's digest and the channel each model was inverted from.

The convex network reads the Blender curves when the organisers release them and the
laboratory curves otherwise (`hac26.data_io.load_inversion_curves`). The render is the
cleaner observation of the shape. It has no lens, sensor, mounting or beam in front of it
and no per-column realignment behind it, its scattering is of the kind the convex operator
assumes, and the laboratory columns of several bodies are out of phase with their own
geometry by tens of degrees. `scripts/reconstruct.py --channel` forces either channel for one
model.

`scripts/assemble_submission.py` is what puts the flow's bodies in place, and
`./reproduce.sh submit` runs it. It copies nothing until every chosen body has passed
`scripts/check_submission.py`, and keeps whatever it replaced under
`results/submission/convex-backup`.

The convex answer is not the body, and on a body with concavities it is not even the body's
hull. A convex inversion returns the convex body whose own shadowing best imitates the
concavities, which is larger than the hull of the body that cast them; of the three public
bodies the answer exceeds the true hull on the one that has concavities and falls short of it
on the two that do not. Correcting it is the subject of the non-convex path below.

That path has two tracks. The submitted bodies come from the flow, and
`scripts/assemble_submission.py` puts them in place. The other track, Gauss-Newton and MAP
refinements of the convex answer, is described below and is not what was submitted here; it
enters a submission through `scripts/select_answers.py`, which judges a refinement on cameras
held out of its own fit, and `results/submission/selection.json` then records what it chose.

The convex network was trained against a photometric kernel that has since been measured to
be wrong (`notes/photometry.md`). An unrolled scheme is an estimator fitted against one
operator and is not the same estimator against another, so the checkpoint records which law
it was trained with and is run against that one; its answers are unchanged. Retraining it
against the corrected operator is outstanding work and is the largest single thing left.
`hac26/solvers/convex_direct.py` solves the same problem directly, on the corrected operator
and without training, and does not yet score as well; its docstring says by how much.

`scripts/benchmark.py` scores any directory of reconstructions against the released shapes
with the organisers' own measures, which is how a change to the method earns its place.

```
python scripts/benchmark.py results/public results/gn
```

## Install

```
pip install -e ".[torch]"
```

The exact forward model renders with nvdiffrast on a GPU. nvdiffrast is not on PyPI and
compiles CUDA at install time, and `scripts/setup_toolchain.sh` builds it without root. The
calibration, the flow training and the non-convex reconstructions render with it. The
submission and the tests do not, since the tests run the same code on a slow pure-torch
rasteriser.

Both x86 and ARM hosts are handled. The toolchain reads the machine and fetches the CUDA
redistributable NVIDIA publishes for it, which for an ARM server such as a Grace-Hopper node
is the one named `linux-sbsa` rather than the `linux-aarch64` that means Jetson. Set
`TORCH_CUDA_ARCH_LIST` to cover every card the build may land on -- `8.0;8.9;9.0+PTX` spans
A100, L40S and Hopper -- since an extension built for one card fails at the first kernel
launch on an older one. The venv and the compiled extension each record the machine they were
made on and are rebuilt rather than reused when a tree is reached from the other, which a
shared filesystem makes easy to do by accident. `make check` prints the machine beside the
torch build.

## Data

The challenge data is not stored in the repo. Fetch it with

```
python scripts/fetch_data.py
```

which streams the organisers' Dropbox folder into `dataset/raw/`, lifts out the wrapping
directory some releases have, and verifies what arrived. Override the link with `--url` or
`$HAC_DATA_URL`. Or put it there by hand under the organisers' original directory names.

Either way, check it with

```
python scripts/check_data.py
```

which verifies it against `dataset/MANIFEST.sha256` and names anything missing or changed.
Do that after every download, because the organisers have re-released these files more than
once and a partial refresh is silent. `--write` regenerates the manifest, for a refresh you meant to
make.

## The non-convex path

The correction from the convex answer to the body is two moves at once: the hull shrinks and
the surface is carved. Neither is worth making alone. Laid on the convex answer of the
non-convex public model separately and scored against the released render, the reshaping
alone raises the misfit and the carve alone barely lowers it, while the two together lower it
by more than half and take the overlap with the truth from 0.69 to 0.99. That is why a search
which moves one coordinate direction at a time walks away from the answer, and it is the
reason every carve of a fixed hull that this repository has measured scored no better than
not carving at all.

Everything here needs an instrument fitted to the channel it inverts, which
`scripts/calibrate.py` writes from the public models' released shapes.

```
python scripts/calibrate.py                                # the laboratory channel
python scripts/calibrate.py --channel blender --models 1 3 # the Blender render
```

The two channels are different instruments (`hac26/forward/mesh/instrument.py`). The
laboratory curves come through a lens, a sensor and bounce light off a matte white print, and
that instrument is fitted. The rendered channel is not fitted at all: a render has no lens, no
point spread, no penumbra and no spline transfer, its camera is at infinity, and its one
constant is the exponent of the view transform, which belongs to the channel rather than to
any body and is measured once. `Instrument.blender_start` is that instrument rather than a
starting point, and `notes/photometry.md` derives what the rig measures and records where the
code had it wrong. The calibration prints the residual of the exact forward model at
the true shape divided by the noise, per geometry, which is the number that says how well
the chain matches the channel, and a travel table that names any parameter still moving
when the step budget ran out.

The correction from the convex answer to the body is one function on the sphere. A convex
inversion returns a body larger than the true hull, because enlarging a convex body is how it
imitates the shadowing of a concavity, so the correction shrinks the hull in some directions and
carves it in others -- and those are not two separate moves but the low and the high angular
degrees of the same displacement. The body is the convex core displaced inward by a depth
indexed by the direction of the surface point, the nine reshaping coefficients carry that
displacement's degrees up to two, and a depth on the nodes carries the rest
(`hac26/field.py`). Below a bound the representation states, the body still contains the centre
the depths are measured from, so it is star-shaped and extracts as one closed surface by
construction rather than by repair.

The fit moves the whole of that function in one damped Gauss-Newton step with a secant
Jacobian, coarse to fine in angular degree and restarted from several shrunken and carved
bodies (`hac26/solvers/gauss_newton.py`). Nothing differentiates the renderer.

What it minimises is not the misfit. Both released reductions are sums over a thresholded
image, so their error is the quantisation of the boundary of the lit region, and a body with
more surface has more boundary to place accurately: a corrugation one grid cell wide, laid on a
convex answer, lowers the misfit more than moving a third of the way to the body does and moves
the overlap by a thirtieth as much. The objective therefore carries the body's surface area
beside its misfit, which is what charges a rough surface and leaves a smooth dent of any depth
nearly free, and it carries it as a logarithm plus an area so the balance survives the descent.
`notes/objective.md` measures all of that, gives the window the weight has to lie in, and
records the three things the penalty exposed that a ridge on the coefficients had been hiding.
One more thing decides where the ladder stops. The first variation of area under a normal
displacement weights the displacement by the surface's mean curvature, so an oscillation of zero
mean is nearly free of area at first order: a displacement at the scale of the node spacing buys
misfit almost without paying for it, and no weight charges it while still leaving the body a
minimum. The ladder therefore stops at an angular degree whose wavelength is many extraction
pitches, and that bound is load-bearing rather than tuning -- the same ladder without it ends
below the convex answer it started from.
Two consequences reach the rest of the pipeline: a body whose convex answer already explains
its curves is left alone, because there the penalty costs overlap while improving the misfit
and nothing downstream could catch it; and `select_answers.py` judges a correction by the
functional it was fitted under, because on the misfit alone that gate prefers a corrugated body
to a shaped one.

Run the non-convex public model first, because its truth is released and the gate below is
calibrated on it, then the scored models.

```
scripts/run_nonconvex.sh
```

which calibrates the channel if it has not been calibrated, runs both solvers on the public
model with a concavity and then on the scored models, and decides. Every setting is a variable
at the top of it and can be overridden from the environment; `scripts/reconstruct_gn.py
--help` runs one model of one solver on its own. Every body is a unit of work: one already
written against the same instrument is skipped, and both solvers checkpoint inside a body, so
a job that stops part way carries on where it stopped rather than starting over.

Cameras are held out of every fit, and the body's misfit on them is written beside the convex
answer's on the same cameras. That pair is the only test of whether a shape was recovered
rather than curves fitted: a body fitted on all of them can reach any misfit by overfitting.
`select_answers.py` accepts a correction for a scored model where it beats its convex answer
on its own held-out cameras. Passing `--calibrate` tightens that to the margin a public
model's correction reached, which is one body's margin asked of every other; the runbook does
not, because a public run that falls short would then stand every convex answer, and a convex
answer is not a safe default but a body known to be missing the concavities the challenge is
about.

`scripts/reconstruct_map.py` searches the same problem differently. It minimises the same
objective from the convex answer alone, by the adjoint rather than by a secant Jacobian: one
gradient costs two rendering passes where a secant Jacobian costs one render per coordinate,
which is about a hundred times as much optimisation per render, and it buys that by descending
into one basin instead of sweeping a designed grid of starts. Neither dominates, so the
runbook runs both. They measure the written body through one function at one resolution and
hold out the same cameras, so their answers for a body are comparable, and
`select_answers.py` is given both directories and decides per model on the cameras neither
fit saw.

How many directions the depth is carried on is not a free choice. A carve the field can only
hold blurred fits the curves worse than no carve at all, because a shadow is cast by an edge, so
the representation has to be able to hold the body before any solver can find it.
`notes/representation.md` measures what the family can hold, and the answer is that it is not
the limit: fitted to the released non-convex body's own surface it reaches an overlap of 0.997,
where every fit of that body from its curves returns about 0.75.

Read the calibration's residual at the released shapes before reading any reconstruction. On
the rendered channel a model of this kind reaches about 0.004 against curves of a released
body, so a residual much above that is a fault in the chain and not a property of the data.
Written in the fit's own coordinates, the non-convex public body has a lower misfit than the
body the fit reaches from its convex answer, and the fit released from the body stays there,
so the curves do prefer the body; but the two are separated by less than the forward model's
own error at the released shapes. A search that selects on the misfit is only selecting
between them once that residual is well below the separation, which is why the calibration's
number is the first one to read, and why `select_answers.py` judges a correction on cameras
held out of its own fit rather than on the misfit it was fitted to. `notes/identifiability.md`
carries the measurement.

The flow pipeline, `scripts/run_remote_pipeline.sh`, builds a shape library, fits shape codes
to it, renders a corpus with the exact operator and the convex stage's starts, trains the
prior and the data part of a conditional flow (`hac26/solvers/lpd_flow.py`,
`scripts/train_lpd.py`) and reconstructs every model from several draws with a polish on the
exact misfit (`scripts/reconstruct_lpd.py`). Each stage writes a marker under `runs/.done/`
recording its settings and the source it ran with, and is skipped only while both still
match, so a dropped run is safe to relaunch and a change to the code reruns the stages below
it. `scripts/run_smoke_test.sh` runs the same stages small, as a wiring check. Every setting
is a variable at the top of the pipeline script and can be overridden from the environment.
Training states lie on the straight line between noise and body only, and `train_lpd.py`
says why states from the sampler's own trajectory are not scored. The flow's answers are
scored beside the convex ones and reach the submission by the same gate as any other
correction. The pipeline's last stage is the non-convex track above, which is what writes
`results/submission`.

`submit_csf3.sh` is that pipeline as a batch job, with the partitions, the modules and the
GPU architectures of one cluster settled in it. `submit_csf3_gn.sh` is the correction and the
submission on their own: they read no corpus and no flow, only the calibration and the
committed convex answers, so they need none of the training. It asks for hours rather than
days, and for a rasterisation-shaped GPU rather than a training-shaped one, because a
Jacobian column there is one render of one candidate. Run it with `SCORED="4"` first to see
what one body costs before trusting a wallclock for all of them.

## Tests

```
pytest
pytest -m "not slow"     # skip the extraction-scale ones
```

## Layout

```
hac26/forward/    forward models; see forward/__init__.py
hac26/solvers/    lpd_convex, lpd_flow, gauss_newton, minkowski, operator, output
hac26/scoring/    official (the organisers' measures), voxel, side_view
hac26/            conventions, geometry, field, shapes, noise, shape_library, curves_mesh,
                  data_io, recon, library_io, library_metrics
scripts/          entry points; reconstruct_lpd.py is the flow's reconstruction and
                  assemble_submission.py makes those bodies the submission.
                  make_submission.py builds the convex answers they rest on, and
                  select_answers.py belongs to the Gauss-Newton and MAP track
notes/            measurements that settle a choice made in the code
models/           the trained convex solver; the calibrations calibrate.py writes go here
results/          submission/ the scored models, public/ the public ones, public_scores.json
dataset/          challenge data, not tracked
runs/             training output, not tracked
tests/
```

There are two forward models, an analytic operator for convex bodies and the exact mesh
chain (shadows from a rasterised sun view, radiosity, rasterisation from every camera, the
sensor model), which is differentiable in the mesh. `hac26/forward/__init__.py` describes
both.

`lpd_convex` is an unrolled primal-dual network on the convex operator, trained on synthetic
convex bodies, and its output is the submitted body and the starting support of the non-convex
path. `lpd_flow` is a conditional flow that generates a non-convex correction on top of that
convex start, reading at every step the residual of the exact model and its adjoint
(`hac26/solvers/operator.py`).
