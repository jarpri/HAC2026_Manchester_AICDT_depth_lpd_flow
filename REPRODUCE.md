# Reproducing the submission

Everything below runs on one machine with one GPU. There is no cluster anywhere in it: the
stages are loops over models in plain bash, and where a site has several GPUs the per-model
commands are independent and can be spread across them however that site expects.

Start with

```
./reproduce.sh
```

which prints the three stages, what each needs, and what each costs. Then pick one.

| | what it does | needs | time |
| --- | --- | --- | --- |
| `./reproduce.sh verify` | checks and re-scores the bodies as shipped | CPU | minutes |
| `./reproduce.sh reconstruct` | rebuilds them from the shipped weights | one GPU | about 1.5 h per model |
| `./reproduce.sh train` | rebuilds the weights from the data | one GPU | about a day |
| `./reproduce.sh submit` | puts those bodies into `results/submission` | CPU | seconds |

`verify` is the one to run first. It reads only files already in the repository and the
released data, so it settles what was submitted and what it scores before any GPU is
involved. `reconstruct` is the one that reproduces the result. `train` is there for
completeness and is the only stage that takes a day.

## Before any of them

```
make venv          # interpreter and dependencies; see README.md if this is unfamiliar
make data          # streams the organisers' release into dataset/raw
make check-data    # verifies it against dataset/MANIFEST.sha256
```

`reconstruct` and `train` additionally need nvdiffrast, which is not on PyPI and compiles
CUDA at install time. `scripts/setup_toolchain.sh` builds it without root, and the Install
section of README.md explains the arguments that matter on an unfamiliar card. `verify` needs
none of it.

## How closely this will match

`verify` matches exactly, because it re-scores the shipped files rather than rebuilding them.

`reconstruct` will come close but not to the last digit. `train` will not come close at all.
The forward model renders with nvdiffrast, and neither it nor the CUDA reductions beneath it
are bitwise reproducible across different GPUs. One body rebuilt on a different card differs
from ours in about the third decimal of its score; several thousand training steps compound
that into weights which are genuinely different, though they should be no better or worse.

That is the reason the trained weights are shipped in `models/` rather than left to be
regenerated. `reconstruct` begins after the point where the divergence accumulates, which is
what makes it worth running.

## What the submitted bodies were made with

`reproduce.sh` carries these at the top, and they are the settings behind the files in
`results/`. Overriding any of them is a different run, which is fine, but it is then not this
one.

```
RECON_SAMPLES   24     candidate bodies drawn per model
RECON_POLISH    20     gradient steps each candidate is refined by
RECON_STEPS     16     sampler steps from noise to a body
RECON_RES       96     extraction resolution of the written mesh
RECON_GUIDANCE  1.0    weight on the data part of the velocity
```

Two of those deserve a word. `RECON_SAMPLES` matters more than its size suggests: a draw is
discarded if it comes out as several disconnected pieces or if the renderer will not accept
it, and on the public bodies about three quarters of draws are discarded for one of those
reasons. Twenty-four draws leaves roughly six to choose between. Fewer would risk a model
with none, which is a model with no answer at all.

`RECON_GUIDANCE` is 1.0, the weight the flow was trained at. The pipeline can measure a
better one instead -- that is stage 5c, `scripts/decision_check.py`, which reconstructs
held-out bodies at each candidate weight and scores them against their known shapes -- but it
is off by default because at the settings it inherits it is 4096 reconstructions and writes
nothing until the last of them. `RUN_DECISION=1` turns it on for anyone with the compute to
run it at a size that means something.

## What is submitted

The submitted bodies are the end of the pipeline: for each scored model, the flow's
reconstruction of it. `reproduce.sh reconstruct` writes them into `results/submission` when it
finishes, and `reproduce.sh submit` does that step on its own if the reconstructions are
already there.

Where the flow produced nothing for a model -- see below -- that model's convex answer stays
where it is rather than leaving a gap. `results/submission/provenance.json` records, per
model, which directory its body came from and that file's sha256, so which bodies are flow
reconstructions and which are convex answers is on the record rather than left to be inferred.
Whatever was replaced is kept under `results/submission/convex-backup`.

Nothing is copied until every chosen body has passed `scripts/check_submission.py`, which
requires one watertight component of positive volume, consistent winding, the challenge pose,
and a largest distance from the spin axis equal to the published bounding radius. A submission
half replaced by files that do not pass would be worse than one not replaced at all.

## Models that produce nothing

A model whose draws all come out fragmented, or which the renderer refuses, ends with no body
rather than with a bad one: `reconstruct_lpd.py` says `no valid draw candidate has finite
misfit` and stops. `reproduce.sh` reports it and carries on to the next model, and that
model's convex answer stands as its submitted body. This is expected on some models and is
not a failure of the run.

## Layout

The pieces a referee is most likely to want:

```
reproduce.sh                    the three stages above
scripts/run_remote_pipeline.sh  what `train` runs; resumable, one marker per stage
scripts/reconstruct.py          the convex inversion
scripts/reconstruct_lpd.py      the flow's correction of it
scripts/reconstruct_gn.py       the Gauss-Newton refinement
hac26/scoring/                  the organisers' two measures
models/                         the trained networks and the fitted instrument
results/submission/             the submitted bodies, with a README of their own
docs/                           the reasoning behind the forward model and the objective
```

README.md is the longer account: what the challenge is, how the forward model is built, and
why the non-convex path is shaped the way it is.
