#!/usr/bin/env bash
# A small run of the pipeline, to check the wiring before a full run on a remote machine.
#
#   scripts/run_smoke_test.sh
#
# The first run creates and activates a .venv through scripts/_venv_setup.sh and installs the
# dependencies if they are missing; later runs skip straight to the work.
#
# Runs a few bodies at a coarse grid and a few optimiser steps per stage. It needs nvdiffrast
# on a GPU, like the pipeline itself, because every stage after the fit renders with the
# exact forward model; on a machine without one, `pytest tests` is the wiring test, since the
# tests run the same code on the pure-torch rasteriser. It does not need dataset/raw: without
# it the instrument is the uncalibrated default and reconstruction is skipped.
#
# Everything is written under runs/smoke/ and dataset/generated/shapes_smoke/, never to
# runs/corpus_codes.npz, runs/corpus.npz, runs/lpd_flow.pt, models/ or the real shape
# library, so this is safe to run alongside a real scripts/run_remote_pipeline.sh run.
#
# What this proves: the shape library builds valid, non-convex, single-component bodies;
# fit_shapes.py fits codes to them; build_corpus.py renders them with the exact operator and
# runs the convex stage on them; train_prior.py and train_lpd.py train on that corpus, in
# both runs (one expert, then branched into several); decision_check.py reconstructs a
# held-out body; the resulting flow checkpoint loads and runs in reconstruct_lpd.py.
#
# What this does not prove: that the results are any good. A few bodies and a few steps
# cannot learn anything; only that every stage runs and the shapes of everything agree.
set -uo pipefail
cd "$(dirname "$0")/.."

# Four, not two. train_lpd holds one body out, train_prior needs at least one to train on,
# and fit_shapes refuses a corpus whose median fitted Dice is below its floor -- over two
# bodies that median is one awkward body away from tripping, and a smoke test that fails at
# random proves nothing. Four is the smallest count that exercises every stage reliably.
N_BODIES=${N_BODIES:-4}
LIB_RES=${LIB_RES:-32}
LIB_WORKERS=${LIB_WORKERS:-$(nproc 2>/dev/null || echo 2)}
FIT_WORKERS=${FIT_WORKERS:-$LIB_WORKERS}
# Not set here. Both the floor the fit refuses to run under and the default it uses instead
# are counts per depth, so they move with the representation: a number written here went
# stale the moment the field grew from a lattice to 2560 depths, and would go stale again.
# The default is above the floor rather than on it, so the fitted-Dice check this stage makes
# is deciding on the bodies rather than on the sampling. Set FIT_POINTS to override.
FIT_POINTS=${FIT_POINTS:-}
FLOW_STEPS=${FLOW_STEPS:-8}
FLOW_PHASES=${FLOW_PHASES:-16}     # few phases keep the run short
# The extraction has to resolve the angular scale of the depth field, or the body the
# operator renders is coarser than the one its coefficients describe and the run exercises a
# regime no real one is in. At 2560 directions that scale is about a tenth of the body, and
# the grid pitch is twice the extraction extent over this number, so sixteen -- chosen when
# the field was a coarse lattice -- no longer reaches it and twenty-four does.
FLOW_OPERATOR_RES=${FLOW_OPERATOR_RES:-24}
# The sensor, for every stage that renders. The cost of this pipeline is dominated by the
# pixels, so a whole-pipeline check at the calibrated 320x568 costs what a run costs and says
# nothing more about whether the stages agree; the numbers below are meaningless at this size
# and the stages say so themselves.
SMOKE_HEIGHT=${SMOKE_HEIGHT:-48}
SMOKE_WIDTH=${SMOKE_WIDTH:-80}
SMOKE_SUN_RES=${SMOKE_SUN_RES:-32}
SIZE=(--height "$SMOKE_HEIGHT" --width "$SMOKE_WIDTH" --sun-res "$SMOKE_SUN_RES")
# The released meshes the calibration renders, decimated. The visibility test that builds
# their patches runs on the CPU and is most of a body's cost here.
TRUTH_FACES=${TRUTH_FACES:-2000}
DESIGN_N=${DESIGN_N:-4096}
CONVEX_CKPT=${CONVEX_CKPT:-models/lpd_convex.pt}

# shellcheck disable=SC1091
source scripts/_venv_setup.sh   # creates and activates the venv, installs deps, sets PY

LIB_DIR=dataset/generated/shapes_smoke
OUT=runs/smoke
mkdir -p "$OUT" logs

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
t_start=$(date +%s)

# Runs one stage and stops the script with a message if it fails. The command's own exit
# status is read from PIPESTATUS, since the output goes through tee. ALLOW holds statuses that
# are not failures of the stage: the solvers report a refused export as 3, which means the fit
# ran to the end and its numbers are written but the extracted mesh is not a closed solid.
# That is a property of the body at this resolution rather than of the wiring, and a wiring
# test that stopped there would have checked every stage and reported a failure.
ALLOW=""
run() {
  local desc="$1" logfile="$2"; shift 2
  "$@" > >(tee "$logfile") 2>&1
  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    for ok in $ALLOW; do
      if [ "$status" = "$ok" ]; then
        log "    NOTE: $desc exited $status, which this test allows -- see $logfile"
        return 0
      fi
    done
    log "FAILED: $desc (exit $status) -- see $logfile"
    log "command was: $*"
    exit "$status"
  fi
}

log "using interpreter: $($PY --version 2>&1) at $(command -v "$PY")"

# The same check the long runs make. Here it is worth it for a different reason: a smoke test
# that fails five stages in because a package is missing has told you about the package and
# nothing about the algorithm.
run "preflight" logs/smoke_preflight.log \
  "$PY" scripts/preflight.py --data-dir "${DATA_DIR:-dataset/raw}" --skip-render

if [ -z "${HAC26_SOFTWARE_RASTER:-}" ] && ! "$PY" -c "import nvdiffrast" >/dev/null 2>&1; then
  log "nvdiffrast is not importable, so the exact forward model cannot render here."
  log "Run scripts/setup_toolchain.sh on a GPU machine, or run 'pytest tests' for a wiring"
  log "test on the CPU."
  exit 1
fi

log "=== 1/14 shape library: $N_BODIES bodies at res=$LIB_RES"
run "shape library" logs/smoke_library.log \
  "$PY" scripts/build_shape_library.py \
    --n "$N_BODIES" --out "$LIB_DIR" --workers "$LIB_WORKERS" --res "$LIB_RES" \
    --report-sample "$N_BODIES"
tail -20 logs/smoke_library.log

log "=== 2/14 spherical design (no-op if hac26/design${DESIGN_N}.npy is already checked in)"
if [ -f "hac26/design${DESIGN_N}.npy" ]; then
  log "    hac26/design${DESIGN_N}.npy already exists" | tee logs/smoke_design.log
else
  run "spherical design" logs/smoke_design.log "$PY" scripts/make_design.py --n "$DESIGN_N"
fi
tail -5 logs/smoke_design.log

log "=== 3/14 fit_shapes: per-body fit over the smoke library"
run "fit_shapes" logs/smoke_fit.log \
  "$PY" scripts/fit_shapes.py \
    --bodies "$N_BODIES" --shapes-dir "$LIB_DIR" \
    --workers "$FIT_WORKERS" ${FIT_POINTS:+--points "$FIT_POINTS"} \
    --out "$OUT/corpus_codes.npz"
tail -20 logs/smoke_fit.log

# The instrument: the calibrated one when it exists and loads, otherwise the uncalibrated
# default, which is enough to check the wiring and nothing else. A calibration written by an
# older Instrument does not load, and is treated as absent.
valid_instrument() {
  "$PY" -c "import sys; sys.path.insert(0, '.')
from hac26.forward.mesh.instrument import Instrument
Instrument.load('models/instrument_calibration.pt')" >/dev/null 2>&1
}
if [ -f models/instrument_calibration.pt ] && valid_instrument; then
  CAL=models/instrument_calibration.pt
else
  CAL=$OUT/instrument_default.pt
  "$PY" -c "import sys; sys.path.insert(0, '.')
from hac26.forward.mesh.instrument import Instrument
Instrument().save('$CAL')"
  if [ -f models/instrument_calibration.pt ]; then
    log "    NOTE: models/instrument_calibration.pt does not load with this Instrument; using the"
  else
    log "    NOTE: no models/instrument_calibration.pt; using the"
  fi
  log "    UNCALIBRATED default instrument ($CAL). Fine for a wiring test; the numbers below"
  log "    are meaningless."
fi

# The convex stage: the real checkpoint when it exists, otherwise an untrained one of the
# same kind, which is enough to check the wiring and nothing else.
if [ -f "$CONVEX_CKPT" ]; then
  CONVEX=$CONVEX_CKPT
else
  CONVEX=$OUT/convex_untrained.pt
  "$PY" -c "import sys; sys.path.insert(0, '.')
import torch
from dataclasses import asdict
from hac26.train import Preset, build_model
pr = Preset(ch=8, n_iter=2, n_primal=3, n_dual=3, n_theta=12, n_phi=24, r_cond=True)
torch.save({'preset': asdict(pr), 'model': build_model(pr, 'cpu')[0].state_dict()}, '$CONVEX')"
  log "    NOTE: no $CONVEX_CKPT; using an UNTRAINED convex stage ($CONVEX). Fine for a"
  log "    wiring test; the starts it makes are meaningless."
fi

log "=== 4/14 build_corpus: curves and convex starts of the smoke bodies"
rm -rf "$OUT/corpus.npz" "$OUT/corpus.npz.parts"
run "build_corpus" logs/smoke_corpus.log \
  "$PY" scripts/build_corpus.py \
    --bodies "$N_BODIES" --phases "$FLOW_PHASES" --operator-res "$FLOW_OPERATOR_RES" \
    --codes-file "$OUT/corpus_codes.npz" --calibration "$CAL" --convex "$CONVEX" \
    "${SIZE[@]}" --out "$OUT/corpus.npz"
tail -5 logs/smoke_corpus.log

log "=== 5/14 train_prior: the prior part over the smoke corpus"
run "train_prior" logs/smoke_prior.log \
  "$PY" scripts/train_prior.py \
    --steps 200 --batch 8 --val-bodies 2 --val-every 50 \
    --log-every 50 --corpus "$OUT/corpus.npz" --out "$OUT/prior_flow.pt"
tail -5 logs/smoke_prior.log

log "=== 6/14 train_lpd: the data part over the smoke corpus, one expert"
run "train_lpd" logs/smoke_flow.log \
  "$PY" scripts/train_lpd.py \
    --steps "$FLOW_STEPS" --batch 1 --experts 1 \
    --val-bodies 2 --val-every 10 --patience 2 \
    --ckpt-every 10 --log-every 5 --no-resume \
    --corpus "$OUT/corpus.npz" "${SIZE[@]}" \
    --calibration "$CAL" --prior "$OUT/prior_flow.pt" \
    --out "$OUT/lpd_flow.pt"
tail -20 logs/smoke_flow.log

log "=== 7/14 train_lpd: the same run continued, branched into its experts"
run "train_lpd (experts)" logs/smoke_flow_experts.log \
  "$PY" scripts/train_lpd.py \
    --steps "$FLOW_STEPS" --extra-steps 4 --batch 1 \
    --val-bodies 2 --val-every 2 --patience 2 \
    --ckpt-every 2 --log-every 1 \
    --corpus "$OUT/corpus.npz" "${SIZE[@]}" \
    --calibration "$CAL" --prior "$OUT/prior_flow.pt" \
    --out "$OUT/lpd_flow.pt"
tail -12 logs/smoke_flow_experts.log
grep -q "branched from 1 to" logs/smoke_flow_experts.log || { log "FAILED: the second run did not branch"; exit 1; }

log "=== 8/14 decision_check: a held-out smoke body, two draws"
run "decision_check" logs/smoke_decision.log \
  "$PY" scripts/decision_check.py --bodies 1 --samples 2 --polish-steps 2 --res 24 \
    --val-bodies 2 --side-points 20000 \
    --ckpt "$OUT/lpd_flow.pt" --corpus "$OUT/corpus.npz" --calibration "$CAL" \
    "${SIZE[@]}" --out "$OUT/decision_check.json"
tail -12 logs/smoke_decision.log

if [ -d dataset/raw ]; then
  log "=== 9/14 convex: dataset/raw is present, the convex start of model 1"
  mkdir -p results/smoke
  run "reconstruct (convex)" logs/smoke_convex.log \
    "$PY" scripts/reconstruct.py --ckpt "$CONVEX" --model 1 --fit-cylinder \
      --out results/smoke/convex_Asteroid01.stl
  log "=== 10/14 reconstruct: model 1 from that start"
  run "reconstruct_lpd" logs/smoke_reconstruct.log \
    "$PY" scripts/reconstruct_lpd.py --model 1 --samples 2 --res 24 --polish-steps 2 \
      --hold-out-geoms 2 \
      --phases "$FLOW_PHASES" --operator-res "$FLOW_OPERATOR_RES" \
      --ckpt "$OUT/lpd_flow.pt" --calibration "$CAL" "${SIZE[@]}" \
      --support-from results/smoke/convex_Asteroid01.stl \
      --medoid-volume-only --out results/smoke/Asteroid01.stl
  tail -20 logs/smoke_reconstruct.log
  log "    wrote results/smoke/Asteroid01.stl"

  # The non-convex track, which the stages above never touch: it needs a calibrated
  # instrument and nothing else the flow produced. Model 3 rather than 1, because it is the
  # only released body with a concavity to find, so it is the one where a correction that
  # does nothing is visible.
  log "=== 11/14 calibrate: two steps on the blender channel, models 1 and 3"
  run "calibrate" logs/smoke_calibrate.log \
    "$PY" scripts/calibrate.py --channel blender --models 1 3 --steps 2 \
      --phases 8 "${SIZE[@]}" --truth-faces "$TRUTH_FACES" \
      --out "$OUT/instrument_smoke.pt" --report "$OUT/instrument_smoke.json"

  # 80 designed starts rather than a handful: the grid is caps first and waists from index
  # 72, so anything under that never exercises the family added for multi-lobed bodies.
  # --max-degree 16 drops the top stage of the ladder, which is more than half its renders
  # at one render per coordinate, and changes nothing about the path being checked: the
  # extraction, the polish, the export and the measurement are the same code at any ceiling.
  log "=== 12/14 reconstruct_gn: model 3, one iteration a stage, both start families"
  ALLOW=3   # a refused export; see run() above
  run "reconstruct_gn" logs/smoke_gn.log \
    "$PY" scripts/reconstruct_gn.py --model 3 --channel blender \
      --calibration "$OUT/instrument_smoke.pt" \
      --max-stage-iters 1 --max-degree 16 \
      --restarts 2 --restart-keep 1 --screen-starts 80 \
      --phases 8 --operator-res 24 --export-res 32 --export-phases 8 \
      "${SIZE[@]}" --hold-out-geoms 2 --out results/smoke/gn/Asteroid03.stl
  tail -8 logs/smoke_gn.log

  log "=== 13/14 reconstruct_map: model 3, a few descent steps on the same objective"
  run "reconstruct_map" logs/smoke_map.log \
    "$PY" scripts/reconstruct_map.py --model 3 --channel blender \
      --calibration "$OUT/instrument_smoke.pt" \
      --steps 4 --every 2 --ckpt-every 2 \
      --phases 8 --operator-res 24 --export-res 32 --export-phases 8 \
      "${SIZE[@]}" --hold-out-geoms 2 \
      --out results/smoke/map/Asteroid03.stl
  tail -8 logs/smoke_map.log
  ALLOW=""

  # Both tracks at once, which is what run_nonconvex.sh does: the decision between two
  # solvers of one body is made on the cameras held out of both fits. Model 3 is public, so
  # this also checks the path that reads a released truth. --into keeps the smoke test out of
  # the real submission.
  log "=== 14/14 select_answers: both tracks, one decision, then the submission check"
  # A track whose export was refused has no STL, and the selection keeps the convex answer
  # for it and says so; with both refused the selection is still a complete submission.
  run "select_answers" logs/smoke_select.log \
    "$PY" scripts/select_answers.py --refined results/smoke/gn results/smoke/map \
      --models 3 --into results/smoke/submission
  run "check_submission" logs/smoke_check.log \
    "$PY" scripts/check_submission.py results/smoke/submission
  log "    the chosen bodies are under results/smoke/submission/"
else
  log "=== 9/14 onwards: skipped (dataset/raw not present)"
  log "    Both need the real measured curves, so they can't run offline. Everything up to"
  log "    here (library, fit, corpus, prior, flow, decision check) is proven wired; download"
  log "    dataset/raw to also exercise the last two stages."
fi

dt=$(( $(date +%s) - t_start ))
log "=== smoke test complete in ${dt}s"
log "    library:  $LIB_DIR/report.md"
log "    codes:    $OUT/corpus_codes.npz"
log "    corpus:   $OUT/corpus.npz"
log "    prior:    $OUT/prior_flow.pt"
log "    flow:     $OUT/lpd_flow.pt"
log "    the non-convex track, when dataset/raw was present:"
log "    instrument: $OUT/instrument_smoke.pt"
log "    bodies:     results/smoke/{gn,map}/Asteroid03.stl"
log "    submission: results/smoke/submission/ (selection.json says which track won)"
log ""
log "If this all ran without error, scripts/run_remote_pipeline.sh should too. Nothing"
log "here touched runs/corpus_codes.npz, runs/corpus.npz, runs/lpd_flow.pt, models/ or"
log "dataset/generated/shapes -- delete runs/smoke/ and dataset/generated/shapes_smoke/"
log "whenever you like."
