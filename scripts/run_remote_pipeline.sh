#!/usr/bin/env bash
# Build the shape library, then run the whole pipeline against it end to end, on a fresh
# or resumed remote machine.
#
#   git clone <repo> hac26 && cd hac26
#   scripts/run_remote_pipeline.sh
#
# or from a laptop, to a machine that already has the repo:
#
#   rsync -az --exclude runs --exclude dataset/generated . remote:hac26/
#   ssh remote 'cd hac26 && nohup scripts/run_remote_pipeline.sh > pipeline.out 2>&1 &'
#
# scripts/_venv_setup.sh creates a .venv if there is none, activates it, and installs the
# dependencies when they are not already importable, so a repeat run does not download torch
# again.
#
# Every stage writes a marker file under runs/.done/ when it finishes, recording the settings
# AND the source it ran with, and is skipped on the next invocation only if both still match.
# A pre-empted or disconnected run can therefore be relaunched as is, and editing a shape
# generator, a loss or the forward model reruns the stages downstream of it without anything
# having to be asked for: the alternative is a run that continues from an artefact the current
# code would not have produced. --force-stage remains for redoing a stage nothing has changed.
#
#   ./scripts/run_remote_pipeline.sh                      # run everything, skip done stages
#   ./scripts/run_remote_pipeline.sh --force-stage fit     # redo `fit` and everything after
#   N_BODIES=2000 ./scripts/run_remote_pipeline.sh         # override any variable below
#
# Stages, in order, and what each needs:
#
#   0. models       scripts/fetch_shape_models.py    -- downloads public asteroid shape
#                   models into SHAPE_MODELS_DIR; skipped with FETCH_MODELS=0
#   0b. objects     scripts/fetch_objects.py         -- everyday printable objects from
#                   Thingi10K into SHAPE_MODELS_DIR/objects; only with FETCH_OBJECTS=1
#   1. library      scripts/build_shape_library.py  -- CPU only; draws on the shape models
#                   and objects when SHAPE_MODELS_DIR has any
#   2. design       scripts/make_design.py           -- GPU if available, else CPU
#   3. calibrate    scripts/calibrate.py             -- needs dataset/raw and nvdiffrast;
#                   skipped if models/instrument_calibration.pt is already present
#   4. fit          scripts/fit_shapes.py            -- fits codes to the library
#   4b. corpus      scripts/build_corpus.py          -- renders every body and runs the convex
#                   stage on it; needs nvdiffrast and the convex checkpoint CONVEX_CKPT
#   4c. prior       scripts/train_prior.py           -- the prior part of the flow; no operator
#   5. flow         scripts/train_lpd.py             -- the data part with one expert, on the
#                   straight line between noise and body; needs nvdiffrast on a GPU
#   5b. flow-experts the same run continued, branched into its experts so that each interval
#                   of t has its own velocity; the main training phase (see train_lpd.py)
#   5c. decision    scripts/decision_check.py         -- reconstructs held-out corpus bodies
#                   and scores every rule for picking the answer against their truth
#   6. convex       scripts/reconstruct.py, all ten models -- the starts the flow corrects;
#                   needs dataset/raw
#   6b. reconstruct scripts/reconstruct_lpd.py, all ten models -- needs nvdiffrast
#   7. score        hac26/scoring/voxel.py and side_view.py on the public models --
#                   needs dataset/raw
#   8. nonconvex    scripts/run_nonconvex.sh -- calibrates the rendered channel, corrects
#                   every body with both solvers and decides which corrections are submitted.
#                   This is the stage that writes results/submission; NONCONVEX=0 skips it.
#
# The exact forward model renders with nvdiffrast, which scripts/setup_toolchain.sh builds;
# the pipeline stops before the calibration if it cannot be imported.
#
# All stdout and stderr also go to logs/<stage>.log.
set -uo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------- configuration
N_BODIES=${N_BODIES:-1000}
LIB_SEED=${LIB_SEED:-0}
LIB_WORKERS=${LIB_WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}
LIB_RES=${LIB_RES:-64}
LIB_DIR=${LIB_DIR:-dataset/generated/shapes}
SHAPE_MODELS_DIR=${SHAPE_MODELS_DIR:-dataset/shape_models}
FETCH_MODELS=${FETCH_MODELS:-1}
FETCH_OBJECTS=${FETCH_OBJECTS:-0}   # needs `pip install thingi10k` and a few GB of download
N_OBJECTS=${N_OBJECTS:-600}

DESIGN_N=${DESIGN_N:-4096}
DESIGN_DEVICE=${DESIGN_DEVICE:-}

FIT_WORKERS=${FIT_WORKERS:-$LIB_WORKERS}
FIT_POINTS=${FIT_POINTS:-60000}   # surface samples per body, about twenty-three per
                                  # depth. The fit refuses fewer than twelve per depth
                                  # and reproduces a deeply carved body at seventeen,
                                  # so this number is only meaningful against
                                  # field.N_NODES and has to be reread whenever the
                                  # representation changes; unset it to take the floor
                                  # the code derives (scripts/fit_shapes.py)
CODES_FILE=runs/corpus_codes.npz

CONVEX_CKPT=${CONVEX_CKPT:-models/lpd_convex.pt}   # the convex stage, whose starts the flow corrects
CORPUS_FILE=runs/corpus.npz
CORPUS_WORKERS=${CORPUS_WORKERS:-8}

PRIOR_STEPS=${PRIOR_STEPS:-20000}   # a cap: the prior stops early once the held-out loss plateaus
PRIOR_BATCH=${PRIOR_BATCH:-64}

FLOW_STEPS=${FLOW_STEPS:-1000}   # a cap: training stops early once the held-out loss plateaus
FLOW_PHASES=${FLOW_PHASES:-96}
FLOW_BATCH=${FLOW_BATCH:-2}
FLOW_VAL_BODIES=${FLOW_VAL_BODIES:-16}  # held out of training: early stopping scores them and
                                        # the decision check reconstructs them; 0 turns both off
FLOW_VAL_EVERY=${FLOW_VAL_EVERY:-200}
FLOW_PATIENCE=${FLOW_PATIENCE:-5}
FLOW_CKPT_EVERY=${FLOW_CKPT_EVERY:-100}   # steps between resumable checkpoints; 0 disables
FLOW_CKPT=${FLOW_CKPT:-runs/lpd_flow.pt.ckpt}   # under runs/, not /tmp: it has to outlive
                                                # the job that wrote it
FLOW_LOG_EVERY=${FLOW_LOG_EVERY:-10}
FLOW_OPERATOR_RES=${FLOW_OPERATOR_RES:-32}
FLOW_TRAIN_GEOMS=${FLOW_TRAIN_GEOMS:-28}   # geometries the operator renders per step; all of them
FLOW_FIT_WEIGHT=${FLOW_FIT_WEIGHT:-1.0}   # the interval it applies over is the last expert's
                                          # and is derived from their number, so it is not a
                                          # setting here (train_lpd.FIT_FROM)
FLOW_EXTRA_STEPS=${FLOW_EXTRA_STEPS:-1000}  # cap on the second run's extra steps: branched
                                            # into experts, the main phase; 0 skips it

NONCONVEX=${NONCONVEX:-1}            # 1 runs scripts/run_nonconvex.sh at the end, which is
                                     # the stage that writes results/submission
NONCONVEX_TIME_BUDGET=${NONCONVEX_TIME_BUDGET:-0}   # seconds one body of one solver may take
                                     # there; 0 is no cap. On a job with a wallclock this is
                                     # what stops a queue of ten bodies being spent on the
                                     # first of them
RECON_SAMPLES=${RECON_SAMPLES:-64}   # draws per model. They are the candidates and they are
                                     # what the consensus bodies are built from, so this also
                                     # sets how finely a consensus level can be placed: the
                                     # occupied fraction of a voxel takes only k/RECON_SAMPLES.
                                     # The medoid is quadratic in it and measured at about four
                                     # minutes a model here; everything else is linear.
RECON_POLISH_STEPS=${RECON_POLISH_STEPS:-30}   # most gradient steps of the polish per draw; 0 skips it
RECON_RES=${RECON_RES:-96}
RECON_SNAP=${RECON_SNAP:-0}
# Weight on the data part of the velocity when sampling (lpd_flow.LPDFlow.velocity).
# One is the model as trained. The decision stage reconstructs held-out bodies at each
# of RECON_GUIDANCE_SWEEP and names the weight that scores best. When RECON_GUIDANCE is
# not set by the caller, the pipeline reads that value before reconstruction.
RECON_GUIDANCE_WAS_SET=${RECON_GUIDANCE+x}
RECON_GUIDANCE=${RECON_GUIDANCE:-}
RECON_GUIDANCE_SWEEP=${RECON_GUIDANCE_SWEEP:-1.0 1.5 2.0 3.0}
# Whether to measure that weight rather than take the default. Off, because measuring it
# costs FLOW_VAL_BODIES x |RECON_GUIDANCE_SWEEP| x RECON_SAMPLES draws -- 4096 at the
# settings above, each about what one challenge body costs -- and decision_check.py writes
# its one JSON at the end, so a run stopped by a wallclock keeps nothing. Left on by default
# it is the stage a pipeline reaches after training and does not return from.
#
# Shrinking it instead was considered and rejected: about a quarter of draws survive the
# single-component and finite-misfit gates, so a cut-down sweep separates the weights on a
# handful of usable candidates, and decision_check itself reports that the weights are often
# within 1e-3 of each other in mean Dice ("not a margin: leaving RECON_GUIDANCE alone is as
# good"). An estimate that noisy would override a sound default on close to a coin flip.
#
# Set RUN_DECISION=1 with the compute to do it properly, or pass RECON_GUIDANCE directly.
RUN_DECISION=${RUN_DECISION:-0}
MEDOID_VOLUME_ONLY=${MEDOID_VOLUME_ONLY:-0}
MEDOID_SIDE_POINTS=${MEDOID_SIDE_POINTS:-200000}
MEDOID_SIDE_DIRS=${MEDOID_SIDE_DIRS:-36}
MEDOID_SIDE_RES=${MEDOID_SIDE_RES:-512}
MEDOID_SIDE_MODE=${MEDOID_SIDE_MODE:-side}

# shellcheck disable=SC1091
source scripts/_venv_setup.sh   # creates and activates the venv, installs deps, sets PY

DATA_DIR=${DATA_DIR:-dataset/raw}
FORCE_STAGE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --force-stage) FORCE_STAGE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p runs runs/.done logs results/lpd

# Everything the run needs, checked while it is still cheap to fix. A missing wheel or an
# unbuilt extension otherwise surfaces hours in, and some of those failures do not look like
# what they are: a missing decimation package used to read as a body with no curves.
$PY scripts/preflight.py --data-dir "$DATA_DIR" || {
  echo "preflight failed; not starting a long run. See above." >&2; exit 1; }

# ---------------------------------------------------------------- helpers
STAGES_AFTER_FORCE=0

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# A stage is skipped when its marker records the signature this run would produce, so a
# signature naming only the settings lets a stage be skipped after the code that produces its
# artefact has been rewritten: the run then continues from an artefact the current code would
# not have made, and nothing says so. The digests below put the source into the signature, so
# editing a shape generator, a loss or the forward model invalidates the stages downstream of
# it and no others.
src_digest() {
  $PY - "$@" <<'PYEOF'
import hashlib, pathlib, sys
h = hashlib.sha256()
for name in sorted(sys.argv[1:]):
    p = pathlib.Path(name)
    h.update(name.encode())
    h.update(p.read_bytes() if p.is_file() else b"<missing>")
print(h.hexdigest()[:16])
PYEOF
}

SRC_LIBRARY=$(src_digest hac26/shape_library.py hac26/shapes.py hac26/library_io.py \
                         hac26/library_metrics.py scripts/build_shape_library.py)
SRC_FORWARD=$(src_digest hac26/conventions.py hac26/geometry.py hac26/noise.py \
                         hac26/forward/mesh/exact.py hac26/forward/mesh/instrument.py \
                         hac26/forward/mesh/radiosity.py hac26/forward/mesh/raster.py \
                         hac26/forward/mesh/sensor.py hac26/forward/shared/coarea.py \
                         hac26/forward/shared/common.py \
                         hac26/forward/shared/software_raster.py scripts/calibrate.py)
SRC_FIT=$(src_digest hac26/field.py scripts/fit_shapes.py)
SRC_CORPUS=$(src_digest scripts/build_corpus.py hac26/solvers/operator.py \
                        hac26/solvers/lpd_convex.py)
SRC_FLOW=$(src_digest hac26/solvers/lpd_flow.py scripts/train_lpd.py scripts/train_prior.py)
SRC_OUTPUT=$(src_digest scripts/reconstruct_lpd.py scripts/decision_check.py \
                        hac26/solvers/output.py hac26/scoring/side_view.py \
                        hac26/scoring/voxel.py hac26/recon.py)

stage_signature() {
  case "$1" in
    models)
      printf 'stage=models\nSHAPE_MODELS_DIR=%s\n' "$SHAPE_MODELS_DIR"
      ;;
    objects)
      printf 'stage=objects\nSHAPE_MODELS_DIR=%s\nN_OBJECTS=%s\n' "$SHAPE_MODELS_DIR" "$N_OBJECTS"
      ;;
    library)
      printf 'stage=library\nN_BODIES=%s\nLIB_SEED=%s\nLIB_RES=%s\nLIB_DIR=%s\nSHAPE_MODELS=%s\nSRC=%s\n' \
        "$N_BODIES" "$LIB_SEED" "$LIB_RES" "$LIB_DIR" "$(shape_model_list)" "$SRC_LIBRARY"
      ;;
    design)
      printf 'stage=design\nDESIGN_N=%s\nDESIGN_DEVICE=%s\n' \
        "$DESIGN_N" "$DESIGN_DEVICE"
      ;;
    calibrate)
      printf 'stage=calibrate\nDATA_DIR=%s\nSRC=%s\n' "$DATA_DIR" "$SRC_FORWARD"
      ;;
    fit)
      stage_signature library | sed 's/^stage=library$/stage=fit/'
      printf 'DESIGN_N=%s\nFIT_POINTS=%s\nCODES_FILE=%s\nSRC=%s\n' \
        "$DESIGN_N" "$FIT_POINTS" "$CODES_FILE" "$SRC_FIT"
      ;;
    # Each later stage's signature extends the one before it, so a change anywhere upstream
    # reruns everything downstream.
    corpus)
      stage_signature fit | sed 's/^stage=fit$/stage=corpus/'
      printf 'FLOW_PHASES=%s\nFLOW_OPERATOR_RES=%s\nCONVEX_CKPT=%s\nCORPUS_FILE=%s\nSRC=%s\nSRC_FWD=%s\n' \
        "$FLOW_PHASES" "$FLOW_OPERATOR_RES" "$CONVEX_CKPT" "$CORPUS_FILE" "$SRC_CORPUS" "$SRC_FORWARD"
      ;;
    prior)
      stage_signature corpus | sed 's/^stage=corpus$/stage=prior/'
      printf 'PRIOR_STEPS=%s\nPRIOR_BATCH=%s\nFLOW_VAL_BODIES=%s\nSRC=%s\n' \
        "$PRIOR_STEPS" "$PRIOR_BATCH" "$FLOW_VAL_BODIES" "$SRC_FLOW"
      ;;
    flow)
      stage_signature prior | sed 's/^stage=prior$/stage=flow/'
      printf 'FLOW_STEPS=%s\nFLOW_BATCH=%s\nFLOW_VAL_EVERY=%s\nFLOW_PATIENCE=%s\nFLOW_CKPT_EVERY=%s\nFLOW_CKPT=%s\nFLOW_LOG_EVERY=%s\nFLOW_TRAIN_GEOMS=%s\nFLOW_FIT_WEIGHT=%s\n' \
        "$FLOW_STEPS" "$FLOW_BATCH" "$FLOW_VAL_EVERY" "$FLOW_PATIENCE" \
        "$FLOW_CKPT_EVERY" "$FLOW_CKPT" "$FLOW_LOG_EVERY" "$FLOW_TRAIN_GEOMS" \
        "$FLOW_FIT_WEIGHT"
      ;;
    flow-experts)
      stage_signature flow | sed 's/^stage=flow$/stage=flow-experts/'
      printf 'FLOW_EXTRA_STEPS=%s\n' "$FLOW_EXTRA_STEPS"
      ;;
    decision)
      stage_signature flow-experts | sed 's/^stage=flow-experts$/stage=decision/'
      printf 'RECON_SAMPLES=%s\nRECON_POLISH_STEPS=%s\nRECON_RES=%s\nSWEEP=%s\nSIDE_POINTS=%s\nSRC=%s\n' \
        "$RECON_SAMPLES" "$RECON_POLISH_STEPS" "$RECON_RES" "$RECON_GUIDANCE_SWEEP" \
        "$MEDOID_SIDE_POINTS" "$SRC_OUTPUT"
      ;;
    convex)
      printf 'stage=convex\nDATA_DIR=%s\nCONVEX_CKPT=%s\nSRC=%s\nSRC_FWD=%s\n' \
        "$DATA_DIR" "$CONVEX_CKPT" "$SRC_CORPUS" "$SRC_FORWARD"
      ;;
    reconstruct)
      printf 'stage=reconstruct\nDESIGN_N=%s\nFLOW_STEPS=%s\nFLOW_PHASES=%s\nFLOW_BATCH=%s\nFLOW_OPERATOR_RES=%s\nFLOW_TRAIN_GEOMS=%s\nFLOW_EXTRA_STEPS=%s\nCONVEX_CKPT=%s\nRECON_SAMPLES=%s\nRECON_POLISH_STEPS=%s\nRECON_RES=%s\nRECON_SNAP=%s\nMEDOID_VOLUME_ONLY=%s\nMEDOID_SIDE_POINTS=%s\nMEDOID_SIDE_DIRS=%s\nMEDOID_SIDE_RES=%s\nMEDOID_SIDE_MODE=%s\n' \
        "$DESIGN_N" "$FLOW_STEPS" "$FLOW_PHASES" "$FLOW_BATCH" "$FLOW_OPERATOR_RES" \
        "$FLOW_TRAIN_GEOMS" "$FLOW_EXTRA_STEPS" "$CONVEX_CKPT" \
        "$RECON_SAMPLES" "$RECON_POLISH_STEPS" "$RECON_RES" "$RECON_SNAP" \
        "$MEDOID_VOLUME_ONLY" "$MEDOID_SIDE_POINTS" "$MEDOID_SIDE_DIRS" \
        "$MEDOID_SIDE_RES" "$MEDOID_SIDE_MODE"
      printf 'RECON_GUIDANCE=%s\nSRC=%s\nSRC_FLOW=%s\nSRC_FWD=%s\n' \
        "$RECON_GUIDANCE" "$SRC_OUTPUT" "$SRC_FLOW" "$SRC_FORWARD"
      ;;
    score)
      printf 'stage=score\nDATA_DIR=%s\nRECON_DIR=results/lpd\nRECON_SAMPLES=%s\nRECON_RES=%s\nRECON_SNAP=%s\nMEDOID_VOLUME_ONLY=%s\nMEDOID_SIDE_POINTS=%s\nMEDOID_SIDE_DIRS=%s\nMEDOID_SIDE_RES=%s\nMEDOID_SIDE_MODE=%s\n' \
        "$DATA_DIR" "$RECON_SAMPLES" "$RECON_RES" "$RECON_SNAP" \
        "$MEDOID_VOLUME_ONLY" "$MEDOID_SIDE_POINTS" "$MEDOID_SIDE_DIRS" \
        "$MEDOID_SIDE_RES" "$MEDOID_SIDE_MODE"
      printf 'SRC=%s\n' "$SRC_OUTPUT"
      ;;
    *)
      printf 'stage=%s\n' "$1"
      ;;
  esac
}

should_run() {
  # A stage runs if it was named by --force-stage, if a stage before it was, or if it has no
  # marker recording exactly this run's settings.
  local stage="$1"
  if [ "$stage" = "$FORCE_STAGE" ]; then STAGES_AFTER_FORCE=1; fi
  if [ "$STAGES_AFTER_FORCE" = "1" ]; then return 0; fi
  local marker="runs/.done/$stage"
  if [ ! -f "$marker" ]; then return 0; fi
  local want have
  want="$(stage_signature "$stage")"
  have="$(cat "$marker")"
  if [ "$want" != "$have" ]; then
    log "=== $stage: config changed since marker was written; rerunning"
    return 0
  fi
  return 1
}

mark_done() { stage_signature "$1" > "runs/.done/$1"; }

run_stage() {
  # run_stage NAME OUTPUT_FILE -- CMD...
  local name="$1" out="$2"; shift 2
  if ! should_run "$name"; then
    log "=== $name: skipped (already done -- rm runs/.done/$name or use --force-stage to redo)"
    return 0
  fi
  log "=== $name: starting"
  # Append rather than truncate: a stage relaunched after a pre-emption continues the earlier
  # attempt, whose log shows what it already did.
  echo "=== $name: starting $(date -u +%FT%TZ) ===" >> "logs/$name.log"
  if "$@" 2>&1 | tee -a "logs/$name.log"; then
    mark_done "$name"
    log "=== $name: done"
  else
    log "=== $name: FAILED -- see logs/$name.log"
    exit 1
  fi
}

valid_design() {
  $PY -c "import sys, numpy as np; n = int(sys.argv[1]); x = np.load(f'hac26/design{n}.npy'); assert x.shape == (n, 3); assert np.isfinite(x).all(); assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-6)" "$1"
}

# ---------------------------------------------------------------- 0. real shape models
# Asteroid models are downloaded here. Everyday objects (scripts/fetch_objects.py, which needs
# the thingi10k package and a long download) are fetched when FETCH_OBJECTS=1; they are an
# addition to the library, not a requirement, so a failure here is logged and the run goes
# on with whatever is under $SHAPE_MODELS_DIR/objects.
if [ "$FETCH_MODELS" = "1" ]; then
  run_stage models "$SHAPE_MODELS_DIR" \
    $PY scripts/fetch_shape_models.py --out "$SHAPE_MODELS_DIR"
else
  log "=== models: skipped (FETCH_MODELS=0)"
fi
if [ "$FETCH_OBJECTS" != "1" ]; then
  log "=== objects: skipped (FETCH_OBJECTS=0); using $SHAPE_MODELS_DIR/objects if present"
elif should_run objects; then
  log "=== objects: starting"
  echo "=== objects: starting $(date -u +%FT%TZ) ===" >> logs/objects.log
  if ! $PY -c "import thingi10k" >/dev/null 2>&1; then
    # in a subshell: a venv without pip must not stop the run over an optional stage
    ( _ensure_pip && python -m pip install -q thingi10k ) 2>&1 | tee -a logs/objects.log || true
  fi
  if $PY scripts/fetch_objects.py --out "$SHAPE_MODELS_DIR/objects" --n "$N_OBJECTS" \
      2>&1 | tee -a logs/objects.log; then
    mark_done objects
    log "=== objects: done"
  else
    log "=== objects: FAILED -- see logs/objects.log; continuing without new objects"
  fi
else
  log "=== objects: skipped (already done)"
fi

# ---------------------------------------------------------------- 1. shape library
# The library's signature names the shape-model and object files, so adding one reruns it.
shape_model_list() {
  [ -d "$SHAPE_MODELS_DIR" ] && find "$SHAPE_MODELS_DIR" -maxdepth 2 -type f 2>/dev/null \
    | grep -Ei '\.(obj|wf|stl|ply|tab|txt)$' | sort | tr '\n' ' '
}
LIB_ARGS=(--n "$N_BODIES" --seed "$LIB_SEED" --out "$LIB_DIR" --workers "$LIB_WORKERS"
  --res "$LIB_RES")
if [ -n "$(shape_model_list)" ]; then
  LIB_ARGS+=(--shape-models "$SHAPE_MODELS_DIR")
else
  log "=== library: no shape models under $SHAPE_MODELS_DIR; procedural families only"
fi
run_stage library "$LIB_DIR/manifest.json" \
  $PY scripts/build_shape_library.py "${LIB_ARGS[@]}"

# ---------------------------------------------------------------- 2. spherical design
DESIGN_ARGS=(--n "$DESIGN_N")
if [ -n "$DESIGN_DEVICE" ]; then
  DESIGN_ARGS+=(--device "$DESIGN_DEVICE")
fi
if should_run design; then
  if [ "$FORCE_STAGE" != "design" ] && [ -f "hac26/design${DESIGN_N}.npy" ] \
      && valid_design "$DESIGN_N"; then
    log "=== design: existing hac26/design${DESIGN_N}.npy is valid"
    mark_done design
  else
    run_stage design "hac26/design${DESIGN_N}.npy" \
      $PY scripts/make_design.py "${DESIGN_ARGS[@]}"
  fi
else
  log "=== design: skipped (already done -- rm runs/.done/design or use --force-stage to redo)"
fi

# ---------------------------------------------------------------- the rasteriser
# Everything from here on renders with the exact forward model. HAC26_SOFTWARE_RASTER=1
# selects the slow pure-torch stand-in, which is only for tests on a machine without a GPU.
if [ -z "${HAC26_SOFTWARE_RASTER:-}" ] && ! $PY -c "import nvdiffrast" >/dev/null 2>&1; then
  log "!!! nvdiffrast is not importable. Run scripts/setup_toolchain.sh first (it builds the"
  log "!!! CUDA toolchain nvdiffrast needs), then rerun this script."
  exit 1
fi

# ---------------------------------------------------------------- 3. instrument calibration
valid_instrument() {
  $PY -c "import sys; sys.path.insert(0, '.')
from hac26.forward.mesh.instrument import Instrument
Instrument.load('models/instrument_calibration.pt')" >/dev/null 2>&1
}
if [ "$STAGES_AFTER_FORCE" != "1" ] && [ "$FORCE_STAGE" != "calibrate" ] \
    && [ -f models/instrument_calibration.pt ] && valid_instrument; then
  log "=== calibrate: skipped (models/instrument_calibration.pt already present)"
  mark_done calibrate
elif [ -d "$DATA_DIR" ]; then
  if [ -f models/instrument_calibration.pt ] && ! valid_instrument; then
    log "=== calibrate: models/instrument_calibration.pt does not load into the current"
    log "    Instrument, so it is refitted. Commit the new one so later runs skip this stage."
  fi
  run_stage calibrate models/instrument_calibration.pt \
    $PY scripts/calibrate.py --data-dir "$DATA_DIR"
else
  log "!!! calibrate: $DATA_DIR not present and no models/instrument_calibration.pt."
  log "!!! Training and reconstruction need the calibrated instrument -- stopping here."
  exit 1
fi

# ---------------------------------------------------------------- 4. per-body codes
run_stage fit "$CODES_FILE" \
  $PY scripts/fit_shapes.py \
    --bodies "$N_BODIES" --shapes-dir "$LIB_DIR" --seed "$LIB_SEED" \
    --workers "$FIT_WORKERS" \
    --points "$FIT_POINTS" \
    --out "$CODES_FILE"

# ---------------------------------------------------------------- 4b. the corpus
# Every body's curves from the exact operator, and the start the convex stage makes from
# them, which is what the flow learns to correct. Resumable body by body under
# $CORPUS_FILE.parts/, so a pre-empted job loses at most one body.
if [ ! -f "$CONVEX_CKPT" ]; then
  log "!!! corpus: $CONVEX_CKPT not present. The flow trains from the convex stage's starts,"
  log "!!! so its checkpoint is needed here (scripts/export_model.py writes it)."
  exit 1
fi
run_stage corpus "$CORPUS_FILE" \
  $PY scripts/build_corpus.py \
    --bodies "$N_BODIES" --phases "$FLOW_PHASES" --operator-res "$FLOW_OPERATOR_RES" \
    --codes-file "$CODES_FILE" --convex "$CONVEX_CKPT" --out "$CORPUS_FILE" \
    --workers "$CORPUS_WORKERS"

# ---------------------------------------------------------------- 4c. the prior part
run_stage prior runs/prior_flow.pt \
  $PY scripts/train_prior.py \
    --steps "$PRIOR_STEPS" --batch "$PRIOR_BATCH" \
    --val-bodies "$FLOW_VAL_BODIES" --corpus "$CORPUS_FILE" --out runs/prior_flow.pt

# ---------------------------------------------------------------- 5. flow
# One expert first; the second run branches it into the experts (train_lpd.py --experts).
run_stage flow runs/lpd_flow.pt \
  $PY scripts/train_lpd.py \
    --steps "$FLOW_STEPS" --batch "$FLOW_BATCH" --experts 1 \
    --train-geoms "$FLOW_TRAIN_GEOMS" --out runs/lpd_flow.pt \
    --val-bodies "$FLOW_VAL_BODIES" --val-every "$FLOW_VAL_EVERY" \
    --patience "$FLOW_PATIENCE" \
    --ckpt-every "$FLOW_CKPT_EVERY" --ckpt-file "$FLOW_CKPT" \
    --log-every "$FLOW_LOG_EVERY" \
    --fit-weight "$FLOW_FIT_WEIGHT" \
    --corpus "$CORPUS_FILE"

# ---------------------------------------------------------------- 5b. flow, branched
# The same run continued from its checkpoint, branched into the default number of experts, so
# that each interval of t gets its own velocity, for up to FLOW_EXTRA_STEPS more steps. This
# is the main phase; the first run only prepares it.
if [ "$FLOW_EXTRA_STEPS" -gt 0 ]; then
  run_stage flow-experts runs/lpd_flow.pt \
    $PY scripts/train_lpd.py \
      --steps "$FLOW_STEPS" --extra-steps "$FLOW_EXTRA_STEPS" \
      --batch "$FLOW_BATCH" \
      --train-geoms "$FLOW_TRAIN_GEOMS" --out runs/lpd_flow.pt \
      --val-bodies "$FLOW_VAL_BODIES" --val-every "$FLOW_VAL_EVERY" \
      --patience "$FLOW_PATIENCE" \
      --ckpt-every "$FLOW_CKPT_EVERY" --ckpt-file "$FLOW_CKPT" \
      --log-every "$FLOW_LOG_EVERY" \
      --fit-weight "$FLOW_FIT_WEIGHT" \
      --corpus "$CORPUS_FILE"
else
  log "=== flow-experts: skipped (FLOW_EXTRA_STEPS=0)"
fi

# ---------------------------------------------------------------- 5c. the decision rule
# Held-out corpus bodies are reconstructed as the challenge models will be, and every rule for
# picking the answer is scored against their truth (see decision_check.py). The summary is in
# logs/decision.log and runs/decision_check.json.
if [ "$RUN_DECISION" = "1" ] && [ "$FLOW_VAL_BODIES" -gt 0 ]; then
  run_stage decision runs/decision_check.json \
    $PY scripts/decision_check.py \
      --ckpt runs/lpd_flow.pt --corpus "$CORPUS_FILE" --val-bodies "$FLOW_VAL_BODIES" \
      --bodies "$FLOW_VAL_BODIES" --samples "$RECON_SAMPLES" \
      --polish-steps "$RECON_POLISH_STEPS" --res "$RECON_RES" \
      --guidance $RECON_GUIDANCE_SWEEP \
      --side-points "$MEDOID_SIDE_POINTS" --out runs/decision_check.json
elif [ "$FLOW_VAL_BODIES" -gt 0 ]; then
  log "=== decision: skipped (RUN_DECISION=0; reconstruction uses RECON_GUIDANCE, default 1.0)"
else
  log "=== decision: skipped (FLOW_VAL_BODIES=0)"
fi

if [ -f runs/decision_check.json ]; then
  BEST_GUIDANCE=$($PY -c "import json, sys; p=sys.argv[1]; d=json.load(open(p)); v=d.get('best_guidance'); print('' if v is None else v)" runs/decision_check.json)
  if [ -z "${RECON_GUIDANCE:-}" ] && [ -n "$BEST_GUIDANCE" ]; then
    RECON_GUIDANCE="$BEST_GUIDANCE"
    log "=== reconstruct: using best guidance from runs/decision_check.json: $RECON_GUIDANCE"
  elif [ -n "${RECON_GUIDANCE_WAS_SET:-}" ] && [ -n "$BEST_GUIDANCE" ] \
       && [ "$RECON_GUIDANCE" != "$BEST_GUIDANCE" ]; then
    log "!!! reconstruct: RECON_GUIDANCE=$RECON_GUIDANCE overrides decision best $BEST_GUIDANCE"
  fi
fi
if [ -z "${RECON_GUIDANCE:-}" ]; then
  RECON_GUIDANCE=1.0
  if [ "$RUN_DECISION" = "1" ]; then
    log "!!! reconstruct: the decision stage ran but named no weight; falling back to 1.0"
  else
    # Not a warning: with RUN_DECISION=0 this is the designed path, and one is the weight
    # the flow was trained at (lpd_flow.LPDFlow.velocity).
    log "=== reconstruct: guidance 1.0, the weight the flow was trained at (RUN_DECISION=0)"
  fi
fi

# ---------------------------------------------------------------- 6. the convex starts
# The convex stage's reconstruction of every model, made the way the corpus starts were: the
# same checkpoint and the same decode, so the flow meets at reconstruction what it trained
# on. A model is redone when its STL is missing or older than its curves or the checkpoint,
# so re-downloaded data is picked up.
if [ ! -d "$DATA_DIR" ]; then
  log "!!! convex: $DATA_DIR not present; the measured curves are needed from here on."
  exit 1
fi
if should_run convex; then
  log "=== convex: starting (10 models)"
  mkdir -p results/convex
  ok=1
  for M in 1 2 3 4 5 6 7 8 9 10; do
    P=$(printf "%02d" "$M")
    OUT="results/convex/Asteroid$P.stl"
    if [ -s "$OUT" ] && [ ! "$CONVEX_CKPT" -nt "$OUT" ] && [ "$STAGES_AFTER_FORCE" != "1" ] \
       && [ -z "$(find "$DATA_DIR" -name "Asteroid*${P}_lightcurve_*" -newer "$OUT" 2>/dev/null)" ]; then
      log "  --- model $M: $OUT is current"
      continue
    fi
    log "  --- model $M -> $OUT"
    if ! $PY scripts/reconstruct.py --ckpt "$CONVEX_CKPT" --model "$M" --data-dir "$DATA_DIR" \
        --out "$OUT" 2>&1 | tee -a logs/convex.log; then
      log "  --- model $M FAILED"
      ok=0
    fi
  done
  [ "$ok" = "1" ] && mark_done convex || { log "=== convex: FAILED"; exit 1; }
  log "=== convex: done"
else
  log "=== convex: skipped (already done)"
fi

# ---------------------------------------------------------------- 6b. reconstruct all ten
if should_run reconstruct; then
  log "=== reconstruct: starting (10 models)"
  ok=1
  for M in 1 2 3 4 5 6 7 8 9 10; do
    P=$(printf "%02d" "$M")
    OUT="results/lpd/Asteroid$P.stl"
    # Each model is its own unit of work, so a job that dies on one model costs only that
    # model. The .json is written last, so a model counts as done only when both files exist
    # and are newer than the flow and the convex start they came from.
    if [ -s "$OUT" ] && [ -s "results/lpd/Asteroid$P.json" ] \
       && [ ! runs/lpd_flow.pt -nt "$OUT" ] && [ ! "results/convex/Asteroid$P.stl" -nt "$OUT" ] \
       && [ "$STAGES_AFTER_FORCE" != "1" ]; then
      log "  --- model $M: skipped ($OUT already written -- rm it to redo just this one)"
      continue
    fi
    log "  --- model $M -> $OUT (started $(date -u +%H:%M:%S))"
    RECON_ARGS=(--model "$M" --samples "$RECON_SAMPLES" --res "$RECON_RES"
      --phases "$FLOW_PHASES" --operator-res "$FLOW_OPERATOR_RES"
      --polish-steps "$RECON_POLISH_STEPS" --guidance "$RECON_GUIDANCE"
      --ckpt runs/lpd_flow.pt --data-dir "$DATA_DIR" --out "$OUT"
      --medoid-side-points "$MEDOID_SIDE_POINTS"
      --medoid-side-dirs "$MEDOID_SIDE_DIRS"
      --medoid-side-res "$MEDOID_SIDE_RES"
      --medoid-side-mode "$MEDOID_SIDE_MODE")
    if [ "$RECON_SNAP" = "1" ]; then
      RECON_ARGS+=(--snap)
    fi
    if [ "$MEDOID_VOLUME_ONLY" = "1" ]; then
      RECON_ARGS+=(--medoid-volume-only)
    fi
    if ! $PY scripts/reconstruct_lpd.py "${RECON_ARGS[@]}" \
        2>&1 | tee -a logs/reconstruct.log; then
      log "  --- model $M FAILED"
      ok=0
    fi
  done
  [ "$ok" = "1" ] && mark_done reconstruct || { log "=== reconstruct: FAILED"; exit 1; }
  log "=== reconstruct: done"
else
  log "=== reconstruct: skipped (already done)"
fi

# ---------------------------------------------------------------- 7. score the public models
# The flow's answers and the convex starts they came from, side by side: the flow has to
# beat its start on the non-convex public body without losing on the near-convex ones, since
# a carved-in dent that is not there costs as much as a missed one.
# PYTHONPATH: the two scorers are the only entry points outside scripts/, and the scripts are
# what put the repo root on sys.path -- _venv_setup.sh installs the dependencies by name and
# not the package itself, so without this `import hac26` fails and the stage cannot start.
# The two side-view runs need separate --out paths: they share one default, so the flow's run
# would otherwise overwrite the convex baseline and leave only half the comparison on disk.
if [ -d "$DATA_DIR" ]; then
  run_stage score "" env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" bash -c "
    echo '--- convex starts' &&
    $PY hac26/scoring/voxel.py --stl results/convex/Asteroid0{1,2,3}.stl --models 1 2 3 &&
    $PY hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/convex --out runs/side_view_convex.json &&
    echo '--- flow' &&
    $PY hac26/scoring/voxel.py --stl results/lpd/Asteroid0{1,2,3}.stl --models 1 2 3 &&
    $PY hac26/scoring/side_view.py --models 1 2 3 --recon-dir results/lpd --out runs/side_view_lpd.json
  "
else
  log "=== score: skipped ($DATA_DIR not present)"
fi

# ---------------------------------------------------------------- 8. the non-convex track
# The stage that produces a submission. It is a runbook of its own, with its own per-body
# resume, so it is run rather than wrapped in a stage marker: a second invocation of this
# pipeline skips the bodies that are written and retries the ones that are not.
if [ "$NONCONVEX" = "1" ] && [ -d "$DATA_DIR" ]; then
  log "=== nonconvex: starting (both solvers, then the selection)"
  if DATA_DIR="$DATA_DIR" PY="$PY" TIME_BUDGET="$NONCONVEX_TIME_BUDGET" \
     ./scripts/run_nonconvex.sh 2>&1 | tee -a logs/nonconvex.log; then
    log "=== nonconvex: done"
  else
    log "=== nonconvex: FAILED -- see logs/nonconvex.log. The convex answers stand wherever"
    log "    no correction was written, so results/submission is still a submission."
  fi
elif [ "$NONCONVEX" != "1" ]; then
  log "=== nonconvex: skipped (NONCONVEX=0)"
else
  log "=== nonconvex: skipped ($DATA_DIR not present)"
fi

log "=== pipeline complete"
log "    library:  $LIB_DIR ($N_BODIES bodies; see $LIB_DIR/report.md)"
log "    normals:  $DESIGN_N"
log "    corpus:   $CORPUS_FILE"
log "    flow ckpt: runs/lpd_flow.pt"
log "    reconstructions: results/lpd/Asteroid*.stl"
