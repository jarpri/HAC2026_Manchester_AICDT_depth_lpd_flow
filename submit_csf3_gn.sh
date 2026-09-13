#!/bin/bash --login
# The non-convex correction on one CSF3 GPU: calibrate, correct the calibrating public model
# and every scored model with both solvers, then select between them and check the submission.
# Modelled on submit_csf3.sh (module loads, scratch, the Makefile's venv/toolchain/check), but
# runs scripts/run_nonconvex.sh instead of the full pipeline -- this needs a calibrated
# instrument and dataset/raw, not a shape library, a trained flow, or Thingi10K objects, so it
# does not fetch or build any of those.
#
#   ./submit_csf3_gn.sh                        # gpuL (L40S 48GB), the default below
#   CSF_PARTITION=gpuA ./submit_csf3_gn.sh     # A100 80GB
#   SCORED="4" ./submit_csf3_gn.sh             # one scored model, to see real per-model cost
#                                               # before committing to all of SCORED's default
#                                               # "4 5 6 7 8 9 10" -- see run_nonconvex.sh
#   TRACKS=gn ./submit_csf3_gn.sh              # one solver rather than both
#   sbatch submit_csf3_gn.sh                   # gpuL only; ignores CSF_PARTITION
#
# Every scripts/run_nonconvex.sh variable (CHANNEL, TRACKS, GN_DIR, MAP_DIR, RESTARTS,
# MAX_DEGREE, MAP_STEPS, TIME_BUDGET, REDO, PHASES, HOLD_OUT_GEOMS, CALIBRATE_STEPS,
# CALIBRATE_MODELS, CALIBRATION_MODEL, SCORED, DATA_DIR) can be set in the environment before
# calling this script; nothing here overrides them.
#
# TIME_BUDGET is the one to reach for if the wallclock below turns out to be too short. It
# bounds a single body of a single solver from below rather than above: the check falls
# between units of work, and the Gauss-Newton ladder's longest unit is one iteration of its
# top stage, at (L+1)^2 - 9 renders. Size the wallclock as bodies x (TIME_BUDGET + one such
# iteration), and lower MAX_DEGREE to shorten it. Both solvers checkpoint inside a body, so a
# run that spends its budget writes the best body it reached and the next job continues from
# there rather than starting that body again.
#
# gpuL rather than submit_csf3.sh's gpuA: reconstruct_gn.py's cost is dominated by rendering
# one candidate mesh at a time (no batching across restarts or the ladder's own coordinates --
# each Jacobian column is its own render), which is exactly the workload L40S's Ada Lovelace
# design (RT cores, rasterisation-oriented) suits better than A100's Ampere, tensor/bandwidth
# design for large-batch training -- the reason submit_csf3.sh picks gpuA is that it also runs
# the LPD's own training stages, which this script does not.
#
# No real per-candidate timing exists for this workload yet. Start with SCORED="4" (one
# model) to see what a real run costs on this GPU before trusting the 10h default below,
# which now has to cover two solvers over eight bodies -- the calibrating model plus seven
# scored ones -- rather than one. A body already written against the same instrument is
# skipped, so resubmitting this job after the wallclock continues the queue.
#SBATCH --job-name=hac26-gauss-newton
#SBATCH --partition=gpuL
#SBATCH -G 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8          # reconstruct_gn.py's own render loop is single-candidate,
                                    # not internally parallel, but preflight/data loading and
                                    # the CPU-side mesh extraction each candidate needs benefit
                                    # from a few cores -- 8 matches submit_csf3.sh's own gpuL
                                    # allowance without taking the full 12 a GPU permits
#SBATCH -t 10:00:00
#SBATCH --output=logs/csf3_gn_%j.out
#SBATCH --error=logs/csf3_gn_%j.err

# ---------------------------------------------------------------- submission
if [ -z "${SLURM_JOB_ID:-}" ]; then
  set -euo pipefail
  PARTITION=${CSF_PARTITION:-gpuL}
  case "$PARTITION" in
    gpuA|gpuL) ;;
    *) echo "CSF_PARTITION=$PARTITION: this script only offers gpuA or gpuL (free-tier" >&2
       echo "  access; gpuH/gpuH_short need an allocation account -- see submit_csf3.sh)" >&2
       exit 1 ;;
  esac
  cd "$(dirname "$0")"
  mkdir -p logs
  exec sbatch -p "$PARTITION" -c 8 -t "${CSF_TIME:-10:00:00}" "$(basename "$0")" "$@"
fi

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

if [ -n "${CSF_PARTITION:-}" ] && [ "$CSF_PARTITION" != "${SLURM_JOB_PARTITION:-}" ]; then
  echo "ERROR: CSF_PARTITION=$CSF_PARTITION, but this job is on ${SLURM_JOB_PARTITION:-?}." >&2
  echo "       Submit with ./submit_csf3_gn.sh rather than sbatch to use it." >&2
  exit 1
fi

# ---------------------------------------------------------------- modules (see submit_csf3.sh)
module purge
module load tools/env/proxy2
module load cuda/12.6.2
module load python/3.13.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi || { echo "ERROR: no GPU visible; this job needs one." >&2; exit 1; }

# ---------------------------------------------------------------- scratch
# Only dataset/raw and runs/ -- this pipeline needs neither a shape library nor Thingi10K
# objects, unlike submit_csf3.sh's full-pipeline scratch setup.
SCRATCH=${SCRATCH_DIR:-$HOME/scratch/hac26}
mkdir -p "$SCRATCH/cache"
export XDG_CACHE_HOME="$SCRATCH/cache"
export UV_CACHE_DIR="$SCRATCH/cache/uv"
mkdir -p dataset

# The path the scripts use, and where the bytes should live. A path that is already a
# symlink is left alone. A populated path with empty scratch is moved rather than deleted, so
# a dataset that has been fetched and validated once -- or rsynced from a machine that did --
# is not fetched again, and the organisers' hosted files cannot drift out from under a
# checked-in dataset/MANIFEST.sha256 between one job and the next. Both populated is the one
# case this cannot decide: scratch is what accumulates across jobs and may hold a training
# checkpoint, the checkout may hold the data someone just copied in, and picking either would
# throw away the other. It stops instead, while stopping is still cheap.
# submit_csf3.sh carries the same function; keep the two the same.
_scratch_link() {
  local path="$1" target="$2"
  [ -L "$path" ] && return 0
  local here="" there=""
  [ -d "$path" ] && here=$(ls -A "$path" 2>/dev/null | head -1)
  [ -d "$target" ] && there=$(ls -A "$target" 2>/dev/null | head -1)
  if [ -n "$here" ] && [ -n "$there" ]; then
    echo "ERROR: $path and $target both have content, and only one can be kept." >&2
    echo "       Scratch is what carries over between jobs; the checkout is what a copy" >&2
    echo "       lands in. Delete whichever is stale and resubmit." >&2
    exit 1
  fi
  if [ -n "$here" ]; then
    echo "  $path already has content; moving it to $target rather than fetching it again"
    rm -rf "$target"; mv "$path" "$target"
  else
    rm -rf "$path"; mkdir -p "$target"
  fi
  ln -s "$target" "$path"
}
_scratch_link dataset/raw "$SCRATCH/raw"
_scratch_link runs "$SCRATCH/runs"
echo "scratch: $SCRATCH"; df -h "$SCRATCH" | tail -1

# ---------------------------------------------------------------- environment
set -e
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.9}   # A100, L40S (see submit_csf3.sh)
make venv CUDA=12 2>&1 | tee logs/csf3_gn_venv.log
make toolchain      2>&1 | tee logs/csf3_gn_toolchain.log
make check          2>&1 | tee logs/csf3_gn_check.log
if ! .venv/bin/python -c "import nvdiffrast.torch" >/dev/null 2>&1; then
  echo "ERROR: nvdiffrast did not build/import; see logs/csf3_gn_toolchain.log. Aborting" >&2
  echo "       rather than silently falling back to the far slower software rasteriser." >&2
  exit 1
fi
set +e

# ---------------------------------------------------------------- data
make data 2>&1 | tee logs/csf3_gn_data.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "ERROR: make data failed; see logs/csf3_gn_data.log. Nothing downstream can run" >&2
  echo "       without dataset/raw. Stopping." >&2
  exit 1
fi

# ---------------------------------------------------------------- the actual run
# scripts/run_nonconvex.sh handles calibration itself (skips it when an already-fitted,
# still-loadable instrument exists), then corrects the calibrating model and every scored
# model with each solver, then selects between them and checks the submission. Its own
# docstring lists every variable it reads; nothing is duplicated here. Its exit status is the
# submission check's, so a tree holding a file an evaluator would read inside out fails this
# job rather than passing quietly.
PY=.venv/bin/python ./scripts/run_nonconvex.sh 2>&1 | tee logs/csf3_gn_run.log
run_status=${PIPESTATUS[0]}

INSTRUMENT_PATH=$(.venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); \
from calibrate import OUT_INSTRUMENT; print(OUT_INSTRUMENT['${CHANNEL:-blender}'])")
echo "=== done (run_nonconvex.sh exited $run_status); on this machine:"
echo "    ${GN_DIR:-results/gn}/ and ${MAP_DIR:-results/map}/   the two solvers' bodies"
echo "    results/submission/                the chosen ones, with selection.json saying"
echo "                                       which solver each came from and why"
echo "Pull them to your laptop with (run FROM the laptop):"
echo "  rsync -avz \$USER@csf3.itservices.manchester.ac.uk:~/hac26/results/ \\"
echo "      /Users/user/Desktop/hac-2026/results/"
echo "If this run calibrated an instrument (run_nonconvex.sh skips it when an already-fitted"
echo "one already loads), it is at $INSTRUMENT_PATH -- pull and commit that too."
exit "$run_status"
