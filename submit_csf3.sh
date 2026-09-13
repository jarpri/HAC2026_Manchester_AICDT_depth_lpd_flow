#!/bin/bash --login
# hac26 full pipeline on CSF3: whatever branch is checked out, at N_BODIES=1500.
# The branch is not named here because the job runs the working tree it is submitted
# from, and a name written here goes stale the moment the work moves.
#
#   ./submit_csf3.sh                            # gpuA (A100 80GB), 4-day limit
#   CSF_PARTITION=gpuH_short ./submit_csf3.sh   # H200, 1-day limit
#   CSF_PARTITION=gpuH ./submit_csf3.sh         # H200, 4-day limit
#   CSF_PARTITION=gpuL ./submit_csf3.sh         # L40S 48GB, 4-day limit
#   ./submit_csf3.sh -d afterany:1234           # other arguments go to sbatch
#   sbatch submit_csf3.sh                       # gpuA only; ignores CSF_PARTITION
#
# CSF_TIME overrides the wallclock and CSF_ACCOUNT the H200 account code.
#
# This is the whole pipeline, which ends with the correction and the submission.
# submit_csf3_gn.sh is that last part on its own: it reads no corpus and no flow, only the
# calibration and the committed convex answers, so it needs neither the library nor any
# training and it asks for hours on a rasterisation-shaped GPU rather than days on a
# training-shaped one. That is the job to submit when a submission is wanted before the
# training finishes.
#
# =============================================================================
# Partition, modules and the torch build are all confirmed against this cluster:
# sinfo, module avail, and nvidia-smi reporting driver 595.71.05.
# =============================================================================
#SBATCH --job-name=hac26-pipeline
#SBATCH --partition=gpuA                  # 19 nodes, 4-day limit
#SBATCH -G 1
#SBATCH -n 1                              # one task; the body is a single process
#SBATCH --cpus-per-task=12                # gpuA allows <=12 cores per GPU
#SBATCH -t 4-0                            # gpuA maximum; the run resumes if it is hit
#SBATCH --output=logs/csf3_%j.out
#SBATCH --error=logs/csf3_%j.err

# ---------------------------------------------------------------- submission
# Slurm reads the #SBATCH lines before any shell runs and expands no variables in
# them, so they cannot take the partition from the environment. Run outside a job,
# this file submits itself instead, passing what the partition needs on the sbatch
# command line, which overrides the lines above. H200 differs from the default in
# three ways: it needs an account, it allows at most 8 cores per GPU, and gpuH_short
# rejects any wallclock over one day.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  set -euo pipefail
  PARTITION=${CSF_PARTITION:-gpuA}
  H200_ACCOUNT=${CSF_ACCOUNT:-gpu-h200-fse-pgdr}
  case "$PARTITION" in
    gpuA|gpuL)  ARGS=(-c 12 -t "${CSF_TIME:-4-0}") ;;
    gpuH)       ARGS=(-c 8  -t "${CSF_TIME:-4-0}" -A "$H200_ACCOUNT") ;;
    gpuH_short) ARGS=(-c 8  -t "${CSF_TIME:-1-0}" -A "$H200_ACCOUNT") ;;
    *) echo "CSF_PARTITION=$PARTITION: expected gpuA, gpuL, gpuH or gpuH_short" >&2
       exit 1 ;;
  esac
  # Slurm opens -o and -e before the job starts, and the job starts in the directory
  # it was submitted from, so both have to be settled here.
  cd "$(dirname "$0")"
  mkdir -p logs
  exec sbatch -p "$PARTITION" "${ARGS[@]}" "$@" "$(basename "$0")"
fi

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-${SGE_O_WORKDIR:-$PWD}}"
mkdir -p logs

# `CSF_PARTITION=gpuH sbatch submit_csf3.sh` lands on gpuA, because sbatch never
# runs the block above. Stop now rather than spend days on the wrong GPU.
if [ -n "${CSF_PARTITION:-}" ] && [ "$CSF_PARTITION" != "${SLURM_JOB_PARTITION:-}" ]; then
  echo "ERROR: CSF_PARTITION=$CSF_PARTITION, but this job is on ${SLURM_JOB_PARTITION:-?}." >&2
  echo "       Submit with ./submit_csf3.sh rather than sbatch to use it." >&2
  exit 1
fi

# ---------------------------------------------------------------- modules
# The Makefile installs torch itself against the driver it finds, so the module
# system only has to supply a CUDA toolkit (for the nvdiffrast build) and a
# Python >= 3.10. The proxy module is loaded in case outbound HTTP needs it,
# but it does not set http_proxy on these nodes and downloads work regardless:
# torch came from download.pytorch.org and all 28 JPL/PDS shape models arrived
# with it unset. So the line below reports the value rather than warning about it.
module purge
module load tools/env/proxy2                 # http_proxy is unset without this
module load cuda/12.6.2                      # the only CUDA module on the merged CSF
module load python/3.13.1                    # the Makefile builds its own venv from it

# Required whenever -c is used, or every library that reads the core count will
# oversubscribe against the multiprocessing pools the pipeline already runs.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "proxy: ${http_proxy:-unset (fine; direct egress works on these nodes)}"
nvidia-smi || echo "WARNING: no GPU visible; corpus, flow and reconstruct will fail"

# ---------------------------------------------------------------- scratch
# Home is quota'd and the bulky parts of a run do not fit in it: the Thingi10K
# cache reached 39G on its own and the challenge archive died mid-extract with
# ENOSPC. Everything large is put on scratch and reached through symlinks, so
# the scripts keep using their own default paths -- `make data` runs
# fetch_data.py with no arguments, so its destination cannot be set by an
# environment variable, and a symlink is the only thing that redirects it.
SCRATCH=${SCRATCH_DIR:-$HOME/scratch/hac26}
mkdir -p "$SCRATCH"/{cache,raw,generated,shape_models,runs}
export XDG_CACHE_HOME="$SCRATCH/cache"    # pip, and thingi10k via platformdirs
export UV_CACHE_DIR="$SCRATCH/cache/uv"   # named outright: uv's cache reached 39G
                                          # in home on the first run, and it is the
                                          # Makefile's installer whenever uv is on PATH
export HF_HOME="$SCRATCH/cache/huggingface"
# Only dataset/raw is symlinked, because `make data` runs fetch_data.py with no arguments
# and its destination cannot be set from the environment. Everywhere a variable will do, an
# absolute path is used instead: run_remote_pipeline.sh finds the shape models with
# `find "$SHAPE_MODELS_DIR" -maxdepth 2 -type f`, and find does not descend into a symlinked
# directory without -L, so pointing that variable at a symlink silently yields no models and
# the library is built with the "real" family empty.
mkdir -p dataset
# The path the scripts use, and where the bytes should live. A path that is already a
# symlink is left alone. A populated path with empty scratch is moved rather than deleted, so
# a dataset that has been fetched and validated once -- or rsynced from a machine that did --
# is not fetched again, and the organisers' hosted files cannot drift out from under a
# checked-in dataset/MANIFEST.sha256 between one job and the next. Both populated is the one
# case this cannot decide: scratch is what accumulates across jobs and may hold a training
# checkpoint, the checkout may hold the data someone just copied in, and picking either would
# throw away the other. It stops instead, while stopping is still cheap.
# submit_csf3_gn.sh carries the same function; keep the two the same.
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
export SHAPE_MODELS_DIR="$SCRATCH/shape_models"
export LIB_DIR="$SCRATCH/generated/shapes"
mkdir -p "$SHAPE_MODELS_DIR" "$LIB_DIR"
echo "scratch: $SCRATCH"; df -h "$SCRATCH" | tail -1

# ---------------------------------------------------------------- sizing
# build_corpus.py renders every body once with the exact forward model on the
# GPU and runs the convex stage on it, so wall time is linear in N_BODIES. That
# stage, not the library build, is what sets the runtime.
export PY_VERSION=3.13                # match the module; the Makefile default is 3.12
export N_BODIES=${N_BODIES:-1500}
export LIB_RES=${LIB_RES:-64}
export LIB_WORKERS=${LIB_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
export LIB_SEED=${LIB_SEED:-0}

# The "object" family (Thingi10K everyday objects) and the "real" family (JPL
# and PDS asteroid models) are 12% and 14% of the intended family weights, and
# LibrarySpec.weights() silently drops both while their directories are empty.
export FETCH_OBJECTS=1
export N_OBJECTS=${N_OBJECTS:-600}
export FETCH_MODELS=1

# ---------------------------------------------------------------- environment
# Built here rather than left to _venv_setup.sh inside the pipeline, so a build
# failure surfaces in its own log instead of a hundred lines into stage one.
set -e
# CUDA=12, not auto. Auto reads the DRIVER, and 595.71.05 is new enough that it would
# choose a CUDA 13 torch (cu130) -- but cuda/12.6.2 is the only toolkit module here, and
# nvdiffrast is compiled with that toolkit against torch's headers. cu129 keeps the two
# on the same CUDA major version. The venv records this, so later `make` calls keep it.
#
# nvdiffrast is compiled for the GPUs listed in TORCH_CUDA_ARCH_LIST. Unset, torch builds
# for whichever card the build job landed on, and the build is shared by every later job:
# one made on an H200 (sm_90) fails on an A100 with "no kernel image is available". So all
# three CSF3 cards are named: A100 8.0, L40S 8.9, H200 9.0, plus PTX for anything newer.
# setup_toolchain.sh records the list and rebuilds when it changes.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0;8.9;9.0+PTX}
make venv CUDA=12 2>&1 | tee logs/csf3_venv.log
make toolchain      2>&1 | tee logs/csf3_toolchain.log
make check          2>&1 | tee logs/csf3_check.log
set +e

# ---------------------------------------------------------------- data
# Several GB from the organisers' Dropbox. Downloads only when dataset/raw is
# empty; if it is already populated and matches dataset/MANIFEST.sha256 this is
# a no-op. No calibration is committed -- models/ holds only the convex stage --
# so the curves are what the calibration is fitted against, and every stage from
# there on needs the instrument it writes. A failure here therefore costs the
# run and not only its last stages, which is why the pipeline stops at the
# calibration rather than going on without one.
make data 2>&1 | tee logs/csf3_data.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "ERROR: data fetch failed; see logs/csf3_data.log. The calibration, and so every" >&2
  echo "       stage after it, needs dataset/raw. Stopping." >&2
  exit 1
fi

# ---------------------------------------------------------------- run
# Each stage writes a marker under runs/.done/ and is skipped if already
# complete, so a job that hits the wall clock can be resubmitted as is. The
# non-convex track keeps its state per body instead, so it too carries on
# where a killed job stopped.
./scripts/run_remote_pipeline.sh
