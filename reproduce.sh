#!/usr/bin/env bash
#
# Reproduce the submitted reconstructions.
#
#   ./reproduce.sh              what each stage costs, and what it needs
#   ./reproduce.sh verify       check and re-score the shipped bodies   (CPU, minutes)
#   ./reproduce.sh reconstruct  rebuild them from the shipped weights   (one GPU, hours)
#   ./reproduce.sh submit       make those bodies the submission        (CPU, seconds)
#   ./reproduce.sh train        rebuild the weights from the data       (one GPU, about a day)
#
# There is no cluster in any of this. Each stage is a loop over models in plain bash, so it
# runs on one machine with one GPU; if you have several, the per-model commands it prints are
# independent and can be spread across them in whatever way your site expects.
#
# On matching our numbers exactly: `verify` will, because it re-scores the files as shipped.
# `reconstruct` will come close but not to the last digit, and `train` will not. The forward
# model renders with nvdiffrast, and neither it nor the CUDA reductions under it are
# bitwise reproducible across different GPUs, so a body rebuilt on your card differs from
# ours in the third decimal of its score and by a little more after several thousand training
# steps have amplified it. That is why the trained weights are shipped: `reconstruct` starts
# after the part where the divergence accumulates.

set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
DATA_DIR=${DATA_DIR:-dataset/raw}
MODELS=${MODELS:-1 2 3 4 5 6 7 8 9 10}
PUBLIC=${PUBLIC:-1 2 3}

FLOW_CKPT=${FLOW_CKPT:-models/lpd_flow.pt}
CONVEX_CKPT=${CONVEX_CKPT:-models/lpd_convex.pt}
CALIBRATION=${CALIBRATION:-models/instrument_blender.pt}

# The settings the submitted bodies were made with. Changing them makes a different run,
# which is fine, but then it is no longer the one being reproduced.
RECON_SAMPLES=${RECON_SAMPLES:-24}     # candidate bodies drawn per model
RECON_POLISH=${RECON_POLISH:-20}       # gradient steps each candidate is refined by
RECON_STEPS=${RECON_STEPS:-16}         # sampler steps from noise to a body
RECON_RES=${RECON_RES:-96}             # extraction resolution of the written mesh
RECON_GUIDANCE=${RECON_GUIDANCE:-1.0}  # weight on the data part; 1.0 is what the flow trained at
# A draw whose mesh extracts as a body plus a few specks is otherwise refused for having more
# than one component. Above this fraction the largest piece is kept and the specks dropped;
# below it the pieces are comparable, which a bilobed body would also look like, so the draw
# is left alone. 0 disables it.
RECON_REPAIR=${RECON_REPAIR:-0.98}
FLOW_PHASES=${FLOW_PHASES:-96}
FLOW_OPERATOR_RES=${FLOW_OPERATOR_RES:-32}

say()  { printf '\n=== %s\n' "$*"; }
step() { printf '  %s\n' "$*"; }
die()  { printf '\n!!! %s\n' "$*" >&2; exit 1; }

need_python() {
  [ -x "$PY" ] || die "no interpreter at $PY. Run 'make venv' first, or set PY to one."
}

need_data() {
  [ -d "$DATA_DIR" ] || die "no data at $DATA_DIR. Run 'python scripts/fetch_data.py' first."
}

need_file() {
  [ -s "$1" ] || die "missing $1${2:+ ($2)}"
}

usage() {
  cat <<'TXT'
Stages, what they need, and roughly what they cost.

  verify        Checks the shipped bodies under results/submission are well formed and in
                the challenge pose, then scores the public ones against the released shapes
                with the organisers' two measures. Reads the data and the STL files; runs no
                network and needs no GPU. Minutes.

  reconstruct   Rebuilds every body from the shipped weights: the convex inversion first,
                then the flow's correction of it. Needs a GPU with nvdiffrast built (see
                "Install" in README.md) and the three checkpoints under models/. Allow about
                an hour and a half per model on an A100 -- most of it is the sampler, which
                renders the body from every camera at each of its steps.

  submit        Puts the flow's bodies into results/submission, preferring the newest run
                and leaving a convex answer in place for any model the flow could not
                answer, then checks every file. Copies nothing until all of them pass.
                Seconds, no GPU. `reconstruct` does this for you at the end.

  train         Rebuilds the weights themselves, from the shape library up. This is
                scripts/run_remote_pipeline.sh, which is resumable: every stage writes a
                marker under runs/.done and is skipped if its settings and sources still
                match, so an interrupted run can be relaunched as it stands. About a day on
                one GPU.

Environment variables worth knowing: PY (interpreter, default .venv/bin/python), DATA_DIR
(default dataset/raw), MODELS (default 1..10), and the RECON_* settings at the top of this
file, which are the ones the submission was made with.
TXT
}

stage_verify() {
  need_python; need_data
  say "checking the released data against dataset/MANIFEST.sha256"
  $PY scripts/check_data.py --data-dir "$DATA_DIR" || die "the data does not match the manifest"

  say "checking the submitted bodies are well formed and correctly posed"
  $PY scripts/check_submission.py results/submission || die "a submitted body failed its checks"

  say "scoring the public bodies against the released shapes"
  # PYTHONPATH: the two scorers are the only entry points outside scripts/, and it is the
  # scripts that put the repository root on sys.path.
  for dir in results/public results/lpd; do
    [ -d "$dir" ] || continue
    ls "$dir"/Asteroid0[123].stl >/dev/null 2>&1 || continue
    step "$dir"
    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" $PY hac26/scoring/voxel.py \
      --stl "$dir"/Asteroid0{1,2,3}.stl --models $PUBLIC --data-dir "$DATA_DIR"
    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" $PY hac26/scoring/side_view.py \
      --models $PUBLIC --recon-dir "$dir" --data-dir "$DATA_DIR" \
      --out "runs/side_view_$(basename "$dir").json"
  done
  say "done"
}

stage_reconstruct() {
  need_python; need_data
  need_file "$CONVEX_CKPT"  "the convex network"
  need_file "$FLOW_CKPT"    "the trained flow"
  need_file "$CALIBRATION"  "the instrument fitted to the Blender channel"
  mkdir -p results/convex results/lpd runs

  # The convex inversion. It is the body the flow corrects, so it comes first, and it is
  # cheap: seconds per model.
  say "convex inversion (the starting body for each model)"
  for m in $MODELS; do
    printf -v nn '%02d' "$m"
    out="results/convex/Asteroid$nn.stl"
    if [ -s "$out" ] && [ ! "$CONVEX_CKPT" -nt "$out" ]; then step "model $m: $out is current"; continue; fi
    step "model $m -> $out"
    $PY scripts/reconstruct.py --ckpt "$CONVEX_CKPT" --model "$m" \
        --data-dir "$DATA_DIR" --out "$out" || die "model $m failed the convex stage"
  done

  # The flow's correction. Each model is independent, and a model already written is left
  # alone, so this can be interrupted and restarted, or split across machines by setting
  # MODELS on each.
  say "flow reconstruction ($RECON_SAMPLES draws per model, guidance $RECON_GUIDANCE)"
  for m in $MODELS; do
    printf -v nn '%02d' "$m"
    out="results/lpd/Asteroid$nn.stl"
    if [ -s "$out" ] && [ -s "results/lpd/Asteroid$nn.json" ] && [ ! "$FLOW_CKPT" -nt "$out" ]; then
      step "model $m: $out is current"; continue
    fi
    step "model $m -> $out (about an hour and a half)"
    $PY scripts/reconstruct_lpd.py --model "$m" \
        --ckpt "$FLOW_CKPT" --calibration "$CALIBRATION" --data-dir "$DATA_DIR" \
        --samples "$RECON_SAMPLES" --polish-steps "$RECON_POLISH" --steps "$RECON_STEPS" \
        --res "$RECON_RES" --guidance "$RECON_GUIDANCE" \
        --repair-components "$RECON_REPAIR" \
        --phases "$FLOW_PHASES" --operator-res "$FLOW_OPERATOR_RES" \
        --out "$out"
    rc=$?
    # A model whose draws all come out in several pieces, or which no draw renders, is
    # reported and passed over rather than stopping the run: its convex answer stands.
    [ $rc -eq 0 ] || step "model $m produced no body (exit $rc); its convex answer stands"
  done

  say "assembling the submission from what was produced"
  $PY scripts/assemble_submission.py --from results/lpd-late results/lpd

  say "scoring what was produced"
  stage_verify
}

stage_train() {
  need_python; need_data
  say "full pipeline (scripts/run_remote_pipeline.sh)"
  step "resumable: stages already done are skipped, so this can be relaunched as it stands"
  step "stage 5c, which measures the guidance weight, is off by default -- see RUN_DECISION"
  exec scripts/run_remote_pipeline.sh "$@"
}

case "${1:-}" in
  verify)      stage_verify ;;
  submit)      need_python; $PY scripts/assemble_submission.py --from results/lpd-late results/lpd ;;
  reconstruct) stage_reconstruct ;;
  train)       shift; stage_train "$@" ;;
  ""|-h|--help|help) usage ;;
  *) die "unknown stage '${1}'. One of: verify, reconstruct, submit, train (no argument for help)." ;;
esac
