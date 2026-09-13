#!/usr/bin/env python3
"""Produce the submission: the convex stage's body for every challenge model, checked and,
where a truth is released, scored with the organisers' measures.

    python scripts/make_submission.py
    python scripts/make_submission.py --models 4 5 --data-dir dataset/raw

The recipe is fixed here rather than on the command line so that the files under
results/submission are the output of one stated procedure. The convex LPD reads the curves
of the released channel hac26.data_io.load_inversion_curves picks, the Blender render when
it is present and the laboratory curves otherwise, and the body's width is set from the
published bounding-cylinder radius, the one number the mean-normalised curves do not carry.
The scored models go to results/submission, the public ones to results/public
(reconstruct.answer_path), every file passes scripts/check_submission.py's inspection before
the script exits zero, and the public files are scored against the released shapes into
results/public_scores.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import score_one                                     # noqa: E402
from check_submission import inspect                                # noqa: E402
from hac26.conventions import CYLINDER_R, PUBLIC_MODELS             # noqa: E402
from hac26.data_io import public_stl                                # noqa: E402
from hac26.recon import save_submission_stl                         # noqa: E402
from reconstruct import answer_path, load_checkpoints, reconstruct_convex   # noqa: E402

CKPT = "models/lpd_convex.pt"     # the trained convex LPD
CHANNEL = "auto"                  # the Blender render when released, else the lab curves
FIT_CYLINDER = True               # xy scaled to the published radius
SMOOTH = 0                        # no support smoothing
SCORES_FILE = "results/public_scores.json"
VOXEL_PITCH = 0.05                # the pitch of the organisers' released example
PROJECTION_DIRS = 4               # angles averaged in the projection measure


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--ckpt", default=CKPT)
    a = ap.parse_args()

    torch.set_num_threads(4)
    loaded = load_checkpoints([a.ckpt])
    report = {"checkpoint": a.ckpt, "checkpoint_sha256": sha256(a.ckpt), "channel": CHANNEL,
              "fit_cylinder": FIT_CYLINDER, "smooth": SMOOTH, "models": {}}
    failed = 0
    for M in a.models:
        out = answer_path(M)
        out.parent.mkdir(parents=True, exist_ok=True)
        v, f, info = reconstruct_convex(M, a.data_dir, loaded, CHANNEL, smooth=SMOOTH,
                                        fit_cylinder=FIT_CYLINDER)
        info.update(save_submission_stl(str(out), v, f, cylinder_radius=CYLINDER_R[M]))
        check = inspect(out, CYLINDER_R[M])
        info["check"] = check
        info["path"] = str(out)
        failed += bool(check["fails"])
        mark = "FAIL" if check["fails"] else "ok"
        print(f"model {M:>2}  {info['channel']:8s}  {check['faces']:>6} faces  "
              f"vol {check['volume']:.3f}  r_xy {check['r_xy']:.3f}  {mark}  "
              f"{'; '.join(check['fails'])}", flush=True)
        report["models"][M] = info

    public = [M for M in a.models if M in PUBLIC_MODELS
              and Path(public_stl(a.data_dir, M)).exists()]
    if public:
        print(f"\n{'model':>5} {'voxel':>7} {'proj':>7} {'SCORE':>7}")
        with tempfile.TemporaryDirectory() as td:
            for M in public:
                r = score_one(public_stl(a.data_dir, M), answer_path(M), Path(td),
                              VOXEL_PITCH, False, PROJECTION_DIRS)
                report["models"][M]["score"] = r
                print(f"{M:>5} {r['voxel']:>7.4f} {r['proj_released']:>7.4f} {r['score']:>7.4f}",
                      flush=True)
        tot = sum(report["models"][M]["score"]["score"] for M in public)
        print(f"{'SUM':>5} {tot:>23.4f}   over {len(public)} models (max {2 * len(public)})")
    if set(a.models) == set(range(1, 11)):
        Path(SCORES_FILE).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {SCORES_FILE}")
    if failed:
        raise SystemExit(f"{failed} file(s) failed the submission check")


if __name__ == "__main__":
    main()
