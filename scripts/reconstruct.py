#!/usr/bin/env python3
"""Reconstruct a challenge model with the convex LPD, from one checkpoint or a
Minkowski-average ensemble, and write the posed STL.

Uses the decode path of eval_exact.py, so what is written is what that script scores.
The curves come from the Blender channel when it is released and from the laboratory
channel otherwise (hac26.data_io.load_inversion_curves); --channel forces one of them.

    python scripts/reconstruct.py --ckpt models/lpd_convex.pt --fit-cylinder \
        --model 4 --out results/submission/Asteroid04.stl
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_exact import decode, ensemble_body, predict_h  # noqa: E402
from hac26.conventions import CYLINDER_R, PUBLIC_MODELS  # noqa: E402
from hac26.data_io import CHANNELS, load_inversion_curves  # noqa: E402
from hac26.recon import save_submission_stl  # noqa: E402
from hac26.train import load_net  # noqa: E402

SUBMISSION_DIR = "results/submission"   # the convex answers of the scored models
PUBLIC_DIR = "results/public"           # the same recipe on the public models


def answer_path(model: int) -> Path:
    """Where the convex stage's answer for a model is written by scripts/make_submission.py
    and read by every stage that starts from it."""
    d = PUBLIC_DIR if model in PUBLIC_MODELS else SUBMISSION_DIR
    return Path(d) / f"Asteroid{model:02d}.stl"


def load_checkpoints(paths) -> list:
    """(net, preset, grid) of every checkpoint, on the CPU."""
    return [load_net(c, device="cpu") for c in paths]


def reconstruct_convex(model: int, data_dir: str, loaded: list, channel: str = "auto",
                       ensemble: bool = False, smooth: int = 0,
                       fit_cylinder: bool = True) -> tuple:
    """The convex stage's body for one model: (verts, faces, info). `loaded` is the list
    `load_checkpoints` returns; with several entries and `ensemble` the bodies are
    Minkowski-averaged, otherwise the first checkpoint answers."""
    pr = loaded[0][1]
    data = load_inversion_curves(data_dir, model, m=pr.m, channel=channel)
    if not data["files"]:
        raise FileNotFoundError(f"no lightcurve files for model {model} under {data_dir}")
    R = CYLINDER_R.get(model)
    h_list = [predict_h(n, g, data["curves"], data["mask"], R) for n, _, g in loaded]
    grids = [g for _, _, g in loaded]
    if ensemble and len(loaded) > 1:
        v, f = ensemble_body(h_list, grids, R, smooth, fit_cylinder)
    else:
        v, f = decode(h_list[0], grids[0], smooth, R, fit_cylinder)
    info = {"model": model, "channel": data["channel"], "files": data["files"],
            "n_ckpt": len(loaded), "ensemble": bool(ensemble and len(loaded) > 1),
            "smooth": smooth, "fit_cylinder": bool(fit_cylinder)}
    return v, f, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="checkpoint files")
    ap.add_argument("--model", type=int, required=True, help="challenge model number")
    ap.add_argument("--out", required=True, help="output STL path")
    ap.add_argument("--data-dir", default="dataset/raw", help="challenge data directory")
    ap.add_argument("--channel", choices=CHANNELS, default="auto",
                    help="which released curves to invert: the Blender render when it is "
                         "present and the laboratory curves otherwise (auto), or one of "
                         "them by name")
    ap.add_argument("--ensemble", action="store_true",
                    help="Minkowski-average all checkpoints instead of using the first")
    ap.add_argument("--smooth", type=int, default=0,
                    help="half-width of the support smoothing window; 0 disables it")
    ap.add_argument("--fit-cylinder", action="store_true",
                    help="scale xy to the published bounding-cylinder radius")
    args = ap.parse_args()

    torch.set_num_threads(4)
    loaded = load_checkpoints(args.ckpt)
    v, f, info = reconstruct_convex(args.model, args.data_dir, loaded, args.channel,
                                    args.ensemble, args.smooth, args.fit_cylinder)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    info.update(save_submission_stl(args.out, v, f, cylinder_radius=CYLINDER_R.get(args.model)))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
