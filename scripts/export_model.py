#!/usr/bin/env python3
"""Export a training checkpoint as a slim file for version control.

The buffers in hac26.train.REGENERABLE_BUFFERS (`op.A`, `tags`, `coords`) are functions
of the preset and are rebuilt by build_model on load, so they are dropped; `op.A` is most
of a training checkpoint's size. The optimiser state is dropped too. The slim file is
reloaded once to check it.

    python scripts/export_model.py --ckpt <training-ckpt> --out models/lpd_convex.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hac26.train import REGENERABLE_BUFFERS, load_net  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_dice2/lpd_gpu_final.pt",
                    help="training checkpoint to export")
    ap.add_argument("--out", default="models/lpd_convex.pt", help="slim output file")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    slim = {k: v for k, v in ck["model"].items() if k not in REGENERABLE_BUFFERS}
    dropped = sorted(set(ck["model"]) - set(slim))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": slim, "preset": ck["preset"], "step": ck.get("step")}, out)

    before = Path(args.ckpt).stat().st_size / 1e6
    after = out.stat().st_size / 1e6
    print(f"dropped (regenerated from the preset): {dropped}")
    print(f"{before:.1f} MB -> {after:.1f} MB")

    # check that the slim file loads
    net, pr, grid = load_net(str(out), device="cpu")
    n = sum(p.numel() for p in net.parameters())
    print(f"reloaded OK: {n} parameters, grid {pr.n_theta}x{pr.n_phi}, "
          f"support_head={net.support_head}, r_cond={net.r_cond}")


if __name__ == "__main__":
    main()
