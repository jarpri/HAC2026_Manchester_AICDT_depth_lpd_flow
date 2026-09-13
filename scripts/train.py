#!/usr/bin/env python3
"""Train the convex LPD (hac26.train). Command-line options override preset fields.

    python scripts/train.py --preset smoke
    python scripts/train.py --preset gpu --out checkpoints
    python scripts/train.py --preset gpu --steps 20000 --resume checkpoints/lpd_gpu_step2000.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.train import PRESETS, Preset, auto_device, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="gpu", choices=sorted(PRESETS),
                    help="named preset from hac26.train.PRESETS")
    ap.add_argument("--steps", type=int, default=None, help="number of training steps")
    ap.add_argument("--batch", type=int, default=None, help="batch size")
    ap.add_argument("--out", default="checkpoints", help="checkpoint directory")
    ap.add_argument("--device", default=None, help="torch device; default: auto-detect")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument("--noise-profile-mode", choices=["measured", "flat"],
                    default=None, help="measured = the per-curve profile from hac26.noise; "
                    "flat = the same noise level on every curve")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from, with its optimiser state")
    ap.add_argument("--warm-start", default=None,
                    help="copy the weights of an existing, possibly ungated, checkpoint "
                         "before training (see hac26.train.warm_start)")
    ap.add_argument("--lr", type=float, default=None,
                    help="peak learning rate; use a smaller one than the preset's when "
                         "warm-starting")
    ap.add_argument("--data", default=None,
                    help="root of a mesh dataset with optional stored curves "
                         "(hac26.adapter); omit to train on synthetic shapes only")
    ap.add_argument("--mix", type=float, default=0.25,
                    help="fraction of synthetic shapes mixed into a --data run")
    ap.add_argument("--support", action="store_true",
                    help="add the support-function head and train h(u)")
    ap.add_argument("--canonical-r", action="store_true",
                    help="train on the shape scaled to xy radius 1; pair with "
                         "reconstruct.py --fit-cylinder")
    ap.add_argument("--r-cond", action="store_true",
                    help="give the a-priori bounding radius to the network as an input "
                         "(the published cylinder radius at test time)")
    ap.add_argument("--egi-weight", type=float, default=None,
                    help="weight on the EGI loss")
    ap.add_argument("--dice-weight", type=float, default=None,
                    help="weight on the Dice loss of hac26.radial; needs --support")
    ap.add_argument("--h-mse-weight", type=float, default=None,
                    help="weight on the support MSE; keep it non-zero with --dice-weight, "
                         "since Dice cannot fix the scale")
    ap.add_argument("--gate-rank", type=int, default=None,
                    help="rank of the occlusion gate; 0 disables it")
    ap.add_argument("--p-flat", type=float, default=None,
                    help="fraction of flat-faced training bodies (platonic solids, prisms, "
                         "faceted and bilobed shapes)")
    ap.add_argument("--n-rays", type=int, default=None,
                    help="number of sphere directions for the Dice loss")
    ap.add_argument("--workers", type=int, default=None,
                    help="number of DataLoader workers")
    args = ap.parse_args()
    pr: Preset = PRESETS[args.preset]
    if args.workers is not None:
        pr.num_workers = args.workers
    if args.support:
        pr.support_head = True
    if args.canonical_r:
        pr.canonical_r = True
    if args.r_cond:
        pr.r_cond = True
    if args.egi_weight is not None:
        pr.egi_weight = args.egi_weight
    if args.dice_weight is not None:
        pr.dice_weight = args.dice_weight
    if args.h_mse_weight is not None:
        pr.h_mse_weight = args.h_mse_weight
    if args.p_flat is not None:
        pr.p_flat = args.p_flat
    if args.gate_rank is not None:
        pr.gate_rank = args.gate_rank
    if args.n_rays is not None:
        pr.n_rays = args.n_rays
    if args.steps is not None:
        pr.steps = args.steps
    if args.batch is not None:
        pr.batch = args.batch
    if args.seed is not None:
        pr.seed = args.seed
    if args.noise_profile_mode is not None:
        pr.noise_profile_mode = args.noise_profile_mode
    if args.lr is not None:
        pr.lr = args.lr
    dev = args.device or auto_device()
    dataset = None
    if args.data:
        from hac26.adapter import FigurineCurves, load_pairs
        from hac26.forward.convex_egi import LEGACY, stack_A
        from hac26.geometry import build_cameras, make_grid
        grid = make_grid(pr.n_theta, pr.n_phi)
        A, _ = stack_A(grid, build_cameras(), pr.m,
                       law=getattr(pr, "photometry", LEGACY),
                       c_lambert=pr.c_lambert, sigma=pr.sigma, delta=pr.delta)
        from hac26.radial import fibonacci_sphere
        pairs = load_pairs(args.data, grid, A, eps=pr.eps_norm,
                           canonical_r=pr.canonical_r,
                           rays=fibonacci_sphere(pr.n_rays) if pr.dice_weight else None)
        print(f"team dataset: {len(pairs)} mesh/curve pairs from {args.data}")
        dataset = FigurineCurves(pairs, pr, mix_synthetic=args.mix, grid=grid, A=A)
    print(f"preset={pr.name} device={dev} steps={pr.steps} batch={pr.batch} "
          f"grid={pr.n_theta}x{pr.n_phi} I={pr.n_iter} ch={pr.ch}")
    final = train(pr, out_dir=args.out, device=dev, resume=args.resume, dataset=dataset,
                  warm_start_from=args.warm_start)
    print(f"final checkpoint: {final}")


if __name__ == "__main__":
    main()
