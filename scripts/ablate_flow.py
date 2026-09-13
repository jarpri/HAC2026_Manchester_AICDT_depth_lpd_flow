#!/usr/bin/env python3
"""Does the trained flow use the curves, or has it memorised the corpus?

The flow sees x_t = (1 - t) x0 + t x1, which reveals x1 more and more as t grows, and with a
small corpus a network can recognise which body it is near without consulting a curve. This
script evaluates the trained flow twice on the same draws: the full velocity, prior plus
data part, and the prior's velocity alone, which reads no curve. It does that on held-out
bodies by default, then repeats the full arm with shuffled curves and with no curves. If the
real curves help only on training bodies, the network has memorised the corpus; if shuffled or
masked curves help as much as the real curves, the data branch is not using the measurement.
The flow and occupancy terms are compared; the data-fit term is left out of both arms.

Both arms come from one call of train_lpd.flow_loss, off one operator call, so they differ by
the data part and nothing else; this script only supplies the draws.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.conventions import cameras, psi_grid          # noqa: E402
from hac26.field import N_DIR                            # noqa: E402
from hac26.solvers.lpd_flow import N_MODES, LPDFlow, geometry_tags   # noqa: E402
from hac26.solvers.operator import CodeOperator          # noqa: E402
from train_lpd import (CALIBRATION, CORPUS, FIT_FROM, OCC_WEIGHT, Corpus,   # noqa: E402
                       add_render_flags, check_flow_metadata, file_digest, flow_loss,
                       held_out, load_corpus, load_flow_file, load_instrument,
                       model_error_scale, noise_sigma, probe_centres, probe_field,
                       render_from, render_tag, smooth_noise_like)


def carving(corpus: Corpus) -> torch.Tensor:
    """Fraction of the probes inside each body's hull that its fitted non-convex code
    removes."""
    g = corpus.codes[:, N_DIR:]
    o = probe_centres(corpus.support_true)
    body = (probe_field(corpus.support_true, g, o) < 0).float().sum((1, 2))
    hull = (probe_field(corpus.support_true, torch.zeros_like(g), o) < 0).float().sum((1, 2))
    return (1.0 - body / hull.clamp_min(1.0)).clamp_min(0.0)


def carve_bin(x: float) -> str:
    if x < 0.05:
        return "low"
    if x < 0.15:
        return "medium"
    return "high"


def with_curves(corpus: Corpus, curves: torch.Tensor,
                turned_counts: torch.Tensor | None = None) -> Corpus:
    return Corpus(corpus.codes, curves, corpus.turned_counts if turned_counts is None else turned_counts,
                  corpus.support, corpus.support_true, corpus.radius, corpus.index)


def shuffled_curves(corpus: Corpus, pool: torch.Tensor, generator: torch.Generator) -> Corpus:
    perm = torch.arange(len(corpus.codes), device=corpus.codes.device)
    if len(pool) > 1:
        shuffled = pool[torch.randperm(len(pool), generator=generator).to(pool.device)]
        if torch.equal(shuffled, pool):
            shuffled = torch.roll(shuffled, 1)
        perm[pool] = shuffled
    return with_curves(corpus, corpus.curves[perm], corpus.turned_counts[perm])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/lpd_flow.pt")
    ap.add_argument("--corpus", default=CORPUS, help="must match the training run")
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--calibration", default=CALIBRATION,
                    help="must match the training run")
    ap.add_argument("--val-bodies", type=int, default=8,
                    help="must match the training run, so the same bodies are held out")
    ap.add_argument("--split", choices=("heldout", "train", "all"), default="heldout",
                    help="which bodies to draw from; heldout tests generalisation, train tests "
                         "memorisation")
    ap.add_argument("--occ-weight", type=float, default=OCC_WEIGHT,
                    help="must match the training run")
    ap.add_argument("--occ-eps", type=float, default=None,
                    help="must match the training run")
    ap.add_argument("--fit-from", type=float, default=FIT_FROM,
                    help="must match the training run; used for metadata consistency")
    add_render_flags(ap)
    a = ap.parse_args()
    render = render_from(a, "comparison")

    data, meta = load_corpus(a.corpus)
    if meta["calibration"] != file_digest(a.calibration):
        raise SystemExit(f"{a.corpus} was built with another calibration than {a.calibration}")
    phases, op_res = int(meta["phases"]), int(meta["operator_res"])
    M = min(N_MODES, phases // 2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inst = load_instrument(a.calibration, dev)
    eta = model_error_scale(inst)
    op = CodeOperator(inst, psi_grid(phases), res=op_res, config=render, device=dev)
    sd, flow_meta = load_flow_file(a.ckpt, map_location="cpu")
    check_flow_metadata(flow_meta, corpus=a.corpus, calibration=a.calibration,
                        phases=phases, operator_res=op_res, context=a.ckpt, render=render_tag(render))
    net = LPDFlow.from_state_dict(sd)
    net = net.to(dev).eval()
    data = data.to(dev)
    codes = data.codes
    carved = carving(data)

    C = len(cameras())
    tag = geometry_tags().to(dev)
    mask = torch.ones(1, C, device=dev)

    # Split off by the same rule train_lpd.py uses (held_out, by codes-file index). Held-out
    # mode asks whether the curves help on bodies the flow did not train on.
    n_val = max(0, min(a.val_bodies, len(codes) - 1))
    is_val = np.isin(data.index.cpu().numpy(), held_out(int(meta["bodies"]), n_val))
    if a.split == "heldout":
        pool = torch.nonzero(torch.as_tensor(is_val)).flatten()
    elif a.split == "train":
        pool = torch.nonzero(torch.as_tensor(~is_val)).flatten()
    else:
        pool = torch.arange(len(codes))
    pool = pool.to(dev)
    if len(pool) == 0:
        raise SystemExit(f"no bodies available for split {a.split!r}; check --val-bodies")
    print(f"  split {a.split}: {len(pool)} bodies, carved "
          f"{float(carved[pool].min()):.2f}-{float(carved[pool].max()):.2f}", flush=True)

    # t is stratified over the bins and continuous within each, as in training. The bins are
    # only for reporting.
    n_bins = 6
    gen = torch.Generator().manual_seed(0)
    idx = pool[torch.randint(0, len(pool), (a.draws,), generator=gen)].to(dev)
    x0 = torch.randn(a.draws, codes.shape[1], dtype=codes.dtype, generator=gen).to(dev)
    t = (((torch.arange(a.draws) % n_bins).to(codes.dtype)
          + torch.rand(a.draws, generator=gen).to(codes.dtype)) / n_bins).to(dev)
    sigma = noise_sigma(a.draws, generator=gen).to(dev)
    xi = torch.randn(a.draws, C, 2, phases, generator=gen).to(dev)
    zeta = smooth_noise_like(data.curves[idx].cpu(), generator=gen).to(dev)

    arms = [("real curves", data, mask, zeta)]
    if len(pool) > 1:
        shuf = shuffled_curves(data, pool, torch.Generator().manual_seed(17))
        arms.append(("shuffled curves", shuf, mask,
                     smooth_noise_like(shuf.curves[idx].cpu(), generator=gen).to(dev)))
    arms.append(("no curves", data, torch.zeros_like(mask), zeta))

    results = {name: {"full": [], "prior": [], "dropped": 0,
                      "per_t": {k: ([], []) for k in range(n_bins)},
                      "per_carve": {k: ([], []) for k in ("low", "medium", "high")}}
               for name, _, _, _ in arms}
    for d in range(a.draws):
        sl = slice(d, d + 1)
        kb = d % n_bins
        cb = carve_bin(float(carved[idx[d]]))
        line = [f"  draw {d:>3}  t={float(t[d]):.3f}  carved {float(carved[idx[d]]):.2f}"]
        for name, corp, arm_mask, arm_zeta in arms:
            with torch.no_grad():
                r_, z_, bad = flow_loss(net, op, corp, eta, idx[sl], x0[sl], t[sl], M, tag,
                                        arm_mask, sigma=sigma[sl], xi=xi[sl],
                                        zeta=arm_zeta[sl], ablate=True,
                                        occ_weight=a.occ_weight, occ_eps=a.occ_eps,
                                        fit_from=a.fit_from)
            r_, z_ = float(r_), float(z_)
            row = results[name]
            row["dropped"] += int(bad)
            row["full"].append(r_); row["prior"].append(z_)
            row["per_t"][kb][0].append(r_); row["per_t"][kb][1].append(z_)
            row["per_carve"][cb][0].append(r_); row["per_carve"][cb][1].append(z_)
            line.append(f"{name}: {z_ - r_:+.5f}")
        print("   ".join(line), flush=True)

    for name, row in results.items():
        r_, z_ = float(np.mean(row["full"])), float(np.mean(row["prior"]))
        print(f"\n  {name}:")
        print(f"    prior plus data part : {r_:.5f}")
        print(f"    prior only           : {z_:.5f}")
        print(f"    data-branch margin   : {z_ - r_:+.5f}  "
              f"({100*(z_-r_)/max(z_,1e-12):+.1f}%)")
        print("    per t bin:")
        for k in range(n_bins):
            rr, zz = row["per_t"][k]
            if rr:
                print(f"      t={(k+0.5)/n_bins:.3f}   full {np.mean(rr):.5f}   "
                      f"prior only {np.mean(zz):.5f}   margin {np.mean(zz)-np.mean(rr):+.5f}")
        print("    per carving bin:")
        for bname in ("low", "medium", "high"):
            rr, zz = row["per_carve"][bname]
            if rr:
                print(f"      {bname:<6} n={len(rr):>3}   full {np.mean(rr):.5f}   "
                      f"prior only {np.mean(zz):.5f}   margin {np.mean(zz)-np.mean(rr):+.5f}")
        if row["dropped"]:
            print(f"    WARNING: {row['dropped']}/{a.draws} draws had no curves "
                  f"(degenerate mesh or unusable patches), pulling the margin toward zero.")
    print("\n  Real held-out curves should beat prior-only by more than shuffled or no curves,"
          "\n  especially in the high-carve bin. A margin near zero means the curves are"
          "\n  decoration; a shuffled/no-curve margin as large as the real margin means the"
          "\n  data branch is not using the measurement.")


if __name__ == "__main__":
    main()
