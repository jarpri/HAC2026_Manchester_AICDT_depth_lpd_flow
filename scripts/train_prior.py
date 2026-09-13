#!/usr/bin/env python3
"""Train the prior part of the flow: an unconditional flow over the corpus codes.

This is the part of the velocity that knows what bodies look like and reads no data, only
the body's published radius (see hac26.solvers.lpd_flow). It needs no operator, so it
trains in minutes at a large batch for
many passes over the corpus, which the operator-bound training of the data part
(scripts/train_lpd.py) cannot afford. The objective is the same as there without the data:
for x0 ~ N(0, I), a corpus body x1 (its dh block the correction from its convex start to its
true hull, scripts/build_corpus.py) and t ~ U[0, 1), the velocity is scored against x1 - x0
and the endpoint it implies, x_t + (1 - t) v, by its occupancy against the body's
(train_lpd.step_loss). The bodies are turned by random quarter turns as there
(train_lpd.quarter_turns). The network is conditioned on the convex start, as the data part
is.

The codec is fitted here from the codes and saved with the prior; train_lpd.py takes both
from this file, so the prior and the data part share one whitened code.

The same bodies are held out as in train_lpd.py (train_lpd.held_out), scored every
--val-every steps at fixed draws for early stopping; the saved weights are the best-scoring
ones, averaged over recent steps as in train_lpd.py.

Checkpoints every --ckpt-every steps to --ckpt-file (default <--out>.ckpt) and resumes from it
by default, carrying the optimiser, the averaged weights, the best state and the early-stopping
counters, so a resumed run stops at the step an uninterrupted one would have. A run that is too
slow to finish can be finalised from where it reached rather than abandoned: rerun it with
--steps at or below the checkpointed step and it writes the product from the best state it has
without training further.

Writes --out (default runs/prior_flow.pt): the prior's weights, the codec, and metadata.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.solvers.lpd_flow import LPDFlow                                # noqa: E402
from train_lpd import (CORPUS, OCC_WEIGHT, PRIOR, Corpus, EMA, _Swapped, _enable_tf32,   # noqa: E402
                       _hms, _now, cond_channels, file_digest, held_out, load_corpus,
                       occ_eps_default, quarter_turns, step_loss)


def prior_loss(net, corpus: Corpus, idx, x0, t, occ_weight, occ_eps, turns=None):
    """The step loss of the prior part alone for one batch of bodies `idx`, given the draws
    (x0, t) and, optionally, the quarter turns (B,) to apply to the bodies. Returns (total,
    flow term, occupancy term)."""
    if turns is None:
        codes, sup, sup_true = corpus.codes[idx], corpus.support[idx], corpus.support_true[idx]
    else:
        codes, _, sup, sup_true = quarter_turns(corpus, idx, turns)
    x1 = net.codec.encode(codes)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    sph, node = cond_channels(sup, device=codes.device)
    v = net.prior_velocity(xt, t, corpus.radius[idx], sph, node)
    total, flow, occ, _ = step_loss(net, xt, t, v, x1, sup_true, codes, sup, occ_weight, occ_eps)
    return total, flow, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--corpus", default=CORPUS, help="written by scripts/build_corpus.py")
    ap.add_argument("--out", default=PRIOR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-bodies", type=int, default=8,
                    help="bodies held out; must match train_lpd.py so the split agrees")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=10,
                    help="evaluations without improvement before stopping; 0 never stops")
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--occ-weight", type=float, default=OCC_WEIGHT,
                    help="must match train_lpd.py")
    ap.add_argument("--occ-eps", type=float, default=None, help="must match train_lpd.py")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500,
                    help="steps between checkpoints; 0 writes none, which loses the run to a "
                         "wallclock")
    ap.add_argument("--ckpt-file", default=None,
                    help="resumable checkpoint; by default <--out>.ckpt")
    ap.add_argument("--no-resume", action="store_true",
                    help="start from step 0 even when a checkpoint for these settings is there")
    a = ap.parse_args()
    ckpt_path = Path(a.ckpt_file or f"{a.out}.ckpt")
    _enable_tf32()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data, cmeta = load_corpus(a.corpus)
    data = data.to(dev)
    codes = data.codes
    n_val = max(0, min(a.val_bodies, len(codes) - 1))
    is_val = torch.as_tensor(np.isin(data.index.cpu().numpy(),
                                     held_out(int(cmeta["bodies"]), n_val)))
    val_idx = torch.nonzero(is_val).flatten().to(dev)
    train_idx = torch.nonzero(~is_val).flatten().to(dev)
    occ_eps = a.occ_eps if a.occ_eps is not None else occ_eps_default()

    net = LPDFlow().to(dev)
    net.codec.fit(codes)
    print(f"  codec: g scale {float(net.codec.g_s):.5f}, dh centre {float(net.codec.mu[0]):.5f} "
          f"sd {float(net.codec.sd[0]):.5f}, g sd {float(net.codec.sd[1]):.5f}", flush=True)
    phases = int(cmeta["phases"])
    augment = phases % 4 == 0
    print(f"  {len(train_idx)} training bodies, {len(val_idx)} held out; batch {a.batch}, "
          f"up to {a.steps} steps; "
          + ("quarter turns on" if augment else
             f"NOTE: {phases} phases is not divisible by 4, so the bodies are not turned"),
          flush=True)
    opt = torch.optim.Adam(net.prior.parameters(), lr=a.lr)
    ema_decay = min(a.ema, 1.0 - 1.0 / max(a.steps / 10.0, 10.0)) if a.ema else 0.0
    ema = EMA(net, decay=ema_decay)

    gen = torch.Generator().manual_seed(1234)
    n_v = len(val_idx)
    val_x0 = torch.randn(n_v, codes.shape[1], generator=gen).to(dev)
    val_t = ((torch.arange(n_v, dtype=torch.float32) + 0.5) / max(n_v, 1)).to(dev)

    def validate():
        was_training = net.training
        net.eval()
        with torch.no_grad(), _Swapped(net, ema):
            l, f, o = prior_loss(net, data, val_idx, val_x0, val_t, a.occ_weight, occ_eps)
        net.train(was_training)
        return float(l), float(f), float(o)

    best, best_state, best_step, stale = float("inf"), None, -1, 0
    start_step, elapsed_before = 0, 0.0
    # The early-stopping state is checkpointed with the weights and not only beside them. A
    # resume that restarted the patience window would train far past the step this run would
    # have stopped at, and one that lost `best` would take a worse state as its answer.
    keys = {"corpus": file_digest(a.corpus), "bodies": int(len(codes)),
            "occ_weight": float(a.occ_weight), "occ_eps": float(occ_eps),
            "batch": int(a.batch), "lr": float(a.lr), "val_bodies": int(n_val)}
    if ckpt_path.exists() and not a.no_resume:
        st = torch.load(ckpt_path, map_location=dev, weights_only=False)
        if st.get("keys") != keys:
            print(f"  [{_now()}] {ckpt_path} was written under other settings, so it is "
                  f"ignored and the run starts from step 0", flush=True)
        else:
            net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"])
            ema.shadow = {k: v.to(dev) for k, v in st["ema"].items()}
            ema.n = int(st["ema_n"])
            best, best_step, stale = float(st["best"]), int(st["best_step"]), int(st["stale"])
            best_state = st["best_state"]
            start_step, elapsed_before = int(st["step"]) + 1, float(st["elapsed"])
            torch.set_rng_state(st["rng"].cpu())
            print(f"  [{_now()}] resumed {ckpt_path} at step {start_step} "
                  f"(best {best:.5f} from step {best_step}, {stale}/{a.patience} stale)",
                  flush=True)

    def save_ckpt(step):
        """Written under a temporary name and renamed, so a kill mid-write leaves the previous
        checkpoint rather than a truncated one."""
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_name(ckpt_path.name + ".part")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "ema": ema.shadow,
                    "ema_n": ema.n, "best": best, "best_state": best_state,
                    "best_step": best_step, "stale": stale, "step": step,
                    "elapsed": elapsed_before + (time.time() - t0), "rng": torch.get_rng_state(),
                    "keys": keys}, tmp)
        tmp.replace(ckpt_path)

    t0 = time.time()
    s = start_step - 1
    for s in range(start_step, a.steps):
        idx = train_idx[torch.randint(0, len(train_idx), (a.batch,), device=dev)]
        x0 = torch.randn(a.batch, codes.shape[1], device=dev)
        t = ((torch.arange(a.batch, dtype=torch.float32) + torch.rand(a.batch)) / a.batch)
        t = t[torch.randperm(a.batch)].to(dev)
        turns = torch.randint(0, 4, (a.batch,), device=dev) if augment else None
        loss, flow, occ = prior_loss(net, data, idx, x0, t, a.occ_weight, occ_eps, turns)
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(net)
        if a.log_every and (s % a.log_every == 0 or s == a.steps - 1):
            # elapsed counts the whole run and not this process, so a resumed job reports
            # the training's age rather than its own, which is what a rate is read from
            print(f"  [{_now()}] step {s:>6}  loss {float(loss):.5f}  (flow {float(flow):.5f}, "
                  f"occupancy {float(occ):.5f})  elapsed "
                  f"{_hms(elapsed_before + (time.time() - t0))}", flush=True)
        if n_v and ((s + 1) % a.val_every == 0 or s == a.steps - 1):
            vl, vf, vo = validate()
            if vl < best - a.min_delta:
                best, best_step, stale = vl, s, 0
                best_state = ema.state(net)
                print(f"  [{_now()}] step {s:>6}  val {vl:.5f}  (flow {vf:.5f}, "
                      f"occupancy {vo:.5f})  best", flush=True)
            else:
                stale += 1
                print(f"  [{_now()}] step {s:>6}  val {vl:.5f}  (no improvement on {best:.5f} "
                      f"from step {best_step}, {stale}/{a.patience})", flush=True)
                if a.patience and stale >= a.patience:
                    print(f"  early stop at step {s}", flush=True)
                    save_ckpt(s)
                    break
        if a.ckpt_every and (s + 1) % a.ckpt_every == 0:
            save_ckpt(s)
    steps_trained = s + 1
    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"  restored step {best_step} (val {best:.5f})", flush=True)
    else:
        net.load_state_dict(ema.state(net))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prior": net.prior.state_dict(), "codec": net.codec.state_dict(),
                "meta": {"bodies": int(len(codes)), "steps_trained": int(steps_trained),
                         "best_step": int(best_step), "val": float(best),
                         "n_val": int(n_v), "occ_weight": float(a.occ_weight),
                         "occ_eps": float(occ_eps), "corpus": file_digest(a.corpus)}},
               a.out)
    print(f"[{_now()}] wrote {a.out} after "
          f"{_hms(elapsed_before + (time.time() - t0))}", flush=True)


if __name__ == "__main__":
    main()
