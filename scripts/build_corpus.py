#!/usr/bin/env python3
"""Build the flow's training corpus: each fitted body's curves and the convex start the flow
has to correct.

The flow starts from the convex stage's reconstruction and learns the correction to it, so
the corpus has to hold the corrections that stage needs on bodies like these. For each body of
the codes file (scripts/fit_shapes.py) it records the radius the body was mounted with, the
width over half-height that the challenge publishes as a model's cylinder radius (a library
without that record gets one drawn from the published range); the exact forward model's
noise-free curves at that radius, with the count curves the body would have under the
thresholds of the other three quarter frames, so that training can turn it about its spin
axis exactly (train_lpd.quarter_turns); and the support the convex stage reconstructs from
a noisy realisation of those curves, made as at reconstruction (scripts/reconstruct.py: the
same checkpoint, decode and per-curve normalisation) with noise and model error of the
sizes the flow trains with. The code's dh block is the band-limited correction from that
support to the body's hull.

The convex start comes from one noisy realisation while training draws fresh noise every
step (train_lpd.flow_loss); the convex stage's error comes from what it cannot represent,
not from the noise, so the two need not agree.

The operator renders on the GPU and the convex stage runs on the CPU in --workers processes
alongside it. Resumable body by body under <out>.parts/. Writes --out (default
runs/corpus.npz), which train_prior.py, train_lpd.py, ablate_flow.py and decision_check.py
read through train_lpd.load_corpus.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_exact import decode, predict_h                                          # noqa: E402
from hac26.conventions import CYLINDER_R, cameras, psi_grid                       # noqa: E402
from hac26.data_io import resample_curves                                         # noqa: E402
from hac26.field import (CODE_DIM, DESIGN_N, EXTRACT_RES, KNN, N_DIR,             # noqa: E402
                         N_NODES, NODE_BETA, SH_DEGREE, DepthSphere, design_sha, real_sh,
                         dir_design, spherical_design)
from hac26.forward.convex_egi import normalize_np                                 # noqa: E402
from hac26.solvers.operator import CodeOperator                                   # noqa: E402
from hac26.train import load_net                                                  # noqa: E402
from train_lpd import (CALIBRATION, CORPUS, _enable_tf32, add_render_flags,   # noqa: E402
                       dh_expand, file_digest, inv_softplus, load_instrument,
                       model_error_scale, noise_sigma, render_from, smooth_noise_like,
                       support_from_mesh)

CODES = "runs/corpus_codes.npz"       # written by scripts/fit_shapes.py
CONVEX = "models/lpd_convex.pt"       # the convex stage's checkpoint, as scripts/reconstruct.py


def load_codes(codes_file: str, n: int):
    """The fitted codes, true hull supports and mounted radii of the first n bodies of the
    codes file (all of them for n = 0), after checking that they were fitted against the
    designs this code uses. Returns (codes, support, radius, n); a radius is NaN where the
    library did not record one."""
    if not Path(codes_file).exists():
        raise SystemExit(f"{codes_file} missing -- run scripts/fit_shapes.py first")
    zz = np.load(codes_file)
    if "codes" not in zz.files or "support" not in zz.files:
        raise SystemExit(f"{codes_file} must contain 'codes' and 'support' arrays")
    n = len(zz["codes"]) if n <= 0 else n
    if len(zz["codes"]) < n:
        raise SystemExit(f"{codes_file} contains {len(zz['codes'])} bodies, "
                         f"but --bodies requested {n}")
    if zz["codes"].ndim != 2 or zz["codes"].shape[1] != CODE_DIM:
        raise SystemExit(f"{codes_file} codes have shape {zz['codes'].shape}, but "
                         f"CODE_DIM={CODE_DIM}; rerun scripts/fit_shapes.py")
    if zz["support"].ndim != 2 or zz["support"].shape[1] != DESIGN_N:
        raise SystemExit(f"{codes_file} support has shape {zz['support'].shape}, "
                         f"but DESIGN_N={DESIGN_N}; rerun fit_shapes.py")
    # h is indexed BY NORMAL, so matching lengths is not enough: a design generated
    # independently on another machine has the same n and different points, and pairing the
    # two silently reindexes every body.
    meta = json.loads(str(zz["meta"])) if "meta" in zz.files else {}
    for key, live, what in (("design_sha", lambda: design_sha(spherical_design()),
                             f"hac26/design{DESIGN_N}.npy"),
                            ("dir_sha", lambda: design_sha(dir_design(N_DIR)),
                             f"hac26/design{N_DIR}.npy, the dh directions")):
        if key in meta and meta[key] != live():
            raise SystemExit(
                f"{codes_file} was fitted against {key} {meta[key]}, but {what} is {live()}. "
                f"Support and dh are indexed BY DIRECTION, so these cannot be mixed. Use the "
                f"design the codes were fitted with, or rerun scripts/fit_shapes.py.")
    radius = zz["radius"][:n] if "radius" in zz.files else np.full(n, np.nan)
    return zz["codes"][:n], zz["support"][:n], radius.astype(float), n


def corpus_radius(i: int, recorded: float) -> float:
    """The xy radius corpus body i is rendered at: the one its mounting gave it when the
    library recorded it, otherwise log-uniform over the range of the published radii,
    widened a little, and fixed by the body index."""
    if np.isfinite(recorded) and recorded > 0:
        return float(recorded)
    lo, hi = 0.9 * min(CYLINDER_R.values()), 1.1 * max(CYLINDER_R.values())
    u = np.random.default_rng(1000 + i).random()
    return float(lo * (hi / lo) ** u)


def corpus_meta(n, phases, op_res, calibration, convex, render) -> dict:
    """Everything the corpus depends on, stored with it. `schema` is bumped whenever the
    operator or the layout changes, so an older corpus is rebuilt rather than reused.

    `render` is the discretisation the curves were actually rendered at and not the
    calibrated one, so that a corpus built small to check the wiring is visibly a different
    corpus and is rebuilt rather than trained on."""
    return {
        "schema": 9,
        "bodies": int(n),
        "phases": int(phases),
        "n_geoms": int(len(cameras())),
        "operator_res": int(op_res),
        "design_n": int(DESIGN_N),
        "code_dim": int(CODE_DIM),
        # The node set, because the second block of the code is depths on it. CODE_DIM pins
        # only how many there are; changing which directions they are, or the width of their
        # kernels, changes what every depth means.
        "n_nodes": int(N_NODES),
        "node_sha": design_sha(DepthSphere(N_NODES).u.numpy()),
        "knn": int(KNN),
        "node_beta": float(NODE_BETA),
        "calibration": file_digest(calibration),
        "convex": file_digest(convex),
        "render": asdict(render),
    }


def convex_start(net, grid, n_phases: int, curves: torch.Tensor, radius: float) -> torch.Tensor:
    """The support the convex stage reconstructs from noisy curves (G, 2, P), made as
    scripts/reconstruct.py makes it: the intensity block over the binary block, resampled to
    the checkpoint's phase count, each curve divided by its mean, the checkpoint's support
    decoded to a hull, and that hull's support on the core's normals."""
    c = curves.cpu().numpy().astype(np.float64)
    stack = normalize_np(resample_curves(np.concatenate([c[:, 0], c[:, 1]], 0), n_phases))
    h = predict_h(net, grid, stack.astype(np.float32), np.ones(len(stack), np.float32), radius)
    v, f = decode(h, grid, 0, radius, False)
    return support_from_mesh(v, f)


_WORKER: dict = {}


def _worker_init(convex: str):
    torch.set_num_threads(1)
    _WORKER["net"], _WORKER["pr"], _WORKER["grid"] = load_net(convex, device="cpu")


def _worker_start(curves: torch.Tensor, radius: float) -> torch.Tensor:
    return convex_start(_WORKER["net"], _WORKER["grid"], _WORKER["pr"].m, curves, radius)


def correction_matrix() -> torch.Tensor:
    """(N_DIR, DESIGN_N): the least-squares fit of harmonics up to SH_DEGREE to a difference
    on the core's normals, evaluated on the dh directions. dh_expand takes the result back to
    the band-limited difference."""
    y_dir = real_sh(dir_design(N_DIR), SH_DEGREE)
    y_nrm = real_sh(spherical_design(DESIGN_N), SH_DEGREE)
    return torch.from_numpy(y_dir @ np.linalg.pinv(y_nrm))


def correction(h_true: torch.Tensor, h_start: torch.Tensor, to_dh: torch.Tensor):
    """The dh (N_DIR,) whose expansion best takes the start's support to the true one:
    inv_softplus(h_true) - inv_softplus(h_start) projected onto what dh can express, with
    `to_dh` from correction_matrix. Returns (dh, rms of the part of the difference dh cannot
    express)."""
    d = inv_softplus(h_true.double()) - inv_softplus(h_start.double())
    dh = to_dh @ d
    left = d - dh_expand().double() @ dh
    return dh.float(), float(left.pow(2).mean().sqrt())


class _Done:
    """A future that has already run, for the in-process path."""

    def __init__(self, fn, *args):
        try:
            self._value, self._exc = fn(*args), None
        except Exception as exc:                       # noqa: BLE001  raised by result()
            self._value, self._exc = None, exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value


# What a body's part holds, and therefore what a resumed body must bring back: the corpus
# stacks each of these at the end, so a field saved and not reloaded is a crash after the
# stage has done all its work.
PART_FIELDS = ("code", "curve", "turned_counts", "support", "support_true", "radius")


def _load_part(path: Path, expected: dict, i: int):
    """A body's saved part, or None when it is missing, unreadable or from other settings.
    `bodies` is left out of the comparison: body i does not depend on how many were asked."""
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
    except Exception as exc:                           # noqa: BLE001  corrupt: redo
        print(f"  ignoring unreadable part {path}: {exc}", flush=True)
        return None
    if int(z["body_index"]) != i or any(meta.get(k) != v for k, v in expected.items()
                                        if k != "bodies"):
        print(f"  ignoring stale part {path}", flush=True)
        return None
    # Every field the part was saved with, because every one of them is stacked into the
    # corpus at the end. Returning a subset made a resumed body raise on the final write,
    # after the whole stage had already paid for its rendering.
    return {k: z[k] for k in PART_FIELDS}


def _save_part(path: Path, i: int, part: dict, expected: dict):
    """Written under a temporary name and renamed, so a job killed mid-write leaves either the
    previous part or none, never half of one. A truncated part would be caught by _load_part
    and redone, but only after it had been read, and a rename costs nothing."""
    missing = [k for k in PART_FIELDS if k not in part]
    if missing:
        raise KeyError(f"body {i} is missing {missing} from its part; the corpus stacks every "
                       f"field of PART_FIELDS and would fail at the final write instead")
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temporary name has to end in .npz itself: np.savez silently appends .npz to any
    # filename that does not already end with it (_out_tmp below already works around this
    # for the final combined write; this per-part write needs the same treatment), so a plain
    # "<name>.npz.part" is actually written as "<name>.npz.part.npz" and the rename below
    # then fails with FileNotFoundError every time, having never found the file it wrote.
    tmp = path.with_suffix(".part.npz")
    np.savez(tmp, body_index=int(i), meta=json.dumps(expected, sort_keys=True), **part)
    tmp.replace(path)


def _out_tmp(out: str) -> str:
    """The stage's product is written here and renamed onto `out` by _out_commit, so a job
    killed mid-write leaves the previous file rather than a truncated one. numpy appends .npz
    to a name without it, so the temporary carries the extension already."""
    return f"{out}.writing.npz"


def _out_commit(out: str) -> None:
    Path(f"{out}.writing.npz").replace(out if out.endswith(".npz") else f"{out}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=0,
                    help="how many bodies of the codes file to use; 0 takes all of them")
    ap.add_argument("--phases", type=int, default=96,
                    help="rotation phases per curve; reconstruction uses the same count")
    ap.add_argument("--operator-res", type=int, default=EXTRACT_RES,
                    help="mesh extraction resolution of the operator")
    ap.add_argument("--codes-file", default=CODES, help="output of scripts/fit_shapes.py")
    ap.add_argument("--calibration", default=CALIBRATION,
                    help="the Instrument written by scripts/calibrate.py")
    ap.add_argument("--convex", default=CONVEX,
                    help="the convex stage's checkpoint, the one scripts/reconstruct.py uses")
    ap.add_argument("--out", default=CORPUS)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                    help="CPU processes running the convex stage; 0 runs it in this process")
    add_render_flags(ap)
    a = ap.parse_args()
    render = render_from(a, "corpus")
    _enable_tf32()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not Path(a.convex).exists():
        raise SystemExit(f"{a.convex} missing -- the convex stage's checkpoint is needed to "
                         f"make the starts the flow trains from")

    all_codes, all_sup, all_rad, n = load_codes(a.codes_file, a.bodies)
    if not np.isfinite(all_rad).all():
        print(f"  NOTE: {int((~np.isfinite(all_rad)).sum())} of {n} bodies carry no mounted "
              f"radius; those get one drawn from the published range", flush=True)
    inst = load_instrument(a.calibration, dev)
    eta = model_error_scale(inst)
    op = CodeOperator(inst, psi_grid(a.phases), res=a.operator_res, config=render, device=dev)
    expected = corpus_meta(n, a.phases, a.operator_res, a.calibration, a.convex, render)
    to_dh = correction_matrix()
    part_dir = Path(f"{a.out}.parts")
    pool = None
    if a.workers > 0:
        pool = ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"),
                                   initializer=_worker_init, initargs=(a.convex,))
    else:
        _worker_init(a.convex)
    print(f"  convex stage: {a.convex} on {a.workers or 1} CPU process(es); exact operator "
          f"on {dev}, {a.phases} phases, extraction res {a.operator_res}", flush=True)

    parts: dict = {}          # body index -> its arrays
    pending: dict = {}        # body index -> (future, code, curves, turned counts, radius, start)

    def finish(i):
        fut, code, cur, turned, radius, t0 = pending.pop(i)
        try:
            h_start = fut.result()
        except Exception as exc:                       # noqa: BLE001  no start, no body
            print(f"  body {i}: the convex stage failed on it ({exc}), skipped", flush=True)
            return
        dh, left = correction(torch.tensor(all_sup[i]), h_start, to_dh)
        code[:N_DIR] = dh
        parts[i] = {"code": code.numpy(), "curve": cur.numpy(), "turned_counts": turned.numpy(),
                    "support": h_start.numpy(), "support_true": all_sup[i],
                    "radius": np.float32(radius)}
        _save_part(part_dir / f"body_{i:05d}.npz", i, parts[i], expected)
        size = float(dh.pow(2).mean().sqrt())
        print(f"  body {i}: radius {radius:.2f}, correction rms {size:.4f} (beyond dh "
              f"{left:.4f}), {time.time()-t0:.1f}s", flush=True)

    for i in range(n):
        t0 = time.time()
        part = _load_part(part_dir / f"body_{i:05d}.npz", expected, i)
        if part is not None:
            parts[i] = part
            print(f"  body {i}: resumed", flush=True)
            continue
        code = torch.tensor(all_codes[i]); h_true = torch.tensor(all_sup[i])
        radius = corpus_radius(i, all_rad[i])
        rendered = op.curves_turned(h_true, code, radius)
        if rendered is None:
            print(f"  body {i}: no curves (degenerate mesh or unusable patches), skipped",
                  flush=True)
            continue
        cur, turned = rendered[0].cpu(), rendered[1].cpu()
        # one realisation of the noise and the model error, at the sizes training draws
        gen = torch.Generator().manual_seed(i)
        sigma = noise_sigma(1, generator=gen)[0]
        noisy = (cur + sigma[..., None] * torch.randn(cur.shape, generator=gen)
                 + eta[..., None] * smooth_noise_like(cur, generator=gen))
        if pool is None:
            fut = _Done(_worker_start, noisy, radius)
        else:
            fut = pool.submit(_worker_start, noisy, radius)
        pending[i] = (fut, code, cur, turned, radius, t0)
        # keep the queue short, so parts are written as bodies finish and a killed job loses
        # little
        while len(pending) > 2 * max(a.workers, 1) or (pool is None and pending):
            finish(next(iter(pending)))
    while pending:
        finish(next(iter(pending)))
    if pool is not None:
        pool.shutdown()
    if not parts:
        raise SystemExit("no corpus body could be rendered")

    index = sorted(parts)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(_out_tmp(a.out),
             codes=np.stack([parts[i]["code"] for i in index]).astype(np.float32),
             curves=np.stack([parts[i]["curve"] for i in index]).astype(np.float32),
             turned_counts=np.stack([parts[i]["turned_counts"] for i in index]).astype(np.float32),
             support=np.stack([parts[i]["support"] for i in index]).astype(np.float32),
             support_true=np.stack([parts[i]["support_true"] for i in index]).astype(np.float32),
             radius=np.asarray([parts[i]["radius"] for i in index], dtype=np.float32),
             index=np.asarray(index, dtype=np.int64),
             meta=json.dumps(expected, sort_keys=True))
    _out_commit(a.out)
    dh_all = np.stack([parts[i]["code"][:N_DIR] for i in index])
    print(f"  wrote {a.out}: {len(parts)} of {n} bodies; correction rms per body "
          f"{np.sqrt((dh_all ** 2).mean(1)).min():.4f}-{np.sqrt((dh_all ** 2).mean(1)).max():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
