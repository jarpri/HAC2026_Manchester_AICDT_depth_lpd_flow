#!/usr/bin/env python3
"""Fit the instrument against the curves of the public models, whose shapes are released.

    python scripts/calibrate.py                                # the laboratory channel
    python scripts/calibrate.py --channel blender --models 1 3 # the organisers' render

The two released channels are different instruments. The laboratory curves come through a
lens, a sensor and bounce light off a matte white print; the Blender render has none of
those and a far camera, so it is fitted from its own start (Instrument.blender_start) and
written to its own file, and a reconstruction against the render reads that file.

What is fitted, all at once and all by gradient: the albedo rho, the source radius delta, the
camera distance, the intensity threshold tau_i, the per-curve pedestal, the per-curve model
error eta, the sensor chain (PSF width, vignetting, OETF knots, saturation), and one start
phase psi0 per public body. The thresholds pass a gradient through the coarea formula, the
rest through the rendering itself. The objective is the Gaussian log-likelihood of the real
curves given the rendered ones,

    sum over present curves and phases of  (pred - real)^2 / s^2 + log s^2,
    s_c^2 = sigma_c^2 + eta_c^2,

with sigma_c the measurement noise of each curve, estimated from its own high-frequency
content at the files' native frame rate (hac26.noise). The log term is what stops the fit
from explaining every residual by a larger eta, and eta is where the A/B mounting mismatch
between the two columns of a geometry belongs.

psi0 is first found by a search over whole-frame shifts of the rendered curves against the
data, within an eighth of a turn either way, then refined with everything else. Its fitted
values are the check on hac26.conventions.PSI0: if they agree with each other and differ
from PSI0, PSI0 is wrong.

The released meshes are decimated to TRUTH_FACES faces before rendering; the interreflection
runs on the operator's usual patches.

Adam moves a parameter by about lr per step whatever the gradient, so `--steps x --lr` is a
hard cap on how far any of them can travel in raw (unsquashed) space. A fit that spends most
of that cap stopped because the run ended, not because it converged, and its parameters are
wherever the cap left them. The run therefore stops early once the likelihood plateaus, and
prints how far every parameter travelled against its budget, naming the ones that were still
moving. Read that table before the residuals: while it names anything, the residuals are
those of a truncated fit.

The fit is hours and everything downstream needs the instrument it writes, so it checkpoints
every --ckpt-every steps to --ckpt-file (default <--out>.ckpt) and resumes from it by default,
carrying the optimiser, the start phases and the likelihood history, so a resumed run stops on
the same plateau an uninterrupted one would. --time-budget stops it between steps and then
writes the instrument it had reached, which a run killed by a wallclock does not.

Writes the Instrument to models/instrument_calibration.pt, and the fitted psi0 per body with
the per-geometry residual report and the movement table to
models/instrument_calibration.json. The residual at the true shape divided by the noise, per
geometry, is the number that says whether the forward model reproduces the organisers'
processing; everything downstream rests on it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.conventions import PUBLIC_MODELS, SENSE, cameras, psi_grid   # noqa: E402
from hac26.data_io import (N_CAMS, load_model_curves, native_sigma,    # noqa: E402
                           public_stl)
from hac26.forward.mesh.exact import (ExactForward, RenderConfig, decimate, normalise,   # noqa: E402
                                      normalise_vjp)
from hac26.forward.mesh.instrument import Instrument                  # noqa: E402
from hac26.noise import ab_mismatch                                   # noqa: E402
from hac26.shapes import rescale_touch_z                              # noqa: E402
from hac26.stl_io import load_stl                                     # noqa: E402

TRUTH_FACES = 20000       # faces the released meshes are decimated to before rendering
PSI0_SEARCH = 1.0 / 8.0   # the start phase is searched within this fraction of a turn each way
ETA_FLOOR = 4e-3       # smallest per-curve model error the saved calibration will claim.
                       # The residual of a curve at the true shape is partly the
                       # discretisation of one particular mesh at one resolution, which does
                       # not transfer to a body of a different size, and a curve whose model
                       # error is fitted below that carries a weight the measurement does not
                       # earn. The value is the spread of the per-curve residual across the
                       # public bodies, which the report prints.
OUT_INSTRUMENT = {"real": "models/instrument_calibration.pt",
                  "blender": "models/instrument_blender.pt"}
OUT_REPORT = {"real": "models/instrument_calibration.json",
              "blender": "models/instrument_blender.json"}


def load_truth(data_dir: str, model: int, device: str, faces: int = TRUTH_FACES):
    """The released mesh of a public model, posed and decimated, as torch tensors."""
    v, f = load_stl(public_stl(data_dir, model))
    # centre_xy=False: the released STL is already posed on the rotation axis, and moving it
    # onto its own centroid would move it off (hac26.shapes.rescale_touch_z)
    v = rescale_touch_z(v, f, centre_xy=False)
    v, f = decimate(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64), faces)
    return (torch.tensor(v, dtype=torch.float32, device=device),
            torch.tensor(f, dtype=torch.long, device=device))


def load_data(data_dir: str, model: int, phases: int, device: str, channel: str = "real"):
    """The mean-normalised curves (N_CAMS, 2, P) of one channel, which geometries are present
    (N_CAMS,) and the measured noise per curve (N_CAMS, 2).

    The noise comes from the high-frequency content of each curve at the files' own frame
    rate, not from the difference of the two columns of a geometry: those two columns are
    separate recordings of the body in its two mountings, so their difference is dominated by
    the A/B mismatch and runs 1-277x the actual noise, median 12x (hac26.noise). That
    difference is reported beside sigma as a diagnostic and is left for eta to absorb.
    """
    d = load_model_curves(data_dir, model, m=phases, use_blender=(channel == "blender"))
    if not d["files"]:
        raise FileNotFoundError(f"model {model}: no {channel} curve files under {data_dir}")
    pairs = np.stack([d["curves"][:N_CAMS], d["curves"][N_CAMS:]], axis=1)
    present = (d["mask"][:N_CAMS] > 0) & (d["mask"][N_CAMS:] > 0)
    sigma = native_sigma(d).reshape(2, N_CAMS).T
    # The mismatch is a diagnostic and is printed beside sigma; it enters neither the fit nor
    # eta. It is the disagreement between two independent recordings of the body in its two
    # mountings, and a deterministic render has no second recording: the organisers' Blender
    # files reproduce the duplicated column exactly, so the duplicate drop leaves no pair with
    # both columns and there is nothing to measure. That is a property of a synthetic channel,
    # not a fault in the data, so it is reported as unavailable and the fit goes on.
    try:
        mismatch = ab_mismatch(d["curves"], d["mask"]).reshape(2, N_CAMS).T
    except ValueError:
        mismatch = np.full((N_CAMS, 2), np.nan, dtype=np.float32)
    return (torch.tensor(pairs, dtype=torch.float32, device=device),
            torch.tensor(present, device=device),
            torch.tensor(sigma, dtype=torch.float32, device=device),
            torch.tensor(mismatch, dtype=torch.float32, device=device))


def initial_psi0(fwd: ExactForward, verts, faces, real, present, sigma, mesh=None) -> float:
    """The whole-frame shift of the rendered curves that fits the data best, as a start
    phase. Shifting psi0 by one grid step moves every curve by one frame, so one rendering
    serves every candidate."""
    P = real.shape[-1]
    pred = normalise(fwd.raw_curves(verts, faces, psi0=0.0, mesh=mesh))
    best, best_j = float("inf"), 0
    for j in range(-int(P * PSI0_SEARCH), int(P * PSI0_SEARCH) + 1):
        r = (torch.roll(pred, -j, dims=-1) - real) / sigma[..., None]
        m = float((r ** 2)[present].mean())
        if m < best:
            best, best_j = m, j
    return float(SENSE * 2.0 * np.pi * best_j / P)


def nll(pred, real, present, sigma, eta):
    """The Gaussian negative log-likelihood per present curve and phase, and its derivative
    with respect to the prediction."""
    s2 = sigma ** 2 + eta ** 2
    r = pred - real
    n = float(present.sum()) * real.shape[-1]
    loss = (((r ** 2 / s2[..., None]) + torch.log(s2)[..., None]) * present[:, None, None]).sum() / n
    cot = 2.0 * r / s2[..., None] * present[:, None, None] / n
    return loss, cot


def likelihood_cotangent(raw, body, eta):
    """The cotangent on the unnormalised curves: the likelihood's derivative with respect to
    the normalised curves, taken back through the normalisation."""
    _, cot = nll(normalise(raw), body["real"], body["present"], body["sigma"], eta)
    return normalise_vjp(raw, cot)


def curve_residual(pred, real, present) -> torch.Tensor:
    """RMS residual of every curve over the phases, in curve units, as (N_CAMS, 2), with the
    geometries that were not measured contributing zero.

    `present` says which geometry was recorded, one flag per geometry, while the curves carry
    a geometry, a column and a phase. The mask therefore belongs on the first axis; broadcast
    from the right it would meet the column axis instead, which is the shape this function
    exists to keep in one place. Zero rather than dropped because the caller takes a maximum
    over bodies, and a geometry with no data can never win one.
    """
    r = (pred - real) * present[:, None, None]
    return r.pow(2).mean(-1).sqrt()


def residual_report(pred, real, present, sigma, eta) -> dict:
    """RMS residual per geometry over the phases, divided by the noise alone and by the total
    scale sqrt(sigma^2 + eta^2), for the intensity and the binary curves."""
    r = (pred - real)
    per_sigma = (r / sigma[..., None]).pow(2).mean(-1).sqrt()
    per_s = (r / torch.sqrt(sigma ** 2 + eta ** 2)[..., None]).pow(2).mean(-1).sqrt()
    keep = present.cpu().numpy()
    out = {}
    for k, name in enumerate(("intensity", "binary")):
        a = np.where(keep, per_sigma[:, k].cpu().numpy(), np.nan)
        b = np.where(keep, per_s[:, k].cpu().numpy(), np.nan)
        out[name] = {"per_sigma": a.tolist(), "per_s": b.tolist()}
    return out


def _shift(curves: torch.Tensor, frac: float) -> torch.Tensor:
    """Circularly shift along the phase axis by `frac` frames, linearly interpolated. The
    curves are periodic, so this is exact wraparound; fractional because the shifts worth
    seeing are smaller than one frame of the calibration's grid -- the organisers' realignment
    of model 1 was up to 12 frames of 841, which is 0.7 of a frame at 48 phases."""
    P = curves.shape[-1]
    idx = torch.arange(P, dtype=torch.float32, device=curves.device) - frac
    i0 = torch.floor(idx)
    w = (idx - i0).to(curves.dtype)
    i0 = i0.long() % P
    return curves[..., i0] * (1 - w) + curves[..., (i0 + 1) % P] * w


def phase_offset_report(pred, real, present, sigma, search: float = PSI0_SEARCH,
                        step: float = 0.125) -> dict:
    """The shift each azimuth group would still like, in degrees of rotation, at the fitted
    psi0.

    The calibration fits one start phase per body, which is right if the body's frames are
    aligned with each other. The organisers align them per azimuth: the 17/25 Aug 2026 update
    to model 1's real curves shifted each azimuth's four columns by its own whole-frame offset
    (0 deg by -12 frames of 841, 90 and 135 by +3, 225 by -4, 270 by -2, 45 and 315 not at
    all), leaving the shifted curves matching the old ones at corr = 1.0000. No single psi0
    absorbs that, so a residual per-azimuth misalignment lands in the residual table looking
    like forward-model error -- and it costs most where the curves move fastest, which is the
    high phase angles that carry the most shape.

    Reported, not fitted: seven more free parameters per body would explain away real misfit
    just as readily. Groups that all want the same shift mean psi0 is off; groups that
    disagree mean the curves are not aligned with each other, and the fix is a fresh download
    rather than a wider fit (scripts/check_data.py).
    """
    P = real.shape[-1]
    w = present[:, None, None] / sigma[..., None]
    span = P * search
    grid = [k * step for k in range(-int(span / step), int(span / step) + 1)]
    out = {}
    for az in sorted({c.azimuth_deg for c in cameras()}):
        idx = [i for i, c in enumerate(cameras()) if c.azimuth_deg == az]
        pr, rl, ww = pred[idx], real[idx], w[idx]
        best, best_f = float("inf"), 0.0
        for f in grid:
            m = float(((_shift(pr, f) - rl) * ww).pow(2).mean())
            if m < best:
                best, best_f = m, f
        out[str(az)] = 360.0 * best_f / P
    return out


def print_phase_offsets(model: int, off: dict, frames: int) -> None:
    """One row per body, in degrees of rotation. All zero is a correctly aligned set."""
    az = sorted(off, key=float)
    res = 360.0 * 0.125 / frames
    print(f"    model {model}: " + "  ".join(f"{float(a):>3.0f}deg {off[a]:+6.2f}" for a in az)
          + f"   (deg, resolution {res:.2f})")


def movement_report(start: dict, now: dict, budget: dict) -> dict:
    """How far each fitted parameter travelled in its raw (unsquashed) space, against how far
    the optimiser could have moved it.

    Adam's step is about lr in magnitude whatever the gradient, so `steps * lr` is a hard cap
    on the travel of any parameter. A parameter that spends most of that cap has not
    converged -- it stopped because the run ended. Every quantity here is stored through a
    squashing function, so the raw space is the one the cap applies in.
    """
    out = {}
    for name, x0 in start.items():
        moved = float((now[name] - x0).abs().max())
        out[name] = {"moved": moved, "budget": budget[name],
                     "fraction": moved / max(budget[name], 1e-12)}
    return out


def print_movement(rep: dict, limit: float = 0.5) -> list:
    """The movement table, and the names that used more than `limit` of their budget."""
    print("\n[budget] travel of each parameter in raw space, against steps x lr")
    limited = [n for n, r in rep.items() if r["fraction"] > limit]
    for name, r in sorted(rep.items(), key=lambda kv: -kv[1]["fraction"]):
        flag = "  <-- still moving when the run ended" if r["fraction"] > limit else ""
        print(f"    {name:<24} {r['moved']:8.3f} of {r['budget']:7.3f}  "
              f"({100 * r['fraction']:5.1f}%){flag}")
    if limited:
        print(f"  !!! {len(limited)} parameter(s) used more than {100 * limit:.0f}% of the "
              f"travel the step budget allows: {', '.join(limited)}.")
        print("  !!! The fit is bounded by --steps, not by the data. Rerun with more steps "
              "(or a larger --lr) until this list is empty before trusting the residuals.")
        if "raw_eta" in limited:
            # Said here because the advice above does not apply to this one and would be
            # followed. eta runs downhill until the likelihood's log term stops it, and on a
            # channel the chain reproduces closely it wants a smaller value than the floor
            # the saved instrument will clamp it to. Travel toward a bound is not an
            # unconverged fit, and no number of steps clears it.
            print(f"  !!! raw_eta is the exception: the saved instrument floors the model "
                  f"error at {ETA_FLOOR}, so on a channel whose residual is below that, eta "
                  f"travels its whole budget toward a bound it cannot pass. Read the eta line "
                  f"below rather than adding steps for this one.")
    else:
        print("  every parameter settled well inside its budget")
    return limited


def print_report(model: int, rep: dict) -> None:
    """One row per camera kind, one column per azimuth, for each curve type and each
    denominator; nothing is aggregated except the medians on the last line."""
    cams = cameras()
    kinds = ("hor_a", "hor_b", "top", "bottom")
    azimuths = sorted({c.azimuth_deg for c in cams})
    print(f"  model {model}:" + " " * 22 + " ".join(f"{z:>6.0f}" for z in azimuths))
    for name in ("intensity", "binary"):
        for denom in ("per_sigma", "per_s"):
            vals = np.asarray(rep[name][denom])
            for kind in kinds:
                row = [vals[i] for i, c in enumerate(cams) if c.kind == kind]
                print(f"    {name:<9} /{denom[4:]:<5} {kind:<7} "
                      + " ".join(f"{v:>6.1f}" for v in row))
        a = np.asarray(rep[name]["per_sigma"]); b = np.asarray(rep[name]["per_s"])
        print(f"    {name:<9} median /sigma {np.nanmedian(a):.2f}, median /s "
              f"{np.nanmedian(b):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=int, default=48)
    ap.add_argument("--steps", type=int, default=600,
                    help="cap on the number of steps. Adam moves a parameter by about lr per "
                         "step, so steps x lr is a hard cap on how far any of them can "
                         "travel in raw space; the run reports what each one used")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="stop early once the mean -logL has improved by less than this over "
                         "the last --patience steps")
    ap.add_argument("--patience", type=int, default=60,
                    help="window of steps the --tol improvement is measured over")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--channel", choices=("real", "blender"), default="real",
                    help="which released curves to fit: the laboratory recordings or the "
                         "organisers' Blender render, each from its own starting instrument")
    ap.add_argument("--models", nargs="+", type=int, default=list(PUBLIC_MODELS),
                    help="public bodies to fit on. One eta is shared across them, so a body "
                         "the chain cannot reproduce raises the model error admitted for "
                         "every other body and for every secret model fitted against the "
                         "result. Model 2, the sawed-off cube, misses by 4.2x the combined "
                         "noise where models 1 and 3 sit at 0.2-0.7x: its flat faces put "
                         "Otsu in a regime the chain does not reproduce. Every secret model "
                         "is round-regime like 1 and 3, so fitting on the cube buys nothing "
                         "and costs the noise floor.")
    ap.add_argument("--out", default=None,
                    help=f"instrument file; by channel, {OUT_INSTRUMENT}")
    ap.add_argument("--report", default=None,
                    help=f"report file; by channel, {OUT_REPORT}")
    render = RenderConfig()
    ap.add_argument("--phase-chunk", type=int, default=render.phase_chunk,
                    help="phases per rendering batch; with --geom-chunk it sets the GPU "
                         "memory the rendering takes, not the result")
    ap.add_argument("--geom-chunk", type=int, default=render.geom_chunk,
                    help="geometries per rendering batch")
    ap.add_argument("--truth-faces", type=int, default=TRUTH_FACES,
                    help="faces the released meshes are decimated to before rendering")
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="seconds after which the fit stops between steps and writes the "
                         "instrument it has reached; 0 is no budget. A fit killed by a "
                         "wallclock instead writes nothing at all, and everything downstream "
                         "needs an instrument to load")
    ap.add_argument("--ckpt-every", type=int, default=25,
                    help="steps between checkpoints; 0 writes none, which loses the fit to a "
                         "wallclock")
    ap.add_argument("--ckpt-file", default=None,
                    help="resumable checkpoint; by default <--out>.ckpt")
    ap.add_argument("--no-resume", action="store_true",
                    help="start from the channel's own starting instrument even when a "
                         "checkpoint for these settings is there")
    ap.add_argument("--height", type=int, default=render.height,
                    help="sensor image height; with --width and --sun-res, a lower value "
                         "checks the wiring on a CPU and is not a calibration")
    ap.add_argument("--width", type=int, default=render.width)
    ap.add_argument("--sun-res", type=int, default=render.sun_res)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a.out = a.out or OUT_INSTRUMENT[a.channel]
    a.report = a.report or OUT_REPORT[a.channel]
    if any(M not in PUBLIC_MODELS for M in a.models):
        raise SystemExit(f"only the public models {PUBLIC_MODELS} have a released shape")

    inst = (Instrument() if a.channel == "real" else Instrument.blender_start()).to(dev)
    render = RenderConfig(height=a.height, width=a.width, sun_res=a.sun_res,
                          phase_chunk=a.phase_chunk, geom_chunk=a.geom_chunk)
    if render != RenderConfig(phase_chunk=a.phase_chunk, geom_chunk=a.geom_chunk):
        print(f"  NOTE: rendering at {render.height}x{render.width}, sun view {render.sun_res}; "
              f"a fit at a reduced resolution checks the wiring and is not a calibration",
              flush=True)
    fwd = ExactForward(inst, psi_grid(a.phases), render, device=dev)
    fit_params = [p for _, p in inst.fitted_parameters()]
    print(f"[start] {a.channel} channel, models {a.models}: {inst.summary()}", flush=True)

    bodies = {}
    for M in a.models:
        t0 = time.time()
        verts, faces = load_truth(a.data_dir, M, dev, a.truth_faces)
        real, present, sigma, mismatch = load_data(a.data_dir, M, a.phases, dev, a.channel)
        # The form factors of the patches belong to the mesh, and only the instrument moves
        # during the fit, so they are built once here. Rebuilt every step they were most of the
        # cost of a step: the visibility ray test of a decimated public body runs on the CPU.
        mesh = fwd.mesh_constants(verts, faces)
        with torch.no_grad():
            psi0 = initial_psi0(fwd, verts, faces, real, present, sigma, mesh)
        bodies[M] = dict(verts=verts, faces=faces, mesh=mesh, real=real, present=present,
                         sigma=sigma,
                         psi0=torch.tensor(psi0, device=dev, requires_grad=True))
        print(f"  model {M}: {len(faces)} faces, {int(present.sum())}/{N_CAMS} geometries, "
              f"noise median {float(sigma.median()):.4f} ("
              + ("A/B mismatch not measurable: no geometry has two independent recordings"
                 if bool(torch.isnan(mismatch).all()) else
                 f"A/B mismatch {float(mismatch.median()):.4f}, "
                 f"{float(mismatch.median()/sigma.median()):.0f}x") + "), "
              f"start phase {np.degrees(psi0):+.1f} deg ({time.time()-t0:.0f}s)", flush=True)

    # A channel whose instrument is measured rather than fitted has no render-path parameter to
    # move: fitted_parameters() is empty by construction there, and the start phase is already
    # the best of every whole-frame shift, searched exactly rather than descended to. Nothing
    # that reaches the rendering then changes from step to step, so the predictions are
    # constant: they are rendered once here and only eta is fitted, whose gradient is direct
    # through the likelihood. That takes the render, and the gradient through it, out of the
    # loop entirely -- which is both what the channel means and the only way the fit can be
    # stated, since a gradient with respect to a start phase alone is the one thing the vjp is
    # then being asked for and it is not a quantity this instrument has.
    render_fitted = bool(fit_params)
    if not render_fitted:
        with torch.no_grad():
            for b in bodies.values():
                b["psi0"] = b["psi0"].detach()
                b["pred"] = normalise(fwd.raw_curves(b["verts"], b["faces"],
                                                     psi0=float(b["psi0"]), mesh=b["mesh"]))
        print("  the instrument of this channel is measured rather than fitted, so the curves "
              "are rendered once and only the model error is fitted", flush=True)

    groups = [{"params": fit_params + [inst.raw_eta]}]
    if render_fitted:
        groups.append({"params": [b["psi0"] for b in bodies.values()], "lr": a.lr / 10})
    opt = torch.optim.Adam(groups, lr=a.lr)

    # The fit is hours and everything downstream needs the instrument it writes, so a run
    # killed by a wallclock must not leave nothing behind. What is checkpointed is the
    # instrument, the start phases, the optimiser and the likelihood history: with the
    # history, a resumed run stops on the same plateau an uninterrupted one would, rather
    # than restarting a patience window that has already been spent. --steps is deliberately
    # not part of the identity, because raising the cap and continuing is a thing to want; it
    # only changes the travel budget the report is read against, which the report states.
    ckpt_path = Path(a.ckpt_file or f"{a.out}.ckpt")
    keys = {"channel": a.channel, "models": list(a.models), "phases": int(a.phases),
            "truth_faces": int(a.truth_faces), "lr": float(a.lr),
            "render": f"{render.height}x{render.width}x{render.sun_res}"}
    history, start_step, elapsed_before = [], 0, 0.0
    if ckpt_path.exists() and not a.no_resume:
        st = torch.load(ckpt_path, map_location=dev, weights_only=False)
        if st.get("keys") != keys:
            print(f"  {ckpt_path} was written under other settings, so it is ignored and the "
                  f"fit starts from the channel's own start", flush=True)
        else:
            inst.load_state_dict(st["instrument"])
            opt.load_state_dict(st["opt"])
            with torch.no_grad():
                for M, b in bodies.items():
                    if str(M) in st["psi0"]:
                        b["psi0"].copy_(torch.as_tensor(st["psi0"][str(M)], device=dev))
            history = list(st["history"])
            start_step, elapsed_before = int(st["step"]) + 1, float(st["elapsed"])
            print(f"  resumed {ckpt_path} at step {start_step} of {a.steps} "
                  f"({elapsed_before:.0f}s of fitting before this run)", flush=True)

    def save_ckpt(step: int) -> None:
        """Written under a temporary name and renamed, so a kill mid-write leaves the
        previous checkpoint rather than a truncated one."""
        if not a.ckpt_every:
            return
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_name(ckpt_path.name + ".part")
        torch.save({"keys": keys, "instrument": inst.state_dict(), "opt": opt.state_dict(),
                    "psi0": {str(M): float(b["psi0"]) for M, b in bodies.items()},
                    "history": history, "step": step,
                    "elapsed": elapsed_before + (time.time() - t_fit)}, tmp)
        tmp.replace(ckpt_path)
    # raw-space starting point and travel budget of everything being fitted, for the
    # convergence report at the end
    named = {n: p for n, p in inst.fitted_parameters()}
    named["raw_eta"] = inst.raw_eta
    if render_fitted:
        named.update({f"psi0[{M}]": b["psi0"] for M, b in bodies.items()})
    # The travel is measured from where this run started, which on a resumed run is the
    # checkpointed instrument and not the channel's own start, so the budget it is read
    # against is this run's steps and not every run's.
    start = {n: p.detach().clone() for n, p in named.items()}
    per_step = {n: (a.lr / 10 if n.startswith("psi0") else a.lr) for n in named}
    steps_before = len(history)
    t_fit = time.time()

    print(f"[fit] up to {a.steps} steps over {len(bodies)} bodies at {a.phases} phases "
          f"(early stop: -logL improving by < {a.tol:g} over {a.patience} steps)"
          + (f", time budget {a.time_budget:.0f}s" if a.time_budget else ""), flush=True)
    stopped_on_time = False
    for step in range(start_step, a.steps):
        opt.zero_grad()
        total = 0.0
        for M, b in bodies.items():
            eta = inst.eta.reshape(2, N_CAMS).T
            if render_fitted:
                # the curves and the gradient of the likelihood with respect to every
                # instrument parameter and this body's start phase, through the rendering
                raw, _, grads = fwd.vjp(
                    b["verts"], b["faces"], lambda r: likelihood_cotangent(r, b, eta.detach()),
                    psi0=b["psi0"], params=fit_params + [b["psi0"]], mesh=b["mesh"])
                for p, g in zip(fit_params + [b["psi0"]], grads):
                    p.grad = g if p.grad is None else p.grad + g
                pred = normalise(raw)
            else:
                pred = b["pred"]          # constant: nothing reaching the render is fitted
            # eta enters only through the likelihood, so its gradient is direct
            loss, _ = nll(pred, b["real"], b["present"], b["sigma"], eta)
            loss.backward()
            total += float(loss.detach())
        opt.step()
        mean_loss = total / len(bodies)
        history.append(mean_loss)
        if step % 10 == 0 or step == a.steps - 1:
            print(f"  step {step:>4}  -logL {mean_loss:.4f}  {inst.summary()}; "
                  f"psi0 " + ", ".join(f"{np.degrees(float(b['psi0'])):+.1f}"
                                       for b in bodies.values()) + " deg", flush=True)
        if len(history) > a.patience and min(history[:-a.patience]) - mean_loss < a.tol:
            print(f"  stopped at step {step}: -logL improved by less than {a.tol:g} over the "
                  f"last {a.patience} steps", flush=True)
            save_ckpt(step)
            break
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0:
            save_ckpt(step)
        if a.time_budget and elapsed_before + (time.time() - t_fit) > a.time_budget:
            # The report and the save below run either way, so stopping here writes the
            # instrument the fit has reached rather than losing it. The checkpoint stays, so
            # a later run continues from this step instead of starting over.
            print(f"  the time budget stopped the fit at step {step}, with -logL at "
                  f"{mean_loss:.4f}; the instrument it had reached is written and "
                  f"{ckpt_path} continues it", flush=True)
            save_ckpt(step)
            stopped_on_time = True
            break
    steps_run = len(history)
    budget = {n: v * max(steps_run - steps_before, 1) for n, v in per_step.items()}

    print("\n[report] RMS residual at the true shape, per geometry (azimuth:value)")
    print("    /sigma  against the measurement noise alone")
    print("    /s      against sqrt(sigma^2 + eta^2), eta being the model error the fit admits")
    moved = movement_report(start, {n: p.detach() for n, p in named.items()}, budget)
    limited = print_movement(moved)
    report = {"channel": a.channel, "models": a.models,
              "psi0_deg": {}, "residual": {}, "phase_offset_deg": {},
              "instrument": inst.summary(),
              "steps_run": steps_run, "steps_before_this_run": steps_before,
              "stopped_on_time": stopped_on_time,
              "movement": moved, "budget_limited": limited}
    with torch.no_grad():
        eta = inst.eta.reshape(2, N_CAMS).T
        worst_eta = torch.zeros_like(eta)
        for M, b in bodies.items():
            pred = normalise(fwd.raw_curves(b["verts"], b["faces"], psi0=float(b["psi0"]),
                                            mesh=b["mesh"]))
            rep = residual_report(pred, b["real"], b["present"], b["sigma"], eta)
            print_report(M, rep)
            report["residual"][M] = rep
            report["psi0_deg"][M] = float(np.degrees(float(b["psi0"])))
            report["phase_offset_deg"][M] = phase_offset_report(
                pred, b["real"], b["present"], b["sigma"])
            worst_eta = torch.maximum(worst_eta,
                                      curve_residual(pred, b["real"], b["present"]))
        # The likelihood fits one eta per curve across the bodies, which is near the pooled
        # residual and therefore under-covers the worst of them. What a curve's model error
        # has to cover is the worst body the method will meet, and three public bodies are
        # the only sample of that there is, so the saved eta is the largest residual any of
        # them leaves. The floor is there because the smallest residuals are the
        # discretisation of one mesh at one resolution and do not transfer to a body of a
        # different size; without it one curve of one body would carry the whole likelihood.
        inst.raw_eta.copy_(torch.log(torch.expm1(
            worst_eta.T.reshape(-1).clamp_min(ETA_FLOOR))))
        report["eta_pooled_median"] = float(eta.median())
        report["eta_worst_body_median"] = float(inst.eta.median())
        report["eta_floor"] = ETA_FLOOR
    print(f"\n[eta] per-curve model error raised from the pooled fit to the worst public "
          f"body's residual, floored at {ETA_FLOOR}: median "
          f"{report['eta_pooled_median']:.4f} -> {report['eta_worst_body_median']:.4f}")
    print("\n[alignment] whole-frame shift each azimuth still wants at the fitted psi0")
    print("    all zero = the body's frames agree with each other; a nonzero row means that")
    print("    azimuth is misaligned with the others, which no single psi0 can absorb and")
    print("    which the residual table above will show as forward-model error")
    for M, off in report["phase_offset_deg"].items():
        print_phase_offsets(M, off, a.phases)
    worst = max((abs(v) for off in report["phase_offset_deg"].values()
                 for v in off.values()), default=0.0)
    if worst > 360.0 * 0.25 / a.phases:
        print("  !!! Some azimuths are misaligned. Check the download against the manifest")
        print("  !!! (scripts/check_data.py) before reading the residuals: the organisers have")
        print("  !!! realigned these curves once already.")
    print(f"  fitted: {inst.summary()}")
    print("  start phases: " + ", ".join(f"model {M} {v:+.2f} deg"
                                          for M, v in report["psi0_deg"].items()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    inst.save(a.out)
    Path(a.report).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out} and {a.report}")


if __name__ == "__main__":
    main()
