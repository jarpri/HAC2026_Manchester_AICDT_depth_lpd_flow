#!/usr/bin/env python3
"""Refine the convex answer of one model by descending the exact misfit of one channel.

    python scripts/reconstruct_map.py --model 3 --channel blender --hold-out-geoms 6 \
        --out results/map/Asteroid03.stl

No flow, no training, no corpus. The convex stage's answer is the base support h; the code,
the band-limited correction dh of that support and the depths that carve, starts at zero,
which is the convex body, and moves by gradient descent on the whitened misfit through the
exact forward model, with a ridge on the depths as the only prior.

A lightcurve misfit is not the score, and a body can fit the curves better while resembling
the truth less, so the run measures both. --hold-out-geoms keeps cameras out of the fit and
reports the misfit on them, which is the difference between recovering a shape and fitting
curves; on a public model the Dice against the released shape is printed at every checkpoint.
The written body is extracted at --export-res and measured at --export-phases, both
independent of the grid and the phase count the descent could afford, because the number that
decides whether it enters the submission is a property of the body rather than of what the
descent cost. Those measurements are recorded beside the descent's own, and the objective
beside every misfit: scripts/select_answers.py reads them, judges this body on the functional
it was fitted under, and decides per model whether it replaces the convex answer.

The step is a backtracking line search along the sign-normalised gradient, per block. The
objective's curvature varies over orders of magnitude across the coordinates, and dh and g
are in different units, so a fixed step or a single scale lets one block set the step for
both; halving until the objective falls, and growing the step when it does, needs only the
direction.

A descent whose extracted mesh the export guard refuses exits 3 rather than 1. The descent
finished in that case and its numbers are written; only the mesh is unusable, and the
coefficients beside it (.code.npz) extract again without repeating the descent.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.conventions import CYLINDER_R, PUBLIC_MODELS, psi_grid        # noqa: E402
from hac26.data_io import (held_out_geoms, load_inversion_curves,   # noqa: E402
                           public_stl)
from hac26.field import (CODE_DIM, EXTRACT_EXTENT, EXTRACT_RES, N_DIR,   # noqa: E402
                         depth_cap, extract_mesh)
from hac26.recon import dice, fit_to_cylinder, mesh_occupancy            # noqa: E402
from hac26.shapes import rescale_touch_z                                 # noqa: E402
from hac26.solvers.operator import CodeOperator                          # noqa: E402
from hac26.solvers.output import export_stl, restore_constraints         # noqa: E402
from calibrate import ETA_FLOOR                                          # noqa: E402
from reconstruct import answer_path                                      # noqa: E402
from reconstruct_lpd import (curve_pairs, curve_weight, json_default,   # noqa: E402
                             measured_geometries, residual_scale,
                             support_from_convex, whitened_misfit)
from hac26.solvers.gauss_newton import AREA_WEIGHT, AREA_WINDOW           # noqa: E402
from train_lpd import (INSTRUMENT, _enable_tf32, add_render_flags,   # noqa: E402
                       load_instrument, render_from)

EXPORT_REFUSED = 3    # exit status of a run whose fit finished and whose extracted mesh the
                      # export guard refused. It is not the status of a run that broke: the
                      # numbers are written, the coefficients are beside them, and only the
                      # extraction has to be redone. Both solvers use it, so a runbook can
                      # tell a body it has no mesh for from a stage that failed.
OCC_RES = 128         # grid of the Dice reported at the checkpoints
EXPORT_RES = EXTRACT_RES   # default extraction resolution of the written mesh: the
                           # operator's own, so that by default the body written is the one
                           # whose misfit was accepted
TARGET_SIGMA = 1.0    # stop once the answer explains the data to the noise level
MAX_HALVINGS = 8      # trial steps per iteration before declaring convergence
STEP_GROW = 1.6       # a step that works makes the next trial bolder


def truth_dice(verts, faces, model: int, data_dir: str) -> float:
    """Dice of a posed reconstruction against the released truth, or nan when there is none."""
    if model not in PUBLIC_MODELS or not Path(public_stl(data_dir, model)).exists():
        return float("nan")
    import trimesh
    t = trimesh.load(public_stl(data_dir, model), process=False)
    tf = np.asarray(t.faces)
    tv = rescale_touch_z(np.asarray(t.vertices), tf, centre_xy=False)
    rv = rescale_touch_z(np.asarray(verts), np.asarray(faces), centre_xy=False)
    e = max(float(np.abs(rv).max()), float(np.abs(tv).max())) * 1.05
    return float(dice(mesh_occupancy(rv, np.asarray(faces), OCC_RES, e),
                      mesh_occupancy(tv, tf, OCC_RES, e)))


def convex_dice(stl: str, model: int, data_dir: str) -> float:
    """Dice of the answer a refinement started from, against the released truth. It is what
    says whether a refinement moved a public body toward its truth or away from it, and
    therefore whether the misfit ratio it reached may be trusted on a body whose truth is
    secret; scripts/select_answers.py reads both."""
    import trimesh
    m = trimesh.load(stl, process=False)
    return truth_dice(np.asarray(m.vertices), np.asarray(m.faces), model, data_dir)


def convexity(verts, faces) -> float:
    """Volume over convex-hull volume; nan for a mesh whose hull cannot be built."""
    import trimesh
    from scipy.spatial import ConvexHull
    m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=True)
    try:
        return float(abs(m.volume) / ConvexHull(np.asarray(m.vertices)).volume)
    except Exception:                                    # noqa: BLE001
        return float("nan")


def export_measure(op_x: CodeOperator, support, radius: float, data, scale, weight,
                   area_weight: float):
    """measure(code, geoms, c=None) -> (misfit, objective) of a correction on those cameras,
    at the resolution and phase count a written body is scored at.

    The objective goes beside the misfit because it is what was minimised. A gate that reads
    the misfit alone prefers a corrugated body to a shaped one, which is the thing the area
    term exists to stop, so selecting on a different functional from the one that was
    minimised undoes the fit. Both solvers measure through here, so that a body from either
    carries one number with one definition and the two can be compared.

    `c` is the reshaping of the hull, which one solver fits beside the code and the other
    carries in the code's own first block (field.split_code); a body that uses neither passes
    nothing.
    """
    def measure(code, geoms, c=None) -> tuple:
        g = [int(i) for i in geoms]
        if not g:
            return float("nan"), float("nan")
        out = op_x.curves_with_shape(support, code, radius, geoms=g, c=c)
        if out is None:
            return float("inf"), float("inf")
        cur, area, _ = out
        chi = whitened_misfit(cur.cpu(), data, scale, g, weight)
        if not np.isfinite(chi):
            return chi, float("nan")
        return chi, float(np.log(max(chi ** 2, 1e-300)) + area_weight * area)
    return measure


def posed_mesh(op: CodeOperator, support, code, radius: float, res: int):
    """(verts, faces) of the code's body in the challenge pose at the physical radius, as
    numpy arrays, or None when the body is degenerate."""
    m = op.mesh(support, code, res=res)
    if m is None:
        return None
    v = fit_to_cylinder(restore_constraints(
        CodeOperator.canonical(m[0], m[1]).cpu().numpy(), radius), radius)
    return v, m[1].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--channel", choices=("real", "blender"), default="blender",
                    help="the released curves to fit and, with it, the instrument fitted to "
                         "them")
    ap.add_argument("--calibration", default=None,
                    help=f"instrument file; by channel, {INSTRUMENT}")
    ap.add_argument("--support-from", default=None,
                    help="STL whose support is the base; by default the convex answer under "
                         "results/")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05,
                    help="first trial step, in RMS code units per coordinate; the line search "
                         "adapts it, so it is a starting scale and not a schedule")
    ap.add_argument("--l2", type=float, default=0.0,
                    help="ridge on the depths. Zero by default, and deliberately: a ridge "
                         "charges a coefficient by how large it is, so it prefers a shallow "
                         "answer to a deep one and therefore prefers this fit's own answer to "
                         "the body. The surface area does the work instead, and charges the "
                         "thing that is actually wrong with a bad answer")
    ap.add_argument("--area-weight", type=float, default=AREA_WEIGHT,
                    help="weight of the posed body's surface area in the objective, in "
                         f"inverse area of the canonical pose; measured window {AREA_WINDOW}")
    ap.add_argument("--volume-floor", type=float, default=0.50,
                    help="smallest volume an accepted body may have, as a fraction of the "
                         "convex answer's; 0 turns the floor off")
    ap.add_argument("--ckpt-every", type=int, default=25,
                    help="steps between checkpoints; 0 writes none")
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="seconds after which the descent stops between steps and writes the "
                         "best body it reached; 0 is no budget. The checkpoint is left "
                         "behind, so rerunning continues rather than starting over")
    ap.add_argument("--no-resume", action="store_true",
                    help="start from the convex answer even when a checkpoint is there")
    ap.add_argument("--phases", type=int, default=96)
    ap.add_argument("--operator-res", type=int, default=EXTRACT_RES,
                    help="extraction resolution of the descent's operator")
    ap.add_argument("--export-phases", type=int, default=96,
                    help="phases the written body is measured at. Finer than the descent's "
                         "grid, because the number that decides whether this body enters the "
                         "submission is a property of the body and not of the grid the "
                         "descent could afford")
    ap.add_argument("--export-res", type=int, default=EXPORT_RES,
                    help="extraction resolution of the written body and of the measurement "
                         "it is judged on")
    ap.add_argument("--hold-out-geoms", type=int, default=5,
                    help="cameras kept out of the fit; their misfit is the only honest test "
                         "of the body that was written, and scripts/select_answers.py "
                         "refuses a refinement that held out none")
    ap.add_argument("--every", type=int, default=25, help="steps between diagnostics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    add_render_flags(ap)
    a = ap.parse_args()
    render = render_from(a, "reconstruction")

    torch.manual_seed(a.seed)
    _enable_tf32()
    dev = a.device if torch.cuda.is_available() else "cpu"
    R = CYLINDER_R[a.model]
    inst = load_instrument(a.calibration or INSTRUMENT[a.channel], device=dev)
    psi = psi_grid(a.phases)
    op = CodeOperator(inst, psi, res=a.operator_res, config=render, device=dev)
    op_x = CodeOperator(inst, psi_grid(a.export_phases), res=a.export_res, config=render,
                        device=dev)

    sup_stl = a.support_from or str(answer_path(a.model))
    support = support_from_convex(sup_stl)

    d = load_inversion_curves(a.data_dir, a.model, m=a.phases, channel=a.channel)
    if set(d["files"]) != {"intensity", "binary"}:
        raise SystemExit(f"model {a.model} needs both {a.channel} curve files under "
                         f"{a.data_dir}; found {sorted(d['files'])}")
    data = curve_pairs(d["curves"])
    weight = curve_weight(d["mask"])
    if d["duplicate_columns"] or d["count_curves_refused"]:
        print(f"  {len(d['duplicate_columns'])} repeated columns and "
              f"{len(d['count_curves_refused'])} count curves dropped; "
              f"{int(weight.sum())} curves fitted", flush=True)
    scale = residual_scale(d, inst.eta).clamp_min(ETA_FLOOR)
    print(f"  curves: {d['channel']}   eta median {float(inst.eta.median()):.4f}   "
          f"median scale {float(scale.median()):.4f}", flush=True)
    present = measured_geometries(d["mask"])

    held = held_out_geoms(present, a.hold_out_geoms)
    fit_geoms = [g for g in present if g not in held]
    print(f"model {a.model}  R={R}  h from {Path(sup_stl).name}  "
          f"fit on {len(fit_geoms)} geometries" + (f", holding out {held}" if held else ""),
          flush=True)

    d_fit = data[fit_geoms].to(dev)
    s_fit = scale[fit_geoms].to(dev)

    w_fit = weight[fit_geoms].to(dev)
    n_obs = float(w_fit.sum()) * data.shape[-1]

    def cot_fn(cur):
        return 2.0 * w_fit[..., None] * (cur - d_fit) / (s_fit[..., None] ** 2) / n_obs

    if not (a.area_weight == 0.0 or AREA_WINDOW[0] <= a.area_weight <= AREA_WINDOW[1]):
        raise SystemExit(f"area weight {a.area_weight} is outside the measured window "
                         f"{AREA_WINDOW}; below it a smooth dent of the size the correction "
                         f"has is not charged and above it the body stops being the minimum")
    cap = depth_cap(support.numpy() if hasattr(support, "numpy") else np.asarray(support))
    floor = [0.0]          # set from the convex answer's own volume, once it is rendered
    # Why a trial was turned away, counted where the reason is known. A line search that ends
    # with no step is the one outcome that says nothing on its own: a body refused by the
    # volume floor, one that carves past its own centre, one the renderer will not render and
    # one that simply does not lower the objective are four different situations wanting four
    # different things done about them, and they all look like "no step" from outside.
    refused = {"depth cap": 0, "volume floor": 0, "no curves": 0, "objective": 0}

    def objective(z):
        """(log chi^2 + weight x area, chi) of the body `z` makes, or (inf, None) when it is
        not a body.

        The same functional CarveFit minimises, and for the same reason: a ridge on the
        coefficients charges how large they are, which prefers a shallow answer to a deep one,
        while the area charges the surface the body actually has. The volume and the area come
        from the mesh the extraction has already built, so they cost a per cent of the render
        rather than one of their own.

        The floor on the volume and the star-shaped bound on the depths are refusals here and
        not clips, because a clipped trial is a different trial and the line search would then
        be halving a step that no longer means what it takes it to mean.
        """
        if float(z[N_DIR:].max()) > cap:
            refused["depth cap"] += 1
            return float("inf"), None
        out = op.curves_with_shape(support, z, R, geoms=fit_geoms)
        if out is None:
            refused["no curves"] += 1
            return float("inf"), None
        cur, area, vol = out
        if vol < floor[0]:
            refused["volume floor"] += 1
            return float("inf"), None
        chi = whitened_misfit(cur.cpu(), data, scale, fit_geoms, weight)
        ridge = a.l2 * float((z[N_DIR:] ** 2).sum())
        return float(np.log(max(chi ** 2, 1e-300)) + a.area_weight * area + ridge), chi

    def area_gradient(z):
        """(area, d area / d z) of the canonically posed body, by autograd through the
        extraction.

        The extraction's vertices are differentiable in the code, so the area is too. The
        derivative holds the triangulation fixed while a real step also re-triangulates, so it
        is a direction rather than an exact derivative -- which is enough, because every trial
        step is accepted or rejected on the true objective above and not on this."""
        zz = z.detach().clone().requires_grad_(True)
        dh, g = zz[:N_DIR], zz[N_DIR:]
        # op.mesh sets the support on every call; this path builds the mesh itself, so it has
        # to do the same or the body is whatever the last call left behind
        op.body.set_support(support.to(dev))
        m = extract_mesh(lambda y: op.body(y, dh=dh, a=g), EXTRACT_EXTENT,
                         res=a.operator_res, device=dev, grad=True)
        if m is None:
            return None, None
        vv, ff = m
        vc = CodeOperator.canonical(vv, ff)
        p0, p1, p2 = (vc[ff[:, i]] for i in range(3))
        A = 0.5 * torch.cross(p1 - p0, p2 - p0, dim=1).norm(dim=1).sum()
        gz, = torch.autograd.grad(A, zz)
        return float(A.detach()), gz.detach()

    def held_misfit(operator, z):
        if not held:
            return float("nan")
        ch = operator.curves(support, z, R, geoms=held)
        return whitened_misfit(ch.cpu(), data, scale, held, weight) if ch is not None \
            else float("inf")

    d_x = load_inversion_curves(a.data_dir, a.model, m=a.export_phases, channel=a.channel)
    measure_at_export = export_measure(op_x, support, R, curve_pairs(d_x["curves"]),
                                       residual_scale(d_x, inst.eta).clamp_min(ETA_FLOOR),
                                       weight, a.area_weight)

    code = torch.zeros(CODE_DIM, device=dev)
    hist, best = [], None
    t0 = time.time()
    step = a.lr
    start_it, elapsed_before = 0, 0.0

    # The floor is a fraction of the convex answer's own volume, so it is set from the answer
    # rather than named: the cheapest surface in this representation is a hull shrink and the
    # misfit barely resists one, so without it the descent walks the volume past the body.
    out0 = op.curves_with_shape(support, torch.zeros(CODE_DIM, device=dev), R, geoms=fit_geoms)
    if out0 is None:
        raise SystemExit("the convex answer does not render; nothing to correct")
    vol0, area0 = float(out0[2]), float(out0[1])
    floor[0] = float(a.volume_floor) * vol0
    print(f"  convex answer: area {area0:.3f}, volume {vol0:.3f}; below {floor[0]:.3f} of "
          f"volume a body is refused, and the depth may reach {cap:.3f} before the body stops "
          f"containing its own centre", flush=True)

    ckpt_path = Path(f"{a.out}.map.ckpt") if a.out else None
    if ckpt_path is not None and ckpt_path.exists() and not a.no_resume:
        st = torch.load(ckpt_path, map_location=dev, weights_only=False)
        if st.get("keys") != {"model": a.model, "channel": a.channel,
                              "operator_res": a.operator_res, "phases": a.phases,
                              "area_weight": float(a.area_weight)}:
            print(f"  {ckpt_path} was written under other settings, so it is ignored",
                  flush=True)
        else:
            code = st["code"].to(dev)
            start_it, step, elapsed_before = int(st["step"]) + 1, float(st["lr"]), float(st["elapsed"])
            best, hist = st["best"], list(st["hist"])
            print(f"  resumed {ckpt_path} at step {start_it}", flush=True)

    def save_ckpt(it, lr_now):
        """Written under a temporary name and renamed, so a kill mid-write leaves the previous
        checkpoint rather than a truncated one. The descent is hundreds of steps and a body is
        worth an hour, so losing it to a wallclock is the failure this prevents."""
        if ckpt_path is None or not a.ckpt_every:
            return
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_name(ckpt_path.name + ".part")
        torch.save({"code": code.detach().cpu(), "step": it, "lr": lr_now, "best": best,
                    "hist": hist, "elapsed": elapsed_before + (time.time() - t0),
                    "keys": {"model": a.model, "channel": a.channel,
                             "operator_res": a.operator_res, "phases": a.phases,
                             "area_weight": float(a.area_weight)}}, tmp)
        tmp.replace(ckpt_path)

    J, chi = objective(code)
    time_limited = False
    zero = torch.zeros(CODE_DIM, device=dev)
    convex_fit, convex_fit_obj = measure_at_export(zero, fit_geoms)
    convex_held, convex_held_obj = measure_at_export(zero, held)
    print(f"  the convex answer, at the resolution the written body is measured at: "
          f"chi_fit {convex_fit:.3f}"
          + (f", chi_held {convex_held:.3f}" if held else ""), flush=True)
    for it in range(start_it, a.steps + 1):
        if it % a.every == 0 or it == a.steps:
            row = {"step": it, "chi_fit": chi, "objective": J, "step_size": step,
                   "seconds": round(time.time() - t0, 1)}
            m = posed_mesh(op, support, code, R, a.export_res)
            if m is not None:
                row["dice"] = truth_dice(m[0], m[1], a.model, a.data_dir)
                row["convexity"] = convexity(m[0], m[1])
            if held:
                row["chi_held"] = held_misfit(op, code)
            # Compared on the functional being minimised and never on the misfit alone.
            # Between two bodies the misfit prefers the rougher one, which is the comparison
            # notes/objective.md says may never be made.
            if best is None or J < best["objective"]:
                best = {**row, "code": code.detach().clone()}
            hist.append(row)
            print(f"  step {it:>4}  chi_fit {chi:7.3f}"
                  + (f"  chi_held {row['chi_held']:7.3f}" if held else "")
                  + f"  dice {row.get('dice', float('nan')):.4f}"
                    f"  convexity {row.get('convexity', float('nan')):.3f}"
                    f"  step {step:.4f}  [{row['seconds']:.0f}s]", flush=True)
        if it == a.steps or (chi is not None and chi <= TARGET_SIGMA):
            break
        if a.time_budget and elapsed_before + (time.time() - t0) > a.time_budget:
            # The best body so far is already tracked and is what gets written, so stopping
            # here costs the steps not taken and nothing that was found.
            print(f"  the time budget stopped the descent at step {it}; the body written is "
                  f"the best of the {len(hist)} checkpoints reached", flush=True)
            save_ckpt(it, step)
            time_limited = True
            break

        _, grad = op.adjoint(support, code, R, cot_fn, geoms=fit_geoms)
        if grad is None:
            print("  the body has no curves; stopping", flush=True)
            break
        # d(log chi^2)/dz is the misfit gradient over chi^2: the adjoint returns the
        # derivative of chi^2 itself, and the log is what makes the balance against the area
        # scale free, so that a body whose misfit is ten times smaller is not thereby allowed
        # ten times the surface.
        grad = grad.detach() / max(chi ** 2, 1e-12)
        if a.area_weight:
            _, g_area = area_gradient(code)
            if g_area is not None:
                grad = grad + a.area_weight * g_area
        grad[N_DIR:] += 2.0 * a.l2 * code[N_DIR:]
        # one unit-RMS direction per block, so that neither the few dh coordinates nor the
        # many amplitudes set the step for the other
        direction = torch.zeros_like(grad)
        for sl in (slice(0, N_DIR), slice(N_DIR, None)):
            r = float(grad[sl].pow(2).mean().sqrt())
            if np.isfinite(r) and r > 0:
                direction[sl] = -grad[sl] / r
        if float(direction.abs().max()) == 0.0:
            print("  the gradient vanished; stopping", flush=True)
            break

        moved = False
        before = dict(refused)
        for _ in range(MAX_HALVINGS):
            trial = code + step * direction
            J_t, chi_t = objective(trial)
            if J_t < J:
                code, J, chi = trial, J_t, chi_t
                step *= STEP_GROW
                moved = True
                break
            if np.isfinite(J_t):
                refused["objective"] += 1
            step *= 0.5
        if a.ckpt_every and (it + 1) % a.ckpt_every == 0:
            save_ckpt(it, step)
        if not moved:
            here = {k: refused[k] - before[k] for k in refused}
            why = ", ".join(f"{n} by the {k}" for k, n in here.items() if n) or "none"
            print(f"  no step down to {step:.2e} lowers the objective; converged at step "
                  f"{it}. Of the {MAX_HALVINGS} trials: {why} refused.", flush=True)
            if it == start_it:
                # Nothing was ever accepted, so the body about to be written is the convex
                # answer unchanged. Said plainly, because a run that wrote its input and
                # exited zero otherwise reads as a run that worked.
                print("  !!! no trial was accepted at all, so the body written is the convex "
                      "answer itself and this run found no correction. The refusals above "
                      "say which bound the first step met.", flush=True)
            break

    if a.out:
        z = best["code"] if best is not None else code.detach()
        # The code is written before anything is asked of the mesh. A deeply carved body can
        # pinch and be refused at the export guard, and losing the coefficients with it means
        # descending again rather than extracting again.
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(Path(a.out).with_suffix(".code.npz"), code=z.cpu().numpy(),
                 support=support.cpu().numpy())
        m = posed_mesh(op, support, z, R, a.export_res)
        if m is None:
            raise SystemExit("the final body is degenerate; the code is in the .code.npz "
                             "beside it")
        v, f = m
        fit_x, fit_x_obj = measure_at_export(z, fit_geoms)
        held_x, held_x_obj = measure_at_export(z, held)
        export_error = None
        try:
            rep = export_stl(a.out, v, f)
        except ValueError as exc:
            export_error, rep = str(exc), {"refused": str(exc)}
            print(f"  !!! {exc}", flush=True)
        meta = {"model": a.model, "channel": d["channel"], "radius": R, "steps": a.steps,
                "lr": a.lr, "l2": a.l2, "phases": a.phases, "operator_res": a.operator_res,
                "export_res": a.export_res, "export_phases": a.export_phases,
                "held_out": held, "fit_geoms": fit_geoms, "history": hist, "export": rep,
                "curves_fitted": int(weight.sum()),
                "chi_fit_convex": convex_fit, "chi_held_convex": convex_held,
                "chi_fit_export": fit_x, "chi_held_export": held_x,
                "objective_fit_convex": convex_fit_obj,
                "objective_held_convex": convex_held_obj,
                "objective_fit_export": fit_x_obj, "objective_held_export": held_x_obj,
                "final_dice": truth_dice(v, f, a.model, a.data_dir),
                "convex_dice": convex_dice(sup_stl, a.model, a.data_dir),
                "final_convexity": convexity(v, f),
                "area_weight": a.area_weight, "volume_floor": a.volume_floor,
                "time_limited": time_limited, "steps_taken": len(hist),
                "trials_refused": refused,
                "export_refused": export_error}
        Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2,
                                                               default=json_default))
        if export_error is not None:
            # A distinct status, because this is not the same failure as a run that broke:
            # the descent finished, its numbers are in the JSON, and only the extraction is
            # unusable. A runbook can carry on past it, and the wiring test can count it as a
            # stage that ran.
            print(f"model {a.model}: the descent finished and its code is written beside "
                  f"this, but the extracted mesh is not a closed solid and was refused. "
                  f"Re-extract from {Path(a.out).with_suffix('.code.npz')} rather than "
                  f"descending again.", flush=True)
            raise SystemExit(EXPORT_REFUSED)
        print(f"  wrote {a.out}  ({rep['faces']} faces, volume {rep['volume']:.3f}); "
              f"at export resolution chi_fit {fit_x:.3f}"
              + (f", chi_held {held_x:.3f} against the convex answer's {convex_held:.3f}"
                 if held else ""), flush=True)


if __name__ == "__main__":
    main()
