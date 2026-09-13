#!/usr/bin/env python3
"""Recover one body from its curves by fitting the hull's reshaping and the carve together.

    python scripts/reconstruct_gn.py --model 3 --channel blender --hold-out-geoms 5 \
        --out results/gn/Asteroid03.stl

The convex stage's answer supplies the base support h. It is not the body's hull: a convex
inversion of a non-convex body returns the convex body whose own shadowing best imitates the
concavities, which is larger. Of the three public bodies the answer exceeds the body's own
hull on the one that has concavities and falls short of it on the two that do not, which is
the mechanism and not a bias of the network. The fit therefore does not treat h as fixed and
does not treat the hull as something to be corrected before carving. The correction is one
function on the sphere -- a displacement of the core's surface, inward where the body has a
concavity and outward where the convex answer overshot -- and the reshaping of the hull is
simply its first three degrees.

The step is damped Gauss-Newton with a secant Jacobian, coarse to fine in the angular degree of
that function (hac26.solvers.gauss_newton). Nothing differentiates the renderer. The ladder
stops well short of the scale at which a displacement stops costing surface area, because there
the penalty below cannot charge it and the fit would spend its whole budget buying misfit with
texture. The fit is started several times, once from the convex answer itself and otherwise
from a shrunken hull carved by a single spherical cap drawn from a grid over where the cap is,
how wide it is and how deep, because the first linearisation from the convex answer is taken
where neither half of the correction is yet doing anything and the step there goes into the
carve alone, which is the convex inversion's own mistake made once more.

What is minimised carries the body's surface area beside its misfit, because the misfit of a
rendered body reports how finely its surface is resolved almost as strongly as it reports
whether the shape is right. The run is therefore two phases: to convergence under the
penalised objective, then a polish on the misfit alone with the volume still held, which
recovers the misfit without giving the shape back.

A body whose convex answer already explains its curves is left alone. There is no concavity
there for the correction to find, and an objective that charges surface will trade overlap for
a misfit it does not need; that failure raises the misfit's opinion of the body while lowering
its overlap, so it cannot be caught afterwards and is refused before the fit instead.

--hold-out-geoms keeps cameras out of the fit and reports the written body's misfit and
objective on them beside the convex answer's on the same cameras. That pair is what
scripts/select_answers.py reads, and it is the only test of whether a shape was recovered
rather than curves fitted. On a public model the overlap with the released shape is reported
as well.

A fit whose extracted mesh the export guard refuses exits 3 rather than 1. The fit
finished in that case and its numbers are written; only the mesh is unusable, and the
coefficients beside it (.fit.npz) extract again without repeating the fit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.conventions import CYLINDER_R, psi_grid                       # noqa: E402
from hac26.data_io import held_out_geoms, load_inversion_curves   # noqa: E402
from hac26.field import (CODE_DIM, EXTRACT_RES, N_DIR, N_NODES, N_RADIAL,  # noqa: E402
                         DepthSphere, depth_cap, node_kernel)
from hac26.recon import fit_to_cylinder                                  # noqa: E402
from hac26.solvers.gauss_newton import (AREA_WEIGHT, AREA_WINDOW,    # noqa: E402
                                        DEFAULT_STAGES, N_STARTS, POLISH_STAGES,
                                        SCREEN_STAGES, STEP_C, STEP_G, TARGET_SIGMA,
                                        VOLUME_TRUST, CarveFit, Stage, conjunction_start)
from hac26.solvers.operator import CodeOperator                          # noqa: E402
from hac26.solvers.output import export_stl, restore_constraints         # noqa: E402
from reconstruct import answer_path                                      # noqa: E402
from reconstruct_lpd import (curve_pairs, curve_weight, json_default,   # noqa: E402
                             measured_geometries, residual_scale,
                             support_from_convex)
from calibrate import ETA_FLOOR                                          # noqa: E402
from reconstruct_map import (EXPORT_REFUSED, EXPORT_RES, convex_dice,   # noqa: E402
                             convexity, export_measure, truth_dice)
from train_lpd import (INSTRUMENT, _enable_tf32, _hms, add_render_flags,   # noqa: E402
                       load_instrument, render_from, render_tag)

RESTARTS = 9               # starts given the coarse screen: the convex answer and the best
                           # eight of the designed grid, which are now chosen by rendering
                           # every start once and ranking rather than by taking the first eight
                           # indices. The screen answers a question a render cannot -- whether
                           # a start descends -- and costs about a hundred renders against the
                           # ladder's thousands, so a handful of them is what is affordable
                           # here and the ranking above is what decides which handful. Measured
                           # also, the random draws this grid replaces were worth nothing at
                           # all, so it is the spread and not the count that earns its place.
RESTART_KEEP = 2           # starts carried to the end of the ladder
VOLUME_FLOOR = 0.50        # smallest volume an accepted body may have, as a fraction of the
                           # convex answer's. The cheapest surface area in this representation
                           # is a hull shrink and the misfit barely resists one, so without a
                           # floor the fit walks the volume down past the body and loses more
                           # overlap than the carve gains; notes/objective.md measures both
                           # runs. The value is calibrated on the one released non-convex body,
                           # which sits comfortably above it, and it inherits exactly the
                           # weakness --min-convex-sigmas has: what would make it principled is
                           # the distribution of that ratio over the shape library, which is the
                           # same corpus a learned acceptance gate would need.
MIN_CONVEX_SIGMAS = 6.5    # how badly the convex answer has to fit before a body is expected
                           # to have concavity to find, in model errors. A body under this is
                           # flagged in the metadata and corrected anyway; the flag is a warning
                           # to the selection, not a refusal. Measured on the most nearly convex
                           # public body, correcting where there is nothing to find costs a
                           # sixth of the overlap while improving the misfit, so no gate that
                           # reads a misfit afterwards catches it -- but that body is one whose
                           # convex inversion had already recovered it, and reading a secret
                           # body's fate from it is the inference this project does not make.
                           # scripts/select_answers.py decides instead, on geometries held out
                           # of the body's own fit.
                           #
                           # The unit is the calibrated model error, and that is a unit which
                           # moves: the released curves carry a measured noise two orders below
                           # it, so the residual is divided by very nearly the calibration's own
                           # eta and every sigma reported here scales inversely with it. Taking
                           # the sawed-off cube out of the calibration lowered eta and lifted
                           # both of the bodies this threshold was placed between, and the value
                           # above is the band that separates them under the calibration it was
                           # measured on and under the narrower one in use. notes/objective.md
                           # records the band and the one measurement that would replace it.


def load_start_code(path: str, index: int | None) -> tuple:
    """(support, g, label) from a reconstruct_lpd .codes.npz, to start the fit at that body.

    The file holds every draw's code and the support they were all decoded against. The
    solver's g is the carving on N_NODES directions and sits in the last N_NODES entries of a
    code (see render(): `code[-N_NODES:] = g`), so the flow's carving transfers verbatim. The
    leading N_DIR entries are the hull correction, which does not transfer: it is already in
    the support this file carries.

    `index` picks the draw. Left out, the sibling .json's chosen candidate is used, so the
    default is the body the flow actually published rather than an arbitrary draw.
    """
    z = np.load(path)
    codes, support = z["codes"], z["support"]
    if index is None:
        side = Path(path).with_suffix("").with_suffix(".json")
        index = 0
        if side.exists():
            try:
                index = int(json.loads(side.read_text()).get("candidate", 0))
            except (ValueError, TypeError, json.JSONDecodeError):
                index = 0
    if not 0 <= index < len(codes):
        raise SystemExit(f"{path} holds {len(codes)} draws; draw {index} was asked for")
    code = np.asarray(codes[index], dtype=float)
    dh, g = code[:N_DIR], code[-N_NODES:]
    print(f"  starting from draw {index} of {path}: carving reaches {g.max():.3f}, "
          f"hull correction {dh.min():+.3f}..{dh.max():+.3f}, "
          f"base support {support.min():.3f}-{support.max():.3f}", flush=True)
    return (torch.tensor(support, dtype=torch.float32), dh, g, f"{path}#draw{index}")


def curve_index(weight: torch.Tensor, geoms) -> tuple:
    """(geometries, per-geometry curve mask) of the curves a fit uses, and the flat index of
    those curves in the operator's (G, 2, P) output."""
    g = [int(i) for i in geoms]
    w = weight[g] > 0
    return g, w


def flat_curves(cur: torch.Tensor, keep: torch.Tensor) -> np.ndarray:
    """The kept curves of an operator result (G, 2, P), flattened in a fixed order."""
    return cur.detach().cpu().numpy()[keep.numpy()].ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--channel", choices=("real", "blender"), default="blender")
    ap.add_argument("--calibration", default=None,
                    help=f"instrument file; by channel, {INSTRUMENT}")
    ap.add_argument("--support-from", default=None,
                    help="STL whose support is the base; by default the convex answer")
    ap.add_argument("--code-from", default=None,
                    help="a reconstruct_lpd <out>.codes.npz to START FROM, rather than from "
                         "the convex answer. --support-from takes only the SUPPORT of a mesh, "
                         "which is a max over vertices and therefore its convex hull, so it "
                         "inherits a flow body's hull and throws away its carving. The "
                         "carving is the last N_NODES entries of the flow's code -- the same "
                         "slots this solver's own g occupies in render() -- so it can be "
                         "carried over exactly. The base support is taken from the file too, "
                         "which is the support the flow actually ran at rather than one "
                         "re-derived from its meshed output.")
    ap.add_argument("--code-index", type=int, default=None,
                    help="which draw in --code-from to start from; by default the one that "
                         "file's sibling .json chose as the answer, else draw 0")
    ap.add_argument("--phases", type=int, default=48)
    ap.add_argument("--export-phases", type=int, default=96,
                    help="phases the written body's misfits are measured at")
    ap.add_argument("--operator-res", type=int, default=EXTRACT_RES,
                    help="extraction resolution of the fit's operator; it has to resolve the "
                         "angular scale of the depth field")
    ap.add_argument("--export-res", type=int, default=EXPORT_RES)
    ap.add_argument("--hold-out-geoms", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--screen-starts", type=int, default=N_STARTS,
                    help="designed starts rendered once and ranked before the coarse screen; "
                         "the whole grid by default, since a render each is a small part of "
                         "one ladder")
    ap.add_argument("--restart-keep", type=int, default=RESTART_KEEP,
                    help="screened starts carried to the end of the ladder. This is what a "
                         "run costs, the ladder being far more renders than the screening, "
                         "so it is the dial for a run against a deadline")
    ap.add_argument("--area-weight", type=float, default=AREA_WEIGHT,
                    help="weight of the posed body's surface area in the objective, in "
                         f"inverse area of the canonical pose; measured window {AREA_WINDOW}, "
                         "and zero minimises the misfit alone")
    ap.add_argument("--volume-trust", type=float, default=VOLUME_TRUST,
                    help="largest fractional change of volume an accepted step may make")
    ap.add_argument("--volume-floor", type=float, default=VOLUME_FLOOR,
                    help="smallest volume an accepted body may have, as a fraction of the "
                         "convex answer's; 0 turns the floor off")
    ap.add_argument("--min-convex-sigmas", type=float, default=MIN_CONVEX_SIGMAS,
                    help="flag a body whose convex answer already explains its curves to "
                         "fewer than this many model errors; it is corrected either way and "
                         "the selection decides. The unit is the calibration's own eta and "
                         "moves with it, so a refitted instrument needs the threshold read "
                         "again")
    ap.add_argument("--step-g", type=float, default=STEP_G,
                    help="secant step of a carve coordinate, in body units of depth")
    ap.add_argument("--step-c", type=float, default=STEP_C,
                    help="secant step of a reshaping coefficient, in body units")
    ap.add_argument("--max-degree", type=int, default=0,
                    help="drop every stage of every ladder above this spherical-harmonic "
                         "degree, and lower a ladder that would then be empty to it; 0 "
                         "leaves the designed ladder. Lowering the top degree gives a "
                         "smoother correction at a fraction of the cost, because a "
                         "degree-L stage costs (L+1)^2 - 9 renders an iteration, and it is "
                         "the answer to a run that is too slow to finish -- the bound the "
                         "ladder's top degree carries is an upper one (notes/objective.md), "
                         "so a lower ceiling is safe where a higher one is not")
    ap.add_argument("--max-stage-iters", type=int, default=0,
                    help="cap every stage of every ladder at this many iterations; 0 leaves "
                         "each at its designed count. A run capped here is bounded by the cap "
                         "and not by the curves, which the written body records as "
                         "budget_limited; the point of the flag is a check that the whole path "
                         "executes, or an answer by a fixed time")
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="seconds after which the fit stops between units of work and writes "
                         "the best finished body; 0 is no budget. A run that stops this way "
                         "records time_limited and leaves its checkpoint, so rerunning it "
                         "continues rather than starting over. It is a floor and not a cap: "
                         "the check falls between units, and the longest unit is one "
                         "iteration of the ladder's top stage, which is (L+1)^2 - 9 renders, "
                         "so the worst case is the budget plus that. --max-degree is what "
                         "bounds it")
    ap.add_argument("--ckpt-file", default=None,
                    help="resumable checkpoint; by default <--out>.gn.ckpt. The fit is a "
                         "sweep of independent starts, so what is checkpointed is the "
                         "finished ones: a killed run loses at most the start it was on")
    ap.add_argument("--no-resume", action="store_true",
                    help="start the sweep from the beginning even when a checkpoint for "
                         "these settings is there")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    add_render_flags(ap)
    a = ap.parse_args()
    # named for the discretisation and not `render`, which is the closure below that the
    # solver calls to render a trial
    render_cfg = render_from(a, "reconstruction")
    # The first nine coordinates of the field are the reshaping, which the solver fits
    # separately, so a stage's coordinates are the degrees above them: below degree three a
    # stage has none and the ladder would fit nothing.
    if a.max_degree and a.max_degree < 3:
        raise SystemExit(f"--max-degree {a.max_degree} leaves a stage with no coordinates: "
                         f"the first nine belong to the reshaping, so three is the lowest "
                         f"degree that carves at all")

    torch.manual_seed(a.seed)
    _enable_tf32()
    dev = a.device if torch.cuda.is_available() else "cpu"
    R = CYLINDER_R[a.model]
    inst = load_instrument(a.calibration or INSTRUMENT[a.channel], device=dev)
    op = CodeOperator(inst, psi_grid(a.phases), res=a.operator_res, config=render_cfg,
                      device=dev)

    sup_stl = a.support_from or str(answer_path(a.model))
    start_dh = start_g = None
    if a.code_from:
        support, start_dh, start_g, sup_stl = load_start_code(a.code_from, a.code_index)
    else:
        support = support_from_convex(sup_stl)

    d = load_inversion_curves(a.data_dir, a.model, m=a.phases, channel=a.channel)
    if set(d["files"]) != {"intensity", "binary"}:
        raise SystemExit(f"model {a.model} needs both {a.channel} curve files under "
                         f"{a.data_dir}; found {sorted(d['files'])}")
    data = curve_pairs(d["curves"])                                  # (N_CAMS, 2, P)
    weight = curve_weight(d["mask"])                                 # (N_CAMS, 2)
    scale = residual_scale(d, inst.eta).clamp_min(ETA_FLOOR)         # (N_CAMS, 2)
    present = measured_geometries(d["mask"])
    held = held_out_geoms(present, a.hold_out_geoms)
    fit_geoms = [g for g in present if g not in held]
    print(f"model {a.model}  R={R}  channel {d['channel']}  h from {Path(sup_stl).name}\n"
          f"  {int(weight.sum())} curves of {len(present)} geometries; fitting on "
          f"{len(fit_geoms)}" + (f", holding out {held}" if held else "")
          + f"; model error median {float(scale.median()):.4f}", flush=True)
    if d["duplicate_columns"] or d["count_curves_refused"]:
        print(f"  {len(d['duplicate_columns'])} repeated columns and "
              f"{len(d['count_curves_refused'])} count curves are not measurements and were "
              f"dropped", flush=True)

    fit_g, keep_fit = curve_index(weight, fit_geoms)
    data_fit = flat_curves(data[fit_g], keep_fit)
    scale_fit = np.repeat(scale[fit_g].numpy()[keep_fit.numpy()], data.shape[-1])
    # The code every trial is built on. render() writes only the carving block, so anything
    # the leading N_DIR entries should carry has to be put here. The support a flow body was
    # decoded against is its CONVEX start, not its hull: the hull correction lives in dh, and
    # leaving dh at zero would cut a carve sized for the corrected hull into the uncorrected
    # one. That shrinks the body onto the volume floor and every trial step is then refused.
    zero_code = torch.zeros(CODE_DIM, device=dev)
    if start_dh is not None:
        zero_code[:N_DIR] = torch.tensor(start_dh, dtype=torch.float32, device=dev)
    floor = [0.0]           # set from the convex answer's own volume, once it is rendered
    refused_volume = [0]    # trials the floor turned away, counted here because this is where
                            # the reason is known: inside the solver a body under the floor is
                            # indistinguishable from one the forward model will not render

    def render(c, g):
        code = zero_code.clone()
        code[-N_NODES:] = torch.tensor(np.asarray(g), dtype=torch.float32, device=dev)
        out = op.curves_with_shape(support, code, R, geoms=fit_g,
                                   c=torch.tensor(np.asarray(c), dtype=torch.float32,
                                                  device=dev))
        if out is None:
            return None
        cur, area, vol = out
        # A body below the floor is refused here rather than in the solver, because the solver
        # already has a path for a body the forward model will not render and this is the same
        # kind of refusal: the line search sees a trial that did not come back and shortens.
        if vol < floor[0]:
            refused_volume[0] += 1
            return None
        return flat_curves(cur, keep_fit), area, vol

    nodes = DepthSphere(N_NODES).u.numpy()
    kernel = node_kernel()
    cap = depth_cap(support.numpy())

    def new_fit(seed):
        return CarveFit(render, data_fit, scale_fit, kernel, nodes,
                        n_radial=N_RADIAL, area_weight=a.area_weight,
                        volume_trust=a.volume_trust, depth_cap=cap,
                        step_g=a.step_g, step_c=a.step_c, seed=seed)

    zeros = (np.zeros(N_RADIAL), np.zeros(N_NODES))
    gate = new_fit(a.seed)
    r0, area0, vol0 = gate._render(*zeros)
    if r0 is None:
        raise SystemExit("the convex answer does not render; nothing to correct")
    floor[0] = float(a.volume_floor) * vol0
    convex_sigmas = float(np.linalg.norm(r0))
    print(f"  the depth may reach {cap:.3f} body units before the body stops containing its "
          f"own centre; below {floor[0]:.3f} of volume a body is refused", flush=True)
    print(f"  the convex answer explains the fitted curves to {convex_sigmas:.2f} model "
          f"errors; area {area0:.3f}, volume {vol0:.3f}", flush=True)
    # A body whose convex answer already fits is flagged and still corrected. The measurement
    # behind the threshold is model 1, whose released body holds 1.009 of its convex answer's
    # volume, so it is a body the convex inversion had already got right; deciding a secret
    # body from it is the inference this project does not make. The correction is run, the flag
    # is written into the metadata, and scripts/select_answers.py chooses between the two on
    # geometries held out of the fit, which is evidence about this body rather than about
    # model 1. Refusing here would instead settle the question before any evidence exists.
    convex_explains = convex_sigmas < a.min_convex_sigmas
    if convex_explains:
        print(f"  WARNING: the convex answer already explains the fitted curves to "
              f"{convex_sigmas:.2f} model errors, under the {a.min_convex_sigmas:g} a body "
              f"normally has concavity to find at. The correction is run and flagged; the "
              f"selection decides on the held-out geometries.", flush=True)

    # The grid is swept in two tiers, because the two questions cost differently. Ranking a
    # start needs the number the fit minimises at that start, and that is one render; telling a
    # start that descends from one that does not needs the coarse stage, and that is about a
    # hundred. So every designed start is rendered once and ranked, and only the best few are
    # given the coarse screen below. The whole grid at one render each costs about what
    # screening nine starts used to, which is what makes a grid of this size affordable.
    t0 = time.time()
    # The sweep is a list of independent units -- one render per designed start, then one
    # coarse screen per carried start, then one full ladder per kept start -- so what a
    # checkpoint has to hold is which of them are finished. There is no optimiser state to
    # restore and nothing partial to reconstruct: a killed run loses the unit it was in and
    # nothing else, which is what makes this safe to rely on rather than only cheap.
    ckpt_path = Path(a.ckpt_file or f"{a.out}.gn.ckpt") if a.out else None
    keys = {"model": a.model, "channel": a.channel, "support": Path(sup_stl).name,
            "phases": a.phases, "operator_res": a.operator_res,
            "render": render_tag(render_cfg),
            "hold_out_geoms": a.hold_out_geoms, "area_weight": a.area_weight,
            "volume_trust": a.volume_trust, "volume_floor": a.volume_floor,
            "screen_starts": int(min(a.screen_starts, N_STARTS)), "restarts": a.restarts,
            "restart_keep": a.restart_keep, "max_stage_iters": a.max_stage_iters,
            "max_degree": a.max_degree,
            "step_g": a.step_g, "step_c": a.step_c, "seed": a.seed}
    st = {"pool": [], "next_start": 0, "skipped_depth": 0, "skipped_floor": 0,
          "screened": [], "next_screen": 0, "done": [], "renders": 0, "refused_volume": 0,
          "elapsed": 0.0}
    if ckpt_path is not None and ckpt_path.exists() and not a.no_resume:
        prev = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if prev.get("keys") != keys:
            print(f"  {ckpt_path} was written under other settings, so it is ignored and the "
                  f"sweep starts from the beginning", flush=True)
        else:
            st = prev["state"]
            print(f"  resumed {ckpt_path}: {st['next_start']} starts ranked, "
                  f"{st['next_screen']} screened, {len(st['done'])} ladders finished "
                  f"({_hms(st['elapsed'])} of fitting before this run)", flush=True)

    def save() -> None:
        """Written under a temporary name and renamed, so a kill mid-write leaves the
        previous checkpoint rather than a truncated one."""
        if ckpt_path is None:
            return
        st["renders"] = gate.renders
        st["refused_volume"] = refused_volume[0]
        st["elapsed"] = elapsed_before + (time.time() - t0)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_name(ckpt_path.name + ".part")
        torch.save({"keys": keys, "state": st}, tmp)
        tmp.replace(ckpt_path)

    elapsed_before = float(st["elapsed"])
    # added to, not replaced: this run has already rendered the convex answer once, and the
    # count the checkpoint carries is what every run before it spent
    gate.renders += int(st["renders"])
    refused_volume[0] = int(st["refused_volume"])

    def out_of_time() -> bool:
        return bool(a.time_budget) and (elapsed_before + time.time() - t0) > a.time_budget

    ranked_from = gate.renders
    pool = st["pool"]
    limit = int(min(a.screen_starts, N_STARTS))
    for i in range(st["next_start"], limit):
        c0, g0, rec = conjunction_start(nodes, i, n_radial=N_RADIAL)
        # A carve deeper than a body has room for is not a body, so it is passed over rather
        # than shortened: a start clipped to the bound is a different start from the one the
        # grid means, and the grid would then no longer be a spread.
        if float(g0.max()) > cap:
            st["skipped_depth"] += 1
        else:
            r_i, area_i, _ = gate._render(c0, g0)
            if r_i is None:
                st["skipped_floor"] += 1
            else:
                pool.append({"objective": gate.objective(r_i, area_i), "c": c0, "g": g0,
                             "recipe": {"start": rec["kind"], **rec}})
        st["next_start"] = i + 1
        save()
        if (i + 1) % 20 == 0 or i + 1 == limit:
            print(f"    ranked {i + 1}/{limit} designed starts [{time.time() - t0:.0f}s]",
                  flush=True)
        if out_of_time():
            print(f"    the time budget stopped the ranking at {i + 1} of {limit} starts",
                  flush=True)
            break
    skipped_depth, skipped_floor = st["skipped_depth"], st["skipped_floor"]
    if not pool and start_g is None:
        raise SystemExit("no designed start is a body this model has room for; there is "
                         "nothing to rank and nothing to fit")
    pool.sort(key=lambda q: q["objective"])
    take = max(0, a.restarts - 1)
    # The first start is the convex answer unless a body was handed in: then it is that
    # body's own carving at c = 0, because the hull it came with is already the base support.
    first = (np.zeros(N_RADIAL), np.zeros(N_NODES) if start_g is None else start_g)
    starts = [first] + [(q["c"], q["g"]) for q in pool[:take]]
    recipes = ([{"start": "convex answer" if start_g is None else f"body from {a.code_from}"}]
               + [q["recipe"] for q in pool[:take]])
    kept_kinds = Counter(q["recipe"]["kind"] for q in pool[:take])
    print(f"  ranked {len(pool)} designed starts on one render each "
          f"({gate.renders - ranked_from} renders, {time.time() - t0:.0f}s); carrying "
          + (", ".join(f"{n} {k}" for k, n in sorted(kept_kinds.items())) or "none")
          + " to the coarse screen"
          + (f"; {skipped_depth} carve deeper than this body's own centre allows" if
             skipped_depth else "")
          + (f"; {skipped_floor} fall under the volume floor" if skipped_floor else ""),
          flush=True)
    def last(hist, key):
        rows = [h for h in hist if key in h]
        return rows[-1][key] if rows else float("inf")

    def capped(stages):
        """`stages` with every iteration count held at --max-stage-iters and every degree at
        --max-degree. A ladder left empty by the degree bound becomes one stage at that
        degree, with the first stage's iteration count, so that a bound below the designed
        ladder still fits something. Both caps are applied here rather than inside the solver
        so that the designed ladder stays the one thing a reader of gauss_newton.py sees, and
        a shortened run is visibly a shortened run."""
        if a.max_degree:
            kept = tuple(sg for sg in stages if sg.degree and sg.degree <= a.max_degree)
            stages = kept or (Stage(a.max_degree, stages[0].n_dirs, stages[0].iters),)
        if not a.max_stage_iters:
            return stages
        return tuple(Stage(st.degree, st.n_dirs, min(st.iters, a.max_stage_iters))
                     for st in stages)

    screen_stages, ladder_stages, polish_stages = (capped(SCREEN_STAGES),
                                                   capped(DEFAULT_STAGES),
                                                   capped(POLISH_STAGES))

    def penalised(chi, area) -> float:
        """The functional a finished body is compared under, here and downstream. It is not
        the one the polish minimises, which is why both sides of the polish are scored with
        it."""
        return float(np.log(max(chi ** 2, 1e-300)) + a.area_weight * area)

    def still_descending(hist) -> bool:
        """True when the ladder's last iteration still took a step, so the fit stopped on its
        iteration count rather than on the data. A capped result is not a converged one and
        the written body has to say which it is."""
        rows = [h for h in hist if "accepted" in h]
        return bool(rows) and bool(rows[-1]["accepted"])

    seen_floor = [0]

    def show(row):
        """One line per iteration, and on an iteration that took no step, how many of its
        trials the volume floor turned away.

        A stage that finds no step says nothing on its own about why, and the two cases want
        different things done about them: a fit that has run out of shape to find is
        finished, while one whose every trial is refused by the floor is pinned against a
        bound and will be refused at every degree above this one too, at a Jacobian each.
        """
        turned = refused_volume[0] - seen_floor[0]
        seen_floor[0] = refused_volume[0]
        print(f"    {row['stage']:>14}  it {row['iteration']}  chi {row['chi']:.4f}"
              f"  area {row['area']:.3f}  volume {row['volume']:.3f}"
              f"  {'step' if row['accepted'] else 'no step'}"
              + ("" if row["accepted"] or not turned else
                 f" ({turned} trials refused by the volume floor)")
              + f"  [{time.time()-t0:.0f}s]", flush=True)

    screened = st["screened"]
    for i in range(st["next_screen"], len(starts)):
        c0, g0 = starts[i]
        f = new_fit(a.seed + i)
        c, g, hist = f.run(c0, g0, stages=screen_stages, target=TARGET_SIGMA)
        screened.append({"i": i, "objective": last(hist, "objective"),
                         "chi": last(hist, "chi"), "c": c, "g": g, "renders": f.renders})
        st["next_screen"] = i + 1
        save()
        print(f"  start {i} ({recipes[i]['start']}): objective "
              f"{screened[-1]['objective']:.4f}, chi {screened[-1]['chi']:.4f} after "
              f"{screen_stages[0].iters} coarse steps, {f.renders} renders "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if out_of_time():
            print(f"    the time budget stopped the coarse screen at {i + 1} of "
                  f"{len(starts)} starts", flush=True)
            break
    # Starts are compared on what is being minimised. A start that has bought misfit with
    # surface is not ahead of one that has not.
    screened.sort(key=lambda q: q["objective"])

    # Which starts are finished, by their index and not by how many there are: a resumed run
    # screens more starts, which re-sorts the ranking, and a start that has moved would
    # otherwise be fitted twice while another was skipped. A start fitted under an earlier
    # ranking still counts -- it was fitted to the end and scored on the same functional, so
    # it is a candidate whether or not it would be carried today.
    done = st["done"]
    fitted = {int(q["start"]) for q in done}
    keep = [q for q in screened[:max(1, a.restart_keep)] if int(q["i"]) not in fitted]
    for s in keep:
        f = new_fit(a.seed + s["i"] + 100)
        floor_before = refused_volume[0]
        cu, gu, hist = f.run(s["c"], s["g"], stages=ladder_stages, target=TARGET_SIGMA,
                             log=show)
        chi_u, area_u = last(hist, "chi"), last(hist, "area")
        obj_u = penalised(chi_u, area_u)
        # The penalty has put the shape where it goes and left the misfit above where the data
        # alone would put it. Minimising the misfit alone from there, with the trust regions
        # still on, is meant to recover the misfit without giving the shape back. It is
        # minimising a functional that is not the one this body is judged by, here or in
        # select_answers.py, so the body it starts from is kept and the two are compared under
        # the penalty. Without that, a polish that buys misfit with surface is written out and
        # nothing downstream can see that it happened.
        print("    polish, on the misfit alone", flush=True)
        cp, gp, polish = f.run(cu, gu, stages=polish_stages, target=TARGET_SIGMA,
                               area_weight=0.0, log=show)
        chi_p, area_p = last(hist + polish, "chi"), last(hist + polish, "area")
        obj_p = penalised(chi_p, area_p)
        keep_polish = obj_p <= obj_u
        c, g, chi, area, obj = ((cp, gp, chi_p, area_p, obj_p) if keep_polish else
                                (cu, gu, chi_u, area_u, obj_u))
        if not keep_polish:
            print(f"    the polish raised the objective from {obj_u:.4f} to {obj_p:.4f}; "
                  f"the body it started from is kept", flush=True)
        hist = hist + polish
        # Two finished starts are compared under the penalty, not under the misfit the polish
        # was run on. Between two bodies the misfit alone prefers the rougher one, which is the
        # comparison notes/objective.md says may never be made and the one select_answers.py is
        # careful to avoid. They are compared on the functional the shape was fitted under,
        # which is also the one the written body is judged by downstream.
        print(f"  start {s['i']} finished at chi {chi:.4f}, area {area:.3f}, objective "
              f"{obj:.4f} ({f.renders + s['renders']} renders)", flush=True)
        done.append({"objective": obj, "chi": chi, "c": c, "g": g, "start": s["i"],
                     "history": hist, "renders": f.renders + s["renders"],
                     "recipe": recipes[s["i"]], "refused_depth": f.refused_depth,
                     "refused_volume": refused_volume[0] - floor_before,
                     "polished": keep_polish, "budget_limited": still_descending(hist)})
        save()
        if out_of_time() and s is not keep[-1]:
            print(f"    the time budget stopped the ladder after {len(done)} of "
                  f"{len(keep) + len(fitted)} kept starts", flush=True)
            break
    time_limited = out_of_time()
    if not done:
        # Nothing finished, so there is no body to write and nothing to select. The
        # checkpoint holds every unit that did finish, so the answer is to run this again --
        # with a larger budget, or a shorter ladder -- rather than to write out a start that
        # has had no fit.
        raise SystemExit(
            f"model {a.model}: the time budget ran out before any start finished its ladder. "
            f"The ranked starts and the coarse screen are in "
            f"{ckpt_path if ckpt_path is not None else 'no checkpoint (no --out)'}; rerun to "
            f"continue from there, with a larger --time-budget or a smaller "
            f"--max-stage-iters.")
    # Two finished starts are compared under the penalty, not under the misfit the polish was
    # run on; the comparison is made here so that it is the same one whether the starts were
    # fitted in one run or across several.
    best = min(done, key=lambda q: q["objective"])
    if time_limited:
        print(f"  !!! the time budget stopped this fit; the body written is the best of the "
              f"{len(done)} start(s) that finished, not of the {max(1, a.restart_keep)} the "
              f"ladder was given. The checkpoint holds them, so rerunning this continues "
              f"with the rest.", flush=True)
    if best["budget_limited"]:
        print("  !!! the ladder was still taking steps at its last iteration, so this body "
              "is bounded by the iteration counts and not by the curves", flush=True)
    if best["refused_volume"]:
        print(f"  the volume floor turned away {best['refused_volume']} trials of the kept "
              f"start; a large count is a fit pressed against it rather than one stopped by "
              f"it once", flush=True)

    if not a.out:
        return

    # the body as it would be submitted, and its misfits, measured at the export resolution
    # and the export phase count, which is what a scored body is
    op_x = CodeOperator(inst, psi_grid(a.export_phases), res=a.export_res, config=render_cfg,
                        device=dev)
    d_x = load_inversion_curves(a.data_dir, a.model, m=a.export_phases, channel=a.channel)
    data_x = curve_pairs(d_x["curves"])
    scale_x = residual_scale(d_x, inst.eta).clamp_min(ETA_FLOOR)

    # One measurement, shared with the other solver, so that a body from either carries the
    # same number and scripts/select_answers.py can compare them. The reshaping travels
    # beside the code here and inside it there, which is the one difference between the two.
    measure = export_measure(op_x, support, R, data_x, scale_x, weight, a.area_weight)

    def measure_at_export(c, g, geoms):
        code = zero_code.clone()
        code[-N_NODES:] = torch.tensor(np.asarray(g), dtype=torch.float32, device=dev)
        return measure(code, geoms,
                       c=torch.tensor(np.asarray(c), dtype=torch.float32, device=dev))

    convex_fit, convex_fit_obj = measure_at_export(*zeros, fit_geoms)
    convex_held, convex_held_obj = measure_at_export(*zeros, held)
    fit_x, fit_x_obj = measure_at_export(best["c"], best["g"], fit_geoms)
    held_x, held_x_obj = measure_at_export(best["c"], best["g"], held)

    # The fit is written before anything is asked of the mesh. Extraction and export can
    # refuse a body the fit spent hours on -- a deeply carved surface can pinch, and the guard
    # is right to refuse it -- and losing the coefficients with it means refitting rather than
    # re-exporting. With this file a body can be extracted again at another resolution, or
    # after a better repair, for the cost of one extraction.
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.out).with_suffix(".fit.npz"), c=best["c"], g=best["g"],
             support=support.cpu().numpy())
    print(f"  wrote {Path(a.out).with_suffix('.fit.npz')}: the fitted coefficients, before "
          f"the mesh is extracted", flush=True)

    code = zero_code.clone()
    code[-N_NODES:] = torch.tensor(best["g"], dtype=torch.float32, device=dev)
    m = op_x.mesh(support, code, res=a.export_res,
                  c=torch.tensor(best["c"], dtype=torch.float32, device=dev))
    if m is None:
        raise SystemExit("the fitted body is degenerate; the fit is in the .fit.npz beside it")
    v = fit_to_cylinder(restore_constraints(
        CodeOperator.canonical(m[0], m[1]).cpu().numpy(), R), R)
    f = m[1].cpu().numpy()
    # A refused export still writes the report, so the run says why it has no body rather than
    # leaving a directory with nothing in it. The refusal stands: the STL is not written.
    export_error = None
    try:
        rep = export_stl(a.out, v, f)
    except ValueError as exc:
        export_error, rep = str(exc), {"refused": str(exc)}
        print(f"  !!! {exc}", flush=True)
    meta = {"model": a.model, "channel": d["channel"], "radius": R, "phases": a.phases,
            "export_phases": a.export_phases, "operator_res": a.operator_res,
            "export_res": a.export_res, "held_out": held, "fit_geoms": fit_g,
            "curves_fitted": int(weight.sum()), "area_weight": a.area_weight,
            "volume_trust": a.volume_trust, "volume_floor": a.volume_floor,
            "convex_volume": vol0, "depth_cap": cap, "convex_sigmas": convex_sigmas,
            "convex_explains_curves": convex_explains,
            "step_g": a.step_g, "step_c": a.step_c, "restarts": a.restarts,
            "start": best["recipe"], "renders": best["renders"],
            "restart_keep": a.restart_keep, "polished": best["polished"],
            "screen_starts": a.screen_starts, "starts_ranked": len(pool),
            "max_stage_iters": a.max_stage_iters,
            "max_degree": a.max_degree,
            "time_limited": time_limited, "ladders_finished": len(done),
            "ladders_asked": max(1, a.restart_keep),
            "starts_over_depth_cap": skipped_depth, "starts_under_floor": skipped_floor,
            "budget_limited": best["budget_limited"],
            "refused_volume": best["refused_volume"],
            "chi_fit": best["chi"], "objective_fit": best["objective"],
            "history": best["history"], "export": rep,
            "chi_fit_convex": convex_fit, "chi_held_convex": convex_held,
            "chi_fit_export": fit_x, "chi_held_export": held_x,
            "objective_fit_convex": convex_fit_obj, "objective_held_convex": convex_held_obj,
            "objective_fit_export": fit_x_obj, "objective_held_export": held_x_obj,
            "reshaping": best["c"].tolist(),
            "carve_depth_max": float(np.abs(kernel @ best["g"]).max()),
            # how often the star-shaped bound refused a trial. A run that never hits it is not
            # held back by it; one that hits it constantly is a body the correction wants to
            # carve past its own centre, and that is worth seeing rather than inferring.
            "depth_refusals": best["refused_depth"],
            "final_dice": truth_dice(v, f, a.model, a.data_dir),
            "convex_dice": convex_dice(sup_stl, a.model, a.data_dir),
            "final_convexity": convexity(v, f),
            "export_refused": export_error}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2,
                                                           default=json_default))
    if export_error is not None:
        # A distinct status, because this is not the same failure as a run that broke: the
        # fit finished, its numbers are in the JSON, and only the extraction is unusable. A
        # runbook can carry on past it, and the wiring test can count it as a stage that ran.
        print(f"model {a.model}: the fit finished and is written beside this, but the "
              f"extracted mesh is not a closed solid and was refused. Re-extract from "
              f"{Path(a.out).with_suffix('.fit.npz')} rather than refitting.", flush=True)
        raise SystemExit(EXPORT_REFUSED)
    print(f"  wrote {a.out} ({rep['faces']} faces, volume {rep['volume']:.3f}); at export "
          f"resolution chi_fit {fit_x:.3f} against the convex answer's {convex_fit:.3f}"
          + (f", chi_held {held_x:.3f} against {convex_held:.3f}" if held else "")
          + f"; dice {meta['final_dice']:.4f}, convexity {meta['final_convexity']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
