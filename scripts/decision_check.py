#!/usr/bin/env python3
"""Check the rule that picks the answer, on bodies whose truth is known.

reconstruct_lpd.py chooses the answer to a rock among the draws and their consensus bodies by
expected score against the draws. Whether that rule beats the alternatives cannot be checked
on the challenge models, since only three truths are public and those enter the calibration.
The held-out corpus bodies can check it: their truth is known, they never enter training, and
their curves are the exact model's curves with noise and model error at the sizes training
draws. For each of them this script reconstructs the body as reconstruct_lpd.py does (draws,
polish, candidates) and scores every candidate against the truth with both challenge
measures. It reports, per body and on average, how the rule's pick compares with the rules
one could use instead: the draw that fits the curves best, the draw closest to the other
draws, each consensus level on its own, and the best candidate in hindsight.

The same bodies also settle the one sampling choice that is not fixed by training: how far
the sampler is allowed to follow the curves away from the prior (--guidance, and
lpd_flow.LPDFlow.velocity). Each body is reconstructed once at each weight, from the same
random starts, so what is being compared is the weight and not the draws. Held-out corpus
bodies are the only bodies whose truth is known and which the training has not seen, so they
are the only place this can be measured rather than guessed; the cost is one reconstruction
per weight per body, which is small beside the training it follows.

The bodies are spread over how deeply carved they are, so the check covers smooth and
deeply carved bodies alike. Writes --out as JSON.
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

from hac26.conventions import cameras, psi_grid                             # noqa: E402
from hac26.field import N_DIR, apply_constraints                           # noqa: E402
from hac26.recon import dice, fit_to_cylinder, mesh_occupancy              # noqa: E402
from hac26.scoring.side_view import (measure_outlines, outline_extent,   # noqa: E402
                                     outline_set, surface_points)
from hac26.solvers.lpd_flow import CHURN, N_MODES, N_STEPS, LPDFlow, geometry_tags   # noqa: E402
from hac26.solvers.operator import CodeOperator                            # noqa: E402
from hac26.solvers.output import metric_medoid                             # noqa: E402
from reconstruct_lpd import (CONSENSUS_LEVELS, OCC_RES, consensus_bodies, decode,   # noqa: E402
                             candidate_diagnostic, dice_optimal_level, json_default,
                             make_resid_fn, mesh_misfit_by_geom, polish)
from train_lpd import (CALIBRATION, CORPUS, _enable_tf32, add_render_flags,   # noqa: E402
                       check_flow_metadata, cond_channels, file_digest, held_out,
                       load_corpus, load_flow_file, load_instrument, model_error_scale,
                       noise_sigma, probe_centres, probe_field, render_from, render_tag,
                       smooth_noise_like)

RULES = (("vote", "best_fit", "medoid", "oracle", "oracle_side", "consensus_opt")
         + tuple(f"consensus_{lv:g}" for lv in CONSENSUS_LEVELS))


def carving(corpus) -> np.ndarray:
    """How deeply carved each corpus body is: one minus the share of the probes inside its hull
    that are also inside the body."""
    g = corpus.codes[:, N_DIR:]
    o = probe_centres(corpus.support_true)
    body = (probe_field(corpus.support_true, g, o) < 0).float().sum((1, 2))
    hull = (probe_field(corpus.support_true, torch.zeros_like(g), o) < 0).float().sum((1, 2))
    return (1.0 - body / hull.clamp_min(1.0)).clamp_min(0.0).cpu().numpy()


def spread_over(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of k entries spread evenly over the sorted values (all of them when k is not
    smaller than their number)."""
    order = np.argsort(values)
    if k >= len(order):
        return order
    return order[np.round(np.linspace(0, len(order) - 1, k)).astype(int)]


def carve_bin(x: float) -> str:
    if x < 0.05:
        return "low"
    if x < 0.15:
        return "medium"
    return "high"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/lpd_flow.pt")
    ap.add_argument("--corpus", default=CORPUS, help="must match the training run")
    ap.add_argument("--calibration", default=CALIBRATION, help="must match the training run")
    ap.add_argument("--val-bodies", type=int, default=8,
                    help="must match the training run, so the same bodies are held out")
    ap.add_argument("--bodies", type=int, default=8,
                    help="held-out bodies to check, spread over how carved they are")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--churn", type=float, default=CHURN)
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0],
                    help="weights on the data part of the velocity to compare "
                         "(lpd_flow.LPDFlow.velocity). Each body is reconstructed once per "
                         "weight from the same starts, so the comparison is between the "
                         "weights and not between the draws")
    ap.add_argument("--polish-steps", type=int, default=30)
    ap.add_argument("--res", type=int, default=64, help="extraction resolution of the meshes")
    ap.add_argument("--side-points", type=int, default=200000,
                    help="surface samples per body for the side-view measure")
    ap.add_argument("--score-res", type=int, default=160,
                    help="side of the grid the candidates are scored against the truth on. "
                         "Must differ from reconstruct_lpd.OCC_RES, the grid they are "
                         "selected on and the consensus bodies are built from, or the "
                         "comparison is made on the discretisation it is meant to be "
                         "independent of")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/decision_check.json")
    add_render_flags(ap)
    a = ap.parse_args()
    render = render_from(a, "check")
    if a.score_res == OCC_RES:
        raise SystemExit(f"--score-res {a.score_res} is the selection grid OCC_RES; the "
                         f"candidates would be scored on the grid they were chosen on and "
                         f"the consensus bodies built from. Pick another side.")
    _enable_tf32()
    torch.manual_seed(a.seed)

    data, meta = load_corpus(a.corpus)
    if meta["calibration"] != file_digest(a.calibration):
        raise SystemExit(f"{a.corpus} was built with another calibration than {a.calibration}")
    phases, op_res = int(meta["phases"]), int(meta["operator_res"])
    M = min(N_MODES, phases // 2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inst = load_instrument(a.calibration, dev)
    eta = model_error_scale(inst)
    op = CodeOperator(inst, psi_grid(phases), res=op_res, config=render, device=dev)
    # the network runs on the CPU and the operator on the GPU, as reconstruct_lpd.py runs
    # them: the corpus tensors and the sampler's state stay on one device throughout
    sd, flow_meta = load_flow_file(a.ckpt, map_location="cpu")
    if "checkpoint_step" in flow_meta:
        print(f"  {a.ckpt} is a training checkpoint at step {flow_meta['checkpoint_step']}, "
              f"using {'best' if flow_meta.get('loaded_best_state') else 'current'} weights "
              f"from step {flow_meta['loaded_step']}", flush=True)
    check_flow_metadata(flow_meta, corpus=a.corpus, calibration=a.calibration,
                        phases=phases, operator_res=op_res, render=render_tag(render),
                        context=a.ckpt)
    net = LPDFlow.from_state_dict(sd)
    net.eval()
    C = len(cameras())
    tag, mask = geometry_tags(), torch.ones(1, C)
    geoms = list(range(C))

    n_val = max(0, min(a.val_bodies, len(data.codes) - 1))
    is_val = np.isin(data.index.numpy(), held_out(int(meta["bodies"]), n_val))
    val_rows = np.nonzero(is_val)[0]
    if len(val_rows) == 0:
        raise SystemExit("no held-out bodies; train with --val-bodies above zero")
    carved = carving(data)
    rows = val_rows[spread_over(carved[val_rows], a.bodies)]
    print(f"  {len(rows)} held-out bodies, carved {carved[rows].min():.2f}-{carved[rows].max():.2f}",
          flush=True)

    results = []
    for b in rows.tolist():
        gen = torch.Generator().manual_seed(a.seed * 100003 + int(data.index[b]))
        R = float(data.radius[b])
        support, sup_true = data.support[b], data.support_true[b]
        sigma = noise_sigma(1, generator=gen)[0]                                  # (G, 2)
        clean = data.curves[b]
        curves = (clean + sigma[..., None] * torch.randn(clean.shape, generator=gen)
                  + eta[..., None] * smooth_noise_like(clean, generator=gen))
        scale = torch.sqrt(sigma ** 2 + eta ** 2)

        cond = cond_channels(support)
        # the truth is the same for every weight, so it is built once, before the sweep
        true_code = data.codes[b].clone()
        true_code[:N_DIR] = 0.0
        truth = op.mesh(sup_true, true_code, res=a.res)
        if truth is None:
            print(f"  body {int(data.index[b])}: the truth has no surface at this resolution, "
                  f"skipped", flush=True)
            continue
        tv = fit_to_cylinder(apply_constraints(truth[0].cpu().numpy(), 1.0), R)
        tf = truth[1].cpu().numpy()
        zero_code = torch.zeros_like(data.codes[b])
        cv, cf, _ = decode(op, zero_code, support, res=a.res)
        convex_mesh = None if cv is None else (fit_to_cylinder(cv, R), cf)

        for w in a.guidance:
            t0 = time.time()
            torch.manual_seed(a.seed * 7919 + int(data.index[b]))   # same starts at every weight
            codes = net.sample(make_resid_fn(net, op, curves, scale, mask, M, cond, support, R),
                               tag.expand(a.samples, -1, -1), mask.expand(a.samples, -1), cond, R,
                               batch=a.samples, n_steps=a.steps, churn=a.churn, guidance=w)
            fits = []
            for i in range(a.samples):
                if a.polish_steps > 0:
                    codes[i], _, chi, _ = polish(net, op, codes[i], support, R, curves, scale, geoms,
                                                 a.polish_steps)
                fits.append(chi if a.polish_steps > 0 else float("nan"))
            raw = net.codec.decode(codes)

            meshes, chis, kept = [], [], []
            for i in range(a.samples):
                v, f, _ = decode(op, raw[i], support, res=a.res)
                if v is None:
                    continue
                kept.append(i)
                chis.append(float(mesh_misfit_by_geom(op, v, f, R, curves, scale).pow(2).mean().sqrt()))
                meshes.append((fit_to_cylinder(v, R), f))
            if len(meshes) < 2:
                print(f"  body {int(data.index[b])}: fewer than two draws decoded, skipped", flush=True)
                continue
            n_draws = len(meshes)
            ext = max(float(np.abs(mv).max()) for mv, _ in meshes + [(tv, tf)]) * 1.05
            occs = [mesh_occupancy(mv, mf, OCC_RES, ext) for mv, mf in meshes]
            # rounded as the levels below are: consensus_bodies names each body by the level
            # it was asked for, so an unrounded opt would never be found among them
            opt = round(float(dice_optimal_level(occs, ext, R)), 9)
            # asking for a level twice builds and scores the same body twice; the rules
            # consensus_opt and consensus_<lv> then simply name one shared candidate
            asked = tuple(dict.fromkeys(round(float(x), 9)
                                        for x in CONSENSUS_LEVELS + (opt,)))
            extra = consensus_bodies(occs, ext, R, levels=asked)
            levels = [lv for lv, _, _ in extra]
            candidates = meshes + [(mv, mf) for _, mv, mf in extra]
            sources = ([{"kind": "draw", "label": f"draw {i}", "draw": int(i)}
                        for i in range(n_draws)]
                       + [{"kind": "consensus", "label": f"consensus at level {lv:g}",
                           "level": float(lv)} for lv, _, _ in extra])
            occs = occs + [mesh_occupancy(mv, mf, OCC_RES, ext) for _, mv, mf in extra]
            candidate_fits = list(chis)
            for _, mv, mf in extra:
                candidate_fits.append(float(mesh_misfit_by_geom(op, mv / np.array([R, R, 1.0]),
                                                                mf, R, curves, scale)
                                            .pow(2).mean().sqrt()))
            candidate_diagnostics = [
                candidate_diagnostic(src["kind"], src["label"], candidates[i][0],
                                     candidates[i][1], candidate_fits[i])
                for i, src in enumerate(sources)
            ]
            eligible = [i for i, row in enumerate(candidate_diagnostics) if row["eligible"]]
            eligible_draws = [i for i in eligible if i < n_draws]
            if not eligible_draws:
                print(f"  body {int(data.index[b])}: no valid draw candidate at guidance {w}, "
                      f"skipped", flush=True)
                continue
            eligible_set = set(eligible)
            eligible_occs = [occs[i] for i in eligible]
            eligible_candidates = [candidates[i] for i in eligible]
            eligible_outlines = [surface_points(v, f, n=a.side_points, seed=a.seed + 1 + i)
                                 for i, (v, f) in enumerate(eligible_candidates)]
            vote = eligible[metric_medoid(eligible_occs, eligible_outlines,
                                          n_ref=len(eligible_draws))]
            medoid_draw_occs = [occs[i] for i in eligible_draws]
            medoid = eligible_draws[metric_medoid(medoid_draw_occs, n_ref=len(eligible_draws))]
            best_fit = eligible_draws[int(np.argmin([candidate_fits[i] for i in eligible_draws]))]

            # Every eligible candidate against the truth, with both measures. The grid here is
            # --score-res rather than OCC_RES on purpose (the script refuses the two equal):
            # the candidates are chosen on OCC_RES and a consensus body is built out of it, so
            # scoring on the same grid would flatter whichever candidate that grid happens to
            # suit.
            truth_occ = mesh_occupancy(tv, tf, a.score_res, ext)
            truth_pts = surface_points(tv, tf, n=a.side_points, seed=a.seed)
            score_outlines = {i: surface_points(candidates[i][0], candidates[i][1],
                                                n=a.side_points, seed=a.seed + 101 + i)
                              for i in eligible}
            conv_pts = (surface_points(convex_mesh[0], convex_mesh[1], n=a.side_points,
                                       seed=a.seed + 999)
                        if convex_mesh is not None else None)
            # the outlines need their own extent: the occupancy extent above is the half-width of
            # a cube, which is not wide enough for a projection (see side_view.outline_extent)
            oext = outline_extent(list(score_outlines.values()) + [truth_pts]
                                  + ([] if conv_pts is None else [conv_pts]))
            truth_out = outline_set(truth_pts, oext)
            scores = [(None, None)] * len(candidates)
            for i, pts in score_outlines.items():
                v, f = candidates[i]
                d = dice(mesh_occupancy(v, f, a.score_res, ext), truth_occ)
                s = measure_outlines(outline_set(pts, oext), truth_out)["assd_mean"]
                scores[i] = (d, s)
            convex_score = None
            if convex_mesh is not None and conv_pts is not None:
                conv_d = dice(mesh_occupancy(convex_mesh[0], convex_mesh[1], a.score_res, ext),
                              truth_occ)
                conv_s = measure_outlines(outline_set(conv_pts, oext), truth_out)["assd_mean"]
                convex_score = {"dice": float(conv_d), "side_assd": float(conv_s)}
            picks = {"vote": vote, "best_fit": best_fit, "medoid": medoid,
                     # one ceiling per measure: the best-Dice candidate is not generally the
                     # best-outline one, and reporting only the first understates what the
                     # second measure was reachable
                     "oracle": max(eligible, key=lambda i: scores[i][0]),
                     "oracle_side": min(eligible, key=lambda i: scores[i][1])}
            for lv in CONSENSUS_LEVELS:            # a level with no closed surface has no candidate
                ci = n_draws + levels.index(lv) if lv in levels else None
                picks[f"consensus_{lv:g}"] = ci if ci in eligible_set else None
            ci = n_draws + levels.index(opt) if opt in levels else None
            picks["consensus_opt"] = ci if ci in eligible_set else None
            row = {"body": int(data.index[b]), "guidance": float(w),
                   "consensus_opt_level": float(opt),
                   "consensus_opt_is_fixed_level": bool(
                       any(abs(opt - lv) < 1e-9 for lv in CONSENSUS_LEVELS)),
                   "consensus_levels_built": [float(x) for x in levels],
                   "carved": float(carved[b]), "carve_bin": carve_bin(float(carved[b])),
                   "radius": R,
                   # both lists are indexed by surviving draw, as `picked` is: a draw that
                   # failed to decode is absent from all three, so entry j of one describes
                   # the same draw as entry j of the others
                   "draws": n_draws, "misfit_sigma": chis,
                   "polished_misfit_sigma": [fits[i] for i in kept],
                   "candidate_diagnostics": candidate_diagnostics,
                   "eligible_candidates": [int(i) for i in eligible],
                   "convex_start": convex_score,
                   "dice": {r: (None if k is None else scores[k][0]) for r, k in picks.items()},
                   "side_assd": {r: (None if k is None else scores[k][1]) for r, k in picks.items()},
                   "picked": picks, "seconds": time.time() - t0}
            results.append(row)
            print(f"  body {row['body']:>5} carved {row['carved']:.2f} guidance {w:.2f}: dice "
                  + ", ".join(f"{r} {row['dice'][r]:.3f}" for r in RULES
                              if row['dice'][r] is not None)
                  + f"  ({row['seconds']:.0f}s)", flush=True)

    if not results:
        raise SystemExit("no body could be checked")

    # A body that completed at one weight and not at another must not be in one weight's
    # mean and missing from the next: the weights would then be compared over different
    # bodies, and the failures are not independent of what is being swept -- a large
    # guidance is itself what pushes a draw off the corpus and leaves it undecodable, so a
    # weight could win by destroying the hard bodies rather than by reconstructing them.
    per_body = {}
    for x in results:
        per_body.setdefault(x["body"], set()).add(float(x["guidance"]))
    n_weights = len({float(w) for w in a.guidance})
    common = {b for b, ws in per_body.items() if len(ws) == n_weights}
    dropped = sorted(set(per_body) - common)
    if dropped and common:
        print(f"\n  {len(dropped)} of {len(per_body)} bodies did not complete at every "
              f"guidance weight and are left out of the means below: {dropped}")
    elif dropped:
        print("\n  WARNING: no body completed at every guidance weight, so the means below "
              "are over a different set of bodies for each weight and cannot be compared.")
        common = set(per_body)

    def finite_of(rows_, key, r):
        return [x[key][r] for x in rows_
                if x[key][r] is not None and np.isfinite(x[key][r])]

    def mean_of(rows_, key, r):
        # inf is a value that occurs here -- measure_outlines returns it when no direction
        # has both outlines, mesh_misfit_by_geom when the renderer rejects a mesh -- and a
        # single one would make the mean inf, hiding every body that did score
        vals = finite_of(rows_, key, r)
        return float(np.mean(vals)) if vals else None

    def nonfinite_of(rows_, key, r):
        return sum(1 for x in rows_
                   if x[key][r] is not None and not np.isfinite(x[key][r]))

    summary = {}
    for w in a.guidance:
        rows_ = [x for x in results
                 if x["guidance"] == float(w) and x["body"] in common]
        summary[f"{w:g}"] = {r: {"dice": mean_of(rows_, "dice", r),
                                 "side_assd": mean_of(rows_, "side_assd", r),
                                 "bodies": len(finite_of(rows_, "dice", r)),
                                 "nonfinite": nonfinite_of(rows_, "side_assd", r)}
                             for r in RULES}
        bins = {}
        for name in ("low", "medium", "high"):
            br = [x for x in rows_ if x["carve_bin"] == name and x.get("convex_start")]
            if br:
                vote_d = [x["dice"]["vote"] for x in br if x["dice"]["vote"] is not None]
                base_d = [x["convex_start"]["dice"] for x in br]
                bins[name] = {"bodies": len(br),
                              "convex_dice": float(np.mean(base_d)),
                              "vote_dice": float(np.mean(vote_d)) if vote_d else None,
                              "dice_gain": (float(np.mean(vote_d) - np.mean(base_d))
                                            if vote_d else None)}
        summary[f"{w:g}"]["by_carve_bin"] = bins
    print("\n  mean over the bodies every weight completed, per guidance weight:")
    for w in a.guidance:
        for r in RULES:
            st = summary[f"{w:g}"][r]
            if st["dice"] is None:
                print(f"    guidance {w:<5g} {r:<15} no candidate on any body")
                continue
            sv = "   n/a" if st["side_assd"] is None else f"{st['side_assd']:.4f}"
            nf = f"  [{st['nonfinite']} non-finite]" if st["nonfinite"] else ""
            print(f"    guidance {w:<5g} {r:<15} dice {st['dice']:.4f}   "
                  f"side-view distance {sv}   ({st['bodies']} bodies){nf}")
        for name, st in summary[f"{w:g}"].get("by_carve_bin", {}).items():
            if st["vote_dice"] is not None:
                print(f"    guidance {w:<5g} carved {name:<6} vote dice {st['vote_dice']:.4f} "
                      f"vs convex {st['convex_dice']:.4f}  gain {st['dice_gain']:+.4f} "
                      f"({st['bodies']} bodies)")
        print("")

    # The challenge sums the two measures, so the weight is chosen by both, ranked the way
    # metric_medoid ranks candidates. Choosing on Dice alone can name a weight that is
    # losing the other half of the score by more than it gains on this half.
    usable = [w for w in a.guidance
              if summary[f"{w:g}"]["vote"]["dice"] is not None
              and summary[f"{w:g}"]["vote"]["side_assd"] is not None]
    best = (None, None)
    if usable:
        dm = np.array([summary[f"{w:g}"]["vote"]["dice"] for w in usable], float)
        sm = np.array([summary[f"{w:g}"]["vote"]["side_assd"] for w in usable], float)
        rank = np.argsort(np.argsort(-dm)) + np.argsort(np.argsort(sm))
        i = int(np.argmin(rank))
        best = (usable[i], float(dm[i]))
        print(f"  the rule the reconstruction uses ('vote') scores best at guidance "
              f"{best[0]:g} (dice {dm[i]:.4f}, side-view distance {sm[i]:.4f}, by the rank "
              f"sum of the two). Set RECON_GUIDANCE to it.")
        if len(usable) > 1 and float(np.ptp(dm)) < 1e-3:
            print(f"  -- but the mean Dice spans only {float(np.ptp(dm)):.2g} across the "
                  f"weights, which is not a margin: leaving RECON_GUIDANCE alone is as good.")
    print("\n  'oracle' and 'oracle_side' are the best candidate in hindsight under each "
          "measure on its own -- the ceiling of any rule on that measure alone, not on both.")
    out = {"ckpt": a.ckpt, "corpus": a.corpus, "samples": a.samples, "steps": a.steps,
           "polish_steps": a.polish_steps, "rules": list(RULES),
           "guidance": [float(w) for w in a.guidance],
           "best_guidance": (None if best[0] is None else float(best[0])),
           "best_guidance_criterion": ("rank sum of the mean Dice and the mean side-view "
                                       "distance of the 'vote' rule, over the bodies that "
                                       "completed at every guidance weight"),
           "score_res": int(a.score_res), "occ_res": int(OCC_RES),
           "val_bodies": int(a.val_bodies), "res": int(a.res), "seed": int(a.seed),
           "side_points": int(a.side_points), "churn": float(a.churn),
           "bodies_in_summary": sorted(common), "bodies_dropped": dropped,
           "summary": summary, "bodies": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=json_default))
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
