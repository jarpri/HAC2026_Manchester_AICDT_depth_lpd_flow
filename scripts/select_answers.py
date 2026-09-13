#!/usr/bin/env python3
"""Decide, per scored model, which body enters the submission: a refinement or the convex answer.

    python scripts/select_answers.py --refined results/gn results/map \
        --calibrate results/gn/Asteroid03.json

A refinement is accepted for a model when its number on the geometries held out of its fit,
measured on the body as written, is at most `ratio` times the convex answer's number on the
same geometries, and the file passes the submission check. The held-out geometries are the
whole of the argument: a body fitted on every camera can reach any misfit by shape or by
overfitting, and only a camera the fit never saw separates the two.

Several --refined directories can be given, which is how two solvers of the same problem are
decided between. Each body is first compared with the convex answer on its own held-out
cameras, and the candidates are then ranked by that comparison: a ratio is what transfers,
since each one is a like-for-like measurement against the same convex body on the same
cameras. The best candidate that also passes the submission check is the one copied. Both
solvers measure through one function at one resolution (reconstruct_map.export_measure), and
both hold out the same cameras for the same count (data_io.held_out_geoms), so in the normal
case ranking by the ratio and ranking by the held-out number are the same ordering; when two
runs held out different cameras the ranking is between two different tests and the run says so.

The number is whichever functional the refinement was fitted under, and `held_pair` says why
it has to be. A correction fitted under an objective that charges the body's surface as well
as its misfit, judged here on the misfit alone, would be thrown away in favour of a corrugated
body that fits the curves better and looks less like the truth.

The ratio is one by default: a refinement stands where it fits the held-out geometries at
least as well as the convex answer does, which is evidence about the body being decided. It
can be tightened by a measured quantity instead. `--calibrate` reads a public model's
refinement, checks that the refinement moved that body toward its released truth rather than
away from it, and takes the misfit ratio it reached; a refinement is then trusted only where
it beats its convex answer by at least as much. That is one body's margin asked of every
other, so it is offered and not required: a public run that falls short would otherwise stand
every convex answer in the submission, and a convex answer is not a safe default here but a
body known to be missing the concavities the challenge is about. `--ratio` sets the number
directly.

The accepted files are copied over the convex answers under results/submission, which
scripts/make_submission.py regenerates in seconds, and results/submission/selection.json
records the choice, the candidates it was made among and the numbers behind it, so the
submitted directory says what it contains and where each body came from.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_submission import inspect                       # noqa: E402
from hac26.conventions import CYLINDER_R, PUBLIC_MODELS    # noqa: E402
from reconstruct import answer_path                        # noqa: E402

SELECTION_FILE = "results/submission/selection.json"
RATIO_MARGIN = 1.0    # how much of a public model's ratio a scored model has to reproduce.
                      # One asks for the same improvement; the ratio is one measurement of
                      # one body, and asking a secret body to beat it is not warranted.


def held_pair(meta: dict) -> tuple:
    """(the correction's held-out number, the convex answer's) on whichever functional the
    correction was fitted under.

    A correction fitted under a penalised objective must be judged by it. The penalty exists
    because the misfit of a rendered body reports how finely its surface is resolved as well
    as whether its shape is right, so on the misfit alone this gate prefers a corrugated body
    to a shaped one and would throw the better body away. A run that minimised the misfit
    alone reports no objective and is judged on the misfit, which is the same thing for it.

    The objective is a logarithm plus an area, so it is a difference rather than a ratio that
    is meaningful; the two are put on a common footing by exponentiating, under which a ratio
    of exponentials is the ratio of misfits the same body would have at equal area.
    """
    held, base = meta.get("objective_held_export"), meta.get("objective_held_convex")
    if held is not None and base is not None and held == held and base == base:
        return float(np.exp(0.5 * held)), float(np.exp(0.5 * base))
    return meta.get("chi_held_export"), meta.get("chi_held_convex")


def calibrated_ratio(meta: dict) -> float:
    """The largest accepted misfit ratio, from a public model's refinement of known truth.

    A refinement that lowered the misfit while lowering the overlap with the truth is the
    failure this gate exists to catch, so it yields no ratio at all rather than a lenient
    one. So does a refinement of a body whose truth is not released, or one fitted on every
    camera, whose misfit is not a test of anything.
    """
    if not meta.get("held_out"):
        raise SystemExit("the calibrating run held out no geometries, so its misfit ratio "
                         "is not a test of a shape")
    dice, base_dice = meta.get("final_dice"), meta.get("convex_dice")
    if dice is None or base_dice is None or not (dice == dice and base_dice == base_dice):
        raise SystemExit("the calibrating run is of a model whose truth is not released, so "
                         "there is nothing to calibrate against")
    if dice <= base_dice:
        raise SystemExit(f"the calibrating refinement took the overlap from {base_dice:.4f} "
                         f"to {dice:.4f}, so a lower misfit is not evidence of a better "
                         f"shape and the convex answers stand")
    held, convex = held_pair(meta)
    if not held or not convex or not (held < float("inf")):
        raise SystemExit("the calibrating run has no held-out misfit")
    return RATIO_MARGIN * held / convex


def decide(meta: dict, ratio: float) -> tuple:
    """(accept, reason) for one refinement's JSON. A refinement fitted without held-out
    geometries has no honest number and is refused.

    The reason names the fraction of the convex answer's number the refinement reached, and
    not the two numbers themselves: under the penalised objective they are exponentials
    carrying the body's area as well as its misfit, so their size means nothing on its own
    and printing them invites reading an area as a misfit. The numbers are in the selection
    file for anyone who wants them, with the plain misfits beside them.
    """
    held, base = held_pair(meta)
    if not meta.get("held_out"):
        return False, "no geometries were held out of the fit"
    if held is None or base is None or not (held < float("inf")):
        return False, "the written body has no held-out misfit"
    if not base:
        return False, "the convex answer has no held-out number to compare with"
    # Where the truth is released, it decides, and no misfit argues with it. A refinement
    # that fit the curves better while moving away from the shape is the failure this gate
    # exists to catch -- the same failure calibrated_ratio refuses to take a ratio from --
    # and on a body whose convex answer is already close to its hull there is no concavity to
    # find, so the objective buys its misfit with overlap. Both numbers are nan for a scored
    # model, where there is no truth to appeal to, and the check is then silent.
    dice, base_dice = meta.get("final_dice"), meta.get("convex_dice")
    if dice is not None and base_dice is not None and dice == dice and base_dice == base_dice:
        if dice <= base_dice:
            return False, (f"the released truth says otherwise: the refinement took the "
                           f"overlap from {base_dice:.4f} to {dice:.4f}")
    got = held / base
    if held > ratio * base:
        return False, (f"held out {got:.3f} of the convex answer's number, above the "
                       f"{ratio:g} this run accepts")
    return True, (f"held out {got:.3f} of the convex answer's number, at or below the "
                  f"{ratio:g} this run accepts")


def candidates(dirs, model: int, ratio: float) -> list:
    """Every refinement of one model that is on disk, ranked best first.

    The key is the body's held-out number over the convex answer's on the same cameras. That
    ratio is what compares across runs: each one is the same convex body measured on the same
    held-out cameras as its challenger, so the ratio says how much of the misfit the
    correction removed whatever cameras it was tested on, while the numbers themselves do not
    if two runs held out different ones.
    """
    out = []
    for d in dirs:
        stl = Path(d) / f"Asteroid{model:02d}.stl"
        js = stl.with_suffix(".json")
        if not (stl.exists() and js.exists()):
            continue
        meta = json.loads(js.read_text())
        held, base = held_pair(meta)
        accept, reason = decide(meta, ratio)
        key = (float(held) / float(base)) if (held and base and np.isfinite(held)
                                              and np.isfinite(base)) else float("inf")
        out.append({"dir": str(d), "stl": stl, "meta": meta, "held": held, "base": base,
                    "gain": key, "accept": accept, "reason": reason,
                    "held_out": list(meta.get("held_out") or [])})
    out.sort(key=lambda q: q["gain"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined", required=True, nargs="+",
                    help="directories of Asteroid<NN>.stl and .json written by "
                         "reconstruct_gn.py or reconstruct_map.py. Give more than one to "
                         "decide between two solvers of the same body")
    ap.add_argument("--calibrate",
                    help="a public model's refinement JSON from the same run; the accepted "
                         "misfit ratio is the one it reached, and only if it improved the "
                         "overlap with the released truth")
    ap.add_argument("--ratio", type=float,
                    help="the accepted ratio directly, when the calibrating run is not at "
                         "hand; largest held-out misfit of a refinement as a fraction of the "
                         "convex answer's")
    ap.add_argument("--models", nargs="+", type=int,
                    default=[M for M in range(1, 11) if M not in PUBLIC_MODELS])
    ap.add_argument("--into", default=None,
                    help="directory the chosen bodies are written to; by default the one "
                         "answer_path names, which is the submission itself. A directory "
                         "other than the default is seeded with the convex answer for every "
                         "model first, so what it holds is a complete submission either way")
    a = ap.parse_args()
    if a.calibrate is not None and a.ratio is not None:
        raise SystemExit("give either --calibrate, to measure the ratio on a public model, "
                         "or --ratio to set it, not both")
    calibration = json.loads(Path(a.calibrate).read_text()) if a.calibrate else None
    # With neither, the ratio is one: a refinement is accepted when it fits the geometries held
    # out of its own fit at least as well as the convex answer does. That is evidence about the
    # body being decided. A public model's ratio is a tightening on top of it, and it is one
    # body's margin asked of every other, so it is offered rather than required -- a public run
    # that falls short would otherwise stand every convex answer in the submission.
    ratio = (a.ratio if a.ratio is not None else
             calibrated_ratio(calibration) if calibration is not None else 1.0)
    if not 0.0 < ratio <= 1.0:
        raise SystemExit(f"the accepted ratio {ratio:g} is outside (0, 1]: a refinement that "
                         f"fits the held-out geometries no better than the convex answer is "
                         f"not accepted")
    if calibration is not None:
        print(f"ratio {ratio:.3f}, from model {calibration['model']}, whose refinement took "
              f"the overlap with its released truth from {calibration['convex_dice']:.4f} to "
              f"{calibration['final_dice']:.4f}", flush=True)
    elif a.ratio is None:
        print("ratio 1.000: a refinement stands where it fits the held-out geometries at "
              "least as well as the convex answer. Pass --calibrate to require a public "
              "model's own margin as well.", flush=True)

    into = Path(a.into) if a.into else None
    # A public model's convex answer is the support every fit of that body starts from and the
    # baseline every comparison of it is made against (reconstruct.answer_path). Writing a
    # refinement over it would move the start of the next run and leave the numbers labelled
    # convex describing a body that is not, so a public model is only ever selected into a
    # directory named here.
    public = [M for M in a.models if M in PUBLIC_MODELS]
    if public and into is None:
        raise SystemExit(
            f"model(s) {public} are public, and their answers under results/public are the "
            f"support every fit starts from and the baseline it is compared with. Give "
            f"--into to write a chosen body somewhere else; the submission itself is the "
            f"scored models.")
    if into is not None:
        into.mkdir(parents=True, exist_ok=True)
        for M in a.models:
            src = answer_path(M)
            dst = into / Path(src).name
            if not dst.exists():
                shutil.copyfile(src, dst)
        print(f"seeded {into} with the convex answer for {len(a.models)} models; what it "
              f"holds is a complete submission whatever is accepted below", flush=True)
    selection_file = str(into / "selection.json") if into else SELECTION_FILE
    selection = {"ratio": ratio, "refined_dirs": list(a.refined),
                 "into": str(into) if into else None,
                 "calibrated_on": a.calibrate, "models": {}}
    for M in a.models:
        target = str(into / Path(answer_path(M)).name) if into else answer_path(M)
        cands = candidates(a.refined, M, ratio)
        entry = {"answer": "convex", "candidates": [
            {"dir": c["dir"], "held": c["held"], "held_convex": c["base"],
             "gain": c["gain"], "accept": c["accept"], "reason": c["reason"],
             "held_out": c["held_out"]} for c in cands]}
        if not cands:
            entry["reason"] = "no refinement written"
        else:
            sets = {tuple(c["held_out"]) for c in cands}
            if len(sets) > 1:
                print(f"model {M:>2}: the candidates held out different cameras "
                      + "; ".join(f"{c['dir']} {c['held_out']}" for c in cands)
                      + ". They are ranked by how much of the convex answer's misfit each "
                        "removed on its own cameras, which is a comparison of two different "
                        "tests.", flush=True)
                entry["held_out_disagree"] = True
            chosen, refused = None, []
            for c in cands:
                if not c["accept"]:
                    refused.append(f"{c['dir']}: {c['reason']}")
                    continue
                check = inspect(c["stl"], CYLINDER_R[M])
                if check["fails"]:
                    refused.append(f"{c['dir']}: fails the submission check: "
                                   + "; ".join(check["fails"]))
                    continue
                chosen = c
                break
            if chosen is None:
                entry["reason"] = "; ".join(refused)
            else:
                shutil.copyfile(chosen["stl"], target)
                entry.update({"answer": "refined", "from": chosen["dir"],
                              "reason": chosen["reason"],
                              "held_convex": chosen["base"], "held_refined": chosen["held"],
                              "chi_held_convex": chosen["meta"].get("chi_held_convex"),
                              "chi_held_export": chosen["meta"].get("chi_held_export"),
                              "held_out": chosen["held_out"]})
                if refused:
                    entry["passed_over"] = refused
        selection["models"][M] = entry
        origin = f" from {entry['from']}" if entry.get("from") else ""
        print(f"model {M:>2}: {entry['answer']:8s}{origin}  {entry['reason']}", flush=True)
    Path(selection_file).parent.mkdir(parents=True, exist_ok=True)
    Path(selection_file).write_text(json.dumps(selection, indent=2))
    print(f"wrote {selection_file}")


if __name__ == "__main__":
    main()
