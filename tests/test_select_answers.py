"""The rule that lets a refined body replace a convex answer."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_answers import calibrated_ratio, decide  # noqa: E402


def test_a_refinement_must_beat_the_convex_answer_on_held_out_geometries_by_the_ratio():
    meta = {"held_out": [3, 9], "chi_held_convex": 2.0, "chi_held_export": 1.2}
    assert decide(meta, 0.7)[0]
    assert decide(meta, 0.6)[0]
    assert not decide(meta, 0.5)[0]


def test_a_fit_without_held_out_geometries_or_without_a_written_misfit_is_refused():
    assert not decide({"held_out": [], "chi_held_convex": 2.0, "chi_held_export": 0.5}, 0.9)[0]
    assert not decide({"held_out": [3], "chi_held_convex": 2.0,
                       "chi_held_export": float("inf")}, 0.9)[0]
    assert not decide({"held_out": [3]}, 0.9)[0]


def test_the_ratio_comes_from_a_public_refinement_that_improved_the_overlap():
    """The gate's number is measured, not chosen: it is the misfit ratio a refinement of a
    body with released truth reached, and it is only a number at all if that refinement moved
    the body toward its truth."""
    good = {"model": 3, "held_out": [1, 2], "convex_dice": 0.71, "final_dice": 0.98,
            "chi_held_convex": 2.0, "chi_held_export": 0.8}
    assert calibrated_ratio(good) == pytest.approx(0.4)

    # a refinement that fit the curves better while moving away from the truth is the failure
    # the gate exists to catch, so it licenses nothing
    for bad in ({**good, "final_dice": 0.70},                   # the overlap fell
                {**good, "final_dice": good["convex_dice"]},    # and did not move
                {**good, "held_out": []},                       # nothing was held out
                {**good, "convex_dice": float("nan"),           # the truth is not released
                 "final_dice": float("nan")},
                {**good, "chi_held_export": float("inf")}):     # nothing was written
        with pytest.raises(SystemExit):
            calibrated_ratio(bad)


def test_the_number_judged_is_the_functional_the_correction_was_fitted_under():
    """A correction fitted under the penalised objective is judged by it, and one fitted on
    the misfit alone by the misfit. Judging the first on the misfit would prefer a corrugated
    body to a shaped one, which is what the penalty exists to stop."""
    from select_answers import held_pair
    import numpy as np
    both = {"held_out": [1], "chi_held_export": 9.0, "chi_held_convex": 1.0,
            "objective_held_export": -2.0, "objective_held_convex": 0.0}
    held, base = held_pair(both)
    assert (held, base) == pytest.approx((np.exp(-1.0), 1.0))
    assert decide(both, 1.0)[0]                      # accepted on the objective
    assert not decide({k: v for k, v in both.items()
                       if not k.startswith("objective")} | {"held_out": [1]}, 1.0)[0]
    assert held_pair({"chi_held_export": 0.5, "chi_held_convex": 1.0}) == (0.5, 1.0)


def _written(d: Path, model: int, meta: dict) -> None:
    import json
    d.mkdir(parents=True, exist_ok=True)
    (d / f"Asteroid{model:02d}.stl").write_bytes(b"")
    (d / f"Asteroid{model:02d}.json").write_text(json.dumps(meta))


def test_two_solvers_of_one_body_are_ranked_by_what_each_removed_of_its_own_misfit(tmp_path):
    """Two directories, one body: the candidates come back best first, and the key is the
    held-out number over the convex answer's rather than the number itself, so that two runs
    tested on different cameras still compare."""
    from select_answers import candidates
    _written(tmp_path / "a", 4, {"held_out": [0, 9], "chi_held_convex": 2.0,
                                 "chi_held_export": 1.4})
    _written(tmp_path / "b", 4, {"held_out": [0, 9], "chi_held_convex": 2.0,
                                 "chi_held_export": 0.8})
    cs = candidates([tmp_path / "a", tmp_path / "b"], 4, 1.0)
    assert [Path(c["dir"]).name for c in cs] == ["b", "a"]
    assert all(c["accept"] for c in cs)
    assert cs[0]["gain"] == pytest.approx(0.4)

    # a body that is worse than its convex answer is still a candidate and still refused,
    # so that the run says why rather than silently offering nothing
    _written(tmp_path / "c", 4, {"held_out": [0, 9], "chi_held_convex": 2.0,
                                 "chi_held_export": 3.0})
    cs = candidates([tmp_path / "a", tmp_path / "c"], 4, 1.0)
    assert [c["accept"] for c in cs] == [True, False]

    # nothing written for a model contributes nothing, and a model with no candidate at all
    # leaves the convex answer standing
    assert candidates([tmp_path / "a", tmp_path / "b"], 5, 1.0) == []


def test_a_run_that_held_out_nothing_is_not_a_candidate_to_rank_above_one_that_did(tmp_path):
    """The held-out cameras are the whole of the test, so a fit that used every camera cannot
    win the comparison however low its misfit."""
    from select_answers import candidates
    _written(tmp_path / "all", 4, {"held_out": [], "chi_held_convex": 2.0,
                                   "chi_held_export": 0.1})
    _written(tmp_path / "held", 4, {"held_out": [0, 9], "chi_held_convex": 2.0,
                                    "chi_held_export": 1.9})
    cs = candidates([tmp_path / "all", tmp_path / "held"], 4, 1.0)
    accepted = [Path(c["dir"]).name for c in cs if c["accept"]]
    assert accepted == ["held"]


def test_a_released_truth_overrules_the_misfit():
    """On a public body the truth is known, and a refinement that fit the curves better while
    moving away from the shape is the failure this gate exists to catch. Measured on model 1,
    whose convex answer already overlapped its truth at 0.983: the refinement improved the
    held-out misfit by a third and took the overlap to 0.647."""
    better = {"held_out": [0, 27], "chi_held_convex": 3.8906, "chi_held_export": 2.5075,
              "convex_dice": 0.9831, "final_dice": 0.6474}
    assert not decide(better, 1.0)[0]
    assert "0.9831 to 0.6474" in decide(better, 1.0)[1]
    # the same numbers with the overlap improving are accepted
    assert decide({**better, "final_dice": 0.9900}, 1.0)[0]
    # and a scored model has no truth to appeal to, so the check says nothing there
    assert decide({**better, "convex_dice": float("nan"),
                   "final_dice": float("nan")}, 1.0)[0]
    assert decide({k: v for k, v in better.items() if "dice" not in k}, 1.0)[0]
