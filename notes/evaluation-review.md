# Evaluation review: the scoring measures, the answer rule, and the decision harness

Review of everything between a reconstruction and a number: `hac26/scoring/` (the two
measures the challenge scores on), the rule in `hac26/solvers/output.py` and
`scripts/reconstruct_lpd.py` that picks which candidate is submitted, and
`scripts/decision_check.py`, the harness that decides which rule and which guidance weight
to use.

The method was three independent read-throughs of the three areas, each given the code and
`docs/challenge_info.md` and nothing else, followed by verification of every load-bearing
claim against the repo's own committed artefacts. Nothing here rests on a reviewer's
say-so: each finding below is either reproduced numerically in this note or traced to a
line that can be read. Where a claim could not be checked it is labelled as such.

The constraint throughout was that the real pipeline cannot be run here -- no GPU, no
`dataset/raw`, no nvdiffrast -- so the evidence is the ten committed reconstructions in
`results/lpd/`, synthetic ensembles that reproduce the same conditions, and the test suite.

## Status

| | finding | status |
|---|---|---|
| 1 | a consensus level equal to k/n_draws makes a degenerate, inside-out body | **fixed** -- `reconstruct_lpd.off_lattice_level` |
| 2 | `export_stl` repairs in the wrong order and writes a body it knows is broken | **fixed** -- fill then orient, validate, refuse |
| 3 | `rescale_touch_z` returns `nan` on a flat body, which scores **1.0** | **fixed** -- raises, as `shape_library.pose` does |
| 4 | nothing on the voxel path checks what it was handed | **fixed** -- `voxel.load_one_solid`, empty-grid guard |
| 5 | `side_view.py` exits 0 having scored nothing | **fixed** -- names what is missing, exits non-zero |
| 6 | `decision_check`'s independent scoring grid is the selection grid | **fixed** -- `--score-res`, refuses to equal `OCC_RES` |
| 7 | per-weight means average over different body sets | **fixed** -- restricted to bodies every weight completed |
| 8 | the guidance weight is chosen on Dice alone | **fixed** -- rank sum of both measures, criterion recorded |
| 9 | `inf` poisons a mean; rules vanish from the table | **fixed** -- non-finite counted, not averaged |
| 10 | `oracle` is a Dice-only ceiling captioned as the ceiling of any rule | **fixed** -- `oracle_side` added, caption corrected |
| 11 | `misfit_sigma` and `polished_misfit_sigma` use different index conventions | **fixed** -- both indexed by surviving draw |
| 12 | `dice_optimal_level` returns a level it just found produces nothing | **fixed** -- falls back to the last level that worked |
| 13 | a duplicated level builds and scores the same body twice | **fixed** -- deduped in both call sites |
| 14 | `MEDOID_SIDE_POINTS` is passed to the decision stage but not in its signature | **fixed** -- added |
| 15 | the two scorers cannot import `hac26` under the pipeline's venv | **fixed** -- `PYTHONPATH` at the call sites |
| 16 | both `side_view.py` runs shared one default `--out` and clobbered each other | **fixed** -- separate files under `runs/` |
| 17 | consensus bodies ship narrower than the published radius | **open, deliberately** -- a modelling choice, see below |
| 18 | consensus bodies are scored in-sample against the draws that built them | **open** -- the sharpest finding; the fix has a real cost |
| 19 | the scoring grid is a cube sized by the widest dimension | **open** -- model 10 is selected on 31 z-layers |
| 20 | `binary_closing` may erase the concavities the measure exists to detect | **open, unverified** |
| 21 | the held-out bodies also select the checkpoint by early stopping | **open** -- a training-side decision |
| 22 | the side-view measure is not normalised to [0, 1] as the spec requires | **open** -- works as a ranking proxy, is not the measure |
| 23 | `answer_misfit_sigma` is `inf` for every consensus answer | **partly explained by 1**; see below |

Findings 1 to 16 are changes in the working tree, with tests. The suite goes from 152
passed to 168 passed, 6 skipped, none failed.

---

## 1. The one that reached the submission

Three of the ten committed reconstructions are not solids. Measured directly from the
files in `results/lpd/`:

```
model  answer                     zero-area faces   signed volume    r/R
  2    consensus at level 0.5      6046 / 29526        -5.235       0.953
  4    consensus at level 0.5      4822 / 20052        -5.845       0.990
  7    consensus at level 0.5      6200 / 24084        -3.912       0.990
 10    consensus at level 0.35        0 / 11912       +38.673       0.996
  1,3,5,6,8,9  (draw answers)         0               positive      1.000
```

A fifth of the triangles have zero area and the signed volume is negative -- the surfaces
are inside out. The repo's own JSON agrees: `"watertight": false, "filled_holes": true` for
exactly those three.

The cause is arithmetic. With `--samples 8`, the fraction of draws occupying a voxel takes
only the values k/8, and `CONSENSUS_LEVELS` contains 0.5, which *is* 4/8. Marching cubes on
a level exactly equal to sampled values places vertices on grid points and emits zero-area
triangles and pinch points. The one consensus answer at a level not of that form -- model
10 at 0.35 -- is clean. Reproduced on a synthetic 8-draw ensemble:

```
                            degenerate faces   watertight
level 0.35 (off-lattice)         0 / 22796        True
level 0.5  (== 4/8)            962 / 22730        False
```

Nothing about this is specific to the data; `--samples 7` would have hidden it entirely.

The arithmetic that caused it is also what sets how good a consensus body can be, which is the
reason the draw count is now 64 rather than 8. The occupied fraction of a voxel takes only the
values k / samples, so at eight draws a level set can be placed in nine ways and
`dice_optimal_level` is choosing among nine bodies; at sixty-four it is choosing among
sixty-five. The draws are therefore not only the candidates being scored, they are the
resolution of the consensus bodies, and the count is the binding constraint on both.

What it costs was measured rather than assumed. Everything in the step is linear in the count
except `metric_medoid`, which compares every candidate against every draw and is quadratic:
with one outline set built per body at about a second, and a boundary comparison between two of
them at about 77 milliseconds, the selection runs about four minutes a model at 64 draws
against a quarter of a minute at 8. That is affordable beside the rest of a run. Note that the
side of the comparison grid was chosen against ensembles of eight draws, and the quantisation
that tied it there no longer binds at this count, so `OCC_RES` is now a number worth measuring
again rather than one that is known to be right.

`reconstruct_lpd.off_lattice_level` moves a requested level down to the middle of the cell
below it, (k − 0.5)/n_draws. That keeps exactly the voxels the level meant -- those where
at least k of the draws agree -- while passing strictly between attainable values, so every
triangle has area. The invariant `prob > used ≡ prob >= asked` is asserted in
`test_off_lattice_level_keeps_the_voxels_the_level_meant` for every attainable level, and
it agrees with the set `test_the_consensus_grid_is_fine_enough_not_to_swamp_the_choice`
already treats as intended. Levels not on the lattice are returned unchanged, so model 10's
0.35 is untouched, and the level reported back is still the one that was asked for, since
that is what names the candidate.

After the fix, both levels come back with zero degenerate faces and watertight.

## 2. What was written anyway

`export_stl` ran `fix_normals()` *before* `fill_holes()`, so the patches hole-filling added
were never oriented; it measured the volume before the repair rather than after; and when
the mesh was still not watertight it recorded that fact in the report and exported the file
regardless. `main()` merged the report into the JSON without a warning, and
`hac26/submission.py` does not check watertightness either, so all ten models passed it
while three were broken.

The order is now fill, then orient, with zero-area faces dropped first because they carry
no orientation, and the volume measured afterwards so the number describes the body on
disk. If the result is still not a closed solid of positive volume it raises instead of
writing.

Run against the three broken meshes, the new order does not merely reject them -- it
repairs them:

```
model 2:  -5.235 -> +5.235   watertight True   z [-1,1]   r_max 1.3530 (unchanged)
model 4:  -5.845 -> +5.845   watertight True   z [-1,1]   r_max 1.4600 (unchanged)
model 7:  -3.912 -> +3.912   watertight True   z [-1,1]   r_max 1.1930 (unchanged)
model 10, model 1 (already sound): identical face count and volume
```

The magnitude is preserved and only the sign changes, which is what dropping zero-area
faces and re-orienting a now-closed surface should do. This is defence in depth: finding 1
stops the degenerate body being built, and this stops one being written if it ever is.

## 3. A score that cannot fail

`rescale_touch_z` divided by `zmax - zmin` with no guard. Its sibling
`shape_library.pose` raises `ValueError("degenerate body: zero z extent")` on exactly this
condition. The chain, traced end to end:

```
flat mesh -> rescale_touch_z          -> [inf, nan]
          -> extent = max|v| * 1.05   -> nan
          -> every parity comparison False, both grids empty
          -> dice(empty, empty)       -> 1.0
```

Two defects compound: the missing guard, and `dice`'s documented "1 when both are empty"
convention, which converts *both voxelisations failed* into a perfect score. It needs a
degenerate input to fire, but it fails upward and silently, which is the worst available
direction.

`rescale_touch_z` now raises. `dice`'s convention is left alone -- it is relied on
elsewhere as a ranking device -- and the empty case is caught where it matters instead:
`voxel.score` raises if either body voxelises to nothing.

## 4-5. What the scorers accept, and what they claim

`mesh_occupancy` decides inside by a parity scan up each column, which is valid only for a
closed surface: a hole inverts every cell in the column through it, so an open mesh scores
*wrongly* rather than failing. Nothing checked. Measured on `Asteroid01.stl` with a hole
punched in it -- geometrically the same solid -- Dice falls to 0.885 at a 0.4-radius hole,
with no diagnostic.

`voxel.load_one_solid` now rejects a multi-solid STL with a useful message and warns when a
mesh is not closed. It warns rather than raises: the truth STLs are the organisers' files
and cannot be inspected from here, and breaking the score stage on their data would be
worse than the thing being guarded against. Our own side is now guaranteed closed by
finding 2.

`side_view.py` computed its only meaningful row, `shipped_vs_truth`, under `if
rf.exists():` with no `else`. A missing reconstruction printed two healthy-looking baseline
rows, wrote well-formed JSON, and exited 0. It now names every missing input and exits
non-zero.

## 6-14. The decision harness

`decision_check.py` is an experiment, and the findings are the ones you put to an
experiment.

**The control that did nothing.** A comment explains that the scoring grid is "fixed at 128
rather than following `OCC_RES` on purpose", so that candidates are not scored on the grid
they were selected on. But `OCC_RES` *is* 128 and the extent is the same variable, so the
scoring call recomputed an array bit-identical to the one already held -- consensus bodies
were scored on the grid they were built out of, and the occupancy was computed twice. There
is now a `--score-res` (default 160) and the script refuses to start if it equals
`OCC_RES`.

**Means over different bodies.** A body that failed to decode two draws was skipped for
*that weight only*, so each weight's mean could be over a different subset. The failures are
not independent of what is being swept: a large guidance is itself what pushes a draw off
the corpus. A weight could win by destroying the hard bodies rather than by reconstructing
them. The means are now restricted to bodies that completed at every weight, and the
dropped ones are named.

**Half the criterion.** The recommended guidance weight -- the number transcribed into
`RECON_GUIDANCE` for the real run -- was chosen by mean Dice alone, ignoring the side-view
column the script had just computed, while the challenge sums both. It is now chosen by the
rank sum of the two, the way `metric_medoid` ranks candidates, the criterion is stated in
the printed line and recorded in the JSON, and a spread below 1e-3 is reported as not being
a margin.

**`inf` in an average.** `mean_of` filtered `None` but not `inf`, and both
`measure_outlines` and `mesh_misfit_by_geom` return it, so one bad body made a rule's mean
infinite and hid every body that did score. Non-finite values are now counted and reported
separately rather than averaged. Rules that produced no candidate print an explicit row
instead of vanishing.

**Ceilings.** `oracle` is the best-Dice candidate, but its side-view number is that
candidate's, not the best reachable one, while the caption called it "the ceiling of any
rule". `oracle_side` now gives the second measure its own ceiling and the caption says the
ceilings are per measure.

**Two index conventions in one row.** `polished_misfit_sigma` had one entry per draw;
`misfit_sigma` skipped draws that failed to decode, and `picked` indexes the latter. One
degenerate draw misaligned them, so anyone asking "did polishing help?" would correlate
mismatched pairs. Both are now indexed by surviving draw.

**Provenance.** `dice_optimal_level` returned the level it had just found produced nothing;
it now falls back to the last level that did. A derived level equal to a fixed one built
and scored the same body twice; both call sites now dedupe, and the row records whether the
derived level coincided with a fixed one. The output JSON gained `val_bodies`, `seed`,
`res`, `side_points`, `churn`, `score_res`, `occ_res`, the bodies in and out of the
summary, and the criterion behind `best_guidance` -- none of which was recoverable before.

`MEDOID_SIDE_POINTS` was passed to the decision stage but omitted from its cache signature,
so changing it reran `reconstruct` and `score` but not `decision`, leaving stale numbers
that claimed to describe the new setting.

## 15-16. The score stage could not start

`hac26/scoring/voxel.py` copied the `sys.path.insert(..., parents[1])` idiom from
`scripts/`, but it sits one level deeper, so it inserted the *inside* of the package rather
than the directory containing it; `side_view.py` had no such line at all. Since
`_venv_setup.sh` deliberately installs the dependencies by name and not the package --
sound reasoning, and these two are the only entry points outside `scripts/` -- the score
stage died with `ModuleNotFoundError: No module named 'hac26'` on a fresh machine, after
training and all ten reconstructions had completed. Fixed at the call sites in both
wrappers, which keeps it out of `SRC_OUTPUT` and reruns nothing.

Both `side_view.py` invocations also shared the default `--out`, so the flow's run
overwrote the convex baseline and only half the comparison survived. They now write
`runs/side_view_convex.json` and `runs/side_view_lpd.json`.

---

## Left alone deliberately

**17. The width of a consensus body.** `restore_constraints` caps the xy radius at
`R * 1.03` rather than setting it, and the cap never binds, so consensus answers ship at
0.953 to 0.996 of the published radius while draws ship at exactly 1.000. The challenge
states R is the *minimal* enclosing radius and the released truths sit on it
(1.1198 against a published 1.12, and so on), so a body 4.7% narrow loses Dice outright and
carries a systematic offset into the side-view measure; inside `metric_medoid` it is also
compared against draws that are all wider.

The comment defending the cap argues the level-set offset is a distance while a rescale is
a proportion, and that the two are "the same half per cent" -- a premise the repo's own
output contradicts at 4.7%. But that does not establish the opposite: scaling a body that
is narrow at one azimuth inflates it everywhere, and which is better for the summed score
is an empirical question, not an argument. It is exactly what the decision harness exists
to settle, and the harness can now measure it. Changing it silently could make the results
worse, so it is left as it is and flagged.

**18. Consensus bodies are scored in-sample.** In `metric_medoid`, a draw is scored against
the *other* draws, but a consensus body -- which is a function of all of them -- is scored
against every draw that helped build it. That is an in-sample score competing against
out-of-sample scores. The bias was measured at 0.007 to 0.022 Dice as the draws disagree
more, against a real difference between candidates of about 0.02 that the code's own
comments cite; a case was constructed where in-sample picks the consensus body and the
honest estimate picks a draw. The bias points one way, toward consensus bodies, which is
what four of the ten answers are -- and those four are the ones that came out broken.

The honest fix is to score a consensus body against draw *j* using the level set built from
the other draws, which costs n_draws extra marching-cubes runs and surface samplings per
level. That is a redesign with a real compute cost and it changes which body is submitted,
so it needs a decision rather than a commit.

## Still open

**19.** The occupancy grid is a cube sized by the largest single coordinate, but the pose
fixes |z| ≤ 1 while R ranges from 0.67 to 3.95, so a wide body loses vertical resolution:

```
model 1   extent 1.160   110 of 128 z-cells
model 9   extent 1.050   122 of 128 z-cells
model 10  extent 4.093    31 of 128 z-cells
```

This is not only a reporting effect: `reconstruct_lpd` builds its selection grid the same
way, so model 10's answer is *chosen*, and its consensus bodies *built*, on 31 vertical
layers. The challenge says "the same bounding box", not a cube. An axis-aligned box would
fix it, and it changes every selection, so it should be measured first.

**20.** `binary_closing(img, iterations=2)` fills any concave notch narrower than roughly
0.018 model units at res 512. Closing only ever adds pixels, so a convex outline is
unaffected while a concave truth is pushed toward convexity -- precisely the feature the
module docstring says the measure exists to detect. It is the one uncommented line in a
file where everything else is justified at length. Not verified numerically; worth an
experiment before touching.

**21.** `decision_check`'s docstring says the held-out bodies "never enter training".
Gradients never saw them, but `train_lpd` sets `best_state` -- the shipped weights -- at
whichever step scored best on those same bodies, so model selection did. Separately,
`held_out(n, k)` is the first k of one fixed permutation, so `held_out(n, 16) ⊇ held_out(n,
4)`: running with a larger `--val-bodies` than training used silently scores bodies the
flow did train on, and nothing detects it. Fixing this means splitting the held-out set in
the training script, which is a training-side decision.

**22.** The side-view module calls itself "the challenge's second scoring measure" but
returns unbounded distances, lower-is-better, while the spec requires [0, 1],
higher-is-better, summed with the voxel score. It works as the internal ranking proxy that
the pipeline uses it as; it is not the measure, and the two JSON files cannot be added.

**23.** Every consensus answer reports `answer_misfit_sigma: inf`, which blanks the
`--hold-out-geoms` generalisation check for those models. Finding 1 explains three of the
four -- a mesh with 6000 zero-area triangles will break the radiosity quadrature -- but
model 10 is geometrically clean and still reports `inf`, so at least one cause remains.
`mesh_misfit_by_geom` catches `RadiosityError` and discards the message, and the three
possible causes carry different messages, so ten minutes on the GPU machine will say which.
Do that before writing a fix.

---

## Tests

`tests/test_scoring.py` is new: `hac26/scoring/` previously had no tests at all. It covers
the pose (uniform scale about the axis, idempotent, raises on a degenerate body), the voxel
measure against the challenge's own formula, the side-view measure (symmetry, a body
against itself, that the extent bounds every projection, that a wider body reads as further
away), and what `export_stl` will and will not write.

Three regression tests were added to `tests/test_decision.py`: that a consensus body at a
majority level is a closed solid with no degenerate faces, that the level nudge keeps
exactly the voxels the level meant, and that `dice_optimal_level` returns a level that
produces a body.

These were checked to bite. With `off_lattice_level` reverted to a no-op, the consensus
test fails with 536 degenerate faces and `watertight=False`. This mattered because the
existing `test_candidates_are_scored_against_the_draws_only` passes identically with the
behaviour it claims to pin removed entirely, so a passing test here was not evidence.

## Consequences for the pipeline

`reconstruct_lpd.py`, `output.py` and `decision_check.py` are hashed into `SRC_OUTPUT`, so
the `decision`, `reconstruct` and `score` stages will rerun on the next invocation. That is
correct -- the code that produces those artefacts changed -- but it is hours of GPU, so
bundle anything else that needs a rerun with it.

`export_stl` can now abort a model rather than writing a bad file. With finding 1 fixed it
should not fire.

**The ten STLs in `results/lpd/` are still the old ones.** They were produced before
`dice_optimal_level` existed -- every committed JSON records `consensus_levels: [0.35, 0.5,
0.65]`, a constant the current code no longer has -- so they were not chosen by the rule the
repo now contains, independently of being broken. They need regenerating. Re-exporting the
three broken ones through the fixed `export_stl` repairs the files in place if valid
geometry is needed before a full rerun, but it does not change which body was chosen.

## Reproduction

The measurements above need numpy, scipy, scikit-image, trimesh and CPU torch, plus
`fast_simplification` and `rtree` for the full suite; no GPU and no `dataset/raw`.

```
python -m venv .venv
.venv/bin/pip install numpy scipy scikit-image trimesh pytest torch fast_simplification rtree
.venv/bin/python -m pytest -q            # 168 passed, 6 skipped
```

The shipped-STL table is read straight out of `results/lpd/` with `hac26.stl_io.load_stl`:
signed volume by the divergence theorem over the faces, degenerate faces by cross-product
norm below 1e-14, `r/R` against `hac26.conventions.CYLINDER_R`. The before-and-after on the
consensus levels uses an 8-draw synthetic ensemble of a ball with a dent in a different
place in each draw, at grid side 48 and extent 1.2, which reproduces the same k/n_draws
lattice the real draws have.
