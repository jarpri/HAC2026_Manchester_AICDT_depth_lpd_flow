# The submitted bodies

`Asteroid04.stl` to `Asteroid10.stl`, one per scored model, in the challenge pose: the
rotation axis is z, the body touches z = 1 and z = -1, the light is at minus infinity on x,
and the body stands as it did at frame 0 of the lightcurves. Each file is one watertight
component of positive volume with consistent winding whose largest distance from the axis
is the published bounding radius, which `scripts/check_submission.py results/submission`
verifies.

Each file is either the convex stage's answer for that model or a correction of it that was
accepted in its place, and `selection.json` beside them says which, per model, and on what
numbers. The convex answers are written by `scripts/make_submission.py` from
`models/lpd_convex.pt` and the released Blender curves of each model, in seconds on a CPU;
the corrections come from the non-convex path and enter only through
`scripts/select_answers.py`, which requires a correction to beat its convex answer on
cameras kept out of its own fit by as much as a correction of a public model did, and
requires that public correction to have moved that body toward its released shape. Until a
correction has passed, every file here is a convex answer.

The same recipe applied to the public models is under `results/public/`, and
`results/public_scores.json` holds their scores under the organisers' two measures together
with the checkpoint digest and the channel each model was inverted from.
