# What is minimised, and why it is not the misfit

A reconstruction is scored on how much of the body it overlaps. It is fitted on how well its
curves match. Those are different functionals and this note measures how differently they
rank the same bodies, which is what decides whether minimising the second can ever return a
body that scores on the first.

## The misfit reports sharpness as much as shape

Take the convex stage's answer for model 3 and move it in two ways. One is toward the body:
blend the answer's signed distance with the body's, which changes the gross shape and leaves
the surface as smooth as it was. The other is a corrugation: add a random field of one grid
cell's width and a few hundredths of a body radius in amplitude, which leaves the gross shape
where it is.

Measured, through an orthographic ray cast with the measured photometry, at 24 phases and 96
pixels over the 21 distinct geometries: every corrugation of two to four hundredths lowers the
misfit by 0.013 to 0.023 and moves the overlap by 0.003. Moving thirty per cent of the way to
the body lowers the misfit by 0.011 and moves the overlap by 0.081. So a corrugation buys more
misfit than the right direction does and buys almost no overlap with it.

The reason is that both reductions are sums over a thresholded image. Their error is the
quantisation of the boundary of the lit region, and a finer surface has more boundary to place
more accurately. Nothing about that is a property of this renderer; the organisers' own
reduction has it.

## Area is the functional that separates them

The two directions differ in one measurable way. A corrugation raises the body's surface area
by 0.55 to 0.71; moving toward the body lowers it by 0.38 to 0.72. The surface area of a
closed body is the total variation of its indicator, and it is what charges a rough surface
while leaving a smooth dent of any depth nearly free. A ridge on the coefficients does not: it
charges by how large a coefficient is, so it prefers a shallow answer to a deep one, and
measured against the released body it prefers the fit's own answer by eight per cent.

## The weight has to be scale free

Under misfit-squared plus a weight times area, the released body is the minimiser of a span of
fifteen posed bodies only for weights at or below 6.6, while a corrugation one cell wide is not
charged until the weight is above 34.5. There is no fixed weight that does both; the window is
empty by a factor of five.

The reason is that the gradient of the squared misfit falls with the misfit while the gradient
of an area term does not. Making the objective scale free in the misfit,

    G = log chi^2 + mu A,

fixes the balance: stationarity then compares a relative change of misfit with an absolute
change of area. The window opens, and where it opens to depends on the representation, which is
the subject of the next section. In the one this repository now uses the top of it is measured:
around the released body a further uniform carve of 0.03 is refused for any mu below 2.48, so
above that a weight walks a correct answer away, and `AREA_WINDOW` stops at 2.40. The bottom,
0.30, is assumed: nothing breaks below it and the penalty simply does less, and what is measured
is the end of that road -- with the penalty off altogether the same ladder converges having
barely moved, at an overlap of 0.7226 against the 0.7477 it reaches with it, so the penalty is
worth 0.025 of overlap and removing it is not the fix for anything. `AREA_WEIGHT` sits inside
the window and `CarveFit` refuses a value outside it.

The penalty does not forbid concavities, and that was tested rather than argued. A hemispherical
pit of radius 0.15 cut into the released body at 35 places over its surface costs between
0.004 and 0.073 of objective and is paid between 0.005 and 0.964 by the misfit: 33 of the 35
are paid for, by a median factor of ten. The two that are not are places where the pit changes
the curves by less than the model error, which the data do not determine in any case.

## What a displacement changes, and why the ladder has a ceiling

The measurements above were made on a representation that added blobs of material. The body is
now the convex core displaced inward by a depth, and for a *displacement* the area behaves
differently in a way that decides the recipe rather than the weight.

The first variation of area under a normal displacement d of a surface is the integral of d
against twice the mean curvature. At first order the area therefore sees only the
curvature-weighted average of the displacement, and an oscillation of zero mean is nearly free;
what a rough displacement raises is the second-order term in its tangential gradient. In a
lattice of blobs a corrugation is new structure that genuinely adds surface at first order; in a
depth field it is a wrinkle in a surface that already exists.

Measured from the convex answer of model 3, a displacement at the scale of the node spacing
lowers the log misfit by 0.15 to 0.57 and changes the area by anything between -0.045 and
+0.112 depending on the seed: one seed in three lowers the area *and* the misfit together, so no
weight charges it at all, and the others need a weight above 1.9 while the released body stops
being a minimiser above 2.48. **The window is empty for that direction and no weight closes
it.** What closes it is taking the direction out of the search: the ladder in
`hac26/solvers/gauss_newton.py` stops at an angular degree whose wavelength is ten extraction
pitches at the surface, which is not a corrugation, and the same ladder run without that cap
ends at an overlap of 0.677 -- below the 0.709 it started from -- while lowering the misfit.
Inside the capped search the weight is back inside a window it can sit in, and the job it is
left doing, which it does, is keeping the body a minimiser against a further uniform carve.

The same bound is why a stage that searches random smooth fields on the nodes builds them by
smoothing over at least two node spacings. One application of the node kernel to white noise is
exactly the direction above.

## When the penalty helps, and when it does not

A concavity adds surface, so a body carries more area than its own hull, and an area penalty
taken at face value should push away from a carved body rather than toward it. What makes it
push the right way on model 3 is not the concavity but the inflation: a convex inversion
imitates shadowing by enlarging the hull, and the enlarged body it returns carries more surface
than the true one despite the true one's dents. That is a property of the convex answer and not
of the body, so it has to be checked body by body rather than assumed.

Posed canonically, the released bodies and the convex answers the correction starts from are:

| body | truth area | truth volume | answer area | answer volume |
|---|---|---|---|---|
| 1 | 12.280 | 3.965 | 12.061 | 3.928 |
| 2 | 14.820 | 3.883 | 12.612 | 3.409 |
| 3 | 9.091 | 1.885 | 10.915 | 3.280 |

The truths are the released meshes themselves and the answers are extracted from their supports
at a coarse resolution, which moves the last digit and not the comparison.

Only model 3 was inflated. Its answer holds 1.74 times the body's volume and 1.20 times its
area, so at the weight above the penalty pays about 1.6 of objective toward the truth and
removes the barrier the misfit alone puts in the way. Models 1 and 2 were not inflated at all --
their answers hold *less* volume than the bodies do -- and on model 2 the body carries 2.2 more
area than its answer, so the same term charges the truth about 2.0 to get there and the misfit
has to find that much on its own.

This is the mechanism the gate on the convex misfit is really guarding, which is more than its
own comment claims for it. An answer that over-inflates is an answer that explains the curves
badly, so the number that says a body has concavity to find is the same number that says the
penalty will be pointing toward it; on the two bodies where both are known they agree, model 1
sitting near the noise with an answer that was never inflated and model 3 far from it with one
that was. Two bodies are not a calibration, and what that gate needs is the same corpus the
floor needs. What the table settles is narrower and firmer: the penalty is not a property of
the objective alone, and a body whose convex answer already fits should be left alone for this
reason as well as the one recorded below.

## The floor on the volume

The penalty's risk is the hull shrink, and the trust region bounds each step rather than the
walk. Measured, the remedied ladder on model 3 without a floor takes the volume from the
convex answer's 2.518 to 0.864 and the overlap from 0.7086 down to 0.6541, lowering the misfit
throughout; with the volume floored at half the convex answer's the same ladder stops at 1.262
and ends at 0.7477. The floor is therefore a stop on a walk and not a bound on a step, which is
why it lives in the render `scripts/reconstruct_gn.py` builds -- a body below it is refused the
way a body the forward model cannot render is refused -- rather than in `CarveFit`.

It is calibrated on one body. Model 3's own volume is 0.580 of its convex answer's, so the
floor sits below it with room; what would make the number principled rather than calibrated is
the distribution of that ratio over the shape library, and that is the same corpus the gate on
the convex answer's misfit would need.

## How wide the trust region on each step has to be

Because the floor takes the walk, the region on each step is left bounding one step, and its
width is then a question about the journey rather than about the collapse. The journey is the
gap the floor's own calibration names, and it can be read straight off the meshes without a
fit: posed canonically, model 3's released body holds 1.885 of volume against the 3.280 of the
convex answer the correction starts from, a ratio of 0.575, so a correction that arrives has
given up two fifths of a volume. A region of a fraction f crosses that in at least
log(0.575) / log(1 - f) maximal steps, which is seven steps at a twelfth of the volume and two
at a quarter.

Seven is more than the screening has. A start is ranked on two iterations of the coarse stage,
so under the tighter region every start reaching that ranking is still most of the way back at
the convex answer they were all given, and the ranking is sorting them on a correction that has
not happened yet; the whole point of a designed spread of starts is the shape of the correction
each one carries, and two capped steps have to be enough to tell one from another. At a quarter
those two iterations are about the length of the journey, so the screening compares bodies
rather than intentions. Nothing else moves with it: the line search still takes the best of five
lengths inside whatever region it is given, the depth region still bounds the carve, and the
floor still refuses a body that has gone too far.

The floor is what binds after this change rather than the region, which is where a stop on a
walk belongs. Two maximal steps at a quarter reach 0.5625 of the convex answer's volume against
a floor at 0.50, so on a body whose own volume sits where model 3's does, the fit arrives with
the floor a tenth of a volume away. On a body more strongly non-convex than the released one it arrives
underneath it, and the floor is then not a stop on an overshoot but a ceiling on the answer.

How much of the library that covers can be read off the library's own design rather than
guessed. `LibrarySpec.convexity_bins` opens at 0.55 of the hull volume and
`convexity_shares` gives that deepest band a fifth of the bodies, unbounded below, because a
sampler whose deepest edge is higher stops at the first body that crosses it and never makes a
deeply carved one. The floor is measured against the convex *answer* rather than the hull, and
the answer is the larger of the two -- that is why the correction from it shrinks the hull, and
model 3's own correction shifts it inward by 0.087 body units -- so a body's volume over the
convex answer sits below its volume over its hull. A floor at half the convex answer's volume
therefore stands at or above the top of the band the library gives a fifth of its bodies to.
Against the working assumption that the secret bodies are strongly non-convex, that is a
ceiling in the wrong place, and it is a stronger bias toward convex answers than the gate on
the convex misfit is: the gate declines to correct a body, while the floor corrects it and
stops it short. Setting it from the library's own bands rather than from the one released body
is what it needs, and nothing about the correction below depends on the value.

## What the penalty exposed

Three things had to be added with it, and each is a defect the ridge was hiding rather than a
price of the penalty.

A direction the curves cannot see has no curvature in the Gauss-Newton matrix, so damping
relative to that curvature does not bound a step into it. While the right-hand side was the
Jacobian applied to the residual, it lay in the range of the Jacobian and the question did not
arise. The area's gradient does not lie there, and the coefficients then grow without limit in
directions that do not move the surface: measured, a first coarse step reached a carve
seventeen hundred body radii deep, while rendering a body of overlap 0.748. Flooring the
damped diagonal at a hundredth of its mean bounds it, and the same step then reaches a carve
of 0.055 and the same body.

The cheapest area in this representation is a hull shrink. A step has to be shortened to stay
inside a trust region rather than merely refused by it, or a coarse stage collapses the body
and every trial is refused; the volume's and the carve depth's own secant derivatives say how
long a step may be, and both are needed, since a step that carves in one place and fills in
another leaves the volume where it was.

And a body whose convex answer already explains its curves has no concavity to find. There the
penalty trades overlap for a misfit the body does not need, and it does so while improving the
misfit, so nothing downstream can catch it. The ratio of the convex answer's misfit to the
channel's model error is read before the fit instead: it is 7.9 on model 3 and 1.8 on model 1,
and running the penalty on model 1 costs 0.166 of overlap.

Those two numbers were read under a calibration that has since changed, and the unit moved with
it. The released curves carry a measured noise of about 0.001 per curve against a model error
in the hundredths, so the residual is divided by very nearly the calibration's own eta and every
sigma quoted here scales inversely with it; taking the sawed-off cube out of the calibration,
which is right for its own reasons, lowered eta and lifted both anchors together. A gate placed
between 1.8 and 7.9 therefore sits below the near-convex body once eta falls by much more than a
factor of two, which is the direction it fell. The threshold is set at 6.5, the band that stays
above model 1 and below model 3 under the calibration the anchors were read on and under the
narrower one in use, and the asymmetry says to sit high inside that band rather than in its
middle: correcting a body that should have been left alone costs 0.166 of overlap where
correcting one that needed it gains 0.045.

What replaces the band is one measurement, not a redesign: render the convex answers of models 1
and 3 against their own curves under the calibration actually being used, and read the two
anchors again. That is two renders and it is worth doing, though it is no longer urgent: the
threshold now flags a body rather than refusing it, because the body it was measured on is one
whose convex inversion had already recovered it, and a secret body's fate is not read from
that. The correction runs either way and `scripts/select_answers.py` decides on geometries held
out of the body's own fit.

## What it is worth

The numbers on model 3, from the convex stage's answer, through the same stand-in renderer,
are in the table below; the branch's own previous answer is the row marked as the ridge.

| what was minimised | misfit | overlap | carve depth | hull shift |
|---|---|---|---|---|
| nothing: the convex stage's answer | 0.0787 | 0.7080 | 0 | 0 |
| the misfit, under a ridge on the coefficients | 0.0376 | 0.7205 | 0.224 | +0.016 |
| the misfit and the area, then polished on the misfit | 0.0557 | 0.7533 | 0.226 | +0.042 |

Both fits are from the same start, on the same curves, through the same renderer, at 24 phases
and 96 pixels. The penalised one reaches an overlap 0.045 above the convex answer where the
other reaches 0.013, and its hull correction is three times larger and in the direction the
body needs, the body's own being +0.087.

It reaches that at a *worse* misfit, which is the whole point and the thing to watch. A
selection that reads the misfit alone prefers the second row to the third and would throw the
better body away, so `scripts/select_answers.py` reads the functional the correction was
fitted under. Nothing that compares two corrections may read the misfit on its own, and that
includes the polish: it minimises the misfit alone by design, so the body it starts from is
kept and the two are scored under the penalty, the better one being the one written. A polish
that buys its misfit with surface is then visible instead of exported.

That selection stands a correction where it beats its own convex answer on the cameras held
out of its own fit, which is evidence about the body being decided. `--calibrate` tightens it
to the ratio model 3's refinement reached, and the table above says why that is a demanding
test rather than a neutral one: model 3 is the released body whose convex answer was inflated,
so the area term pays toward its truth and is part of the ratio it set. A secret body whose
convex answer was not inflated has to find the same ratio with that term working against it.
That is why the tightening is offered and not required -- a public run that fell short would
otherwise stand every convex answer in the submission, and a convex answer is not a safe
default but a body known to be missing the concavities the challenge is about. Either way the
rule is biased toward the convex answer, which stands unless the correction is shown to be
better; the selection is seconds to re-run once the scored numbers exist, and it is the place
to revisit with them in hand.

Two solvers now offer a correction for each body, and the same rule decides between them:
each is compared with the same convex answer on the same held-out cameras, and the one that
removed more of its misfit is the candidate. Ranking them needs both to be measuring one
thing, so they minimise the same functional, extract and measure the written body through one
function at one resolution, and hold out the same cameras
(`reconstruct_map.export_measure`, `data_io.held_out_geoms`). Comparing a body fitted under
the penalty with one fitted on the misfit alone, or two bodies scored on different cameras,
would be the comparison this note says may never be made.

The volume falls from the convex answer's 2.52 to 1.27 against the body's 1.46, so the penalty
overshoots the shrink by about a seventh of the volume even with the trust region holding each
step. That is the direction its risk was known to lie in and it is the first thing to look at
if the overlap stops improving.
