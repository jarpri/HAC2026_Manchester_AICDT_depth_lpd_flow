# What the depth field can hold

The body is a convex core displaced inward by a depth carried on directions
(`hac26/field.py`). How many directions it is carried on decides what a fit can reach, and
the question is not how well the family can approximate a shape but how well it can hold
the one shape the challenge actually poses: a body whose own hull is wrong in the places a
convex inversion cannot see.

## The target

The released non-convex body, model 3, posed into the canonical frame. Against its own
convex hull it has 0.768 of the volume and 0.869 of the overlap, so the whole of what any
correction has to supply is that 0.13 of overlap. The other two released bodies are their
own hulls to 0.997 and 0.9999 and say nothing about a correction at all; nothing here is
inferred from them.

## The fit that measures the family

The protocol is the one this note used for the lattice, so that the rows are comparable. The
core is pinned to the **convex stage's answer** for model 3 -- not to the body's own hull,
because that answer is what a reconstruction actually starts from -- the correction is fitted to
the body, the zero level is extracted on a 96-cubed grid over a half-width of 1.30, and the
overlap is taken against the released shape at 128 cubed.

What is fitted is an identity rather than an approximation: on the body's own surface the field
has to vanish, so the depth wanted in the direction of a surface point is exactly minus the
core's field there, which is the distance from that point out to the core's surface along the
normal. The fit is therefore a linear least-squares problem over points sampled by area over the
body's surface, twelve per coefficient, with a ridge of 1e-3 of the mean diagonal, and it needs
no signed distance anywhere.

| correction | coefficients | surface residual | deepest carve | dice | components |
|---|---|---|---|---|---|
| lattice, 24 sites per axis | 13824 | — | — | 0.9907 | 1 |
| depths on 640 directions | 640 | 0.0089 | 0.682 | 0.9914 | 1 |
| depths on 2560 directions | 2560 | 0.0043 | 0.682 | 0.9966 | 1 |

**640 depths hold the body as well as 13824 amplitudes, and 2560 hold it better.** The surface
residual -- how far the fitted field leaves the body's own surface from the zero level, in body
units -- halves between the two node counts, and at 2560 it is a hundredth of the carve it is
holding. The reason is not resolution but direction: a Gaussian in space smooths the carve at
the kernel's width along the surface normal as well as across it, and the normal is the
direction a shadow edge lives in, while a depth is smoothed across the surface only.

**The curve misfit of these bodies is not measured here and is the first thing the GPU run
should read.** The lattice's floor on this protocol is a misfit of 0.026, against 0.011 for the
released body itself and 0.078 for the convex answer, and that column is what decides whether a
fit can ever be selected on the curves. It is a numpy ray cast over thirty-five thousand
triangles at 96 pixels, 24 frames and 21 geometries; one geometry at four frames does not finish
in a minute on a CPU. The overlap column above says the family holds the body; only the misfit
column says whether the curves can find it, and the two are measured elsewhere in this
repository to rank bodies differently.

**The star-shaped bound binds.** Started from the convex stage's answer rather than from the
body's hull, the deepest carve the body needs is exactly at the bound at both node counts, with
five of the directions held there. It costs almost nothing here -- five directions out of 640
and out of 2560 -- but it is not slack, and on a body whose convex answer is inflated further it
would be the thing that decides how deep a carve can go.

## What each rung of the ladder can hold

The family is not the only cap on what a fit can reach. The solver never searches the depths
directly; each of its stages searches the real spherical harmonics up to a degree, so a stage's
ceiling is what that degree can hold and not what the 2560 directions can. The same protocol
answers it: the core is pinned to the convex stage's answer for model 3, the body's own surface
is sampled, and the depths are solved inside each stage's coordinates together with the nine
reshaping coefficients the fit carries beside them. The columns are scaled the way `CarveFit`
scales them, by the peak depth each makes in the field, without which the unnormalised
harmonics reach 1e31 by degree 24 and the solve returns nothing. The overlap below is taken on
a 64-cubed grid rather than the 128-cubed of the table above, which is why the last row reads
0.994 where that table reads 0.9966; the rows here are comparable with each other.

| stage | coefficients | rms field residual | volume | overlap |
|---|---|---|---|---|
| the convex answer, uncorrected | 0 | 0.192 | 3.279 | 0.706 |
| degree 4 | 16 | 0.044 | 2.075 | 0.929 |
| degree 10 | 112 | 0.020 | 1.938 | 0.969 |
| degree 16 | 280 | 0.010 | 1.927 | 0.985 |
| degree 24 | 616 | 0.007 | 1.871 | 0.992 |
| degree 32 | 1080 | 0.005 | 1.907 | 0.992 |
| all directions | 2560 | 0.004 | 1.900 | 0.994 |

The body's own volume is 1.885, so every rung from degree 4 up recovers it to within a few per
cent, and the overlap is what separates them. What the table settles is that the ladder's
ceiling is not what limits a reconstruction: the coarsest stage of all can hold the released
body at 0.929 and the third at 0.969, both above what the challenge is being aimed at, and
degree 24 buys 0.007 over degree 16 for twice the renders an iteration.

The cost runs the other way, since a secant Jacobian spends one render per coefficient. A
degree-24 iteration is thirteen degree-4 iterations, so a ladder that spends its iterations
evenly spends most of them on the rung that carries least. That is why `DEFAULT_STAGES` gives
the coarse rungs most of the iterations and leaves degree 24 as a refinement, and why the
polish runs at degree 16.

This is an upper bound on a stage and not a prediction of a run. It is fitted to the body's own
surface, which a reconstruction does not have; what a reconstruction has is the curves, and
whether they determine the same body is the separate question notes/identifiability.md asks. The
table says only that when a fit stops short of these numbers, the degree cap is not the reason.

## What the grid of starts can hold

The ladder is local, so what it can reach is bounded by where it begins as well as by what its
coordinates span. The grid of starts was eight spherical caps by three radii by three depths,
and a cap is a crater: the field it writes is a single connected region of one depth. No
setting of those three numbers makes a neck, so a body with one cannot be started near.

Whether a scored body has one is the question, and it is not settled by the released three.
Sweeping sixty directions for an interior minimum of the body's width, model 3 reaches 23 per
cent of its width against the sawed-off cube's 20 -- and the cube is exactly convex, its volume
equalling its hull's, so twenty per cent is what this measure reads on a body with no neck at
all. Model 3 is short of its hull by a fifth of its volume without being bilobed. What does say
a neck must be reachable is the shape library, which gives its bilobe and trilobe families just
under a third of the bodies between them; that is this project's own statement of what a scored
body may be, and a grid that cannot start near one is missing a third of its job.

The second family is a neck. A body's two lobes lie on an axis, and the neck between them is
the band of carve around the great circle perpendicular to it, which is the directions u with
u . axis near zero; `waist_depths` writes a constant depth over the band within a half-width of
that circle. The axes are the eight the caps use, in the band about the equator, together with
the spin axis itself, because the challenge fixes the axis of rotation but not which way a body
was mounted on it: a waist about an equatorial axis is the neck of a body lying across the
axis, and a waist about the spin axis the neck of one standing on it. Half-widths run from
twelve to thirty degrees, and the mean spacing of 2560 nodes is about four degrees, so the
narrowest band is six spacings across and is carried rather than smoothed away.

The uniform shrink came out of the grid at the same time, from three levels to one. Measured on
the released bodies, the canonical pose divides a uniform inward displacement almost entirely
back out, since z is rescaled to span [-1, 1] and the widest radius to one and an offset is
nearly a similarity: the deepest level of the range moves the canonical volume of model 3 by a
twentieth and of model 1 by a hundredth. Three levels were therefore three bodies that a
ranking cannot separate, and they would have filled a shortlist with copies of one start. The
level is kept non-zero because its purpose is the linearisation rather than the body -- it puts
the secant probe of the hull coefficient somewhere the hull is already moving.

The grid is 72 caps and 81 waists, and it is swept whole rather than truncated, because the two
questions about a start cost differently. Ranking a start needs the number the fit minimises at
that start, which is one render; telling a start that descends from one that does not needs the
coarse stage, which is about a hundred. So every start is rendered once and ranked, and the
best eight are given the coarse screen. The whole grid at one render each is 153 renders
against the roughly twelve thousand a body costs, and the screen it feeds is unchanged.

## The node set, and why it is not a plain spiral

A quarter turn about the spin axis is an exact symmetry of the problem and is what gives the
corpus four training pairs per body at no cost in renders. That is only an exact symmetry of
the *code* if turning a body permutes its depths. A plain Fibonacci spiral is not invariant
under a quarter turn, so a turn would have to resample the field instead -- and resampling
smooths it. Measured on model 3, the resampled code differs from a fit of the turned body by
twelve per cent of the depths in the root mean square, which is five times the fit's own
residual, and by forty per cent of the deepest carve at the worst direction. A corpus
augmented that way would teach the flow codes that no fit of those bodies would produce.

The node set is therefore a quarter of the sphere's worth of directions and its three
rotations, and the quarter carries a spiral of its own: latitudes evenly spaced in cosine,
azimuths advancing by the golden ratio *of the quadrant*. Advancing instead by the whole
sphere's golden angle and folding it into the quadrant leaves consecutive latitudes near the
poles at nearly the same azimuth, and the closest pair of nodes then sits at half the spacing
of a plain spiral. Taking the golden ratio inside the quadrant costs nothing:

| node set | nearest neighbour, in units of the mean spacing | | | sixth neighbour |
|---|---|---|---|---|
| | min | median | max | max |
| plain spiral | 0.872 | 0.950 | 0.990 | 1.590 |
| quarter-turn invariant | 0.852 | 0.954 | 0.988 | 1.518 |

The turn is then a shift of whole blocks, exact to machine precision.

## The extraction

The surface is extracted by FlexiCubes on a grid of `EXTRACT_RES` cubes over
`[-EXTRACT_EXTENT, EXTRACT_EXTENT]`. Two things about it are worth recording because they
decide whether a fit of thousands of renders is affordable at all.

The grid itself is built arithmetically rather than by deduplicating the corners of every
cube separately. The vendored construction sorts eight times as many points as the grid has,
which at 128 cubes a side is seventeen million rows of three and several gigabytes; it does
not complete in a seven-gigabyte machine. The arithmetic construction agrees with it exactly,
vertices and cube corners alike, and the suite checks that at four sizes.

The core's field and the depth's neighbour search are both cached on the identity of the
points they were computed for. Through a fit of the carve the support does not move and the
grid is one tensor, so both are computed once and reused on every later render. Measured on
one core pair: a first extraction at 88 cubes a side costs 14 s and every later one 0.36 s,
and at 128 cubes 38 s and 0.95 s. Without the caches every render pays the first figure,
which is the difference between a fit that takes half an hour and one that takes a day.
