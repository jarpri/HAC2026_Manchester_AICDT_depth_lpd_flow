"""Fitting a body to its curves by damped Gauss-Newton, without differentiating the renderer.

The unknown is the pair (c, a): nine coefficients that reshape the convex core and one depth
per node of the correction field. Both move in the same step, because the correction from a
convex inversion's answer to the body is a conjunction of the two and neither half lowers the
misfit on its own. A method that takes one coordinate direction at a time, or one block at a
time, walks away from the answer.

c and a are the same function on the sphere at different angular scales -- c carries its
degrees up to two and a the rest -- so the search over them is one ladder in angular degree,
from a stage that can only move the hull to a stage that can carve a dent a tenth of the body
across. Every stage's coordinates are scaled so that one unit of a coordinate is one body unit
of depth in the field it actually makes, which is what lets one finite-difference step and one
trust region serve all of them.

The ladder stops at a degree whose angular wavelength is many extraction pitches, and that bound
is load-bearing rather than tuning. The first variation of area under a normal displacement is
the integral of the displacement against twice the mean curvature, so an oscillation of zero
mean is nearly free of area at first order; a displacement at the node scale therefore buys
misfit almost without paying for it, and no weight on the area charges it while still leaving
the body a minimiser. Capping the degree takes that direction out of the search instead, which
is the only thing that removes it.

No derivative of the renderer is used. The chain's derivative with respect to the vertices is
missing its boundary term on the pure-torch rasteriser and has never been compared against a
finite difference on the other, and the misfit is in any case a rough function of the code at
the step sizes a line search takes. A secant Jacobian costs one render per active coordinate
and removes both risks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..field import RADIAL_DEGREE, real_sh

__all__ = ["Stage", "DEFAULT_STAGES", "SCREEN_STAGES", "POLISH_STAGES", "CarveFit",
           "degree_basis", "node_subspace_basis", "cap_depths", "waist_depths",
           "conjunction_start", "N_CAP_STARTS", "N_WAIST_STARTS",
           "start_recipe", "N_STARTS", "SHRINK_RANGE", "STEP_G", "STEP_C",
           "AREA_WEIGHT", "AREA_WINDOW", "VOLUME_TRUST", "DEPTH_TRUST", "TARGET_SIGMA",
           "DAMP_FLOOR", "SMOOTHING_LENGTHS", "CAP_RADII_DEG", "CAP_DEPTHS", "CAP_BAND_DEG"]

# The secant steps, in body units of the canonical pose. Each is the size the answer's own
# correction is likely to have, not a small number. Measured on the released non-convex body:
# the response of a curve to a carve is concave in its depth, so the direction a column points
# in has not settled at a probe of 0.05 and has by 0.20, and a Jacobian taken at the shorter
# probe is a linearisation of the wrong regime. STEP_G is a carve depth and STEP_C a
# displacement of the hull.
STEP_G = 0.20
STEP_C = 0.08

# Weight of the surface area in the objective, in inverse area of the canonical pose.
#
# What the penalty is for depends on the representation, and this one is a displacement of a
# surface rather than a sum of blobs. The first variation of area under a normal displacement d
# is the integral of 2Hd over the surface, so at first order the area sees only the
# mean-curvature-weighted average of the displacement and an oscillation of zero mean is nearly
# free; what a rough displacement raises is the second-order term in its tangential gradient.
# Measured on the released non-convex body, a displacement at the node scale is therefore
# profitable at any weight that leaves the body a minimiser, and in one seed of three it lowers
# the area as well as the misfit, so no weight charges it at all. The degree cap on the ladder
# is what removes that direction, not this weight.
#
# What the weight is left doing, and does, is keeping the body a minimiser against a further
# uniform carve. The top of the window is measured: above it the released body is beaten by
# itself carved a little further, and a weight there would walk a correct answer away. The
# bottom is assumed rather than measured -- nothing breaks below it, the penalty simply stops
# doing enough to matter, and what is measured is the end of that road, since turning it off
# altogether costs overlap the misfit alone will not recover.
AREA_WEIGHT = 0.9
AREA_WINDOW = (0.30, 2.40)

# Largest change of volume an accepted step may make, as a fraction of the volume it starts
# from. The region is there for the single step. The cheapest area in this representation is a
# hull shrink, the linear term of a penalised step points straight down it, and unconstrained
# the first step of a coarse stage leaves no body at all; _trust_length shortens such a step
# rather than refusing it, because a region that refuses every trial is a fit that cannot move.
#
# It is not what keeps the fit from walking the volume away, although it was once asked to be.
# The floor in scripts/reconstruct_gn.py does that, and it acts on the body rather than on the
# step, so it costs an honest correction nothing. What sets the width here is therefore the
# journey rather than the collapse: a convex inversion returns a hull larger than the body's,
# so a correction that arrives has given up a large part of a volume, and a region much tighter
# than this spends the whole ladder crossing it. The screening stage binds hardest, ranking a
# start on two iterations; under a tighter region the bodies it ranks have barely left the
# convex answer they all started from, and it is then sorting starts on a difference that is
# not yet there. notes/objective.md works the width out against the journey it has to allow.
VOLUME_TRUST = 0.25

# Deepest carve one accepted step may add or remove, in body units. The node kernel's rows sum
# to one, so the largest depth the field makes is at most the largest coefficient and this
# region is an exact bound on the step in body units. It is needed because nothing else bounds
# the coefficients: what the objective charges is the body's surface and its misfit, and both
# are properties of the level set, so coefficients may grow in the directions that do not move
# it. A step is a carve, and a carve deeper than the body is not one.
DEPTH_TRUST = 0.5

# Smallest curvature the damping will believe, as a fraction of the mean over the stage's
# coordinates. See _normal_equations. A direction the curves cannot see has no curvature, and
# damping relative to curvature does not bound a step into it once the right-hand side carries
# the area's gradient, which does not lie in the range of the Jacobian. The floor sits at a
# hundredth of the mean and binds nothing where the curvature is healthy, which in a field
# indexed by direction is everywhere: every node has a surface point.
DAMP_FLOOR = 1e-2

# The misfit, in model errors, at which the fit stops. It is not one: one asks the body to
# explain the curves to the noise, and the representation cannot, its own floor at the
# released non-convex body being about this. A target below the floor makes the stopping rule
# dead and lets the fit spend its budget buying misfit with surface.
#
# The unit is the calibrated model error and it moves with the calibration, exactly as the
# threshold in scripts/reconstruct_gn.py does; the floor above was read before the sawed-off
# cube left the calibration. The direction it moved is the safe one, since a smaller model
# error puts the floor further above this number and the rule simply never fires, which
# spends the whole budget rather than stopping early. It is still a number to read again
# alongside the other, and a later calibration that raises the model error would make it stop
# short without saying so.
TARGET_SIGMA = 2.6

# Correlation lengths of the random smooth fields a subspace stage searches, in node spacings.
# One application of the node kernel smooths by about one spacing, so a field is built by
# applying it the square of the length times. The shortest length here is two spacings and not
# one, and that is the same bound the degree cap enforces by another route: a single application
# of the kernel leaves exactly the node-scale oscillation the area cannot charge.
SMOOTHING_LENGTHS = (2.0, 4.0, 8.0)

# The designed spread of starts. Measured on the released non-convex body, not one of twelve
# random draws of a shrunken hull carved by a random slab beat the plain convex answer, and the
# seven extra draws bought nothing; what a start should be is not a draw but a spread over the
# shape the correction can have. The carve of a start is therefore one spherical cap written
# straight into the coefficients, since a coefficient is a depth and needs no rescaling, and the
# grid below is swept axis first so that the first few starts differ in the one thing the
# ladder's own first stage cannot fix cheaply, which is where the dent is.
CAP_BAND_DEG = 30.0                    # half-height of the band of axes, from the equator. A
                                       # concavity near a pole is seen at grazing incidence from
                                       # every camera of the rig, so it is the part of a body a
                                       # convex answer is already closest to right about.
CAP_AXES = 8                           # axes across that band, quasi-uniformly
CAP_RADII_DEG = (35.0, 20.0, 50.0)     # angular radius of the cap, middle level first
CAP_DEPTHS = (0.30, 0.15, 0.45)        # its depth, in body units, middle level first
SHRINK_RANGE = (0.02, 0.14)            # uniform inward displacements of the hull a start draws
                                       # from, in body units of the canonical pose, where a body
                                       # spans [-1, 1]. The range covers a body that is barely
                                       # non-convex through a contact binary.
SHRINK_LEVELS = (0.5,)                 # positions in that range. One, not three. Measured on
                                       # the released bodies, the canonical pose divides a
                                       # uniform inward displacement almost entirely back out
                                       # -- z is rescaled to span [-1, 1] and the widest radius
                                       # to one, and an offset is nearly a similarity -- so the
                                       # deepest level of the range moves the canonical volume
                                       # of model 3 by a twentieth and of model 1 by a
                                       # hundredth. Three levels therefore make three bodies a
                                       # ranking cannot tell apart, and they would fill a
                                       # shortlist with copies of one start. The level is kept
                                       # non-zero because its purpose is the linearisation
                                       # rather than the body: it puts the secant probe of the
                                       # hull coefficient somewhere the hull is already moving.

# The waist family. A cap is a crater: one connected region at one depth, and no setting of its
# axis, radius and depth makes a neck. The shape library gives its bilobe and trilobe families
# just under a third of the bodies between them (LibrarySpec.family_weights), which is this
# project's own statement of what a scored body may be, so a grid that cannot start near a
# multi-lobed body is a grid missing a third of what it is meant to cover. A neck is a band of
# carve around the great circle perpendicular to the axis the lobes lie on, which is what
# waist_depths writes.
#
# Not measured from the released bodies, and not inferred from them. Sweeping sixty directions
# for an interior minimum of the body's width, model 3 reaches 23 per cent against the sawed-off
# cube's 20 -- and the cube is exactly convex, so that is what the measure reads on a body with
# no neck at all. Model 3 is a fifth of its hull's volume short without being bilobed, so this
# family is carried for the library's sake and not for its.
WAIST_AXES = 8                         # lobe axes across the same band the caps use. The spin
                                       # axis is added to them in _waist_axes, since a body may
                                       # equally lie across the axis of rotation or stand on it.
WAIST_HALFWIDTHS_DEG = (20.0, 12.0, 30.0)   # half-width of the carved band, middle level first.
                                       # At this many nodes the mean spacing is about four
                                       # degrees, so the narrowest band here is six spacings
                                       # across and is resolved rather than smoothed away.
WAIST_DEPTHS = (0.35, 0.20, 0.50)      # its depth, in body units, middle level first. Deeper
                                       # than a crater because a neck is the deepest feature a
                                       # body of this kind has; a start deeper than a given
                                       # body's star-shaped bound is passed over by the caller.

N_CAP_STARTS = CAP_AXES * len(CAP_RADII_DEG) * len(CAP_DEPTHS) * len(SHRINK_LEVELS)
N_WAIST_STARTS = ((WAIST_AXES + 1) * len(WAIST_HALFWIDTHS_DEG) * len(WAIST_DEPTHS)
                  * len(SHRINK_LEVELS))
N_STARTS = N_CAP_STARTS + N_WAIST_STARTS


@dataclass(frozen=True)
class Stage:
    """One refinement of the correction. `degree` is the largest spherical-harmonic degree the
    stage's coordinates reach, or 0 for a random subspace of `n_dirs` smooth fields."""
    degree: int
    n_dirs: int
    iters: int

    @property
    def name(self) -> str:
        return f"degree {self.degree}" if self.degree else f"subspace {self.n_dirs}"


# The ladder. A degree-L stage has (L+1)^2 - 9 coordinates, so this is 16, 40, 112, 280 and 616,
# and a secant Jacobian spends one render on each; with the line search a stage costs about
# (L+1)^2 + 25 renders an iteration. It ends at degree 24 because that is where the family
# already reaches the floor of what any correction of this size can do, and because its angular
# wavelength is ten extraction pitches at the surface -- far from the node scale the area cannot
# charge. There is deliberately no node-space stage: measured, the same ladder with one ends
# below the convex answer it started from.
#
# How the iterations are shared out follows what each degree is worth, which is measured in
# notes/representation.md by fitting the released body's own surface inside each stage's
# coordinates: degree 4 alone carries the overlap from the convex answer's 0.706 to 0.929,
# degree 10 reaches 0.969, and degree 24 adds 0.007 over degree 16. The cost runs the other way,
# a degree-24 iteration being thirteen times a degree-4 one, so the iterations are concentrated
# where the shape is and the top of the ladder is left as a refinement. A coarse stage also
# needs the iterations for a second reason: it is where the volume journey is made, and the
# trust region allows a fixed fraction of it per step.
DEFAULT_STAGES = (Stage(degree=4, n_dirs=0, iters=10),
                  Stage(degree=6, n_dirs=0, iters=8),
                  Stage(degree=10, n_dirs=0, iters=6),
                  Stage(degree=16, n_dirs=0, iters=4),
                  Stage(degree=24, n_dirs=0, iters=2))

# What every start is judged on. Measured, two iterations of the coarsest stage take almost the
# whole of the overlap the full ladder reaches, so this is a screening that sees the answer.
SCREEN_STAGES = (Stage(degree=4, n_dirs=0, iters=2),)

# Run with the penalty off, after the penalised phase has put the shape where it goes. At degree
# 16 rather than 24: the polish recovers misfit at a shape the penalised phase has already
# chosen, and the measurement above puts all but 0.007 of the reachable overlap inside degree
# 16, so the wider stage spends twice the renders refining a body the curves have already fixed.
POLISH_STAGES = (Stage(degree=16, n_dirs=0, iters=3),)


def degree_basis(nodes: np.ndarray, degree: int, skip: int = RADIAL_DEGREE) -> np.ndarray:
    """(n_nodes, (degree+1)^2 - (skip+1)^2): the real spherical harmonics of degree skip+1 to
    `degree`, sampled at the nodes.

    The degrees at or below `skip` are left out because c already carries exactly them: the
    reshaping adds sum_{l <= RADIAL_DEGREE} c_lm Ybar_lm to the same field from the same centre,
    so including them here would put identical columns in the Jacobian and spend a render
    measuring a direction the fit already has.

    The columns are returned unscaled. `CarveFit._basis` scales each to unit peak depth in the
    field it makes, which is a stronger statement than scaling it at the nodes and is what makes
    one secant step and one trust region right for every coordinate of every stage.
    """
    degree, skip = int(degree), int(skip)
    if degree <= skip:
        raise ValueError(f"a degree-{degree} stage has no coordinates above the degree-{skip} "
                         f"reshaping the fit already carries")
    return real_sh(np.asarray(nodes, dtype=float), degree)[:, (skip + 1) ** 2:]


def node_subspace_basis(kernel, n_dirs: int, rng,
                        lengths=SMOOTHING_LENGTHS) -> np.ndarray:
    """(n_nodes, n_dirs): smooth random fields on the node set, orthonormalised.

    Smooth rather than white because the fit is being asked for a shape and not for a texture: a
    white field on the nodes is an oscillation at the node scale, which the area penalty cannot
    charge at any weight that leaves the body a minimiser, and a search that can reach it will
    spend its budget there. The smoothing is the node kernel itself, applied the square of a
    correlation length times, since one application averages a node over its nearest neighbours
    and so smooths by about one node spacing. The shortest length is two spacings for the same
    reason the ladder's degree is capped, and one application of the kernel must not appear here.
    """
    k = int(kernel.shape[0])
    cols, per = [], max(1, int(n_dirs) // len(lengths))
    for length in lengths:
        for _ in range(per):
            q = rng.standard_normal(k)
            for _ in range(int(round(float(length) ** 2))):
                q = kernel @ q
            cols.append(q)
    while len(cols) < int(n_dirs):                     # a count the lengths do not divide
        q = rng.standard_normal(k)
        for _ in range(int(round(float(lengths[0]) ** 2))):
            q = kernel @ q
        cols.append(q)
    return np.linalg.qr(np.stack(cols[:int(n_dirs)], 1))[0]


def cap_depths(nodes: np.ndarray, axis, radius_deg: float, depth: float) -> np.ndarray:
    """The coefficients of a single spherical cap: `depth` on the nodes within `radius_deg` of
    `axis`, zero elsewhere.

    Written straight into the coefficients with no rescaling, because a coefficient is a depth:
    the node weights are a partition of unity, so a constant over a region is reproduced exactly
    inside it and the field falls to zero across about one node spacing at the rim. That rim is
    the sharp edge of a concavity, which is the feature a convex inversion cannot see at all.
    """
    u = np.asarray(nodes, dtype=float)
    n = np.asarray(axis, dtype=float).ravel()
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    return np.where(u @ n >= np.cos(np.deg2rad(float(radius_deg))), float(depth), 0.0)


def waist_depths(nodes: np.ndarray, axis, halfwidth_deg: float, depth: float) -> np.ndarray:
    """The coefficients of a neck: `depth` on the nodes within `halfwidth_deg` of the great
    circle perpendicular to `axis`, zero elsewhere.

    A direction u sits on that great circle exactly where u . axis vanishes, and within
    `halfwidth_deg` of it where |u . axis| is at most the sine of that angle, which is the
    test below. `axis` is the line the two lobes lie on, so the band runs around the body
    between them.

    Written straight into the coefficients with no rescaling, for the reason cap_depths is: the
    node weights are a partition of unity, so a constant over a region is reproduced exactly
    inside it and the field falls to zero across about one node spacing at each rim. A neck has
    two rims rather than one, and both are the sharp edge a convex inversion cannot see.
    """
    u = np.asarray(nodes, dtype=float)
    n = np.asarray(axis, dtype=float).ravel()
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    return np.where(np.abs(u @ n) <= np.sin(np.deg2rad(float(halfwidth_deg))),
                    float(depth), 0.0)


def _cap_axes(n: int = CAP_AXES, band_deg: float = CAP_BAND_DEG) -> np.ndarray:
    """`n` quasi-uniform axes in the band within `band_deg` of the equator, by the same spiral
    the nodes use, restricted to that band."""
    i = np.arange(int(n)) + 0.5
    z = np.sin(np.deg2rad(float(band_deg))) * (2.0 * i / int(n) - 1.0)
    azim = np.pi * (1.0 + 5.0 ** 0.5) * i
    r = np.sqrt(np.maximum(1.0 - z ** 2, 0.0))
    return np.stack([r * np.cos(azim), r * np.sin(azim), z], axis=1)


def _waist_axes(n: int = WAIST_AXES, band_deg: float = CAP_BAND_DEG) -> np.ndarray:
    """The lobe axes a neck is taken about: the band the cap axes use, and the spin axis.

    A waist about an equatorial axis is the neck of a body lying across the axis of rotation,
    and a waist about the spin axis the neck of one standing on it; the challenge fixes the
    axis but not which way a body was mounted on it, so both are in the grid. An axis and its
    antipode give the same band, so the band's own axes are not doubled.
    """
    return np.concatenate([_cap_axes(n, band_deg),
                           np.array([[0.0, 0.0, 1.0]], dtype=float)], axis=0)


def start_recipe(index: int) -> dict:
    """The `index`-th start of the designed grid, as its family and that family's numbers.

    The grid is two families laid end to end, the caps first and then the waists, each swept
    with the axis varying fastest and every other factor at its middle level first. A run which
    can afford only part of a family therefore spends it on where the feature is rather than on
    how large it is: the depth and the width are what the ladder's own first stage corrects most
    cheaply, and the axis is not.
    """
    i = int(index)
    if i < N_CAP_STARTS:
        axes = _cap_axes()
        axis = axes[i % len(axes)]
        i //= len(axes)
        radius = CAP_RADII_DEG[i % len(CAP_RADII_DEG)]
        i //= len(CAP_RADII_DEG)
        depth = CAP_DEPTHS[i % len(CAP_DEPTHS)]
        i //= len(CAP_DEPTHS)
        rec = {"kind": "cap", "axis": axis.tolist(), "radius_deg": float(radius),
               "depth": float(depth)}
    else:
        i -= N_CAP_STARTS
        axes = _waist_axes()
        axis = axes[i % len(axes)]
        i //= len(axes)
        half = WAIST_HALFWIDTHS_DEG[i % len(WAIST_HALFWIDTHS_DEG)]
        i //= len(WAIST_HALFWIDTHS_DEG)
        depth = WAIST_DEPTHS[i % len(WAIST_DEPTHS)]
        i //= len(WAIST_DEPTHS)
        rec = {"kind": "waist", "axis": axis.tolist(), "halfwidth_deg": float(half),
               "depth": float(depth)}
    level = SHRINK_LEVELS[i % len(SHRINK_LEVELS)]
    rec["shrink"] = float(SHRINK_RANGE[0] + level * (SHRINK_RANGE[1] - SHRINK_RANGE[0]))
    return rec


def conjunction_start(nodes: np.ndarray, index: int, n_radial: int = 9) -> tuple:
    """A start with both halves of the correction present: (c, a, recipe).

    A convex inversion of a non-convex body does not return that body's hull. It returns a
    larger one, because enlarging the hull is how a convex body imitates the shadowing of a
    concavity, so the correction from that answer to the body shrinks the hull and carves it at
    the same time. A start that leaves the hull where the convex inversion put it asks the fit to
    find both from a linearisation taken where neither is active, and there a hull shrink alone
    raises the misfit; the fit then spends the step on the carve, which is the convex inversion's
    own mistake made once more one level down. Pairing the shrink with the cap puts the first
    linearisation somewhere both are already doing something.

    The shrink is uniform, the degree-zero coefficient alone, because that is the part of the
    excess that does not depend on which way a body is turned; the rest is left to the fit.

    The carve is a crater or a neck, by the family the index falls in. A crater is what a grid
    of caps can make and a neck is not, and the shape library gives its bilobe and trilobe
    families just under a third of the bodies between them, so a grid without the second family
    cannot start near a third of what it is meant to cover, at any size.
    """
    rec = start_recipe(index)
    a = (waist_depths(nodes, rec["axis"], rec["halfwidth_deg"], rec["depth"])
         if rec["kind"] == "waist" else
         cap_depths(nodes, rec["axis"], rec["radius_deg"], rec["depth"]))
    c = np.zeros(int(n_radial))
    c[0] = rec["shrink"]
    return c, a, rec


class CarveFit:
    """Damped Gauss-Newton on (c, a) against one body's curves.

    `render(c, a)` returns the kept curves as a flat array in the same order as `data` together
    with the surface area and the volume of the canonically posed body, or None for a body the
    forward model refuses. `scale` is the per-entry model error the residual is divided by, so
    the reported misfit is in standard deviations of it.

    `kernel` is what the coefficients make at the nodes themselves, and `nodes` are the
    directions they are carried on. The kernel is used for three things that all follow from its
    rows summing to one: scaling a coordinate to unit peak depth, bounding a step's depth
    exactly, and smoothing the random fields of a subspace stage.

    Both secant steps are taken over the size the answer's own correction is likely to have
    rather than at zero, because the response of the curves to either half of the correction
    begins rather than scales: a dent shadows itself only once it is deep enough to, and a hull
    shrink lowers the misfit only once there is a carve for it to uncover.
    """

    def __init__(self, render, data: np.ndarray, scale: np.ndarray, kernel,
                 nodes: np.ndarray, n_radial: int = 9,
                 area_weight: float = AREA_WEIGHT, volume_trust: float = VOLUME_TRUST,
                 depth_trust: float = DEPTH_TRUST, depth_cap: float | None = None,
                 step_g: float = STEP_G, step_c: float = STEP_C,
                 damping=(1e-1, 1e-2, 1e-3, 1.0, 10.0),
                 lengths=(1.0, 0.5, 0.25, 0.1, 0.04),
                 subspace_tries: int = 3, seed: int = 0):
        if area_weight and not AREA_WINDOW[0] <= area_weight <= AREA_WINDOW[1]:
            raise ValueError(f"area weight {area_weight} is outside the measured window "
                             f"{AREA_WINDOW}; below it a smooth dent of the size the correction "
                             f"has is not charged and above it the body stops being the minimum")
        self.render = render
        self.data = np.asarray(data, dtype=np.float64).ravel()
        self.iscale = 1.0 / np.asarray(scale, dtype=np.float64).ravel()
        self.n_obs = len(self.data)
        self.kernel = kernel
        self.nodes = np.asarray(nodes, dtype=float)
        self.n_nodes = len(self.nodes)
        if kernel.shape[0] != self.n_nodes or kernel.shape[1] != self.n_nodes:
            raise ValueError(f"the kernel is {kernel.shape} for {self.n_nodes} nodes")
        self.n_radial = int(n_radial)
        self.area_weight = float(area_weight)
        self.volume_trust = float(volume_trust)
        self.depth_trust = float(depth_trust)
        self.depth_cap = None if depth_cap is None else float(depth_cap)
        self.step_g, self.step_c = float(step_g), float(step_c)
        self.damping, self.lengths = tuple(damping), tuple(lengths)
        self.subspace_tries = int(subspace_tries)
        self.rng = np.random.default_rng(seed)
        self.renders = 0
        self.refused_depth = 0

    # ------------------------------------------------------------------ the objective
    def residual(self, y) -> np.ndarray | None:
        """The whitened residual, scaled so that its Euclidean norm is the misfit in standard
        deviations."""
        if y is None:
            return None
        r = (np.asarray(y, dtype=np.float64).ravel() - self.data) * self.iscale
        return r / np.sqrt(self.n_obs)

    def _render(self, c, a):
        """(residual, area, volume) of the body at (c, a), or (None, nan, nan).

        `render` returns the kept curves together with the surface area and the volume of the
        canonically posed body. Both come from the mesh the extraction has already built, so
        they cost a per cent of the render, and every Jacobian column therefore carries the
        secant derivative of the area for nothing."""
        self.renders += 1
        out = self.render(c, a)
        if out is None:
            return None, float("nan"), float("nan")
        y, area, vol = out
        return self.residual(y), float(area), float(vol)

    def objective(self, r, area: float) -> float:
        """log chi^2 plus the area of the body, which is the total variation of its indicator.

        Area rather than a ridge on the coefficients because the two charge different things. A
        ridge charges by how large a coefficient is, so it prefers a shallow answer to a deep one
        and prefers the fit's own answer to the body. Area charges by how much surface a shape
        has, so a smooth dent of any depth passes nearly free while what it refuses is a body
        that has grown surface it does not need -- above all a further uniform carve into a body
        that already explains its curves.

        The logarithm rather than a plain weight because the two terms have to stay in balance as
        the fit descends. The gradient of chi^2 falls with chi while the gradient of an area term
        does not, so no fixed weight both charges at the start and leaves the body a minimum at
        the end. Stationarity then compares a relative change of misfit with an absolute change
        of area.
        """
        rr = float(r @ r)
        return float(np.log(max(rr, 1e-300)) + self.area_weight * float(area))

    def _basis(self, stage: Stage) -> np.ndarray:
        """The stage's coordinates as columns over the node depths, scaled so that one unit of a
        coordinate is one body unit of depth in the field the column actually makes.

        Scaled through the kernel rather than at the nodes because the two differ: the node
        weights smooth a column over its neighbours, and a column whose angular wavelength is a
        few node spacings comes out of that smoothing shallower than it went in. Scaling by what
        the field reaches is what makes one finite-difference step and one depth trust region
        right for a degree-4 coordinate and a degree-24 one alike.
        """
        if stage.degree:
            b = degree_basis(self.nodes, stage.degree, self.n_radial_degree)
        else:
            b = node_subspace_basis(self.kernel, stage.n_dirs, self.rng)
        peak = np.zeros(b.shape[1])
        for i in range(0, b.shape[1], 256):                    # a block at a time
            sl = slice(i, min(i + 256, b.shape[1]))
            peak[sl] = np.abs(self.kernel @ b[:, sl]).max(axis=0)
        return b / np.maximum(peak, 1e-12)[None, :]

    @property
    def n_radial_degree(self) -> int:
        """The largest spherical-harmonic degree c carries, which is what a degree stage skips.

        Read from the number of reshaping coefficients rather than imported, so that a caller
        which passes a different n_radial gets a basis that skips exactly what its own c covers.
        """
        d = int(round(np.sqrt(self.n_radial))) - 1
        if (d + 1) ** 2 != self.n_radial:
            raise ValueError(f"{self.n_radial} reshaping coefficients are not the harmonics of "
                             f"a whole degree, so a degree stage cannot know what to skip")
        return d

    # ------------------------------------------------------------------ one iteration
    def _jacobian(self, c, a, basis, r0, area0, vol0):
        """(J, da, dv): the secant derivatives of the residual, the area and the volume, in the
        stage's coordinates. The last two cost no render of their own, since the body whose
        residual a column measures is the body whose area and volume it measures."""
        m = basis.shape[1]
        J = np.zeros((len(r0), self.n_radial + m))
        da = np.zeros(self.n_radial + m)
        dv = np.zeros(self.n_radial + m)
        for j in range(self.n_radial):
            cc = c.copy()
            cc[j] += self.step_c
            r, ar, vr = self._render(cc, a)
            if r is not None:
                J[:, j] = (r - r0) / self.step_c
                da[j] = (ar - area0) / self.step_c
                dv[j] = (vr - vol0) / self.step_c
        for j in range(m):
            r, ar, vr = self._render(c, a + self.step_g * basis[:, j])
            if r is not None:
                J[:, self.n_radial + j] = (r - r0) / self.step_g
                da[self.n_radial + j] = (ar - area0) / self.step_g
                dv[self.n_radial + j] = (vr - vol0) / self.step_g
        return J, da, dv

    def _trust_length(self, step, basis, vprime, vol) -> float:
        """The longest step length that keeps both the change of volume and the depth of the
        carve it adds inside their trust regions, capped at one.

        A region has to shorten the step rather than merely refuse it. The cheapest area in this
        representation is a hull shrink, so the linear term of a penalised step points down it
        and can be enormous: unconstrained, the first step of a coarse stage collapses the body
        altogether, and a line search over a fixed ladder of lengths then finds every one of them
        outside the region and takes no step at all.

        The two regions bound different things and both are needed. Volume alone does not bound
        the depths, because a step that carves deeply in one place and fills deeply in another
        leaves the volume where it was; the objective does not bound them either, since it
        charges the surface and the misfit and both are properties of the level set. The depth of
        the carve a step makes is what says whether it is a step between bodies.
        """
        rate = abs(float(vprime @ step))
        lv = 1.0 if rate <= 1e-12 else self.volume_trust * max(abs(vol), 1e-9) / rate
        depth = float(np.abs(self.kernel @ (basis @ step[self.n_radial:])).max())
        ld = 1.0 if depth <= 1e-12 else self.depth_trust / depth
        return float(min(1.0, lv, ld))

    def _normal_equations(self, J, r0, da):
        """(A, b, diagonal) of the penalised Gauss-Newton system, before the damping.

        Differentiating log(r.r) + mu A gives the ordinary Gauss-Newton system with one extra
        linear term. The matrix is unchanged, so the penalty costs nothing in conditioning:

            (J^T J) d = -J^T r - (mu/2)(r^T r) a'.
        """
        A = J.T @ J
        b = -(J.T @ r0) - 0.5 * self.area_weight * float(r0 @ r0) * da
        d = np.diag(A).copy()
        pos = d[d > 0]
        mean = float(pos.mean()) if pos.size else 1.0
        # A direction the curves cannot see has no curvature, and the damping, being relative to
        # the curvature, does not bound the step there. That is harmless while the right side is
        # J^T r, which lies in the range of J; it is not harmless once the right side carries the
        # area's gradient, which does not, and the step in such a direction then grows without
        # bound as the damping is loosened.
        d = np.maximum(d, DAMP_FLOOR * mean)
        return A, b, d

    @staticmethod
    def _solve(A, b, d, mu):
        try:
            return np.linalg.solve(A + mu * np.diag(d), b)
        except np.linalg.LinAlgError:
            return np.zeros(len(b))

    def run(self, c0, a0, stages=DEFAULT_STAGES, target: float = TARGET_SIGMA, log=None,
            area_weight: float | None = None):
        """Fit from (c0, a0). Returns (c, a, history); the history has one row per accepted or
        refused iteration.

        `area_weight` overrides the penalty for this run alone, and zero turns it off, which is
        what the polish uses: once the shape is where the penalty puts it, minimising the misfit
        alone with the trust regions still on recovers the misfit without giving the shape back.
        """
        was, self.area_weight = self.area_weight, (self.area_weight if area_weight is None
                                                   else float(area_weight))
        try:
            return self._run(c0, a0, stages, target, log)
        finally:
            self.area_weight = was

    def _run(self, c0, a0, stages, target, log):
        c, a = np.asarray(c0, float).copy(), np.asarray(a0, float).copy()
        history = []
        r, area, vol = self._render(c, a)
        if r is None:
            return c, a, [{"stage": "start", "chi": float("inf"), "note": "no curves"}]
        chi = float(np.linalg.norm(r))
        for stage in stages:
            basis = self._basis(stage)
            tries = 0
            it = 0
            while it < stage.iters:
                if chi <= target:
                    break
                J, da, vprime = self._jacobian(c, a, basis, r, area, vol)
                A, rhs, diag = self._normal_equations(J, r, da)
                obj = self.objective(r, area)
                # The best of the trials rather than the first that descends. A damping and a
                # step length that happen to be tried early are not the best step available, and
                # the trials cost fifteen renders against the Jacobian's own hundreds.
                best = None
                for mu in self.damping:
                    step = self._solve(A, rhs, diag, mu)
                    cap = self._trust_length(step, basis, vprime, vol)
                    for length in (cap * np.asarray(self.lengths)):
                        cc = c + length * step[:self.n_radial]
                        aa = a + length * (basis @ step[self.n_radial:])
                        # The star-shaped bound is a refusal and not a clip: a clipped step is a
                        # different step, and the line search would then be choosing among
                        # lengths that no longer mean what it thinks they mean.
                        if self.depth_cap is not None and float(aa.max()) > self.depth_cap:
                            self.refused_depth += 1
                            continue
                        rr, aar, vv = self._render(cc, aa)
                        if rr is None:
                            continue
                        if abs(vv - vol) > self.volume_trust * max(abs(vol), 1e-9):
                            continue          # outside the trust region on volume
                        o = self.objective(rr, aar)
                        if o < obj and (best is None or o < best[0]):
                            best = (o, cc, aa, rr, aar, vv)
                moved = best is not None
                if moved:
                    _, c, a, r, area, vol = best
                    chi = float(np.linalg.norm(r))
                row = {"stage": stage.name, "iteration": it, "chi": chi, "area": area,
                       "volume": vol, "objective": self.objective(r, area),
                       "area_weight": self.area_weight, "accepted": moved,
                       "renders": self.renders}
                history.append(row)
                if log is not None:
                    log(row)
                if not moved:
                    # a subspace that finds no step is redrawn; a fixed degree basis that finds
                    # none has converged for this stage
                    tries += 1
                    if stage.degree or tries >= self.subspace_tries:
                        break
                    basis = self._basis(stage)
                    continue
                it += 1
            if chi <= target:
                break
        return c, a, history
