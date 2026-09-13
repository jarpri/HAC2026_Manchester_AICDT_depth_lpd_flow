"""The derivative-free fit: what a coordinate means, and that one step of it moves a body
toward the one its curves came from."""
import numpy as np
import pytest
import torch

from hac26.conventions import psi_grid
from hac26.field import (CODE_DIM, N_NODES, N_RADIAL, RADIAL_DEGREE, DepthSphere,
                         ImplicitBody, depth_cap, node_kernel, radial_basis)
from hac26.forward.mesh.exact import RenderConfig
from hac26.forward.mesh.instrument import Instrument
from hac26.shapes import canonicalize_r, icosphere, mesh_support, rescale_touch_z
from hac26.solvers.gauss_newton import (AREA_WEIGHT, AREA_WINDOW, CAP_DEPTHS, DEPTH_TRUST,
                                        N_CAP_STARTS, N_STARTS, N_WAIST_STARTS,
                                        SMOOTHING_LENGTHS, VOLUME_TRUST, CarveFit, Stage,
                                        cap_depths, conjunction_start, degree_basis,
                                        node_subspace_basis, start_recipe, waist_depths)
from hac26.solvers.operator import CodeOperator

TINY = RenderConfig(height=24, width=40, supersample=1, sun_res=64, n_source=1,
                    phase_chunk=2, geom_chunk=1, radiosity_faces=48)
NODES = DepthSphere(N_NODES).u.numpy()


def test_the_node_kernel_is_the_depth_at_the_nodes_and_averages():
    """K a is the depth the coefficients make at the nodes, and it is what tells a solver what
    a coordinate is worth without spending a render on it. Its rows sum to one, which is what
    makes a coefficient a depth: a constant is reproduced exactly, nothing overshoots, and a
    bound on the coefficients is a bound on the depth in body units."""
    K = node_kernel()
    assert np.abs(np.asarray(K.sum(axis=1)).ravel() - 1.0).max() < 1e-6
    assert (K.toarray() >= 0.0).all()
    rng = np.random.default_rng(0)
    a = rng.standard_normal(N_NODES)
    made = K @ a
    assert np.abs(made).max() <= np.abs(a).max() + 1e-6
    assert np.abs(K @ np.full(N_NODES, 0.37) - 0.37).max() < 1e-6


def test_a_stage_coordinate_is_a_carve_depth():
    """Every stage's coordinates are scaled so that one unit is one body unit of depth in the
    field the coordinate actually makes. Without it a low-degree coordinate, which displaces a
    whole face at once, and a high-degree one, which the node weights smooth away by half,
    would mean different things and one finite-difference step could not be right for both."""
    K = node_kernel()
    fit = CarveFit(lambda c, g: None, np.zeros(4), np.ones(4), K, NODES)
    for stage in (Stage(4, 0, 1), Stage(10, 0, 1), Stage(24, 0, 1), Stage(0, 8, 1)):
        basis = fit._basis(stage)
        assert np.allclose(np.abs(K @ basis).max(axis=0), 1.0, atol=1e-6), stage.name


def test_a_degree_stage_carries_the_degrees_the_reshaping_does_not():
    """The reshaping already carries every harmonic up to degree two of the same field, so a
    degree stage has to start above them: a column that repeated one would spend a render
    measuring a direction the fit already has, and would put two identical columns in the
    Jacobian."""
    for degree in (4, 6, 10, 16, 24):
        b = degree_basis(NODES, degree)
        assert b.shape == (N_NODES, (degree + 1) ** 2 - (RADIAL_DEGREE + 1) ** 2)
        q = radial_basis(torch.tensor(NODES, dtype=torch.float64)).numpy()
        cos = ((q / np.linalg.norm(q, axis=0)).T
               @ (b / np.linalg.norm(b, axis=0)))
        # the nodes are a quasi-uniform quadrature, not an exact one, so the harmonics are
        # orthogonal on them only to that quadrature's own error, which grows with the degree
        # of the column; what would be a defect is a column that repeats a reshaping direction,
        # and that would read near one
        assert np.abs(cos).max() < 0.01, degree
    with pytest.raises(ValueError):
        degree_basis(NODES, RADIAL_DEGREE)


def test_the_subspace_directions_are_smoother_than_the_node_spacing():
    """A displacement at the node scale is what the area penalty cannot charge at any weight
    that leaves a body the minimum, so it must not be reachable. One application of the node
    kernel is exactly that displacement; the shortest correlation length here is two node
    spacings, which is four applications."""
    K = node_kernel()
    assert min(SMOOTHING_LENGTHS) >= 2.0
    rng = np.random.default_rng(0)
    b = node_subspace_basis(K, 12, rng)
    assert b.shape == (N_NODES, 12)
    assert np.abs(b.T @ b - np.eye(12)).max() < 1e-8
    # Smoothing again barely changes them, which is what being smooth on this graph means. The
    # comparison is against what the excluded directions keep: white noise on the nodes, and
    # white noise smoothed once, which is the corrugation the ladder's degree cap is there to
    # exclude and which no column here may resemble.
    white = rng.standard_normal((N_NODES, 12))
    once = K @ white

    def kept(x):
        return float((np.linalg.norm(K @ x, axis=0) / np.linalg.norm(x, axis=0)).min())

    assert kept(b) > 1.2 * kept(once) > 1.4 * kept(white)


def test_a_designed_start_carves_the_cap_it_asks_for():
    """A start is a spherical cap of a requested depth written straight into the coefficients,
    and the field it makes has to be that deep: the weights are a partition of unity, so a
    constant over a region is reproduced exactly inside it and no rescaling is needed."""
    K = node_kernel()
    seen = set()
    for i in range(24):
        c, a, rec = conjunction_start(NODES, i)
        assert float(np.abs(K @ a).max()) == pytest.approx(rec["depth"], rel=1e-3)
        assert rec["depth"] in CAP_DEPTHS
        assert c[0] == pytest.approx(rec["shrink"])
        assert (a >= 0.0).all() and a.max() == pytest.approx(rec["depth"])
        seen.add(tuple(np.round(rec["axis"], 6)))
    # the grid is swept axis first, so a run that can afford only a few starts spends them on
    # where the dent is and not on how deep it is
    assert len(seen) == len({tuple(np.round(start_recipe(i)["axis"], 6)) for i in range(24)})
    assert start_recipe(0)["axis"] != start_recipe(1)["axis"]
    assert start_recipe(0)["depth"] == start_recipe(1)["depth"]
    # a cap is a cap: only the nodes inside its angular radius are carved
    a = cap_depths(NODES, [0.0, 0.0, 1.0], 30.0, 0.2)
    assert np.array_equal(a > 0, NODES[:, 2] >= np.cos(np.deg2rad(30.0)))


def test_a_designed_waist_carves_the_band_it_asks_for():
    """The grid's second family is a neck rather than a crater, because the one released
    non-convex body is a contact binary and no cap can make a neck. A waist about an axis is
    the band within its half-width of the great circle perpendicular to that axis, and the
    field it makes has to be as deep as it asks for, the weights being a partition of unity."""
    K = node_kernel()
    # a waist about the spin axis is exactly the equatorial band
    a = waist_depths(NODES, [0.0, 0.0, 1.0], 20.0, 0.3)
    assert np.array_equal(a > 0, np.abs(NODES[:, 2]) <= np.sin(np.deg2rad(20.0)))
    assert float(np.abs(K @ a).max()) == pytest.approx(0.3, rel=1e-3)
    # a waist about an equatorial axis runs through both poles, which a cap in the grid's own
    # band can never reach
    b = waist_depths(NODES, [1.0, 0.0, 0.0], 20.0, 0.3)
    assert b[int(np.argmax(NODES[:, 2]))] > 0 and b[int(np.argmin(NODES[:, 2]))] > 0
    # half-width 30 degrees is half the sphere by Archimedes: the band |u.n| <= sin(30) has
    # exactly half the area, so the fraction of nodes in it is a check on the node set too
    half = waist_depths(NODES, [0.0, 0.0, 1.0], 30.0, 0.3)
    assert float((half > 0).mean()) == pytest.approx(0.5, abs=0.02)


def test_the_start_grid_holds_both_families_and_repeats_none_of_them():
    """A run that can afford only part of the grid takes it in order, so the caps come first
    and the axis varies fastest inside each family. Every index has to be a different start:
    a grid that repeats a recipe is spending a ranking slot on a body it has already seen."""
    assert N_STARTS == N_CAP_STARTS + N_WAIST_STARTS
    kinds = [start_recipe(i)["kind"] for i in range(N_STARTS)]
    assert set(kinds[:N_CAP_STARTS]) == {"cap"} and set(kinds[N_CAP_STARTS:]) == {"waist"}
    keys = {repr(sorted(((k, tuple(v) if isinstance(v, list) else v)
                         for k, v in start_recipe(i).items()), key=str))
            for i in range(N_STARTS)}
    assert len(keys) == N_STARTS
    # every start is a body the representation can hold: one depth, written straight in
    for i in (0, N_CAP_STARTS - 1, N_CAP_STARTS, N_STARTS - 1):
        c, a, rec = conjunction_start(NODES, i)
        assert a.max() == pytest.approx(rec["depth"]) and (a >= 0.0).all()
        assert c[0] == pytest.approx(rec["shrink"])


def _operator_and_bodies():


    """A small operator, a convex support, and a body carved out of it by a known cap."""
    v, f = icosphere(2)
    v = np.asarray(v) * np.array([1.0, 0.82, 0.72])
    v = canonicalize_r(rescale_touch_z(v, f, centre_xy=False))
    support = torch.tensor(mesh_support(v, ImplicitBody().core.n.numpy()),
                           dtype=torch.float32)
    op = CodeOperator(Instrument.blender_start(), psi_grid(2), res=16, config=TINY,
                      device="cpu", backend="software")
    K = node_kernel()
    # a dent in the side, the feature a convex inversion cannot see, at a depth that needs no
    # rescaling because a coefficient is a depth
    g_true = cap_depths(NODES, [1.0, 0.0, 0.0], 40.0, 0.35)
    c_true = np.zeros(N_RADIAL)
    # and a hull too large by about what a convex inversion of such a body leaves
    c_true[0] = 0.10
    return op, support, K, g_true, c_true


@pytest.mark.slow
def test_one_step_moves_a_body_and_leaves_it_a_body():
    """The fit is given the curves of a carved body and started from the convex body it was
    carved out of. One step of the coarsest stage has to lower what is being minimised, and
    the body it leaves has to still be a body: its volume inside the trust region and the
    carve it added no deeper than a carve can be.

    Those are the properties of the machinery and they are what a test can settle. Whether the
    fit moves a body *toward* its truth is a property of a run against that body's whole set
    of geometries, and is measured in notes/representation.md; asked of one coarse stage
    against a handful of cameras it measures a regime no reconstruction is in, where a
    relative change of misfit is small and the penalty sets the step on its own.
    """
    op, support, K, g_true, c_true = _operator_and_bodies()
    geoms = [12]                                     # one high phase-angle camera
    code = torch.zeros(CODE_DIM)

    def render(c, g):
        z = code.clone()
        z[-N_NODES:] = torch.tensor(np.asarray(g), dtype=torch.float32)
        out = op.curves_with_shape(support, z, 1.0, geoms=geoms,
                                   c=torch.tensor(np.asarray(c), dtype=torch.float32))
        if out is None:
            return None
        cur, area, vol = out
        return cur.numpy().ravel(), area, vol

    data = render(c_true, g_true)[0]
    # The model error the fit is given has to be the one it is used with. What is minimised is
    # scale free in the misfit, so a body a hundred model errors from its curves is a regime
    # no reconstruction starts in; a calibrated model error puts the convex answer a few
    # errors away and this scale does the same here.
    scale = np.full(len(data), 0.12)
    fit = CarveFit(render, data, scale, K, NODES, n_radial=N_RADIAL, seed=0,
                   depth_cap=depth_cap(support.numpy()))

    r0, area0, vol0 = fit._render(np.zeros(N_RADIAL), np.zeros(N_NODES))
    obj0 = fit.objective(r0, area0)
    c, g, hist = fit.run(np.zeros(N_RADIAL), np.zeros(N_NODES),
                         stages=(Stage(4, 0, 2),), target=0.0)
    assert hist and hist[0]["accepted"], "no damping and no step length improved the objective"
    assert hist[-1]["objective"] < obj0
    # Each accepted step moves the volume by at most the region's fraction of the volume it
    # starts from, so after n of them the volume is inside vol0 (1 +- trust)^n. Bounding it by
    # n times the region instead would be the wrong test on the growing side, and a fixed
    # multiple of the region would go stale the moment the region is rewidened.
    steps = sum(1 for row in hist if row.get("accepted"))
    assert abs(hist[-1]["volume"] - vol0) <= vol0 * ((1.0 + VOLUME_TRUST) ** steps - 1.0)
    assert float(np.abs(K @ g).max()) < 3.0 * DEPTH_TRUST
    assert np.isfinite(g).all() and np.isfinite(c).all()


def test_the_objective_charges_surface_and_not_depth():
    """A body that has grown surface and one that has not are ranked by the surface, whatever
    their coefficients are.

    That is the whole reason the penalty is on area: a ridge on the coefficients would prefer a
    shallow answer to a deep one and so prefer the fit's own answer to the body, and the misfit
    prefers a finely resolved surface whether or not its shape is right."""
    K = node_kernel()
    fit = CarveFit(lambda c, g: None, np.zeros(4), np.ones(4), K, NODES,
                   area_weight=AREA_WEIGHT)
    r = np.full(4, 0.5)
    smooth, rough = 9.0, 9.6
    assert fit.objective(r, smooth) < fit.objective(r, rough)
    # and the balance is scale free: a body whose misfit is ten times smaller is not thereby
    # allowed ten times the surface
    better = np.full(4, 0.05)
    assert (fit.objective(better, rough) - fit.objective(better, smooth)
            == pytest.approx(fit.objective(r, rough) - fit.objective(r, smooth)))
    # with the penalty off it is the misfit alone
    off = CarveFit(lambda c, g: None, np.zeros(4), np.ones(4), K, NODES, area_weight=0.0)
    assert off.objective(r, smooth) == off.objective(r, rough)


def test_the_area_weight_is_held_inside_the_window_it_was_measured_in():
    """Outside it the objective is the wrong one in a way no run would report: below the
    window a smooth dent of the size the correction has is not charged at all, and above the
    window the body stops being the minimum against a further carve."""
    K = node_kernel()
    for bad in (AREA_WINDOW[0] - 0.1, AREA_WINDOW[1] + 0.1):
        with pytest.raises(ValueError):
            CarveFit(lambda c, g: None, np.zeros(4), np.ones(4), K, NODES, area_weight=bad)


def test_a_step_that_moves_the_volume_too_far_is_refused():
    """The cheapest area in this representation is a hull shrink, so an objective that
    charges area walks the body away to nothing unless the volume is held. The trust region
    is what holds it, and a fit whose every trial leaves it must take no step at all."""
    K = node_kernel()
    calls = {"n": 0}

    def render(c, g):
        calls["n"] += 1
        # every body after the first is half the volume of the first, and fits perfectly
        if calls["n"] == 1:
            return np.ones(4), 9.0, 1.0
        return np.zeros(4), 1.0, 0.5

    fit = CarveFit(render, np.zeros(4), np.ones(4), K, NODES, n_radial=N_RADIAL,
                   volume_trust=0.08)
    _, _, hist = fit.run(np.zeros(N_RADIAL), np.zeros(N_NODES), stages=(Stage(4, 0, 1),),
                         target=0.0)
    assert hist and not hist[0]["accepted"], "a step halving the volume was accepted"


def test_a_step_past_the_star_shaped_bound_is_refused_and_not_clipped():
    """Past the bound the body stops containing the centre its depths are measured from, so it
    stops being star-shaped and extracts as several pieces. A trial that crosses it is refused
    without being rendered, and not shortened to sit on it: a clipped step is a different step,
    and the line search would then be choosing among lengths that no longer mean what it takes
    them to mean."""
    K = node_kernel()
    seen = []

    def render(c, g):
        seen.append(float(np.max(g)))
        return np.zeros(4), 1.0, 1.0

    fit = CarveFit(render, np.zeros(4), np.ones(4), K, NODES, n_radial=N_RADIAL,
                   depth_cap=0.05)
    fit.run(np.zeros(N_RADIAL), np.zeros(N_NODES), stages=(Stage(4, 0, 1),), target=0.0)
    assert max(seen) <= 0.05 + 1e-9 or fit.refused_depth > 0
