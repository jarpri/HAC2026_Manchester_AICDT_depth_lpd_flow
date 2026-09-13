"""Tests of hac26.field, the implicit shape representation.

The gate is the cube: encode it with the support h alone, extract a mesh, and require the
faces planar to within one grid cell and Dice above 0.99 against the analytic cube. The other
tests pin what the cube does not touch: the design of normals, that the core is a plain max,
that the depth correction is signed, is a partition of unity and is independent of the query
batch, that the grid the extraction runs on is the one FlexiCubes builds, that dh is
band-limited, and that the pose constraints allow the published radius its tolerance.
"""
import numpy as np
import pytest
import torch

from hac26.field import (DESIGN_N, DESIGN_T, EXTRACT_EXTENT, KNN, N_NODES, ConvexCore,
                         DepthSphere, ImplicitBody, _design_residual, apply_constraints,
                         core_centre, depth_cap, extract_mesh, spherical_design, voxel_grid)

A = 1.0            # cube half-side
RES = 128


def cube_support(normals: np.ndarray, a: float = A) -> np.ndarray:
    """h(n) = max over the cube's vertices of n.v = a(|nx| + |ny| + |nz|)."""
    return a * np.abs(normals).sum(1)


# ------------------------------------------------------------------ the fixed normals

def test_design_is_a_ten_design():
    """The cached design has DESIGN_N unit normals and a small worst-degree residual."""
    x = spherical_design()
    assert x.shape == (DESIGN_N, 3)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-9)
    # an exact design has zero residual at every degree
    assert _design_residual(x, DESIGN_T) < 1e-5


def test_design_contains_the_axis_directions():
    """All six axis directions are in the design; the core needs them to represent a cube
    exactly."""
    x = spherical_design()
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        assert np.abs(x - e).sum(1).min() < 1e-9
        assert np.abs(x + e).sum(1).min() < 1e-9


# ------------------------------------------------------------------ the core

def test_core_is_max_not_log_sum_exp():
    """With the cube's support, the core is exactly -A at the centre and exactly zero at the
    face centres; a log-sum-exp core would put the zero set inside the true surface."""
    n = spherical_design()
    core = ConvexCore(n)
    core.set_support(torch.tensor(cube_support(n), dtype=torch.float32))
    y = torch.tensor([[0.0, 0.0, 0.0], [A, 0.0, 0.0], [0.0, 0.0, A]], dtype=torch.float32)
    f = core(y).detach().numpy()
    assert f[0] == pytest.approx(-A, abs=1e-5)       # centre: distance to the nearest face
    assert f[1] == pytest.approx(0.0, abs=1e-5)      # face centres sit exactly on the surface
    assert f[2] == pytest.approx(0.0, abs=1e-5)


def test_support_roundtrip():
    """set_support followed by reading h returns the same support."""
    n = spherical_design()
    h = cube_support(n)
    core = ConvexCore(n)
    core.set_support(torch.tensor(h, dtype=torch.float32))
    assert core.h.detach().numpy() == pytest.approx(h, rel=1e-4)


def test_core_h_is_non_negative():
    """h stays non-negative whatever the raw parameter holds."""
    core = ConvexCore(spherical_design())
    with torch.no_grad():
        core.raw_h.copy_(torch.full((DESIGN_N,), -50.0))
    assert (core.h >= 0).all()


# ------------------------------------------------------------------ the correction

def test_a_coefficient_is_a_depth_and_the_weights_are_a_partition_of_unity():
    """The weights of a direction sum to one over the nodes it reads, which is what makes a
    coefficient a depth rather than an amplitude: a constant is reproduced exactly, the field
    is bounded by the largest coefficient, and it cannot overshoot between nodes. Everything
    the solver does with a bound on the coefficients rests on this."""
    rep = DepthSphere()
    y = torch.randn(512, 3)
    idx, w = rep.weights(y.numpy())
    assert idx.shape == (512, KNN)
    assert np.abs(w.sum(1) - 1.0).max() < 1e-6 and (w >= 0).all()
    with torch.no_grad():
        rep.a.fill_(0.37)
    assert float((rep(y) - 0.37).abs().max().detach()) < 1e-6
    with torch.no_grad():
        rep.a.normal_(0, 0.1)
    d = rep(y)
    assert float(d.abs().max().detach()) <= float(rep.a.abs().max().detach()) + 1e-6
    assert float(d.min().detach()) < 0 < float(d.max().detach())          # signed: it grows as well as carves


def test_correction_does_not_depend_on_the_query_batch():
    """The correction is a function of the query point alone: evaluating the points in two
    chunks gives the same values as one call. extract_mesh evaluates its grid in chunks, and
    both the depth's neighbour search and the core's own field are cached per chunk."""
    torch.manual_seed(0)
    rep = DepthSphere()
    with torch.no_grad():
        rep.a.normal_(0, 0.1)
    y = torch.randn(3000, 3) * 0.5
    parts = torch.cat([rep(y[:2000]), rep(y[2000:])])
    assert float((rep(y) - parts).abs().max()) < 1e-6

    n = spherical_design(64)
    body = ImplicitBody(normals=n)
    body.set_support(torch.tensor(cube_support(n), dtype=torch.float32))
    with torch.no_grad():
        body.delta.a.normal_(0, 0.05)
    whole = body(y)
    assert float((torch.cat([body(y[:2000]), body(y[2000:])]) - whole).abs().max()) < 1e-6
    # and a cached value is the value: asking again for the same points cannot drift
    assert float((body(y) - whole).abs().max()) == 0.0


def test_the_node_set_is_fixed_and_never_travels_with_a_checkpoint():
    """`a` is the only parameter and the only entry of the state dict; the nodes and the kernel
    width are constants of the representation, so a saved state cannot redefine what another
    body's depths mean."""
    rep = DepthSphere()
    assert [n for n, _ in rep.named_parameters()] == ["a"]
    assert list(rep.state_dict().keys()) == ["a"]
    assert rep.a.numel() == N_NODES
    assert float((rep.u.norm(dim=1) - 1.0).abs().max()) < 1e-6


def test_the_centre_follows_from_the_support_and_moves_with_the_body():
    """The depth is measured from a point that has to be a function of the support alone, or
    the code would not determine the body without a second thing carried beside it. It has to
    move with the body, or the same shape mounted differently would be a different code, and it
    has to be inside, or there is no room to carve at all. The deepest carve the star-shaped
    bound allows is then a fraction of how far the nearest face is from it."""
    n = spherical_design(DESIGN_N)
    h = cube_support(n)
    o = core_centre(h, n)
    assert float(np.abs(o).max()) < 1e-4                    # a cube about the origin
    assert depth_cap(h, o, normals=n) == pytest.approx(0.90 * float((h - n @ o).min()))
    moved = np.array([0.3, -0.2, 0.1])
    off = cube_support(n) + n @ moved                       # the same cube, mounted elsewhere
    assert np.allclose(core_centre(off, n), moved, atol=1e-3)
    # and the carve is measured from it, so a body mounted off the origin keeps its room
    assert depth_cap(off, normals=n) == pytest.approx(depth_cap(h, normals=n), rel=1e-3)
    assert depth_cap(off, normals=n) > depth_cap(off, np.zeros(3), normals=n)


def test_the_extraction_grid_is_the_one_flexicubes_builds():
    """The grid is built arithmetically rather than by deduplicating the corners of every cube,
    which is the largest allocation anything here makes and does not fit in memory at the
    extraction resolution. The two constructions have to agree exactly, vertices and cube
    corners alike, or every extracted body is subtly wrong."""
    from hac26.vendor.flexicubes import FlexiCubes
    fc = FlexiCubes(device="cpu")
    for res in (2, 5, 8, 16):
        v0, c0 = fc.construct_voxel_grid(res)
        v1, c1 = voxel_grid(fc, res, "cpu")
        assert v1.shape == v0.shape and torch.allclose(v0, v1, atol=1e-6)
        assert c1.shape == c0.shape and bool((c0 == c1).all())


def test_dh_is_band_limited_whatever_the_flow_emits():
    """The expanded dh is band-limited to degree SH_DEGREE exactly in the argument of the
    softplus, and approximately in h itself (the softplus slope varies across normals), and
    the resulting support stays positive. An out-of-band dh would kill facets, and a dead
    facet has a zero row in the area Jacobian and so no gradient at all.
    """
    n = spherical_design(64)
    body = ImplicitBody(normals=n)
    body.set_support(torch.tensor(cube_support(n), dtype=torch.float32))
    from hac26.field import real_sh
    y5 = torch.tensor(real_sh(n, 5), dtype=torch.float32)
    torch.manual_seed(0)
    with torch.no_grad():
        body.dh.normal_(0, 0.02)
        arg = body.dh_expand @ body.dh                       # the argument: exactly band-limited
        moved = body.support() - body.core.h                 # h itself: approximately so
    r_arg = arg - y5 @ torch.linalg.lstsq(y5, arg).solution
    assert float(r_arg.norm() / arg.norm()) < 1e-4
    r_h = moved - y5 @ torch.linalg.lstsq(y5, moved).solution
    assert float(r_h.norm() / moved.norm()) < 0.10
    assert bool((body.support() > 0).all())                  # positivity is automatic


def test_every_parameter_block_receives_gradient():
    """Every parameter of ImplicitBody (the support, the depths and dh) receives a non-zero
    gradient from a loss on the field. The caches the field keeps must not break that: a caller
    differentiating through the support has to have the maxima taken again."""
    n = spherical_design(64)
    body = ImplicitBody(normals=n)
    body.set_support(torch.tensor(cube_support(n), dtype=torch.float32))
    (body(torch.randn(512, 3) * 0.5) ** 2).mean().backward()
    for name, prm in body.named_parameters():
        assert prm.grad is not None and float(prm.grad.abs().max()) > 0, f"{name} is dead"


# ------------------------------------------------------------------ constraints

def test_constraints_are_applied_to_vertices():
    """apply_constraints rescales z to [-1, 1] and caps the xy radius at R (1 + tol)."""
    v = np.array([[0.3, 0.0, -4.0], [0.0, 0.4, 6.0], [2.0, 0.0, 1.0]])
    out = apply_constraints(v, radius=1.0, tol=0.03)
    assert out[:, 2].min() == pytest.approx(-1.0, abs=1e-12)
    assert out[:, 2].max() == pytest.approx(+1.0, abs=1e-12)
    r = np.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2).max()
    assert r == pytest.approx(1.03, rel=1e-9)      # clamped to R(1+tol), not to R


def test_radius_tolerance_does_not_shrink_a_body_inside_it():
    """A body whose xy radius is within the tolerance of R is left alone."""
    v = np.array([[1.02, 0, -1.0], [0, 0, 1.0], [-1.02, 0, 0.0]])
    out = apply_constraints(v, radius=1.0, tol=0.03)
    assert np.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2).max() == pytest.approx(1.02, rel=1e-9)


# ------------------------------------------------------------------ the gate

@pytest.mark.slow
def test_cube_extraction_is_planar_and_matches():
    """A cube encoded with h alone extracts with every vertex on a face to within one grid
    cell, and Dice above 0.99 against the analytic cube."""
    n = spherical_design()
    body = ImplicitBody(normals=n)
    body.set_support(torch.tensor(cube_support(n), dtype=torch.float32))
    # the depths and dh are zero, so this is the core alone
    extent = EXTRACT_EXTENT + 0.5
    verts, faces = extract_mesh(lambda y: body(y), extent, res=RES)
    assert len(verts) > 0 and len(faces) > 0

    cell = 2.0 * extent / RES

    # planarity: every vertex of the cube's zero set must lie on one of the six faces,
    # i.e. its largest coordinate magnitude must be A, to within one grid cell
    dev = np.abs(np.abs(verts).max(1) - A)
    assert dev.max() < cell, f"max deviation {dev.max():.5f} exceeds one cell {cell:.5f}"

    # Dice against the analytic cube on a common voxel grid
    g = (np.arange(96) + 0.5) / 96 * 2 * extent - extent
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    truth = (np.abs(X) <= A) & (np.abs(Y) <= A) & (np.abs(Z) <= A)
    # mesh_occupancy rather than trimesh.contains: contains() casts a ray per point and is
    # too expensive on a grid this size without the optional embreex dependency;
    # mesh_occupancy uses the same cell-centre grid and needs nothing optional.
    from hac26.recon import mesh_occupancy
    got = mesh_occupancy(verts, faces, 96, extent)
    dice = 2.0 * (got & truth).sum() / (got.sum() + truth.sum())
    assert dice > 0.99, f"Dice {dice:.4f}"
