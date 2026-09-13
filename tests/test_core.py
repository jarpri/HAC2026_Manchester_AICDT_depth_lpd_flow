"""Tests of the convex forward model, its derivatives and adjoints, the Minkowski solver and
the curve-file parser. Tests that need torch are skipped when it is not installed."""
import numpy as np
import pytest

from hac26.data_io import read_curves29, write_curves29
from hac26.forward.convex_egi import (build_A, deriv_adjoint_np, dn_adjoint_np, dn_np,
                           forward_np, normalize_np, stack_A)
from hac26.geometry import (build_cameras, make_grid, project_closure)
from hac26.solvers.minkowski import solve_minkowski
from hac26.shapes import (face_normals_areas, icosphere, mesh_curves_convex,
                          mesh_to_egi, sample_training_shape)

RNG = np.random.default_rng(0)


def small_setup(nt=6, nphi=12, m=24):
    grid = make_grid(nt, nphi)
    cams = build_cameras()
    A, types = stack_A(grid, cams, m)
    return grid, cams, A, types


def test_camera_table():
    """hac26.geometry.build_cameras gives the released column order and phase angles."""
    cams = build_cameras()
    assert len(cams) == 28
    a0 = [c for c in cams if c.azimuth_deg == 0.0]
    assert [c.kind for c in a0] == ["hor_a", "hor_b", "top", "bottom"]
    # the horizontal camera at azimuth 0 looks along the light: phase angle 0
    assert abs(a0[0].phase_angle_deg) < 1e-12
    # at azimuth 0 the top camera's phase angle equals its elevation
    assert abs(a0[2].phase_angle_deg - 21.0) < 1e-9


def test_pole_columns_zero():
    """The pole normals are never lit by a light in the equatorial plane, so their columns
    of A are exactly zero."""
    cams = build_cameras()
    normals = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0]])
    A = build_A(normals, cams + cams, 16,
                ["intensity"] * 28 + ["binary"] * 28)
    assert np.all(A[:, :, 0] == 0.0)
    assert np.all(A[:, :, 1] == 0.0)
    assert A[:, :, 2].max() > 0  # sanity: equatorial normal does contribute


def test_dn_derivative_and_adjoint():
    """dn_np matches a finite difference of normalize_np, and dn_adjoint_np is its adjoint."""
    y = RNG.uniform(1.0, 2.0, size=(5, 30))
    v = RNG.standard_normal((5, 30))
    w = RNG.standard_normal((5, 30))
    # finite differences of N
    t = 1e-7
    fd = (normalize_np(y + t * v) - normalize_np(y - t * v)) / (2 * t)
    assert np.allclose(fd, dn_np(y, v), atol=1e-5)
    # adjoint identity <DN v, w> = <v, DN* w>
    lhs = float((dn_np(y, v) * w).sum())
    rhs = float((v * dn_adjoint_np(y, w)).sum())
    assert abs(lhs - rhs) < 1e-10 * max(1.0, abs(lhs))


def test_scale_invariance():
    """The normalised forward model does not change when the EGI is scaled."""
    grid, cams, A, types = small_setup()
    g = RNG.uniform(0, 1, grid.n)
    assert np.allclose(forward_np(A, g), forward_np(A, 7.3 * g))


def test_deriv_adjoint_vs_fd():
    """<h, dF(g) v> from a finite difference equals <v, deriv_adjoint_np(A, g, h)>."""
    grid, cams, A, types = small_setup()
    g = RNG.uniform(0.1, 1.0, grid.n)
    h = RNG.standard_normal(A.shape[:2])
    t = 1e-6
    v = RNG.standard_normal(grid.n)
    fd = float(((forward_np(A, g + t * v) - forward_np(A, g - t * v)) / (2 * t) * h).sum())
    an = float(v @ deriv_adjoint_np(A, g, h))
    assert abs(fd - an) < 1e-4 * max(1.0, abs(fd))


def test_azimuthal_equivariance():
    """Rotating the EGI by one phi cell shifts every raw curve cyclically by m / n_phi frames,
    exactly on the grid."""
    nt, nphi, m = 6, 12, 24
    grid = make_grid(nt, nphi)
    # The psi0 offset keeps the samples away from mu0 = 0, where the discontinuous binary
    # kernel turns floating-point sign noise into whole-pixel flips.
    A, types = stack_A(grid, build_cameras(), m, psi0=0.1234)
    g = RNG.uniform(0, 1, grid.n)
    gimg = g.reshape(nt, nphi)
    g_rot = np.roll(gimg, 1, axis=1).reshape(-1)  # body rotated by beta = 2*pi/nphi
    s = m // nphi
    y = np.einsum("cmn,n->cm", A, g)
    y_rot = np.einsum("cmn,n->cm", A, g_rot)
    # y_rot[k] = y[k + s]  (body rotated by +beta <=> curves advanced by s frames)
    assert np.allclose(y_rot, np.roll(y, -s, axis=1), atol=1e-10)


def test_closure_projection():
    """project_closure keeps the EGI non-negative and makes its area vector vanish."""
    grid = make_grid(8, 16)
    g = RNG.uniform(0, 1, grid.n)
    gp = project_closure(g, grid.normals)
    assert gp.min() >= 0
    assert np.linalg.norm(grid.normals.T @ gp) < 1e-5 * gp.sum()


def test_minkowski_cube():
    """Six axis normals with equal areas solve to a unit cube."""
    U = np.array([[1., 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    sol = solve_minkowski(U, np.ones(6))
    assert sol["egi_l1"] < 0.02, sol["message"]
    a = sol["areas_kept"]
    assert np.allclose(a / a.mean(), 1.0, atol=0.02)  # equal facet areas -> cube
    assert abs(sol["volume"] - 1.0) < 1e-3


def test_minkowski_roundtrip_and_bruteforce():
    """An EGI g is solved to a Minkowski polytope; (a) the polytope's EGI is close to g in
    proportion, and (b) the mesh curves of the polytope match A @ g_back, since its facet
    normals are grid nodes by construction."""
    nt, nphi, m = 8, 16, 32
    grid = make_grid(nt, nphi)
    cams = build_cameras()
    g = np.zeros(grid.n)
    idx = RNG.choice(grid.n, 40, replace=False)
    g[idx] = RNG.uniform(0.5, 1.5, 40)
    g = project_closure(g, grid.normals)
    sol = solve_minkowski(grid.normals, g, drop_tol=1e-6)
    assert sol["egi_l1"] < 0.1, (sol["egi_l1"], sol["message"])
    # (a) proportional EGI roundtrip
    g_back = mesh_to_egi(sol["verts"], sol["faces"], grid, close=False)
    p, pb = g / g.sum(), g_back / g_back.sum()
    assert np.abs(p - pb).sum() < 0.05
    # (b) the matrix route against the facet sum on the same polytope; the psi0 offset keeps
    # the samples away from the level the binary kernel steps at, where a discontinuous
    # kernel amplifies floating-point differences between the two normal computations. The
    # matrix is built from the same extended Gaussian image the facet sum will be taken over,
    # so both derive the same threshold from the same first frame; a matrix built without one
    # thresholds at zero and counts a different set of facets.
    A, types = stack_A(grid, cams, m, areas=g_back, psi0=0.789)
    y_mat = np.einsum("cmn,n->cm", A, g_back)
    y_brt = mesh_curves_convex(sol["verts"], sol["faces"], cams + cams, m, types,
                               psi0=0.789)
    denom = np.abs(y_brt).max()
    assert np.abs(y_mat - y_brt).max() < 5e-3 * denom


def test_sample_shape_and_egi():
    """A sampled training shape has a non-negative closed EGI and spans z in [-1, 1]."""
    grid = make_grid(12, 24)
    s = sample_training_shape(np.random.default_rng(3), grid)
    assert s["g"].min() >= 0 and s["g"].sum() > 0
    assert abs(s["verts"][:, 2].max() - 1) < 1e-9 and abs(s["verts"][:, 2].min() + 1) < 1e-9
    assert np.linalg.norm(grid.normals.T @ s["g"]) < 1e-4 * s["g"].sum()


def test_parser_roundtrip(tmp_path):
    """write_curves29 followed by read_curves29 returns the time and curves unchanged."""
    m = 17
    time = np.arange(m, dtype=float)
    curves = RNG.uniform(0.5, 1.5, size=(28, m))
    p = tmp_path / "Asteroid99_lightcurve_intensity.txt"
    write_curves29(str(p), time, curves)
    back = read_curves29(str(p))
    assert np.allclose(back["time"], time)
    assert np.allclose(back["curves"], curves)


# ---------------- torch-dependent tests ----------------------------------------------
def test_torch_operator_matches_numpy():
    """ConvexPhotometricOperator agrees with forward_np."""
    torch = pytest.importorskip("torch")
    from hac26.forward.convex_egi import ConvexPhotometricOperator
    grid, cams, A, types = small_setup()
    op = ConvexPhotometricOperator(A)
    g = RNG.uniform(0.1, 1.0, grid.n)
    y_np = forward_np(A, g)
    y_t = op(torch.tensor(g, dtype=torch.float32)[None])[0].numpy()
    assert np.abs(y_np - y_t).max() < 1e-4


def test_torch_deriv_adjoint_matches_autograd():
    """The operator's closed-form deriv_adjoint equals the autograd gradient."""
    torch = pytest.importorskip("torch")
    from hac26.forward.convex_egi import ConvexPhotometricOperator
    grid, cams, A, types = small_setup()
    op = ConvexPhotometricOperator(A)
    op.double()
    g = torch.tensor(RNG.uniform(0.1, 1.0, grid.n)[None], requires_grad=True)
    h = torch.tensor(RNG.standard_normal(A.shape[:2]))[None]
    (op(g) * h).sum().backward()
    closed = op.deriv_adjoint(g.detach(), h)
    assert torch.allclose(g.grad, closed, atol=1e-9)


def test_lpd_forward_backward():
    """LPDNet returns an EGI that sums to one per sample, and every trainable parameter
    receives a gradient."""
    torch = pytest.importorskip("torch")
    from hac26.forward.convex_egi import ConvexPhotometricOperator
    from hac26.solvers.lpd_convex import LPDNet
    nt, nphi, m = 6, 12, 24
    grid, cams, A, types = small_setup(nt, nphi, m)
    op = ConvexPhotometricOperator(A)
    net = LPDNet(op, nt, nphi, cams + cams, types, n_iter=3, n_primal=3,
                 n_dual=3, ch=8)
    d = torch.rand(2, A.shape[0], m)
    mask = torch.ones(2, A.shape[0])
    p, g = net(d, mask)
    assert p.shape == (2, grid.n)
    assert torch.allclose(p.sum(1), torch.ones(2), atol=1e-5)
    p.sum().backward()
    assert all(q.grad is not None for q in net.parameters() if q.requires_grad)


def test_the_count_is_thresholded_at_the_level_the_first_frame_gives():
    """The organisers count pixels above Otsu's level of each video's own first frame, so the
    level is derived and not fitted. Otsu's criterion needs only the histogram of a frame, and
    for a convex body that histogram is known from the facets, so the level can be computed
    without rendering anything. It has to be the level a rendered frame of the same body
    gives, and it has to be insensitive to how much of the frame the body is set in, since
    the framing is not something the released files say."""
    from hac26.forward.convex_egi import otsu_threshold
    from hac26.conventions import TRANSFER_EXPONENT

    v, f = icosphere(3)
    v = np.asarray(v) * np.array([1.0, 0.7, 1.2])
    n, a = face_normals_areas(v, np.asarray(f))
    s = np.array([-1.0, 0.0, 0.0])
    view = np.array([-0.7, 0.7, 0.0]) / np.sqrt(2.0)
    mu, mu0 = n @ view, n @ s
    proj = 0.5 * float(np.abs(mu) @ a)

    # the same level, whatever the body is set in
    levels = [otsu_threshold(mu, mu0, a, k * proj) for k in (1.5, 2.5, 5.0)]
    assert max(levels) - min(levels) < 0.01, levels
    assert 0.0 < levels[0] < 1.0

    # and it is the level a histogram of the frame gives, by the criterion's own definition:
    # no other level separates the frame into two parts of larger between-class variance
    c = levels[1]
    lit = (mu > 0) & (mu0 > 0)
    val = np.where(lit, np.abs(mu0) ** TRANSFER_EXPONENT, 0.0)
    w = np.where(lit, a * mu, 0.0)
    dark = max(2.5 * proj - w.sum(), 0.0)

    def between(level):
        hi = val > level ** TRANSFER_EXPONENT
        w1 = float(w[hi].sum())
        w0 = float(w[~hi].sum()) + dark
        if w0 <= 0 or w1 <= 0:
            return -1.0
        m1 = float((w[hi] * val[hi]).sum()) / w1
        m0 = float((w[~hi] * val[~hi]).sum()) / w0
        return w0 * w1 * (m0 - m1) ** 2

    best = max(np.linspace(0.02, 0.95, 94), key=between)
    assert abs(best - c) < 0.03, (best, c)
