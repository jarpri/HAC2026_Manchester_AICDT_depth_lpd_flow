"""The code operator on the software rasteriser: curves of a code, and the adjoint back onto
the code."""
import numpy as np
import pytest
import torch

from hac26.conventions import psi_grid
from hac26.field import CODE_DIM, N_DIR, ImplicitBody
from hac26.forward.mesh.exact import RenderConfig
from hac26.forward.mesh.instrument import Instrument
from hac26.shapes import mesh_support, icosphere
from hac26.solvers.operator import CodeOperator

SMALL = RenderConfig(height=24, width=40, supersample=1, sun_res=64, phase_chunk=4,
                     radiosity_faces=48)


def _operator(P=4):
    return CodeOperator(Instrument(quantise=False), psi_grid(P), res=16, config=SMALL,
                        device="cpu", backend="software")


def _ellipsoid_support():
    v, _ = icosphere(2)
    v = v * np.array([0.9, 0.7, 1.0])
    n = ImplicitBody().core.n.numpy()
    return torch.tensor(mesh_support(v, n), dtype=torch.float32)


def test_curves_have_the_right_shape_and_are_normalised():
    op = _operator()
    cur = op.curves(_ellipsoid_support(), torch.zeros(CODE_DIM), 1.2, geoms=[0, 7])
    assert cur.shape == (2, 2, 4)
    assert torch.allclose(cur.mean(-1), torch.ones(2, 2), atol=1e-5)


def test_adjoint_reproduces_the_curves_and_reaches_both_blocks():
    """The adjoint pass returns the same curves as the plain pass and a finite gradient on
    both the dh block and the g block."""
    op = _operator()
    h = _ellipsoid_support()
    code = torch.zeros(CODE_DIM)
    code[N_DIR:] = 0.05 * torch.randn(CODE_DIM - N_DIR, generator=torch.Generator().manual_seed(0))
    plain = op.curves(h, code, 1.2, geoms=[0, 7])
    # the cotangent of half the sum of squares; a constant cotangent would be degenerate,
    # since the sum of a mean-normalised curve does not depend on the body
    cur, grad = op.adjoint(h, code, 1.2, lambda c: c, geoms=[0, 7])
    assert torch.allclose(plain, cur)
    assert grad.shape == (CODE_DIM,) and torch.isfinite(grad).all()
    assert grad[:N_DIR].abs().sum() > 0 and grad[N_DIR:].abs().sum() > 0


def test_a_degenerate_code_gives_no_curves():
    """Amplitudes large enough to push the whole field positive leave no surface to extract."""
    op = _operator()
    code = torch.zeros(CODE_DIM)
    code[N_DIR:] = 50.0
    assert op.curves(_ellipsoid_support(), code, 1.0) is None
    assert op.adjoint(_ellipsoid_support(), code, 1.0, lambda c: c) == (None, None)


def test_the_radius_changes_the_curves():
    """The same canonical body rendered at two radii gives different curves: a body stretched
    sideways looks different to the horizontal cameras."""
    op = _operator()
    h = _ellipsoid_support()
    a = op.curves(h, torch.zeros(CODE_DIM), 1.0, geoms=[0, 9])
    b = op.curves(h, torch.zeros(CODE_DIM), 2.0, geoms=[0, 9])
    assert not torch.allclose(a, b, atol=1e-3)


def test_canonical_pose_is_imposed_and_differentiable():
    """A mesh stretched and scaled comes back touching z = -1 and +1, with its largest xy
    radius one, and the pose has a gradient in the vertices."""
    v, f = icosphere(2)
    v = torch.tensor(v * np.array([0.6, 1.4, 0.3]) + np.array([0.0, 0.0, 0.5]),
                     dtype=torch.float32).requires_grad_(True)
    ft = torch.tensor(f)
    c = CodeOperator.canonical(v, ft)
    assert float(c[:, 2].min()) == -1.0 and float(c[:, 2].max()) == 1.0
    assert abs(float(c[:, :2].norm(dim=1).max()) - 1.0) < 1e-6
    p = CodeOperator.physical(v, ft, 1.7)
    assert abs(float(p[:, :2].norm(dim=1).max()) - 1.7) < 1e-5
    p.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_canonical_pose_does_not_move_the_body_off_the_rotation_axis():
    """The pose rescales; it must not translate in xy. The rotation axis is the z axis of the
    frame the body is already in, not the line through its centre of mass, so a body sitting
    off the axis has to stay there -- that offset is geometry, and the released public STLs
    carry one."""
    v, f = icosphere(2)
    off = torch.tensor(v + np.array([0.35, -0.2, 0.0]), dtype=torch.float32)
    ft = torch.tensor(f)
    c = CodeOperator.canonical(off, ft)
    scale = float(c[:, :2].norm(dim=1).max()) / float(off[:, :2].norm(dim=1).max())
    assert torch.allclose(c[:, :2], off[:, :2] * scale, atol=1e-6)   # a pure scaling
    assert float(c[:, :2].mean(0).norm()) > 0.1                      # still off the axis


def test_the_radial_term_displaces_the_surface_by_its_own_amount():
    """The reshaping coefficients are lengths. On a convex core the field's gradient has unit
    norm on a facet, so adding a constant field moves the level set by that constant, and a
    degree-two coefficient moves the surface along its own direction by about its size. This
    is why the reshaping is added to the field rather than to the support values inside a
    softplus, whose slope varies across the normals."""
    from hac26.field import ImplicitBody, N_RADIAL, radial_basis

    body = ImplicitBody()
    body.set_support(torch.full((body.core.n.shape[0],), 0.8))

    def radius_along(u, c):
        """Where the field crosses zero along the ray u, by bisection."""
        lo, hi = torch.tensor(0.05), torch.tensor(3.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            f = body(mid * u[None], c=c)[0]
            lo, hi = torch.where(f < 0, mid, lo), torch.where(f < 0, hi, mid)
        return float(0.5 * (lo + hi))

    u = torch.tensor([0.48, -0.64, 0.6])
    u = u / u.norm()
    base = radius_along(u, None)
    assert base == pytest.approx(0.8, abs=0.02)          # the polytope's own facet

    for k, amount in ((0, 0.12), (3, -0.09), (6, 0.07)):
        c = torch.zeros(N_RADIAL)
        c[k] = amount
        expected = base - amount * float(radial_basis(u[None])[0, k])
        assert radius_along(u, c) == pytest.approx(expected, abs=0.02)

    # zero coefficients are the body the code alone describes
    y = torch.randn(64, 3)
    assert torch.allclose(body(y, c=torch.zeros(N_RADIAL)), body(y), atol=1e-6)
