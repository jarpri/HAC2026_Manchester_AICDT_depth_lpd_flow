"""The coarea derivative against finite differences.

A smooth image is built from N_PARAM random bumps, and the analytic derivative of each
thresholded reduction with respect to the bump amplitudes is compared with a central finite
difference.

The pixel count is an integer, so a finite difference of it is quantised. The derivative
being checked is that of the continuum area the count approximates, so the finite difference
is taken on an up-sampled image, with a step large enough that the count moves by many pixels
and small enough to stay in the linear regime.
"""
import numpy as np
import pytest
import torch

from hac26.forward.shared.coarea import contour_weights, threshold_count, threshold_sum

N_PARAM = 100
H = W = 256
TAU = 0.5


def _field(theta: torch.Tensor, centres: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """A smooth image built from N_PARAM broad Gaussian bumps, u(x) = sum_k theta_k G_k(x).

    Smooth on the pixel scale, so the finite-difference gradient magnitude is meaningful.
    """
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H, dtype=theta.dtype),
                            torch.linspace(0, 1, W, dtype=theta.dtype), indexing="ij")
    d2 = ((yy[None] - centres[:, 0, None, None]) ** 2
          + (xx[None] - centres[:, 1, None, None]) ** 2)
    # Scaled so u spans roughly [0, 1] and the level set at TAU is a substantial closed
    # contour inside the frame rather than a sliver at the border.
    bumps = theta[:, None, None] * torch.exp(-d2 / (2 * widths[:, None, None] ** 2))
    return bumps.sum(0) / theta.shape[0] * 8.0


@pytest.fixture(scope="module")
def setup():
    g = torch.Generator().manual_seed(7)
    centres = torch.rand(N_PARAM, 2, generator=g, dtype=torch.float64)
    widths = 0.08 + 0.05 * torch.rand(N_PARAM, generator=g, dtype=torch.float64)
    theta = 0.5 + 0.5 * torch.rand(N_PARAM, generator=g, dtype=torch.float64)
    return centres, widths, theta


def _reference(fn, field, up=8):
    """Near-continuum value of the reduction, evaluated on an `up`-times up-sampled field and
    scaled back to the original pixel area.

    The coarea formula differentiates the area of the level set, and the pixel count is an
    integer approximation of it; on the original grid a finite difference of the count is
    quantised and measures that quantisation rather than the derivative.
    """
    import torch.nn.functional as Fn
    u = Fn.interpolate(field[None, None], scale_factor=up, mode="bicubic",
                       align_corners=True)[0, 0]
    return float(fn(u, TAU)) / up ** 2


def _fd_vs_analytic(fn, setup, step):
    """Analytic gradient of fn with respect to theta, and its central finite difference."""
    centres, widths, theta = setup
    t = theta.clone().requires_grad_(True)
    fn(_field(t, centres, widths), TAU).backward()
    ana = t.grad.detach().clone()
    fd = torch.zeros_like(ana)
    for k in range(N_PARAM):
        tp = theta.clone(); tp[k] += step
        tm = theta.clone(); tm[k] -= step
        fd[k] = (_reference(fn, _field(tp, centres, widths))
                 - _reference(fn, _field(tm, centres, widths))) / (2 * step)
    return ana, fd


@pytest.mark.slow
def test_count_derivative_matches_finite_differences(setup):
    """dN/dtheta: correlation above 0.999 and median relative error below 1% over the
    parameters that move the contour."""
    ana, fd = _fd_vs_analytic(threshold_count, setup, step=1e-1)
    keep = fd.abs() > 0.02 * fd.abs().max()      # parameters that move the contour
    rel = ((ana[keep] - fd[keep]).abs() / fd[keep].abs()).median()
    corr = float(np.corrcoef(ana.numpy(), fd.numpy())[0, 1])
    print(f"\n  dN/dtheta : {int(keep.sum())} active params, median rel err {float(rel):.4f}, "
          f"corr {corr:.6f}")
    assert corr > 0.999
    assert float(rel) < 0.01


@pytest.mark.slow
def test_sum_derivative_matches_finite_differences(setup):
    """dI/dtheta: the same bounds as for the count."""
    ana, fd = _fd_vs_analytic(threshold_sum, setup, step=1e-1)
    keep = fd.abs() > 0.02 * fd.abs().max()
    rel = ((ana[keep] - fd[keep]).abs() / fd[keep].abs()).median()
    corr = float(np.corrcoef(ana.numpy(), fd.numpy())[0, 1])
    print(f"  dI/dtheta : {int(keep.sum())} active params, median rel err {float(rel):.4f}, "
          f"corr {corr:.6f}")
    assert corr > 0.999
    assert float(rel) < 0.01


def test_contour_weights_reproduce_the_level_set_length(setup):
    """sum_k w_k |grad u|(rows_k, cols_k) must equal the total contour length to within 5%."""
    centres, widths, theta = setup
    u = _field(theta, centres, widths)
    b, r, c, w = contour_weights(u[None], torch.tensor([TAU], dtype=u.dtype))
    from skimage.measure import find_contours
    length = sum(float(np.sqrt(((k[1:] - k[:-1]) ** 2).sum(1)).sum())
                 for k in find_contours(u.numpy(), TAU))
    gy, gx = np.gradient(u.numpy())
    g = torch.as_tensor(np.sqrt(gx ** 2 + gy ** 2))
    recovered = float((w * g[r, c]).sum())
    assert recovered == pytest.approx(length, rel=0.05)


def test_no_softening_anywhere():
    """The forward values are the hard count and the hard sum, not softened versions."""
    u = torch.tensor([[0.0, 0.4], [0.6, 1.0]], dtype=torch.float64, requires_grad=True)
    assert float(threshold_count(u, 0.5)) == 2.0
    assert float(threshold_sum(u, 0.5)) == pytest.approx(1.6)


def test_batched_images_and_thresholds(setup):
    """A batch with one threshold per image gives the same counts, sums and gradients as the
    images one at a time."""
    centres, widths, theta = setup
    u = _field(theta, centres, widths)
    ub = torch.stack([u, 0.7 * u]).requires_grad_(True)
    taus = torch.tensor([TAU, 0.3], dtype=u.dtype, requires_grad=True)
    n = threshold_count(ub, taus)
    assert n.shape == (2,)
    assert float(n[0]) == float((u > TAU).sum()) and float(n[1]) == float((0.7 * u > 0.3).sum())
    (n.sum() + threshold_sum(ub, taus).sum()).backward()
    g_batch, gt_batch = ub.grad.clone(), taus.grad.clone()
    for k, (img, t) in enumerate(((u, TAU), (0.7 * u, 0.3))):
        x = img.detach().clone().requires_grad_(True)
        tt = torch.tensor(t, dtype=u.dtype, requires_grad=True)
        (threshold_count(x, tt) + threshold_sum(x, tt)).backward()
        assert torch.allclose(x.grad, g_batch[k])
        assert torch.allclose(tt.grad, gt_batch[k])


def test_threshold_derivative_matches_finite_differences(setup):
    """d/dtau of the count equals minus the contour weight, checked against a central
    finite difference of the near-continuum area."""
    centres, widths, theta = setup
    u = _field(theta, centres, widths)
    tau = torch.tensor(TAU, dtype=u.dtype, requires_grad=True)
    threshold_count(u, tau).backward()
    step = 2e-3
    fd = (_ref_at(u, TAU + step) - _ref_at(u, TAU - step)) / (2 * step)
    assert float(tau.grad) == pytest.approx(fd, rel=0.1)


def _ref_at(u, tau, up=8):
    import torch.nn.functional as Fn
    uu = Fn.interpolate(u[None, None], scale_factor=up, mode="bicubic",
                        align_corners=True)[0, 0]
    return float((uu > tau).sum()) / up ** 2
