"""The sensor chain."""
import torch

from hac26.forward.mesh.sensor import SensorModel


def _oetf_by_index(s: SensorModel, x: torch.Tensor) -> torch.Tensor:
    """The OETF as first written, indexing the knots by the segment of each pixel."""
    k = s.oetf_knots()
    n = len(k) - 1
    u = x.clamp(0.0, 1.0) * n
    i = u.floor().clamp(max=n - 1)
    t = u - i
    i = i.long()
    return torch.lerp(k[i], k[i + 1], t)


def test_the_oetf_matches_the_indexed_spline_in_value_and_gradient():
    """SensorModel.oetf picks its knots by masks because the backward of an index into nine
    knots is slow; the values and both gradients must be those of the index, on the knots,
    at 0 and 1 and outside them as well as between."""
    torch.manual_seed(0)
    s = SensorModel()
    with torch.no_grad():
        s.raw_oetf.copy_(torch.randn(len(s.raw_oetf)))
    n = len(s.raw_oetf)
    x = torch.cat([torch.rand(4000) * 1.2 - 0.1, torch.arange(n + 1) / n,
                   torch.tensor([1.5, -0.5])]).requires_grad_(True)
    w = torch.randn_like(x)
    out = []
    for f in (s.oetf, lambda y: _oetf_by_index(s, y)):
        y = f(x)
        out.append((y, *torch.autograd.grad((w * y).sum(), [x, s.raw_oetf])))
    for mine, ref in zip(*out):
        assert torch.allclose(mine, ref, atol=1e-6)
