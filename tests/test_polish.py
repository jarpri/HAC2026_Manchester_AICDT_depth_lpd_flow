"""The last step of a reconstruction and the training term that mirrors it: the polish of a
draw on the exact misfit, and the data-fit term's value and gradient."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hac26.conventions import psi_grid                                         # noqa: E402
from hac26.field import (CODE_DIM, N_DIR, N_NODES, DepthSphere,               # noqa: E402
                         ImplicitBody, node_kernel)
from hac26.forward.mesh.exact import RenderConfig                              # noqa: E402
from hac26.forward.mesh.instrument import Instrument                           # noqa: E402
from hac26.shapes import icosphere, mesh_support                              # noqa: E402
from hac26.solvers.lpd_flow import LPDFlow                                     # noqa: E402
from hac26.solvers.operator import CodeOperator                                # noqa: E402
from reconstruct_lpd import POLISH_TARGET, polish, whitened_misfit             # noqa: E402
from train_lpd import FIT_KNEE, data_fit, with_gradient                        # noqa: E402

# The polish is a line search on the rendered misfit, so the render has to be fine enough
# that the misfit falls smoothly along the descent direction. At 24x40 with no supersampling
# it does not: the body covers ~15 pixels across, the binary count moves in whole pixels, and
# the misfit wobbles by more between neighbouring step sizes than the step buys. The line
# search then stops on whichever of its five step sizes happens to land on a decrease, which
# is luck rather than a property of the polish. At this resolution the misfit is monotone
# below the first step size and the descent is a descent.
SMALL = RenderConfig(height=48, width=80, supersample=2, sun_res=128, phase_chunk=4,
                     radiosity_faces=48)
GEOMS = [0, 7]

_NODES = None
_KERNEL = None


def _nodes() -> np.ndarray:
    global _NODES
    if _NODES is None:
        _NODES = DepthSphere(N_NODES).u.numpy()
    return _NODES


def _smooth(a: torch.Tensor, spacings: float = 3.0) -> torch.Tensor:
    """`a` smoothed over about `spacings` node spacings, by repeated application of the node
    kernel.

    The bodies here stand in for what a fit of a real body produces, and that is smooth at the
    node scale: the depths come from a least-squares solve over surface samples with a ridge,
    and the search that moves them is capped in angular degree. White noise on the nodes is a
    body no stage of this pipeline makes, and its rendered misfit is rough at exactly the scale
    a finite-difference check steps over."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = node_kernel()
    out = a.detach().numpy().astype(np.float64)
    for _ in range(int(round(spacings ** 2))):
        out = _KERNEL @ out
    return torch.tensor(out, dtype=torch.float32)


def _at_depth(a: torch.Tensor, depth: float) -> torch.Tensor:
    """Depths rescaled so that the deepest of them is `depth`.

    The bodies here are named by the dents they have, and a dent is a depth. The node weights
    are a partition of unity, so the field is bounded by the largest coefficient and reaches it
    wherever a few neighbours agree; scaling the largest coefficient therefore names the dent in
    body units at any node count."""
    made = float(a.detach().abs().max())
    return a * (depth / made) if made > 1e-9 else a * 0.0


def _blob(direction, width_deg: float) -> torch.Tensor:
    """A single dent: a smooth cap of the given angular width, in the given direction."""
    u = _nodes()
    d = np.asarray(direction, float); d = d / np.linalg.norm(d)
    ang = np.arccos(np.clip(u @ d, -1.0, 1.0))
    return torch.tensor(np.exp(-(ang / np.deg2rad(width_deg)) ** 2), dtype=torch.float32)


def _setup():
    op = CodeOperator(Instrument(quantise=False), psi_grid(4), res=16, config=SMALL,
                      device="cpu", backend="software")
    v, _ = icosphere(2)
    v = v * [0.9, 0.7, 1.0]
    h = torch.tensor(mesh_support(v, ImplicitBody().core.n.numpy()), dtype=torch.float32)
    target = torch.zeros(CODE_DIM)
    # the dent the data know about, off both axes so that no symmetry hides it
    target[N_DIR:] = _at_depth(_blob((0.35, -0.45, 0.0), 25.0), 0.30)
    data = op.curves(h, target, 1.2, geoms=GEOMS)       # (2, 2, P) of the geometries used
    full = torch.ones(28, 2, data.shape[-1])
    full[GEOMS] = data
    scale = torch.full((28, 2), 0.01)
    net = LPDFlow(n_experts=1)
    gen = torch.Generator().manual_seed(0)
    codes = torch.zeros(8, CODE_DIM)
    codes[:, :N_DIR] = 0.05 * torch.randn(8, N_DIR, generator=gen)
    for i in range(len(codes)):                         # bodies dented to a tenth of a radius
        codes[i, N_DIR:] = _at_depth(_smooth(torch.randn(N_NODES, generator=gen)), 0.10)
    net.codec.fit(codes)
    start = torch.zeros(CODE_DIM)                       # a generic body without the dent
    start[N_DIR:] = _at_depth(_smooth(torch.randn(N_NODES, generator=gen)), 0.05)
    return op, h, net, full, scale, net.codec.encode(start)


def test_polish_lowers_the_misfit_and_stops_at_the_noise_level():
    op, h, net, data, scale, z = _setup()
    z2, before, after, n_it = polish(net, op, z, h, 1.2, data, scale, GEOMS, steps=3)
    assert before > POLISH_TARGET
    assert after < before and 1 <= n_it <= 3
    cur = op.curves(h, net.codec.decode(z2), 1.2, geoms=GEOMS)
    assert abs(whitened_misfit(cur, data, scale, GEOMS) - after) < 1e-4
    # a draw already at the noise level is left where it is
    fine = torch.full((28, 2), 100.0)                   # a scale so large that chi < 1
    z3, b3, a3, n3 = polish(net, op, z, h, 1.2, data, fine, GEOMS, steps=3)
    assert n3 == 0 and torch.equal(z3, z) and a3 == b3 <= POLISH_TARGET


def test_data_fit_is_zero_at_the_noise_level_and_its_gradient_points_downhill():
    op, h, net, data, scale, z = _setup()
    z = z[None]
    mask = torch.zeros(1, 28)
    mask[0, GEOMS] = 1.0
    val, grad, dropped = data_fit(net, op, z, h[None], torch.tensor([1.2]), data[None],
                                  scale[None], GEOMS, mask)
    assert dropped == 0 and float(val) > 0 and torch.isfinite(grad).all()
    # steps against the gradient land lower than the same steps along it, and the best of
    # them below the start (the test renderer is coarse and its misfit rough at small scales,
    # so the check is over a few step sizes rather than one)
    direction = grad / grad.pow(2).mean().sqrt()
    downs, ups = [], []
    for eps in (0.003, 0.01, 0.03):
        downs.append(float(data_fit(net, op, z - eps * direction, h[None], torch.tensor([1.2]),
                                    data[None], scale[None], GEOMS, mask)[0]))
        ups.append(float(data_fit(net, op, z + eps * direction, h[None], torch.tensor([1.2]),
                                  data[None], scale[None], GEOMS, mask)[0]))
    assert all(d < u for d, u in zip(downs, ups)) and min(downs) < float(val)
    # at the noise level the term and its gradient vanish
    fine = torch.full((1, 28, 2), 100.0)
    val3, grad3, _ = data_fit(net, op, z, h[None], torch.tensor([1.2]), data[None], fine,
                              GEOMS, mask)
    assert float(val3) == 0.0 and float(grad3.abs().max()) == 0.0
    # far above the noise the term grows in proportion to the excess rather than as its
    # square: it goes on separating a poor body from a hopeless one, and goes on pulling,
    # where a term that levelled off would report the same value and no gradient for both
    coarse = torch.full((1, 28, 2), 1e-3)
    val4, grad4, _ = data_fit(net, op, z, h[None], torch.tensor([1.2]), data[None], coarse,
                              GEOMS, mask)
    coarser = torch.full((1, 28, 2), 1e-4)
    val5, grad5, _ = data_fit(net, op, z, h[None], torch.tensor([1.2]), data[None], coarser,
                              GEOMS, mask)
    assert float(val5) > float(val4) > 0
    assert float(grad4.abs().max()) > 0 and float(grad5.abs().max()) > 0
    cur = op.curves(h, net.codec.decode(z[0]), 1.2, geoms=GEOMS)
    excess = whitened_misfit(cur, data, coarse[0], GEOMS) - 1.0
    assert excess > FIT_KNEE
    assert abs(float(val4) - FIT_KNEE * (2 * excess - FIT_KNEE)) < 1e-3 * float(val4)
    # with_gradient carries the value and the gradient into autograd
    x = z.clone().requires_grad_(True)
    s = with_gradient(val, x, grad)
    s.backward()
    assert torch.allclose(x.grad, grad) and float(s) == float(val)
