"""The training loss end to end on the software rasteriser: the operator at the prior's
endpoint estimate, the reader and expert in the loop, the data-fit term for late t, turned
bodies and the ablation arm."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hac26.conventions import cameras, psi_grid                                # noqa: E402
from hac26.field import CODE_DIM, DESIGN_N, N_DIR, ImplicitBody                # noqa: E402
from hac26.forward.mesh.exact import RenderConfig                              # noqa: E402
from hac26.forward.mesh.instrument import Instrument                           # noqa: E402
from hac26.shapes import icosphere, mesh_support                              # noqa: E402
from hac26.solvers.lpd_flow import LPDFlow, geometry_tags                     # noqa: E402
from hac26.solvers.operator import CodeOperator                                # noqa: E402
from train_lpd import (FIT_FROM, OCC_LOGIT, OCC_WEIGHT, Corpus, Diag,          # noqa: E402
                       flow_loss, model_error_scale, occ_eps_default, step_loss)

SMALL = RenderConfig(height=24, width=40, supersample=1, sun_res=64, phase_chunk=4,
                     radiosity_faces=48)


def _corpus(op):
    """Two bodies with all their curves and turned counts, at P = 4 phases."""
    gen = torch.Generator().manual_seed(0)
    n = ImplicitBody().core.n.numpy()
    codes, curves, turned, sups = [], [], [], []
    for scale in ([0.9, 0.7, 1.0], [1.0, 0.8, 1.1]):
        v, _ = icosphere(2)
        h = torch.tensor(mesh_support(v * scale, n), dtype=torch.float32)
        code = torch.zeros(CODE_DIM)
        code[N_DIR:] = 0.05 * torch.randn(CODE_DIM - N_DIR, generator=gen)
        c, t = op.curves_turned(h, code, 1.2)
        codes.append(code); curves.append(c); turned.append(t); sups.append(h)
    stack = lambda xs: torch.stack(xs)                                           # noqa: E731
    return Corpus(stack(codes), stack(curves), stack(turned), stack(sups), stack(sups),
                  torch.tensor([1.2, 1.2]), torch.tensor([0, 1]))


def test_flow_loss_trains_reader_and_expert_with_every_term():
    inst = Instrument(quantise=False)
    op = CodeOperator(inst, psi_grid(4), res=16, config=SMALL, device="cpu",
                      backend="software")
    corpus = _corpus(op)
    net = LPDFlow(n_experts=1)
    net.codec.fit(corpus.codes)
    net.prior.requires_grad_(False)        # as train_lpd.load_prior leaves it
    with torch.no_grad():                  # open the zero-initialised output paths, so the
        for p in net.experts.parameters():   # reader's gradient is not zero by construction
            p.add_(0.01 * torch.randn_like(p))
    eta = model_error_scale(inst)
    C = len(cameras())
    tag, mask = geometry_tags(), torch.ones(1, C)
    idx = torch.tensor([0, 1])
    x0 = torch.randn(2, CODE_DIM, generator=torch.Generator().manual_seed(1))
    t = torch.tensor([0.3, 0.5 * (FIT_FROM + 1.0)])         # one early draw, one in the fit interval
    turns = torch.tensor([0, 1])
    loss, diag = flow_loss(net, op, corpus, eta, idx, x0, t, 2, tag, mask, train_geoms=2,
                           turns=turns, return_diag=True)
    assert isinstance(diag, Diag) and torch.isfinite(loss)
    assert diag.fit > 0 and diag.flow > 0 and diag.occ > 0 and diag.dropped == 0
    loss.backward()
    reader_grad = sum(float(p.grad.abs().sum()) for p in net.reader.parameters()
                      if p.grad is not None)
    expert_grad = sum(float(p.grad.abs().sum()) for p in net.experts[0].parameters()
                      if p.grad is not None)
    assert reader_grad > 0 and expert_grad > 0
    assert all(p.grad is None for p in net.prior.parameters())     # the prior is not trained here
    # the ablation arm: the prior alone, on the same draws
    full, prior_only, dropped = flow_loss(net, op, corpus, eta, idx, x0, t, 2, tag, mask,
                                          train_geoms=2, turns=turns, ablate=True)
    assert torch.isfinite(full) and torch.isfinite(prior_only) and dropped == 0


def _carved_body():
    """A unit ball and the same ball with a pocket carved into it: the hull support, the true
    code, and which of the probe rays the carve reaches."""
    from hac26.field import DepthSphere, N_NODES
    nodes = DepthSphere(N_NODES).u
    axis = torch.tensor([1.0, 0.0, 0.0])               # a pocket in the side of the ball
    carved = nodes @ axis > np.cos(np.deg2rad(25.0))
    sup = torch.ones(1, DESIGN_N)                      # the support of the unit ball
    code = torch.zeros(1, CODE_DIM)
    code[0, N_DIR:][carved] = 0.45                     # depths that take the surface inward
    return sup, code, carved


def _occ(net, sup, h_base, code_true, z_est):
    """The occupancy term alone, for an endpoint estimate z_est in the whitened code."""
    zero = torch.zeros_like(z_est)
    x1 = net.codec.encode(code_true)
    return float(step_loss(net, zero, torch.zeros(1), z_est, x1, sup, code_true, h_base,
                           OCC_WEIGHT, occ_eps_default())[2])


def test_the_occupancy_term_is_mostly_about_the_carve(monkeypatch):
    """The convex stage already supplies the hull, so the term has to be dominated by the
    probes the hull gets wrong. Missing the carve entirely costs far more with the weighting
    than without it, while a convex body's term is untouched."""
    import train_lpd
    sup, code_true, carved = _carved_body()
    net = LPDFlow(n_experts=1)
    net.codec.fit(torch.cat([code_true, torch.zeros(1, CODE_DIM)]))
    hull = net.codec.encode(torch.zeros(1, CODE_DIM))          # the body without its carve
    truth = net.codec.encode(code_true)

    def gap():
        return _occ(net, sup, sup, code_true, hull) - _occ(net, sup, sup, code_true, truth)

    weighted = gap()
    monkeypatch.setattr(train_lpd, "CARVE_WEIGHT", 0.0)
    plain = gap()
    assert weighted > 5.0 * plain, (weighted, plain)
    # the carve is a few percent of the probes, and the weighting is what makes it count
    assert float(carved.float().mean()) < 0.1


def test_the_occupancy_term_is_bounded_by_a_wild_field():
    """A decoded field of many probe spacings says no more about where the surface is than
    one of a single spacing, and unbounded it would swamp the step."""
    sup, code_true, _ = _carved_body()
    net = LPDFlow(n_experts=1)
    net.codec.fit(torch.cat([code_true, torch.zeros(1, CODE_DIM)]))
    wild = torch.full((1, CODE_DIM), 6.0)                      # far outside anything trained
    val = _occ(net, sup, sup, code_true, wild)
    assert np.isfinite(val) and val < OCC_LOGIT, val
