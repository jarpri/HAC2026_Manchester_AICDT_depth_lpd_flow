"""The flow network: the codec's pullback, the routing of times to experts, the inputs, and
the sampler."""
import torch

from hac26.conventions import cameras
from hac26.field import CODE_DIM, N_DIR, N_NODES
from hac26.solvers.lpd_flow import (N_FEAT, N_NODE_CH, N_SPHERE_CH, LPDFlow,
                                    PrimalNet, churn_step, descent_scale, flow_inputs,
                                    geometry_tags, time_embed)


def _codec():
    net = LPDFlow(n_experts=2)
    gen = torch.Generator().manual_seed(0)
    codes = torch.zeros(6, CODE_DIM)
    codes[:, :N_DIR] = 0.1 * torch.randn(6, N_DIR, generator=gen)
    codes[:, N_DIR:] = 0.05 * torch.randn(6, CODE_DIM - N_DIR, generator=gen)
    net.codec.fit(codes)
    return net


def test_codec_whitens_both_blocks_and_inverts():
    """After the fit each block of the whitened corpus has a spread of order one (the g block
    is scaled robustly through asinh, so not exactly one), and decode undoes encode."""
    net = _codec()
    gen = torch.Generator().manual_seed(1)
    codes = torch.zeros(200, CODE_DIM)
    codes[:, :N_DIR] = 0.1 * torch.randn(200, N_DIR, generator=gen) + 0.02
    codes[:, N_DIR:] = 0.05 * torch.randn(200, CODE_DIM - N_DIR, generator=gen)
    net.codec.fit(codes)
    z = net.codec.encode(codes)
    assert abs(float(z[:, :N_DIR].std()) - 1.0) < 0.15
    assert abs(float(z[:, :N_DIR].mean())) < 0.1
    assert 0.5 < float(z[:, N_DIR:].std()) < 1.5
    assert torch.allclose(net.codec.decode(z), codes, atol=1e-5)


def test_pullback_matches_autograd_through_decode():
    """A gradient with respect to the raw code, pulled back through the codec, equals the
    gradient autograd gives through decode."""
    net = _codec()
    z = torch.randn(3, CODE_DIM, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
    g_raw = torch.randn(3, CODE_DIM, generator=torch.Generator().manual_seed(2))
    (net.codec.decode(z) * g_raw).sum().backward()
    assert torch.allclose(z.grad, net.codec.pullback(z.detach(), g_raw), atol=1e-5, rtol=1e-4)


def _inputs(B, C):
    sph0 = torch.zeros(1, N_DIR, N_SPHERE_CH)
    node0 = torch.zeros(1, N_NODES, N_NODE_CH)
    grad = torch.randn(B, CODE_DIM)
    grad[1] = 0.0                                          # a body without an adjoint
    return flow_inputs(torch.randn(B, C, 40, N_FEAT), torch.ones(B, C), sph0, node0, grad)


def test_inputs_carry_a_unit_direction_and_its_size():
    inp = _inputs(3, len(cameras()))
    assert inp.adj[:, :N_DIR].pow(2).mean(1)[[0, 2]].allclose(torch.ones(2), atol=1e-5)
    assert inp.adj[1].abs().sum() == 0.0
    assert torch.allclose(inp.sphere[0, :, -1], inp.adj[0, :N_DIR])
    assert torch.allclose(inp.node[0, :, -1], inp.adj[0, N_DIR:])
    assert inp.adj_log.shape == (3, 2)


def test_each_time_goes_to_its_own_expert():
    """With two experts, times below one half use the first and the rest the second, and
    the batched velocity equals the expert's own output."""
    net = _codec().eval()
    C = len(cameras())
    B = 4
    t = torch.tensor([0.1, 0.4, 0.6, 0.95])
    code = torch.randn(B, CODE_DIM)
    rad = torch.full((B,), 1.2)
    tag = geometry_tags().expand(B, -1, -1)
    inp = _inputs(B, C)
    v = net.velocity(code, t, rad, tag, inp)
    assert net.expert_of(t).tolist() == [0, 0, 1, 1]
    summary = net.reader(inp.resid, tag, inp.mask)
    for e, sel in ((0, [0, 1]), (1, [2, 3])):
        direct = (net.prior_velocity(code[sel], t[sel], rad[sel], inp.sphere[sel], inp.node[sel])
                  + net.experts[e](code[sel], t[sel], summary[sel], time_embed(t[sel]),
                                   torch.log(rad[sel]), inp.select(sel)))
        assert torch.allclose(v[sel], direct, atol=1e-6)


def test_branching_copies_the_trained_expert_and_shares_the_reader():
    """A one-expert network branched into four gives the same velocity at every time, since
    each new expert starts as a copy; the edges split [0, 1) into quarters; and the reader is
    one module."""
    net = _codec().eval()
    net.experts = net.experts[:1]
    net.edges = net._edges(1, None)
    with torch.no_grad():                 # make the expert say something
        for p in net.experts[0].parameters():
            p.add_(0.01 * torch.randn_like(p))
    C = len(cameras())
    B = 4
    t = torch.tensor([0.1, 0.4, 0.6, 0.95])
    code = torch.randn(B, CODE_DIM)
    inp = _inputs(B, C)
    tag = geometry_tags().expand(B, -1, -1)
    before = net.velocity(code, t, torch.ones(B), tag, inp)
    net.branch(4)
    assert net.edges.tolist() == [0.25, 0.5, 0.75]
    assert net.expert_of(t).tolist() == [0, 1, 2, 3]
    assert len(net.experts) == 4 and len(list(net.reader.parameters())) > 0
    after = net.velocity(code, t, torch.ones(B), tag, inp)
    assert torch.allclose(before, after, atol=1e-5)      # float32 rounding, summed differently


def test_a_fresh_data_part_is_the_closed_form_descent_step():
    """The branches and the skip of a fresh expert are zero-initialised and its correction to
    the descent step is too, so at the start of the data part's training the velocity is the
    prior's plus the step the conditional velocity asks for (descent_scale) and nothing else.
    A body whose adjoint is zero gets no step."""
    net = _codec().eval()
    with torch.no_grad():                 # a prior that says something, unlike a fresh one
        net.prior.gain[-1].bias.fill_(0.3)
    C = len(cameras())
    B = 3
    t = torch.tensor([0.2, 0.5, 0.8])
    code = torch.randn(B, CODE_DIM)
    inp = _inputs(B, C)
    v = net.velocity(code, t, torch.ones(B), geometry_tags().expand(B, -1, -1), inp)
    prior = net.prior_velocity(code, t, torch.ones(B), inp.sphere, inp.node)
    size = descent_scale(t)[:, None] * torch.exp(inp.adj_log)
    want = PrimalNet._capped(torch.cat([size[:, :1] * inp.adj[:, :N_DIR],
                                        size[:, 1:] * inp.adj[:, N_DIR:]], -1), code)
    assert torch.allclose(v - prior, want, atol=1e-6)
    assert prior.abs().sum() > 0 and want.abs().sum() > 0
    assert float(want[1].abs().max()) < 1e-5      # the body with no adjoint gets no step


def test_sampler_runs_with_and_without_noise():
    net = _codec().eval()
    C = len(cameras())
    sph0 = torch.zeros(1, N_DIR, N_SPHERE_CH)
    node0 = torch.zeros(1, N_NODES, N_NODE_CH)
    calls = []

    def resid_fn(z, t):
        calls.append(float(t[0]))
        return flow_inputs(torch.zeros(len(z), C, 40, N_FEAT), torch.ones(len(z), C), sph0, node0,
                           torch.randn(len(z), CODE_DIM))
    tag = geometry_tags().expand(2, -1, -1)
    mask = torch.ones(2, C)
    for churn in (0.0, 0.5):
        x = net.sample(resid_fn, tag, mask, (sph0, node0), 1.0, batch=2, n_steps=4, churn=churn)
        assert x.shape == (2, CODE_DIM) and torch.isfinite(x).all()
    assert calls[:4] == [0.0, 0.25, 0.5, 0.75]


def test_last_step_adds_no_noise():
    """With churn on, every step but the last is random; the last is the plain flow step, so
    no noise is left in the answer."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(3, CODE_DIM, generator=gen)
    v = torch.randn(3, CODE_DIM, generator=gen)
    dt = 0.25
    a, b = churn_step(x, v, 0.5, dt, 0.5), churn_step(x, v, 0.5, dt, 0.5)
    assert not torch.allclose(a, b)
    last = churn_step(x, v, 0.75, dt, 0.5)
    assert torch.allclose(last, x + v * dt)
    assert torch.allclose(churn_step(x, v, 0.5, dt, 0.0), x + v * dt)


def test_time_embedding_is_smooth_and_tells_the_ends_apart():
    """Nearby times give nearby embeddings, so the network sees t as a smooth quantity, while
    t = 0 and t = 1 are distinct."""
    t = torch.linspace(0.0, 0.99, 50)
    d = (time_embed(t + 1e-3) - time_embed(t)).abs().max()
    assert float(d) < 0.5
    ends = (time_embed(torch.tensor([0.0])) - time_embed(torch.tensor([1.0]))).abs().max()
    assert float(ends) > 1.0


def test_from_state_dict_restores_the_expert_count_and_edges():
    net = LPDFlow(n_experts=3, edges=(0.5, 0.9))
    net.codec.fit(torch.randn(6, CODE_DIM) * 0.1)
    again = LPDFlow.from_state_dict(net.state_dict())
    assert len(again.experts) == 3 and torch.allclose(again.edges, torch.tensor([0.5, 0.9]))
    assert again.expert_of(torch.tensor([0.2, 0.6, 0.95])).tolist() == [0, 1, 2]


def test_the_summary_is_a_response_to_the_curves():
    """The dual carries mode embeddings, geometry tags and its own biases, so its raw output
    contains a large part no residual influences. The summary the primal reads has to be what
    the curves changed, or the data arrives as a small perturbation on a constant."""
    from hac26.solvers.lpd_flow import N_FEAT, N_MODES
    net = _codec().eval()
    C = len(cameras())
    K = 32
    tag = geometry_tags().expand(K, -1, -1)
    mask = torch.ones(K, C)
    gen = torch.Generator().manual_seed(3)
    for scale in (1.0, 0.05):
        with torch.no_grad():
            s = net.reader(scale * torch.randn(K, C, N_MODES, N_FEAT, generator=gen), tag, mask)
        common, varying = float(s.mean(0).norm()), float(s.std(0).norm())
        assert common < 2.0 * varying, (scale, common, varying)
    # no curves at all leaves nothing to summarise
    with torch.no_grad():
        empty = net.reader(torch.zeros(K, C, N_MODES, N_FEAT), tag, mask)
    assert float(empty.abs().max()) == 0.0


def test_guidance_weights_the_data_part_and_leaves_the_prior_alone():
    """The velocity is the prior's plus the data part's, which is the pair an interpolation
    between an unconditional and a conditional field is built from, so the interpolation is a
    weight on the data part alone. At weight one the velocity is the model as trained, and the
    weight never touches the prior."""
    net = _codec().eval()
    with torch.no_grad():                 # a prior and a data part that both say something
        net.prior.gain[-1].bias.fill_(0.3)
        for e in net.experts:
            for p in e.parameters():
                p.data.add_(0.05 * torch.randn_like(p))
    C, B = len(cameras()), 3
    t, code, rad = torch.tensor([0.2, 0.5, 0.8]), torch.randn(3, CODE_DIM), torch.ones(3)
    inp, tag = _inputs(B, C), geometry_tags().expand(B, -1, -1)
    prior = net.prior_velocity(code, t, rad, inp.sphere, inp.node)
    datav = net.data_velocity(code, t, rad, tag, inp)
    assert datav.abs().sum() > 0          # otherwise the weight has nothing to act on
    assert torch.allclose(net.velocity(code, t, rad, tag, inp), prior + datav)
    assert torch.allclose(net.velocity(code, t, rad, tag, inp, guidance=1.0), prior + datav)
    # the weight ramps with t: neutral where the residual is about the prior's guess rather
    # than about the body, full where the endpoint estimate is nearly the answer
    for w in (0.0, 2.0, 3.5):
        got = net.velocity(code, t, rad, tag, inp, guidance=w)
        ramp = (1.0 + (w - 1.0) * t).reshape(-1, 1)
        assert torch.allclose(got, prior + ramp * datav, atol=1e-6)
    t0 = torch.zeros(B)
    d0 = net.data_velocity(code, t0, rad, tag, inp)
    p0 = net.prior_velocity(code, t0, rad, inp.sphere, inp.node)
    for w in (1.0, 2.0, 5.0):             # at t = 0 every weight is the model as trained
        assert torch.allclose(net.velocity(code, t0, rad, tag, inp, guidance=w), p0 + d0,
                              atol=1e-6)
    t1 = torch.ones(B)
    p1 = net.prior_velocity(code, t1, rad, inp.sphere, inp.node)
    assert torch.allclose(net.velocity(code, t1, rad, tag, inp, guidance=0.0), p1, atol=1e-6)


def test_supervising_the_samplers_own_states_collapses_the_draws():
    """Why training states are taken from the straight line only. A state the sampler
    reaches is a function of the start, the churn noise and the data, all independent of the
    body given the data, so a velocity scored against (x1 - x) / (1 - t) at such states has
    the posterior mean as its optimum and carries every draw there. On a two-point body with
    noisy data the line objective keeps both points and the rollout objective keeps neither.
    The sampler is churn_step itself."""
    import torch.nn as nn
    from hac26.solvers.lpd_flow import churn_step, time_embed

    torch.manual_seed(0)
    n_steps, batch = 8, 128

    class V(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2 + time_embed(torch.zeros(1)).shape[1], 48),
                                     nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 1))

        def forward(self, x, t, d):
            return self.net(torch.cat([x, d, time_embed(t)], 1))

    def draw_data():
        x1 = torch.where(torch.rand(batch, 1) < 0.5, 1.0, -1.0)
        return x1, x1 + torch.randn(batch, 1)

    def train(rollout: bool):
        v = V()
        opt = torch.optim.Adam(v.parameters(), lr=3e-3)
        for _ in range(1200):
            x1, d = draw_data()
            x0 = torch.randn(batch, 1)
            if not rollout:
                t = torch.rand(batch, 1)
                xt = (1 - t) * x0 + t * x1
                states = [(xt, t)]
            else:
                states, x = [], x0
                with torch.no_grad():
                    for s in range(n_steps):
                        t = torch.full((batch, 1), s / n_steps)
                        states.append((x.clone(), t))
                        x = churn_step(x, v(x, t[:, 0], d), s / n_steps, 1 / n_steps, 0.5)
            loss = sum(((v(x, t[:, 0], d) - (x1 - x) / (1 - t)) ** 2).mean()
                       for x, t in states) / len(states)
            opt.zero_grad(); loss.backward(); opt.step()
        return v

    def sample(v, d_value: float, n: int = 2000):
        d = torch.full((n, 1), d_value)
        x = torch.randn(n, 1)
        with torch.no_grad():
            for s in range(n_steps):
                t = torch.full((n,), s / n_steps)
                x = churn_step(x, v(x, t, d), s / n_steps, 1 / n_steps, 0.5)
        return x[:, 0]

    line = sample(train(rollout=False), 0.0)
    rolled = sample(train(rollout=True), 0.0)
    # at d = 0 the posterior is the two points with equal weight: spread one, mean zero
    assert float(line.std()) > 0.7
    assert float(((line.abs() - 1.0).abs() < 0.35).float().mean()) > 0.7
    assert float(rolled.std()) < 0.25
    assert float(((rolled.abs() - 1.0).abs() < 0.35).float().mean()) < 0.3
