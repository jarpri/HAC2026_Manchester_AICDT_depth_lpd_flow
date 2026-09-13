"""The learned primal-dual solver, written as a conditional flow.

The curves do not determine the body. The spindle r(z) = R(1 - |z|/2) and the hourglass
r(z) = R(1/2 + |z|/2) have the same volume, the same silhouette from every equatorial
direction, the same R and the same z extent, and both are axisymmetric, so every
mean-normalised curve is identically one for both. A network trained to predict one shape
from the curves learns the average of the bodies that fit, which here is neither of them and
smoother than both. So the solver is a flow: it starts from noise and moves toward a body
that fits, and several runs give several bodies that fit, from which the answer is chosen
(scripts/reconstruct_lpd.py).

The state is the code x = (dh, a), whitened by CodeCodec. a holds signed depths on a fixed
set of directions and is the only part that can make a body non-convex. dh is a band-limited
correction to the support function h, sampled on N_DIR directions. h itself is not in the
flow: the convex stage recovers the hull well from a linear operator, and sampling all of h
would make the network rederive that and would let noise move the size of the body. But the
convex stage assumes convexity, and a non-convex body is darker than its hull because it
shadows itself, so its h is wrong in the direction of a convex answer. dh corrects that. The
corpus dh is the correction the convex stage's own reconstruction of each training body needs
(scripts/build_corpus.py), so the flow learns the errors that stage makes. Both blocks live on
a sphere -- dh on the design directions, the carve on the depth nodes -- so both have the same
network, a convolution on the sphere, at their own resolutions.

The velocity is a prior part plus a data part, because the posterior is the prior times the
likelihood: grad log p(x_t | d) = grad log p(x_t) + grad log p(d | x_t), and the velocity of
the flow is an affine function of the score. The prior part (PriorNet) is the velocity of a
flow over the corpus codes given only the published radius; it reads no data and trains
without the operator (scripts/train_prior.py), so it can train for as long as it takes. The
data part (Reader, PrimalNet) reads the residual and the adjoint and starts at zero, so at the
start of its training the flow is the prior and everything it learns is a correction toward
the data. In the terms of the learned primal-dual method, the prior part is the proximal
step and the data part is the dual step and the adjoint.

Splitting the velocity this way also makes the sampler adjustable in a way a single network
would not be. The prior part alone is the flow with no knowledge of these curves and the sum
is the flow conditioned on them, which is exactly the pair an interpolation between an
unconditional and a conditional field is built from, and the interpolation reduces here to a
weight on the data part alone (LPDFlow.velocity). At weight one the sampler integrates the
model as trained. Above one it follows the curves further from the prior, which is aimed at
this problem's own failure: the prior is the average of a corpus of bodies and is therefore
smoother than any of them, and it is the curves, not the prior, that say this body has a hole
in it. The weight is ramped from one at t = 0 to its full size at t = 1, because what the data
part reads early in t is the residual at a body the prior made out of noise, which says
something about the prior's guess and not about this rock. It is a sampling choice, not a
training one, so it is measured against held-out bodies rather than assumed
(scripts/decision_check.py).

The data the network reads are the Fourier coefficients along the rotation angle psi of the
mean-normalised curves, orders m = 1..N_MODES: of the measured curves, and of the residual
against the operator's prediction divided by each curve's noise and model error, so that a
residual of one means one standard deviation for every geometry. The state itself is part
noise, so both are taken at the body the prior's velocity says the state is heading for,
x_t + (1 - t) v_prior (LPDFlow.sample). The network also reads the adjoint of the operator
there applied to the whitened residual, the direction in code space along which the
predicted curves move toward the data: each block of it as a field on that block's own
sphere. The same direction enters the velocity directly
with a learned step size per block (PrimalNet), so that a step of gradient descent on the
misfit is available to the network as two numbers.

For a convex body the forward map is block-diagonal in m: rotating the body by psi
multiplies the harmonic Y_lm by e^{-i m psi}, so Fourier order m of a curve depends only on
the shape's harmonics of order m. The dual network therefore never mixes orders; it attends
across geometries at fixed m, which also makes it indifferent to missing geometries. Orders
meet only in the primal's conditioning, which reads all of them at once.

Early in t the state is mostly noise and the velocity has to come from the curves; late in
t the state is nearly the body and the velocity is a clean-up. These are different jobs, so
t is cut into N_EXPERTS intervals with an expert (PrimalNet) each, while the reader is one
network shared by all of them: what a residual means does not depend on t, only the response
does. One data part is trained on all of t first and then copied into the experts, each of
which continues on its own interval (LPDFlow.branch), so every expert starts from everything
that was learned. Training draws t at random and trains the expert that owns it, so the cost
of a training step does not depend on the number of experts or of sampling steps.

The training interpolant is the straight line x_t = (1 - t) x0 + t x1; it defines what a
state at time t is, the body with a known share of noise mixed in, and gives the training
target of lowest variance. Any velocity trained on it is also a denoiser, x1_hat = x_t +
(1 - t) v, and a score. The sampler integrates the velocity in N_STEPS steps with the
operator at each, and adds noise on the way (CHURN): the flow and the stochastic equation
dx = (v + eps s) dt + sqrt(2 eps) dW, with s the score of the flow's marginal, have the same
distribution at every t, so the noise changes nothing about what the draws represent and
only decorrelates them. The noise level of a step is the one at the step's end, so the last
step is a plain flow step (churn_step).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from hac26.conventions import cameras
from hac26.field import CODE_DIM, N_DIR, N_NODES, dir_design, node_design

__all__ = ["CODE_DIM", "N_DIR", "N_NODES", "N_MODES", "N_STEPS", "N_EXPERTS", "CHURN",
           "OP_GAIN", "DESCENT_CAP", "T_DIM",
           "T_FREQ", "N_FEAT", "N_SPHERE_CH", "N_NODE_CH", "FlowInputs", "flow_inputs",
           "DualSetTransformer", "SphereBranch", "PrimalNet", "PriorNet",
           "Reader", "CodeCodec", "LPDFlow", "fourier_embed", "time_embed", "GUIDANCE",
           "churn_step", "descent_scale", "geometry_tags"]

N_MODES = 40
N_STEPS = 16        # sampling steps by default; the operator runs at each. Training does not
                    # depend on it.
N_EXPERTS = 4       # velocity networks, one per equal interval of t
CHURN = 0.5         # noise in the sampler, eps(t) = CHURN (1 - t); 0 is the plain flow
DESCENT_CAP = 0.5   # largest descent step per block, as a share of that block of the state
OP_GAIN = 300.0     # fallback for tr(A' Sigma^-1 A) / CODE_DIM: how much whitened misfit a
                    # unit whitened change of the code makes, for the operator at the run's
                    # geometries and phases. It sets descent_scale and nothing else, and that
                    # profile is flat enough in it that the order of magnitude is what matters.
                    # It depends on the code's dimension, on the whitening, and therefore on the
                    # corpus, so a trained model carries its own measured value in a buffer
                    # (PrimalNet.op_gain) and this constant is only what an untrained one starts
                    # from. scripts/train_lpd.py measures it from the corpus it is about to
                    # train on, by rendering pairs of bodies and taking the ratio of the
                    # whitened squared curve difference to the whitened squared code
                    # difference.
GUIDANCE = 1.0      # weight on the data part of the velocity at t = 1, ramped from one at
                    # t = 0. One is the model as trained; above one the draws follow the
                    # curves further from the prior. See LPDFlow.velocity, and
                    # scripts/decision_check.py, which measures the score against held-out
                    # bodies over a range of weights.
T_DIM = 32          # width of the TIME embedding, which is not the mode embedding
T_FREQ = (0.5, 64.0)  # slowest and fastest time feature, in cycles across [0, 1]
N_FEAT = 8          # per (geometry, mode) input of the dual: real and imaginary parts of the
                    # whitened residual and of the data, for the intensity and the count
N_SPHERE_CH = 5     # dh-branch channels besides dh_t: h on the directions, the direction
                    # (3), the adjoint's dh part
N_NODE_CH = 5       # depth-branch channels besides a_t: how far out the core's surface is
                    # along that node's ray, which is how much room there is to carve there,
                    # the direction (3), and the adjoint's depth part. There is no inside
                    # indicator and no coordinate triple beyond the direction, because in a
                    # field indexed by direction every coordinate is on the surface -- which is
                    # the whole reason the lattice's two wasted channels are gone.
G_LIMIT = 4.0       # the decode saturates at this multiple of the corpus's largest fitted
                    # amplitude: outside anything the flow should emit, well inside where
                    # float32 sinh overflows.


def fourier_embed(m: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """Embedding of the rotation order m. The only way m enters the dual network."""
    k = torch.arange(dim // 2, device=m.device, dtype=torch.float32)
    a = m[..., None].float() / (10.0 ** (2 * k / dim))
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


def geometry_tags() -> torch.Tensor:
    """(1, C, 4) per-geometry inputs of the dual: cos and sin of the azimuth, sin of the
    elevation, and a constant one."""
    tag = torch.zeros(1, len(cameras()), 4)
    for i, cam in enumerate(cameras()):
        tag[0, i] = torch.tensor([np.cos(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.azimuth_deg)),
                                  np.sin(np.radians(cam.elevation_deg)), 1.0])
    return tag


class FlowInputs(NamedTuple):
    """What the network reads besides the state and the time: the dual's input, the
    geometry mask, the two branches' channels with the adjoint slots filled, the adjoint
    direction (unit root mean square per block, in whitened units) and the log of its size
    per block. flow_inputs builds one."""
    resid: torch.Tensor        # (B, C, N_MODES, N_FEAT)
    mask: torch.Tensor         # (B, C)
    sphere: torch.Tensor       # (B, N_DIR, N_SPHERE_CH)
    node: torch.Tensor         # (B, N_NODES, N_NODE_CH)
    adj: torch.Tensor          # (B, CODE_DIM)
    adj_log: torch.Tensor      # (B, 2)

    def select(self, sel) -> "FlowInputs":
        return FlowInputs(*(f[sel] for f in self))


def flow_inputs(resid, mask, sphere0, node0, grad_z) -> FlowInputs:
    """The inputs from the dual's features (B, C, N_MODES, N_FEAT), the geometry mask (B, C),
    the branch channels with empty adjoint slots, and the adjoint of the whitened misfit in
    whitened units (B, CODE_DIM); zeros for a body without curves. Each block of the adjoint
    is divided by its root mean square, which goes into adj_log: the direction is the
    information, its size is a separate number."""
    B = grad_z.shape[0]
    blocks = [grad_z[:, :N_DIR], grad_z[:, N_DIR:]]
    rms = torch.stack([b.pow(2).mean(1).sqrt() for b in blocks], 1)               # (B, 2)
    unit = torch.cat([b / r[:, None].clamp_min(1e-12) for b, r in zip(blocks, rms.T)], 1)
    sphere = sphere0.expand(B, -1, -1).clone()
    node = node0.expand(B, -1, -1).clone()
    sphere[..., -1] = unit[:, :N_DIR]
    node[..., -1] = unit[:, N_DIR:]
    return FlowInputs(resid, mask, sphere, node, unit, torch.log(rms.clamp_min(1e-6)))


def time_embed(t: torch.Tensor, dim: int = T_DIM) -> torch.Tensor:
    """Embedding of t in [0, 1]: sines and cosines at dim/2 frequencies spaced geometrically
    from T_FREQ[0] to T_FREQ[1] cycles across the interval. Separate from fourier_embed, whose
    frequencies are chosen for integer m. The slowest feature tells t = 0 from t = 1; the
    fastest resolves a few hundredths of t, finer than any step the sampler takes. Faster
    features would differ between two nearly equal times and be noise to the network.
    """
    n = dim // 2
    k = torch.arange(n, device=t.device, dtype=torch.float32) / max(n - 1, 1)
    f = T_FREQ[0] * (T_FREQ[1] / T_FREQ[0]) ** k
    a = t[..., None].float() * (2.0 * np.pi) * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


def descent_scale(t: torch.Tensor, gain: float = OP_GAIN) -> torch.Tensor:
    """The multiple of the operator's adjoint that the conditional velocity asks for at time t.

    With the interpolant x_t = (1-t) x0 + t x1 and a whitened prior, the exact conditional
    velocity is v_prior + ((1-t)/D) A' (Sigma + v_t A A')^-1 r, where D = (1-t)^2 + t^2,
    v_t = (1-t)^2 / D is the prior's conditional variance of the endpoint, and r is the
    residual at the endpoint estimate. The operator returns A' Sigma^-1 r, so the two differ by
    the whitening of the residual by the endpoint's own spread as well as by the noise. Taking
    A A' for a multiple of the identity, that difference collapses to

        (1 - t) / [ (1 - t)^2 + t^2 + gain (1 - t)^2 ],

    which is what this returns. It rises from 1/(1+gain) at t = 0 to 1/(2(sqrt(2+gain) - 1)) at
    t = 1 - 1/sqrt(2+gain) and falls to zero at t = 1. The data part is a correction on top of
    it (PrimalNet), so an error in `gain` costs the correction some of its range and nothing
    else."""
    u = 1.0 - t
    return u / (u * u * (1.0 + gain) + t * t)


def churn_step(x: torch.Tensor, v: torch.Tensor, t: float, dt: float, churn: float) -> torch.Tensor:
    """One step of the sampler from time t with velocity v. With churn 0 it is the flow step
    x + v dt. Otherwise it is a step of the stochastic equation of the module docstring,
    dx = (v + eps s) dt + sqrt(2 eps) dW, with s = -(x - t v) / (1 - t) the score of the
    flow's marginal at t and eps = churn (1 - t - dt), the noise level at the end of the step:
    the last step, which nothing follows, is then the plain flow step x + v dt, which lands
    exactly on the endpoint the velocity implies."""
    if churn <= 0:
        return x + v * dt
    eps = churn * max(0.0, 1.0 - t - dt)
    s = -(x - t * v) / (1.0 - t)
    return x + (v + eps * s) * dt + np.sqrt(2.0 * eps * dt) * torch.randn_like(x)


class DualSetTransformer(nn.Module):
    """Attention across GEOMETRIES at fixed m, weights shared across m.

    Input  (B, C, M, F) with C geometries and M rotation orders, plus a geometry mask.
    Output (B, C, M, width). No operation mixes different m.
    """

    def __init__(self, in_feat: int = N_FEAT, width: int = 96, heads: int = 4,
                 blocks: int = 3, m_dim: int = 16):
        super().__init__()
        self.inp = nn.Linear(in_feat + m_dim + 4, width)
        self.att = nn.ModuleList([nn.MultiheadAttention(width, heads, batch_first=True)
                                  for _ in range(blocks)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                                nn.SiLU(), nn.Linear(width, width))
                                  for _ in range(blocks)])
        self.m_dim = m_dim

    def forward(self, feats, geom_tag, mask, modes):
        B, C, M, _ = feats.shape
        me = fourier_embed(modes, self.m_dim)[None, None].expand(B, C, M, self.m_dim)
        x = self.inp(torch.cat([feats, me, geom_tag[:, :, None, :].expand(B, C, M, 4)], -1))
        kpm = (mask < 0.5)                                   # True where a geometry is absent
        for att, mlp in zip(self.att, self.mlp):
            y = x.permute(0, 2, 1, 3).reshape(B * M, C, -1)  # attend over C, at fixed m
            km = kpm[:, None, :].expand(B, M, C).reshape(B * M, C)
            km = torch.where(km.all(-1, keepdim=True), torch.zeros_like(km), km)
            # need_weights=False routes to scaled_dot_product_attention instead of
            # materialising the (B*M*heads, C, C) attention matrix and its head-mean, neither
            # of which is read. Same output, and flash attention on CUDA.
            y, _ = att(y, y, y, key_padding_mask=km, need_weights=False)
            x = x + y.reshape(B, M, C, -1).permute(0, 2, 1, 3)
            x = x + mlp(x)
        return x


N_OPS = 5           # fixed operators a convolution on the sphere is built from


def _sphere_operators(dirs: np.ndarray, k: int = 8) -> torch.Tensor:
    """The five fixed operators of a convolution on the sphere, stacked into one sparse
    (N_OPS * N, N) matrix: the identity, the mean over the k nearest directions, two tangential
    first moments and one second moment.

    A sphere has no global grid, so a convolution is built from these fixed operators instead
    of a stencil. Together they carry the same information a 3x3 stencil carries on a plane.

    Sparse, because the branches run on direction sets of very different sizes: each operator
    has k + 1 non-zeros per row wherever it has any, so at the depth nodes the dense form would
    be a hundred and thirty megabytes of almost entirely zeros and every product would visit all
    of it. Stacked into one matrix so that applying all five is one product.
    """
    n = len(dirs)
    g = dirs @ dirs.T
    np.fill_diagonal(g, -2.0)
    nb = np.argsort(-g, axis=1)[:, :k]                      # k nearest by cosine
    # a right-handed tangent frame at each direction, seeded from the least-aligned axis
    seed = np.zeros_like(dirs)
    seed[np.arange(n), np.argmin(np.abs(dirs), axis=1)] = 1.0
    e1 = seed - (seed * dirs).sum(1, keepdims=True) * dirs
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(dirs, e1)
    off = dirs[nb] - dirs[:, None, :]                       # (n, k, 3)
    d = np.linalg.norm(off, axis=2) + 1e-12
    # column 0 of each row is the direction itself, so the identity needs no separate entry
    idx = np.concatenate([np.arange(n)[:, None], nb], axis=1)               # (n, k+1)
    w = np.zeros((N_OPS, n, k + 1), dtype=np.float32)
    w[0, :, 0] = 1.0
    w[1, :, 1:] = 1.0 / k
    w[2, :, 1:] = (off * e1[:, None, :]).sum(2) / d / k
    w[3, :, 1:] = (off * e2[:, None, :]).sum(2) / d / k
    w[4, :, 1:] = d ** 2 / (d ** 2).mean() / k
    rows = (np.arange(N_OPS)[:, None, None] * n
            + np.arange(n)[None, :, None]).repeat(k + 1, axis=2).ravel()
    cols = np.broadcast_to(idx[None], (N_OPS, n, k + 1)).ravel()
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])).long(),
        torch.from_numpy(w.reshape(-1)), (N_OPS * n, n),
        check_invariants=False).coalesce()


class SphereConv(nn.Module):
    """y_o = sum_{s,c} W[s, c, o] (S_s x)_c + b. Weights shared across directions.

    The operators are a buffer and not a parameter, and they are not persistent: they follow
    from the direction set alone, which is a constant of the representation, so a checkpoint
    must not carry a copy of them.
    """

    def __init__(self, ops: torch.Tensor, c_in: int, c_out: int):
        super().__init__()
        self.register_buffer("S", ops, persistent=False)
        self.n_dirs = ops.shape[1]
        self.w = nn.Parameter(torch.randn(N_OPS, c_in, c_out) / np.sqrt(N_OPS * c_in))
        self.b = nn.Parameter(torch.zeros(c_out))

    def forward(self, x):                                   # x: (B, N, C_in)
        b, n, c = x.shape
        y = torch.sparse.mm(self.S, x.transpose(0, 1).reshape(n, b * c))
        return torch.einsum("snbc,sco->bno", y.reshape(N_OPS, n, b, c), self.w) + self.b


class _FiLM(nn.Module):
    """Per-channel scale and shift computed from the conditioning vector.

    Zero-initialised, so at the start it passes its input straight through and the
    conditioning has no effect until the weights move.
    """

    def __init__(self, cond_dim: int, width: int):
        super().__init__()
        self.f = nn.Linear(cond_dim, 2 * width)
        nn.init.zeros_(self.f.weight); nn.init.zeros_(self.f.bias)
        self.width = width

    def forward(self, h, cond, spatial_dims: int):
        a, b = self.f(cond).chunk(2, -1)
        shape = (h.shape[0], self.width) + (1,) * spatial_dims
        if spatial_dims == 0:                               # (B, N, C) layout
            return h * (1 + a[:, None, :]) + b[:, None, :]
        return h * (1 + a.reshape(shape)) + b.reshape(shape)


class SphereBranch(nn.Module):
    """Velocity for one block of the code: a convolution on the sphere over the directions that
    block is carried on.

    Both blocks of the code live on a sphere, so this is the only branch there is. For dh the
    directions are `dir_design(N_DIR)` and the channels are, in order: dh_t, the base support
    resampled onto the directions, the three components of the direction, and the dh part of the
    operator's adjoint applied to the whitened residual, scaled per sample. For the depth field
    the directions are the nodes and the second channel is how far out the core's surface lies
    along that node's ray. train_lpd.cond_channels builds the middle four of each and
    flow_inputs fills the last; the prior part leaves the last one out.

    The adjoint channel is an input, never added to the velocity: it says in which direction
    the misfit falls fastest, and the network decides how far to go.
    """

    def __init__(self, cond_dim: int, dirs: np.ndarray, width: int = 128, blocks: int = 4,
                 in_ch: int = 1 + N_SPHERE_CH):
        super().__init__()
        ops = _sphere_operators(np.asarray(dirs, dtype=float))
        self.n_dirs = len(dirs)
        self.inp = SphereConv(ops, in_ch, width)
        self.conv = nn.ModuleList([nn.ModuleList([SphereConv(ops, width, width),
                                                  SphereConv(ops, width, width)])
                                   for _ in range(blocks)])
        self.film = nn.ModuleList([_FiLM(cond_dim, width) for _ in range(blocks)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(blocks)])
        self.head = SphereConv(ops, width, 1)
        nn.init.zeros_(self.head.w); nn.init.zeros_(self.head.b)
        self.act = nn.SiLU()

    def forward(self, x, cond):                             # x: (B, n_dirs, in_ch)
        h = self.inp(x)
        for (c1, c2), film, norm in zip(self.conv, self.film, self.norm):
            y = c2(self.act(c1(norm(h))))
            h = h + film(y, cond, spatial_dims=0)
        return self.head(h)[..., 0]                         # (B, n_dirs)


class CodeCodec(nn.Module):
    """Maps between raw code units and the whitened space the flow works in.

    Whitening: x0 is drawn from N(0, I) and the target is x1 - x0, so both endpoints have to
    live on the same scale. Raw dh and g are much narrower than a unit Gaussian.

    asinh on the depths: they are heavy-tailed, and the tail is the deep carves. Clipping it,
    or modelling it as Gaussian, would push the flow toward convex answers.

    One scale and one offset per block, not per coordinate: the depth branch is a convolution on
    the sphere that reads every node with the same weights, and a per-node transform would undo
    that. The scale is a median absolute deviation rather than a standard deviation so the heavy
    tail does not set it.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("g_s", torch.ones(1))
        self.register_buffer("mu", torch.zeros(2))     # [dh, a], one scalar each
        self.register_buffer("sd", torch.ones(2))
        self.register_buffer("u_lim", torch.full((1,), 80.0))

    @torch.no_grad()
    def fit(self, codes: torch.Tensor) -> None:
        """`codes` are the raw corpus codes: dh the correction from each body's convex start
        to its true hull, g its fitted amplitudes."""
        dh, g = self._split(codes)
        self.g_s.fill_(float(g.abs().median().clamp_min(1e-12)))
        u = torch.asinh(g / self.g_s)
        for k, block in enumerate((dh, u)):
            med = block.median()
            mad = (block - med).abs().median().clamp_min(1e-8) * 1.4826   # sigma of a Gaussian
            self.mu[k] = float(med)
            self.sd[k] = float(mad)
        # where decode saturates, as a bound on the amplitude converted through asinh
        g_lim = float(g.abs().max()) * G_LIMIT
        self.u_lim.fill_(float(np.arcsinh(g_lim / float(self.g_s))))

    def _split(self, x):
        return x[..., :N_DIR], x[..., N_DIR:]

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        dh, g = self._split(raw)
        u = torch.asinh(g / self.g_s)
        return torch.cat([(dh - self.mu[0]) / self.sd[0],
                          (u - self.mu[1]) / self.sd[1]], -1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z -> raw code, saturating.

        sinh is unbounded and float32 sinh overflows, so a large excursion in z would make g
        infinite and the field non-finite. The clamp on u caps the amplitude at G_LIMIT times
        the corpus's largest fitted amplitude: it never binds on anything the flow has been
        taught to produce, and it stops a single spike from breaking every later stage. It is
        applied per coordinate, so an in-range site is never touched by an out-of-range
        neighbour.
        """
        zd, zg = self._split(z)
        u = (zg * self.sd[1] + self.mu[1]).clamp(-self.u_lim, self.u_lim)
        return torch.cat([zd * self.sd[0] + self.mu[0], self.g_s * torch.sinh(u)], -1)

    def pullback(self, z: torch.Tensor, grad_raw: torch.Tensor) -> torch.Tensor:
        """A gradient with respect to the raw code at decode(z), taken back to a gradient with
        respect to z: the chain rule through decode, zero where the clamp binds."""
        zd, zg = self._split(z)
        gd, gg = self._split(grad_raw)
        u = zg * self.sd[1] + self.mu[1]
        inside = (u.abs() < self.u_lim).to(gg.dtype)
        return torch.cat([gd * self.sd[0],
                          gg * self.g_s * torch.cosh(u.clamp(-self.u_lim, self.u_lim))
                          * self.sd[1] * inside], -1)


class PrimalNet(nn.Module):
    """The two branches, the shared conditioning, a time-dependent skip gain, and a learned
    step along the adjoint.

    The conditioning vector is built from the dual's summary, the time embedding and the log
    of the body's published xy radius. The radius is needed because the code describes the
    body in the canonical frame while the curves are those of the body at its physical
    radius, so the same curves mean a different canonical body at a different radius.

    The skip path exists because the velocity target x1 - x0 equals (x1 - x_t)/(1 - t), so
    its dominant term is the input scaled by a function of t. The gain is per block, since dh
    and g have different scales, and zero-initialised, so training starts from the branches
    alone.

    The adjoint step adds, per block, a multiple of the adjoint of the whitened data misfit: a
    step of gradient descent on it. The multiple is the one the conditional velocity asks for
    (descent_scale) times one plus a learned correction that depends on t and on the size of
    the gradient. The correction is zero-initialised, so an untrained data part is the descent
    step at the size the closed form prescribes rather than nothing at all, and training
    spends itself on the part of the correction that no scale can supply; a correction of
    minus one cancels the term where it is wrong. The branches still read the direction as an
    input and can shape it.

    The contribution is capped per block at DESCENT_CAP times the size of the state, so that a
    mis-set OP_GAIN cannot drive the first steps to a body the operator cannot render.
    """

    def __init__(self, summary_dim: int, cond_width: int = 256):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(summary_dim + T_DIM + 1, cond_width), nn.SiLU(),
                                  nn.Linear(cond_width, cond_width))
        self.sphere = SphereBranch(cond_width, dir_design(N_DIR))
        self.node = SphereBranch(cond_width, node_design(N_NODES), width=64,
                                 in_ch=1 + N_NODE_CH)
        self.register_buffer("op_gain", torch.tensor(float(OP_GAIN)))
        self.gain = nn.Sequential(nn.Linear(T_DIM, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.gain[-1].weight); nn.init.zeros_(self.gain[-1].bias)
        self.step = nn.Sequential(nn.Linear(T_DIM + 2, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.step[-1].weight); nn.init.zeros_(self.step[-1].bias)

    def forward(self, code, t, summary, t_embed, log_radius, inp: FlowInputs):
        c = self.cond(torch.cat([summary, t_embed, log_radius[:, None]], -1))
        v_dh = self.sphere(torch.cat([code[:, None, :N_DIR].transpose(1, 2), inp.sphere], -1), c)
        v_g = self.node(torch.cat([code[:, N_DIR:, None], inp.node], -1), c)
        gain = self.gain(t_embed)
        # inp.adj is the unit direction per block and inp.adj_log the log of the size taken
        # out of it, so the two together are the adjoint itself
        step = descent_scale(t, float(self.op_gain))[:, None] * torch.exp(inp.adj_log) \
            * (1.0 + self.step(torch.cat([t_embed, inp.adj_log], -1)))
        skip = torch.cat([gain[:, :1] * code[:, :N_DIR], gain[:, 1:] * code[:, N_DIR:]], -1)
        descent = torch.cat([step[:, :1] * inp.adj[:, :N_DIR], step[:, 1:] * inp.adj[:, N_DIR:]], -1)
        descent = self._capped(descent, code)
        return skip + descent + torch.cat([v_dh, v_g], -1)

    @staticmethod
    def _capped(descent, code):
        """Each block of the descent step held to DESCENT_CAP times the size of that block of
        the state."""
        out = []
        for sl in (slice(None, N_DIR), slice(N_DIR, None)):
            d, z = descent[:, sl], code[:, sl]
            lim = DESCENT_CAP * z.norm(dim=1, keepdim=True).clamp_min(1e-6)
            n = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
            out.append(d * torch.clamp(lim / n, max=1.0))
        return torch.cat(out, -1)


class PriorNet(nn.Module):
    """The velocity of the unconditional flow over codes: the two branches on the body's own
    channels (the adjoint slot left out), conditioned on time and on the body's published
    radius, with the same time-dependent skip gain as PrimalNet. It reads no data. It does
    read the radius: the code describes the body in the canonical frame, where the width is
    one and the height two, so a body mounted along its long axis and one mounted across it
    give different canonical shapes, and the radius is what tells them apart."""

    def __init__(self, cond_width: int = 256):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(T_DIM + 1, cond_width), nn.SiLU(),
                                  nn.Linear(cond_width, cond_width))
        self.sphere = SphereBranch(cond_width, dir_design(N_DIR), in_ch=N_SPHERE_CH)
        self.node = SphereBranch(cond_width, node_design(N_NODES), width=64,
                                 in_ch=N_NODE_CH)
        self.gain = nn.Sequential(nn.Linear(T_DIM, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.gain[-1].weight); nn.init.zeros_(self.gain[-1].bias)

    def forward(self, code, t, radius, sphere_ch, node_ch):
        """code (B, CODE_DIM), t (B,), radius (B,), and the branch channels with or without
        their adjoint slot, which is dropped here."""
        te = time_embed(t)
        c = self.cond(torch.cat([te, torch.log(radius)[:, None]], -1))
        sph = sphere_ch[..., :N_SPHERE_CH - 1]
        nod = node_ch[..., :N_NODE_CH - 1]
        v_dh = self.sphere(torch.cat([code[:, None, :N_DIR].transpose(1, 2), sph], -1), c)
        v_g = self.node(torch.cat([code[:, N_DIR:, None], nod], -1), c)
        gain = self.gain(te)
        skip = torch.cat([gain[:, :1] * code[:, :N_DIR], gain[:, 1:] * code[:, N_DIR:]], -1)
        return skip + torch.cat([v_dh, v_g], -1)


class Reader(nn.Module):
    """The part of the data network that interprets the curves: the dual, and the pooling of
    its output into one summary vector per body. What a residual means does not depend on t,
    only the response to it does, so the reader is one network shared by all experts, and it
    trains on every sample at every time.

    The summary is what the curves CHANGED, not what the network emits when they are present.
    The dual carries a mode embedding, a geometry tag and its own biases, so its output on any
    input already contains a part that no residual influences, and that part is much the
    larger of the two: the primal would have to read the data as a small perturbation on a
    constant, and the smaller the residual the worse the ratio, which is exactly the late part
    of the flow where the fine shape is settled. Subtracting the network's own response to
    zero features leaves the response to the data. The second pass costs a fraction of one
    operator call.
    """

    def __init__(self, width: int = 96, n_modes: int = N_MODES, mode_feat: int = 16):
        super().__init__()
        self.dual = DualSetTransformer(width=width)
        # Pool over geometries only, then project each mode with one shared Linear, so the
        # summary keeps which mode carried the signal. Mixing across modes happens later, in
        # the primal's conditioning. No bias: a bias is a constant the data cannot move.
        self.mode_proj = nn.Linear(width, mode_feat, bias=False)
        self.register_buffer("modes", torch.arange(1, n_modes + 1), persistent=False)
        self.summary_dim = n_modes * mode_feat

    def forward(self, resid, geom_tag, mask):
        d = (self.dual(resid, geom_tag, mask, self.modes)
             - self.dual(torch.zeros_like(resid), geom_tag, mask, self.modes))
        w = mask[:, :, None, None]
        pooled = (d * w).sum(1) / w.sum(1).clamp_min(1e-6)       # (B, M, width) -- true mean
        # Mode slots the caller zero-filled (orders above what the phase count supports) carry
        # no data. They are detected from the input rather than from a constructor argument,
        # so a checkpoint cannot disagree with a run.
        live = (resid.abs().sum((1, 3)) > 0).float()[..., None]  # (B, M, 1)
        return (self.mode_proj(pooled) * live).reshape(resid.shape[0], -1)


class LPDFlow(nn.Module):
    """The velocity field of the conditional flow: the prior part plus the data part, which is
    one reader shared by the experts and one expert (a PrimalNet) per interval of t; and the
    sampler that integrates it.

    `edges` are the interior boundaries of the experts' intervals, equal by default. They are
    a buffer, so a checkpoint carries its own split.
    """

    def __init__(self, width: int = 96, n_modes: int = N_MODES, n_experts: int = N_EXPERTS,
                 mode_feat: int = 16, edges=None):
        super().__init__()
        self.prior = PriorNet()
        self.reader = Reader(width, n_modes, mode_feat)
        self.experts = nn.ModuleList([PrimalNet(self.reader.summary_dim)
                                      for _ in range(n_experts)])
        self.register_buffer("edges", self._edges(n_experts, edges))
        self.codec = CodeCodec()
        self.n_modes = n_modes

    def set_op_gain(self, gain: float) -> None:
        """Set the operator's gain on every expert, in whitened code units.

        It is a property of the operator, the geometries and the whitening together, so it
        belongs to a trained model and not to the module: the whitening is fitted to a corpus
        and the code's dimension is a constant of the representation, and both of them move it.
        scripts/train_lpd.py measures it on the corpus it is about to train on and calls this,
        so a checkpoint carries the value its own training used and nothing has to remember a
        number written down somewhere else."""
        for e in self.experts:
            e.op_gain.fill_(float(gain))

    @classmethod
    def from_state_dict(cls, state: dict, **kw) -> "LPDFlow":
        """A network of the shape a saved state dict describes, loaded with it: the expert
        count is read off the keys and the edges come with the buffer."""
        n = len({k.split(".")[1] for k in state if k.startswith("experts.")})
        net = cls(n_experts=max(n, 1), **kw)
        net.load_state_dict(state)
        return net

    @staticmethod
    def _edges(n: int, edges) -> torch.Tensor:
        e = torch.arange(1, n, dtype=torch.float32) / n if edges is None \
            else torch.as_tensor(edges, dtype=torch.float32)
        if len(e) != n - 1 or (len(e) and not (0 < e.min() and e.max() < 1 and (e.diff() > 0).all())):
            raise ValueError(f"{n} experts need {n - 1} increasing edges inside (0, 1), got {e.tolist()}")
        return e

    def expert_of(self, t: torch.Tensor) -> torch.Tensor:
        """Index of the expert that owns each time in t (B,): expert k owns
        [edges[k-1], edges[k])."""
        return torch.bucketize(t, self.edges, right=True)

    def branch(self, n: int, edges=None) -> None:
        """Split the data part into n experts: each new interval starts as a copy of the
        expert that owns its midpoint now, so every expert begins with everything that was
        learned and only specialises from there (scripts/train_lpd.py --experts). The reader
        stays shared."""
        import copy
        new_edges = self._edges(n, edges).to(self.edges.device)
        ends = torch.cat([torch.zeros(1, device=new_edges.device), new_edges,
                          torch.ones(1, device=new_edges.device)])
        mid = 0.5 * (ends[:-1] + ends[1:])
        owners = self.expert_of(mid).tolist()
        self.experts = nn.ModuleList([copy.deepcopy(self.experts[k]) for k in owners])
        self.edges = new_edges

    def prior_velocity(self, code, t, radius, sphere_ch, node_ch):
        """The prior part alone: the flow over codes without data."""
        return self.prior(code, t, radius, sphere_ch, node_ch)

    def data_velocity(self, code, t, radius, geom_tag, inp: FlowInputs):
        """The data part alone: the shared reader's summary, then the expert that owns each
        sample's t."""
        summary = self.reader(inp.resid, geom_tag, inp.mask)
        te = time_embed(t)
        which = self.expert_of(t)
        out = torch.zeros_like(code)
        for e in which.unique().tolist():
            sel = which == e
            out[sel] = self.experts[e](code[sel], t[sel], summary[sel], te[sel],
                                       torch.log(radius[sel]), inp.select(sel))
        return out

    def velocity(self, code, t, radius, geom_tag, inp: FlowInputs, guidance: float = 1.0):
        """The velocity at `code` (B, CODE_DIM) and time t (B,), for bodies of published
        radius (B,), from the geometry tags (B, C, 4) and the inputs: prior part plus
        `guidance` times the data part.

        The two parts are the two halves of a guided velocity. The prior is trained on the
        corpus alone and frozen, so it is the flow that knows what a body looks like and
        nothing about these curves; adding the data part gives the flow conditioned on them.
        Writing v_prior + w (v_prior + v_data - v_prior) for the usual interpolation between
        an unconditional and a conditional field leaves v_prior + w v_data, so the weight
        multiplies the data part and nothing else, and w = 1 is the model as trained.

        Above one the sampler follows the curves further from the prior than the model's own
        conditional does. That is worth having here because the failure this method has to
        avoid is a draw that falls back on the prior's typical body, which is smooth: the
        corpus is the average of many bodies and the curves are what say this one has a hole
        in it. It is not free, since the field being integrated is no longer the one whose
        marginals the training matched, and far enough above one the draws leave the corpus
        altogether. scripts/decision_check.py measures the score against held-out bodies over
        a range of weights, so the value to use is read off that rather than assumed.

        The weight is applied as 1 + (guidance - 1) t rather than as a constant, so it is
        neutral at t = 0 and full at t = 1. What the data part reads is the residual of the
        curves at the body the state is heading for, and early in t that body is what the
        prior made of a draw of noise: the residual there is a statement about the prior's
        guess rather than about this rock, and amplifying the response to it amplifies
        nothing useful. Late in t the endpoint estimate is nearly the answer and the residual
        is about the body being reconstructed. The ramp is also the conservative choice, since
        it applies less guidance in total than a constant weight of the same size, and at
        guidance one it is the model as trained at every t.
        """
        v = self.prior_velocity(code, t, radius, inp.sphere, inp.node)
        d = self.data_velocity(code, t, radius, geom_tag, inp)
        if guidance == 1.0:
            return v + d
        return v + (1.0 + (guidance - 1.0) * t.reshape(-1, 1)) * d

    @torch.no_grad()
    def sample(self, resid_fn, geom_tag, mask, cond, radius: float, batch: int = 1,
               n_steps: int = N_STEPS, churn: float = CHURN, device="cpu",
               guidance: float = GUIDANCE):
        """x0 ~ N(0, I), then n_steps steps from t = 0 to 1, in the whitened space
        throughout, for a body of published radius `radius`.

        `resid_fn(code, t)` returns the FlowInputs at that code and applies the operator; it
        receives a whitened code and decodes it itself, and the mask it returns is `mask` with
        the geometries of a draw whose body has no curves switched off, as
        train_lpd.flow_loss does. `cond` is the pair of channel tensors for this body with the
        adjoint slots still empty; h is fixed at reconstruction, so it is constant.

        The operator is applied at the endpoint the prior's velocity implies,
        x1_hat = x + (1 - t) v_prior(x), not at x, as in train_lpd.flow_loss: at the first
        step x is the Gaussian draw, which decodes to a body unlike anything in the corpus,
        and a residual taken there says little about the body being reconstructed. The prior
        is a trained, frozen denoiser, so its estimate is a stable place to look; it costs one
        network forward and no operator call.

        With churn > 0 each step is churn_step's step of the stochastic equation; the last
        step adds no noise. `guidance` weights the data part of the velocity; see velocity.
        """
        sph0, node0 = cond
        rad = torch.full((batch,), float(radius), device=device)
        x = torch.randn(batch, CODE_DIM, device=device)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((batch,), k * dt, device=device)
            x1_hat = x + (1 - t[:, None]) * self.prior_velocity(
                x, t, rad, sph0.expand(batch, -1, -1), node0.expand(batch, -1, -1))
            v = self.velocity(x, t, rad, geom_tag, resid_fn(x1_hat, t), guidance)
            x = churn_step(x, v, k * dt, dt, churn)
        return x
