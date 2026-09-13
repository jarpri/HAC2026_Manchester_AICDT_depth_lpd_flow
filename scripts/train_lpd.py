#!/usr/bin/env python3
"""Train the data part of the flow (hac26.solvers.lpd_flow).

The corpus comes from scripts/build_corpus.py: for each body, the exact forward model's
curves at its radius, the support the convex stage reconstructs from those curves, which is
where the flow starts, and a code whose dh block is the correction from that support to the
body's hull. The prior part of the flow comes from scripts/train_prior.py and is frozen here.

For a draw x0 ~ N(0, I) and a time t, the state is x_t = (1-t) x0 + t x1. The operator is
applied at the body the prior's velocity says the state is heading for, the reader turns the
whitened residual into a summary, the adjoint carries the residual back onto the code, and
the expert that owns t predicts the velocity. The velocity is scored against the one that
would take the state straight to x1 in the time left, with the same weight at every t. The
endpoint it implies is decoded and its inside-or-outside at a cloud of probes is scored
against the body's, so the loss sees the shape and not only the numbers that encode it. For
draws late in t, where that endpoint is nearly the answer, it is rendered once more and must
fit the data to within the noise. See flow_loss.

The real curves carry measurement noise and model error, so the training curves carry both:
Gaussian noise at the measured per-azimuth profile at a level drawn per body, and a
model-error term of the size the calibration fitted for each curve, as smooth in phase as
the curve. The residual is divided by the combined scale of the two, as at reconstruction.
Each body is also turned by a random number of quarter turns about its spin axis, which is
an exact symmetry of the problem (quarter_turns).

--steps is a cap. A few corpus bodies are held out and scored every --val-every steps at
fixed draws; training stops once that score has gone --patience evaluations without
improving, and the saved weights are the best-scoring ones. --val-bodies 0 turns that off.

Every training state lies on the straight line between the draw x0 and the body x1. States
taken from the sampler's own trajectory were tried as a way of training the network on the
states it meets at reconstruction, and are not used, for a reason that is exact rather than
empirical. A state the sampler reaches is a function of x0, the churn noise and the data,
all independent of x1 given the data, so the least-squares optimum of a velocity scored
against (x1 - x) / (1 - t) at such a state is (E[x1 | data] - x) / (1 - t), whatever x is.
A network trained that way carries every draw to the posterior mean, and the draws stop
spanning the bodies behind the data, which is what the flow exists to do. On the line the
state carries x1, the optimum is the conditional expectation given the state, and the
learned flow transports the prior to the posterior.

Training is two runs. The first trains one expert. The second continues its checkpoint
with --experts above one, which copies the best-scoring weights of the first run into one
expert per interval of t, so that each interval's velocity can specialise.

Public bodies are never in the corpus; they appear only at reconstruction.

Training checkpoints every --ckpt-every steps to --ckpt-file (default <--out>.ckpt) and
resumes from it by default, so a run can be spread over several jobs. Only a completed run
writes --out, which is what the pipeline reads; to reconstruct from an unfinished run, point
reconstruct_lpd.py --ckpt at the .ckpt itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.conventions import R_z, cameras, psi_grid                            # noqa: E402
from hac26.data_io import N_CAMS                                                # noqa: E402
from hac26.field import (CODE_DIM, DESIGN_N, EXTRACT_EXTENT, N_DIR,               # noqa: E402
                         N_NODES, DepthSphere, ImplicitBody, dir_design, node_turn,
                         sh_expand, spherical_design, support_resample,
                         support_resample_weights)
from hac26.forward.mesh.exact import RenderConfig                               # noqa: E402
from hac26.forward.mesh.instrument import Instrument                            # noqa: E402
from hac26.noise import NOISE_HI, NOISE_LO, NOISE_PROFILE                       # noqa: E402
from hac26.shapes import canonicalize_r, mesh_support, rescale_touch_z          # noqa: E402
from hac26.solvers.lpd_flow import (N_EXPERTS, N_FEAT, N_MODES, N_NODE_CH,     # noqa: E402
                                    N_SPHERE_CH, LPDFlow, flow_inputs, geometry_tags)
from hac26.solvers.operator import CodeOperator                                 # noqa: E402

from calibrate import OUT_INSTRUMENT as INSTRUMENT   # noqa: E402  per channel, by calibrate.py
CALIBRATION = INSTRUMENT["real"]                      # the laboratory channel's instrument
PRIOR = "runs/prior_flow.pt"                       # written by scripts/train_prior.py
CORPUS = "runs/corpus.npz"                         # written by scripts/build_corpus.py
RENDER = RenderConfig()                            # the operator's discretisation


def add_render_flags(ap) -> None:
    """--height, --width and --sun-res: the sensor and sun-view discretisation of the exact
    forward model, for every stage that renders.

    They default to RENDER, so a run that passes none of them renders at the size the
    pipeline is calibrated at. They exist because the cost of this pipeline is dominated by
    the sensor pixels, and a check of whether the stages agree with each other -- which is
    what a whole-pipeline test on a small library is -- says the same thing at a fraction of
    the size, while at the calibrated size it costs what a run costs.
    """
    ap.add_argument("--height", type=int, default=RENDER.height,
                    help="sensor image height; with --width and --sun-res, a lower value "
                         "checks the wiring at a fraction of the cost and is not a result")
    ap.add_argument("--width", type=int, default=RENDER.width)
    ap.add_argument("--sun-res", type=int, default=RENDER.sun_res)


def render_tag(cfg: RenderConfig) -> str:
    """How a discretisation is written into a file's metadata, so that a learned object and
    the run using it can be compared. It is the sensor and the sun view, which is what the
    flags move; the rest of RenderConfig is chunking and does not change a curve."""
    return f"{cfg.height}x{cfg.width}x{cfg.sun_res}"


def render_from(a, what: str = "run") -> RenderConfig:
    """The configuration --height, --width and --sun-res ask for, with a notice when it is
    not the calibrated one. A reduced size resolves the lit region's boundary more coarsely,
    so its curves are not the instrument's and its numbers are not the pipeline's."""
    cfg = replace(RENDER, height=a.height, width=a.width, sun_res=a.sun_res)
    if cfg != RENDER:
        print(f"  NOTE: rendering at {cfg.height}x{cfg.width}, sun view {cfg.sun_res}; a "
              f"{what} at a reduced size checks the wiring and is not a {what}", flush=True)
    return cfg

OCC_WEIGHT = 1.0       # weight of the occupancy term against the endpoint term. Both are of
                       # order one at initialisation, so one is the neutral choice.
OCC_MARGIN = 0.25      # width of the occupancy target's soft edge, as a fraction of the probe
                       # spacing along a ray. A probe further from the surface than a few of
                       # these saturates and stops contributing, and the probes cannot place the
                       # surface finer than their own spacing anyway.
PROBE_RADII = 12       # probes along each node's ray, from the centre out to the extraction's
                       # extent. The probes sit on the nodes' own rays rather than on a grid in
                       # space, which is what makes the field at them free of any neighbour
                       # search: the depth at a probe is the depth at its own node, so the whole
                       # cloud costs one sparse product with the node kernel and one maximum
                       # over the core's normals per radius.
CARVE_WEIGHT = 20.0    # how much a probe the body's hull gets wrong counts against one it gets
                       # right, in the occupancy term. A carve touches a few percent of the
                       # probes, so this puts about half the term on the carve; see step_loss.
OCC_LOGIT = 12.0       # where the occupancy term saturates, in units of its soft edge, so
                       # about three probe spacings from the surface
FIT_WEIGHT = 1.0       # weight of the data-fit term (data_fit) against the flow term
FIT_FROM = 1.0 - 1.0 / N_EXPERTS   # the data-fit term applies at t from here on: the interval
                                   # the last of the N_EXPERTS experts owns, where the endpoint
                                   # estimate is nearly the answer
FIT_KNEE = 1.0         # excess misfit, in noise standard deviations, beyond which the data-fit
                       # term stops growing as a square and grows in proportion instead. It
                       # bounds the pull the term can exert at the pull a body two standard
                       # deviations from the curves exerts, so a badly placed endpoint cannot
                       # set the direction of the step on its own, while the term keeps both
                       # its full strength in that direction and a value that goes on telling
                       # a poor body from a hopeless one. See data_fit.


# ------------------------------------------------------------------------------ the corpus

class Corpus(NamedTuple):
    """The training bodies, as scripts/build_corpus.py writes them: raw codes (n, CODE_DIM)
    whose dh block is the correction from the convex start to the true hull, their noise-free
    curves (n, G, 2, P), the count curves (n, 3, G, P) they have under the thresholds of the
    other three quarter frames (what a quarter turn needs, quarter_turns), the convex start
    `support` (n, DESIGN_N) the flow begins from, the true hull support `support_true`, the
    radius each was rendered at, and each body's index in the codes file, which is how a
    held-out body is named across scripts."""
    codes: torch.Tensor
    curves: torch.Tensor
    turned_counts: torch.Tensor
    support: torch.Tensor
    support_true: torch.Tensor
    radius: torch.Tensor
    index: torch.Tensor

    def to(self, device) -> "Corpus":
        return Corpus(*(t.to(device) for t in self))


def load_corpus(path: str) -> tuple:
    """(Corpus, metadata) from the file scripts/build_corpus.py wrote."""
    if not Path(path).exists():
        raise SystemExit(f"{path} missing -- run scripts/build_corpus.py first")
    z = np.load(path, allow_pickle=False)
    need = {"codes", "curves", "turned_counts", "support", "support_true", "radius", "index",
            "meta"}
    if not need.issubset(z.files):
        raise SystemExit(f"{path} is not a corpus file of this layout (missing "
                         f"{sorted(need - set(z.files))}); rebuild it with "
                         f"scripts/build_corpus.py")
    meta = json.loads(str(z["meta"]))
    if z["codes"].shape[1] != CODE_DIM or z["support"].shape[1] != DESIGN_N:
        raise SystemExit(f"{path} was built for another code layout; rebuild it")
    corpus = Corpus(*(torch.tensor(z[k]) for k in Corpus._fields))
    print(f"  corpus: {len(corpus.codes)} bodies from {path}, {meta['phases']} phases, "
          f"radii {float(corpus.radius.min()):.2f}-{float(corpus.radius.max()):.2f}",
          flush=True)
    return corpus, meta


def held_out(n_bodies: int, n_val: int) -> np.ndarray:
    """The codes-file indices held out of training: the first n_val of a fixed permutation
    of the n_bodies indices, the same in every script, so a body never scores a network that
    trained on it and the prior and the flow hold out the same bodies."""
    perm = torch.randperm(n_bodies, generator=torch.Generator().manual_seed(0)).numpy()
    return np.sort(perm[:n_val])


def file_digest(path) -> str:
    """Short SHA-256 of a file's bytes, so a checkpoint can tell which inputs made it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def support_from_mesh(verts: np.ndarray, faces: np.ndarray) -> torch.Tensor:
    """The base support h of a body from a mesh of it: posed, brought to the canonical frame
    and evaluated on the core's normals. The convex stage's meshes are in the physical frame
    (xy scaled to the published radius) while the corpus is posed at xy radius 1, so the
    vertices are canonicalised first. A support function is a max over vertices, so it
    cannot be rescaled after the fact."""
    v = canonicalize_r(rescale_touch_z(np.asarray(verts, dtype=np.float64),
                                       np.asarray(faces)))
    n = ImplicitBody().core.n.detach().cpu().numpy()
    return torch.tensor(np.maximum(mesh_support(v, n), 1e-3), dtype=torch.float32)


# ----------------------------------------------------------------------- quarter turns

_TURN_CACHE: dict = {}


def _quarter_turn_maps(device):
    """What a turn of the body by q quarter turns about the spin axis does to a code, for
    q = 1, 2, 3, stacked along the first axis: the permutation of the depth nodes
    (3, N_NODES), the resampling of a support function onto the turned design normals as
    (indices, weights) (3, DESIGN_N, k), and the matrix taking dh on its directions to the
    turned directions (3, N_DIR, N_DIR).

    All three are exact, each for its own reason: the node set is built to be invariant under a
    quarter turn (field.node_design), so a turned body's depths are the same numbers in a
    different order; dh is band-limited, so its matrix is exact on the band; and the support
    function is resampled the way support_from_mesh's normals resample any support. Built once
    per device."""
    dev = torch.device(device)
    if dev not in _TURN_CACHE:
        nrm = spherical_design(DESIGN_N)
        dirs = dir_design(N_DIR)
        perms, idxs, ws, es = [], [], [], []
        for q in (1, 2, 3):
            R = R_z(q * np.pi / 2.0)
            # a turned body's value at y is the unturned body's value at R^T y
            perms.append(torch.from_numpy(node_turn(q, N_NODES)).long())
            idx, w = support_resample_weights(nrm, nrm @ R)
            idxs.append(torch.from_numpy(idx)); ws.append(torch.from_numpy(w))
            es.append(torch.from_numpy(sh_expand(dirs, dirs @ R)))
        _TURN_CACHE[dev] = tuple(torch.stack(x).to(dev) for x in (perms, idxs, ws, es))
    return _TURN_CACHE[dev]


def quarter_turns(corpus: Corpus, idx, q):
    """The bodies `idx` turned by q (B,) quarter turns about the spin axis, as (codes,
    curves, support, support_true). A turned body at a frame is the unturned body a quarter
    of the rotation on, so the code turns with the body (_quarter_turn_maps) and every curve
    shifts by a quarter of its phases. The count curve takes its threshold from the first
    frame, and the turned body's first frame is another frame of the unturned body, so its
    count curve is the unturned body's count under that frame's threshold, which the corpus
    carries (`turned_counts`), shifted. The turn is an exact symmetry of the problem and
    gives four training pairs per body at no operator cost. The phase count must be divisible
    by four; the caller checks."""
    codes, curves, sup, sup_true = (corpus.codes[idx], corpus.curves[idx], corpus.support[idx],
                                    corpus.support_true[idx])
    q = torch.as_tensor(q, device=codes.device)
    P = curves.shape[-1]
    perms, idxs, ws, es = _quarter_turn_maps(codes.device)
    codes, curves, sup, sup_true = codes.clone(), curves.clone(), sup.clone(), sup_true.clone()
    for k in (1, 2, 3):
        sel = q == k
        if not sel.any():
            continue
        perm, ridx, rw, e = perms[k - 1], idxs[k - 1], ws[k - 1], es[k - 1]
        codes[sel, :N_DIR] = codes[sel, :N_DIR] @ e.T
        codes[sel, N_DIR:] = codes[sel, N_DIR:][:, perm]
        sup[sel] = (sup[sel][:, ridx] * rw).sum(-1)
        sup_true[sel] = (sup_true[sel][:, ridx] * rw).sum(-1)
        # the turned body's first frame is the unturned body's frame (4 - k) P / 4, whose counts
        # sit at index 3 - k of turned_counts
        curves[sel, :, 1] = corpus.turned_counts[idx][sel, 3 - k]
        curves[sel] = torch.roll(curves[sel], shifts=k * (P // 4), dims=-1)
    return codes, curves, sup, sup_true


# --------------------------------------------------------------------- the training draws

def occ_eps_default() -> float:
    """The soft edge in model units: OCC_MARGIN of the spacing of the probes along a ray."""
    return OCC_MARGIN * (EXTRACT_EXTENT / PROBE_RADII)


def noise_sigma(n: int, generator=None) -> torch.Tensor:
    """Per-curve noise levels (n, G, 2) for n bodies: one overall level per body, drawn from
    [NOISE_LO, NOISE_HI], times the measured per-azimuth profile."""
    level = NOISE_LO + (NOISE_HI - NOISE_LO) * torch.rand(n, 1, 1, generator=generator)
    profile = torch.from_numpy(NOISE_PROFILE).reshape(2, -1).T          # (G, 2)
    return level * profile[None]


def model_error_scale(inst: Instrument) -> torch.Tensor:
    """The calibration's per-curve model error eta as (G, 2), on the CPU."""
    return inst.eta.detach().cpu().float().reshape(2, N_CAMS).T


def smooth_noise_like(curves: torch.Tensor, generator=None) -> torch.Tensor:
    """Unit-variance random curves with the spectral shape of `curves` (..., P): the
    Fourier amplitudes of each curve about its mean, with random phases. This is what the
    model error of a curve is taken to look like: as smooth in phase as the curve itself."""
    P = curves.shape[-1]
    amp = torch.fft.rfft(curves - curves.mean(-1, keepdim=True), dim=-1).abs()
    phase = torch.rand(amp.shape, generator=generator) * 2.0 * np.pi
    z = torch.fft.irfft(amp.cpu() * torch.exp(1j * phase), n=P, dim=-1)
    z = z - z.mean(-1, keepdim=True)
    return (z / z.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)).to(curves.device)


_EXPAND: dict = {}


def dh_expand(device="cpu") -> torch.Tensor:
    """The matrix (DESIGN_N, N_DIR) taking dh on its directions to the core's normals,
    band-limited: the one ImplicitBody applies (field.sh_expand). Built once per device."""
    dev = torch.device(device)
    if dev not in _EXPAND:
        _EXPAND[dev] = torch.from_numpy(
            sh_expand(dir_design(N_DIR), spherical_design(DESIGN_N))).to(dev)
    return _EXPAND[dev]


def inv_softplus(h: torch.Tensor) -> torch.Tensor:
    """The inverse of softplus, stable at small h."""
    hh = h.clamp_min(1e-6)
    return hh + torch.log(-torch.expm1(-hh))


def support_with(h: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    """softplus(inv_softplus(h) + expand(dh)): the support a base h and a correction dh make,
    the same operation ImplicitBody.support() performs, so the flow is supervised on exactly
    what it emits at reconstruction."""
    e = dh_expand(h.device)
    return torch.nn.functional.softplus(inv_softplus(h) + dh.to(h.device) @ e.T)


# ------------------------------------------------------------------- the network's inputs

_DIR_CACHE = None
_DIR_CACHE_DEV: dict = {}


def cond_channels(support: torch.Tensor, device=None):
    """The per-body input channels of the two branches, for a base support h (B, DESIGN_N),
    with the adjoint slots still empty (lpd_flow.flow_inputs fills them).

    Sphere branch (B, N_DIR, N_SPHERE_CH): h resampled onto the dh directions, the three
    components of each direction, and the adjoint slot. The resample is support_resample, not
    sh_expand: sh_expand keeps only low harmonic degrees, which is right for dh and would
    throw away part of h on a flat-faced body.

    Depth branch (B, N_NODES, N_NODE_CH): how far out the core's surface lies along each
    node's ray, the three components of the node direction, and the adjoint slot. That first
    channel is the room there is to carve at that node -- the quantity field.depth_cap takes the
    smallest of -- and it is the depth branch's whole view of the body it is correcting. There
    is no inside indicator and no coordinate triple: on a sphere of directions every coordinate
    is on the surface, which is the reason this branch has two fewer channels than the lattice
    branch it replaces and none of them wasted.
    """
    global _DIR_CACHE
    dev = device or support.device
    sup = support if support.dim() == 2 else support[None]
    B = sup.shape[0]
    if _DIR_CACHE is None:
        d = dir_design(N_DIR)
        nrm = spherical_design(DESIGN_N)
        _DIR_CACHE = (torch.from_numpy(d.astype(np.float32)),
                      torch.from_numpy(support_resample(nrm, d)),
                      DepthSphere(N_NODES).u)
    if dev not in _DIR_CACHE_DEV:
        _DIR_CACHE_DEV[dev] = tuple(t.to(dev) for t in _DIR_CACHE)
    dirs, to_dir, nodes = _DIR_CACHE_DEV[dev]
    h_dir = sup.to(dev) @ to_dir.T                                    # (B, N_DIR)
    sph = torch.cat([h_dir[..., None],
                     dirs[None].expand(B, -1, -1),
                     torch.zeros(B, N_DIR, 1, device=dev)], -1)       # (B, N_DIR, 5)

    reach = core_reach(sup.to(dev))                                   # (B, N_NODES)
    node = torch.cat([reach[..., None],
                      nodes[None].expand(B, -1, -1),
                      torch.zeros(B, len(nodes), 1, device=dev)], -1)
    assert sph.shape[-1] == N_SPHERE_CH and node.shape[-1] == N_NODE_CH
    return sph, node


def spectrum(curves: torch.Tensor, n_modes: int) -> torch.Tensor:
    """Fourier coefficients of orders 1..n_modes along the phase axis, scaled so that white
    noise of unit variance per phase gives coefficients of unit variance."""
    P = curves.shape[-1]
    return torch.fft.rfft(curves, dim=-1)[..., 1:n_modes + 1] * np.sqrt(2.0 / P)


def residual_features(data: torch.Tensor, pred: torch.Tensor, sigma: torch.Tensor,
                      n_modes: int, geom_mask: torch.Tensor) -> torch.Tensor:
    """The dual network's input (B, G, N_MODES, N_FEAT) from the data (B, G, 2, P), the
    prediction (B, G, 2, P) and the per-curve noise levels (B, G, 2): the spectrum of the
    whitened residual (data - pred) / sigma, through asinh so that a large residual early in
    the flow cannot swamp the network while a residual at the noise level passes unchanged,
    then the spectrum of the data. Geometries with `geom_mask` (B, G) zero carry zeros."""
    B, G, _, _ = data.shape
    r = spectrum((data - pred) / sigma[..., None], n_modes)           # (B, G, 2, M) complex
    d = spectrum(data, n_modes)
    parts = [torch.asinh(r[:, :, 0].real), torch.asinh(r[:, :, 0].imag),
             torch.asinh(r[:, :, 1].real), torch.asinh(r[:, :, 1].imag),
             d[:, :, 0].real, d[:, :, 0].imag, d[:, :, 1].real, d[:, :, 1].imag]
    feats = torch.zeros(B, G, N_MODES, N_FEAT, device=data.device)
    feats[:, :, :r.shape[-1]] = torch.stack(parts, -1)
    return feats * geom_mask[:, :, None, None]


# ------------------------------------------------------------------- the occupancy target

_PROBE_CACHE: dict = {}


def _probe_geometry(dev):
    """Three fixed tensors over the probe cloud: the node directions projected on every design
    normal (N_NODES, DESIGN_N), the radii the probes sit at (PROBE_RADII,), and the node kernel
    (N_NODES, N_NODES). None depends on the body, so all three are built once per device.

    The probes are the nodes' own rays, so the projection does not depend on the radius: a probe
    at radius r in direction u has n . (o + r u) = n . o + r (n . u), and the second factor is
    this matrix. That is what lets a cloud of PROBE_RADII * N_NODES points cost one maximum per
    radius rather than one per point.
    """
    if dev not in _PROBE_CACHE:
        rep = DepthSphere(N_NODES)
        proj = rep.u.double() @ ImplicitBody().core.n.double().T      # (N_NODES, DESIGN_N)
        r = (torch.arange(PROBE_RADII, dtype=torch.float64) + 0.5) / PROBE_RADII * EXTRACT_EXTENT
        kern = torch.from_numpy(rep.matrix(rep.u.numpy()).toarray())
        _PROBE_CACHE[dev] = (proj.float().to(dev), r.float().to(dev), kern.float().to(dev))
    return _PROBE_CACHE[dev]


def core_reach(h: torch.Tensor) -> torch.Tensor:
    """How far the core's surface is from the centre along each node's ray, (B, N_NODES), for
    supports h (B, DESIGN_N).

    Along the ray o + t u the core is the largest of n . o + t (n . u) - h over the normals, so
    it reaches zero at the smallest of (h - n . o) / (n . u) over the normals that face the ray.
    This is the depth at which a carve in that direction would take the surface all the way to
    the centre, so it is exactly the per-node form of field.depth_cap's bound, and it is what
    the depth branch reads to know how much room it has.
    """
    proj, _, _ = _probe_geometry(h.device)                            # (N_NODES, DESIGN_N)
    off = probe_centres(h) @ ImplicitBody().core.n.to(h.device).T     # (B, DESIGN_N)
    far = torch.tensor(float("inf"), device=h.device)
    return torch.stack([torch.where(proj > 1e-6, (h[b] - off[b]) / proj.clamp_min(1e-6),
                                    far).amin(-1) for b in range(len(h))])


def probe_centres(h: torch.Tensor) -> torch.Tensor:
    """The centre each body's depths are measured from, (B, 3), for supports h (B, DESIGN_N).

    The Steiner point of field.core_centre, written as one product so that it is batched and so
    that it carries a gradient: the occupancy term reads the field of a decoded support, the
    centre of that support is part of where the field is, and a centre taken through numpy would
    silently cut the loss off from it."""
    return 3.0 * (h @ ImplicitBody().core.n.to(h.device)) / h.shape[-1]


def probe_field(h: torch.Tensor, a: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    """f at the probe cloud for a batch of bodies, (B, PROBE_RADII, N_NODES): the core at
    support h (B, DESIGN_N) plus the depth field, which on a node's own ray is that node's own
    depth smoothed over its neighbours. This is ImplicitBody.forward evaluated at the probes,
    with the support supplied rather than stored, so it stays differentiable in both h and a."""
    proj, radii, kern = _probe_geometry(h.device)
    depth = a @ kern.T                                                # (B, N_NODES)
    off = centre @ ImplicitBody().core.n.to(h.device).T               # (B, DESIGN_N)
    out = []
    for r in radii:
        core = torch.stack([(r * proj + (off[b] - h[b])).amax(-1) for b in range(len(h))])
        out.append(core + depth)
    return torch.stack(out, 1)


# --------------------------------------------------------------- the operator in the loop

def operator_inputs(net, op: CodeOperator, x1_hat, h, radius, data, sigma, geoms, step_mask,
                    sph, node, M):
    """Run the operator and its adjoint at the endpoint estimate x1_hat (B, CODE_DIM,
    whitened) of every body and build the network's inputs from the result: the whitened
    residual features and the adjoint of the whitened misfit. A body without curves is
    switched off in the returned mask and carries a zero adjoint. Returns
    (inputs, number of bodies dropped)."""
    B, C = step_mask.shape
    dev = x1_hat.device
    gsel = torch.arange(C, device=dev) if geoms is None else torch.tensor(geoms, device=dev)
    raw = net.codec.decode(x1_hat)
    pred = torch.zeros_like(data)
    adj = torch.zeros(B, CODE_DIM, device=dev)
    live = torch.ones(B, device=dev)
    for b in range(B):
        d_b, s_b = data[b, gsel].to(op.device), sigma[b, gsel].to(op.device)
        # the cotangent on the normalised curves: the descent direction of the whitened
        # misfit, (data - A(x)) / sigma^2
        cur, grad = op.adjoint(h[b], raw[b], float(radius[b]),
                               lambda c: (d_b - c) / s_b[..., None] ** 2, geoms=geoms)
        if cur is None:
            live[b] = 0.0
            continue
        pred[b, gsel] = cur.to(dev)
        adj[b] = grad.to(dev)
    step_mask = step_mask * live[:, None]
    feats = residual_features(data, pred, sigma, M, step_mask)
    # the adjoint is taken with respect to the raw code at x1_hat; the network works in the
    # whitened code, so it is taken back through the codec at that point
    return flow_inputs(feats, step_mask, sph, node, net.codec.pullback(x1_hat, adj)), int(B - live.sum())


# ------------------------------------------------------------------------------- the loss

def step_loss(net, xt, t, v, x1, sup_true, code_true, h_base, occ_weight, occ_eps):
    """The two terms scored on a velocity v at the state (xt, t).

    The flow term is the squared error of v against the velocity that takes the state
    straight to x1 in the time left, (x1 - xt) / (1 - t), averaged per block so the many g
    coordinates do not swamp the few dh coordinates. On the straight line that target is
    x1 - x0. Every t weighs the same: a velocity error moves the sampler's answer by the same
    amount whenever it happens, and the late ones are never corrected.

    The occupancy term scores the endpoint the velocity implies, x1_hat = xt + (1 - t) v,
    decoded: the cross-entropy of its inside-or-outside at the lattice sites against the
    corpus body's (true hull support `sup_true`, raw code `code_true`). The endpoint's dh is
    measured from `h_base`, the convex start the operator ran at, exactly as at
    reconstruction.

    The sites are not weighted equally. The convex stage already supplies the hull, so a site
    the hull places correctly asks nothing of the flow, and a body's hull places all but a few
    percent of the sites correctly. Weighted equally, almost all of the term would reward
    reproducing the hull and the carve would be a rounding error in it, which is the one thing
    the flow exists to produce. So a site where the hull and the body disagree counts
    CARVE_WEIGHT times one where they agree, and the weights are normalised per body; on a
    convex body every weight is one and the term is unchanged.

    The cross-entropy saturates at OCC_LOGIT rather than growing with the field. A decoded
    field of several probe spacings says nothing more about where the surface is than one of a
    single spacing, and left unbounded a single probe with a large field contributes hundreds
    to the loss and destabilises the step.

    Returns (total, flow term, occupancy term, decoded endpoint)."""
    err = (v - (x1 - xt) / (1 - t[:, None])) ** 2
    flow = 0.5 * (err[:, :N_DIR].mean() + err[:, N_DIR:].mean())
    x1_hat = xt + (1 - t[:, None]) * v
    with torch.no_grad():
        g_true = code_true[:, N_DIR:]
        o_true = probe_centres(sup_true)
        occ_true = torch.sigmoid(-probe_field(sup_true, g_true, o_true) / occ_eps)
        occ_hull = torch.sigmoid(-probe_field(sup_true, torch.zeros_like(g_true), o_true)
                                 / occ_eps)
        w = 1.0 + CARVE_WEIGHT * (occ_hull - occ_true).abs()
        w = w / w.mean((1, 2), keepdim=True).clamp_min(1e-6)
    raw = net.codec.decode(x1_hat)
    h_est = support_with(h_base, raw[:, :N_DIR])
    f_est = probe_field(h_est, raw[:, N_DIR:], probe_centres(h_est))
    logit = OCC_LOGIT * torch.tanh(-f_est / (occ_eps * OCC_LOGIT))
    occ = (torch.nn.functional.binary_cross_entropy_with_logits(logit, occ_true,
                                                                reduction="none") * w).mean()
    return flow + occ_weight * occ, flow, occ, raw


def data_fit(net, op: CodeOperator, x1_hat, h, radius, data, scale, geoms, step_mask):
    """The data-fit term on endpoint estimates x1_hat (B, CODE_DIM, whitened): each decoded
    endpoint is rendered and chi is the RMS of its whitened residual against the data over the
    geometries used. The term is zero once the endpoint fits the data to within the noise,
    because fitting below the noise fits noise. Above the noise it grows as the square of the
    excess up to FIT_KNEE and in proportion to it beyond, which is what keeps a body far from
    the data from setting the direction of the step: a square makes the pull grow without
    bound in the region where the endpoint estimate is least trustworthy, whereas the
    proportional part holds the pull at a fixed size while leaving its direction, the one the
    curves ask for, untouched. Returns (mean term over the bodies with curves, its gradient
    with respect to x1_hat (B, CODE_DIM), number of bodies without curves). The gradient comes
    from the adjoint of the same operator call, so the term costs one call per body, and the
    caller attaches it to the graph with with_gradient."""
    B, C = step_mask.shape
    dev = x1_hat.device
    gsel = torch.arange(C, device=dev) if geoms is None else torch.tensor(geoms, device=dev)
    raw = net.codec.decode(x1_hat.detach())
    vals = torch.zeros(B, device=dev)
    grad = torch.zeros(B, CODE_DIM, device=dev)
    live = torch.zeros(B, device=dev)
    for b in range(B):
        if step_mask[b].sum() == 0:
            continue
        d_b, s_b = data[b, gsel].to(op.device), scale[b, gsel].to(op.device)
        cur, g = op.adjoint(h[b], raw[b], float(radius[b]),
                            lambda c: (d_b - c) / s_b[..., None] ** 2, geoms=geoms)
        if cur is None:
            continue
        live[b] = 1.0
        r = (d_b - cur) / s_b[..., None]
        chi = r.pow(2).mean().sqrt()
        excess = (chi - 1.0).clamp_min(0.0)
        held = excess.clamp_max(FIT_KNEE)
        vals[b] = held * (2.0 * excess - held)     # excess^2 below the knee, linear above it
        # g is the gradient of -chi^2 n / 2 with respect to the raw code, n the number of
        # residual entries. The term's derivative with respect to the excess is twice the
        # excess below the knee and twice the knee above it, which is `held` either way.
        grad[b] = -(2.0 / r.numel()) * float(held / chi.clamp_min(1e-12)) * g.to(dev)
    n = live.sum().clamp_min(1.0)
    return vals.sum() / n, net.codec.pullback(x1_hat.detach(), grad) / n, int(B - live.sum())


def with_gradient(value: torch.Tensor, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """A scalar equal to `value` whose gradient with respect to `x` is `grad`: for a term whose
    value and gradient were computed outside autograd."""
    return value.detach() + ((x - x.detach()) * grad.detach()).sum()


class Diag(NamedTuple):
    """Numbers flow_loss reports beside the loss: the mean |g| of the endpoint estimate and
    its spread across the batch (what a collapse to the conditional mean would move first),
    the flow, occupancy and data-fit terms, and the bodies dropped for having no curves."""
    g_mean: float
    g_spread: float
    flow: float
    occ: float
    fit: float
    dropped: int


def flow_loss(net, op: CodeOperator, corpus: Corpus, eta, idx, x0, t, M, tag, mask,
              train_geoms=None, sigma=None, xi=None, zeta=None, turns=None,
              return_diag=False, ablate=False, occ_weight=OCC_WEIGHT, occ_eps=None,
              fit_weight=FIT_WEIGHT):
    """The training loss for one batch, given the draws (idx, x0, t, sigma, xi, zeta, turns):
    the flow term and the occupancy term of step_loss, plus the data-fit term of data_fit for
    the draws in the last expert's interval.

    Validation calls this too, with fixed draws, so it scores the same objective through the
    same code, operator included.

    The data. The corpus curves are noise-free; the data the network sees are
    curves + sigma * xi + eta * zeta: xi standard normal with sigma the per-curve noise level
    of this body, drawn by noise_sigma, and zeta a unit-variance curve as smooth as the curve
    itself (smooth_noise_like) with `eta` (G, 2) the calibration's per-curve model error. The
    residual and the adjoint are divided by sqrt(sigma^2 + eta^2), as at reconstruction.
    `turns` (B,) in 0..3 turns each body by quarter turns first (quarter_turns); None leaves
    them as they are.

    The start. The operator runs at the corpus body's convex start, the support the convex
    stage reconstructed from its own curves, and the code's dh block is the correction from
    there to the true hull (scripts/build_corpus.py); so the flow is supervised on exactly
    the correction it has to make at reconstruction.

    The state. x_t is the point of the straight line x_t = (1-t) x0 + t x1 at the drawn t,
    and nothing else; the module docstring says why states off the line are not scored.

    The operator's inputs are taken at the endpoint the prior's velocity implies,
    x1_hat = x_t + (1-t) v_prior(x_t), not at x_t: at small t, x_t is mostly the Gaussian
    draw x0, which decodes to a body unlike anything in the corpus, and a residual taken
    there says little about the body being reconstructed. The prior is trained as a denoiser
    and frozen, so this point is stable while the data part trains; it costs one network
    forward and no operator call.

    The data-fit term. For the draws with t >= FIT_FROM the endpoint the velocity implies,
    x_t + (1-t) v, is rendered once more and must fit the data to within the noise
    (data_fit). There the endpoint is nearly the answer. At earlier t the flow term's target is
    the average of the bodies that could be behind the state, which the data-fit term would
    pull away from, so it is not applied there. It costs one more operator call per such draw.

    A body whose endpoint estimate has no curves (degenerate mesh, unusable patches) is
    dropped from the operator for this step: its geometries are masked and its adjoint is
    zero, so the network sees it as a body without data rather than as data of zero flux.

    `ablate` also scores the prior's velocity alone on the same draws, flow and occupancy
    terms only, and returns (loss without the data-fit term, ablated loss, n_dropped); the
    difference is what the data part contributes.

    `return_diag` returns (loss, Diag).
    """
    if ablate and return_diag:
        raise ValueError("flow_loss: ablate and return_diag return different tuples; "
                         "ask for one or the other")
    if occ_eps is None:
        occ_eps = occ_eps_default()
    B, C = len(idx), tag.shape[1]
    dev = corpus.codes.device
    if turns is None:
        codes, curves, sup, sup_true = (corpus.codes[idx], corpus.curves[idx],
                                        corpus.support[idx], corpus.support_true[idx])
    else:
        codes, curves, sup, sup_true = quarter_turns(corpus, idx, turns)
    P = curves.shape[-1]
    if sigma is None:
        sigma = noise_sigma(B).to(dev)
    if xi is None:
        xi = torch.randn(B, C, 2, P, device=dev)
    if zeta is None:
        zeta = smooth_noise_like(curves)
    eta = eta.to(dev)[None].expand(B, -1, -1)
    x1 = net.codec.encode(codes)
    data = curves + sigma[..., None] * xi + eta[..., None] * zeta       # (B, G, 2, P)
    scale = torch.sqrt(sigma ** 2 + eta ** 2)                            # (B, G, 2)
    radius = corpus.radius[idx]

    geoms = None
    step_mask = mask.expand(B, C)
    if train_geoms is not None and 0 < int(train_geoms) < C:
        geoms = torch.randperm(C)[:int(train_geoms)].sort().values.tolist()
        step_mask = torch.zeros(B, C, device=dev)
        step_mask[:, geoms] = 1.0

    # h the operator runs at: the convex start of each body
    h_base = sup
    sph, node = cond_channels(h_base, device=dev)
    tag_b = tag.expand(B, C, 4)

    # The state, on the straight line.
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    with torch.no_grad():
        v0 = net.prior_velocity(xt, t, radius, sph, node)
        inp, n_dropped = operator_inputs(net, op, xt + (1 - t[:, None]) * v0, h_base, radius,
                                         data, scale, geoms, step_mask, sph, node, M)
    step_mask = inp.mask
    u = net.velocity(xt, t, radius, tag_b, inp)
    loss, flow, occ, raw = step_loss(net, xt, t, u, x1, sup_true, codes, h_base, occ_weight,
                                     occ_eps)
    if ablate:
        return loss, step_loss(net, xt, t, v0, x1, sup_true, codes, h_base, occ_weight,
                               occ_eps)[0], n_dropped

    fit = torch.zeros((), device=dev)
    if fit_weight > 0:
        sel = (t >= FIT_FROM) & (step_mask.sum(1) > 0)
        if bool(sel.any()):
            x1_hat = (xt + (1 - t[:, None]) * u)[sel]
            val, g_fit, n_bad = data_fit(net, op, x1_hat, h_base[sel], radius[sel], data[sel],
                                         scale[sel], geoms, step_mask[sel])
            n_dropped += n_bad
            fit = with_gradient(val, x1_hat, g_fit)
    loss = loss + fit_weight * fit
    if not return_diag:
        return loss
    with torch.no_grad():
        g_hat = raw[:, N_DIR:]
        diag = Diag(float(g_hat.abs().mean()),
                    float(g_hat.std(0).mean()) if len(g_hat) > 1 else float("nan"),
                    float(flow), float(occ), float(fit), n_dropped)
    return loss, diag


def validate(net, op, corpus, eta, val_idx, val_x0, val_t, val_sigma, val_xi, val_zeta, M,
             tag, mask, chunk, occ_weight=OCC_WEIGHT, occ_eps=None, fit_weight=FIT_WEIGHT):
    """Mean loss over the held-out bodies at fixed draws, and the mean Diag, chunked to bound
    memory. The bodies are not turned, so every evaluation scores the same draws."""
    was_training = net.training
    net.eval()
    tot, n, acc, dropped = 0.0, 0, np.zeros(len(Diag._fields) - 1), 0
    with torch.no_grad():
        for i in range(0, len(val_idx), chunk):
            sl = slice(i, i + chunk)
            b = len(val_idx[sl])
            l, d = flow_loss(net, op, corpus, eta, val_idx[sl], val_x0[sl], val_t[sl], M,
                             tag, mask, sigma=val_sigma[sl], xi=val_xi[sl],
                             zeta=val_zeta[sl], return_diag=True, occ_weight=occ_weight,
                             occ_eps=occ_eps, fit_weight=fit_weight)
            tot += b * float(l)
            acc += b * np.array([0.0 if v != v else v for v in d[:-1]])   # a nan spread is 0
            dropped += d.dropped
            n += b
    net.train(was_training)
    m = max(n, 1)
    return tot / m, Diag(*(acc / m), dropped)


# --------------------------------------------------------------------------- the training

def _hms(sec: float) -> str:
    """Seconds as h:mm:ss, for lines a human reads while a job is running."""
    sec = int(max(sec, 0.0))
    return f"{sec // 3600}:{(sec // 60) % 60:02d}:{sec % 60:02d}"


def _now() -> str:
    return time.strftime("%H:%M:%S")


class EMA:
    """Exponential moving average of the weights, with the same warm-up correction Adam uses
    for its moments: the buffer starts at zero and is divided by (1 - decay^n) when read, so
    the average is not biased toward the initial weights for the first 1/(1-decay) steps. The
    buffer must start at zero for that division to be right.
    """

    def __init__(self, net, decay: float = 0.999):
        self.decay = float(decay)
        self.n = 0
        self.shadow = {k: torch.zeros_like(v, dtype=torch.float32)
                       for k, v in net.state_dict().items() if v.dtype.is_floating_point}

    def update(self, net):
        self.n += 1
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                         alpha=1.0 - self.decay)

    def state(self, net):
        """Bias-corrected weights, in the layout state_dict() wants."""
        if self.n == 0:
            return {k: v.detach().clone() for k, v in net.state_dict().items()}
        c = 1.0 - self.decay ** self.n
        out = {}
        for k, v in net.state_dict().items():
            out[k] = (self.shadow[k] / c).to(v.dtype) if k in self.shadow else v.detach().clone()
        return out

    def load(self, d, n):
        # onto the device the shadow was built on; the checkpoint is read onto the CPU
        self.shadow = {k: v.detach().to(self.shadow[k].device if k in self.shadow
                                        else v.device).float().clone()
                       for k, v in d.items()}
        self.n = int(n)


class _Swapped:
    """Run a block with the EMA weights installed, then put the raw ones back."""

    def __init__(self, net, ema):
        self.net, self.ema = net, ema

    def __enter__(self):
        self.saved = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(self.ema.state(self.net))

    def __exit__(self, *exc):
        self.net.load_state_dict(self.saved)
        return False


def save_checkpoint(path, net, opt, step, best, best_state, best_step, stale, elapsed,
                    meta, ema=None):
    """Write a resumable training checkpoint: weights, optimiser state, EMA shadow, the
    best-so-far state, the early-stopping counters and the RNG state, so a resumed run
    continues as an uninterrupted one would have. Written to a temporary name and renamed, so
    a job killed mid-write cannot leave a truncated file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step,
                "best": best, "best_state": best_state, "best_step": best_step,
                "stale": stale, "elapsed": elapsed, "rng": torch.get_rng_state(),
                "ema": None if ema is None else ema.shadow,
                "ema_n": 0 if ema is None else ema.n,
                **meta}, tmp)
    tmp.replace(p)


def _enable_tf32():
    """Allow TF32 matmuls on CUDA. The field evaluation is large matrix products that do not
    need full float32 precision, and TF32 is several times faster on recent GPUs."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_prior(net: LPDFlow, path: str, device: str) -> dict:
    """Install the prior part and the codec from the file scripts/train_prior.py wrote, and
    freeze the prior. Returns the file's metadata."""
    if not Path(path).exists():
        raise SystemExit(f"{path} missing -- run scripts/train_prior.py first")
    st = torch.load(path, map_location="cpu", weights_only=False)
    net.prior.load_state_dict(st["prior"])
    net.codec.load_state_dict(st["codec"])
    net.prior.requires_grad_(False)
    net.to(device)
    print(f"  prior: {path}, trained {st['meta'].get('steps_trained', '?')} steps on "
          f"{st['meta'].get('bodies', '?')} bodies", flush=True)
    return st["meta"]


def load_flow_file(path: str, map_location="cpu") -> tuple:
    """Load a finished flow file or a resumable training checkpoint.

    New finished files are {"state_dict", "meta"}. Older finished files are bare state dicts.
    Resumable checkpoints are accepted for diagnostics and reconstruction by taking their best
    held-out state when available, as reconstruct_lpd.py already did before metadata was added.
    Returns (state_dict, metadata).
    """
    st = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(st, dict) and "state_dict" in st and "meta" in st:
        return st["state_dict"], dict(st.get("meta") or {})
    if isinstance(st, dict) and "net" in st and isinstance(st.get("step"), int):
        meta = {k: v for k, v in st.items()
                if k not in {"net", "opt", "best_state", "rng", "ema"}}
        meta["checkpoint_step"] = int(st["step"])
        meta["loaded_step"] = int(st.get("best_step", st["step"])) if st.get("best_state") else int(st["step"])
        meta["loaded_best_state"] = bool(st.get("best_state") is not None)
        return st.get("best_state") or st["net"], meta
    return st, {}


def check_flow_metadata(meta: dict, *, corpus: str | None = None, calibration: str | None = None,
                        phases: int | None = None, operator_res: int | None = None,
                        render: str | None = None,
                        context: str = "flow checkpoint") -> None:
    """Refuse known mismatches between a flow file and the run trying to use it."""
    if not meta:
        print(f"  WARNING: {context} has no metadata; cannot verify corpus/calibration/operator "
              f"settings", flush=True)
        return
    problems, missing = [], []

    def expect_digest(key, path):
        if path is None:
            return
        if key not in meta:
            missing.append(key)
            return
        live = file_digest(path)
        if meta[key] != live:
            problems.append(f"{key}: checkpoint={meta[key]!r}, current={live!r}")

    def expect_value(key, value):
        if value is None:
            return
        if key not in meta:
            missing.append(key)
            return
        if meta[key] != value:
            problems.append(f"{key}: checkpoint={meta[key]!r}, current={value!r}")

    def expect_int(key, value):
        if value is None:
            return
        if key not in meta:
            missing.append(key)
            return
        if int(meta[key]) != int(value):
            problems.append(f"{key}: checkpoint={meta[key]!r}, current={int(value)!r}")

    expect_digest("corpus", corpus)
    expect_digest("calibration", calibration)
    expect_int("phases", phases)
    expect_int("operator_res", operator_res)
    # A network trained against curves from one discretisation and used against another is
    # being asked about a different instrument, and nothing else here would notice.
    expect_value("render", render)
    if problems:
        raise SystemExit(f"{context} was written for different settings ({'; '.join(problems)})")
    if missing:
        print(f"  WARNING: {context} metadata is missing {', '.join(sorted(set(missing)))}; "
              f"verified the fields it did carry", flush=True)


def load_instrument(path: str, device: str) -> Instrument:
    """The calibrated instrument, frozen: only the calibration fits it. Training and
    reconstruction refuse to run without one: the curves depend on it, and a default
    instrument would be a different forward model."""
    if not Path(path).exists():
        raise SystemExit(f"{path} missing -- run scripts/calibrate.py first")
    inst = Instrument.load(path, device=device).requires_grad_(False)
    print(f"  instrument: {inst.summary()}", flush=True)
    return inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--train-geoms", type=int, default=28,
                    help="number of camera geometries sampled per flow step")
    ap.add_argument("--occ-weight", type=float, default=OCC_WEIGHT,
                    help="weight of the occupancy term on the endpoint estimate against the "
                         "endpoint term; 0 trains the plain flow objective")
    ap.add_argument("--occ-eps", type=float, default=None,
                    help="soft edge of the occupancy target in model units; default is "
                         "OCC_MARGIN of the probe spacing along a ray")
    ap.add_argument("--fit-weight", type=float, default=FIT_WEIGHT,
                    help="weight of the data-fit term on the endpoint estimate for draws in "
                         "the last expert's interval (see flow_loss); 0 turns it off and "
                         "saves its operator call")
    ap.add_argument("--experts", type=int, default=N_EXPERTS,
                    help="experts of the data part, one per interval of t. A run resumed "
                         "from a checkpoint with fewer experts branches: every new expert "
                         "starts as a copy of the one that owned its interval and then "
                         "trains on that interval alone. Train with 1 first, then branch.")
    ap.add_argument("--out", default="runs/lpd_flow.pt")
    ap.add_argument("--corpus", default=CORPUS,
                    help="the corpus written by scripts/build_corpus.py; it carries the "
                         "phase count and the operator resolution")
    ap.add_argument("--calibration", default=CALIBRATION,
                    help="the Instrument written by scripts/calibrate.py")
    ap.add_argument("--prior", default=PRIOR,
                    help="the prior flow and codec written by scripts/train_prior.py")
    # --steps is the cap; training stops earlier when the held-out loss stops improving.
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the global RNG; a resume restores the checkpointed RNG state "
                         "instead")
    ap.add_argument("--val-bodies", type=int, default=8,
                    help="bodies held out of training to score early stopping on; 0 trains "
                         "the full --steps and keeps the final weights")
    ap.add_argument("--val-every", type=int, default=200,
                    help="steps between held-out evaluations")
    ap.add_argument("--patience", type=int, default=5,
                    help="consecutive evaluations without improvement before stopping; "
                         "0 evaluates and checkpoints but never stops early")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="held-out loss must drop by at least this much to count as an "
                         "improvement")
    # --steps is a total across jobs, not a per-job budget: a resumed run trains up to the
    # same cap.
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="steps between resumable checkpoints; 0 disables them (the run "
                         "then has to finish in one job to leave anything behind)")
    ap.add_argument("--ckpt-file", default=None,
                    help="where the resumable checkpoint goes; defaults to <--out>.ckpt. "
                         "Keep it on persistent storage -- a worker's /tmp does not "
                         "survive the job")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="pick training back up from --ckpt-file when it exists "
                         "(--no-resume starts from step 0 and overwrites it)")
    ap.add_argument("--log-every", type=int, default=10,
                    help="steps between training-loss lines")
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA decay on the weights. Validation scores the averaged weights "
                         "and the saved checkpoint is those weights, so the model that is "
                         "selected is the model that was measured. 0 disables it.")
    ap.add_argument("--time-budget", type=float, default=20.0,
                    help="hours. After the first few steps the finish time is projected and "
                         "compared against this, and the largest --steps that would fit is "
                         "printed. It warns rather than exits: the run checkpoints and "
                         "resumes, so an overrun costs a restart, not the work.")
    ap.add_argument("--extra-steps", type=int, default=0,
                    help="when resuming, train this many steps beyond the checkpoint's step "
                         "instead of up to the --steps cap; for the branched second run")
    add_render_flags(ap)
    a = ap.parse_args()
    render = render_from(a, "training")
    _enable_tf32()
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data, cmeta = load_corpus(a.corpus)
    data = data.to(dev)
    codes = data.codes
    if cmeta["calibration"] != file_digest(a.calibration):
        raise SystemExit(f"{a.corpus} was built with another calibration than {a.calibration}; "
                         f"rebuild the corpus or point --calibration at the one it used")
    phases, op_res = int(cmeta["phases"]), int(cmeta["operator_res"])
    inst = load_instrument(a.calibration, dev)
    eta = model_error_scale(inst)
    op = CodeOperator(inst, psi_grid(phases), res=op_res, config=render, device=dev)
    print(f"  exact operator on {dev}, extraction res {op_res}, {len(cameras())} "
          f"geometries, {phases} phases; model error median {float(eta.median()):.4f}",
          flush=True)

    ckpt_path = a.ckpt_file or f"{a.out}.ckpt"
    st = None
    if a.resume and Path(ckpt_path).exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        stale = [k for k in ("corpus", "prior") if st.get(k) != file_digest(getattr(a, k))]
        if stale:
            # weights trained on a corpus or a prior that has since been rebuilt are not
            # progress
            print(f"  WARNING: {ckpt_path} was trained against a different "
                  f"{' and '.join(stale)} -- ignoring it and training from step 0",
                  flush=True)
            st = None
    # The network is built with the checkpoint's expert count and branched to --experts after
    # loading, so a run can be trained with one expert and continued with several.
    n_experts = int(st["n_experts"]) if st is not None else a.experts
    if n_experts > a.experts:
        raise SystemExit(f"{ckpt_path} has {n_experts} experts; --experts {a.experts} cannot "
                         f"merge them. Delete it or pass --no-resume.")
    net = LPDFlow(n_experts=n_experts).to(dev)
    # The prior part and the codec come from scripts/train_prior.py and are frozen here: only
    # the data part trains. The codec is the prior's, so the two parts speak the same
    # whitened code; it travels with every checkpoint to reconstruction.
    pmeta = load_prior(net, a.prior, dev)
    if pmeta.get("corpus") != file_digest(a.corpus):
        raise SystemExit(f"{a.prior} was trained on another corpus than {a.corpus}; rerun "
                         f"scripts/train_prior.py on this one")
    with torch.no_grad():
        z = net.codec.encode(codes)
    print(f"  codec: g scale {float(net.codec.g_s):.5f}, dh sd {float(net.codec.sd[0]):.5f}, "
          f"g sd {float(net.codec.sd[1]):.5f}; corpus in whitened space reaches "
          f"|z| = {float(z[:, N_DIR:].abs().max()):.2f}", flush=True)

    def fresh_optimiser_and_ema():
        # EMA of the weights. The velocity target is noisy, so an average of recent weights
        # is a better estimate than the last iterate. Validation scores the averaged weights
        # and the averaged weights are what is saved, so the model selected is the model
        # measured. The averaging window is capped at a tenth of the run: a longer window
        # would lag into the early phase of training, where the velocity still transports
        # every draw toward the corpus mean, which is smoother and more convex than any real
        # body.
        decay = min(a.ema, 1.0 - 1.0 / max(a.steps / 10.0, 10.0)) if a.ema else 0.0
        if a.ema:
            print(f"  EMA decay {decay:.5f} (window ~{1/(1-decay):.0f} steps of {a.steps})",
                  flush=True)
        params = list(net.reader.parameters()) + list(net.experts.parameters())
        return torch.optim.Adam(params, lr=1e-3), EMA(net, decay=decay)

    opt, ema = fresh_optimiser_and_ema()
    C = len(cameras())
    # The dual uses m = 1..N_MODES, and an rFFT of n phases yields floor(n/2)+1 coefficients,
    # so fewer than 2*N_MODES phases cannot supply them all.
    M = min(N_MODES, phases // 2)
    if M < N_MODES:
        print(f'  WARNING: only {M} modes available at {phases} phases; '
              f'{N_MODES} are required', flush=True)
    tag = geometry_tags().to(dev)
    mask = torch.ones(1, C, device=dev)
    train_geoms = max(1, min(C, int(a.train_geoms)))
    print(f"  training samples {train_geoms}/{C} geometries per step; noise level "
          f"{NOISE_LO:g}-{NOISE_HI:g} of the curve mean", flush=True)
    occ_eps = a.occ_eps if a.occ_eps is not None else occ_eps_default()
    print(f"  occupancy term: weight {a.occ_weight:g}, soft edge {occ_eps:.4f} "
          f"({PROBE_RADII} probes per ray, spacing "
          f"{EXTRACT_EXTENT / PROBE_RADII:.4f})", flush=True)
    print(f"  data-fit term: weight {a.fit_weight:g} on draws with t >= {FIT_FROM:g}"
          + ("" if a.fit_weight > 0 else " (off)"), flush=True)
    augment = phases % 4 == 0
    print("  quarter turns: on, four training pairs per body" if augment else
          f"  NOTE: {phases} phases is not divisible by 4, so the bodies are not turned",
          flush=True)

    # The held-out bodies are named by their index in the codes file through a fixed
    # permutation (held_out), so the split is the same on a resumed or repeated run, the
    # prior held out the same bodies, and a body never scores a network that trained on it.
    g_ref = float(codes[:, N_DIR:].abs().mean())
    n_val = max(0, min(a.val_bodies, len(codes) - 1))
    if n_val < a.val_bodies:
        print(f"  WARNING: corpus has {len(codes)} bodies; holding out {n_val} for "
              f"validation instead of {a.val_bodies}", flush=True)
    is_val = torch.as_tensor(np.isin(data.index.cpu().numpy(),
                                     held_out(int(cmeta["bodies"]), n_val)))
    val_idx = torch.nonzero(is_val).flatten().to(dev)
    train_idx = torch.nonzero(~is_val).flatten().to(dev)
    n_val = len(val_idx)
    if n_val:
        # Fixed draws: the same noise, times and data terms at every evaluation, so a change
        # in the score is a change in the network. The times are spread evenly over [0, 1),
        # matching the continuous t the flow is trained at.
        gen = torch.Generator().manual_seed(1234)
        val_x0 = torch.randn(n_val, codes.shape[1], dtype=codes.dtype,
                             generator=gen).to(dev)
        val_t = ((torch.arange(n_val, dtype=codes.dtype) + 0.5) / max(n_val, 1)).to(dev)
        val_sigma = noise_sigma(n_val, generator=gen).to(dev)
        val_xi = torch.randn(n_val, C, 2, phases, generator=gen).to(dev)
        val_zeta = smooth_noise_like(data.curves[val_idx].cpu(), generator=gen).to(dev)
        print(f"  {len(train_idx)} training bodies, {n_val} held out; validating every "
              f"{a.val_every} steps, patience {a.patience}", flush=True)
    else:
        print("  no held-out bodies: training the full --steps, keeping the final weights",
              flush=True)

    best, best_state, best_step, stale = float("inf"), None, -1, 0
    start_step, elapsed_before = 0, 0.0
    # Everything that defines the objective and the split. A resume with any of these changed
    # is refused, except the expert count, which may grow (branching): `bodies` because the
    # held-out split is a permutation of len(codes), so a different count would move bodies
    # across the split.
    meta = {
        "bodies": int(len(codes)),
        "dim": int(codes.shape[1]),
        "n_modes": int(N_MODES),
        "n_experts": int(a.experts),
        "loss": "velocity_mse_per_block_plus_occupancy_plus_data_fit",
        "occ_weight": float(a.occ_weight),
        "occ_eps": float(occ_eps),
        "fit_weight": float(a.fit_weight),
        "fit_from": float(FIT_FROM),
        "n_val": n_val,
        "phases": phases,
        "operator_res": op_res,
        "train_geoms": train_geoms,
        "render": render_tag(render),
        "corpus": file_digest(a.corpus),
        "prior": file_digest(a.prior),
    }

    if st is not None:
        bad = [f"{k}: checkpoint={st.get(k)!r}, current={v!r}"
               for k, v in meta.items() if k != "n_experts" and st.get(k) != v]
        if bad:
            raise SystemExit(
                f"{ckpt_path} was written for different flow settings "
                f"({'; '.join(bad)}). Delete it or pass --no-resume.")
        net.load_state_dict(st["net"])
        torch.set_rng_state(st["rng"])
        start_step = st["step"] + 1
        best, best_step, stale = st["best"], st["best_step"], st["stale"]
        best_state = st["best_state"]
        elapsed_before = st.get("elapsed", 0.0)
        if n_experts != a.experts and best_state is not None:
            # The weights the first run selected are the ones its validation scored, not the
            # ones it happened to hold at its last step, so the branch starts from them.
            net.load_state_dict(best_state)
            print(f"  branching from the best-scoring weights (step {best_step})", flush=True)
        print(f"  [{_now()}] resumed {ckpt_path} at step {start_step} of {a.steps} "
              f"({_hms(elapsed_before)} trained so far; best val "
              f"{best:.5f} from step {best_step}, {stale}/{a.patience} without "
              f"improvement)", flush=True)
        reset = []
        if n_experts != a.experts:
            # the best weights so far belong to the network before the split; the record
            # starts again for the branched one, with a fresh optimiser and average
            net.branch(a.experts)
            opt, ema = fresh_optimiser_and_ema()
            reset.append(f"branched from {n_experts} to {a.experts} experts at "
                         f"{net.edges.tolist()}")
        else:
            opt.load_state_dict(st["opt"])
            if st.get("ema") is not None:
                ema.load(st["ema"], st.get("ema_n", 0))
        if reset:
            best, best_state, best_step, stale = float("inf"), None, -1, 0
            print(f"  {'; '.join(reset)}: early-stopping record reset", flush=True)
        if a.extra_steps > 0:
            a.steps = start_step + a.extra_steps
            print(f"  training {a.extra_steps} steps beyond the checkpoint, to step "
                  f"{a.steps}", flush=True)
    if start_step >= a.steps:
        print(f"  the checkpoint is already at the --steps cap ({a.steps}); nothing left "
              f"to train -- raise --steps to continue", flush=True)
    stopped_at = a.steps

    t_run = t_step = time.time()
    dropped = seen = 0
    for s in range(start_step, a.steps):
        idx = train_idx[torch.randint(0, len(train_idx), (a.batch,)).to(dev)]
        x0 = torch.randn(a.batch, codes.shape[1], dtype=codes.dtype).to(dev)
        # Continuous t, stratified across the batch: one draw per equal sub-interval of
        # [0, 1), which lowers the variance of the loss estimate at no extra cost.
        t = ((torch.arange(a.batch, dtype=codes.dtype) + torch.rand(a.batch)) / a.batch)
        t = t[torch.randperm(a.batch)].to(dev)
        seen += a.batch
        turns = torch.randint(0, 4, (a.batch,)).to(dev) if augment else None
        loss, parts = flow_loss(net, op, data, eta, idx, x0, t, M, tag, mask,
                                train_geoms=train_geoms, turns=turns,
                                return_diag=True, occ_weight=a.occ_weight, occ_eps=occ_eps,
                                fit_weight=a.fit_weight)
        dropped += parts.dropped
        opt.zero_grad(); loss.backward(); opt.step(); ema.update(net)
        now = time.time()
        step_s = now - t_step
        elapsed = elapsed_before + (now - t_run)
        if a.time_budget and s - start_step == 2:
            # projected from three measured steps, early enough to act on
            per = (now - t_run) / 3.0
            proj = elapsed + per * (a.steps - s - 1)
            n_fit = int((a.time_budget * 3600.0 - elapsed) / max(per, 1e-9)) + s + 1
            print(f"  [budget] {per:.1f}s/step -> {_hms(proj)} projected for {a.steps} steps; "
                  f"{_hms(a.time_budget * 3600)} allowed. Largest --steps that fits: {n_fit}",
                  flush=True)
            if proj > a.time_budget * 3600.0:
                # a warning, not an exit: the run checkpoints and resumes, so an overrun
                # costs a restart, not the work
                print(f"  [budget] WARNING: {_hms(proj)} exceeds the budget by "
                      f"{_hms(proj - a.time_budget * 3600)}. This run will be cut short and "
                      f"resumed from {ckpt_path}; pass --steps {n_fit} if you would rather it "
                      f"finish inside one window.", flush=True)
        if (a.log_every and s % a.log_every == 0) or s == a.steps - 1:
            rate = (now - t_run) / (s - start_step + 1)
            print(f"  [{_now()}] step {s:>5}  loss {float(loss.detach()):.5f}  "
                  f"(flow {parts.flow:.5f}, occupancy {parts.occ:.5f}, "
                  f"data fit {parts.fit:.5f})  dropped {dropped}/{seen} states"
                  f"  {step_s:.1f}s/step  elapsed {_hms(elapsed)}  "
                  f"eta {_hms(rate * (a.steps - s - 1))}", flush=True)
            if dropped > 0.5 * seen:
                # a step that drops most of its batch is not training; say so every time
                print(f"  WARNING: {dropped} of {seen} states had no curves so far -- the "
                      f"operator is failing on most endpoint estimates", flush=True)

        stop = False
        if n_val and ((s + 1) % a.val_every == 0 or s == a.steps - 1):
            t_val = time.time()
            with _Swapped(net, ema):
                vl, diag = validate(net, op, data, eta, val_idx, val_x0, val_t,
                                    val_sigma, val_xi, val_zeta, M, tag, mask, a.batch,
                                    occ_weight=a.occ_weight, occ_eps=occ_eps,
                                    fit_weight=a.fit_weight)
            # A collapse check: |g| of the model's own endpoint estimate against the corpus.
            # A flow that has regressed to the mean produces amplitudes smaller than any real
            # body, and a small spread across draws means it produces the same body
            # regardless of x0.
            print(f"  [{_now()}] step {s:>5}  |g|hat {diag.g_mean:.5f} vs corpus "
                  f"{g_ref:.5f} ({100*diag.g_mean/max(g_ref,1e-12):.0f}%), "
                  f"across-draw spread {diag.g_spread:.5f}; val flow {diag.flow:.5f}, "
                  f"occupancy {diag.occ:.5f}, data fit {diag.fit:.5f}, "
                  f"dropped {diag.dropped}/{n_val}", flush=True)
            if vl < best - a.min_delta:
                best, best_step, stale = vl, s, 0
                best_state = ema.state(net)      # ship the weights that were scored
                print(f"  [{_now()}] step {s:>5}  val {vl:.5f}  (best, "
                      f"{time.time()-t_val:.0f}s)", flush=True)
            else:
                stale += 1
                print(f"  [{_now()}] step {s:>5}  val {vl:.5f}  (no improvement on "
                      f"{best:.5f} from step {best_step}, {stale}/{a.patience}, "
                      f"{time.time()-t_val:.0f}s)", flush=True)
                if a.patience and stale >= a.patience:
                    stopped_at = s + 1
                    stop = True

        # After the evaluation, so the checkpoint carries the best state it just found.
        if a.ckpt_every and (stop or (s + 1) % a.ckpt_every == 0 or s == a.steps - 1):
            save_checkpoint(ckpt_path, net, opt, s, best, best_state, best_step, stale,
                            elapsed_before + (time.time() - t_run), meta, ema=ema)
            print(f"  [{_now()}] step {s:>5}  checkpointed to {ckpt_path}", flush=True)
        if stop:
            print(f"  early stop at step {s}: {stale} evaluations without improvement",
                  flush=True)
            break
        t_step = time.time()

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"  restored step {best_step} (val {best:.5f}) after {stopped_at} steps",
              flush=True)
    elif a.ema:
        # no held-out bodies, or no evaluation ever improved: nothing selected a step, so keep
        # the averaged weights
        net.load_state_dict(ema.state(net))
        print(f"  no best checkpoint was selected; keeping the EMA weights over {ema.n} "
              f"steps", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # With the metadata, not as a bare state dict: load_flow_file reads it and
    # check_flow_metadata refuses a flow used against a corpus, a calibration, a phase grid,
    # an extraction resolution or a sensor it was not trained under. Written without it, the
    # finished file was the one link in the chain with no digest -- the instrument carries a
    # rig digest and the corpus records the calibration it was rendered with, and the flow
    # between them carried nothing.
    torch.save({"state_dict": net.state_dict(),
                "meta": {**meta, "steps_trained": int(stopped_at),
                         "best_step": int(best_step), "val": float(best)}}, a.out)
    print(f"[{_now()}] wrote {a.out} after {_hms(elapsed_before + (time.time() - t_run))} "
          f"of training", flush=True)


if __name__ == "__main__":
    main()
