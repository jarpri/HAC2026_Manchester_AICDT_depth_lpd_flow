"""Learned primal-dual network on the convex operator (Adler & Oektem, arXiv:1707.06474,
Alg. 3).

Unrolled scheme, I iterations with their own parameters:
    h_i = h_{i-1} + Gamma_i( h_{i-1}, T(softplus(f_{i-1}^{(2)})), d, tags, mask )
    f_i = f_{i-1} + Lambda_i( f_{i-1}, [dT(softplus(f_{i-1}^{(1)}))]^T h_i^{(1)}, coords )
    return p = softplus(f_I^{(1)}) / sum(...)
where T = N_eps o A is the convex photometric operator (forward/convex_egi.py) and the
derivative adjoint is the closed form A^T o DN^T, chained with softplus' = sigmoid.

Design choices:
- Dual blocks convolve along the frame axis only, with circular padding, because a curve
  is one full revolution.
- Primal blocks are 2D CNNs on the (theta, phi) grid of normals, circular in phi.
- Each curve carries conditioning channels ("tags"): azimuth/360, elevation/90, phase/180,
  is_binary; plus the availability mask for curves missing from the data.
- The network predicts the scale-free EGI direction p = g/sum(g), since mean-normalised
  curves carry no absolute scale; the size is fixed downstream by posing to z in [-1, 1].
- The last conv of each block is zero-initialised, so every update starts at zero.

Optional parts, all off by default: conditioning on the a-priori bounding radius (r_cond), a
support-function head, and a low-rank occlusion gate; see LPDNet.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hac26.forward.convex_egi import ConvexPhotometricOperator


def make_tags(cameras: list, curve_types: list) -> torch.Tensor:
    """Per-curve conditioning channels, shape (C, 4): azimuth/360, elevation/90, phase/180,
    is_binary."""
    rows = []
    for cam, ctype in zip(cameras, curve_types):
        rows.append([cam.azimuth_deg / 360.0,
                     cam.elevation_deg / 90.0,
                     cam.phase_angle_deg / 180.0,
                     1.0 if ctype == "binary" else 0.0])
    return torch.tensor(rows, dtype=torch.float32)  # (C, 4)


class DualBlock(nn.Module):
    """Three 1x3 convolutions along the frame axis with circular padding,
    (B, n_in, C, m) -> (B, n_dual, C, m). The caller adds the output to its state; the last
    conv starts at zero so the update starts at zero."""

    def __init__(self, n_dual: int, n_in: int, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(n_in, ch, (1, 3))
        self.c2 = nn.Conv2d(ch, ch, (1, 3))
        self.c3 = nn.Conv2d(ch, n_dual, (1, 3))
        self.a1, self.a2 = nn.PReLU(ch), nn.PReLU(ch)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (1, 1, 0, 0), mode="circular")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.c1(self._pad(x)))
        x = self.a2(self.c2(self._pad(x)))
        return self.c3(self._pad(x))


class PrimalBlock(nn.Module):
    """Three 3x3 convolutions on the (theta, phi) grid, circular in phi and replicate-padded
    in theta, (B, n_in, n_theta, n_phi) -> (B, n_primal, n_theta, n_phi). The last conv
    starts at zero."""

    def __init__(self, n_primal: int, n_in: int, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(n_in, ch, 3)
        self.c2 = nn.Conv2d(ch, ch, 3)
        self.c3 = nn.Conv2d(ch, n_primal, 3)
        self.a1, self.a2 = nn.PReLU(ch), nn.PReLU(ch)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 0, 0), mode="circular")      # phi
        return F.pad(x, (0, 0, 1, 1), mode="replicate")  # theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.c1(self._pad(x)))
        x = self.a2(self.c2(self._pad(x)))
        return self.c3(self._pad(x))


class LPDNet(nn.Module):
    """The unrolled network of the module docstring. forward() returns (p, g), or
    (p, g, h) with a support head, where p is the scale-free EGI direction, g the
    unnormalised facet areas and h the support function on the normal grid."""

    def __init__(self, op: ConvexPhotometricOperator, n_theta: int, n_phi: int,
                 cameras: list, curve_types: list, n_iter: int = 15,
                 n_primal: int = 7, n_dual: int = 7, ch: int = 48,
                 support_head: bool = False, r_cond: bool = False,
                 gate_rank: int = 0, gate_bias: float = 3.0):
        super().__init__()
        assert op.n == n_theta * n_phi
        self.op = op
        self.n_theta, self.n_phi = n_theta, n_phi
        self.n_iter, self.n_primal, self.n_dual = n_iter, n_primal, n_dual
        self.support_head = support_head
        # Optional conditioning on the a-priori bounding radius R. Per-curve mean
        # normalization removes the cross-camera amplitudes that carry the body's width
        # relative to its height, so log R is given to the primal blocks as an extra
        # input channel.
        self.r_cond = r_cond
        self.register_buffer("tags", make_tags(cameras, curve_types))  # (C,4)
        th = (np.arange(n_theta) + 0.5) * np.pi / n_theta
        coords = np.stack([np.repeat(np.cos(th)[:, None], n_phi, 1),
                           np.repeat(np.sin(th)[:, None], n_phi, 1)])
        self.register_buffer("coords", torch.as_tensor(coords, dtype=torch.float32))
        n_tag = 4
        self.duals = nn.ModuleList(
            [DualBlock(n_dual, n_dual + 2 + n_tag + 1, ch) for _ in range(n_iter)])
        n_r = 1 if r_cond else 0
        n_back = max(1, gate_rank)
        self.primals = nn.ModuleList(
            [PrimalBlock(n_primal, n_primal + n_back + 2 + n_r, ch) for _ in range(n_iter)])
        # Optional support-function head. The unroll works in EGI space, where the
        # operator lives, but the prediction is the support function h(u): any positive
        # h gives a valid convex body by half-space intersection, and an average of
        # support functions is again a support function, which an average of polytope
        # EGIs is not.
        self.head_h = (PrimalBlock(1, n_primal + 2 + n_r, ch) if support_head else None)

        # --- occlusion gate ---------------------------------------------------------
        # Self-occlusion and cast shadow multiply the operator entrywise. The gate models
        # that factor as a rank-R product: R shape-space factors w_r live in primal
        # channels 1..R and a data-space factor d1 (B, R, C, m) is predicted by a dual-side
        # CNN, so the forward is
        #
        #     y = N( sum_r d1^(r) * (A w_r) )
        #
        # and A itself, with its exact adjoint, stays fixed.
        #
        # d1 is a sigmoid, so it stays in (0, 1) and y stays positive; a negative y would
        # push normalize() onto its eps floor and dn_adjoint onto its 1/eps branch. The
        # gate CNN's output bias starts at +gate_bias for rank 0 and at a small positive
        # value for the other ranks, chosen so the ranks' initial gates sum to one.
        self.gate_rank = gate_rank
        self.gate_bias = gate_bias
        if gate_rank:
            assert n_primal >= 1 + gate_rank, "need a primal channel per gate rank"
            self.gate_d1 = nn.ModuleList(
                [DualBlock(gate_rank, n_dual + 2 + n_tag + 1, ch) for _ in range(n_iter)])
            budget = 1.0 / (1.0 + math.exp(gate_bias))        # sigmoid(-gate_bias)
            t = budget / max(1, gate_rank - 1)                # per-rank share
            corr = math.log(t / (1.0 - t))
            for blk in self.gate_d1:
                with torch.no_grad():
                    blk.c3.bias.fill_(corr)
                    blk.c3.bias[0] = gate_bias

    def _gated_raw(self, w, d1):
        """sum_r d1^(r) * (A w_r):  w (B,R,N), d1 (B,R,C,m) -> (B,C,m)."""
        y = torch.einsum("cmn,brn->brcm", self.op.A, w)
        return (d1 * y).sum(1)

    def _gated_back(self, g1, h0, mask, d1):
        """Adjoint of the gated forward at g1 applied to h0, per rank: (B,R,N). The same
        fixed A is used forward and backward; only the gate weights differ per rank."""
        R = d1.shape[1]
        raw = self._gated_raw(g1[:, None].expand(-1, R, -1), d1)
        r = self.op.dn_adjoint(raw, h0)
        if mask is not None:
            r = r * mask[..., None]
        return torch.einsum("cmn,brcm->brn", self.op.A, d1 * r[:, None])

    def forward(self, d: torch.Tensor, mask: torch.Tensor,
                log_r: torch.Tensor | None = None) -> tuple:
        """d: (B, C, m) normalised curves, zero where missing; mask: (B, C) in {0, 1};
        log_r: (B,) log of the a-priori bounding radius, required when r_cond=True."""
        B, C, m = d.shape
        nt, nph = self.n_theta, self.n_phi
        f = d.new_zeros(B, self.n_primal, nt, nph)
        h = d.new_zeros(B, self.n_dual, C, m)
        tags = self.tags.T[None, :, :, None].expand(B, 4, C, m)
        mch = mask[:, None, :, None].expand(B, 1, C, m)
        coords = self.coords[None].expand(B, 2, nt, nph)
        if self.r_cond:
            if log_r is None:
                raise ValueError("r_cond=True requires log_r")
            rch = log_r.reshape(B, 1, 1, 1).expand(B, 1, nt, nph).to(d.dtype)
            coords = torch.cat([coords, rch], dim=1)
        R = self.gate_rank
        for i in range(self.n_iter):
            if R:
                # mch appears twice on purpose: gate_d1 has the same input width as the
                # dual blocks, but y2 is not available yet (it is computed from d1), so
                # the mask fills that slot. The shipped checkpoint was trained with this
                # layout.
                d1 = torch.sigmoid(
                    self.gate_d1[i](torch.cat([h, d[:, None], mch, tags, mch], dim=1)))
                w = F.softplus(f[:, 1:1 + R].reshape(B, R, -1))
                y2 = self.op.normalize(self._gated_raw(w, d1)) * mask[:, :, None]
            else:
                g2 = F.softplus(f[:, 1].reshape(B, -1))
                y2 = self.op(g2) * mask[:, :, None]
            dual_in = torch.cat([h, y2[:, None], d[:, None], tags, mch], dim=1)
            h = h + self.duals[i](dual_in)
            x1 = f[:, 0].reshape(B, -1)
            g1 = F.softplus(x1)
            if R:
                back = self._gated_back(g1, h[:, 0], mask, d1) * torch.sigmoid(x1)[:, None]
                back = back.reshape(B, R, nt, nph)
            else:
                back = self.op.deriv_adjoint(g1, h[:, 0], mask).reshape(B, 1, nt, nph) \
                    * torch.sigmoid(x1).reshape(B, 1, nt, nph)
            primal_in = torch.cat([f, back, coords], dim=1)
            f = f + self.primals[i](primal_in)
        g_out = F.softplus(f[:, 0].reshape(B, -1))
        p = g_out / g_out.sum(dim=1, keepdim=True).clamp_min(1e-12)
        if self.head_h is None:
            return p, g_out
        # softplus keeps h > 0, so the origin is inside the body; the +1 makes the initial
        # prediction (zero-initialised last conv) a sphere of radius softplus(1).
        h = F.softplus(self.head_h(torch.cat([f, coords], dim=1))[:, 0] + 1.0)
        return p, g_out, h.reshape(B, -1)
