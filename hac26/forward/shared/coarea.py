"""Exact derivatives of the two thresholded reductions, by the coarea formula.

The reductions are discontinuous in the image: a pixel enters the sum only once its value
crosses tau. The coarea formula turns the derivative of such a threshold into an integral over
the level set,

    d/dtheta INT_{u > tau} g = INT_{u = tau} (g / |grad u|) du/dtheta dl

so the gradient is carried by the contour at level tau, weighted by 1/|grad u|. The contour of
the piecewise-linear interpolant of the image is traced by marching squares, and each segment
scatters its length divided by the local gradient magnitude onto the four pixels around its
midpoint.

Everything is batched over images and runs in torch on whatever device the image is on.
`threshold_count` and `threshold_sum` take a batch of images (..., H, W) and one threshold per
image (broadcastable to the batch shape) and return one number per image. Both are
differentiable in the image and in the threshold: d/dtau of the count is minus the total
contour weight, and of the sum minus tau times it.
"""
from __future__ import annotations

import torch

__all__ = ["contour_weights", "threshold_count", "threshold_sum"]

# Marching squares. A cell has corners a (top-left), b (top-right), c (bottom-right),
# d (bottom-left) and edges 0 top, 1 right, 2 bottom, 3 left. The case index is
# (a > 0) << 3 | (b > 0) << 2 | (c > 0) << 1 | (d > 0). Each case has up to two segments,
# each joining two edges; -1 pads. The two saddle cases (5 and 10) have two readings and
# are resolved by the sign of the cell's mean value.
_NONE = (-1, -1)
_SEGMENTS = [
    [_NONE, _NONE],          # 0
    [(3, 2), _NONE],         # 1: d
    [(2, 1), _NONE],         # 2: c
    [(3, 1), _NONE],         # 3: c d
    [(0, 1), _NONE],         # 4: b
    [(0, 1), (3, 2)],        # 5: b d, centre negative: separate b and d
    [(0, 2), _NONE],         # 6: b c
    [(3, 0), _NONE],         # 7: b c d
    [(3, 0), _NONE],         # 8: a
    [(0, 2), _NONE],         # 9: a d
    [(3, 0), (2, 1)],        # 10: a c, centre negative: separate a and c
    [(0, 1), _NONE],         # 11: a c d
    [(3, 1), _NONE],         # 12: a b
    [(2, 1), _NONE],         # 13: a b d
    [(3, 2), _NONE],         # 14: a b c
    [_NONE, _NONE],          # 15
]
_SEGMENTS_CENTRE_POSITIVE = {5: [(3, 0), (2, 1)],     # positive region joins b and d
                             10: [(0, 1), (3, 2)]}    # positive region joins a and c


def _tables(device):
    t = torch.tensor(_SEGMENTS, dtype=torch.long, device=device)            # (16, 2, 2)
    tp = t.clone()
    for k, seg in _SEGMENTS_CENTRE_POSITIVE.items():
        tp[k] = torch.tensor(seg, dtype=torch.long, device=device)
    return t, tp


def _grad_mag(u: torch.Tensor) -> torch.Tensor:
    """|grad u| by central differences (one-sided at the border), in units of 1/pixel."""
    gy = torch.empty_like(u)
    gy[..., 1:-1, :] = 0.5 * (u[..., 2:, :] - u[..., :-2, :])
    gy[..., 0, :] = u[..., 1, :] - u[..., 0, :]
    gy[..., -1, :] = u[..., -1, :] - u[..., -2, :]
    gx = torch.empty_like(u)
    gx[..., :, 1:-1] = 0.5 * (u[..., :, 2:] - u[..., :, :-2])
    gx[..., :, 0] = u[..., :, 1] - u[..., :, 0]
    gx[..., :, -1] = u[..., :, -1] - u[..., :, -2]
    return torch.sqrt(gx ** 2 + gy ** 2)


def contour_weights(u: torch.Tensor, tau: torch.Tensor, eps: float = 1e-8):
    """Scatter weights of the level set of each image at its threshold.

    `u` is (B, H, W), `tau` is (B,). Returns (batch, rows, cols, w), flat over every segment
    end of every image, such that for a perturbation field d

        INT_{u = tau} (d / |grad u|) dl  ~=  sum_k w_k * d[batch_k, rows_k, cols_k]

    Each marching-squares segment contributes its length divided by the gradient magnitude at
    its midpoint (floored at `eps`), spread bilinearly onto the four pixels around the
    midpoint.
    """
    B, H, W = u.shape
    dev = u.device
    s = u - tau.reshape(B, 1, 1)
    a, b = s[:, :-1, :-1], s[:, :-1, 1:]
    d, c = s[:, 1:, :-1], s[:, 1:, 1:]
    case = ((a > 0).long() << 3) | ((b > 0).long() << 2) | ((c > 0).long() << 1) | (d > 0).long()

    # crossing point of each cell edge, as (row, col) in pixel units; only used where the
    # case table asks for that edge, so the values elsewhere do not matter
    def frac(p, q):
        return p / torch.where((p - q).abs() > 1e-30, p - q, torch.full_like(p, 1e-30))
    ci = torch.arange(H - 1, device=dev, dtype=u.dtype).reshape(1, H - 1, 1)
    cj = torch.arange(W - 1, device=dev, dtype=u.dtype).reshape(1, 1, W - 1)
    zero = torch.zeros_like(a)
    edge_r = torch.stack([ci + zero, ci + frac(b, c), ci + 1 + zero, ci + frac(a, d)], -1)
    edge_c = torch.stack([cj + frac(a, b), cj + 1 + zero, cj + frac(d, c), cj + zero], -1)
    # (B, H-1, W-1, 4)

    table, table_pos = _tables(dev)
    centre_pos = (a + b + c + d) > 0
    seg = torch.where(centre_pos[..., None, None], table_pos[case], table[case])   # (B,H-1,W-1,2,2)
    live = seg[..., 0] >= 0                                                        # (B,H-1,W-1,2)
    idx = torch.nonzero(live, as_tuple=True)                                       # cells x segment
    if idx[0].numel() == 0:
        e = torch.zeros(0, dtype=torch.long, device=dev)
        return e, e, e, torch.zeros(0, dtype=u.dtype, device=dev)
    bi, ri, cj_, si = idx
    e0 = seg[bi, ri, cj_, si, 0]
    e1 = seg[bi, ri, cj_, si, 1]
    r0, c0 = edge_r[bi, ri, cj_, e0], edge_c[bi, ri, cj_, e0]
    r1, c1 = edge_r[bi, ri, cj_, e1], edge_c[bi, ri, cj_, e1]
    length = torch.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    mr, mc = 0.5 * (r0 + r1), 0.5 * (c0 + c1)

    g = _grad_mag(u)
    pr = mr.floor().clamp(0, H - 2).long()
    pc = mc.floor().clamp(0, W - 2).long()
    fr, fc = mr - pr, mc - pc
    gm = (g[bi, pr, pc] * (1 - fr) * (1 - fc) + g[bi, pr + 1, pc] * fr * (1 - fc)
          + g[bi, pr, pc + 1] * (1 - fr) * fc + g[bi, pr + 1, pc + 1] * fr * fc)
    w = length / gm.clamp_min(eps)

    rows = torch.stack([pr, pr + 1, pr, pr + 1], 0).reshape(-1)
    cols = torch.stack([pc, pc, pc + 1, pc + 1], 0).reshape(-1)
    wts = torch.stack([w * (1 - fr) * (1 - fc), w * fr * (1 - fc),
                       w * (1 - fr) * fc, w * fr * fc], 0).reshape(-1)
    batch = bi.repeat(4)
    return batch, rows, cols, wts


def _prepare(u: torch.Tensor, tau):
    """Flatten leading dims to one batch axis and broadcast tau to it."""
    lead = u.shape[:-2]
    uf = u.reshape(-1, *u.shape[-2:])
    t = torch.as_tensor(tau, dtype=u.dtype, device=u.device)
    tf = t.expand(lead).reshape(-1) if t.dim() > 0 else t.expand(uf.shape[0])
    return uf, tf, lead


class _ThresholdCount(torch.autograd.Function):
    """N(tau) = sum 1[u > tau] per image, forward exact, backward by coarea."""

    @staticmethod
    def forward(ctx, u, tau):
        with torch.no_grad():
            b, r, c, w = contour_weights(u.detach(), tau.detach())
        ctx.shape = u.shape
        ctx.save_for_backward(b, r, c, w)
        return (u > tau.reshape(-1, 1, 1)).sum((-2, -1)).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        b, r, c, w = ctx.saved_tensors
        grad_u = torch.zeros(ctx.shape, dtype=g.dtype, device=g.device)
        grad_tau = torch.zeros(ctx.shape[0], dtype=g.dtype, device=g.device)
        if len(b):
            gw = g[b] * w
            grad_u.index_put_((b, r, c), gw, accumulate=True)
            grad_tau.index_put_((b,), -gw, accumulate=True)
        return grad_u, grad_tau


class _ThresholdSum(torch.autograd.Function):
    """I(tau) = sum u 1[u > tau] per image: the interior term plus tau times the boundary
    term."""

    @staticmethod
    def forward(ctx, u, tau):
        with torch.no_grad():
            b, r, c, w = contour_weights(u.detach(), tau.detach())
        mask = (u > tau.reshape(-1, 1, 1)).to(u.dtype)
        ctx.save_for_backward(mask, tau, b, r, c, w)
        return (u * mask).sum((-2, -1))

    @staticmethod
    def backward(ctx, g):
        mask, tau, b, r, c, w = ctx.saved_tensors
        grad_u = g.reshape(-1, 1, 1) * mask
        grad_tau = torch.zeros_like(tau)
        if len(b):
            gw = g[b] * tau[b] * w
            grad_u = grad_u.clone()
            grad_u.index_put_((b, r, c), gw, accumulate=True)
            grad_tau.index_put_((b,), -gw, accumulate=True)
        return grad_u, grad_tau


def threshold_count(u: torch.Tensor, tau) -> torch.Tensor:
    """Pixel count above tau for each image in `u` (..., H, W), with the exact coarea
    derivative in the image and in tau. `tau` broadcasts to the batch shape."""
    uf, tf, lead = _prepare(u, tau)
    return _ThresholdCount.apply(uf, tf).reshape(lead)


def threshold_sum(u: torch.Tensor, tau) -> torch.Tensor:
    """Summed value above tau for each image in `u` (..., H, W), with the exact coarea
    derivative in the image and in tau. `tau` broadcasts to the batch shape."""
    uf, tf, lead = _prepare(u, tau)
    return _ThresholdSum.apply(uf, tf).reshape(lead)
