"""A plain-torch stand-in for the three nvdiffrast operations, for tests on machines without
a GPU.

It follows nvdiffrast's conventions: clip-space positions (x, y, z, w), pixel centres at
half-integer positions, the nearest fragment by z / w, and the same output layouts. It does
not antialias: `antialias` returns its input, so position gradients through coverage are zero
here, and `rasterize` runs without recording a graph, since a point-sampled image is
piecewise constant in the positions. Gradients with respect to the attributes are exact. It
is slow and only meant for small meshes and small images. It is selected with
Rasteriser(backend="software") or the environment variable HAC26_SOFTWARE_RASTER=1.
"""
from __future__ import annotations

import torch


class RasterizeCudaContext:
    """Placeholder with the same name as nvdiffrast's context class."""

    def __init__(self, device=None):
        self.device = device


@torch.no_grad()
def rasterize(ctx, pos, tri, resolution, ranges=None, grad_db=True):
    """(rast, rast_db) as nvdiffrast returns them: rast is (B, H, W, 4) holding
    (u, v, z/w, triangle_id + 1), zero where no triangle covers the pixel."""
    if pos.dim() == 2:
        pos = pos[None]
    B, V, _ = pos.shape
    H, W = resolution
    dev = pos.device
    ndc = pos[..., :3] / pos[..., 3:4].clamp_min(1e-12)                    # (B, V, 3)
    p = ndc[:, tri.long()]                                                   # (B, T, 3, 3)
    # pixel centres in NDC; row 0 is y = -1 as in OpenGL
    xs = (torch.arange(W, device=dev, dtype=pos.dtype) + 0.5) / W * 2 - 1
    ys = (torch.arange(H, device=dev, dtype=pos.dtype) + 0.5) / H * 2 - 1
    py, px = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack([px.reshape(-1), py.reshape(-1)], -1)                  # (HW, 2)
    rast = torch.zeros(B, H * W, 4, dtype=pos.dtype, device=dev)
    for b in range(B):
        x0, y0 = p[b, :, 0, 0], p[b, :, 0, 1]
        x1, y1 = p[b, :, 1, 0], p[b, :, 1, 1]
        x2, y2 = p[b, :, 2, 0], p[b, :, 2, 1]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)                # (T,)
        ok = area.abs() > 1e-14
        best_z = torch.full((H * W,), float("inf"), dtype=pos.dtype, device=dev)
        best_t = torch.full((H * W,), -1, dtype=torch.long, device=dev)
        best_uv = torch.zeros(H * W, 2, dtype=pos.dtype, device=dev)
        chunk = 4096
        for i in range(0, H * W, chunk):
            q = pix[i:i + chunk]                                             # (n, 2)
            qx, qy = q[:, :1], q[:, 1:]
            # barycentric weights of the pixel centre in every triangle
            w0 = ((x1 - qx) * (y2 - qy) - (x2 - qx) * (y1 - qy)) / area     # (n, T)
            w1 = ((x2 - qx) * (y0 - qy) - (x0 - qx) * (y2 - qy)) / area
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & ok
            z = w0 * p[b, :, 0, 2] + w1 * p[b, :, 1, 2] + w2 * p[b, :, 2, 2]
            z = torch.where(inside, z, torch.full_like(z, float("inf")))
            zmin, t = z.min(1)
            hit = torch.isfinite(zmin)
            best_z[i:i + chunk] = torch.where(hit, zmin, best_z[i:i + chunk])
            best_t[i:i + chunk] = torch.where(hit, t, best_t[i:i + chunk])
            ar = torch.arange(len(q), device=dev)
            uv = torch.stack([w0[ar, t], w1[ar, t]], -1)   # nvdiffrast's (u, v): the weights
            best_uv[i:i + chunk] = torch.where(hit[:, None], uv, best_uv[i:i + chunk])  # of vertices 0 and 1
        cov = best_t >= 0
        rast[b, :, 0] = best_uv[:, 0] * cov
        rast[b, :, 1] = best_uv[:, 1] * cov
        rast[b, :, 2] = torch.where(cov, best_z, torch.zeros_like(best_z))
        rast[b, :, 3] = (best_t + 1).to(pos.dtype) * cov
    rast = rast.reshape(B, H, W, 4)
    return rast, torch.zeros(B, H, W, 4, dtype=pos.dtype, device=dev)


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """Interpolated attributes (B, H, W, C); zero on uncovered pixels."""
    if attr.dim() == 2:
        attr = attr[None]
    B, H, W, _ = rast.shape
    tid = rast[..., 3].long() - 1                                            # (B, H, W)
    cov = tid >= 0
    tid = tid.clamp_min(0)
    corners = tri.long()[tid]                                                # (B, H, W, 3)
    u, v = rast[..., 0:1], rast[..., 1:2]
    a = attr if attr.shape[0] == B else attr.expand(B, -1, -1)

    def gather(k):
        idx = corners[..., k].reshape(B, -1)                                 # (B, HW)
        return torch.gather(a, 1, idx[..., None].expand(-1, -1, a.shape[-1])).reshape(B, H, W, -1)
    out = u * gather(0) + v * gather(1) + (1 - u - v) * gather(2)
    out = out * cov[..., None]
    return out, torch.zeros(B, H, W, 0, dtype=out.dtype, device=out.device)


def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    """No antialiasing in the software backend."""
    return color


def antialias_construct_topology_hash(tri):
    return None
