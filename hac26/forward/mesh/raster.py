"""Rasterisation with nvdiffrast: the projections, a thin wrapper around the three
operations, and Otsu's threshold, which the organisers use to turn a frame into the pixel
count.

Every mesh rendered here has its vertices duplicated per face (`flat_faces`), so a per-face
attribute such as the radiance can be given as a per-vertex attribute. nvdiffrast then treats
every edge as a silhouette edge and antialiases all of them, which is correct for a
flat-shaded mesh: adjacent faces share the same geometric edge, so the blend at that edge is
the same whichever face is taken to own it.
"""
from __future__ import annotations

import os

import numpy as np
import torch

__all__ = ["perspective", "look_at", "orthographic", "orthographic_view", "flat_faces",
           "Rasteriser", "otsu_threshold"]


def perspective(fov_y_rad, aspect: float, near, far, device=None) -> torch.Tensor:
    """Standard OpenGL-style projection matrix, which is what nvdiffrast expects, clipping
    at the distances `near` and `far` from the eye. Differentiable in the field of view and
    in the two distances when they are tensors."""
    fov = torch.as_tensor(fov_y_rad, dtype=torch.float32, device=device)
    near = torch.as_tensor(near, dtype=torch.float32, device=device)
    far = torch.as_tensor(far, dtype=torch.float32, device=device)
    f = 1.0 / torch.tan(fov / 2.0)
    zero = torch.zeros((), device=device)
    return torch.stack([
        torch.stack([f / aspect, zero, zero, zero]),
        torch.stack([zero, f, zero, zero]),
        torch.stack([zero, zero, (far + near) / (near - far), (2 * far * near) / (near - far)]),
        torch.stack([zero, zero, -torch.ones((), device=device), zero])])


def look_at(eyes, up=(0.0, 0.0, 1.0), device=None) -> torch.Tensor:
    """View matrices (B, 4, 4) for cameras at `eyes` (B, 3) looking at the origin, with `up`
    as the vertical unless a camera sits on it. Differentiable in `eyes`."""
    eyes = torch.as_tensor(eyes, dtype=torch.float32, device=device)
    squeeze = eyes.dim() == 1
    e = eyes.reshape(-1, 3)
    u = torch.as_tensor(up, dtype=torch.float32, device=device)
    u = u / u.norm()
    f = -e / e.norm(dim=1, keepdim=True)
    on_axis = (f.detach() @ u).abs() > 0.999
    alt = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    ub = torch.where(on_axis[:, None], alt[None], u[None].expand_as(f))
    s = torch.linalg.cross(f, ub); s = s / s.norm(dim=1, keepdim=True)
    v = torch.linalg.cross(s, f)
    rot = torch.stack([s, v, -f], 1)                                     # (B, 3, 3)
    trans = -(rot @ e[..., None])                                        # (B, 3, 1)
    last = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    m = torch.cat([torch.cat([rot, trans], 2), last.expand(len(e), 1, 4)], 1)
    return m[0] if squeeze else m


def orthographic(direction, half_width: float, depth: float, device=None) -> torch.Tensor:
    """Matrix taking body-frame points to clip space for a camera looking along -`direction`
    from far away: clip x = (p . u) / half_width, y = (p . v) / half_width and
    z = -(p . direction) / depth, with (u, v, direction) a right-handed frame. A point closer
    to the source along `direction` gets a smaller z, which nvdiffrast treats as nearer.
    Differentiable in `direction` when it is a tensor."""
    d = torch.as_tensor(direction, dtype=torch.float32, device=device)
    d = d / d.norm()
    a = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    if abs(float(d[2].detach())) > 0.9:
        a = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    u = torch.linalg.cross(a, d); u = u / u.norm()
    v = torch.linalg.cross(d, u)
    rows = torch.stack([u / half_width, v / half_width, -d / depth,
                        torch.zeros(3, dtype=torch.float32, device=device)])
    last_col = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32, device=device)
    return torch.cat([rows, last_col], 1)


def orthographic_view(directions, half_height, depth, aspect: float = 1.0,
                      device=None) -> torch.Tensor:
    """Matrices (B, 4, 4) taking body-frame points to clip space for cameras at infinity in
    the `directions` (B, 3), one per direction.

    This is `orthographic` for a batch, with the frame's two axes scaled separately so that a
    rectangular image covers `half_height` of the body vertically and `half_height * aspect`
    horizontally, as the perspective path's field of view does. The frame is the one `look_at`
    builds, so an orthographic image and a distant perspective image of the same body have the
    same orientation and can be compared pixel by pixel.

    A camera at a finite distance has to place its near and far clip planes around the body,
    and the depth buffer then resolves two nearly coincident faces over a range set by that
    distance rather than by the body. Here the range is `[-depth, depth]` about the body
    centre, whatever the camera, which is why the render of a simulated orthographic channel
    goes through this rather than through a very distant perspective camera.
    """
    d = torch.as_tensor(directions, dtype=torch.float32, device=device).reshape(-1, 3)
    d = d / d.norm(dim=1, keepdim=True)
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d.device)
    y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=d.device)
    # the same fallback axis as `orthographic`: a camera looking along z has no frame built
    # from z, and no released geometry is anywhere near it
    a = torch.where((d[:, 2].abs() > 0.9)[:, None], y[None], z[None])
    u = torch.linalg.cross(a, d); u = u / u.norm(dim=1, keepdim=True)
    v = torch.linalg.cross(d, u)
    hh = torch.as_tensor(half_height, dtype=torch.float32, device=d.device)
    dp = torch.as_tensor(depth, dtype=torch.float32, device=d.device)
    rows = torch.stack([u / (hh * aspect), v / hh, -d / dp, torch.zeros_like(d)], 1)
    last = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32,
                        device=d.device).expand(len(d), 4, 1)
    return torch.cat([rows, last], 2)


def flat_faces(verts: torch.Tensor, faces: torch.Tensor):
    """Duplicate the vertices per face: (3F, 3) positions and (F, 3) faces indexing them, so
    a per-face attribute can be given per vertex. Differentiable in `verts`."""
    fv = verts[faces.long()].reshape(-1, 3)
    ff = torch.arange(fv.shape[0], device=verts.device, dtype=torch.int32).reshape(-1, 3)
    return fv, ff


def _backend(name: str | None, device: str = "cuda"):
    """nvdiffrast, or the pure-torch stand-in when asked for by name, by the
    HAC26_SOFTWARE_RASTER environment variable, or because the device is not a CUDA one.

    nvdiffrast has no CPU path, so on a CPU there is one backend and not a choice; defaulting to
    it there turned every --device cpu run into a missing-module traceback from inside the
    forward model, which reads as a broken install rather than as the one thing it is."""
    if name is None:
        name = ("software" if os.environ.get("HAC26_SOFTWARE_RASTER")
                or not str(device).startswith("cuda") else "nvdiffrast")
    if name == "software":
        from hac26.forward.shared import software_raster
        return software_raster
    if name != "nvdiffrast":
        raise ValueError(f"unknown raster backend {name!r}")
    from ..shared._nvdr import load
    return load()


class Rasteriser:
    """Point-sampled rasterisation with antialiased coverage, plus the per-pixel geometry the
    sensor chain needs. Positions must be float32 on the rasteriser's device."""

    def __init__(self, height: int, width: int, supersample: int = 1, device: str = "cuda",
                 backend: str | None = None):
        self.dr = _backend(backend, device)
        self.h, self.w, self.ss = int(height), int(width), int(supersample)
        self.device = device
        self.ctx = self.dr.RasterizeCudaContext(device=device)
        self._px_cache: dict = {}

    @property
    def resolution(self) -> list:
        return [self.h * self.ss, self.w * self.ss]

    def _unit_grid(self):
        """The pixel grid at unit half-height, and the normalised distance from the optical
        axis, both (1, H, W). Neither depends on the field of view: the grid scales with
        tan(fov / 2), and the radius is divided by its own maximum, which scales with it too.
        So this is cached once per resolution rather than once per field of view."""
        key = (self.h, self.w, self.ss)
        if key not in self._px_cache:
            H, W = self.resolution
            aspect = self.w / self.h
            yy = torch.linspace(1.0, -1.0, H, device=self.device)
            xx = torch.linspace(-aspect, aspect, W, device=self.device)
            gx, gy = torch.meshgrid(xx, yy, indexing="xy")
            g2 = gx ** 2 + gy ** 2
            self._px_cache[key] = (g2[None], (g2.sqrt() / g2.sqrt().max())[None])
        return self._px_cache[key]

    def pixel_geometry(self, fov_y_rad: float):
        """cos(off-axis angle) and normalised radius for every supersampled pixel, (1, H, W).

        The field of view moves with the body, since the camera frames each mesh by its own
        extent, so it takes a different value on nearly every call. Only the cosine depends
        on it, and it is one square root over the grid.
        """
        g2, r = self._unit_grid()
        ty = float(np.tan(fov_y_rad / 2.0))
        return 1.0 / torch.sqrt(1.0 + ty * ty * g2), r

    def render(self, pos_clip: torch.Tensor, faces: torch.Tensor, attr: torch.Tensor,
               antialias: bool = True):
        """Rasterise clip-space positions (B, V, 4) with triangles (F, 3) and interpolate the
        per-vertex attribute (V, C). Returns (image (B, H, W, C), rast). Differentiable in
        `attr` and, through the antialiasing, in `pos_clip`."""
        tri = faces.to(torch.int32).contiguous()
        rast, _ = self.dr.rasterize(self.ctx, pos_clip.contiguous(), tri,
                                    resolution=self.resolution)
        img, _ = self.dr.interpolate(attr.contiguous(), rast, tri)
        if antialias:
            img = self.dr.antialias(img, rast, pos_clip.contiguous(), tri)
        return img, rast


def otsu_threshold(images: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Otsu's threshold of each image in `images` (B, H, W): the grey level that maximises the
    variance between the two classes it separates. Values are binned over [0, 1]. Returns
    (B,), not differentiable."""
    B = images.shape[0]
    x = images.detach().reshape(B, -1).clamp(0.0, 1.0)
    idx = (x * (bins - 1)).round().long()
    flat = idx + bins * torch.arange(B, device=x.device)[:, None]
    hist = torch.bincount(flat.reshape(-1), minlength=B * bins).reshape(B, bins).to(x.dtype)
    levels = torch.arange(bins, device=x.device, dtype=x.dtype) / (bins - 1)
    w0 = hist.cumsum(1)                                   # pixels at or below each level
    total = w0[:, -1:]
    w1 = total - w0
    m0 = (hist * levels).cumsum(1)
    mean_total = m0[:, -1:]
    mu0 = m0 / w0.clamp_min(1)
    mu1 = (mean_total - m0) / w1.clamp_min(1)
    between = w0 * w1 * (mu0 - mu1) ** 2
    between[:, -1] = -1.0                                 # a split with an empty class is not one
    k = between.argmax(1)
    # the threshold sits between bin k and bin k + 1: a value is bright if above it
    return (k.to(x.dtype) + 0.5) / (bins - 1)
