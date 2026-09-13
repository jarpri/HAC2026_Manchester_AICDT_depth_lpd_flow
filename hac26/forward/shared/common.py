"""Shared contract for the forward models in this package, and the two reductions.

Each model answers one question: given a body and a viewing/lighting geometry, what are the
two numbers the challenge measures. How the body is parameterised is the model's own concern.

    render(shape, view, sun, **kw) -> (intensity, lit_area)

`view` is omega_c, the direction from the body to the camera; `sun` is the direction from the
body to the source. Both are unit vectors already carried into the body frame by
hac26.conventions.to_body, which is where the rotation sense is defined.

The two reductions are different integrals:

    intensity  I = sum over pixels of val * 1[val > tau_I]      a sum of values
    lit area   N = count of pixels with val > tau_B             a count of pixels

I carries the radiance, N carries only the geometry of the lit region.

The shader emits radiance, not radiance times mu. A Lambertian facet's radiance is
(rho/pi) mu0 and carries no mu; the viewing obliquity enters through the projected area of
the pixel, mu dA, which the image-plane sum already supplies. With val = mu0,

    I = INT mu0+ mu+ dA        Lambert kernel
    N = INT_lit mu+ dA         binary kernel, mu+ alone

which is the pair the convex analytic operator uses.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["reduce_frame", "camera_basis", "pixel_rays", "curves_over_psi"]


def reduce_frame(val: torch.Tensor, px: float, tau_i: float = 0.0,
                 tau_b: float = 0.02) -> tuple:
    """The two reductions from a per-pixel value map. `px` is the area of one pixel."""
    inten = (val * (val > tau_i)).sum() * px
    area = (val > tau_b).sum().to(val.dtype) * px
    return inten, area


def camera_basis(view: torch.Tensor) -> tuple:
    """An orthonormal image basis for an orthographic camera looking along -view."""
    up = torch.tensor([0.0, 0.0, 1.0], device=view.device, dtype=view.dtype)
    if abs(float(view @ up)) > 0.99:
        up = torch.tensor([0.0, 1.0, 0.0], device=view.device, dtype=view.dtype)
    ex = torch.cross(up, view, dim=0); ex = ex / ex.norm()
    ey = torch.cross(view, ex, dim=0); ey = ey / ey.norm()
    return ex, ey


def pixel_rays(view: torch.Tensor, extent: float, res: int) -> tuple:
    """Ray origins on a plane in front of the body, all travelling along -view.

    Rays start on the camera side and travel along -view. Starting at -extent*view and
    marching along +view renders the far surface instead, which appears as opposed cameras
    exchanging values.
    """
    ex, ey = camera_basis(view)
    a = torch.linspace(-extent, extent, res, device=view.device, dtype=view.dtype)
    gx, gy = torch.meshgrid(a, a, indexing="ij")
    o = (gx.reshape(-1, 1) * ex + gy.reshape(-1, 1) * ey) + 2.0 * extent * view
    return o, -view, (2.0 * extent / res) ** 2


def curves_over_psi(render_fn, shape, cam_dir, sun_lab, psi, psi0: float = 0.0, **kw):
    """Both curves over a full rotation, for one camera.

    The body spins about z and the source is fixed in the lab, so both directions are carried
    into the body frame at each phase.
    """
    from hac26.conventions import to_body
    cam = to_body(np.asarray(cam_dir, dtype=float), np.asarray(psi, dtype=float), psi0)
    sun = to_body(np.asarray(sun_lab, dtype=float), np.asarray(psi, dtype=float), psi0)
    I, N = [], []
    for j in range(len(psi)):
        i_, n_ = render_fn(shape,
                           torch.tensor(cam[j], dtype=torch.float32),
                           torch.tensor(sun[j], dtype=torch.float32), **kw)
        I.append(float(i_)); N.append(float(n_))
    return np.asarray(I), np.asarray(N)
