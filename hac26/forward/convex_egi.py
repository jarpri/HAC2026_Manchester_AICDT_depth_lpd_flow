"""Convex forward model: T = N o A, linear in the facet areas of the extended Gaussian image.

A is the photometric tensor on the normal grid,
    A[c, k, i] = s_c( <R3(psi_k) u_i, omega_c>, <R3(psi_k) u_i, OMEGA0> ),
where s_c is the kernel of curve c (see `kernel`), zero where the normal faces away from the
camera or the light. Per-curve constant factors cancel under normalization and are omitted.
N is the per-curve mean normalization N(y) = y / mean(y), implemented with a guard
N_eps(y) = y / max(mean(y), eps); it equals N whenever mean(y) >= eps.

Closed forms used by the LPD:
    DN(y)v      = v/mbar - y*mean(v)/mbar^2
    DN(y)^T w   = w/mbar - (<y,w>/(m*mbar^2)) * ones
    [d(N o A)(g)]^T = A^T o DN(Ag)^T
"""
from __future__ import annotations

import numpy as np

from hac26.conventions import TRANSFER_EXPONENT

# How much of a frame the body is set in, as a multiple of its largest silhouette area. It
# enters only through Otsu's criterion, which weighs the body against the dark part of the
# frame, and the level it gives moves by under a hundredth of the illumination cosine across
# any framing that keeps a body of this shape inside the frame at every phase.
FRAME_OVER_BODY = 2.5

# Which photometric law builds the tensor. MEASURED is the rig's own; LEGACY is what this
# repository used before it was measured and exists only so that a checkpoint trained against
# that tensor can be loaded and run as the estimator it was trained to be.
MEASURED = "measured"
LEGACY = "lommel_seeliger"
from hac26.geometry import OMEGA0, NormalGrid, body_frame_dirs, psi_grid


def legacy_kernel(mu: np.ndarray, mu0: np.ndarray, curve_type: str,
                  c_lambert: float = 0.1, ls_weight: float = 1.0,
                  tau: float = 0.0) -> np.ndarray:
    """The photometric kernel this repository used before the rig's own was measured:
    Lommel-Seeliger plus Lambert for intensity, and for the count a threshold on the product
    of the two cosines.

    It is not what the rig measures, by a factor of fifty on a body where the facet sum and a
    ray cast must agree, and nothing new should be built on it. It is kept because
    `models/lpd_convex.pt` is an unrolled scheme whose weights were fitted against the tensor
    this kernel builds, so the operator has to be reconstructible to load that checkpoint at
    all; a network trained against one operator and run against another is not the estimator
    that was trained.
    """
    rad = np.where((mu > 0.0) & (mu0 > 0.0), mu * mu0, 0.0)
    lit = rad > tau
    if curve_type == "intensity":
        den = np.where(lit, mu + mu0, 1.0)
        return np.where(lit, ls_weight * mu * mu0 / den + c_lambert * mu * mu0, 0.0)
    if curve_type == "binary":
        return np.where(lit, mu, 0.0)
    raise ValueError(curve_type)


def kernel(mu: np.ndarray, mu0: np.ndarray, curve_type: str,
           gamma: float = TRANSFER_EXPONENT, threshold: float = 0.0) -> np.ndarray:
    """Per-normal weight of one curve type, from the cosines to the camera (mu) and to the
    light (mu0).

        intensity   mu * mu0**gamma
        binary      mu  where mu0 > threshold

    Both follow from what the rig measures. A Lambertian facet leaves a radiance proportional
    to mu0 that does not depend on the direction it is seen from; it covers mu times its own
    area of the image; and the stored value of each of its pixels is that radiance through a
    transfer of exponent gamma. Summing the stored values over the facet therefore gives
    mu * mu0**gamma, and counting the pixels above a level gives mu wherever the radiance
    clears that level, which is a condition on mu0 alone.

    The emission cosine enters only as the area a facet covers. A kernel that thresholds the
    product mu*mu0 is thresholding a brightness that varies with the direction of view, which
    a Lambertian surface's does not, and an intensity kernel of the Lommel-Seeliger form is a
    scattering law this rig does not have. On the convex hull of a public body, where the
    facet sum and a ray cast of the same body must agree exactly, these kernels reproduce the
    cast to about a thousandth and those two do not.

    `threshold` is a level of mu0, one per curve; `otsu_threshold` derives it the way the
    organisers do, from the first frame.
    """
    lit = (mu > 0.0) & (mu0 > 0.0)
    if curve_type == "intensity":
        return np.where(lit, mu * np.abs(mu0) ** gamma, 0.0)
    if curve_type == "binary":
        return np.where(lit & (mu0 > threshold), mu, 0.0)
    raise ValueError(curve_type)


def otsu_threshold(mu: np.ndarray, mu0: np.ndarray, areas: np.ndarray,
                   frame_area: float, gamma: float = TRANSFER_EXPONENT,
                   bins: int = 256) -> float:
    """The level of mu0 that Otsu's criterion puts on one frame of this body, without
    rendering it.

    The organisers threshold each video at Otsu's level of its own first frame and hold that
    level for the rotation, so the level is not free and must not be fitted. Otsu's criterion
    needs only the histogram of the frame, and that histogram is known from the facets: a lit
    and visible facet contributes its projected area at the stored value mu0**gamma, and the
    rest of the frame contributes at zero. The returned level is in mu0 rather than in stored
    value, which is where the kernel wants it.
    """
    lit = (mu > 0.0) & (mu0 > 0.0)
    val = np.zeros_like(mu0)
    val[lit] = np.abs(mu0[lit]) ** gamma
    w = np.where(lit, areas * mu, 0.0)
    idx = np.clip(np.round(val * (bins - 1)).astype(int), 0, bins - 1)
    hist = np.bincount(idx, weights=w, minlength=bins).astype(float)
    hist[0] += max(frame_area - w.sum(), 0.0)               # the unlit part of the frame
    lev = np.arange(bins) / (bins - 1)
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * lev)
    mu_0 = m0 / np.maximum(w0, 1e-12)
    mu_1 = (m0[-1] - m0) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (mu_0 - mu_1) ** 2
    k = int(np.argmax(between[:-1]))
    return float(lev[k] ** (1.0 / gamma))


def build_A(normals: np.ndarray, cameras: list, m: int, curve_types: list,
            gamma: float = TRANSFER_EXPONENT, thresholds=None, sigma: float = 1.0,
            delta: float = 1.0, psi0: float = 0.0, law: str = MEASURED,
            c_lambert: float = 0.1) -> np.ndarray:
    """Photometric tensor A of shape (n_curves, m, N) for the given normals.

    `cameras` and `curve_types` are parallel lists, one entry per output curve, and
    `thresholds` gives the level of the illumination cosine for each of them, which only the
    binary ones use. stack_A() builds the full stack of every camera as intensity and then as
    binary.
    """
    psi = psi_grid(m, sigma=sigma, psi0=psi0)
    v0 = body_frame_dirs(OMEGA0, psi)                    # (m,3)
    mu0 = normals @ v0.T                                 # (N,m)
    if thresholds is None:
        thresholds = np.zeros(len(curve_types))
    rows = []
    for cam, ctype, c in zip(cameras, curve_types, np.asarray(thresholds, float)):
        v = body_frame_dirs(cam.omega(delta=delta), psi)  # (m,3)
        mu = normals @ v.T                                # (N,m)
        if law == LEGACY:
            rows.append(legacy_kernel(mu, mu0, ctype, c_lambert=c_lambert).T)
        else:
            rows.append(kernel(mu, mu0, ctype, gamma=gamma, threshold=float(c)).T)
    return np.stack(rows, axis=0)


def curve_thresholds(normals: np.ndarray, areas: np.ndarray, cameras: list, m: int,
                     curve_types: list, gamma: float = TRANSFER_EXPONENT,
                     frame_area: float | None = None, sigma: float = 1.0,
                     delta: float = 1.0, psi0: float = 0.0) -> np.ndarray:
    """The level of the illumination cosine each curve is thresholded at, derived from the
    body's own first frame as the organisers derive theirs from the video's.

    `areas` are the facet areas of the extended Gaussian image, so this is a property of the
    body being modelled and changes as an inversion changes it. Intensity curves get zero,
    which their kernel ignores. `frame_area` is how much of the image the body is set in;
    left out, it is taken as four times the body's largest projected area, which is the order
    a framing that keeps the body inside the frame at every phase gives.
    """
    psi = psi_grid(m, sigma=sigma, psi0=psi0)
    v0 = body_frame_dirs(OMEGA0, psi)
    mu0 = normals @ v0[0]
    out = np.zeros(len(curve_types))
    proj = 0.5 * np.abs(normals @ v0.T).T @ areas          # silhouette area at each phase
    if frame_area is None:
        frame_area = FRAME_OVER_BODY * float(proj.max())
    for j, (cam, ctype) in enumerate(zip(cameras, curve_types)):
        if ctype != "binary":
            continue
        mu = normals @ body_frame_dirs(cam.omega(delta=delta), psi)[0]
        out[j] = otsu_threshold(mu, mu0, areas, frame_area, gamma=gamma)
    return out


def stack_A(grid: NormalGrid, cameras: list, m: int, areas: np.ndarray | None = None,
            gamma: float = TRANSFER_EXPONENT, sigma: float = 1.0, delta: float = 1.0,
            psi0: float = 0.0, frame_area: float | None = None, law: str = MEASURED,
            c_lambert: float = 0.1) -> tuple:
    """Full operator for one model: every camera's intensity curve, then every camera's
    binary curve. Returns (A, curve_types) with A of shape (2 * len(cameras), m, N).

    `areas` is the extended Gaussian image the binary curves' thresholds are derived from. A
    caller that has no estimate yet passes none and gets zero thresholds, which counts the
    whole lit and visible area rather than the part above the level the organisers used.
    """
    cams2 = list(cameras) + list(cameras)
    types = ["intensity"] * len(cameras) + ["binary"] * len(cameras)
    thr = None if (areas is None or law == LEGACY) else curve_thresholds(
        grid.normals, areas, cams2, m, types, gamma=gamma, frame_area=frame_area,
        sigma=sigma, delta=delta, psi0=psi0)
    A = build_A(grid.normals, cams2, m, types, gamma=gamma, thresholds=thr,
                sigma=sigma, delta=delta, psi0=psi0, law=law, c_lambert=c_lambert)
    return A, types


# ---------- numpy reference implementations -----------------------------------------
def normalize_np(y: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """N_eps(y): divide each curve (last axis) by max(mean, eps)."""
    mbar = np.maximum(y.mean(axis=-1, keepdims=True), eps)
    return y / mbar


def dn_np(y: np.ndarray, v: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Derivative of N_eps at y applied to v, per curve (last axis = frames)."""
    mbar = y.mean(axis=-1, keepdims=True)
    guard = mbar >= eps
    mb = np.maximum(mbar, eps)
    mv = v.mean(axis=-1, keepdims=True)
    return np.where(guard, v / mb - y * mv / mb**2, v / eps)


def dn_adjoint_np(y: np.ndarray, w: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Adjoint of dn_np: DN(y)^T w = w/mbar - (<y,w>/(m*mbar^2)) ones where mbar >= eps,
    and w/eps where the guard is active."""
    m = y.shape[-1]
    mbar = y.mean(axis=-1, keepdims=True)
    guard = mbar >= eps
    mb = np.maximum(mbar, eps)
    yw = (y * w).sum(axis=-1, keepdims=True)
    return np.where(guard, w / mb - yw / (m * mb**2) * np.ones_like(w), w / eps)


def forward_np(A: np.ndarray, g: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """T(g) = N_eps(A g); A (C,m,N), g (N,) -> (C,m)."""
    return normalize_np(np.einsum("cmn,n->cm", A, g), eps=eps)


def deriv_adjoint_np(A: np.ndarray, g: np.ndarray, h: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """[dT(g)]^T h = A^T DN(Ag)^T h; h (C,m) -> (N,)."""
    y = np.einsum("cmn,n->cm", A, g)
    return np.einsum("cmn,cm->n", A, dn_adjoint_np(y, h, eps=eps))


# ---------- torch operator ----------------------------------------------------------
try:
    import torch

    class ConvexPhotometricOperator(torch.nn.Module):
        """T = N_eps o A as a torch module, with the closed-form derivative adjoint.

        One buffer, A (C, m, N) float32. The curve mask is not stored here; the calls that
        need it take it as an argument, shape (B, C), with 0 marking a missing curve.
        """

        def __init__(self, A: np.ndarray, eps: float = 1e-3):
            super().__init__()
            self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
            self.eps = float(eps)

        @property
        def n_curves(self) -> int:
            return self.A.shape[0]

        @property
        def m(self) -> int:
            return self.A.shape[1]

        @property
        def n(self) -> int:
            return self.A.shape[2]

        def raw(self, g: "torch.Tensor") -> "torch.Tensor":
            """A g : (B,N) -> (B,C,m)."""
            return torch.einsum("cmn,bn->bcm", self.A, g)

        def normalize(self, y: "torch.Tensor") -> "torch.Tensor":
            """N_eps along the last axis."""
            mbar = y.mean(dim=-1, keepdim=True).clamp_min(self.eps)
            return y / mbar

        def forward(self, g: "torch.Tensor") -> "torch.Tensor":
            return self.normalize(self.raw(g))

        def adjoint_raw(self, r: "torch.Tensor") -> "torch.Tensor":
            """A^T r : (B,C,m) -> (B,N)."""
            return torch.einsum("cmn,bcm->bn", self.A, r)

        def dn_adjoint(self, y: "torch.Tensor", w: "torch.Tensor") -> "torch.Tensor":
            """DN(y)^T w along the last axis; same branches as dn_adjoint_np."""
            mfr = y.shape[-1]
            mbar = y.mean(dim=-1, keepdim=True)
            guard = mbar >= self.eps
            mb = mbar.clamp_min(self.eps)
            yw = (y * w).sum(dim=-1, keepdim=True)
            full = w / mb - yw / (mfr * mb**2)
            return torch.where(guard, full, w / self.eps)

        def deriv_adjoint(self, g: "torch.Tensor", h: "torch.Tensor",
                          mask: "torch.Tensor | None" = None) -> "torch.Tensor":
            """[dT(g)]^T h with optional per-curve mask (B,C): masked curves contribute 0."""
            y = self.raw(g)
            r = self.dn_adjoint(y, h)
            if mask is not None:
                r = r * mask[..., None]
            return self.adjoint_raw(r)

except ImportError:  # torch is optional for the numpy-side of the package
    pass
