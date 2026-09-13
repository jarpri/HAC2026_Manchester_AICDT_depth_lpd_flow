"""Volumetric forward model: a mollified signed distance field, rendered by line integrals.

phi is a signed distance function, negative inside, |grad phi| = 1, so n = grad phi.

Mollifier: Psi_w(d) = f(d/w)/w with f the logistic density, f(t) = 1/(4 cosh^2(t/2)).
Phi_s(x) = sigmoid(x/w) is its CDF.

Opacity uses the occlusion-exact form rather than a density kappa*Psi_w(phi): with

    rho(t) = max(-d/dt ln Phi_s(phi(r(t))), 0)

the transmittance along a segment where phi decreases is T(t) = Phi_s(phi(r(t))) once the ray
starts far outside. That is 1 outside and 0 inside at every incidence angle. A density form
instead integrates to kappa/|omega . n| along the ray, independent of w, which leaves the
body view-angle-dependently semi-transparent at any finite kappa.

The rendering weight is W(t) = -d/dt Phi_s(phi(r(t))) = Psi_w(phi)|dphi/dt|. It peaks at
phi = 0 and integrates to 1 across a crossing.

The two line integrals:

    L_pixel     = INT W(t) c(r(t)) dt
    T_s(x, w_k) = Phi_s( min_{u > u0} phi(x + u w_k) )

The shadow form is the closest-approach transmittance, found by sphere tracing. Its minimum
starts beyond the shading point's own shell, u0 = shell*w/max(n . omega, floor); taken from
u = 0 the minimum at a surface point is phi = 0, giving T_s = 1/2 for every lit point.

Bias in w. The coarea formula plus the Steiner factor J(d) = 1 + 2Hd + Kd^2 gives

    INT Psi_w(phi) g dV = INT dA [ g(1 + K m2 w^2) + 2H m2 w^2 dn_g + (m2 w^2/2) dnn_g ]
                          + O(w^4) + O(e^{-reach/w})

with m2 = pi^2/3. The expansion holds only inside the reach, and the logistic has exponential
tails, so two regimes are not covered: at an edge the reach is zero and the error is O(w); and
across a neck of half-thickness t the tails overlap and add density e^{-2t/w}, filling it.
Both biases favour convexity, so this is a continuation stage and not a final model.
`w_bounds` gives the two limits and `anneal_schedule` the path to them.

Gradients: the soft model is evaluated and the same soft model is differentiated, so the
w-schedule is a continuation method over a well-defined path of objectives. Quadrature nodes
along the ray are fixed; the shape enters through phi evaluated at them. The two image
thresholds stay hard and are differentiated by the coarea formula.
"""
from __future__ import annotations

import numpy as np
import torch

from hac26.forward.shared.coarea import threshold_count, threshold_sum

__all__ = ["M2_LOGISTIC", "SHELL", "N_NODES", "logistic_cdf", "logistic_pdf",
           "normalised_field", "eikonal_penalty", "field_normal", "closest_approach",
           "shell_entry",
           "shadow_transmittance", "march_view", "direct_radiance", "gathered_bounce",
           "render_frame", "reduce_frame_coarea", "shadow_deficit", "w_bounds",
           "anneal_schedule", "curves"]

M2_LOGISTIC = float(np.pi ** 2 / 3.0)      # variance of the unit-scale logistic density
SHELL = 10.0        # gate half-width in units of w. The weight inside it is
                    # 1 - 2/(1 + e^SHELL), so a wider gate loses less; see w_bounds.
N_NODES = 64        # in-shell quadrature nodes per ray, fixed so the count carries no
                    # geometry dependence and the gradient does not jump by a node


# ----------------------------------------------------------------------------- mollifier

def logistic_cdf(x: torch.Tensor, w: float) -> torch.Tensor:
    """Phi_s(x) = sigmoid(x / w)."""
    return torch.sigmoid(x / w)


def logistic_pdf(x: torch.Tensor, w: float) -> torch.Tensor:
    """Psi_w(x) = Phi_s'(x) = f(x/w)/w with f(t) = 1/(4 cosh^2(t/2))."""
    s = torch.sigmoid(x / w)
    return s * (1.0 - s) / w


# ----------------------------------------------------------------------------- the field

def field_normal(field, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """n = grad phi / |grad phi|, by autodiff of the field with respect to position."""
    xr = x.detach().requires_grad_(True)
    phi = field(xr)
    g, = torch.autograd.grad(phi.sum(), xr, create_graph=create_graph)
    return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def eikonal_penalty(field, points: torch.Tensor) -> torch.Tensor:
    """mean (|grad phi| - 1)^2 at sampled points.

    A CONSTRAINT ON THE FIELD, not a prior on the shape. Every formula in this module assumes
    |grad phi| = 1: sphere tracing may not overstep, the closest-approach shadow form reads
    phi as a distance, and the Steiner expansion of the mollified integral is stated in terms
    of the offset distance. A field that is only approximately a distance function violates
    all three.

    It is therefore not interchangeable with a curvature or total-variation penalty, which
    say something about what shapes are like. Those are supplied by the shape prior Gamma in
    hac26.solvers.map_gauss_newton, estimated from a library, and nowhere else.

    The alternative to this penalty is `normalised_field`, which divides by |grad phi| at
    evaluation time instead of driving it to 1 during optimisation.
    """
    x = points.detach().requires_grad_(True)
    v = field(x)
    g, = torch.autograd.grad(v.sum(), x, create_graph=True)
    return ((g.norm(dim=-1) - 1.0) ** 2).mean()


def normalised_field(field):
    """phi / |grad phi|, so the formulas that assume a true SDF hold near the zero set.

    The alternative is an eikonal penalty during optimisation; this is the evaluation-time
    version and costs one extra gradient per call.
    """
    def phi(x):
        xr = x.detach().requires_grad_(True)
        v = field(xr)
        g, = torch.autograd.grad(v.sum(), xr, create_graph=True)
        return field(x) / g.norm(dim=-1).clamp_min(1e-6)
    return phi


# ------------------------------------------------------------------------------ tracing

def closest_approach(o: torch.Tensor, d: torch.Tensor, field, iters: int = 96,
                     tmax: float = 8.0, t0: float = 1e-3):
    """min_{u > 0} phi(o + u d), and the point attaining it.

    Marching by |phi| is the SDF's own guarantee: no step can pass through the surface, and
    once inside the same bound holds for the distance back out.
    """
    t = torch.full(o.shape[:-1], t0, device=o.device, dtype=o.dtype)
    best = field(o + t[..., None] * d)
    best_t = t.clone()
    for _ in range(iters):
        x = o + t[..., None] * d
        v = field(x)
        closer = v < best
        best = torch.where(closer, v, best)
        best_t = torch.where(closer, t, best_t)
        t = torch.minimum(t + v.abs().clamp_min(1e-4),
                          torch.full_like(t, tmax))
    return best, o + best_t[..., None] * d


def shadow_transmittance(x: torch.Tensor, dirs: torch.Tensor, field, w: float,
                         normal: torch.Tensor | None = None, shell: float = SHELL,
                         cos_floor: float = 0.05) -> torch.Tensor:
    """T_s(x, omega_k) = Phi_s(min_u phi(x + u omega_k)), one scalar per (point, direction).

    x is (P, 3) and dirs is (K, 3); the result is (P, K).

    THE MINIMUM STARTS BEYOND THE POINT'S OWN SHELL. Taken literally from u = 0, the minimum
    over a shading point that lies on the surface is phi(x) = 0, so T_s = Phi_s(0) = 1/2 for
    every lit point whether or not anything occludes it -- a uniform halving of the direct
    term that no amount of geometry can produce. The shadow ray must measure occlusion by
    OTHER geometry, so the trace starts where the point's own mollified shell ends,

        u0 = shell * w / max(n . omega, cos_floor)

    at which phi has risen to about shell*w on a locally flat surface. An unoccluded point
    then returns Phi_s(shell) = 1 - 2/(1 + e^shell), which is short of 1 by the same
    telescoping residual as the view gate -- and that is why the zero-phase deficit is small
    rather than identically zero at finite w.
    """
    P, K = x.shape[0], dirs.shape[0]
    n = field_normal(field, x, create_graph=False) if normal is None else normal
    mu = (n[:, None, :] * dirs[None, :, :]).sum(-1).abs().clamp_min(cos_floor)   # (P, K)
    u0 = (shell * w / mu)[..., None]                                            # (P, K, 1)
    o = (x[:, None, :] + u0 * dirs[None, :, :]).reshape(-1, 3)
    d = dirs[None, :, :].expand(P, K, 3).reshape(-1, 3)
    phi_min, _ = closest_approach(o, d, field, t0=0.0)
    return logistic_cdf(phi_min.reshape(P, K), w)


# ------------------------------------------------------------------- view line integral

def shell_entry(o: torch.Tensor, d: torch.Tensor, field, w: float, shell: float = SHELL,
                iters: int = 128, tmax: float = 8.0) -> torch.Tensor:
    """First t at which the ray reaches the shell, phi <= shell * w, by sphere tracing.

    Stepping by phi - shell*w approaches the shell boundary without entering it, which is the
    SDF's own guarantee applied to the offset surface {phi = shell*w}.
    """
    t = torch.zeros(o.shape[:-1], device=o.device, dtype=o.dtype)
    for _ in range(iters):
        v = field(o + t[..., None] * d) - shell * w
        t = torch.where((v > 0) & (t < tmax), t + v.clamp_min(1e-4), t)
    return t


def march_view(o: torch.Tensor, d: torch.Tensor, field, w: float, shade_fn,
               shell: float = SHELL, n_nodes: int = N_NODES, tmax: float = 8.0,
               cos_floor: float = 0.05):
    """L = INT W(t) c(r(t)) dt along each ray, with W = -d/dt Phi_s(phi).

    GATE WIDTH. The weight across a crossing telescopes exactly,
    sum_i W_i = Phi_s(phi_start) - Phi_s(phi_end), so gating at |phi| < D w delivers
    Phi_s(Dw) - Phi_s(-Dw) = 1 - 2/(1 + e^D) rather than 1. That deficit falls off
    exponentially in D, and at a narrow gate it is well above the measurement noise. It is a
    uniform multiplicative factor and so largely cancels under per-curve mean normalisation,
    but not entirely, because tau_I is a fixed level -- and a grazing ray that never crosses
    picks up opacity 1 - Phi_s(phi_min), so a narrow gate puts a discontinuity at the
    silhouette, which is where the binary channel lives.

    COMPOSITING IN LOG SPACE. The transmittance ratio Phi_s(phi_cur)/Phi_s(phi_prev)
    underflows deep inside the body, so the increment is accumulated as

        log T <- log T + logsigmoid(phi_cur / w) - logsigmoid(phi_prev / w)

    clamped at zero increment, which is what makes a receding segment contribute nothing and
    so splits the ray at sign changes of dphi/dt without explicit segmentation. The per-step
    weight is then T (1 - e^dlog), evaluated with expm1.

    FIXED NODE COUNT. Exactly n_nodes quadrature nodes are placed uniformly in t across the
    shell crossing, so the node count does not depend on the geometry and the gradient does
    not jump when a marcher would have taken one step more or fewer. The window length scales
    as 2 shell w / |d . n| at the entry point, since a ray at incidence |d . n| takes that
    much longer in t to cross the same range of phi.
    """
    t_in = shell_entry(o, d, field, w, shell, tmax=tmax).detach()
    n_entry = field_normal(field, o + t_in[..., None] * d, create_graph=False)
    cosang = (n_entry * d).sum(-1).abs().clamp_min(cos_floor)
    span = (2.0 * shell * w) / cosang
    dt = (span / n_nodes).detach()

    log_T = torch.zeros(o.shape[:-1], device=o.device, dtype=o.dtype)
    L = torch.zeros_like(log_T)
    phi_prev = field(o + t_in[..., None] * d)
    for i in range(n_nodes):
        t = (t_in + (i + 1) * dt).detach()
        x = (o + t[..., None] * d).detach()
        phi_cur = field(x)
        dlog = (torch.nn.functional.logsigmoid(phi_cur / w)
                - torch.nn.functional.logsigmoid(phi_prev / w)).clamp(max=0.0)
        T = torch.exp(log_T)
        wgt = T * (-torch.expm1(dlog))
        m = wgt > 1e-9
        if bool(m.any()):
            c = torch.zeros_like(L)
            c[m] = shade_fn(x[m])
            L = L + wgt * c
        log_T = log_T + dlog
        phi_prev = phi_cur
    return L, torch.exp(log_T)


# ------------------------------------------------------------------------------ shading

def direct_radiance(x: torch.Tensor, field, w: float, sun_dirs: torch.Tensor,
                    rho_alb: float = 0.85, e0: float = 1.0,
                    shadows: bool = True) -> torch.Tensor:
    """c(x) = (rho/pi)(E0/K) sum_k (n . omega_k)+ T_s(x, omega_k).

    The K source directions span the source's finite angular disc, so the penumbra is
    physical and stays separate from w: the mollifier is not used to stand in for it.
    """
    n = field_normal(field, x)
    mu0 = (n[:, None, :] * sun_dirs[None, :, :]).sum(-1).clamp_min(0.0)
    ts = (shadow_transmittance(x, sun_dirs, field, w) if shadows
          else torch.ones_like(mu0))
    return (rho_alb / np.pi) * (e0 / sun_dirs.shape[0]) * (mu0 * ts).sum(-1)


def _cosine_directions(n: torch.Tensor, m: int, generator=None) -> torch.Tensor:
    """m cosine-sampled directions about each normal. Returns (P, m, 3)."""
    P = n.shape[0]
    u1 = torch.rand(P, m, device=n.device, dtype=n.dtype, generator=generator)
    u2 = torch.rand(P, m, device=n.device, dtype=n.dtype, generator=generator)
    r = u1.sqrt()
    theta = 2.0 * np.pi * u2
    local = torch.stack([r * torch.cos(theta), r * torch.sin(theta),
                         (1.0 - u1).clamp_min(0.0).sqrt()], dim=-1)
    a = torch.where(n[..., 2:3].abs() < 0.9,
                    torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype).expand_as(n),
                    torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n))
    t1 = torch.cross(a, n, dim=-1); t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t2 = torch.cross(n, t1, dim=-1)
    return (local[..., 0:1] * t1[:, None, :] + local[..., 1:2] * t2[:, None, :]
            + local[..., 2:3] * n[:, None, :])


def gathered_bounce(x: torch.Tensor, field, w: float, sun_dirs: torch.Tensor,
                    m_dirs: int = 32, rho_alb: float = 0.85, e0: float = 1.0,
                    shadows: bool = True):
    """One gathered bounce, reusing the closest-approach trace.

        c1(x) = (rho/pi) sum_m (1 - T_s(x, omega_m)) c0(y_m) (n . omega_m)+ (2 pi / M)

    y_m is the closest-approach point along omega_m. At albedo 0.85 this term is first order.
    """
    n = field_normal(field, x)
    dirs = _cosine_directions(n, m_dirs)                      # (P, M, 3)
    P, M = dirs.shape[0], dirs.shape[1]
    o = (x[:, None, :] + 1e-3 * dirs).reshape(-1, 3)
    dd = dirs.reshape(-1, 3)
    phi_min, y = closest_approach(o, dd, field)
    ts = logistic_cdf(phi_min, w).reshape(P, M)
    c0 = direct_radiance(y, field, w, sun_dirs, rho_alb, e0, shadows).reshape(P, M)
    mu = (n[:, None, :] * dirs).sum(-1).clamp_min(0.0)
    return (rho_alb / np.pi) * ((1.0 - ts) * c0 * mu).sum(-1) * (2.0 * np.pi / M)


# -------------------------------------------------------------------------------- frame

def render_frame(field, view: torch.Tensor, sun_lab_dirs: torch.Tensor, w: float,
                 sensor=None, res: int = 128, extent: float = 1.6, rho_alb: float = 0.85,
                 e0: float = 1.0, bounce: bool = True, m_dirs: int = 32,
                 supersample: int = 4, eye_distance: float = 8.0,
                 shadows: bool = True):
    """One frame's radiance image, through the sensor chain if one is given.

    Returns the image at the reduced resolution, ready for the two thresholded reductions.
    """
    from .shared.common import camera_basis
    ex, ey = camera_basis(view)
    n_px = res * supersample
    a = torch.linspace(-extent, extent, n_px, device=view.device, dtype=view.dtype)
    gx, gy = torch.meshgrid(a, a, indexing="ij")
    o = (gx.reshape(-1, 1) * ex + gy.reshape(-1, 1) * ey) + 2.0 * extent * view
    d = (-view)[None, :].expand(o.shape[0], 3)

    def shade_fn(pts):
        c = direct_radiance(pts, field, w, sun_lab_dirs, rho_alb, e0, shadows)
        if bounce:
            c = c + gathered_bounce(pts, field, w, sun_lab_dirs, m_dirs,
                                    rho_alb, e0, shadows)
        return c

    L, _ = march_view(o, d, field, w, shade_fn)
    img = L.reshape(1, n_px, n_px)
    if sensor is None:
        return img[0]
    # off-axis cosine and normalised radius of each pixel ray, for cos^4 and vignetting
    rr = (gx ** 2 + gy ** 2).sqrt()
    cos_off = (eye_distance / (eye_distance ** 2 + rr ** 2).sqrt())[None]
    radius = (rr / rr.max().clamp_min(1e-9))[None]
    return sensor(img, cos_off, radius, supersample=supersample)[0]


def reduce_frame_coarea(img: torch.Tensor, tau_i: float, tau_b: float):
    """The two observables, with exact derivatives through both thresholds by coarea.

    Distinct from forward.shared.common.reduce_frame, which is the plain non-differentiable
    reduction. The thresholds are not softened to match the geometric softness: they are
    different bandwidths and only the geometric one is physically justified.
    """
    return threshold_sum(img, tau_i), threshold_count(img, tau_b)


def shadow_deficit(field, view: torch.Tensor, sun_dirs: torch.Tensor, w: float,
                   **kw) -> torch.Tensor:
    """D_shadow = sum over pixels of [ L(shadows=False) - L(shadows=True) ].

    Three properties must hold, and each failure is a distinct fault:

      * D >= 0 everywhere. Shadow removes light and never adds it.
      * D = 0 exactly at azimuth 0, elevation 0. There v = s, so a point is shadowed only
        when it is blocked along the view direction, i.e. when it is not visible at all.
      * D varies with psi for a non-convex body. A psi-flat term is annihilated by the
        per-curve mean normalisation and carries no information.

    Identically zero means the shadow ray is not wired in. Non-zero but flat in psi means the
    source is not rotating in the body frame; it must be recomputed per phase as
    s_body(psi) = R_z(-psi - psi0) s_lab.
    """
    lit = render_frame(field, view, sun_dirs, w, shadows=False, **kw)
    shd = render_frame(field, view, sun_dirs, w, shadows=True, **kw)
    return (lit - shd).sum()


# --------------------------------------------------------------------------- w schedule

def w_bounds(t_min: float, radius: float, eps: float = 1e-3, psf_px: float = 1.5,
             body_px: float = 800.0) -> dict:
    """The two hard bounds on w, and their minimum.

    Neck: a neck of half-thickness t_min fills by about e^{-2 t_min / w}, so holding that
    below eps requires w <= 2 t_min / ln(1/eps).

    Silhouette: a grazing ray that does not cross accumulates opacity 1 - Phi_s(phi_min), so
    the outline is blurred over a scale w in phi. Keeping that below the optical PSF
    footprint expressed in object units stops the mollification from biasing the thresholded
    pixel count -- which is what the psf_px / body_px ratio below computes.
    """
    neck = 2.0 * t_min / np.log(1.0 / eps)
    silhouette = 2.0 * (psf_px / body_px) * radius
    return {"neck": neck, "silhouette": silhouette, "w_max": min(neck, silhouette)}


def anneal_schedule(radius: float, w_max: float, levels: int = 6, hold: int = 400):
    """Geometric path from w = 0.05 R down to w_max, held `hold` steps at each level.

    Held rather than swept so the continuation path is followed rather than jumped.
    """
    w0 = 0.05 * radius
    ws = np.geomspace(w0, max(w_max, 1e-6), levels)
    return [(float(w), int(hold)) for w in ws]


# -------------------------------------------------------------------------------- curves

def curves(field, cam_dir, sun_lab, psi, w: float, sensor=None, psi0: float = 0.0,
           delta_rad: float = 0.0, n_source: int = 8, tau_i: float = 0.0,
           tau_b: float | None = None, **kw):
    """Both curves over a rotation, in the body frame, normalised by their mean over psi.

    The field is fixed and the camera and source directions rotate by R_z(-psi - psi0). The
    binary threshold defaults to the full-frame Otsu level of frame 0, as in the rasterised
    pipeline; the intensity threshold is the fixed low one.
    """
    from hac26.conventions import source_directions, to_body
    from .mesh.raster import otsu_threshold

    src = source_directions(delta_rad, n_source)          # (K, 3) in the lab frame
    cam = to_body(np.asarray(cam_dir, dtype=float), np.asarray(psi, dtype=float), psi0)
    frames = []
    for j in range(len(psi)):
        sun_b = np.stack([to_body(s, np.array([psi[j]]), psi0)[0] for s in src])
        img = render_frame(field, torch.tensor(cam[j], dtype=torch.float32),
                           torch.tensor(sun_b, dtype=torch.float32), w, sensor=sensor, **kw)
        frames.append(img)
    if tau_b is None:
        tau_b = otsu_threshold(frames[0])
    I, N = [], []
    for img in frames:
        i_, n_ = reduce_frame_coarea(img, tau_i, tau_b)
        I.append(i_); N.append(n_)
    I = torch.stack(I); N = torch.stack(N)
    return I / I.mean().clamp_min(1e-12), N / N.mean().clamp_min(1e-12)
