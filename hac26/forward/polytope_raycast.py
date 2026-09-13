"""Bodies as unions of convex polytopes, rendered by exact ray casting.

Representation. The data carry orders of magnitude fewer usable dimensions than an SDF on
a voxel grid has unknowns, which is why an unregularised volumetric fit carves wherever the
model error happens to point. Every ground truth is a POLYTOPE, so the natural unknown is
not a level set but a finite set of face distances. A union of K convex polytopes on N
normals has K(N+3) parameters, which at usable K and N is the same order as the information
the data carries -- so the fit is determined rather than regularised into shape.

    K_k(h) = { x : <x - c_k, n_i> <= h_{k,i}  for all i },      body = union_k K_k

Each part is convex by construction, so it needs no convexity penalty and no eikonal term,
and the union is non-convex exactly where parts meet -- which is the contact-binary
geometry of model 3.

Ray casting. For a ray x = o + t d against one polytope, each halfspace gives a scalar
bound on t: <d,n_i> > 0 caps t above, < 0 caps it below. So

    t_enter = max_i lower_i,   t_exit = min_i upper_i,   hit iff t_enter <= t_exit

and the surface normal is the n_i attaining t_enter. This is EXACT -- the only discretised
thing is the pixel grid, not the geometry -- and it is a handful of vectorised reductions,
so it runs on the GPU without a rasteriser, a mesh, or a boolean library. Occlusion and
cast shadows both fall out of the same routine, which is the part the convex operator
cannot express at all.
"""
import numpy as np
import torch


def fibonacci_normals(n=66):
    """Near-uniform directions on S^2. The face-normal dictionary of the polytopes."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], 1).astype(np.float32)


def raycast(o, d, H, C, N, eps=1e-6):
    """Nearest entry into the UNION of K polytopes.

    o (P,3) ray origins, d (3,) shared direction, H (K,Nn) supports, C (K,3) centres,
    N (Nn,3) normals. Returns t (P,), normal (P,3), hit (P,) bool.
    """
    K, Nn = H.shape
    dn = d @ N.T                                     # (Nn,)  <d, n_i>
    on = o @ N.T                                     # (P,Nn) <o, n_i>
    cn = C @ N.T                                     # (K,Nn) <c_k, n_i>
    # bound_i = (h_i + <c,n_i> - <o,n_i>) / <d,n_i>
    num = (H + cn).unsqueeze(1) - on.unsqueeze(0)    # (K,P,Nn)
    big = torch.full_like(dn, 1e9)
    pos, neg = dn > eps, dn < -eps
    q = num / torch.where(pos | neg, dn, torch.ones_like(dn))
    t_exit = torch.where(pos.expand_as(q), q, big).amin(-1)             # (K,P)
    low = torch.where(neg.expand_as(q), q, -big)
    t_ent, arg = low.max(-1)                                            # (K,P)
    # a halfspace parallel to the ray and already violated kills the whole ray
    par = (~pos & ~neg) & (num < 0)     # (Nn,) broadcasts along num's LAST axis
    dead = par.any(-1)
    ok = (t_ent <= t_exit) & (t_exit > 0) & ~dead
    t_ent = torch.where(ok, t_ent.clamp(min=0.0), torch.full_like(t_ent, 1e9))
    t, kbest = t_ent.min(0)                                             # (P,)
    hit = t < 1e8
    nrm = N[arg.gather(0, kbest.unsqueeze(0)).squeeze(0)]
    return t, nrm, hit


def occluded(p, d, H, C, N, eps=1e-4):
    """Does the ray from p along d meet any part? Used for cast shadows.

    The epsilon push-off along d matters: without it every lit point shadows itself at
    t = 0, which silently turns the whole body black.
    """
    t, _, hit = raycast(p + eps * d, d, H, C, N)
    return hit & (t > 0)


def inside(p, H, C, N, tol=1e-4):
    """Strictly interior to ANY part -- such a point is not on the union's boundary."""
    v = (p.unsqueeze(0) @ N.T) - (C @ N.T).unsqueeze(1) - H.unsqueeze(1)
    return (v < -tol).all(-1).any(0)


def render(H, C, N, view, sun, res=64, extent=None, tau_i=0.0, tau_b=0.02):
    """One frame: (intensity, lit area) for a parallel beam and an orthographic camera.

    Mirrors the imaging chain rather than an analytic functional: shade the visible
    surface, then threshold, then sum pixel values for intensity and count them for
    binary. That is what the organisers' pipeline does, and it is the only way the two
    curve types differ by more than a constant.
    """
    dev = H.device
    if extent is None:
        extent = float((H + (C @ N.T)).max()) * 1.25
    up = torch.tensor([0.0, 0.0, 1.0], device=dev)
    if abs(float(view @ up)) > 0.99:
        up = torch.tensor([0.0, 1.0, 0.0], device=dev)
    ex = torch.cross(up, view, dim=0); ex = ex / ex.norm()
    ey = torch.cross(view, ex, dim=0); ey = ey / ey.norm()
    a = torch.linspace(-extent, extent, res, device=dev)
    gx, gy = torch.meshgrid(a, a, indexing="ij")
    # `view` is omega_c, the direction from the body to the camera -- the same convention
    # the analytic operator uses when it writes mu = n . omega_c. Rays therefore start on
    # the camera side and travel along -view. Starting at -extent*view and marching along
    # +view instead renders the far surface, which shows up as the near and far cameras
    # swapping values.
    o = (gx.reshape(-1, 1) * ex + gy.reshape(-1, 1) * ey) + 2.0 * extent * view
    t, nrm, hit = raycast(o, -view, H, C, N)
    mu = nrm @ view
    mu0 = nrm @ sun
    lit = hit & (mu > 0) & (mu0 > 0)
    if lit.any():
        p = o[lit] - t[lit].unsqueeze(1) * view
        sh = occluded(p, sun, H, C, N)
        val = torch.zeros_like(mu)
        # Radiance, not radiance x mu. A Lambertian facet's radiance is rho/pi * mu0; the
        # viewing obliquity enters through the pixel's projected area, which the image-plane
        # sum already supplies, so multiplying by mu here would double-count it.
        val[lit] = torch.where(sh, torch.zeros_like(mu0[lit]), mu0[lit])
    else:
        val = torch.zeros_like(mu)
    px = (2 * extent / res) ** 2
    return (val * (val > tau_i)).sum() * px, ((val > tau_b).sum()).to(val.dtype) * px
