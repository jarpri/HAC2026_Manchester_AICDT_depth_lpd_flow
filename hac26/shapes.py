"""Mesh utilities shared by the convex stage and the pose conventions: simple synthetic
shapes, facet areas and normals binned into the EGI, the challenge pose (rescale_touch_z,
canonicalize_r), the support function of a point set, and a brute-force convex curve
renderer used to cross-check the convex operator in tests.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.special import gammaln, lpmv

from .geometry import OMEGA0, NormalGrid, body_frame_dirs, cell_index, project_closure, psi_grid
from hac26.conventions import TRANSFER_EXPONENT
from hac26.forward.convex_egi import curve_thresholds, kernel


# ---------- icosphere ---------------------------------------------------------------
def icosphere(subdiv: int = 3) -> tuple:
    """Unit icosphere (verts, faces). subdiv=3 -> 642 verts, 1280 faces."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                      [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                      [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(subdiv):
        edge_mid: dict = {}
        vlist = list(verts)

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in edge_mid:
                p = vlist[i] + vlist[j]
                p = p / np.linalg.norm(p)
                edge_mid[key] = len(vlist)
                vlist.append(p)
            return edge_mid[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(vlist)
        faces = np.array(new_faces)
    return verts, faces


# ---------- real spherical harmonics -------------------------------------------------
def real_sh_basis(L: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Rows = real orthonormal SH Y for l=1..L (l=0 excluded; it is a pure log-scale
    shift and scale is unidentifiable from normalized curves). Shape (n_coef, npts)."""
    x = np.cos(theta)
    rows = []
    for l in range(1, L + 1):
        nl0 = np.sqrt((2 * l + 1) / (4 * np.pi))
        rows.append(nl0 * lpmv(0, l, x))
        for m in range(1, l + 1):
            nlm = np.sqrt((2 * l + 1) / (4 * np.pi)) * np.exp(
                0.5 * (gammaln(l - m + 1) - gammaln(l + m + 1)))
            plm = lpmv(m, l, x)
            rows.append(np.sqrt(2.0) * nlm * plm * np.cos(m * phi))
            rows.append(np.sqrt(2.0) * nlm * plm * np.sin(m * phi))
    return np.stack(rows, axis=0)


def sh_lognormal_mesh(rng: np.random.Generator, L: int = 6, amp: float = 0.35,
                      decay: float = 1.5, subdiv: int = 3) -> tuple:
    """Star-shaped body r(u) = exp(sum a_lm Y_lm(u)), a_lm ~ N(0, (amp/(1+l)^decay)^2)."""
    u, faces = icosphere(subdiv)
    theta = np.arccos(np.clip(u[:, 2], -1, 1))
    phi = np.mod(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)
    B = real_sh_basis(L, theta, phi)
    ls = np.concatenate([[l] * (2 * l + 1) for l in range(1, L + 1)])
    a = rng.normal(0.0, amp / (1.0 + ls) ** decay)
    r = np.exp(a @ B)
    return u * r[:, None], faces, (a, L)


def random_convex_polytope(rng: np.random.Generator, n_pts: int = 40) -> tuple:
    pts = rng.normal(size=(n_pts, 3)) * rng.uniform(0.5, 1.5, size=3)
    hull = ConvexHull(pts)
    return pts[hull.vertices], None


def ellipsoid_mesh(rng: np.random.Generator, subdiv: int = 3) -> tuple:
    u, faces = icosphere(subdiv)
    axes = rng.uniform(0.5, 1.5, size=3)
    return u * axes, faces, axes


# ---------- convex hull with outward-oriented faces ----------------------------------
def hull_mesh(points: np.ndarray) -> tuple:
    hull = ConvexHull(points)
    verts = points
    faces = hull.simplices.copy()
    eqs = hull.equations[:, :3]
    for i, f in enumerate(faces):
        n = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
        if n @ eqs[i] < 0:
            faces[i] = f[::-1]
    return verts, faces


def face_normals_areas(verts: np.ndarray, faces: np.ndarray) -> tuple:
    """Unit normals and areas of every face; a degenerate face gets a zero normal."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    a2 = np.linalg.norm(cr, axis=1)
    keep = a2 > 1e-14
    n = np.zeros_like(cr)
    n[keep] = cr[keep] / a2[keep, None]
    return n, 0.5 * a2


def mesh_to_egi(verts: np.ndarray, faces: np.ndarray, grid: NormalGrid,
                close: bool = True) -> np.ndarray:
    """Bin facet areas by facet normal into the grid cells (discretized S_K).
    For a closed oriented mesh sum_f area_f n_f = 0 exactly; binning to cell-center
    normals perturbs this, so optionally re-project onto the closure cone."""
    n, a = face_normals_areas(verts, faces)
    idx = cell_index(grid, n)
    g = np.bincount(idx, weights=a, minlength=grid.n).astype(float)
    return project_closure(g, grid.normals) if close else g


def solid_centroid(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Centroid of the solid a closed mesh bounds, by the divergence theorem. Does not depend
    on the triangulation, unlike the vertex mean. hac26.shape_library.pose uses the same
    expression, so a body posed by either lands in the same place."""
    v = np.asarray(verts, float)
    v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    tet = (v0 + v1 + v2) / 4.0
    w = np.einsum("ij,ij->i", v0 + v1 + v2, cr) / 18.0
    tot = w.sum()
    return (tet * w[:, None]).sum(0) / tot if abs(tot) > 1e-12 else v.mean(0)


def rescale_touch_z(verts: np.ndarray, faces: np.ndarray | None = None,
                    centre_xy: bool = True) -> np.ndarray:
    """The challenge pose: uniform scale and translation so that min z = -1 and max z = +1.
    A uniform scale does not change the mean-normalised curves.

    `centre_xy` decides what happens to x and y, and the answer depends on where the mesh
    came from:

    - A mesh that is **already in the challenge frame** -- a released public STL, a
      reconstruction this package produced, anything whose origin is the rotation axis --
      must be left alone: `centre_xy=False`. Its origin is the axis, and moving the body to
      put its centroid there moves it off. The released STLs say so. Posed to z in [-1, 1],
      their largest xy radius about the STL origin matches the published bounding-cylinder
      radius to within 0.6% (1.1198 vs 1.12, 1.4142 vs 1.42, 0.8782 vs 0.88), and centring
      them on the solid centroid makes model 2 read 1.4451 -- larger than the published
      *minimal* enclosing radius, which the true body cannot be -- and displaces it by 0.031.
    - A mesh with **no meaningful origin** -- a procedural library body, a random training
      shape -- has to be mounted on the axis somehow, and its centroid is as good a choice as
      any: `centre_xy=True`, the default, which is also the rule
      `hac26.shape_library.pose` uses so that a body posed either way lands in the same place.

    Pass `faces` whenever two meshes will be compared and `centre_xy` is on. With them the
    centre is the solid centroid; without them it is the vertex mean, which depends on the
    triangulation, so two meshes of the same body would be posed at different centres.
    """
    v = verts.copy()
    if centre_xy:
        c = v.mean(0) if faces is None else solid_centroid(v, faces)
        v[:, 0] -= c[0]
        v[:, 1] -= c[1]
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    if not zmax - zmin > 1e-12:
        # Dividing by it anyway returns inf and nan, a nan extent makes every occupancy grid
        # empty, and dice() reads two empty grids as two identical bodies and returns 1.0 --
        # so a degenerate body would score perfectly. hac26.shape_library.pose raises on
        # exactly this condition; so does this.
        raise ValueError("degenerate body: zero z extent")
    v[:, 2] -= 0.5 * (zmin + zmax)
    return v * (2.0 / (zmax - zmin))


def canonicalize_r(verts: np.ndarray) -> np.ndarray:
    """Scale xy so the largest axis distance is 1, leaving z (already in [-1, 1]) alone.

    The challenge publishes a bounding radius R per model, and the per-curve mean
    normalisation removes most of the information about the body's width relative to its
    height. So every model is trained on the canonical shape with xy radius 1 and the width is
    restored from R at reconstruction:

        train target : canonicalize_r(hull)              (r_max = 1)
        test  output : fit_to_cylinder(prediction, R)    (r_max = R)

    The two are exact inverses."""
    v = verts.copy()
    r = float(np.sqrt((v[:, :2] ** 2).sum(1)).max())
    if r > 1e-12:
        v[:, :2] /= r
    return v


def mesh_support(verts: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Support function h(u) = max over the vertices of <x, u>, sampled on `normals`. Exact
    for the convex hull of `verts`."""
    return (verts @ normals.T).max(axis=0)


# ---------- brute-force convex renderer ----------------------------------------------
def mesh_curves_convex(verts: np.ndarray, faces: np.ndarray, cameras: list, m: int,
                       curve_types: list, gamma: float = TRANSFER_EXPONENT,
                       sigma: float = 1.0, delta: float = 1.0, psi0: float = 0.0,
                       frame_area: float | None = None) -> np.ndarray:
    """Raw (unnormalized) curves of a CONVEX mesh, shape (n_curves, m).

    On a convex body nothing shadows, so summing the kernel over the facets is the whole
    forward model and agrees with a ray cast of the same body. The binary curves are
    thresholded at the level the body's own first frame gives, as the organisers threshold
    theirs at the level their first frame gives."""
    n, a = face_normals_areas(verts, faces)
    psi = psi_grid(m, sigma=sigma, psi0=psi0)
    v0 = body_frame_dirs(OMEGA0, psi)
    mu0 = n @ v0.T
    thr = curve_thresholds(n, a, cameras, m, curve_types, gamma=gamma,
                           frame_area=frame_area, sigma=sigma, delta=delta, psi0=psi0)
    out = []
    for cam, ctype, c in zip(cameras, curve_types, thr):
        v = body_frame_dirs(cam.omega(delta=delta), psi)
        mu = n @ v.T
        out.append(a @ kernel(mu, mu0, ctype, gamma=gamma, threshold=float(c)))
    return np.stack(out, axis=0)


# ---------- training-shape sampler ----------------------------------------------------
def sample_training_shape(rng: np.random.Generator, grid: NormalGrid,
                          p_flat: float = 0.0) -> dict:
    """A random posed body for the convex stage's training: its hull mesh, the EGI of the hull
    and metadata. `p_flat` is the fraction of flat-faced bodies mixed in."""
    meta = None
    if p_flat and rng.random() < p_flat:
        v, kind = sample_flat_shape(rng)
        v = rescale_touch_z(v)
        hv, hf = hull_mesh(v)
        g = mesh_to_egi(hv, hf, grid)
        return {"kind": kind, "verts": hv, "faces": hf, "g": g,
                "p": g / max(g.sum(), 1e-12), "meta": None}
    kind = rng.choice(["sh", "poly", "ellipsoid"], p=[0.6, 0.3, 0.1])
    if kind == "sh":
        v, f, meta = sh_lognormal_mesh(rng, L=int(rng.integers(4, 9)),
                                       amp=rng.uniform(0.2, 0.5))
    elif kind == "poly":
        v, _ = random_convex_polytope(rng, n_pts=int(rng.integers(12, 60)))
    else:
        v, f, meta = ellipsoid_mesh(rng)
    v = rescale_touch_z(v)
    hv, hf = hull_mesh(v)
    g = mesh_to_egi(hv, hf, grid)
    return {"kind": kind, "verts": hv, "faces": hf, "g": g,
            "p": g / max(g.sum(), 1e-12), "meta": meta}


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random orthogonal matrix, reflections included. Fine for orienting a random
    body; not a proper rotation."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.diag(r))


# ---------- flat-faced and few-face bodies -------------------------------------------
# The smooth families above contain no bodies with large flat facets, while the public models
# include one that is a cube. These generators add flat-faced and few-faced bodies.

_PLATONIC = {}


def platonic(kind: str) -> np.ndarray:
    """Vertices of a platonic solid, unit-ish scale (cached)."""
    if kind in _PLATONIC:
        return _PLATONIC[kind]
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    if kind == "tetra":
        v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    elif kind == "cube":
        v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    elif kind == "octa":
        v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    elif kind == "dodeca":
        v = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        v += [[0, s * 1 / phi, t * phi] for s in (-1, 1) for t in (-1, 1)]
        v += [[s * 1 / phi, t * phi, 0] for s in (-1, 1) for t in (-1, 1)]
        v += [[s * phi, 0, t * 1 / phi] for s in (-1, 1) for t in (-1, 1)]
        v = np.array(v, float)
    elif kind == "icosa":
        v = [[0, s, t * phi] for s in (-1, 1) for t in (-1, 1)]
        v += [[s, t * phi, 0] for s in (-1, 1) for t in (-1, 1)]
        v += [[s * phi, 0, t] for s in (-1, 1) for t in (-1, 1)]
        v = np.array(v, float)
    else:
        raise ValueError(kind)
    _PLATONIC[kind] = v / np.linalg.norm(v, axis=1).max()
    return _PLATONIC[kind]


def prism_mesh(rng: np.random.Generator) -> np.ndarray:
    """Right prism on a regular or jittered n-gon: the archetypal few-face body."""
    n = int(rng.integers(3, 9))
    a = np.arange(n) * 2 * np.pi / n + rng.uniform(0, 2 * np.pi)
    rad = 1.0 + rng.normal(0, 0.08, size=n) if rng.random() < 0.5 else np.ones(n)
    ring = np.stack([rad * np.cos(a), rad * np.sin(a)], axis=1)
    ring = ring * rng.uniform(0.6, 1.4, size=2)          # elliptical cross-section
    hz = rng.uniform(0.4, 1.8)
    top = np.hstack([ring * rng.uniform(0.55, 1.0), np.full((n, 1), hz)])   # allow taper
    bot = np.hstack([ring, np.full((n, 1), -hz)])
    return np.vstack([top, bot])


def faceted_mesh(rng: np.random.Generator) -> np.ndarray:
    """A smooth body sliced by a few random half-spaces, giving flat facets.

    This is the physically motivated one: real small bodies acquire flat faces from
    large impacts and from fracture along planes, so a cut ellipsoid is a much better
    model of a faceted asteroid than either a platonic solid or a Gaussian hull.
    """
    v, _, _ = sh_lognormal_mesh(rng, L=int(rng.integers(3, 7)), amp=rng.uniform(0.1, 0.35))
    v = v * rng.uniform(0.6, 1.4, size=3)
    for _ in range(int(rng.integers(1, 6))):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        off = np.quantile(v @ u, rng.uniform(0.55, 0.95))
        proj = v @ u
        cut = proj > off
        if cut.any():
            v[cut] -= np.outer(proj[cut] - off, u)       # project onto the cutting plane
    return v


def bilobe_mesh(rng: np.random.Generator) -> np.ndarray:
    """Hull of two offset ellipsoids -- a contact-binary silhouette. They need not overlap;
    the hull bridges them either way."""
    u, _ = icosphere(2)
    out = []
    sep = rng.uniform(0.4, 1.1)
    for k in (-1, 1):
        ax = rng.uniform(0.45, 1.0, size=3)
        c = np.zeros(3)
        c[0] = k * sep
        out.append(u * ax + c)
    return np.vstack(out)


def sample_flat_shape(rng: np.random.Generator) -> tuple:
    """Draw one flat-faced / few-face body. Returns (points, kind)."""
    kind = rng.choice(["platonic", "prism", "faceted", "bilobe"], p=[0.2, 0.3, 0.35, 0.15])
    if kind == "platonic":
        name = rng.choice(["tetra", "cube", "octa", "dodeca", "icosa"])
        v = platonic(name) * rng.uniform(0.6, 1.5, size=3)     # anisotropic: not just the solid
        kind = f"platonic_{name}"
    elif kind == "prism":
        v = prism_mesh(rng)
    elif kind == "faceted":
        v = faceted_mesh(rng)
    else:
        v = bilobe_mesh(rng)
    return v @ _random_rotation(rng).T, kind
