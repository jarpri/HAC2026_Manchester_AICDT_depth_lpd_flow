"""Training bodies for the flow, generated as level sets.

The secret bodies are 3-D prints, and the three public ones say of what: two real asteroid
shape models (Vesta, a nearly convex spheroid with a polar basin; Mithra, a contact binary)
and one geometric test solid (a sawed-off cube). So the library is drawn from the same
sources, and from one more, since a print need not be an asteroid at all: real asteroid
shape models and everyday printable objects, when directories of them are given, each enter
as a family with random anisotropic scaling and mirroring (`_real`). The procedural
families are the shapes real asteroids come in -- smooth lumpy potatoes, bilobed and
trilobed contact binaries, spinning tops with an equatorial ridge, angular faceted bodies --
plus geometric solids with saw cuts. The modifiers are the large features the lightcurves
and the competition's voxel and side-view measures can see: basins, saw cuts, an added lobe,
a ridge, moderate roughness. Nothing is generated below the scale the extraction grid
resolves: min_feature_radius gives that scale, and every sampled feature is above it by
construction, which tests/test_shape_library.py checks against the ranges themselves rather
than leaving it as a claim.

Convex bodies are kept, since a sawed-off cube is convex and the flow has to learn when
there is nothing to carve. What is set is the mix: `LibrarySpec.convexity_shares` gives the
share of the library in each band of volume over hull volume, and `sample_body` redraws a
body until it lands in the band it was dealt, so the deeply carved bands are covered whatever
the families would give on their own. `Body.convexity` records what came out.

The organisers chose per model how the body sits on its rotation axis: Vesta spins about its
shortest axis, Mithra was mounted along its longest with a tilt. So after extraction each
body is mounted on one of its principal axes, or at random, and tilted (`mount`);
`LibrarySpec.mount_weights` and `tilt_deg` set that distribution.

Bodies are built as fields f(x) with f < 0 inside. Unions are min, cuts are max(f, -g).
These are not true distances away from the zero set, but their sign is exact, and only the
sign and the location of the zero crossing matter here. On the occupancy grid the solid is
made one connected piece without voids before any triangle exists (`_repair`), and marching
cubes returns a closed oriented surface.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field as _dcfield
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

from .shapes import real_sh_basis, solid_centroid

__all__ = [
    "Body", "LibrarySpec", "sample_body", "build_library", "mount",
    "body_from_mesh", "body_from_convex_points", "read_shape_model", "load_shape_models",
    "mesh_volume", "hull_volume", "convexity_ratio", "is_edge_manifold",
    "n_components", "pose", "extract", "voxelise", "solid_frame",
    "sd_sphere", "sd_ellipsoid", "sd_box", "sd_cylinder", "sd_cone", "sd_torus",
    "sd_convex", "sd_star_sh", "op_union", "op_subtract", "op_intersect",
    "op_smooth_union", "op_displace", "decimate_mesh", "min_feature_radius",
]

Field = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- primitives
# Each returns f(p) for p of shape (..., 3), negative inside.

def _rot(axis_z: np.ndarray) -> np.ndarray:
    """Orthonormal frame whose third row is `axis_z`; rows map world -> local."""
    w = np.asarray(axis_z, float)
    w = w / max(np.linalg.norm(w), 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, w); u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return np.stack([u, v, w], axis=0)


def _unit(rng: np.random.Generator) -> np.ndarray:
    u = rng.normal(size=3)
    return u / np.linalg.norm(u)


def _angles(p: np.ndarray) -> tuple:
    """(r, theta, phi) of points (..., 3), with the origin sent to r = 0, theta = 0."""
    r = np.linalg.norm(p, axis=-1)
    safe = np.where(r > 1e-12, r, 1.0)
    u = p / safe[..., None]
    theta = np.arccos(np.clip(u[..., 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(u[..., 1], u[..., 0]), 2.0 * np.pi)
    return r, theta, phi


def sd_sphere(centre=(0, 0, 0), radius: float = 1.0) -> Field:
    c = np.asarray(centre, float)
    return lambda p: np.linalg.norm(p - c, axis=-1) - radius


def sd_ellipsoid(centre=(0, 0, 0), axes=(1, 1, 1), rot: np.ndarray | None = None) -> Field:
    """Ellipsoid field with the correct sign everywhere and approximately the true distance
    near the surface (Inigo Quilez's bound)."""
    c = np.asarray(centre, float)
    a = np.asarray(axes, float)
    R = np.eye(3) if rot is None else np.asarray(rot, float)

    def f(p):
        q = (p - c) @ R.T
        k0 = np.linalg.norm(q / a, axis=-1)
        k1 = np.linalg.norm(q / (a * a), axis=-1)
        return np.where(k1 > 1e-12, k0 * (k0 - 1.0) / np.maximum(k1, 1e-12), k0 - 1.0)
    return f


def sd_box(centre=(0, 0, 0), half=(1, 1, 1), rot: np.ndarray | None = None,
           round_r: float = 0.0) -> Field:
    c = np.asarray(centre, float)
    h = np.asarray(half, float) - round_r
    R = np.eye(3) if rot is None else np.asarray(rot, float)

    def f(p):
        q = np.abs((p - c) @ R.T) - h
        out = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        return out + np.minimum(q.max(axis=-1), 0.0) - round_r
    return f


def sd_cylinder(centre=(0, 0, 0), axis=(0, 0, 1), radius: float = 1.0,
                half_height: float = 1.0, round_r: float = 0.0) -> Field:
    c = np.asarray(centre, float)
    R = _rot(axis)

    def f(p):
        q = (p - c) @ R.T
        d_r = np.linalg.norm(q[..., :2], axis=-1) - (radius - round_r)
        d_z = np.abs(q[..., 2]) - (half_height - round_r)
        d = np.stack([d_r, d_z], axis=-1)
        return (np.linalg.norm(np.maximum(d, 0.0), axis=-1)
                + np.minimum(d.max(axis=-1), 0.0) - round_r)
    return f


def sd_cone(centre=(0, 0, 0), axis=(0, 0, 1), radius: float = 1.0,
            half_height: float = 1.0) -> Field:
    """A cone with its base at -half_height and its apex at +half_height along `axis`. The
    sign is exact; the value is a bound."""
    c = np.asarray(centre, float)
    R = _rot(axis)

    def f(p):
        q = (p - c) @ R.T
        z = q[..., 2]
        rad = np.linalg.norm(q[..., :2], axis=-1)
        # radius allowed at this height, negative above the apex
        allowed = radius * (half_height - z) / (2.0 * half_height)
        side = (rad - allowed) * np.cos(np.arctan2(radius, 2.0 * half_height))
        return np.maximum(side, -half_height - z)
    return f


def sd_torus(centre=(0, 0, 0), axis=(0, 0, 1), major: float = 1.0,
             minor: float = 0.3) -> Field:
    c = np.asarray(centre, float)
    R = _rot(axis)

    def f(p):
        q = (p - c) @ R.T
        return np.hypot(np.linalg.norm(q[..., :2], axis=-1) - major, q[..., 2]) - minor
    return f


def sd_convex(normals: np.ndarray, offsets: np.ndarray) -> Field:
    """Intersection of half-spaces n_j . x <= d_j, as max_j (n_j . x - d_j).

    This is the representation `hac26.field.ConvexCore` uses, so a convex hull enters the
    generator in the solver's own parameterisation.
    """
    n = np.asarray(normals, float)
    d = np.asarray(offsets, float)
    return lambda p: (p @ n.T - d).max(axis=-1)


def sd_star_sh(coeffs: np.ndarray, l_max: int, centre=(0, 0, 0), scale: float = 1.0,
               axes=(1.0, 1.0, 1.0), rot: np.ndarray | None = None) -> Field:
    """|q| - scale * exp(sum a_lm Y_lm(q/|q|)) in the ellipsoid coordinates q = R (x - c) / axes:
    a star-shaped body about its centre, an ellipsoid when the coefficients are zero."""
    c = np.asarray(centre, float)
    a = np.asarray(coeffs, float)
    ax = np.asarray(axes, float)
    R = np.eye(3) if rot is None else np.asarray(rot, float)

    def f(p):
        q = ((p - c) @ R.T) / ax
        r, theta, phi = _angles(q)
        B = real_sh_basis(l_max, theta.ravel(), phi.ravel())
        return r - scale * np.exp(a @ B).reshape(r.shape)
    return f


# --------------------------------------------------------------------------- operators

def op_union(*fs: Field) -> Field:
    return lambda p: np.min(np.stack([f(p) for f in fs], axis=0), axis=0)


def op_intersect(*fs: Field) -> Field:
    return lambda p: np.max(np.stack([f(p) for f in fs], axis=0), axis=0)


def op_subtract(a: Field, *bs: Field) -> Field:
    def f(p):
        out = a(p)
        for b in bs:
            out = np.maximum(out, -b(p))
        return out
    return f


def op_smooth_union(a: Field, b: Field, k: float = 0.1) -> Field:
    """Exponential smooth min of two fields, with `k` the fillet width.

    A hard min leaves a crease where two lobes meet; real necks are filleted.
    """
    def f(p):
        x, y = a(p), b(p)
        m = np.minimum(x, y)
        return m - k * np.log1p(np.exp(-np.abs(x - y) / max(k, 1e-9)))
    return f


def op_displace(a: Field, d: Callable[[np.ndarray], np.ndarray]) -> Field:
    return lambda p: a(p) + d(p)


def _sh_coeffs(rng: np.random.Generator, l_lo: int, l_hi: int, amp: float,
               decay: float) -> np.ndarray:
    """Random real-harmonic coefficients for degrees 1..l_hi, zero below l_lo, with the
    standard deviation of degree l falling as (1 + l)^-decay."""
    ls = np.concatenate([[l] * (2 * l + 1) for l in range(1, l_hi + 1)])
    a = rng.normal(0.0, amp / (1.0 + ls) ** decay)
    a[ls < l_lo] = 0.0
    return a


def _sh_displacement(rng: np.random.Generator, l_lo: int, l_hi: int, amp: float) -> Callable:
    """Radial roughness: a band-limited random field on the sphere, added to f."""
    a = _sh_coeffs(rng, l_lo, l_hi, amp, 1.1)

    def d(p):
        r, theta, phi = _angles(p)
        B = real_sh_basis(l_hi, theta.ravel(), phi.ravel())
        return (a @ B).reshape(r.shape)
    return d


def _ray_radius(f: Field, dirs: np.ndarray, t_max: float = 4.0, iters: int = 40) -> np.ndarray:
    """Distance from the origin to the surface along each unit direction (n, 3), by bisection
    on f(t u) = 0. The origin must be inside. For a body that is not star-shaped about the
    origin this is the last crossing before t_max, which is what a cutter placed from
    outside needs."""
    d = np.asarray(dirs, float)
    lo = np.zeros(len(d)); hi = np.full(len(d), t_max)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        inside = f(mid[:, None] * d) < 0.0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- mesh measures

def mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume by the divergence theorem. Positive for outward-oriented faces."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def hull_volume(verts: np.ndarray) -> float:
    return float(ConvexHull(verts).volume)


def convexity_ratio(verts: np.ndarray, faces: np.ndarray) -> float:
    """volume / hull volume: 1 for a convex body, smaller the more is carved away."""
    hv = hull_volume(verts)
    return abs(mesh_volume(verts, faces)) / hv if hv > 0 else 1.0


def is_edge_manifold(faces: np.ndarray) -> bool:
    """True when every edge is used by exactly two faces, once in each direction.

    This is the combinatorial part of a watertightness check and needs no trimesh.
    """
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    key = np.sort(e, axis=1)
    _, counts = np.unique(key, axis=0, return_counts=True)
    if not np.all(counts == 2):
        return False
    # orientation: each undirected edge must appear once in each direction
    fwd = {(int(a), int(b)) for a, b in e}
    return all((b, a) in fwd for a, b in fwd)


def n_components(verts: np.ndarray, faces: np.ndarray) -> int:
    """Connected components of the surface, by vertex adjacency across triangle edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    n = len(verts)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    used = np.unique(faces)
    lab = connected_components(g, directed=False)[1]
    return int(len(np.unique(lab[used])))


def decimate_mesh(verts: np.ndarray, faces: np.ndarray, extent: float, res: int) -> tuple:
    """Vertex-clustering decimation on a grid of `res` cells across `2 * extent`.

    Vertices in the same cell are merged into the first one seen and the triangles that
    collapse are dropped. Callers that only read the mesh on a grid of their own (occupancy,
    the curve renderer) lose nothing they can resolve and gain a much smaller face count.

    The result need not be edge-manifold: merging vertices can join two distinct edges into
    one. Do not run `is_edge_manifold` or `n_components` on a decimated mesh; decimate a copy
    for metrics and keep the original for validity checks.
    """
    cell = 2.0 * extent / max(res, 1)
    key = np.round(verts / cell).astype(np.int64)
    _, first_idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    v2 = verts[first_idx]
    f2 = inv[faces]
    keep = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    if not keep.any():
        return verts, faces
    return v2, f2[keep]


# --------------------------------------------------------------------------- grid -> mesh

def voxelise(f: Field, extent: float = 1.6, res: int = 96,
             chunk: int = 400_000) -> tuple:
    """Sample f on a res^3 grid over [-extent, extent]^3. Returns (values, spacing, origin).

    The grid is padded by one voxel of positive field on every side, so the zero set cannot
    touch the boundary and marching cubes cannot produce an open surface.
    """
    a = np.linspace(-extent, extent, res)
    g = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3)
    out = np.empty(len(g))
    for i in range(0, len(g), chunk):
        out[i:i + chunk] = f(g[i:i + chunk])
    vol = out.reshape(res, res, res)
    vol = np.pad(vol, 1, mode="constant", constant_values=float(np.abs(vol).max() + 1.0))
    spacing = 2.0 * extent / (res - 1)
    return vol, spacing, -extent - spacing


def _repair(vol: np.ndarray, eps: float) -> tuple:
    """Keep the largest solid component and fill interior voids, by editing the field.

    Returns (vol, n_solid_components, n_voids). Solid uses 6-connectivity, so two pieces
    touching only at a corner count as separate and the smaller is discarded rather than
    welded into a pinch. Background uses 6-connectivity too. The complementary pair (6 for
    solid, 26 for background) is the textbook choice, but marching cubes does not honour it:
    a background pocket joined to the outside only through a corner is not a void by the
    26-connected count, yet the interpolated surface seals it into an interior cavity, which
    the lightcurves cannot see and the surface checks then reject as a second component.
    Counting background at 6-connectivity fills those pockets instead.

    Discarded pieces are removed by raising f to at least `eps` on exactly their voxels.
    Voids are filled by lowering f to at most `-eps` inside them, which deletes the internal
    surface without creating a new crossing, since a void is surrounded by solid.
    """
    solid = vol < 0.0
    if not solid.any():
        return vol, 0, 0
    lab, k = ndimage.label(solid, structure=ndimage.generate_binary_structure(3, 1))
    n_solid = k
    if k > 1:
        sizes = ndimage.sum(solid, lab, index=np.arange(1, k + 1))
        keep = lab == (int(np.argmax(sizes)) + 1)
    else:
        keep = solid
    vol = np.where(solid & ~keep, np.maximum(vol, eps), vol)

    bg = vol >= 0.0
    lab_b, kb = ndimage.label(bg, structure=ndimage.generate_binary_structure(3, 1))
    outer = lab_b[0, 0, 0]
    void = bg & (lab_b != outer)
    n_void = int(kb - 1)
    if void.any():
        vol = np.where(void, np.minimum(vol, -eps), vol)
    return vol, n_solid, n_void


def extract(f: Field, extent: float = 1.6, res: int = 96) -> tuple:
    """Zero level set of f as a closed mesh. Returns (verts, faces, info).

    `_repair` leaves one solid voxel component and no interior void, but the extracted
    surface can still split at a voxel-scale pinch, so callers that need one component
    check `n_components` themselves. `info["clipped"]` is True when the body reached the
    edge of the sampled grid.
    """
    from skimage import measure

    vol, spacing, origin = voxelise(f, extent, res)
    # A field still negative on the outermost real shell gets closed off by the padding with
    # a flat wall that downstream code would read as real geometry. Indices 1 and -2 are the
    # first and last real shells; 0 and -1 are the pad.
    clipped = bool(min(vol[1].min(), vol[-2].min(), vol[:, 1].min(), vol[:, -2].min(),
                       vol[:, :, 1].min(), vol[:, :, -2].min()) < 0.0)
    eps = 1e-3 * max(float(np.abs(vol).max()), 1e-9)
    vol, n_solid, n_void = _repair(vol, eps)
    if not (vol < 0).any():
        raise ValueError("empty body: the field is positive everywhere on the grid")
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0, spacing=(spacing,) * 3)
    verts = verts + origin
    # marching_cubes assumes the inside has the higher value; here the inside is f < 0, so
    # the faces come out facing inward. Flip them, then check the volume sign to be sure.
    faces = faces[:, ::-1].copy()
    if mesh_volume(verts, faces) < 0:
        faces = faces[:, ::-1].copy()
    return verts, faces, {"n_solid_components": n_solid, "n_voids_filled": n_void,
                          "clipped": clipped}


# --------------------------------------------------------------------------- posing

def pose(verts: np.ndarray, radius: float | None = 1.0,
         centre: str = "volume", faces: np.ndarray | None = None) -> np.ndarray:
    """Challenge pose: z spans exactly [-1, 1], the xy centroid is on the z axis, and the
    largest xy radius is `radius`. `radius=None` leaves the xy scale alone.

    Scaling xy separately from z follows `hac26.shapes.canonicalize_r`: the library is built
    at a canonical xy radius and the true width is restored from the published bounding
    radius at reconstruction time.
    """
    v = np.asarray(verts, float).copy()
    # Same centring rule as hac26.shapes.rescale_touch_z, so a body posed here does not move
    # when re-posed there.
    c = (solid_centroid(v, faces) if centre == "volume" and faces is not None
         else v.mean(0))
    v[:, 0] -= c[0]
    v[:, 1] -= c[1]
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    if zmax - zmin < 1e-12:
        raise ValueError("degenerate body: zero z extent")
    v[:, 2] = 2.0 * (v[:, 2] - zmin) / (zmax - zmin) - 1.0
    if radius is not None:
        r = float(np.hypot(v[:, 0], v[:, 1]).max())
        if r > 1e-12:
            v[:, :2] *= radius / r
    return v


def solid_frame(verts: np.ndarray, faces: np.ndarray, res: int = 48) -> np.ndarray:
    """Rows: the principal axes of the solid, longest extent first, as a proper rotation.
    Taken from the occupancy of a grid, so a finely meshed region weighs no more than a
    coarse one."""
    extent = float(np.abs(verts).max()) * 1.05
    v, f = verts, faces
    if len(f) > 3000:
        v, f = decimate_mesh(np.asarray(verts, float), np.asarray(faces, np.int64), extent, res)
    occ = _parity_occupancy(np.asarray(v, float), np.asarray(f, np.int64), extent, res)
    idx = np.argwhere(occ).astype(float)
    if len(idx) < 4:
        return np.eye(3)
    p = idx * (2.0 * extent / (res - 1)) - extent
    p -= p.mean(0)
    _, vecs = np.linalg.eigh(np.cov(p, rowvar=False))
    R = vecs[:, ::-1].T
    if np.linalg.det(R) < 0:
        R[2] *= -1.0
    return R


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation by `angle` about the unit vector `axis`."""
    k = np.asarray(axis, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def mount(verts: np.ndarray, faces: np.ndarray, rng: np.random.Generator,
          mount_weights: dict, tilt_deg: float, max_tilt_deg: float,
          radius: float = 1.0) -> tuple:
    """Put the body on its rotation axis and pose it. Returns (verts, record).

    The axis is one of the body's principal axes (`mount_weights` keys "long", "middle",
    "short") or a random direction ("random"), tilted away by an angle drawn as |N(0,
    tilt_deg)| capped at max_tilt_deg, and spun by a random angle. Then `pose`."""
    v = np.asarray(verts, float)
    kind = _draw(mount_weights, rng)
    tilt = 0.0
    if kind == "random":
        R = _rand_rot(rng)
    else:
        frame = solid_frame(v, faces)                    # rows: long, middle, short
        axis = frame[{"long": 0, "middle": 1, "short": 2}[kind]]
        R = _rot(axis)                                   # sends the chosen axis to z
        tilt = min(abs(rng.normal(0.0, np.radians(tilt_deg))), np.radians(max_tilt_deg))
        ang = rng.uniform(0.0, 2.0 * np.pi)
        R = _rotation_about(np.array([np.cos(ang), np.sin(ang), 0.0]), tilt) @ R
    R = _rotation_about(np.array([0.0, 0.0, 1.0]), rng.uniform(0.0, 2.0 * np.pi)) @ R
    v = v @ R.T
    # The published bounding-cylinder radius of a model is its largest distance from the
    # rotation axis over half its height; posing removes it, so it is recorded here.
    c = solid_centroid(v, faces)
    r_xy = float(np.hypot(v[:, 0] - c[0], v[:, 1] - c[1]).max())
    z_half = 0.5 * float(v[:, 2].max() - v[:, 2].min())
    return pose(v, radius=radius, faces=faces), {"axis": kind, "tilt_deg": float(np.degrees(tilt)),
                                                "cylinder_radius": r_xy / max(z_half, 1e-12)}


# --------------------------------------------------------------------------- the recipes

@dataclass
class LibrarySpec:
    """Everything the sampler is allowed to vary, so a run is reproducible from it."""
    res: int = 96
    extent: float = 1.6
    radius: float = 1.0
    family_weights: dict = _dcfield(default_factory=lambda: {
        "potato": 0.14, "bilobe": 0.24, "trilobe": 0.08, "top": 0.06, "faceted": 0.10,
        "geometric": 0.12, "real": 0.14, "object": 0.12})
    n_modifiers: tuple = (1, 3)          # inclusive range, drawn per body. The lower end is
                                         # one rather than zero: a body with no modifier is
                                         # the bare base family, which for four of the eight
                                         # families is a convex primitive.
    mod_weights: dict = _dcfield(default_factory=lambda: {
        "basin": 0.28, "saw": 0.16, "roughness": 0.16, "waist": 0.12, "bulge": 0.12,
        "ridge": 0.08, "bite": 0.08})
    # Which principal axis becomes the rotation axis. The choice sets the body's bounding
    # radius, since the axis fixes what is height and what is width: mounted on its short
    # axis a body lies down and is wide, on its long axis it stands up and is narrow.
    # Measured over the families, the median radius is 2.9 short, 1.9 middle, 1.2 random and
    # 0.6 long, against published radii of which nine of ten lie between 0.67 and 1.48 and
    # one is 3.95. The weights are set from those measurements so that most of the library
    # sits in the published range while both tails stay covered.
    mount_weights: dict = _dcfield(default_factory=lambda: {
        "random": 0.50, "long": 0.20, "short": 0.20, "middle": 0.10})
    tilt_deg: float = 12.0               # scale of the tilt off the principal axis
    max_tilt_deg: float = 35.0
    # Band edges of volume over hull volume. The lowest edge is what decides whether the
    # library contains deeply carved bodies at all: with the lowest edge at 0.7 the deepest
    # band is unbounded below, the sampler stops at the first body that crosses 0.7, and the
    # band fills up just under its own edge. An edge at 0.55 makes the deepest band a target
    # in its own right.
    convexity_bins: tuple = (0.55, 0.7, 0.85, 0.95)
    convexity_shares: tuple = (0.20, 0.25, 0.25, 0.18, 0.12)  # share of bodies per band; ()
                                                              # leaves the mix to the families
    band_attempts: int = 12              # draws to land in the band before taking the nearest
    shape_models: tuple = ()             # files of real asteroid models, the "real" family
    object_models: tuple = ()            # files of everyday objects, the "object" family
    max_attempts: int = 8

    def weights(self) -> dict:
        """The family weights, with "real" and "object" dropped when there are no files
        for them."""
        w = dict(self.family_weights)
        if not self.shape_models:
            w.pop("real", None)
        if not self.object_models:
            w.pop("object", None)
        return w


@dataclass
class Body:
    """A finished library body: posed mesh, the recipe that made it, and its measurements."""
    verts: np.ndarray
    faces: np.ndarray
    recipe: dict
    info: dict

    @property
    def convexity(self) -> float:
        return self.info["convexity"]


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation.

    The QR of a Gaussian matrix, with the signs of R's diagonal fixed, is uniform over the
    orthogonal group, half of which has determinant minus one. Those are reflections, and
    applying one to a mesh turns it inside out: the vertices move but the winding does not
    follow, so the faces end up pointing inward and every later reader of the mesh has the
    inside and the outside the wrong way round. Negating a column makes the determinant one
    and leaves the distribution uniform over rotations, since composing the reflected half
    with a fixed reflection maps it onto the rotations.
    """
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    return q


def _draw(d: dict, rng: np.random.Generator) -> str:
    ks = list(d)
    w = np.array([d[k] for k in ks], float)
    return ks[int(rng.choice(len(ks), p=w / w.sum()))]


def _potato(rng: np.random.Generator, s: float, centre=(0, 0, 0),
            rot: np.ndarray | None = None, b_min: float = 0.35) -> tuple:
    """A smooth lumpy body: an ellipsoid of size s with its middle axis at least b_min of
    the long one, its radius modulated by harmonics of degree 2 to 4. Returns (field,
    record)."""
    b = np.exp(rng.uniform(np.log(b_min), 0.0))         # elongations up to 1 / b_min
    c = b * np.exp(rng.uniform(np.log(0.45), 0.0))
    axes = s * np.array([1.0, b, c])
    amp = rng.uniform(0.05, 0.28)
    L = 4
    a = _sh_coeffs(rng, 2, L, amp, rng.uniform(0.8, 1.5))
    return (sd_star_sh(a, L, centre=centre, scale=1.0, axes=axes, rot=rot),
            {"axes": (axes / s).round(3).tolist(), "amp": float(amp)})


def _lobes(rng: np.random.Generator, s: float, k: int) -> tuple:
    """k potatoes in a row, each touching or overlapping the next with a filleted neck, and
    bent a little off the line: contact binaries and dog-bones."""
    u = np.array([1.0, 0.0, 0.0])
    fs, recs, fracs = [], [], []
    pos = np.zeros(3)
    r_ahead = None                                       # the previous lobe's radius along u
    for i in range(k):
        size = s if i == 0 else s * rng.uniform(0.4, 1.0)
        # Each lobe's long axis lies near the line, as in the radar contact binaries, with
        # a tilt of up to forty degrees and a random roll about the line.
        w = _unit(rng); w -= (w @ u) * u; w /= np.linalg.norm(w)
        R = (_rotation_about(u, rng.uniform(0.0, 2.0 * np.pi))
             @ _rotation_about(w, rng.uniform(0.0, np.radians(30.0))))
        f, rec = _potato(rng, size, rot=R, b_min=0.55)
        if r_ahead is not None:
            r_back = float(_ray_radius(f, -u[None])[0])  # this lobe's radius toward the last
            # Centre separation as a fraction of the two lobes' radii along the line. At one
            # the lobes touch at a point and the neck is whatever the fillet leaves, which is
            # the dog-bone end of the family; well below it the lobes merge into a single
            # ovoid with no waist at all.
            frac = rng.uniform(0.72, 1.0)
            fracs.append(float(frac))
            bend = _unit(rng); bend -= (bend @ u) * u
            pos = pos + u * frac * (r_ahead + r_back) + bend * rng.uniform(0.0, 0.15) * size
        fs.append(_shift(f, pos))
        recs.append(rec)
        r_ahead = float(_ray_radius(f, u[None])[0])
    # The neck's fillet. The smooth union pushes the surface out by about 0.69 k at the
    # waist, so this is the floor on how narrow a neck the family can express and it is kept
    # well below the lobe size.
    k_fill = s * rng.uniform(0.03, 0.14)
    out = fs[0]
    for g in fs[1:]:
        out = op_smooth_union(out, g, k=k_fill)
    return out, {"n_lobes": k, "fillet": float(k_fill / s), "spacing": fracs, "lobes": recs}


def _shift(f: Field, c: np.ndarray) -> Field:
    c = np.asarray(c, float)
    return lambda p: f(p - c)


def _top(rng: np.random.Generator, s: float) -> tuple:
    """A spinning top: the radius profile r(z) = R (1 - |z/h|^a)^(1/a) on each side of the
    equator, a = 1 a double cone with a sharp ridge, a = 2 an ellipsoid, with a small
    non-axisymmetric modulation."""
    R_eq = s
    h_n, h_s = s * rng.uniform(0.55, 1.0), s * rng.uniform(0.55, 1.0)
    a = rng.uniform(1.0, 2.0)
    L = 3
    coef = _sh_coeffs(rng, 2, L, rng.uniform(0.0, 0.06), 1.0)

    def f(p):
        z = p[..., 2]
        rad = np.hypot(p[..., 0], p[..., 1])
        h = np.where(z >= 0, h_n, h_s)
        t = np.clip(np.abs(z) / h, 0.0, 1.0)
        prof = R_eq * (1.0 - t ** a) ** (1.0 / a)
        _, theta, phi = _angles(p)
        mod = np.exp((coef @ real_sh_basis(L, theta.ravel(), phi.ravel())).reshape(z.shape))
        d = rad - prof * mod
        return np.where(np.abs(z) > h, np.maximum(d, np.abs(z) - h), d)
    return f, {"exponent": float(a), "heights": [float(h_n / s), float(h_s / s)]}


def _faceted(rng: np.random.Generator, s: float) -> tuple:
    """A potato cut by several planes, each removing a cap: an angular body."""
    f, rec = _potato(rng, s)
    n = int(rng.integers(4, 15))
    dirs = np.stack([_unit(rng) for _ in range(n)])
    r = _ray_radius(f, dirs)
    d = r * rng.uniform(0.80, 0.97, n)
    return op_intersect(f, sd_convex(dirs, d)), {"n_planes": n, **rec}


def _geometric(rng: np.random.Generator, s: float) -> tuple:
    """A test solid: box, cylinder, cone, prism, capsule, superellipsoid, or a Platonic solid,
    with zero to three saw cuts that take a corner or an edge off."""
    kind = _draw({"box": 0.25, "cylinder": 0.15, "cone": 0.08, "prism": 0.15,
                  "capsule": 0.10, "superellipsoid": 0.15, "platonic": 0.12}, rng)
    R = _rand_rot(rng)
    rec = {"solid": kind}
    if kind == "box":
        half = s * np.array([1.0, rng.uniform(0.5, 1.0), rng.uniform(0.2, 1.0)])
        f = sd_box(half=half, rot=R, round_r=s * rng.uniform(0.0, 0.12))
        rec["half"] = (half / s).round(3).tolist()
    elif kind == "cylinder":
        f = sd_cylinder(axis=R[2], radius=s * rng.uniform(0.5, 1.0),
                        half_height=s * rng.uniform(0.4, 1.2), round_r=s * rng.uniform(0.0, 0.1))
    elif kind == "cone":
        f = sd_cone(axis=R[2], radius=s * rng.uniform(0.6, 1.0), half_height=s * rng.uniform(0.6, 1.2))
    elif kind == "prism":
        n = int(rng.integers(3, 9))
        ang = np.arange(n) * 2 * np.pi / n
        nrm = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1)
        d = s * rng.uniform(0.55, 1.0) * np.ones(n)
        hz = s * rng.uniform(0.4, 1.2)
        base = sd_convex(np.vstack([nrm, [[0, 0, 1.0], [0, 0, -1.0]]]),
                         np.concatenate([d, [hz, hz]]))
        f = (lambda p, g=base, R=R: g(p @ R.T))
        rec["n_sides"] = n
    elif kind == "capsule":
        r = s * rng.uniform(0.35, 0.7); hz = s * rng.uniform(0.3, 1.0)
        f = sd_cylinder(axis=R[2], radius=r, half_height=hz + r, round_r=r * 0.999)
    elif kind == "superellipsoid":
        ax = s * np.array([1.0, rng.uniform(0.5, 1.0), rng.uniform(0.4, 1.0)])
        e = rng.uniform(2.5, 8.0)

        def f(p, ax=ax, e=e, R=R):
            q = np.abs((p @ R.T) / ax)
            return (q ** e).sum(-1) ** (1.0 / e) - 1.0
        rec["exponent"] = float(e)
    else:
        from .shapes import platonic
        name = ["tetra", "octa", "dodeca", "icosa"][int(rng.integers(0, 4))]
        pts = platonic(name) @ R.T
        hull = ConvexHull(pts)
        f = sd_convex(hull.equations[:, :3], -hull.equations[:, 3] * s)
        rec["solid"] = name
    n_cut = int(rng.choice(4, p=[0.35, 0.35, 0.2, 0.1]))
    if n_cut:
        dirs = np.stack([_unit(rng) for _ in range(n_cut)])
        r = _ray_radius(f, dirs)
        f = op_intersect(f, sd_convex(dirs, r * rng.uniform(0.55, 0.9, n_cut)))
    rec["n_saw_cuts"] = n_cut
    return f, rec


_MODEL_CACHE: dict = {}


def _real(rng: np.random.Generator, spec: "LibrarySpec", files: tuple) -> tuple:
    """One of `files` (a real shape model or an everyday object), stretched by up to fifteen
    percent along each axis of a random frame and mirrored half the time."""
    path = files[int(rng.integers(0, len(files)))]
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = read_shape_model(path)
    v, faces = _MODEL_CACHE[path]
    v = v - solid_centroid(v, faces)
    R = _rand_rot(rng)
    stretch = rng.uniform(0.85, 1.15, 3)
    v = (v @ R.T) * stretch
    if rng.random() < 0.5:
        v = v * np.array([1.0, 1.0, -1.0])
        faces = faces[:, ::-1]
    v = v / max(float(np.linalg.norm(v, axis=1).max()), 1e-12) * GRID_FILL * spec.extent
    occ = _parity_occupancy(v, faces, spec.extent, spec.res)
    return (_field_from_occupancy(occ, spec.extent, spec.res),
            {"model": Path(path).name, "stretch": stretch.round(3).tolist()})


def _base(rng: np.random.Generator, kind: str, s: float, spec: "LibrarySpec") -> tuple:
    if kind == "potato":
        return _potato(rng, s)
    if kind == "bilobe":
        return _lobes(rng, s, 2)
    if kind == "trilobe":
        return _lobes(rng, s, 3)
    if kind == "top":
        return _top(rng, s)
    if kind == "faceted":
        return _faceted(rng, s)
    if kind == "geometric":
        return _geometric(rng, s)
    if kind == "real":
        return _real(rng, spec, spec.shape_models)
    if kind == "object":
        return _real(rng, spec, spec.object_models)
    raise ValueError(kind)


GRID_FILL = 0.85         # a body's largest radius from its centroid, as a share of the
                         # grid's half-width; the rest is margin against clipping


def _centre_and_fit(f: Field, extent: float, res: int = 32) -> Field:
    """Shift the field so its solid centroid is at the origin and scale it so its largest
    radius is GRID_FILL of the grid, from a coarse sample of the solid. The sign of f is
    unchanged, so nothing downstream cares that the values are no longer distances."""
    a = np.linspace(-4.0, 4.0, res)
    g = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = f(g) < 0.0
    if not inside.any():
        return f
    pts = g[inside]
    c = pts.mean(0)
    r_max = float(np.linalg.norm(pts - c, axis=1).max()) + (a[1] - a[0])
    lam = GRID_FILL * extent / max(r_max, 1e-9)
    return lambda p: f(p / lam + c)


def _surface_direction(f: Field, rng: np.random.Generator, s: float) -> tuple:
    """A random direction and the surface radius along it, redrawn while the ray finds
    nothing but a sliver, which happens when the origin is not inside a bent body."""
    for _ in range(8):
        u = _unit(rng)
        rb = float(_ray_radius(f, u[None])[0])
        if rb > 0.25 * s:
            return u, rb
    return u, rb


def min_feature_radius(res: int, extent: float, voxels_across: float = 2.5) -> float:
    """Smallest sphere radius marching cubes can render as a round bowl on a grid of `res`
    samples across `2 * extent`."""
    spacing = 2.0 * extent / max(res - 1, 1)
    return voxels_across * spacing


CUT_SHARE = 0.6          # deepest a cutter may reach, as a share of the body's thickness
                         # along the direction it comes in on. A cut that goes right through
                         # parts the body, and `_repair` keeps the larger piece, so the draw
                         # becomes a fragment carrying the recipe of the body it was cut from
                         # -- silently, since every check in `_finish` runs after the repair.
                         # Measured over the families, uncapped cutters part about a quarter
                         # of all bodies; the base families on their own part none.


def _cut_depth(f: Field, u: np.ndarray, rb: float, depth: float, floor: float) -> float:
    """The depth a cutter coming in along `u` may reach: what was asked for, capped at
    CUT_SHARE of the body's through-thickness there. Returns 0.0 when the body is too thin
    for any cut the grid could render, which the caller takes as "leave this one out"."""
    back = float(_ray_radius(f, -u[None])[0])
    cap = CUT_SHARE * (rb + max(back, 0.0))
    depth = min(depth, cap)
    return depth if depth >= floor else 0.0


def _apply_modifier(f: Field, rng: np.random.Generator, kind: str, s: float,
                    floor: float = 0.0) -> tuple:
    """One large-scale edit of `f`. Every cutter and lobe is placed relative to the body's
    own surface along its direction, so the edit lands on the body whatever its shape.

    `floor` is the smallest feature the extraction grid resolves (min_feature_radius). A cut
    below it is not rendered as a cut: marching cubes returns a body pinched or broken where
    the cutter went, the repair pass then keeps only the largest piece, and what is written to
    the library is a fragment labelled with the recipe of the body it was cut from. So a cut
    that would fall below the floor is widened to it rather than drawn as asked."""
    if kind == "basin":
        k = int(rng.integers(1, 4))
        cuts = []
        for _ in range(k):
            u, rb = _surface_direction(f, rng, s)
            rho = s * rng.uniform(0.3, 0.9)                     # cutter radius
            # How far the cutter dips below the surface, in units of its own radius. A local
            # plane cut to depth d by a sphere of radius rho leaves a mouth of radius
            # sqrt(d (2 rho - d)), so the ratio of mouth to depth is sqrt(2 rho / d - 1):
            # below one the cut is a dish, at one it is a hemispherical bowl, and above one
            # the mouth is narrower than the cut is deep and the rim overhangs. Past two the
            # cutter closes over and leaves a cavity with no mouth, which the repair pass
            # fills in again, so the range stops short of it. The lower end is set by the
            # grid: at the smallest cutter a shallower cut is a dent under one cell deep,
            # which the extraction cannot render as a bowl, so drawing one wastes the draw.
            frac = rng.uniform(0.35, 1.35)
            # The cut has to be renderable: it is `frac * rho` deep and its mouth has radius
            # `rho * sqrt(frac (2 - frac))`, and the smaller of the two decides whether the
            # grid sees a bowl or a pinch. Widening the cutter raises both together.
            rho = max(rho, floor / min(frac, np.sqrt(frac * (2.0 - frac))))
            depth = _cut_depth(f, u, rb, rho * frac, floor)
            if depth <= 0.0:
                continue
            cuts.append(sd_sphere(u * (rb + rho - depth), rho))
        if not cuts:
            return f, {"n_basins": 0}
        return op_subtract(f, *cuts), {"n_basins": len(cuts)}
    if kind == "saw":
        k = int(rng.integers(1, 3))
        dirs = np.stack([_unit(rng) for _ in range(k)])
        r = _ray_radius(f, dirs)
        return op_intersect(f, sd_convex(dirs, r * rng.uniform(0.6, 0.92, k))), {"n_saw": k}
    if kind == "bulge":
        u, rb = _surface_direction(f, rng, s)
        ax = s * rng.uniform(0.2, 0.5, 3)
        lobe = sd_ellipsoid(u * rb * rng.uniform(0.6, 0.95), ax, _rand_rot(rng))
        return op_smooth_union(f, lobe, k=s * rng.uniform(0.03, 0.12)), {"bulge": (ax / s).round(3).tolist()}
    if kind == "roughness":
        L = int(rng.integers(5, 11))
        return op_displace(f, _sh_displacement(rng, 4, L, s * rng.uniform(0.01, 0.05))), {"rough_L": L}
    if kind == "waist":
        # A pinch right around the body: `ridge` with the sign of the displacement reversed,
        # over a wider band. It is the only modifier that moves the whole outline rather than
        # a patch of it, which is what the challenge's side-view measure compares -- the
        # projection of a hull is the hull of the projection, so a waist is a concavity no
        # convex reconstruction can produce from any direction.
        #
        # The amplitude is capped well below the body size on purpose. A pinch deep enough to
        # part the body leaves two pieces, and `_repair` keeps the larger and discards the
        # rest, so what reaches the library is a fragment carrying the recipe of the body it
        # was cut from rather than a rejected draw.
        w = _unit(rng)
        c = rng.uniform(-0.25, 0.25) * s
        width = max(s * rng.uniform(0.18, 0.40), floor)
        amp = s * rng.uniform(0.08, 0.18)

        def d(p, w=w, c=c, amp=amp, width=width):
            return amp * np.exp(-((p @ w - c) ** 2) / (2.0 * width * width))
        return op_displace(f, d), {"waist": float(amp / s)}
    if kind == "bite":
        # One lobe-scale piece taken out of the limb with an ellipsoidal cutter. A `basin` is
        # a sphere dishing a face; this is bigger, single, and not round, so it leaves a
        # facet-and-edge scar rather than a bowl.
        u, rb = _surface_direction(f, rng, s)
        ax = np.maximum(s * rng.uniform(0.35, 0.75, 3), floor)
        R = _rand_rot(rng)
        # How far the cutter reaches from its own centre along u. sd_ellipsoid measures in
        # the frame q = (p - c) @ R.T, so u in that frame is u @ R.T.
        r_u = 1.0 / max(float(np.sqrt((((u @ R.T) / ax) ** 2).sum())), 1e-12)
        # Depth in units of that reach, on the same reasoning as `basin`: at one the cutter's
        # centre sits on the surface and the scar is as deep as it is wide, and past that the
        # mouth narrows. The cutter is widened if either the depth or the mouth would fall
        # below what the grid resolves.
        frac = rng.uniform(0.8, 1.4)
        grow = floor / max(r_u * min(frac, np.sqrt(frac * (2.0 - frac))), 1e-12)
        if grow > 1.0:
            ax, r_u = ax * grow, r_u * grow
        depth = _cut_depth(f, u, rb, r_u * frac, floor)
        if depth <= 0.0:
            return f, {"bite": None}
        centre = u * (rb + r_u - depth)
        return (op_subtract(f, sd_ellipsoid(centre, ax, R)),
                {"bite": (ax / s).round(3).tolist(), "bite_depth": round(depth / s, 3)})
    if kind == "ridge":
        w = _unit(rng)
        c = rng.uniform(-0.3, 0.3) * s
        amp = s * rng.uniform(0.04, 0.12)
        width = s * rng.uniform(0.08, 0.2)

        def d(p, w=w, c=c, amp=amp, width=width):
            return -amp * np.exp(-((p @ w - c) ** 2) / (2.0 * width * width))
        return op_displace(f, d), {"ridge": float(amp / s)}
    raise ValueError(kind)


def _finish(f: Field, rng: np.random.Generator, spec: LibrarySpec, recipe: dict,
            attempt: int) -> Body | None:
    """Extract, check, mount and measure. None when the body fails a check."""
    try:
        v, fc, info = extract(f, extent=spec.extent, res=spec.res)
    except (ValueError, RuntimeError):
        return None
    if len(fc) < 100 or info.get("clipped"):
        return None
    if not is_edge_manifold(fc) or n_components(v, fc) != 1:
        return None
    v, rec_mount = mount(v, fc, rng, spec.mount_weights, spec.tilt_deg, spec.max_tilt_deg,
                         spec.radius)
    # Orientation is checked again after the mount, not only after the extraction. A mesh
    # turned inside out is watertight, winding-consistent and of the right convexity, so
    # every other check here passes it, and the first thing to notice is whatever asks it
    # which side is inside: the signed distance the fit regresses on comes back negated, and
    # the body is fitted as its own complement. Nothing about that is visible in the fit's
    # residual. The repair is the same one the extraction makes.
    if mesh_volume(v, fc) < 0.0:
        fc = fc[:, ::-1].copy()
    recipe["mount"] = rec_mount
    info.update({"convexity": convexity_ratio(v, fc), "attempt": attempt,
                 "n_faces": len(fc), "n_verts": len(v),
                 "cylinder_radius": rec_mount["cylinder_radius"]})
    return Body(v, fc, recipe, info)


def _one_body(rng: np.random.Generator, spec: LibrarySpec) -> Body:
    """One posed, closed, single-component body of a family drawn from the weights. A body
    that fails the checks is redrawn within the same family; RuntimeError after
    `spec.max_attempts` failures."""
    kind = _draw(spec.weights(), rng)
    s = GRID_FILL * spec.extent                          # the size every body is brought to
    severed = None                                       # the best of a bad job, if it comes to it
    for attempt in range(spec.max_attempts):
        f, rec = _base(rng, kind, 1.0, spec)
        if kind not in ("real", "object"):
            f = _centre_and_fit(f, spec.extent)
        recipe = {"base": kind, **rec, "mods": []}
        for _ in range(int(rng.integers(spec.n_modifiers[0], spec.n_modifiers[1] + 1))):
            mk = _draw(spec.mod_weights, rng)
            f, mrec = _apply_modifier(f, rng, mk, s,
                                      min_feature_radius(spec.res, spec.extent))
            recipe["mods"].append({"kind": mk, **mrec})
        body = _finish(f, rng, spec, recipe, attempt)
        if body is None:
            continue
        # `_repair` has already thrown the smaller pieces away by this point, so nothing in
        # `_finish` can see that the body was parted: it is watertight, one component and of
        # a plausible convexity. Only the count `extract` recorded says so. Prefer a body
        # that was never parted -- but prefer a parted one to no body at all, because the
        # caller builds a library of thousands and a raise here kills the whole pool. The
        # count stays in `info` either way, so the manifest says which happened.
        if int(body.info.get("n_solid_components", 1)) > 1:
            if severed is None:
                severed = body
            continue
        return body
    if severed is not None:
        return severed
    raise RuntimeError(f"no {kind} body passed the checks in {spec.max_attempts} attempts")


def sample_body(rng: np.random.Generator, spec: LibrarySpec | None = None) -> Body:
    """One body. With `spec.convexity_shares` set, the body is dealt a band of volume over
    hull volume first and bodies are drawn until one lands in it, up to `spec.band_attempts`,
    after which the nearest miss is kept; so the library's convexity mix follows the shares
    whatever the families' own tendencies. `info["band"]` records the band dealt."""
    spec = spec or LibrarySpec()
    if not spec.convexity_shares:
        return _one_body(rng, spec)
    edges = (-np.inf,) + tuple(spec.convexity_bins) + (np.inf,)
    shares = np.asarray(spec.convexity_shares, float)
    band = int(rng.choice(len(shares), p=shares / shares.sum()))
    lo, hi = edges[band], edges[band + 1]
    best, best_miss = None, np.inf
    for _ in range(spec.band_attempts):
        body = _one_body(rng, spec)
        c = body.convexity
        miss = 0.0 if lo <= c < hi else min(abs(c - lo), abs(c - hi))
        if miss < best_miss:
            best, best_miss = body, miss
        if miss == 0.0:
            break
    best.info["band"] = band
    best.info["band_hit"] = bool(best_miss == 0.0)
    return best


def build_library(n: int, seed: int = 0, spec: LibrarySpec | None = None,
                  progress: bool = False) -> list:
    """`n` independent bodies. Body `i` uses its own generator seeded from (seed, i), so the
    library is reproducible and any single body can be rebuilt without the rest."""
    spec = spec or LibrarySpec()
    out = []
    for i in range(n):
        b = sample_body(np.random.default_rng([seed, i]), spec)
        out.append(b)
        if progress and (i % 10 == 0 or i == n - 1):
            print(f"  body {i + 1}/{n}  base={b.recipe['base']:<10} "
                  f"conv={b.convexity:.3f}  faces={b.info['n_faces']}", flush=True)
    return out


# --------------------------------------------------------------------------- ingestion

def read_shape_model(path: str) -> tuple:
    """(verts, faces) from a shape-model file: STL, Wavefront OBJ, ASCII PLY, or the plain
    vertex and facet lists radar and PDS models come in (`.wf`, `.tab`, `.txt`: lines of
    three numbers, or an index followed by three, with or without `v`/`f` prefixes; faces
    numbered from one). Faces with more than three corners are fanned into triangles."""
    p = Path(path)
    if p.suffix.lower() == ".stl":
        from .stl_io import load_stl
        v, f = load_stl(str(p))
        return np.asarray(v, float), np.asarray(f, np.int64)
    text = p.read_text(errors="ignore").splitlines()
    if p.suffix.lower() == ".ply":
        return _read_ply_ascii(text)
    v_rows, f_rows, plain = [], [], []
    for line in text:
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        key = t[0].lower()
        if key == "v" and len(t) >= 4:
            v_rows.append(t[1:])
        elif key == "f" and len(t) >= 4:
            f_rows.append([x.split("/")[0] for x in t[1:]])
        elif key[0] in "-+.0123456789" and len(t) in (3, 4):
            plain.append(t)
    verts, faces = [], []
    if v_rows:
        # "v i x y z" numbers its rows; then the faces are "f i a b c" too
        indexed = all(len(r) >= 4 and _is_int(r[0]) for r in v_rows)
        verts = [[float(x) for x in (r[1:4] if indexed else r[:3])] for r in v_rows]
        for r in f_rows:
            idx = [int(float(x)) for x in (r[1:] if indexed else r)]
            faces += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    else:
        for t in plain:
            nums = t[1:] if len(t) == 4 else t         # a leading index is dropped
            if all(_is_int(x) for x in nums):
                faces.append([int(float(x)) for x in nums])
            else:
                verts.append([float(x) for x in nums])
    if not verts or not faces:
        raise ValueError(f"{path}: no vertices and faces found")
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    if f.min() == 1:
        f = f - 1
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"{path}: face indices out of range")
    return v, f


def _is_int(x: str) -> bool:
    try:
        return float(x).is_integer() and "." not in x and "e" not in x.lower()
    except ValueError:
        return False


def _read_ply_ascii(lines: list) -> tuple:
    n_v = n_f = 0
    i = 0
    for i, line in enumerate(lines):
        t = line.split()
        if t[:2] == ["element", "vertex"]:
            n_v = int(t[2])
        elif t[:2] == ["element", "face"]:
            n_f = int(t[2])
        elif t and t[0] == "end_header":
            break
    body = lines[i + 1:]
    v = np.array([[float(x) for x in body[k].split()[:3]] for k in range(n_v)])
    faces = []
    for k in range(n_v, n_v + n_f):
        t = [int(x) for x in body[k].split()]
        idx = t[1:1 + t[0]]
        faces += [[idx[0], idx[j], idx[j + 1]] for j in range(1, len(idx) - 1)]
    return v, np.asarray(faces, np.int64)


def load_shape_models(directory: str) -> list:
    """Paths of every readable shape model directly in `directory`. Files that cannot be
    parsed are skipped with a warning."""
    out = []
    for p in sorted(Path(directory).glob("*")):
        if p.suffix.lower() not in (".stl", ".obj", ".ply", ".wf", ".tab", ".txt"):
            continue
        try:
            v, f = read_shape_model(str(p))
            if len(f) >= 100:
                out.append(str(p))
                _MODEL_CACHE[str(p)] = (v, f)
        except Exception as e:                                   # noqa: BLE001
            warnings.warn(f"skipped {p.name}: {e}")
    return out


def body_from_mesh(verts: np.ndarray, faces: np.ndarray,
                   rng: np.random.Generator | None = None,
                   spec: LibrarySpec | None = None,
                   add_modifiers: int = 0) -> Body:
    """Bring an external mesh into the library unchanged in shape: voxelised by ray parity
    along z (which needs it closed but not oriented), repaired, re-meshed, mounted and posed
    like a generated body, with `add_modifiers` edits on top."""
    spec = spec or LibrarySpec()
    rng = rng or np.random.default_rng(0)
    v = np.asarray(verts, float)
    v = v - solid_centroid(v, np.asarray(faces, np.int64))
    v = v / max(float(np.linalg.norm(v, axis=1).max()), 1e-12) * GRID_FILL * spec.extent
    f0 = _field_from_occupancy(_parity_occupancy(v, np.asarray(faces, np.int64),
                                                 spec.extent, spec.res), spec.extent, spec.res)
    for attempt in range(spec.max_attempts):
        f = f0
        recipe = {"base": "mesh", "source_faces": int(len(faces)), "mods": []}
        for _ in range(add_modifiers):
            mk = _draw(spec.mod_weights, rng)
            f, mrec = _apply_modifier(f, rng, mk, GRID_FILL * spec.extent,
                                      min_feature_radius(spec.res, spec.extent))
            recipe["mods"].append({"kind": mk, **mrec})
        body = _finish(f, rng, spec, recipe, attempt)
        if body is not None:
            return body
    raise RuntimeError(f"ingested mesh never passed the checks in {spec.max_attempts} attempts")


def body_from_convex_points(points: np.ndarray, rng: np.random.Generator,
                            spec: LibrarySpec | None = None,
                            n_modifiers: int = 2) -> Body:
    """A convex model (DAMIT) as the starting field, with `n_modifiers` edits on top."""
    spec = spec or LibrarySpec()
    p = np.asarray(points, float)
    p = p - p.mean(0)
    p = p / max(float(np.abs(p).max()), 1e-12)
    hull = ConvexHull(p)
    f0 = _centre_and_fit(sd_convex(hull.equations[:, :3], -hull.equations[:, 3]), spec.extent)
    for attempt in range(spec.max_attempts):
        f = f0
        recipe = {"base": "damit_convex", "n_planes": int(len(hull.equations)), "mods": []}
        for _ in range(n_modifiers):
            mk = _draw(spec.mod_weights, rng)
            f, mrec = _apply_modifier(f, rng, mk, GRID_FILL * spec.extent,
                                      min_feature_radius(spec.res, spec.extent))
            recipe["mods"].append({"kind": mk, **mrec})
        body = _finish(f, rng, spec, recipe, attempt)
        if body is not None:
            return body
    raise RuntimeError("convex model never passed the checks")


PARITY_TILE = 8          # see _parity_occupancy

# Sub-voxel offsets added to the sample columns so a column never lands exactly on a triangle
# edge. The point-in-triangle test is closed on its edges, so a column on an edge shared by
# two triangles would be counted twice and the whole column would invert. The two offsets
# differ so that a column on a face diagonal (x == y, as in any axis-aligned box) is moved
# off it too. Fixed constants keep results reproducible.
_JITTER = 1e-7
_JX = _JITTER * 0.6180339887498949           # 1/phi
_JY = _JITTER * 0.4142135623730951           # sqrt(2) - 1


def _parity_occupancy(verts: np.ndarray, faces: np.ndarray, extent: float,
                      res: int, axis: np.ndarray | None = None,
                      max_elems: float = 4e6) -> np.ndarray:
    """Occupancy of a res^3 grid by counting triangle crossings along each (x, y) column.

    A sample point is inside when the number of crossings strictly above it is odd. This is
    correct for any closed surface whatever its face orientation, and needs no trimesh.

    Columns are walked in tiles, each tile testing only the triangles whose xy box overlaps
    it; the result does not depend on the tile size.

    `axis` replaces the default `linspace(-extent, extent, res)` sample coordinates, for a
    caller whose grid is cell centres (`hac26.recon`). It must have `res` entries.
    """
    a = np.linspace(-extent, extent, res) if axis is None else np.asarray(axis, float)
    if len(a) != res:
        raise ValueError(f"axis has {len(a)} samples but res is {res}")
    zs = a
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    tmin = np.minimum(np.minimum(v0, v1), v2)
    tmax = np.maximum(np.maximum(v0, v1), v2)
    # One more bin than there are planes: bin i holds the crossings with zs[i-1] < z <= zs[i],
    # so bin 0 holds crossings at or below the lowest plane, which no plane should count.
    count = np.zeros((res * res, res + 1), dtype=np.int32)
    # Tile side in columns. Only the speed depends on it, not the result.
    tile = max(1, min(res, PARITY_TILE))
    spacing = float(a[1] - a[0]) if res > 1 else 2.0 * extent
    jx, jy = _JX * spacing, _JY * spacing        # see _JITTER
    for i0 in range(0, res, tile):
        xs = a[i0:i0 + tile]
        in_x = (tmin[:, 0] <= xs[-1]) & (tmax[:, 0] >= xs[0])
        if not in_x.any():
            continue
        for j0 in range(0, res, tile):
            ys = a[j0:j0 + tile]
            sel = np.nonzero(in_x & (tmin[:, 1] <= ys[-1]) & (tmax[:, 1] >= ys[0]))[0]
            if not len(sel):
                continue
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            px, py = X.ravel() + jx, Y.ravel() + jy
            # global column index of each tile column: i * res + j
            gcol = ((np.arange(i0, i0 + len(xs))[:, None] * res)
                    + np.arange(j0, j0 + len(ys))[None, :]).ravel()
            chunk = max(1, int(max_elems // max(len(px), 1)))
            for t in range(0, len(sel), chunk):
                k = sel[t:t + chunk]
                A, B, C = v0[k], v1[k], v2[k]
                d = ((B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0])
                     + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1]))
                ok = np.abs(d) > 1e-14
                if not ok.any():
                    continue
                A, B, C, d = A[ok], B[ok], C[ok], d[ok]
                l1 = ((B[None, :, 1] - C[None, :, 1]) * (px[:, None] - C[None, :, 0])
                      + (C[None, :, 0] - B[None, :, 0]) * (py[:, None] - C[None, :, 1])) / d
                l2 = ((C[None, :, 1] - A[None, :, 1]) * (px[:, None] - C[None, :, 0])
                      + (A[None, :, 0] - C[None, :, 0]) * (py[:, None] - C[None, :, 1])) / d
                l3 = 1.0 - l1 - l2
                inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
                if not inside.any():
                    continue
                zh = l1 * A[None, :, 2] + l2 * B[None, :, 2] + l3 * C[None, :, 2]
                col, tri = np.nonzero(inside)
                idx = np.clip(np.searchsorted(zs, zh[col, tri]), 0, res)
                np.add.at(count, (gcol[col], idx), 1)
    # Crossings strictly above plane j are those in bins j+1 and up; the slice drops bin 0.
    occ = (np.cumsum(count[:, ::-1], axis=1)[:, ::-1] % 2 == 1)[:, 1:]
    return occ.reshape(res, res, res)


def _field_from_occupancy(occ: np.ndarray, extent: float, res: int,
                          smooth: float = 0.7) -> Field:
    """A signed distance field from a binary occupancy: distance to the boundary, negative
    inside, blurred by `smooth` voxels and trilinearly interpolated. The distance of a binary
    mask still carries the voxel steps of the mask; the blur takes them out and leaves every
    feature wider than a voxel or two, so an ingested body comes back as smooth as it went in."""
    sp = 2.0 * extent / (res - 1)
    din = ndimage.distance_transform_edt(occ, sampling=sp)
    dout = ndimage.distance_transform_edt(~occ, sampling=sp)
    sdf = ndimage.gaussian_filter(dout - din, smooth) if smooth > 0 else dout - din
    lo = -extent

    def f(p):
        idx = (np.asarray(p) - lo) / sp
        return ndimage.map_coordinates(sdf, idx.reshape(-1, 3).T, order=1,
                                       mode="nearest").reshape(p.shape[:-1])
    return f
