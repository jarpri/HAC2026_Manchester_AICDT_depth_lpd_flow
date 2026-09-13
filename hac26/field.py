"""The shape representation: a convex core displaced inward by a depth on the sphere.

    f(y) = max_j (n_j . y - h_j) + q(u) + d(u),   u = (y - o) / |y - o|
    q(u) = sum_{l <= 2} c_lm Ybar_lm(u),          d(u) = sum_k a_k psi_k(u)

The body is where f < 0. The core is the intersection of half-spaces on fixed normals n_j
with support values h_j, so on its own it is a convex polytope whose faces lie in the planes
n_j . y = h_j. Everything that makes the body non-convex is a function of direction alone.

Adding a function of direction to the core displaces its level set along the local surface
normal by that amount, because on a facet the core's gradient has unit norm. So a coefficient
of q or d is a depth in body units, and the surface the carve leaves is as sharp as the core's
own facets: the field is smoothed across the surface, where the curves resolve little, and not
along the normal, where a shadow edge lives.

The direction is measured from a centre o rather than from the pose origin, and it indexes the
surface point rather than its normal: the Gauss map of a polytope is not injective, since a
whole facet shares one normal, while the radial map of a body star-shaped about its centre is.

q and d are one function at two angular scales. q carries the degrees up to two -- which is the
correction a convex inversion's hull needs, since it cannot see a concavity and explains the
shadowing with shape -- and d carries the rest. A solver moves the single function coarse to
fine in angular degree, so the hull correction is the first few degrees of the carve and not a
separate object in a separate space.

The max is a plain max, not a smooth one. Autodiff sends the gradient to the half-space that
owns the surface at that point; a smooth maximum would pull every face inward.

A depth may be negative, which pushes the surface outward. The bound that matters is one-sided
and is `depth_cap`: below it the body still contains its centre, is star-shaped, and therefore
extracts as one closed surface.

Everything here lives in the canonical frame (the body's z extent is [-1, 1] and its xy
radius is 1). Callers restore the published width afterwards with fit_to_cylinder; see
hac26/shapes.py::canonicalize_r.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["spherical_design", "design_sha", "DESIGN_N", "DESIGN_T", "DESIGN_ITERS",
           "ConvexCore", "DepthSphere", "ImplicitBody", "extract_mesh", "voxel_grid",
           "apply_constraints", "EXTRACT_EXTENT", "EXTRACT_RES",
           "RADIAL_DEGREE", "N_RADIAL", "radial_basis", "radial_field",
           "node_design", "node_turn", "node_kernel", "depth_cap", "core_centre",
           "N_NODES", "KNN", "NODE_BETA", "DEPTH_CAP_FRAC", "EVAL_CACHE_POINTS",
           "N_DIR", "CODE_DIM", "SH_DEGREE", "dir_design", "sh_expand",
           "support_resample", "support_resample_weights", "real_sh"]

DESIGN_N = 4096        # number of core normals. More normals give smaller facets. This size
                       # needs the committed hac26/design4096.npy: building a design this
                       # large takes hours on a CPU, so spherical_design() refuses to do it
                       # inside a constructor.
DESIGN_T = 10          # spherical design strength
DESIGN_ITERS = 4000    # the only iteration count whose result is allowed into the cache
CORE_CHUNK_ELEMS = 6e7 # cap on the (points x normals) intermediate, in float32 elements

# ------------------------------------------------------------------------ the depth field
N_NODES = 2560        # directions the depth is carried on. It is where the family stops
                      # improving faster than the extraction can show: a quarter of this count
                      # already holds the released non-convex body better than the lattice of
                      # thirteen thousand amplitudes this replaced, and four times it buys
                      # little that a grid of EXTRACT_RES cubes can resolve.
                      # notes/representation.md has the measurement. It is a multiple of four
                      # because node_design is built to be invariant under a quarter turn about
                      # the spin axis.
KNN = 6               # nodes a direction reads. The weights are normalised over them, so they
                      # are a partition of unity: a constant depth is reproduced exactly, the
                      # field is bounded by the largest coefficient, and it cannot ring.
NODE_BETA = 0.75      # kernel width over mean node spacing, sqrt(4 pi / N_NODES) radians.
DEPTH_CAP_FRAC = 0.90 # deepest carve allowed, as a fraction of the smallest support value
                      # about the centre. Above that bound the body stops containing its own
                      # centre and stops being star-shaped, and the extraction then returns
                      # several components; the margin is a margin and not a measurement.

EXTRACT_EXTENT = 1.30             # half-width of the grid extract_mesh runs on, in the
                                  # canonical frame. It has to cover the body, which the pose
                                  # puts inside the unit cylinder, with room for a correction
                                  # that pushes outward.
EXTRACT_RES = 128                 # side of that grid. What sets it is that the extraction is
                                  # what limits the answer once the field is fine enough, so it
                                  # is raised until the field rather than the grid decides how
                                  # sharp a carve can be. See notes/representation.md.
EVAL_CACHE_POINTS = 2 * EXTRACT_RES ** 3
                      # query points the field keeps work for, counting the same point once per
                      # cached quantity. It is twice an extraction grid so that a run which
                      # fits at one resolution and exports at another holds the whole of
                      # whichever it is using; smaller than one grid and every chunk would
                      # evict the chunk wanted next. What is held per point is the depth's six
                      # neighbours and their weights, seventy-two bytes, and the core's own
                      # value, four -- so the cap is a few hundred megabytes at the extraction
                      # resolution, against seconds of neighbour search and a thousand million
                      # comparisons against the core's normals on every render that would
                      # otherwise be repeated.


# ------------------------------------------------------------------- the support correction
SH_DEGREE = 5                     # dh is band-limited to this spherical-harmonic degree. A
                                  # rough dh removes facets from the polytope, and a removed
                                  # facet has no area and therefore no gradient.
N_DIR = 128                       # dh is carried as samples on this many design directions,
                                  # which is what SphereConv needs; the band limit is applied
                                  # by sh_expand() when dh is expanded onto the core normals.
CODE_DIM = N_DIR + N_NODES


# ----------------------------------------------------------------- the fixed normals

def _legendre_gram(g: torch.Tensor, t: int) -> list:
    """P_l applied elementwise to a Gram matrix, by the three-term recurrence."""
    out = [torch.ones_like(g), g]
    for l in range(2, t + 1):
        out.append(((2 * l - 1) * g * out[l - 1] - (l - 1) * out[l - 2]) / l)
    return out


def design_energy(x, t: int = DESIGN_T):
    """Sum of the per-degree design energies; zero exactly for a t-design.

    The identity that makes this the correct objective:

        sum_{i,j} P_l(x_i . x_j) = (4 pi / (2l+1)) sum_m | sum_i Y_lm(x_i) |^2  >= 0

    Each term is non-negative and vanishes precisely when the degree-l harmonic moments do,
    which IS the t-design condition -- and it needs only Legendre polynomials of the Gram
    matrix, so it differentiates trivially, unlike evaluating Y_lm directly.

    Minimising raw monomial means instead does not work: for even l the points being unit
    vectors forces sum_i x_i^2 = n/3, so that target is unreachable.
    """
    xt = torch.as_tensor(x, dtype=torch.float64)
    g = (xt @ xt.T).clamp(-1.0, 1.0)
    P = _legendre_gram(g, t)
    n = len(xt)
    return sum((2 * l + 1) * P[l].sum() / (n * n) for l in range(1, t + 1))


def _design_residual(x: np.ndarray, t: int = DESIGN_T) -> float:
    """Worst single-degree design energy. Zero for an exact t-design."""
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    g = (xt @ xt.T).clamp(-1.0, 1.0)
    P = _legendre_gram(g, t)
    n = len(xt)
    return float(max(abs(P[l].sum().item()) / (n * n) for l in range(1, t + 1)))


def spherical_design(n: int = DESIGN_N, t: int = DESIGN_T, seed: int = 0,
                     iters: int = DESIGN_ITERS) -> np.ndarray:
    """n unit normals forming a spherical t-design, with the six axis directions included.

    The axis directions are pinned so that an axis-aligned box is exactly representable: the
    core reproduces a face exactly only if that face's normal is one of its normals. Pinning
    six points costs the design property almost nothing (`_design_residual` reports it and
    the tests check it).

    The remaining n - 6 points are optimised to zero the harmonic sums that define the
    design, starting from a Fibonacci spiral.

    The result is cached beside this file as design{n}.npy. Large n is refused rather than
    built: the objective holds n x n matrices per Legendre order, so a large build takes
    hours, and running that silently inside a constructor looks like a hang. The error names
    the command that builds and caches the file.
    """
    cache = Path(__file__).with_name(f"design{n}.npy")
    if cache.exists() and t == DESIGN_T:
        x = np.load(cache)
        if len(x) == n:
            return x
    if n > 512:
        raise FileNotFoundError(
            f"{cache} is missing. Generate it once with "
            f"`python scripts/make_design.py --n {n}` (add --device cuda if you have a GPU; "
            f"on a CPU this takes hours). It is cached afterwards.")
    axes = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
    m = n - len(axes)
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    tht = np.pi * (1 + 5 ** 0.5) * i
    free = np.stack([np.cos(tht) * np.sin(phi), np.sin(tht) * np.sin(phi), np.cos(phi)], 1)

    p = torch.tensor(free, dtype=torch.float64, requires_grad=True)
    fixed = torch.tensor(axes, dtype=torch.float64)
    opt = torch.optim.Adam([p], lr=1e-2)
    best, best_x = float("inf"), None
    for _ in range(iters):
        x = torch.cat([fixed, p / p.norm(dim=1, keepdim=True)], 0)
        loss = design_energy(x, t)
        opt.zero_grad(); loss.backward(); opt.step()
        v = abs(float(loss))          # the sum cancels to ~1e-15 and can go slightly
        if v < best:                  # negative; selecting on the signed value latches
            best, best_x = v, x.detach().clone()   # onto numerical noise, not the minimum
    out = best_x.numpy()
    if t == DESIGN_T and iters == DESIGN_ITERS:
        _write_design_cache(cache, out)
    return out


def _write_design_cache(cache: Path, x: np.ndarray) -> None:
    """Write the design file atomically, so a concurrent process cannot read a half-written
    file.

    Only called for the default `iters`: the filename keys on `n` alone, so caching a short
    debug build would silently become what every later caller gets.

    Generation is deterministic, so two processes that race produce the same array and
    whichever rename lands last is still correct.
    """
    import os
    import tempfile
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(cache.parent), suffix=".npy")
        os.close(fd)
        np.save(tmp, x)
        os.replace(tmp, cache)
    except OSError:
        if tmp is not None and Path(tmp).exists():
            try:
                os.unlink(tmp)        # do not leave a partial file behind
            except OSError:
                pass                  # a read-only install is not a reason to fail the run


def design_sha(x: np.ndarray) -> str:
    """Short digest of a design, so a corpus can refuse a design it was not fitted against.

    The number of normals alone does not identify them: two machines that build the cache
    independently can land on different points. h is indexed by normal, so mixing two
    designs silently changes what every entry of h means.
    """
    import hashlib
    # float32, the precision ConvexCore stores the normals at, so the cached file and the
    # model's own buffer hash the same for the same design.
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------- the field

class ConvexCore(nn.Module):
    """max_j (n_j . y - h_j), with h >= 0 enforced by softplus on the raw parameter."""

    def __init__(self, normals: np.ndarray):
        super().__init__()
        self.register_buffer("n", torch.tensor(normals, dtype=torch.float32))
        self.raw_h = nn.Parameter(torch.zeros(len(normals)))
        # The field of a fixed polytope at a fixed set of points. Through a fit of the carve the
        # support does not move at all, and the extraction grid is one tensor, so this is the
        # same array on every one of a fit's thousands of renders -- and recomputing it is the
        # most expensive thing in a render, a maximum over every normal at every grid point.
        self._field_cache: dict = {}
        self._field_points = 0

    @property
    def h(self) -> torch.Tensor:
        return F.softplus(self.raw_h)

    def set_support(self, h: torch.Tensor | np.ndarray) -> None:
        """Set h directly (inverse softplus), e.g. from an analytic support function."""
        h = torch.as_tensor(h, dtype=torch.float32).clamp_min(1e-6)
        with torch.no_grad():
            self.raw_h.copy_(h + torch.log(-torch.expm1(-h)))   # stable softplus inverse

    def forward(self, y: torch.Tensor, chunk: int | None = None,
                h: torch.Tensor | None = None) -> torch.Tensor:
        """max_j (n_j . y - h_j) at the query points y, in chunks.

        `h` may be supplied to override the stored support. ImplicitBody passes the
        dh-corrected support that way, so the correction stays differentiable.

        The (points x normals) intermediate is the memory cost of the whole field. Chunking
        bounds it without changing the result, since the max is taken per point.
        """
        n_norm = self.n.shape[0]
        hh = self.h if h is None else h
        # A cached value is the value, not an approximation of it, but it is not a graph: a
        # caller differentiating through the support has to have the maxima taken again.
        live = torch.is_grad_enabled() and (hh.requires_grad or y.requires_grad)
        key = None
        if not live:
            key = (y.data_ptr(), int(y.shape[0]), str(y.device),
                   hashlib.sha1(hh.detach().cpu().numpy().tobytes()).hexdigest())
            hit = self._field_cache.get(key)
            if hit is not None:
                return hit[1]
        if chunk is None:
            chunk = max(4096, int(CORE_CHUNK_ELEMS // max(n_norm, 1)))
        if y.shape[0] <= chunk:
            out = (y @ self.n.T - hh).amax(dim=-1)              # plain max, not LSE
        else:
            out = torch.cat([(y[i:i + chunk] @ self.n.T - hh).amax(dim=-1)
                             for i in range(0, y.shape[0], chunk)], dim=0)
        if key is not None:
            # the points are kept with the values, so that the tensor cannot be freed while its
            # address is a key and a later allocation cannot land on that address
            self._field_cache[key] = (y, out)
            self._field_points += int(y.shape[0])
            while self._field_points > EVAL_CACHE_POINTS and len(self._field_cache) > 1:
                old = next(iter(self._field_cache))
                self._field_points -= int(self._field_cache.pop(old)[0].shape[0])
        return out


def node_design(n: int = N_NODES) -> np.ndarray:
    """n quasi-uniform directions on the sphere, invariant under a quarter turn about z.

    A quarter turn about the spin axis is an exact symmetry of the problem, and it is what
    gives the corpus four training pairs per body at no cost in renders. That is only true if
    turning a body permutes its depths: if the node set is not itself invariant, a turn has to
    resample the field instead, and resampling smooths it. Measured on the released non-convex
    body, the resampled code differs from a fit of the turned body by twelve per cent of the
    depths in the root mean square -- five times the fit's own residual -- so the corpus would
    be taught codes that no fit of those bodies would produce.

    The set is therefore a quarter of the sphere's worth of directions and its three rotations,
    so the turn is the block shift node_turn. The quarter carries a Fibonacci spiral of its
    own: latitudes evenly spaced in cos, azimuths advancing by the golden ratio *of the
    quadrant*. That last part is what keeps it uniform. Advancing by the whole sphere's golden
    angle and folding it into the quadrant leaves consecutive latitudes near the poles at
    nearly the same azimuth, and the closest pair of nodes comes out at half the spacing of a
    plain spiral; taking the golden ratio inside the quadrant instead gives nearest-neighbour
    spacings as good as a plain spiral's at every quantile, and a slightly better worst case
    over six neighbours, which is the number the weights read.

    Quasi-uniform rather than a subdivided icosahedron because the count is then free: the node
    count is a resolution and wants to be chosen by what the curves resolve, not by whatever a
    subdivision level happens to give.
    """
    n = int(n)
    if n % 4:
        raise ValueError(f"{n} directions cannot be invariant under a quarter turn; the count "
                         f"must be a multiple of four")
    m = n // 4
    i = np.arange(m) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / m)
    azim = (0.5 * np.pi * (5.0 ** 0.5 - 1.0) / 2.0 * i) % (0.5 * np.pi)
    return np.concatenate([np.stack([np.cos(azim + q * 0.5 * np.pi) * np.sin(polar),
                                     np.sin(azim + q * 0.5 * np.pi) * np.sin(polar),
                                     np.cos(polar)], axis=1) for q in range(4)], axis=0)


def node_turn(q: int, n: int = N_NODES) -> np.ndarray:
    """The permutation of the nodes made by turning a body q quarter turns about the spin axis:
    index k of the turned field reads index node_turn(q)[k] of the unturned one.

    The value a turned body has in direction u is the value the body has in direction R^T u, so
    this is the permutation with u[node_turn(q)] = R^T u. It is a shift of whole blocks, because
    node_design lays the four rotations out one after another. Exact, which is the whole reason
    the node set is built the way it is: a turned body's depths are the same numbers in a
    different order, so the symmetry costs nothing and introduces nothing."""
    n = int(n)
    if n % 4:
        raise ValueError(f"{n} directions are not invariant under a quarter turn")
    return np.roll(np.arange(n), int(q) * (n // 4))


def dir_design(n: int = N_DIR) -> np.ndarray:
    """The directions dh is sampled on: a spherical design of n points."""
    return spherical_design(n)


def real_sh(x: np.ndarray, degree: int = SH_DEGREE) -> np.ndarray:
    """Real spherical harmonics up to `degree` at the unit vectors x, without normalisation.

    The basis is only used through a pseudo-inverse, which does not care how the columns are
    scaled. What matters is that the columns span exactly the harmonics of degree <= `degree`.
    """
    from scipy.special import lpmv
    x = np.asarray(x, dtype=np.float64)
    ct = np.clip(x[:, 2], -1.0, 1.0)
    ph = np.arctan2(x[:, 1], x[:, 0])
    cols = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            p_lm = lpmv(abs(m), l, ct)
            if m > 0:
                cols.append(p_lm * np.cos(m * ph))
            elif m < 0:
                cols.append(p_lm * np.sin(-m * ph))
            else:
                cols.append(p_lm)
    return np.stack(cols, 1)                       # (len(x), (degree+1)^2)


RADIAL_DEGREE = 2                          # band limit of the radial reshaping term
N_RADIAL = (RADIAL_DEGREE + 1) ** 2        # its coefficients


def radial_basis(u: torch.Tensor) -> torch.Tensor:
    """The real spherical harmonics of degree at most RADIAL_DEGREE at the unit vectors u
    (n, 3), as (n, N_RADIAL), each scaled to unit root mean square over the sphere.

    Unit root mean square rather than unit integral so that a coefficient is a length. The
    degree-zero column is then identically one, and a coefficient vector of size s displaces
    the surface of a convex core by about s in body units, because on a facet of a polytope
    the core's gradient has unit norm and adding a constant to the field moves the level set
    by that constant.
    """
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    r3, r15 = np.sqrt(3.0), np.sqrt(15.0)
    one = torch.ones_like(x)
    return torch.stack([
        one,
        r3 * y, r3 * z, r3 * x,
        r15 * x * y, r15 * y * z,
        0.5 * np.sqrt(5.0) * (3.0 * z ** 2 - 1.0),
        r15 * x * z, 0.5 * r15 * (x ** 2 - y ** 2)], -1)


def radial_field(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """sum_lm c_lm Ybar_lm(y / |y|) at the points y (n, 3), which the caller has already made
    relative to the body's centre.

    This is the degree-two part of the same depth field DepthSphere carries the rest of, so it
    is measured from the same centre and in the same units. The direction of a point is
    undefined at the origin, which is inside every body the challenge poses and never on a
    level set, so it is guarded and not special-cased.
    """
    return radial_basis(y) @ c


def support_resample_weights(src: np.ndarray, dst: np.ndarray, k: int = 6):
    """The resampling of support_resample as (indices, weights), both (len(dst), k): the k
    source directions each destination direction reads, and their weights. The form to use
    when the dense matrix would be large."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    idx = np.argsort(-(dst @ src.T), axis=1)[:, :k]
    # One batched pseudo-inverse rather than a least-squares solve per direction. pinv gives
    # the least-norm solution of the underdetermined system, which is what lstsq returns.
    A = np.concatenate([np.ones((len(dst), 1, k)),
                        src[idx].transpose(0, 2, 1)], axis=1)         # (D, 4, k)
    b = np.concatenate([np.ones((len(dst), 1)), dst], axis=1)         # (D, 4)
    w = np.einsum("dkj,dj->dk", np.linalg.pinv(A), b)                 # (D, k)
    return idx, w.astype(np.float32)


def support_resample(src: np.ndarray, dst: np.ndarray, k: int = 6) -> np.ndarray:
    """Matrix that resamples a support function from `src` directions onto `dst` directions.

    This is not sh_expand. sh_expand keeps only low harmonic degrees, which is right for dh
    and wrong for h: the support function of a flat-faced body is not band-limited, and the
    public models include flat-faced bodies.

    Within one flat face a support function is an affine function of the direction, so
    weights over the k nearest source directions that reproduce affine functions are exact
    there and interpolate smoothly elsewhere. They are the least-norm solution of

        sum_i w_i = 1,      sum_i w_i n_i = u

    k is larger than the four points that would pin the constraints exactly, because a
    near-degenerate quadruple makes the weights blow up.
    """
    idx, w = support_resample_weights(src, dst, k)
    W = np.zeros((len(dst), len(np.asarray(src))), dtype=np.float32)
    np.put_along_axis(W, idx, w, axis=1)
    return W


def sh_expand(src: np.ndarray, dst: np.ndarray, degree: int = SH_DEGREE) -> np.ndarray:
    """Matrix taking dh sampled on `src` directions to dh sampled on `dst` directions.

    `Y_dst @ pinv(Y_src)`. Because Y has only (degree+1)^2 columns, this is also the band
    limit: whatever is emitted on `src`, only its part of degree <= `degree` reaches h. So
    the band limit is built in rather than encouraged by a penalty.
    """
    y_src, y_dst = real_sh(src, degree), real_sh(dst, degree)
    return (y_dst @ np.linalg.pinv(y_src)).astype(np.float32)


# ----------------------------------------------------------------- the correction field

class DepthSphere(nn.Module):
    """d(u) = sum_k a_k psi_k(u), a depth on the sphere carried on N_NODES directions.

    psi_k is a Gaussian in the angle to node k, normalised over the KNN nearest nodes of the
    query direction. The weights of a direction therefore sum to one, which is what makes a
    coefficient a depth rather than an amplitude: d reproduces a constant exactly, is bounded
    by the largest coefficient, and cannot overshoot between nodes.

    `a` is the only parameter and it is the code. The nodes are fixed and are not part of it.

    The weights depend on the query direction alone, so for a fixed set of query points they
    are a fixed sparse gather with KNN non-zeros per row, built once and reused. A field
    evaluation is then one sparse product, and it is linear in `a`, so a caller may
    differentiate through it.
    """

    def __init__(self, n_nodes: int = N_NODES, knn: int = KNN, beta: float = NODE_BETA,
                 nodes: np.ndarray | None = None):
        super().__init__()
        u = node_design(n_nodes) if nodes is None else np.asarray(nodes, dtype=float)
        # persistent=False: the nodes follow from the constants above and are the same for
        # every body, so a checkpoint must not carry its own copy.
        self.register_buffer("u", torch.tensor(u, dtype=torch.float32), persistent=False)
        self.knn = int(knn)
        self.sigma = float(beta) * float(np.sqrt(4.0 * np.pi / len(u)))
        self.a = nn.Parameter(torch.zeros(len(u)))
        self._tree = None
        # The weights of a query direction depend on the direction alone, so for a fixed set
        # of points they are a fixed gather. The extraction grid is fixed and is the set this
        # is asked for millions of points at a time, so it is cached on the identity of the
        # tensor holding those points; a rebuilt search costs seconds and a cached one costs
        # a hundredth of that.
        self._gather_cache: dict = {}
        self._gather_points = 0

    @property
    def n_nodes(self) -> int:
        return self.u.shape[0]

    def _kdtree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.u.detach().cpu().numpy())
        return self._tree

    def weights(self, direction: np.ndarray):
        """(index, weight) of shape (n, KNN): the nodes each direction reads and by how much."""
        d = np.asarray(direction, dtype=float)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        chord, idx = self._kdtree().query(d, k=self.knn)
        if self.knn == 1:
            chord, idx = chord[:, None], idx[:, None]
        angle = 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))
        w = np.exp(-0.5 * (angle / self.sigma) ** 2)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
        return idx, w

    def gather(self, y: torch.Tensor, centre: torch.Tensor | None = None) -> tuple:
        """(index, weight) as torch tensors for the directions of the points y about `centre`,
        ready to be applied to the coefficients. The centre has no direction from itself; it
        is inside every body the challenge poses and never on a level set, so it is guarded
        rather than special-cased.

        Cached on the identity of the points and the value of the centre, which is what makes
        an extraction affordable: the grid is the same tensor every call and its neighbour
        search is built once. The points are keyed *before* the centre is subtracted, because
        subtracting it allocates a new tensor and a key on that would never hit, and the entry
        holds the points as well as the weights, so the address a key names stays taken.
        """
        c = None if centre is None else centre.detach().cpu().numpy().tobytes()
        key = (y.data_ptr(), int(y.shape[0]), str(y.device), c)
        hit = self._gather_cache.get(key)
        if hit is not None:
            return hit[1:]
        d = y.detach().cpu().numpy()
        if centre is not None:
            d = d - centre.detach().cpu().numpy().reshape(1, 3)
        idx, w = self.weights(d)
        dev = y.device
        out = (torch.as_tensor(idx, dtype=torch.long, device=dev),
               torch.as_tensor(w, dtype=torch.float32, device=dev))
        # y itself is kept, so that the tensor cannot be freed while its address is a key and a
        # later allocation of the same shape cannot land on that address and read this entry
        self._gather_cache[key] = (y,) + out
        self._gather_points += int(y.shape[0])
        # oldest first, so that a run which changes resolution gives up the grid it has
        # finished with rather than the one it is working on
        while self._gather_points > EVAL_CACHE_POINTS and len(self._gather_cache) > 1:
            old = next(iter(self._gather_cache))
            self._gather_points -= int(self._gather_cache.pop(old)[0].shape[0])
        return out

    def forward(self, y: torch.Tensor, a: torch.Tensor | None = None,
                gather: tuple | None = None,
                centre: torch.Tensor | None = None) -> torch.Tensor:
        """d at the directions of the points y about `centre`. `a` may be supplied to override
        the stored coefficients, and `gather` to reuse weights already built for them."""
        aa = self.a if a is None else a
        idx, w = self.gather(y, centre) if gather is None else gather
        return (aa[idx] * w).sum(dim=1)

    def matrix(self, direction: np.ndarray):
        """The sparse (n, n_nodes) evaluation matrix for a set of directions."""
        from scipy import sparse
        idx, w = self.weights(direction)
        rows = np.repeat(np.arange(len(idx)), self.knn)
        return sparse.csr_matrix((w.ravel().astype(np.float32), (rows, idx.ravel())),
                                 shape=(len(idx), self.n_nodes))


def node_kernel(n_nodes: int = N_NODES, knn: int = KNN, beta: float = NODE_BETA):
    """What the coefficients make at the nodes themselves, as a sparse matrix.

    It is nearly the identity, and that is the point: a coefficient is the depth at its own
    node to within the smoothing over its neighbours. A solver uses it to know what a
    coordinate is worth without spending a render on it, and because its rows sum to one a
    bound on the coefficients is a bound on the depth in body units.
    """
    rep = DepthSphere(n_nodes, knn, beta)
    return rep.matrix(rep.u.numpy())


def core_centre(support, normals: np.ndarray | None = None) -> np.ndarray:
    """The Steiner point of the convex core at this support: 3/(4 pi) times the integral of
    h(u) u over the sphere, taken as the mean over the design directions.

    The depth is a displacement measured along a ray, so it needs a point to measure from, and
    that point has to be a function of the support alone: the support is what every stage of the
    pipeline carries beside a code, and a centre stored separately would be a second thing to
    keep in step. It is not the pose origin. Measured on the library's deeply carved bodies, the
    best a body star-shaped about the pose origin can do against the truth is three points of
    overlap below the best about a centre inside the body in the median and twelve points below
    it in the worst case, because the pose puts the origin where the published constraints put
    it and not where the body is.

    The Steiner point is the canonical centre of a convex body: it lies in the interior, it
    moves with the body (for h(u) = h0(u) + o . u with h0 even it returns o exactly), and it
    costs one product against the normals. Three candidates were compared on the released
    bodies, on how much of each body a shape star-shaped about the point can cover: this one,
    the volume centroid, and the Chebyshev centre, which is by construction the point with the
    most room to carve. On the one released body that is substantially non-convex they span two
    thousandths and this one is the best of the three; the Chebyshev centre is the worst on both
    bodies that are not already their own hulls, which is what its own definition predicts, since
    maximising the room to carve puts the point inside the fattest lobe rather than where it can
    see the whole body.

    What decides between this and the volume centroid is not the two thousandths but the cost
    and what happens when it is wrong. The centroid needs the polytope recovered from its four
    thousand half-spaces, which is a hundred times slower and can be refused outright for a
    degenerate support -- and a support decoded by a partly trained network in a training loop
    is sometimes degenerate. This is a product and a mean, and there is no support it cannot
    take.
    """
    h = np.asarray(support, dtype=np.float64).ravel()
    n = spherical_design(DESIGN_N) if normals is None else np.asarray(normals, dtype=float)
    return 3.0 * (h[:, None] * n).mean(axis=0)


def depth_cap(support, centre=None, fraction: float = DEPTH_CAP_FRAC,
              normals: np.ndarray | None = None) -> float:
    """The deepest carve that leaves the body star-shaped about its centre.

    Along a ray from the centre the core is a maximum of affine functions, so it is convex,
    tends to infinity at both ends, and equals minus the smallest support value at the centre.
    The set where the field is negative is therefore one interval, and it contains the centre
    exactly while the depth stays below that smallest support value. Under that one-sided
    bound the body is star-shaped, hence connected, hence the single closed surface
    scripts/check_submission.py requires -- by construction rather than by repair.

    The bound is one-sided. A depth may be as negative as it likes; that pushes the surface
    outward and cannot disconnect it.
    """
    h = np.asarray(support, dtype=float).ravel()
    n = spherical_design(DESIGN_N) if normals is None else np.asarray(normals, dtype=float)
    o = core_centre(h, n) if centre is None else np.asarray(centre, dtype=float).ravel()
    return float(fraction) * float((h - n @ o).min())


class ImplicitBody(nn.Module):
    """f(y) = max_j (n_j . y - h_j) + q(u) + d(u), with u the direction of y from the body's
    centre, h = softplus(inv_softplus(h_base) + expand(dh)), q the degree-two reshaping and d
    the depth field.

    h_base is the support the body starts from: the hull support of a training body, or the
    convex stage's answer at reconstruction. dh is a band-limited correction to it, sampled on
    `dir_design(N_DIR)` and expanded onto the core normals by sh_expand. It is added inside the
    softplus, so h stays positive without a clamp.

    dh exists because the convex stage assumes convexity. A non-convex body is darker than
    its own hull because it shadows itself, and a convex inversion explains that darkness with
    shape, so h_base is wrong in a direction that favours a convex answer.

    q and d are the same object at different angular scales, and writing them that way is what
    dissolves the problem this repository spent a long time fighting. The correction from a
    convex inversion's answer to the body used to be two things -- a reshaping of the hull and
    a carve -- that had to move together and that no search could move together, because they
    lived in different spaces. They are now one function on the sphere: q carries its degrees
    up to two and d the rest. A solver moves it coarse to fine in angular degree, and the hull
    correction is simply the first few degrees of the carve.

    The centre the direction is measured from is a buffer, not a parameter. It is set beside
    the support and is never moved by a fit: the star-shaped bound of `depth_cap` is a
    statement about a fixed centre, and a body carved deeply is much better behaved about its
    own centroid than about the pose origin.
    """

    def __init__(self, normals: np.ndarray | None = None, n_nodes: int = N_NODES):
        super().__init__()
        if normals is None:
            normals = spherical_design(DESIGN_N)
        self.core = ConvexCore(normals)
        self.delta = DepthSphere(n_nodes)
        self.dh = nn.Parameter(torch.zeros(N_DIR))
        self.register_buffer("centre", torch.zeros(3), persistent=False)
        self.register_buffer("dh_expand",
                             torch.from_numpy(sh_expand(dir_design(N_DIR), normals)),
                             persistent=False)

    def set_support(self, h, centre=None) -> None:
        """Set the base support h_base, which is the origin dh is measured from, and with it the
        centre the depth's directions are measured from.

        The centre follows from the support unless one is given, so a caller cannot leave it at
        the pose origin by forgetting it. core_centre keeps what it derives, and the support is
        the same through a fit, so the derivation is paid once however many times this is
        called.
        """
        self.core.set_support(h)
        o = core_centre(h.detach().cpu().numpy() if torch.is_tensor(h) else h,
                        self.core.n.detach().cpu().numpy()) if centre is None else centre
        with torch.no_grad():
            self.centre.copy_(torch.as_tensor(np.asarray(o, dtype=np.float32).ravel()))

    def support(self, dh: torch.Tensor | None = None) -> torch.Tensor:
        """The support actually used: softplus(raw_h + expand(dh)), with `dh` overriding the
        stored correction when given. raw_h is the parameter itself, not a copy, so h stays
        learnable through this call if a caller wants that."""
        d = self.dh if dh is None else dh
        return F.softplus(self.core.raw_h + self.dh_expand @ d)

    def forward(self, y: torch.Tensor, dh: torch.Tensor | None = None,
                a: torch.Tensor | None = None, c: torch.Tensor | None = None,
                gather: tuple | None = None) -> torch.Tensor:
        """f at the points y. `dh`, `a` and `c` override the stored code when given, so a caller
        can differentiate f through parameters it holds itself. `a` is the depth field's
        coefficients and `c` the degree-two reshaping, which is absent unless it is passed.
        `gather` reuses depth weights already built for these points."""
        f = self.core(y, h=self.support(dh)) + self.delta(y, a=a, gather=gather,
                                                          centre=self.centre)
        return f if c is None else f + radial_field(y - self.centre, c)


# ----------------------------------------------------------------- extraction

_GRID_CACHE: dict = {}


def voxel_grid(fc, res: int, device):
    """The (res+1)^3 grid vertices on [-1/2, 1/2]^3 and the eight vertex indices of each of the
    res^3 cubes, in FlexiCubes' own ordering.

    The same thing FlexiCubes.construct_voxel_grid returns, built arithmetically instead of by
    deduplicating the corners of every cube separately. That deduplication sorts eight times as
    many points as the grid has, which at the extraction resolution is seventeen million rows of
    three and several gigabytes, and it is the largest allocation anything here makes. The
    vertices it produces come out in lexicographic order, which is the order a meshgrid gives,
    so a cube's corner is the grid index of its own corner offset and no sort is needed. The
    suite checks the two constructions against each other.
    """
    a = torch.arange(res + 1, device=device, dtype=torch.float32) / res
    verts = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3) - 0.5
    c = fc.cube_corners.to(device).long()
    off = (c[:, 0] * (res + 1) + c[:, 1]) * (res + 1) + c[:, 2]            # (8,)
    i = torch.arange(res, device=device)
    gi, gj, gk = torch.meshgrid(i, i, i, indexing="ij")
    base = ((gi * (res + 1) + gj) * (res + 1) + gk).reshape(-1, 1)
    return verts, base + off[None, :]


def _voxel_grid(res: int, device, extent: float):
    """FlexiCubes plus its voxel grid at the extraction's own scale, cached per
    (res, device, extent).

    The scale is in the key, and the scaled grid is what is cached, because the identity of
    that tensor is what lets the depth field and the convex core reuse the work they did for it.
    A grid rescaled on every call is a new tensor every call, and the neighbour search and the
    core's own field would both be rebuilt two million points at a time.
    """
    from .vendor.flexicubes import FlexiCubes

    key = (int(res), str(device), round(float(extent), 9))
    if key not in _GRID_CACHE:
        fc = FlexiCubes(device=device)
        x, cube = voxel_grid(fc, int(res), device)
        _GRID_CACHE[key] = (fc, x * (2.0 * float(extent)), cube)
    return _GRID_CACHE[key]


def extract_mesh(field, extent: float = EXTRACT_EXTENT, res: int = EXTRACT_RES, device: str = "cpu",
                 chunk: int = 262144, grad: bool = False):
    """The surface f = 0 as a triangle mesh, by FlexiCubes on a res^3 grid over
    [-extent, extent]^3.

    With `grad=False` the result is a pair of numpy arrays. With `grad=True` the vertices are
    a torch tensor that is differentiable through the field values on the grid, so a caller
    that computes `field` from tensors it holds can differentiate the mesh with respect to
    them; the faces are then a torch long tensor.

    `extent` must cover the body. The pose puts every body inside the unit cylinder, and the
    correction can push the surface outward, so the default leaves room for that; a caller
    that passes less gets a body clipped by the grid rather than an error, which is why the
    default is a constant of the module and not an argument anyone should be choosing.
    """
    fc, x_nx3, cube_fx8 = _voxel_grid(res, device, extent)
    with torch.set_grad_enabled(grad):
        sdf = torch.cat([field(x_nx3[i:i + chunk]) for i in range(0, len(x_nx3), chunk)])
        verts, faces, _ = fc(x_nx3, sdf, cube_fx8, res, training=False)
    if grad:
        return verts, faces.long()
    return verts.detach().cpu().numpy(), faces.detach().cpu().numpy()


def apply_constraints(verts: np.ndarray, radius: float, tol: float = 0.03) -> np.ndarray:
    """Apply the challenge's pose constraints to extracted vertices.

    z is rescaled so the body touches -1 and +1 exactly, which the challenge states as
    equalities. The xy radius is then brought down to `radius` only if it exceeds it by more
    than `tol`: the published radius is approximate, and a posed public body can sit slightly
    past it, so clamping hard would shrink true geometry.
    """
    v = np.asarray(verts, dtype=np.float64).copy()
    zmin, zmax = v[:, 2].min(), v[:, 2].max()
    if zmax - zmin < 1e-12:
        raise ValueError("degenerate body: zero z extent")
    v[:, 2] = 2.0 * (v[:, 2] - zmin) / (zmax - zmin) - 1.0
    cap = radius * (1.0 + tol)
    r = float(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).max())
    if r > cap:
        v[:, :2] *= cap / r
    return v
