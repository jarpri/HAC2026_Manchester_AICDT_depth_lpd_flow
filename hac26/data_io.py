"""Reading the challenge curve files.

A curve file has one row per frame and N_CAMS + 1 columns: the frame time, then one curve
per camera in the released column order (per azimuth: two horizontal cameras, top, virtual
bottom), the same order as hac26.geometry.build_cameras(). Files are named
Asteroid<NN>_lightcurve_<intensity|binary>[_blender].txt.

fit_conventions estimates the rotation sense, azimuth handedness and Lambert weight of the
convex operator on a public model whose shape is known.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hac26.forward.convex_egi import normalize_np
from .conventions import PUBLIC_MODELS
from .geometry import build_cameras
from .shapes import hull_mesh, mesh_curves_convex

N_CAMS = 28
COUNT_FLOOR = 0.05     # a count curve whose smallest value falls below this fraction of its
                       # mean is refused; see count_curve_is_usable


def distinct_geometries() -> list:
    """The released columns grouped by camera geometry, as lists of column indices.

    Twenty-eight columns cover twenty-one distinct (azimuth, elevation) pairs: at every
    azimuth the first two columns are the same horizontal camera, recorded in the two
    mountings of the body. A forward model that does not know which mounting a column came
    from predicts one curve for both, so a likelihood that sums over all twenty-eight gives
    those seven geometries twice the weight of the rest.
    """
    groups: dict = {}
    for i, c in enumerate(build_cameras()):
        groups.setdefault((c.azimuth_deg, c.elevation_deg), []).append(i)
    return [groups[k] for k in sorted(groups)]


def held_out_geoms(present, n: int) -> list:
    """`n` of the geometries in `present` to keep out of a fit, spread evenly over the camera
    ordering.

    The held-out cameras are the only honest test of a reconstruction: a body fitted on every
    camera can reach any misfit by shape or by overfitting, and only a camera the fit never
    saw separates the two. That makes how they are chosen part of the measurement rather than
    a detail. The released cameras are ordered azimuth-major with the three distinct
    geometries of each azimuth together, so an even spread over the list takes cameras from
    across the azimuths and across the elevations, and both the fitted and the held-out set
    span the range of viewing geometries. A random draw does not: five drawn at random can
    fall in one azimuth, which leaves that azimuth out of the fit and puts the whole test
    inside it, and it makes two runs of the same body incomparable because they are then
    scored on different cameras. `present` is in camera order and the result is a subset of
    it, so two runs that hold out the same count hold out the same cameras.
    """
    idx = np.asarray(present)
    if not n:
        return []
    if n >= len(idx):
        raise ValueError(f"{n} geometries held out of {len(idx)} present leaves nothing to "
                         f"fit on")
    take = np.unique(np.linspace(0, len(idx) - 1, int(n)).round().astype(int))
    return [int(idx[i]) for i in take]


def duplicate_columns(curves: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """(2 * N_CAMS,) marking every column that repeats an earlier column of its own geometry
    and curve type.

    Which columns repeat is read off the file rather than assumed. In the released renders
    the two horizontal columns of every azimuth carry the same numbers to the last digit, so
    each contributes the same residual twice; in the laboratory files they differ, because
    they are two recordings of two mountings, and nothing is masked.
    """
    dup = np.zeros(len(curves), dtype=bool)
    for cols in distinct_geometries():
        for block in (0, N_CAMS):
            first = cols[0] + block
            for c in cols[1:]:
                if np.allclose(curves[c + block], curves[first], atol=atol, rtol=0.0):
                    dup[c + block] = True
    return dup


def count_curve_is_usable(curve: np.ndarray, floor: float = COUNT_FLOOR) -> bool:
    """Whether a released count curve carries shape rather than the transfer curve.

    The organisers take Otsu's threshold on the first frame and apply it to every frame. On a
    body with large flat facets the first frame's histogram has a mode per facet, and the
    threshold that maximises the between-class variance can fall between two facet modes
    rather than between the body and the background. The count is then the area of whichever
    facets are brighter than that level, and it drops to nothing at the phases where none is.
    A forward model cannot be relied on to land on the same side of that split, so a curve
    whose smallest value is a small fraction of its mean is not a measurement of the shape.
    """
    curve = np.asarray(curve, dtype=float)
    return bool(curve.min() > floor * max(curve.mean(), 1e-12))


def public_stl(data_dir: str, model: int) -> str:
    """Path of a public model's released shape inside the dataset directory."""
    if model not in PUBLIC_MODELS:
        raise ValueError(f"model {model} has no released shape; public models are "
                         f"{PUBLIC_MODELS}")
    return str(Path(data_dir) / f"AsteroidModel0{model}_shape_public" / f"asteroid{model}.stl")


def read_curves29(path: str) -> dict:
    """Read one curve file into {'time': (m,), 'curves': (N_CAMS, m)}. The delimiter, comma
    or whitespace, is detected from the first line."""
    with open(path) as fh:
        first = fh.readline()
    delim = "," if "," in first else None
    mat = np.loadtxt(path, delimiter=delim)
    if mat.ndim == 1:
        mat = mat[None, :]
    if mat.shape[1] != 29:
        raise ValueError(f"{path}: expected 29 columns, got {mat.shape[1]}")
    return {"time": mat[:, 0].copy(), "curves": mat[:, 1:].T.copy()}


def write_curves29(path: str, time: np.ndarray, curves: np.ndarray) -> None:
    """Write curves in the layout read_curves29 reads, whitespace-separated."""
    assert curves.shape[0] == N_CAMS
    np.savetxt(path, np.column_stack([time, curves.T]))


def resample_curves(curves: np.ndarray, m: int) -> np.ndarray:
    """Periodic linear resampling of each curve onto m uniform frames."""
    m0 = curves.shape[-1]
    if m0 == m:
        return curves
    x0 = np.arange(m0 + 1) / m0
    x1 = np.arange(m) / m
    ext = np.concatenate([curves, curves[..., :1]], axis=-1)
    return np.stack([np.interp(x1, x0, c) for c in ext], axis=0)[..., :m]


def load_model_curves(data_dir: str, model_idx: int, m: int = 360,
                      use_blender: bool = False, renormalize: bool = True) -> dict:
    """Assemble one model's [intensity, binary] curve stack, resampled to m frames.

    Returns {'curves': (2 * N_CAMS, m), 'mask': (2 * N_CAMS,), 'files': {type: path},
    'native': {type: (N_CAMS, m0)}}. A missing file leaves its block at zero with mask 0.
    `renormalize` divides each curve by its mean, which leaves an already mean-normalised
    file unchanged.

    'native' holds the curves at the frame rate of the files, before the resampling. The
    noise has to be estimated there: `hac26.noise.sigma_from_highfreq` reads successive
    differences, and resampling ~841 frames down to the operator's phase grid turns those
    into the curvature of the signal. `native_sigma` does that for a whole model.
    """
    import glob as _glob

    suffix = "_blender" if use_blender else ""
    stack = np.zeros((2 * N_CAMS, m))
    mask = np.zeros(2 * N_CAMS, dtype=np.float32)
    found = {}
    native = {}
    for j, ctype in enumerate(("intensity", "binary")):
        # The released archive spells the model number both zero-padded (Asteroid10) and
        # zero-prefixed (Asteroid010), and files sit in nested per-model folders, so both
        # spellings are searched recursively.
        names = {f"Asteroid{model_idx:02d}_lightcurve_{ctype}{suffix}.txt",
                 f"Asteroid0{model_idx}_lightcurve_{ctype}{suffix}.txt"}
        hits: list = []
        for nm in names:
            hits += _glob.glob(str(Path(data_dir) / "**" / nm), recursive=True)
            hits += _glob.glob(str(Path(data_dir) / nm))
        p = Path(sorted(hits)[0]) if hits else None
        if p is not None and p.exists():
            raw = read_curves29(str(p))["curves"]
            cur = resample_curves(raw, m)
            if renormalize:
                raw = normalize_np(raw)
                cur = normalize_np(cur)
            sl = slice(j * N_CAMS, (j + 1) * N_CAMS)
            stack[sl] = cur
            mask[sl] = 1.0
            found[ctype] = str(p)
            native[ctype] = raw
    # A curve that repeats another, or that Otsu's threshold made a step function of the
    # transfer curve rather than of the shape, is not a second measurement and is dropped
    # here rather than by each caller, so that no likelihood in the tree can double-count or
    # fit it by forgetting to ask.
    dup = duplicate_columns(stack) & (mask > 0)
    refused = np.zeros_like(dup)
    for c in range(N_CAMS, 2 * N_CAMS):
        if mask[c] > 0 and not count_curve_is_usable(stack[c]):
            refused[c] = True
    mask[dup | refused] = 0.0
    return {"curves": stack, "mask": mask, "files": found, "native": native,
            "duplicate_columns": np.nonzero(dup)[0].tolist(),
            "count_curves_refused": np.nonzero(refused)[0].tolist()}


CHANNELS = ("auto", "blender", "real")


def load_inversion_curves(data_dir: str, model_idx: int, m: int = 360,
                          channel: str = "auto") -> dict:
    """The curve stack a shape is inverted from, with the channel it came from under
    'channel'.

    The released data carry two recordings of every body, the laboratory curves and the
    organisers' Blender render of the true shape. The render is the cleaner measurement of
    the shape. It has no sensor, no mounting, no beam non-uniformity and no per-column
    realignment in front of it, its camera is far from the body, and its scattering is the
    Lambertian kind the convex operator assumes, whereas the laboratory columns of several
    bodies are out of phase with their own geometry by tens of degrees and one body's
    vertical cameras only match the render with the body upside down. So the render is
    inverted whenever it is released, and the laboratory curves are the fallback for a body
    whose simulated curves the organisers withhold, which their rules allow for the harder
    targets. `channel` forces one or the other; 'auto' takes the render only when both of
    its files, intensity and binary, are present, so that a body is never inverted from
    half of one recording and none of the other.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, not {channel!r}")
    if channel in ("auto", "blender"):
        d = load_model_curves(data_dir, model_idx, m=m, use_blender=True)
        if len(d["files"]) == 2 or (channel == "blender" and d["files"]):
            d["channel"] = "blender"
            return d
        if channel == "blender":
            raise FileNotFoundError(f"model {model_idx}: no Blender curve file under "
                                    f"{data_dir}")
    d = load_model_curves(data_dir, model_idx, m=m, use_blender=False)
    d["channel"] = "real"
    return d


def native_sigma(d: dict) -> np.ndarray:
    """(2 * N_CAMS,) noise sigma per curve of a `load_model_curves` result, estimated at the
    files' own frame rate. Blocks whose file is missing get the median of the present ones;
    a result with no files at all raises."""
    from .noise import sigma_from_highfreq

    out = np.full(2 * N_CAMS, np.nan)
    for j, ctype in enumerate(("intensity", "binary")):
        if ctype in d["native"]:
            out[j * N_CAMS:(j + 1) * N_CAMS] = sigma_from_highfreq(d["native"][ctype])
    if np.isnan(out).all():
        raise ValueError("no curve file was found; cannot estimate the noise")
    return np.where(np.isnan(out), np.nanmedian(out), out)


def fit_conventions(verts: np.ndarray, faces: np.ndarray, curves56: np.ndarray,
                    mask: np.ndarray, m: int) -> dict:
    """Estimate the two signs (sigma, delta) for a public model with a known mesh.

    Minimises the summed squared misfit between the measured normalised curves and the
    normalised convex-operator curves of the mesh's convex hull, over {+-1} x {+-1}. The true
    body may be non-convex; the hull is enough to identify the signs, which is all these
    curves can determine about the conventions. The photometry itself is not fitted here: the
    intensity kernel's exponent is a property of the channel and is measured once
    (conventions.TRANSFER_EXPONENT), and the binary kernel's level is derived from the body's
    own first frame rather than searched. Returns the best candidate with its misfit under
    'err'.
    """
    cams = build_cameras()
    types = ["intensity"] * N_CAMS + ["binary"] * N_CAMS
    hv, hf = hull_mesh(verts)
    best = None
    for sigma in (1.0, -1.0):
        for delta in (1.0, -1.0):
            sim = mesh_curves_convex(hv, hf, cams + cams, m, types,
                                     sigma=sigma, delta=delta)
            r = (normalize_np(sim) - curves56) * mask[:, None]
            err = float((r ** 2).sum())
            if best is None or err < best["err"]:
                best = {"sigma": sigma, "delta": delta, "err": err}
    return best
