"""Adapter between a dataset of meshes, with optional stored curves, and the trainer.

Layout under the root, searched recursively (see load_pairs):
    <name>.stl | <name>.obj             ground-truth mesh
    <name>_curves.npz | <name>.npz      optional curves: either 'curves' (56, m) with an
                                        optional 'mask' (56,), or the team schema read by
                                        curves_from_npz
A mesh without a curves file gets curves simulated with the convex operator A passed in.

FigurineCurves yields the same six-tuples as hac26.train.SyntheticCurves:
(d, mask, p, h, log_r, rho).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from hac26.data_io import resample_curves
from hac26.forward.convex_egi import normalize_np
from hac26.radial import fibonacci_sphere, mesh_radial
from hac26.shapes import (canonicalize_r, hull_mesh, mesh_support, mesh_to_egi,
                          rescale_touch_z)
from hac26.stl_io import load_stl
from hac26.noise import NOISE_PROFILE, apply_noise


def _load_obj(path: str) -> tuple:
    """(verts, faces) from a triangle OBJ file; only 'v' and 'f' lines are read."""
    v, f = [], []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            v.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            f.append([int(w.split("/")[0]) - 1 for w in t[1:4]])
    return np.asarray(v), np.asarray(f)


def curves_from_npz(path: str, m: int, eps: float = 1e-3) -> tuple:
    """(d (56, m) float32, mask (56,) float32) from one stored curves npz.

    Accepts the single-array layout ('curves' (56, m), optional 'mask') and the team schema
    ('intensity' and 'binary' of shape (frames, 28) plus 'azimuth' and 'elevation' (28,)),
    whose columns are reordered into challenge-camera order. The curves are resampled onto
    m frames and then mean-normalised per curve.
    """
    z = np.load(path)
    if "curves" in z:                          # single-array layout
        raw = z["curves"]
    else:                                      # team schema: (frames, 28) x 2 + geometry
        from hac26.geometry import build_cameras
        cols = list(zip(np.round(z["azimuth"], 3), np.round(z["elevation"], 3)))
        order, used = [], set()
        for cam in build_cameras():
            j = next(k for k, (a, e) in enumerate(cols) if k not in used
                     and a == round(cam.azimuth_deg, 3)
                     and abs(e - cam.elevation_deg) < 0.51)
            order.append(j)
            used.add(j)
        raw = np.concatenate([z["intensity"].T[order], z["binary"].T[order]])
    d = normalize_np(resample_curves(np.asarray(raw, dtype=float), m), eps=eps).astype(np.float32)
    mask = np.asarray(z.get("mask", np.ones(len(d)))).astype(np.float32)
    return d, mask


def load_pairs(root: str, grid, A: np.ndarray, eps: float = 1e-3,
               canonical_r: bool = False, rays: np.ndarray | None = None) -> list:
    """One tuple (d, mask, p, h, r_true, rho) per mesh under root: normalised curves
    (56, m), availability mask (56,), EGI direction (N,), support function of the posed
    hull (N,), the hull's largest xy distance, and its radial function on `rays` (a single
    zero when rays is None). With canonical_r the targets p and h are those of the hull
    scaled to xy radius 1.

    m is taken from A (56, m, N); stored curves with another frame count are resampled onto
    it, so a dataset can be trained with any preset."""
    m = A.shape[1]
    pairs = []
    for mp in sorted(glob.glob(str(Path(root) / "**" / "*.stl"), recursive=True)
                     + glob.glob(str(Path(root) / "**" / "*.obj"), recursive=True)):
        verts, faces = (_load_obj(mp) if mp.endswith(".obj") else load_stl(mp))
        verts = rescale_touch_z(verts)
        hv, hf = hull_mesh(verts)
        g_true = mesh_to_egi(hv, hf, grid)      # simulated curves come from the body itself
        if canonical_r:
            hv, hf = hull_mesh(canonicalize_r(hv))
        g = mesh_to_egi(hv, hf, grid)
        p = (g / max(g.sum(), 1e-12)).astype(np.float32)
        h = mesh_support(hv, grid.normals).astype(np.float32)
        r_true = float(np.sqrt((verts[:, :2] ** 2).sum(1)).max())
        cands = [Path(mp).with_suffix("").as_posix() + "_curves.npz",
                 Path(mp).with_suffix(".npz").as_posix()]
        cp = next((c for c in cands if Path(c).exists()), None)
        if cp:
            d, mask = curves_from_npz(cp, m, eps=eps)
        else:  # simulate with the convex operator
            raw = np.einsum("cmn,n->cm", A, g_true)
            d = normalize_np(raw, eps=eps).astype(np.float32)
            mask = np.ones(len(d), dtype=np.float32)
        rho = (mesh_radial(hv, hf, rays).astype(np.float32) if rays is not None
               else np.zeros(1, dtype=np.float32))
        pairs.append((d, mask, p, h, r_true, rho))
    return pairs


class FigurineCurves(IterableDataset):
    """Endless stream over the tuples from load_pairs, with the augmentations of
    SyntheticCurves (noise, small cyclic shifts, curve dropout) applied to the stored
    curves. A fraction `mix_synthetic` of the samples are fresh synthetic shapes instead,
    which needs `grid` and `A`."""

    def __init__(self, pairs: list, pr, mix_synthetic: float = 0.25, grid=None,
                 A: np.ndarray | None = None):
        self.pairs, self.pr = pairs, pr
        self.mix, self.grid, self.A = mix_synthetic, grid, A
        self.rays = (fibonacci_sphere(pr.n_rays)
                     if getattr(pr, "dice_weight", 0.0) else None)

    def __iter__(self):
        from hac26.shapes import sample_training_shape
        wi = get_worker_info()
        rng = np.random.default_rng(self.pr.seed + (wi.id + 1) * 9973 if wi else self.pr.seed)
        while True:
            if self.mix > 0 and rng.random() < self.mix:  # a synthetic shape instead
                s = sample_training_shape(rng, self.grid,
                                          p_flat=getattr(self.pr, "p_flat", 0.0))
                raw = np.einsum("cmn,n->cm", self.A, s["g"])
                d0 = normalize_np(raw, eps=self.pr.eps_norm).astype(np.float32)
                mask, p = np.ones(len(d0), np.float32), s["p"].astype(np.float32)
                tv, tf = s["verts"], s["faces"]
                if getattr(self.pr, "canonical_r", False):
                    tv, tf = hull_mesh(canonicalize_r(tv))
                    gc = mesh_to_egi(tv, tf, self.grid)
                    p = (gc / max(gc.sum(), 1e-12)).astype(np.float32)
                h = mesh_support(tv, self.grid.normals).astype(np.float32)
                r_true = float(np.sqrt((s["verts"][:, :2] ** 2).sum(1)).max())
                rho = (mesh_radial(tv, tf, self.rays).astype(np.float32)
                       if self.rays is not None else np.zeros(1, dtype=np.float32))
            else:
                d0, mask, p, h, r_true, rho = self.pairs[rng.integers(len(self.pairs))]
                d0, mask = d0.copy(), mask.copy()
            C, m = d0.shape
            # the same noise-profile choice as SyntheticCurves; the curves are already
            # mean-normalised, hence relative=False
            prof = None if self.pr.noise_profile_mode == "measured" \
                else np.ones_like(NOISE_PROFILE)
            d0 = apply_noise(d0, rng, self.pr.noise_lo, self.pr.noise_hi,
                             profile=prof, relative=False).astype(np.float32)
            for c in range(C):
                sh = int(rng.integers(-self.pr.shift_max, self.pr.shift_max + 1))
                if sh:
                    d0[c] = np.roll(d0[c], sh)
            drop = rng.random(C) < self.pr.drop_p
            mask = mask * (~drop)
            d0 = d0 * mask[:, None]
            r_in = r_true * float(np.exp(rng.normal(0.0, getattr(self.pr, "r_jitter", 0.05))))
            yield (torch.from_numpy(d0), torch.from_numpy(mask.astype(np.float32)),
                   torch.from_numpy(p), torch.from_numpy(h),
                   torch.tensor(np.log(max(r_in, 1e-6)), dtype=torch.float32),
                   torch.from_numpy(rho))
