#!/usr/bin/env python3
"""Measure the error of the convex stage's support against the released shapes, in the units
the flow's dh correction works in, and compare it with the corrections the corpus trains the
flow to make.

The flow corrects the convex start h_c through h = softplus(inv_softplus(h_c) + expand(dh)),
with dh band-limited to spherical-harmonic degree SH_DEGREE. So the correction the flow has
to make on a public body is

    d = inv_softplus(h_true) - inv_softplus(h_c)

on the core's normals, where h_true is the support of the released shape's convex hull, both
in the canonical frame. Its band-limited part is what dh can express; the remainder is what
no dh can fix and has to be carried by the depths or accepted as error. Both are reported
per model, as root mean squares over the normals. The corpus (scripts/build_corpus.py) is
built the same way on synthetic bodies, so the sizes there should bracket the sizes here; if
the public bodies need larger corrections than any corpus body, the corpus is too convex.

Needs results/convex/AsteroidXX.stl for the public models and dataset/raw.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.conventions import PUBLIC_MODELS                    # noqa: E402
from hac26.data_io import public_stl                           # noqa: E402
from hac26.field import N_DIR, SH_DEGREE, ImplicitBody, real_sh   # noqa: E402
from hac26.shapes import canonicalize_r, mesh_support, rescale_touch_z   # noqa: E402
from hac26.stl_io import load_stl                              # noqa: E402
from reconstruct_lpd import support_from_convex                # noqa: E402
from train_lpd import CORPUS                                   # noqa: E402


def inv_softplus(h: np.ndarray) -> np.ndarray:
    h = np.maximum(h, 1e-6)
    return h + np.log(-np.expm1(-h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--convex-dir", default="results/convex")
    ap.add_argument("--corpus", default=CORPUS)
    a = ap.parse_args()

    normals = ImplicitBody().core.n.numpy().astype(np.float64)
    Y = real_sh(normals, SH_DEGREE)
    band = Y @ np.linalg.pinv(Y)                    # projector onto what dh can express
    print(f"{'model':>5} {'band rms':>9} {'remainder rms':>14} {'max |d|':>8}   "
          f"(inverse-softplus units of the canonical support)")
    rows = []
    for M in PUBLIC_MODELS:
        stl = public_stl(a.data_dir, M)
        if not Path(stl).exists():
            print(f"{M:>5}  ({stl} not found)")
            continue
        h_c = support_from_convex(f"{a.convex_dir}/Asteroid{M:02d}.stl").numpy().astype(np.float64)
        v, f = load_stl(stl)
        v = canonicalize_r(rescale_touch_z(np.asarray(v, dtype=np.float64), np.asarray(f)))
        from scipy.spatial import ConvexHull
        h_t = mesh_support(v[ConvexHull(v).vertices], normals)   # the hull's vertices suffice
        d = inv_softplus(h_t) - inv_softplus(h_c)
        d_band = band @ d
        rows.append(float(np.sqrt((d_band ** 2).mean())))
        print(f"{M:>5} {rows[-1]:>9.4f} {float(np.sqrt(((d - d_band) ** 2).mean())):>14.4f} "
              f"{float(np.abs(d).max()):>8.4f}")
    if not rows:
        return
    print(f"\nThe band rms is the correction the flow has to make on these bodies "
          f"({min(rows):.4f} to {max(rows):.4f}).")
    if Path(a.corpus).exists():
        dh = np.load(a.corpus)["codes"][:, :N_DIR]
        size = np.sqrt((dh ** 2).mean(1))
        print(f"The corpus trains it on corrections of {size.min():.4f} to {size.max():.4f} "
              f"(median {np.median(size):.4f}) over {len(size)} bodies.")
    else:
        print(f"({a.corpus} not found, so no comparison with the corpus corrections.)")


if __name__ == "__main__":
    main()
