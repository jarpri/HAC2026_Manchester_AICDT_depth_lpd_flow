#!/usr/bin/env python3
"""Measure the released public shapes with the shape library's own numbers, next to the
library's, so the library's family weights can be set to cover them.

Per body: the convexity ratio (volume over hull volume); the neck, the deepest dip of the
cross-section along the body's longest axis relative to the larger sections on both sides of
it (near one for a body without a waist, small for a contact binary); and the flat share, the
part of the surface area whose normals pile up in a few directions (near zero for a smooth
body, large for a cube). For the library, the same numbers per family, and the share of each family that
falls inside the public range on all of them. The flow learns its prior from the library, so a
public body outside the library's range on any of these is a body the flow has never seen the
like of.

Needs dataset/raw for the public shapes; --shapes-dir is optional.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.conventions import CYLINDER_R, PUBLIC_MODELS      # noqa: E402
from hac26.data_io import public_stl                          # noqa: E402
from hac26.library_io import load_library_dir                 # noqa: E402
from hac26.library_metrics import design_normals              # noqa: E402
from hac26.recon import mesh_occupancy                        # noqa: E402
from hac26.shape_library import convexity_ratio               # noqa: E402
from hac26.shapes import face_normals_areas, rescale_touch_z  # noqa: E402
from hac26.stl_io import load_stl                             # noqa: E402

RES = 64            # voxel grid of the cross-section measurement
FLAT_BIN_SHARE = 0.02   # a direction bin holding more than this share of the area is a flat face


def neck_ratio(section: np.ndarray) -> float:
    """The deepest dip of a cross-section profile: min over z of section(z) divided by the
    smaller of the largest sections below and above z. One for a profile without a dip."""
    s = section[section > 0].astype(float)
    if len(s) < 3:
        return 1.0
    below = np.maximum.accumulate(s)
    above = np.maximum.accumulate(s[::-1])[::-1]
    return float((s / np.minimum(below, above)).min())


def flat_share(v: np.ndarray, f: np.ndarray) -> float:
    """Share of the surface area whose normals fall in direction bins holding more than
    FLAT_BIN_SHARE of the area each."""
    n, a = face_normals_areas(v, f)
    bins = design_normals()
    share = np.bincount((n @ bins.T).argmax(1), weights=a, minlength=len(bins)) / a.sum()
    return float(share[share > FLAT_BIN_SHARE].sum())


def measures(v: np.ndarray, f: np.ndarray) -> dict:
    """convexity, neck and flat share of a mesh. The neck is taken along the longest
    principal axis of the vertices, so it does not depend on how the body is posed."""
    c = v - v.mean(0)
    _, axes = np.linalg.eigh(c.T @ c)                        # columns: ascending variance
    w = c @ axes                                             # longest axis last -> z
    extent = float(np.abs(w).max()) * 1.05
    occ = mesh_occupancy(w, f, RES, extent)
    return {"convexity": convexity_ratio(v, f),
            "neck": neck_ratio(occ.sum(axis=(0, 1))),
            "flat": flat_share(v, f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--shapes-dir", default=None,
                    help="a library written by scripts/build_shape_library.py")
    ap.add_argument("--library-bodies", type=int, default=400,
                    help="library bodies sampled for the comparison")
    a = ap.parse_args()

    keys = ("convexity", "neck", "flat")
    print(f"{'public':<8} " + " ".join(f"{k:>10}" for k in keys))
    pub = []
    for M in PUBLIC_MODELS:
        stl = public_stl(a.data_dir, M)
        if not Path(stl).exists():
            print(f"{M:<8} ({stl} not found)")
            continue
        v, f = load_stl(stl)
        v = rescale_touch_z(np.asarray(v, dtype=np.float64), np.asarray(f), centre_xy=False)
        m = measures(v, f)
        pub.append(m)
        print(f"{M:<8} " + " ".join(f"{m[k]:>10.3f}" for k in keys)
              + f"   (published radius {CYLINDER_R[M]})")
    if not a.shapes_dir:
        return
    if not pub:
        raise SystemExit("no public shape found; nothing to compare the library with")
    lo = {k: min(m[k] for m in pub) for k in keys}
    hi = {k: max(m[k] for m in pub) for k in keys}

    bodies = load_library_dir(a.shapes_dir, n=a.library_bodies, seed=0, with_entries=True)
    rows = {}
    for v, f, e in bodies:
        rows.setdefault(str(e.get("base", "unknown")), []).append(measures(v, f))
    print(f"\n{'library family':<16} {'n':>4} " + " ".join(f"{k + ' median':>17}" for k in keys)
          + "   share inside the public range on all three")
    for fam, ms in sorted(rows.items()):
        arr = {k: np.array([m[k] for m in ms]) for k in keys}
        inside = np.ones(len(ms), dtype=bool)
        for k in keys:
            inside &= (arr[k] >= lo[k]) & (arr[k] <= hi[k])
        print(f"{fam:<16} {len(ms):>4} "
              + " ".join(f"{np.median(arr[k]):>17.3f}" for k in keys)
              + f"   {inside.mean():.2f}")
    print("\nA family whose median sits outside the public range on a number does not supply "
          "bodies like the public ones on that number; LibrarySpec.family_weights and "
          "mod_weights in hac26/shape_library.py set how much of each family the library has.")


if __name__ == "__main__":
    main()
