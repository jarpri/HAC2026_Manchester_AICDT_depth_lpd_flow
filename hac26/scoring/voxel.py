#!/usr/bin/env python3
"""Voxel overlap of a reconstruction against the public ground truth.

The challenge defines its voxel measure as

    1 - ( #(A \\ B) + #(B \\ A) ) / ( #(A) + #(B) )

which is the Dice coefficient: #(A\\B) + #(B\\A) = #A + #B - 2#(A and B), so the expression
reduces to 2#(A and B) / (#A + #B).

Both meshes are posed with rescale_touch_z before voxelising, since the challenge fixes z to
[-1, 1] and comparing before that pose compares two different frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.data_io import public_stl               # noqa: E402
from hac26.recon import dice, mesh_occupancy       # noqa: E402
from hac26.shapes import rescale_touch_z           # noqa: E402


def load_one_solid(path: str):
    """An STL as a single mesh, warning if it is not closed.

    mesh_occupancy decides inside by a parity scan up each column, which is only valid for a
    closed surface: a hole inverts every cell in the column through it, so an open mesh
    scores wrongly rather than failing. The truth STLs are the organisers' files and are
    taken as they come, so this warns and continues rather than refusing to score.
    """
    import trimesh
    m = trimesh.load(path, process=False)
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"{path}: expected a single solid, got {type(m).__name__} -- a "
                         f"multi-solid STL has no one body to score")
    if not m.is_watertight:
        print(f"  WARNING: {path} is not closed; the parity scan that decides inside is "
              f"only valid for a closed surface, so this score is unreliable", flush=True)
    return m


def score(stl: str, model: int, data_dir: str = "dataset/raw", n: int = 128) -> float:
    """Dice between the reconstruction and the public truth, both posed, on one n^3 grid."""
    r = load_one_solid(stl)
    t = load_one_solid(public_stl(data_dir, model))
    # Both meshes are already in the challenge frame -- the truth as released, the
    # reconstruction as this package builds it -- so the pose only rescales z and leaves the
    # rotation axis where it is. Centring either on its own centroid would slide them apart.
    rv = rescale_touch_z(np.asarray(r.vertices), np.asarray(r.faces), centre_xy=False)
    tv = rescale_touch_z(np.asarray(t.vertices), np.asarray(t.faces), centre_xy=False)
    e = max(float(np.abs(rv).max()), float(np.abs(tv).max())) * 1.05
    occ_r = mesh_occupancy(rv, np.asarray(r.faces), n, e)
    occ_t = mesh_occupancy(tv, np.asarray(t.faces), n, e)
    if not occ_r.any() or not occ_t.any():
        # dice() reads two empty grids as two identical bodies and returns 1.0, so a pair
        # that voxelised to nothing would be reported as a perfect reconstruction
        raise ValueError(f"nothing to compare for model {model}: the reconstruction "
                         f"voxelised to {int(occ_r.sum())} cells and the truth to "
                         f"{int(occ_t.sum())} on a {n}^3 grid of half-width {e:.4g}")
    return float(dice(occ_r, occ_t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", nargs="+", required=True,
                    help="one STL per public model, in the order given by --models")
    ap.add_argument("--models", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    out = {}
    for stl, m in zip(a.stl, a.models):
        out[m] = score(stl, m, a.data_dir, a.n)
        print(f"  model {m}: voxel measure {out[m]:.4f}   {stl}", flush=True)
    print(f"{a.label} summed voxel measure over {len(out)} models: {sum(out.values()):.4f}")
    print(json.dumps({"label": a.label, "dice": out, "sum": sum(out.values())}))


if __name__ == "__main__":
    main()
