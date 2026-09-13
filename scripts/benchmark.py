#!/usr/bin/env python3
"""Score reconstructions of the public models with the organisers' own two measures.

    python scripts/benchmark.py results/convex results/lpd
    python scripts/benchmark.py results/lpd --models 1 2 3 --pitch 0.05 --label "flow v2"
    python scripts/benchmark.py results/lpd --append BENCHMARKS.md

The challenge sums, per model, one voxel score and one projection score, each in [0, 1]:
a maximum of 2 per model and 14 over the seven secret ones. Only models 1-3 have released
truth, so those three are the whole observable, and 6.0 is the most a recipe can show here.

Both measures are run from `hac26/scoring/official.py`, which transcribes the Python voxel
code and ports the MATLAB projection code out of `Evaluation_measures/`.

Posing. The organisers' code does not re-pose anything: it compares the two files as they sit.
That is right for them -- both their truth and our submission are in the challenge frame -- and
wrong for us, because the *released* truth STLs are at the physical scale of the printed model
(z spans 6 to 8, not 2). So the truth is posed into the challenge frame here, and posed with
`centre_xy=False`: it is already on the rotation axis, and moving it onto its own solid
centroid displaces it (see hac26.shapes.rescale_touch_z). The reconstruction is posed the same
way, and for the same reason -- it too is already in the frame -- which is what
`--centre-xy` exists to contradict, so the cost of getting this wrong stays measurable.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.data_io import public_stl                                    # noqa: E402
from hac26.recon import dice, mesh_occupancy                            # noqa: E402
from hac26.scoring.official import (projection_score,                   # noqa: E402
                                    relative_volume_difference_voxelized)
from hac26.shapes import rescale_touch_z                                # noqa: E402
from hac26.stl_io import save_stl                                       # noqa: E402


def posed(path, centre_xy: bool):
    import trimesh
    m = trimesh.load(str(path), process=False)
    v, f = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    return rescale_touch_z(v, f, centre_xy=centre_xy), f


def score_one(truth_stl, recon_stl, tmp: Path, pitch: float, centre_xy: bool,
              n_dirs: int) -> dict:
    import trimesh
    tv, tf = posed(truth_stl, centre_xy)
    rv, rf = posed(recon_stl, centre_xy)
    tp = tmp / "truth_posed.stl"
    rp = tmp / "recon_posed.stl"
    save_stl(str(tp), tv, tf)
    save_stl(str(rp), rv, rf)

    _, m2 = relative_volume_difference_voxelized(str(tp), str(rp), pitch=pitch)
    vox = float(1.0 - m2)

    ang = np.linspace(0.0, 360.0, n_dirs, endpoint=False)
    proj_rel = float(np.mean([projection_score(str(tp), str(rp), t, axis="z") for t in ang]))
    proj_side = [projection_score(str(tp), str(rp), t, axis="side") for t in ang]

    # our own parity-scan Dice, as a cross-check on the organisers' voxeliser
    e = max(float(np.abs(tv).max()), float(np.abs(rv).max())) * 1.05
    ours = dice(mesh_occupancy(rv, rf, 128, e), mesh_occupancy(tv, tf, 128, e))
    return {"voxel": vox, "proj_released": proj_rel,
            "proj_side": float(np.mean(proj_side)), "proj_side_worst": float(min(proj_side)),
            "ours_parity_dice": float(ours),
            "score": vox + proj_rel}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="directories of Asteroid<NN>.stl reconstructions")
    ap.add_argument("--models", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--pitch", type=float, default=0.05,
                    help="voxel pitch in challenge units (the released example uses 0.05)")
    ap.add_argument("--n-dirs", type=int, default=4, help="projection angles to average")
    ap.add_argument("--centre-xy", action="store_true",
                    help="pose by re-centring on the solid centroid (wrong for posed meshes; "
                         "here so the cost of that choice can be measured)")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="", help="also write the table here")
    a = ap.parse_args()

    out = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for d in a.dirs:
            rows = {}
            print(f"\n{d}{'  [' + a.label + ']' if a.label else ''}")
            print(f"  {'model':>5} {'voxel':>7} {'proj':>7} {'SCORE':>7} | "
                  f"{'proj_side':>9} {'parity':>7}")
            for M in a.models:
                rf = Path(d) / f"Asteroid{M:02d}.stl"
                if not rf.exists():
                    continue
                t0 = time.time()
                r = score_one(public_stl(a.data_dir, M), rf, tmp, a.pitch, a.centre_xy,
                              a.n_dirs)
                rows[M] = r
                print(f"  {M:>5} {r['voxel']:>7.4f} {r['proj_released']:>7.4f} "
                      f"{r['score']:>7.4f} | {r['proj_side']:>9.4f} "
                      f"{r['ours_parity_dice']:>7.4f}   ({time.time()-t0:.0f}s)", flush=True)
            if rows:
                tv = sum(r["voxel"] for r in rows.values())
                tp_ = sum(r["proj_released"] for r in rows.values())
                print(f"  {'SUM':>5} {tv:>7.4f} {tp_:>7.4f} {tv + tp_:>7.4f}"
                      f"   over {len(rows)} models (max {2 * len(rows)})")
            out[d] = rows

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
