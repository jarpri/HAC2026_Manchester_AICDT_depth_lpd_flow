#!/usr/bin/env python3
"""Check the rotation and camera conventions against a public body's true shape and curves.

    PYTHONPATH=. python scripts/validate_generation.py --model 1 --data dataset/raw

The true STL of a public model is rendered with `hac26.curves_mesh.render_curves_mesh`, a
CPU renderer, and compared with the measured curves for that model. Agreement is evidence
about the conventions (rotation sense, camera geometry, thresholds), not about the scattering
law: the renderer has no interreflection and no sensor chain, so a residual offset after the
best-fit delta is expected; a sign or phase error is not.

Needs the challenge data under --data (see the README). `--stl` takes a path; without it,
every .stl under --data is searched for one whose name ends in the model number, ignoring
case and zero-padding. Only the public models come with a true shape.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.conventions import cameras                              # noqa: E402
from hac26.curves_mesh import render_curves_mesh                   # noqa: E402
from hac26.data_io import N_CAMS, load_model_curves, fit_conventions  # noqa: E402
from hac26.forward.convex_egi import normalize_np                  # noqa: E402
from hac26.shape_library import pose                               # noqa: E402
from hac26.stl_io import load_stl                                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True, help="public model number")
    ap.add_argument("--data", default="dataset/raw", help="challenge data directory")
    ap.add_argument("--stl", default=None,
                    help="path of the true STL; searched for under --data if omitted")
    ap.add_argument("--m", type=int, default=360, help="frames per revolution")
    ap.add_argument("--res", type=int, default=128, help="renderer image size in pixels")
    a = ap.parse_args()

    stl_path = a.stl
    if stl_path is None:
        # Case and zero-padding vary in the release, so match on the number alone.
        pat = re.compile(rf"asteroid0*{a.model}$", re.IGNORECASE)
        hits = sorted(p for p in Path(a.data).rglob("*.stl") if pat.match(p.stem))
        if not hits:
            raise SystemExit(f"no ground-truth STL for model {a.model} under {a.data}; "
                             f"pass --stl")
        stl_path = str(hits[0])

    verts, faces = load_stl(stl_path)
    verts = pose(verts, radius=1.0, faces=faces)     # challenge pose, same convention
    print(f"loaded {stl_path}: {len(faces)} faces, posed to z in [-1, 1]")

    meas = load_model_curves(a.data, a.model, m=a.m)
    curves56, mask = meas["curves"], meas["mask"]
    if mask.sum() == 0:
        raise SystemExit(f"no measured curves found for model {a.model} under {a.data}")

    fit = fit_conventions(verts, faces, curves56, mask, a.m)
    print(f"fitted conventions on the convex hull: {fit}")

    geoms = cameras() + cameras()
    types = ["intensity"] * N_CAMS + ["binary"] * N_CAMS
    print(f"rendering {len(types)} curves x {a.m} frames at res={a.res} "
          f"(this is the slow, exact step)")
    sim = render_curves_mesh(verts, faces, m=a.m, curve_types=types, geoms=geoms,
                             delta=fit["delta"], res=a.res)
    sim_n = normalize_np(sim)

    err = (sim_n - curves56) * mask[:, None]
    rmse = float(np.sqrt((err ** 2).sum() / max(mask.sum() * a.m, 1)))
    corr = np.array([float(np.corrcoef(sim_n[i], curves56[i])[0, 1])
                     for i in range(len(mask)) if mask[i] > 0])
    print(f"RMSE (mean-normalised units): {rmse:.4f}")
    print(f"per-curve correlation: mean {corr.mean():.3f}, min {corr.min():.3f}")
    print("A mean correlation well above 0, and an RMSE within a small multiple of the "
          "instrument sigma, supports the rotation sense, camera geometry and threshold "
          "conventions. It does not validate the scattering law, which this renderer only "
          "approximates.")


if __name__ == "__main__":
    main()
