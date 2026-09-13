#!/usr/bin/env python3
"""Validate -- and optionally repair -- the STL files that would be submitted.

    python scripts/check_submission.py results/lpd
    python scripts/check_submission.py results/lpd --repair

The challenge is scored by voxelising the reconstruction and comparing it with the truth. How
a voxeliser decides "inside" is not published; a parity scan up a column does not care which
way a face is wound, but a winding-number or normal-based test does, and reads a mesh with
inverted winding as its own complement. A file that scores 0.80 under one and ~0 under the
other is not a file to submit, so this refuses to pass anything that is not a single
watertight component of positive volume with consistent winding.

The pose is checked too, against what the challenge asks for: rotation axis z, the body
touching z = +1 and z = -1, and (given --radius) inside the published bounding cylinder.

Exit status is non-zero if any file fails, so this can gate a submission in CI or a script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.conventions import CYLINDER_R          # noqa: E402
from hac26.solvers.output import export_stl       # noqa: E402

Z_TOL = 1e-3          # how far the extreme z may sit off the +-1 planes
R_TOL = 1e-2          # relative slack on the published bounding radius


def inspect(path: Path, radius: float | None) -> dict:
    """Geometry, topology and pose of one candidate submission file."""
    import trimesh
    m = trimesh.load(str(path), process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    v = np.asarray(m.vertices)
    pieces = m.split(only_watertight=False)
    r_xy = float(np.hypot(v[:, 0], v[:, 1]).max()) if len(v) else float("nan")
    out = {
        "faces": int(len(m.faces)),
        "components": int(len(pieces)),
        "watertight": bool(m.is_watertight),
        "winding_consistent": bool(m.is_winding_consistent),
        "volume": float(m.volume),
        "zmin": float(v[:, 2].min()) if len(v) else float("nan"),
        "zmax": float(v[:, 2].max()) if len(v) else float("nan"),
        "r_xy": r_xy,
    }
    fails = []
    if out["components"] != 1:
        fails.append(f"{out['components']} components")
    if not out["watertight"]:
        fails.append("not watertight")
    if not out["winding_consistent"]:
        fails.append("inconsistent winding")
    if not (out["volume"] > 0.0):
        fails.append(f"volume {out['volume']:.3f} <= 0 (inside out)")
    if abs(out["zmax"] - 1.0) > Z_TOL or abs(out["zmin"] + 1.0) > Z_TOL:
        fails.append(f"z span [{out['zmin']:.4f}, {out['zmax']:.4f}], not [-1, 1]")
    if radius is not None and r_xy > radius * (1.0 + R_TOL):
        fails.append(f"r_xy {r_xy:.3f} outside the published cylinder {radius:.3f}")
    out["fails"] = fails
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="directories holding Asteroid<NN>.stl files")
    ap.add_argument("--repair", action="store_true",
                    help="rewrite any failing file through export_stl and re-check")
    ap.add_argument("--no-radius", action="store_true",
                    help="skip the bounding-cylinder check")
    a = ap.parse_args()

    bad = 0
    for d in a.dirs:
        print(f"\n{d}")
        for path in sorted(Path(d).glob("Asteroid*.stl")):
            model = int(path.stem.replace("Asteroid", ""))
            radius = None if a.no_radius else CYLINDER_R.get(model)
            rep = inspect(path, radius)
            if rep["fails"] and a.repair:
                import trimesh
                m = trimesh.load(str(path), process=True)
                export_stl(str(path), np.asarray(m.vertices), np.asarray(m.faces),
                           strict=False)
                rep = inspect(path, radius)
                rep["repaired"] = True
            mark = "FAIL" if rep["fails"] else "ok  "
            note = "; ".join(rep["fails"]) if rep["fails"] else ""
            if rep.get("repaired"):
                note = ("repaired; " + note) if note else "repaired"
            print(f"  {mark} model {model:>2}  {rep['faces']:>6} faces  "
                  f"vol {rep['volume']:>8.3f}  comps {rep['components']:>4}  "
                  f"r_xy {rep['r_xy']:.3f}  {note}")
            bad += bool(rep["fails"])

    print(f"\n{bad} file(s) failed" if bad else "\nall files pass")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
