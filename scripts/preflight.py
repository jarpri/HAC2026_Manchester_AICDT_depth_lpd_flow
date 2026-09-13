#!/usr/bin/env python3
"""Everything a long run needs, checked before it starts.

    python scripts/preflight.py --data-dir dataset/raw

A pipeline stage that discovers a missing wheel or an unbuilt extension does so after it has
spent time, and some of those failures do not look like what they are: a missing decimation
package used to surface as a body with no curves, which build_corpus skips, so a whole library
could render into an empty corpus. This runs the cheap version of every dependency the
pipeline has -- the imports, one extraction, one render through the exact forward model, the
instrument if one is there, the data -- and says what is wrong while it is still cheap to fix.

Exits non-zero when anything the pipeline cannot run without is missing. The machine and the
torch build are printed because a tree shared between an x86 cluster and an ARM one is the
usual way a venv or a compiled extension comes to be wrong for the host reading it.
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Compiled, and therefore the ones without a wheel for every architecture. Each is named with
# what stops working without it, so a failure says what it costs rather than only what it is.
COMPILED = {
    "numpy": "everything",
    "scipy": "the convex stage and the spherical designs",
    "torch": "everything",
    "trimesh": "every mesh read, written or repaired",
    "skimage": "the marching-cubes extraction of the consensus bodies",
    "fast_simplification": "the radiosity patches; without it every body renders as having "
                           "no curves and a corpus comes out empty",
    "rtree": "trimesh's spatial queries, used by the mesh repairs",
    "embreex": "trimesh's ray queries; the side-view measure is slow without it",
}
OPTIONAL = {"embreex"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--calibration", default=None,
                    help="instrument to check; by default both channels' files if present")
    ap.add_argument("--skip-render", action="store_true",
                    help="check the imports and the data but not the forward model")
    a = ap.parse_args()
    fatal: list[str] = []
    warn: list[str] = []

    print(f"machine      {platform.machine()}  python {platform.python_version()}")
    for name, why in COMPILED.items():
        try:
            m = importlib.import_module(name)
            v = getattr(m, "__version__", "")
            print(f"  {name:<20} ok {v}")
        except Exception as exc:                       # noqa: BLE001  any import failure
            (warn if name in OPTIONAL else fatal).append(f"{name} ({exc}) -- needed for {why}")
            print(f"  {name:<20} MISSING: {exc}")

    if "torch" not in str(fatal):
        import torch
        print(f"torch        {torch.__version__}  cuda {torch.version.cuda or 'none'}  "
              f"available {torch.cuda.is_available()}"
              + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
        try:
            importlib.import_module("nvdiffrast")
            print("  nvdiffrast         ok")
        except Exception as exc:                       # noqa: BLE001
            warn.append(f"nvdiffrast ({exc}) -- the renders fall back to the pure-torch "
                        f"rasteriser, which is far slower and is not what a run should use")
            print(f"  nvdiffrast         MISSING: {exc}")

    d = Path(a.data_dir)
    if not d.is_dir():
        fatal.append(f"{d} is not a directory -- the curves and the released shapes live there")
        print(f"data         MISSING {d}")
    else:
        curves = sorted(d.glob("AsteroidModel*/*/*lightcurve*.txt"))
        print(f"data         {d}: {len(curves)} lightcurve files")
        if not curves:
            fatal.append(f"{d} holds no lightcurve files")

    from calibrate import OUT_INSTRUMENT
    for ch, path in OUT_INSTRUMENT.items():
        want = a.calibration or path
        if a.calibration and ch != "real":
            continue
        if not Path(want).exists():
            print(f"instrument   {ch:<8} none at {want} (it will be fitted)")
            continue
        try:
            from hac26.forward.mesh.instrument import Instrument
            Instrument.load(want)
            print(f"instrument   {ch:<8} {want} loads against these cameras")
        except Exception as exc:                       # noqa: BLE001
            warn.append(f"{want} does not load ({exc}); the stage will refit it")
            print(f"instrument   {ch:<8} {want} will be refitted: {exc}")

    if not a.skip_render and not fatal:
        t0 = time.time()
        try:
            import numpy as np
            import torch
            from hac26.conventions import psi_grid
            from hac26.field import CODE_DIM, N_DIR, N_NODES, DepthSphere
            from hac26.forward.mesh.exact import RenderConfig
            from hac26.forward.mesh.instrument import Instrument
            from hac26.shapes import (canonicalize_r, icosphere, mesh_support,
                                      rescale_touch_z)
            from hac26.solvers.gauss_newton import cap_depths
            from hac26.solvers.operator import CodeOperator
            v, f = icosphere(2)
            v = canonicalize_r(rescale_touch_z(np.asarray(v) * np.array([1.0, .82, .72]),
                                               f, centre_xy=False))
            body = CodeOperator(Instrument.blender_start(), psi_grid(2), res=16,
                                config=RenderConfig(height=48, width=80, sun_res=32),
                                device="cpu", backend="software")
            h = torch.tensor(mesh_support(v, body.body.core.n.numpy()), dtype=torch.float32)
            code = torch.zeros(CODE_DIM)
            code[N_DIR:] = torch.tensor(
                cap_depths(DepthSphere(N_NODES).u.numpy(), [0.8, 0.45, 0.4], 30.0, 0.25),
                dtype=torch.float32)
            out = body.curves_with_shape(h, code, 1.0, geoms=[0])
            if out is None:
                fatal.append("the forward model returned no curves for a carved test body; "
                             "nothing downstream can run")
                print("render       NO CURVES for a carved test body")
            else:
                cur, area, vol = out
                print(f"render       ok: curves {tuple(cur.shape)}, area {area:.3f}, "
                      f"volume {vol:.3f}  [{time.time()-t0:.0f}s]")
        except Exception as exc:                       # noqa: BLE001
            fatal.append(f"the forward model raised on a test body: {exc!r}")
            print(f"render       RAISED {exc!r}")

    print()
    for w in warn:
        print(f"  warning: {w}")
    for x in fatal:
        print(f"  FATAL:   {x}")
    if fatal:
        print(f"\npreflight failed on {len(fatal)} thing(s). A long run started now would "
              f"spend time and fail.")
        return 1
    print("preflight passed" + (f", with {len(warn)} warning(s)" if warn else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
