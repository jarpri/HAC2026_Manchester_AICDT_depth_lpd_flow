#!/usr/bin/env python3
"""Download SPICE DSK shape kernels for mission-imaged non-convex bodies and convert them to
STL for the library's "real" family, alongside scripts/fetch_shape_models.py.

DSK ("digital shape kernel") is the SPICE toolkit's own binary shape format;
hac26.shape_library.read_shape_model does not parse it. This script downloads each body's
DSK from its mission's NAIF/PDS SPICE archive, extracts the vertex/plate mesh with spiceypy,
decimates it when the source is far denser than anything else in the library
(hac26.forward.mesh.exact.decimate), and writes a binary STL -- the same format
scripts/reconstruct.py and hac26.shape_library already read everywhere else, so these bodies
need no special handling: build_shape_library.py --shape-models picks them up exactly like
the JPL radar / PDS spacecraft models.

    python scripts/fetch_dsk_shapes.py                    # into dataset/shape_models

Sources, all mission shape models (not lightcurve inversions):
    67P/Churyumov-Gerasimenko, Lutetia, Steins, Phobos, Deimos  -- Rosetta (ESA/NASA SPICE)
    Bennu                                                       -- OSIRIS-REx
    Ryugu                                                       -- Hayabusa2 (JAXA/NASA SPICE)
    Didymos, Dimorphos                                          -- DART / Hera

Lutetia and Phobos/Deimos duplicate bodies scripts/fetch_shape_models.py or DAMIT can also
reach, but at much higher fidelity (flyby imagery, not lightcurve inversion or radar); the
rest -- 67P, Bennu, Ryugu, Didymos, Dimorphos, Steins -- are new bodies, several of them
(67P, Bennu, Dimorphos) among the most dramatically non-convex real shapes known.

Raw .bds downloads are cached under <out>/_dsk_raw/ (dataset/ is entirely gitignored) so a
rerun does not refetch them; only the target's decimation/output changing invalidates it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import truststore
import urllib.request

truststore.inject_into_ssl()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.forward.mesh.exact import decimate                                     # noqa: E402
from hac26.stl_io import save_stl                                                 # noqa: E402

NAIF = "https://naif.jpl.nasa.gov/pub/naif/pds"
ROSETTA_DSK = f"{NAIF}/data/ro_rl-e_m_a_c-spice-6-v1.0/rossp_1000/DATA/DSK"

# name -> (url, decimate target or None). Resolutions were picked in the low tens-of-thousands
# of plates, matching the scale of the other real models already in the library; Phobos's
# only modest-size file is still far denser than that, so it gets decimated after conversion.
TARGETS = {
    "67p": (f"{ROSETTA_DSK}/ROS_CG_K024_OSPCLPS_N_V2.BDS", None),
    "lutetia_rosetta": (f"{ROSETTA_DSK}/ROS_LU_K025_OSPCLAM_N_V1.BDS", None),
    "steins": (f"{ROSETTA_DSK}/ROS_ST_K020_OSPCLAM_N_V1.BDS", None),
    "phobos": (f"{ROSETTA_DSK}/PHOBOS_K275_DLR_V02.BDS", 8000),
    "deimos": (f"{ROSETTA_DSK}/DEIMOS_K005_THO_V01.BDS", None),
    "bennu": (f"{NAIF}/pds4/orex/orex_spice/spice_kernels/dsk/"
              "bennu_g_12600mm_alt_obj_0000n00000_v021.bds", None),
    "ryugu": (f"{NAIF}/pds4/hyb2/hyb2_spice/spice_kernels/dsk/"
              "ryugu_shape_sfm_49k_v20180804.bds", None),
    "didymos": (f"{NAIF}/pds4/dart/dart_spice/spice_kernels/dsk/"
                "didymos_g_09309mm_spc_0000n00000_v003.bds", None),
    "dimorphos": (f"{NAIF}/pds4/dart/dart_spice/spice_kernels/dsk/"
                  "dimorphos_g_01940mm_spc_0000n00000_v004.bds", None),
}


def download(url: str, dst: Path, timeout: float) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as fh:
        fh.write(r.read())


def dsk_to_mesh(path: Path) -> tuple:
    """(verts, faces) of a DSK's first (type 2, plate/vertex) segment, 0-indexed faces."""
    import spiceypy as spice
    handle = spice.dasopr(str(path))
    try:
        dladsc = spice.dlabfs(handle)
        nv, npl = spice.dskz02(handle, dladsc)
        v = np.array(spice.dskv02(handle, dladsc, 1, nv), dtype=np.float64)
        f = np.array(spice.dskp02(handle, dladsc, 1, npl), dtype=np.int64) - 1
    finally:
        spice.dascls(handle)
    return v, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/shape_models")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS),
                    help="subset of " + ", ".join(TARGETS))
    a = ap.parse_args()
    out = Path(a.out)
    raw_dir = out / "_dsk_raw"

    ok, bad = [], []
    for name in a.targets:
        if name not in TARGETS:
            bad.append(f"{name}: not one of {list(TARGETS)}")
            continue
        url, target_faces = TARGETS[name]
        raw = raw_dir / Path(url).name
        stl_path = out / f"{name}.stl"
        if stl_path.exists():
            ok.append(f"{name}: already at {stl_path}")
            continue
        try:
            download(url, raw, a.timeout)
            v, f = dsk_to_mesh(raw)
            if target_faces and len(f) > target_faces:
                v, f = decimate(v, f, target_faces)
            out.mkdir(parents=True, exist_ok=True)
            save_stl(str(stl_path), v, f, header=name[:80])
            ok.append(f"{name}: {len(v)} vertices, {len(f)} faces -> {stl_path}")
        except Exception as exc:                                  # noqa: BLE001
            bad.append(f"{name}: {exc}")

    print(f"{len(ok)} DSK shapes converted in {out}:")
    for line in ok:
        print(f"  {line}")
    if bad:
        print(f"{len(bad)} failed:")
        for line in bad:
            print(f"  {line}")


if __name__ == "__main__":
    main()
