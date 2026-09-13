#!/usr/bin/env python3
"""Download public asteroid shape models for the library's "real" family.

    python scripts/fetch_shape_models.py            # into dataset/shape_models

Two sources: the radar shape models NASA JPL publishes, and the spacecraft models of Eros,
Mathilde and Itokawa from the PDS Small Bodies Node. Mithra is left out: it is the third
public model, and a public body never enters the training set, or the score on it would
say nothing; Vesta likewise. Files already present are kept; a failed download is reported
and skipped, so the script can be rerun. The directory is then passed to
scripts/build_shape_library.py --shape-models, and scripts/run_remote_pipeline.sh does that
when the directory has files.

Other non-convex models worth adding by hand into the same directory, in OBJ, PLY or STL:
Bennu, Ryugu, Phobos, Deimos, Ida, Gaspra, Lutetia, Steins, Dinkinesh, Didymos, Dimorphos,
Arrokoth and the comets 67P, Tempel 1, Hartley 2, Wild 2 and Borrelly, from the NASA 3D
resources site, the PDS Small Bodies Node and ESA's archives. DAMIT's models are convex, so
they add nothing to this family.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import truststore

# The stdlib ssl module verifies against a static CA bundle that can lag behind the OS's own
# trust store; JPL's host chains through a Sectigo root recent enough to be missing from it,
# which urlopen reports as "self-signed certificate in certificate chain" even though the
# system (and curl) trusts the chain fine. truststore delegates verification to the OS trust
# store instead, so this stays real certificate verification, just against the same roots
# curl already uses.
truststore.inject_into_ssl()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.shape_library import read_shape_model                     # noqa: E402

JPL = "https://echo.jpl.nasa.gov/asteroids/shapes/"
PDS = "https://sbnarchive.psi.edu/pds3/"
MODELS = {
    **{name: JPL + name for name in (
        "psyche.v.final.mod.wf", "kleo.obj", "kleo.v2.7.1148.mod.wf", "betulia.obj",
        "geographos.obj", "bacchus.obj", "rashalom.obj", "toutatis.obj", "hirestoutatis.obj",
        "Nereus_alt1.mod.wf", "castalia.obj", "golevka.obj", "1996hw1.obj", "sk.obj",
        "sf36.v.mod.wf", "1950DA_ProgradeModel.wf", "1950DA_RetrogradeModel.wf", "wt24.obj",
        "ml14.obj", "yorp.obj", "kw4a.obj", "kw4b.obj", "1994CC_nominal.mod.wf", "ky26.obj",
        "ce26.obj", "2008ev5.obj")},
    "eros022540.tab": PDS + "near/NEAR_A_5_COLLECTED_MODELS_V1_0/data/msi/eros022540.tab",
    "253mathilde.tab": PDS + "near/NEAR_A_5_COLLECTED_MODELS_V1_0/data/msi/253mathilde.tab",
    "itokawa_ver64q.tab": PDS + "hayabusa/HAY_A_AMICA_5_ITOKAWASHAPE_V1_0/data/vertex/ver64q.tab",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/shape_models")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok, bad = [], []
    for name, url in MODELS.items():
        dst = out / name
        if not dst.exists():
            try:
                # the PDS archive refuses requests without a browser-like user agent
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    dst.write_bytes(r.read())
            except Exception as exc:                          # noqa: BLE001
                bad.append(f"{name}: {exc}")
                continue
        try:
            v, f = read_shape_model(str(dst))
            ok.append(f"{name}: {len(v)} vertices, {len(f)} faces")
        except Exception as exc:                              # noqa: BLE001
            bad.append(f"{name}: downloaded but unreadable ({exc})")
            dst.rename(dst.with_suffix(dst.suffix + ".unreadable"))
    print(f"{len(ok)} models ready in {out}:")
    for line in ok:
        print(f"  {line}")
    if bad:
        print(f"{len(bad)} not available:")
        for line in bad:
            print(f"  {line}")


if __name__ == "__main__":
    main()
