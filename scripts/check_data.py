#!/usr/bin/env python3
"""Check the downloaded challenge data against dataset/MANIFEST.sha256.

The README tells you to do this and nothing did it, which is how the repository came to pin a
mixed snapshot: models 2 and 3 were refreshed by hand after the organisers re-rendered the
Blender curves on 29 July 2026 (the `.stale29jul` backups that used to be in the manifest are
the trace of it) and model 1 was missed, so its four curve files stayed on the pre-update
versions the organisers have since realigned. A single start phase cannot absorb that -- the
realignment is a different whole-frame shift per azimuth -- so it lands in the calibration's
residuals looking like forward-model error. `calibrate.py` now reports a per-azimuth phase
offset, which is the other half of catching this.

The manifest records paths under `data/raw/`; the README puts the data in `dataset/raw/`.
`--data-dir` is the root the manifest's paths are resolved against, so either layout works.

    python scripts/check_data.py                       # dataset/raw
    python scripts/check_data.py --data-dir data/raw
    python scripts/check_data.py --write               # regenerate the manifest from what is
                                                       # on disk, after a deliberate refresh

Exits non-zero when a file the repository reads is missing or has changed. The download also
carries videos and prose that nothing here opens, and a mismatch in those is printed and
passed over: a reworded Readme is not a data change, and a run that stopped for one would be
stopping on something it never reads.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MANIFEST = Path("dataset/MANIFEST.sha256")
PREFIX = "data/raw/"          # the path prefix the manifest was written with


def is_read(rel: str) -> bool:
    """Whether anything in this repository opens the file, which is what decides whether a
    mismatch against the manifest is fatal.

    The curves are what every fit is measured against and the released meshes are what the
    public scores are measured against, so a changed or missing one of those invalidates a
    run: `data_io.load_model_curves` reads the first and `data_io.public_stl` the second.
    Nothing here opens the rest of the download -- the videos, the prose -- so a mismatch in
    it is worth printing and cannot invalidate anything. The distinction is made here rather
    than by editing the manifest, because a prose file the organisers reword is not a data
    change and refreshing the manifest each time would train the operator to adopt whatever
    is on disk.
    """
    name = Path(rel).name.lower()
    return "_lightcurve_" in name or name.endswith((".stl", ".msh"))


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def entries(manifest: Path):
    """(relative path, hash) for every line, with the manifest's own prefix stripped."""
    out = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rel = rel.strip()
        out.append((rel[len(PREFIX):] if rel.startswith(PREFIX) else rel, digest))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/raw",
                    help="root the manifest's paths are resolved against")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--write", action="store_true",
                    help="rewrite the manifest from what is on disk instead of checking it. "
                         "Only after a refresh you meant to do: it makes whatever you have "
                         "the reference, including a stale copy")
    a = ap.parse_args()
    root = Path(a.data_dir)
    if not root.is_dir():
        print(f"{root} is not a directory; put the challenge data there first")
        return 2

    if a.write:
        files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text("".join(
            f"{sha256(p)}  {PREFIX}{p.relative_to(root)}\n" for p in files))
        print(f"wrote {a.manifest} from {len(files)} files under {root}")
        return 0

    missing, changed, ok = [], [], 0
    for rel, digest in entries(a.manifest):
        path = root / rel
        if not path.exists():
            missing.append(rel)
        elif sha256(path) != digest:
            changed.append(rel)
        else:
            ok += 1

    listed = {rel for rel, _ in entries(a.manifest)}
    extra = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                   if p.is_file() and not p.name.startswith(".")
                   and str(p.relative_to(root)) not in listed)

    print(f"{ok} of {ok + len(missing) + len(changed)} files match {a.manifest}")
    for label, items in (("missing", missing), ("changed", changed), ("not in the manifest", extra)):
        if items:
            read = [q for q in items if is_read(q)]
            print(f"\n{len(items)} {label}"
                  + (f", {len(read)} of them read by this repository" if read and
                     len(read) != len(items) else "") + ":")
            for q in items[:20]:
                print(f"   {q}" + ("" if is_read(q) else "   (not read here)"))
            if len(items) > 20:
                print(f"   ... and {len(items) - 20} more")
    fatal = [q for q in missing + changed if is_read(q)]
    if changed:
        print("\nA changed file is either a download the organisers have since replaced or a\n"
              "manifest entry that was never refreshed. Check the challenge page's News &\n"
              "Updates before assuming the manifest is right, then rerun with --write.")
    if fatal:
        print(f"\n{len(fatal)} of these {'is' if len(fatal) == 1 else 'are'} read by the "
              f"calibration and everything downstream of it.")
        return 1
    if missing or changed:
        print("\nNone of these is read by anything here, so nothing downstream is affected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
