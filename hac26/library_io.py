"""Disk format for a saved shape library: one .npz per body, plus a JSON manifest.

Kept separate from the generator in `shape_library.py` because reading a saved library, as
`scripts/fit_shapes.py` does, has nothing to do with building one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .shape_library import Body

__all__ = ["save_body", "load_body", "write_manifest", "read_manifest", "load_library_dir"]


def save_body(path: str, body: Body) -> None:
    """Write one body as a compressed .npz: float32 verts, int32 faces, recipe and info as
    JSON strings."""
    np.savez_compressed(
        path,
        verts=body.verts.astype(np.float32),
        faces=body.faces.astype(np.int32),
        recipe=json.dumps(body.recipe),
        info=json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in body.info.items()}),
    )


def load_body(path: str) -> Body:
    """Read a body written by `save_body`, with verts as float64 and faces as int64."""
    z = np.load(path, allow_pickle=False)
    return Body(z["verts"].astype(np.float64), z["faces"].astype(np.int64),
               json.loads(str(z["recipe"])), json.loads(str(z["info"])))


def write_manifest(directory: str, entries: list[dict]) -> None:
    """Write `manifest.json` in `directory`. Each entry is a dict with at least "file",
    "index", "base" and "convexity"."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(Path(directory) / "manifest.json", "w") as fh:
        json.dump({"n": len(entries), "entries": entries}, fh, indent=1)


def read_manifest(directory: str) -> dict:
    """The manifest written by `write_manifest`; raises FileNotFoundError if there is none."""
    p = Path(directory) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no manifest.json in {directory}; run scripts/build_shape_library.py first")
    with open(p) as fh:
        return json.load(fh)


def load_library_dir(directory: str, n: int | None = None, seed: int | None = None,
                     with_entries: bool = False) -> list:
    """Bodies from a directory `scripts/build_shape_library.py` wrote, as (verts, faces)
    pairs, or (verts, faces, manifest entry) triples with `with_entries`; the entry carries
    the body's family under "base".

    `seed` shuffles the manifest order before truncating to `n`, so two callers asking for
    different counts do not both get the library's first bodies.
    """
    man = read_manifest(directory)
    entries = list(man["entries"])
    if seed is not None:
        np.random.default_rng(seed).shuffle(entries)
    if n is not None:
        entries = entries[:n]
    out = []
    for e in entries:
        b = load_body(str(Path(directory) / e["file"]))
        out.append((b.verts, b.faces, e) if with_entries else (b.verts, b.faces))
    return out
