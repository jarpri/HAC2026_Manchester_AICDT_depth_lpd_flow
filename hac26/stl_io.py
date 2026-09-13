"""Minimal STL writer and reader with no mesh library dependency.

Binary STL: 80-byte header, uint32 triangle count, then per triangle
float32 normal[3], float32 v0[3], v1[3], v2[3], uint16 attribute. The reader also accepts
ASCII STL.
"""
from __future__ import annotations

import struct

import numpy as np


def save_stl(path: str, verts: np.ndarray, faces: np.ndarray,
             header: str = "hac26") -> None:
    """Write a binary STL; facet normals follow the vertex order of each face."""
    tri = verts[faces].astype(np.float32)              # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(norm > 1e-20, n / np.maximum(norm, 1e-20), 0.0).astype(np.float32)
    F = len(faces)
    rec = np.zeros(F, dtype=[("n", np.float32, 3), ("v", np.float32, (3, 3)),
                             ("attr", np.uint16)])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii", "ignore")[:80].ljust(80, b"\0"))
        fh.write(struct.pack("<I", F))
        fh.write(rec.tobytes())


def load_stl(path: str) -> tuple:
    """(verts (V, 3), faces (F, 3)) from a binary or ASCII STL, duplicate vertices merged."""
    with open(path, "rb") as fh:
        head = fh.read(80)
        rest = fh.read()
    if head[:5].lower() == b"solid" and b"facet" in rest[:1000]:
        return _load_ascii(path)
    F = struct.unpack("<I", rest[:4])[0]
    rec = np.frombuffer(rest[4:4 + F * 50],
                        dtype=[("n", np.float32, 3), ("v", np.float32, (3, 3)),
                               ("attr", np.uint16)])
    tri = rec["v"].reshape(-1, 3).astype(float)
    verts, inv = np.unique(tri.round(decimals=7), axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3)


def _load_ascii(path: str) -> tuple:
    """ASCII STL: every 'vertex' line in order, three per triangle."""
    pts = []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if t and t[0] == "vertex":
                pts.append([float(t[1]), float(t[2]), float(t[3])])
    tri = np.asarray(pts)
    verts, inv = np.unique(tri.round(decimals=7), axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3)
