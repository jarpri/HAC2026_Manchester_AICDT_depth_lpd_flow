"""Vendored FlexiCubes (NVIDIA nv-tlabs/FlexiCubes), Apache License 2.0.

Unmodified copies of flexicubes.py and tables.py; LICENSE is the upstream file. Vendored
rather than depended on because the package is not published to PyPI, and pinning the
extraction code matters: the specification requires FlexiCubes specifically, since
Marching Cubes staircasing inflates the silhouette perimeter and that error goes straight
into the binary channel.
"""
from .flexicubes import FlexiCubes

__all__ = ["FlexiCubes"]
