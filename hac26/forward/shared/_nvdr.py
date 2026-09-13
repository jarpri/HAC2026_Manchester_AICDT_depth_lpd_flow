"""Import nvdiffrast with its CUDA runtime preloaded.

The extension is built against CUDA 12.9 (scripts/setup_toolchain.sh says why) while torch
ships its own CUDA 13 runtime, so `_nvdiffrast_c.so` needs a libcudart.so.12 that is not on
the default loader path. Preloading it with RTLD_GLOBAL before the import avoids setting
LD_LIBRARY_PATH in every shell. Setting HAC26_SKIP_CUDART_PRELOAD skips the preload.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

_CANDIDATES = [Path.home() / ".local/cuda129/lib", Path.home() / ".local/cuda129/lib64"]


def _preload() -> None:
    if os.environ.get("HAC26_SKIP_CUDART_PRELOAD"):
        return
    for d in _CANDIDATES:
        so = d / "libcudart.so.12"
        if so.exists():
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            return


def load():
    """Return nvdiffrast.torch, or raise ImportError with the reason."""
    _preload()
    import nvdiffrast.torch as dr
    return dr
