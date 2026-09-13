"""Loaders for the two model files shipped in this directory.

    lpd_convex.pt                the trained convex LPD, written by scripts/export_model.py.
                                 It carries its own `preset`, so the network is rebuilt from
                                 the file alone.

    instrument_calibration.pt    the Instrument fitted to the public models' real curves by
                                 scripts/calibrate.py: albedo, source radius, camera distance,
                                 intensity threshold, per-curve pedestal and model error, and
                                 the sensor chain. The fitted start phases live in the
                                 report next to it, instrument_calibration.json.

Both are torch pickles, so torch.load executes their contents; load only copies you trust.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_lpd(device: str = "cpu"):
    """(net, preset, grid) for the trained convex LPD, via hac26.train.load_net."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    from hac26.train import load_net
    return load_net(str(HERE / "lpd_convex.pt"), device=device)


def load_calibration(device: str = "cpu"):
    """The calibrated Instrument."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    from hac26.forward.mesh.instrument import Instrument
    return Instrument.load(HERE / "instrument_calibration.pt", device=device)
