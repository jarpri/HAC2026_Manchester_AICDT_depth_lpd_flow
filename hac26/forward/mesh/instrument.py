"""Everything about the measurement that is not the shape: the scene and sensor parameters
the calibration fits on the public models and the exact forward model then uses.

    rho           albedo of the surface, in the radiosity solve
    delta         angular radius of the source disc, in radians
    eye_distance  distance of every camera from the body centre, in canonical units
    tau_i         the fixed threshold below which pixels do not count toward the intensity
                  curve (the binary threshold is Otsu's, computed from the first frame and
                  not a parameter)
    pedestal      a per-curve offset added to every pixel value before thresholding
    eta           a per-curve model-error scale: the part of the residual at the true shape
                  that the noise does not explain. It weights the residual in the
                  calibration and at reconstruction; it does not enter the rendering
    sensor        the chain from radiance to pixel value: SensorModel for the laboratory
                  camera, PowerTransfer for a rendered channel
    interreflection
                  whether light bounces between facets. The laboratory body is matte white
                  and bounces light into its concavities; a rendering without bounce light
                  reproduces a simulated channel that was made without it
    orthographic  whether the cameras sit at infinity. A rendered channel's camera does, and
                  its own projection resolves depth over a range set by the body rather than
                  by the camera distance; eye_distance then means nothing and is not fitted

Every quantity with a range is stored through a squashing function so it stays in range:
sigmoid for rho and tau_i, softplus for delta, eye_distance and eta. `interreflection` and
`orthographic` are switches and are saved with the parameters, so a loaded instrument renders
as it was fitted; a caller cannot forget to set them, which is why they live here and not in
the RenderConfig, whose fields are resolutions.

The released data carry two channels, the laboratory curves and the organisers' Blender
render of the true shape, and they are not the same measurement. The render has no lens,
no sensor and no bounce light, and its camera sits far from the body, so an instrument fitted
to the laboratory curves is the wrong instrument for it. `Instrument.blender_start` is the
starting point of a calibration against the render: cameras at infinity, no interreflection
and a power-law encoding of the sRGB kind, with the threshold, the transfer curve and the
source disc free.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hac26.conventions import TRANSFER_EXPONENT

from .sensor import PowerTransfer, SensorModel

__all__ = ["Instrument", "N_CURVES"]

N_CURVES = 56          # the released curves: every camera geometry, intensity then binary


# Parameters are stored unsquashed, so a value of exactly zero is minus infinity there. A
# stored infinity is finite in the forward direction but poisons any arithmetic on the
# parameter itself, so the inverses are floored: below this the squashed value is zero to
# float32 anyway.
_UNSQUASHED_FLOOR = -30.0


def _inv_softplus(x: float) -> float:
    if x <= 0.0:
        return _UNSQUASHED_FLOOR
    return float(max(np.log(np.expm1(x)), _UNSQUASHED_FLOOR))


def _inv_sigmoid(x: float) -> float:
    if x <= 0.0:
        return _UNSQUASHED_FLOOR
    if x >= 1.0:
        return -_UNSQUASHED_FLOOR
    return float(np.clip(np.log(x / (1.0 - x)), _UNSQUASHED_FLOOR, -_UNSQUASHED_FLOOR))


class Instrument(nn.Module):
    """The fitted scene and sensor parameters. Construct with the starting values; load a
    calibration with `load`."""

    # rho starts at the middle of the range of asteroid analogue materials rather than near
    # one. It is fitted, so this is a starting point, but it is the starting point of a
    # parameter that sets how strongly light bounces between facets, and the interreflection
    # amplification 1 / (1 - rho * max row sum) is several-fold near one and about a quarter
    # at a fifth. Bounced light fills concavities, so a high albedo drags a carved body's
    # curves toward its convex hull's; measured on two library bodies, the separation between
    # a body and its own hull in the intensity curves is several times larger at 0.2 than at
    # 0.85, while the binary curves, which are geometry, barely move.
    def __init__(self, rho: float = 0.20, delta_deg: float = 1.0, eye_distance: float = 8.0,
                 tau_i: float = 0.02, eta: float = 0.02, n_curves: int = N_CURVES,
                 quantise: bool = True, interreflection: bool = True,
                 orthographic: bool = False, sensor: nn.Module | None = None):
        super().__init__()
        self.sensor = SensorModel(quantise=quantise) if sensor is None else sensor
        # buffers rather than attributes, so that save and load carry them
        self.register_buffer("interreflection", torch.tensor(bool(interreflection)))
        self.register_buffer("orthographic", torch.tensor(bool(orthographic)))
        self.raw_rho = nn.Parameter(torch.tensor(_inv_sigmoid(rho)))
        self.raw_delta = nn.Parameter(torch.tensor(_inv_softplus(np.radians(delta_deg))))
        self.raw_eye = nn.Parameter(torch.tensor(_inv_softplus(eye_distance)))
        self.raw_tau_i = nn.Parameter(torch.tensor(_inv_sigmoid(tau_i)))
        self.n_curves = n_curves
        self.raw_pedestal = nn.Parameter(torch.full((n_curves // 2,), -6.0))
        self.raw_eta = nn.Parameter(torch.full((n_curves,), _inv_softplus(eta)))

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_rho)

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self.raw_delta)

    @property
    def eye_distance(self) -> torch.Tensor:
        return F.softplus(self.raw_eye)

    @property
    def tau_i(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_tau_i)

    @property
    def pedestal(self) -> torch.Tensor:
        """The per-curve offset added to the image before the curve is formed, laid out as
        the curves are: the intensity curves first, then the binary ones.

        The intensity offset is held below tau_i. The background of a rendered frame is
        exactly zero, so an offset above the threshold puts every background pixel into the
        summed intensity; with a body covering a few per cent of the frame that constant is
        several times the signal, and after the per-curve mean normalisation it flattens the
        curve. Since the threshold is what separates the body from the background, the offset
        belongs below it.

        The binary offset is zero and is not fitted. The binary threshold is Otsu's threshold
        of the same image, which moves with an added constant, so the count above it is
        unchanged by the offset: the offset is a direction the curves cannot see, and fitting
        it only lets the calibration wander along it.
        """
        ped_i = self.tau_i * torch.sigmoid(self.raw_pedestal)
        return torch.cat([ped_i, torch.zeros_like(ped_i)])

    @property
    def eta(self) -> torch.Tensor:
        return F.softplus(self.raw_eta)

    @classmethod
    def blender_start(cls, delta_deg: float = 0.0, tau_i: float = 0.0,
                      eta: float = 0.02) -> "Instrument":
        """The instrument of the released render. Not a starting point: every one of these is
        measured against a released body rather than fitted against it.

        The camera is at infinity, the source is a parallel beam, there is no bounce light,
        and the transfer from radiance to stored value is the power law of the renderer's
        view transform at the measured exponent. The albedo is kept at one because without
        interreflection it only scales the radiance, and the scale is removed when each curve
        is divided by its own mean. What a calibration would otherwise be free to move --
        a penumbra, a lens falloff, a point spread, a spline transfer -- are all absent from
        a render, and each of them is a direction a fit would use to absorb an error of
        shape instead."""
        return cls(rho=0.999, delta_deg=delta_deg, tau_i=tau_i, eta=eta,
                   interreflection=False, orthographic=True,
                   sensor=PowerTransfer(gamma=TRANSFER_EXPONENT))

    def fitted_parameters(self) -> list:
        """(name, parameter) of everything a calibration moves. Without interreflection the
        albedo is a pure scale of the radiance, indistinguishable from the sensor's
        saturation, so it is not fitted then. Cameras at infinity have no distance and no
        lens whose falloff could be fitted, so neither moves. eta enters the likelihood and
        not the rendering and is fitted separately."""
        skip = {"raw_eta"}
        if not bool(self.interreflection):
            skip.add("raw_rho")
        if bool(self.orthographic):
            skip |= {"raw_eye", "sensor.raw_vignette"}
        if bool(self.orthographic) and not bool(self.interreflection):
            return []            # the rendered channel: measured, not fitted
        return [(n, p) for n, p in self.named_parameters() if n not in skip]

    def scene_parameters(self) -> list:
        """The parameters that change the rendered geometry or transport, as opposed to the
        sensor chain: rho, delta and the eye distance."""
        return [self.raw_rho, self.raw_delta, self.raw_eye]

    def summary(self) -> str:
        def f(x):
            return float(x.detach())
        return (f"rho {f(self.rho):.3f}, source radius {np.degrees(f(self.delta)):.2f} deg, "
                f"eye distance "
                f"{'infinite' if bool(self.orthographic) else format(f(self.eye_distance), '.2f')}"
                f", tau_i {f(self.tau_i):.4f}, {self.sensor.describe()}, "
                f"interreflection {'on' if bool(self.interreflection) else 'off'}, "
                f"eta median {f(self.eta.median()):.4f}")

    def save(self, path) -> None:
        """Write the fitted parameters, and with them the geometry they were fitted against.

        See conventions.geometry_digest: the cameras and the transfer are not part of the
        instrument, so a file fitted under one set of them loads happily under another and is
        then the answer to a different question."""
        from ...conventions import geometry_digest
        state = dict(self.state_dict())
        state["_geometry"] = torch.tensor(
            list(bytes.fromhex(geometry_digest())), dtype=torch.uint8)
        torch.save(state, path)

    @classmethod
    def load(cls, path, device="cpu") -> "Instrument":
        """A saved instrument, with the sensor chain it was saved with. Which chain that is
        follows from the keys: a rendered channel's PowerTransfer has a gamma where the
        laboratory camera's SensorModel has spline knots."""
        from ...conventions import geometry_digest
        state = torch.load(path, map_location="cpu", weights_only=True)
        saved = state.pop("_geometry", None)
        if saved is None:
            # A file with no digest was written before the digest existed, so the cameras and
            # the transfer it was fitted against are unknown rather than equal to these. That
            # is the same failure as a mismatch and has to be refused the same way: the
            # pipeline skips the calibration when this file is present, so an instrument
            # carried over from another tree would set the model error every threshold
            # downstream is measured in, in silence.
            raise RuntimeError(
                f"{path} records no camera geometry, so it was written before the geometry "
                f"was recorded and there is no way to tell what it was fitted against. Rerun "
                f"scripts/calibrate.py rather than using it.")
        was = bytes(saved.tolist()).hex()
        if was != geometry_digest():
            raise RuntimeError(
                f"{path} was fitted against different cameras or a different transfer "
                f"({was} against {geometry_digest()}). Every parameter in it is the answer "
                f"to a different question, so rerun scripts/calibrate.py rather than using "
                f"it; the pipeline skips the calibration when this file is present.")
        inst = cls(sensor=PowerTransfer() if "sensor.raw_gamma" in state else None)
        try:
            inst.load_state_dict(state)
        except (RuntimeError, TypeError) as exc:
            raise RuntimeError(f"{path} is not a saved Instrument; rerun "
                               f"scripts/calibrate.py to write one") from exc
        return inst.to(device)
