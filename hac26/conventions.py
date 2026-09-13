"""Conventions: frames, light, cameras, rotation, geometric constraints.

Everything is computed in the body frame: the mesh never moves and the camera and light
directions are carried into it, which makes the transport phase-independent since body, mount
and turntable are mutually rigid.

Tabulated camera azimuths are measured from the light direction, so the lab azimuth is
180 + az. psi0 is the phase of the body at the first frame. The convex stage and the flow
both assume PSI0, so their frames agree; scripts/calibrate.py fits psi0 per public body
against the released shapes and reports it, which is the check that PSI0 is right.

SENSE is fixed by correlating forward-modelled curves for the public bodies against the real
ones, not by the published wording, which is ambiguous about the direction of rotation.

The published bounding-cylinder radius is treated as an approximation rather than a hard
bound: posed to z in [-1, 1] a public body can sit right at or slightly past its published R,
so clamping to it would shrink true geometry.
"""
from __future__ import annotations

import numpy as np

__all__ = ["S_LAB", "AZIMUTHS_DEG", "TOP_ELEVATION_DEG", "CAM_KINDS", "FRAMES",
           "TRANSFER_EXPONENT",
           "CYLINDER_R", "PUBLIC_MODELS", "Camera", "cameras", "camera_vector",
           "lab_azimuth_deg", "phase_angle_deg", "R_z", "to_body", "source_directions",
           "psi_grid", "SENSE", "PSI0"]

S_LAB = np.array([-1.0, 0.0, 0.0])

AZIMUTHS_DEG = (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0)

# Top-camera elevation per azimuth; the "virtual bottom" camera sits at its negative.
# Measured against the released render of a public body, not taken from the published table,
# which gives 26 degrees at azimuth 135. Every other entry agrees with the table; that one
# does not, and a two-degree error there costs a factor of three in the residual of the two
# columns at the largest phase angle, where the shadows are longest.
TOP_ELEVATION_DEG = {0.0: 21.0, 45.0: 26.0, 90.0: 26.0, 135.0: 24.0,
                     225.0: 24.0, 270.0: 24.0, 315.0: 24.0}

# Column order within each azimuth group, matching the released curve files.
CAM_KINDS = ("hor_a", "hor_b", "top", "bottom")

FRAMES = 360

# Published bounding-cylinder radius of each model, from the challenge page. The published
# value is approximate; see the module docstring.
CYLINDER_R = {1: 1.12, 2: 1.42, 3: 0.88, 4: 1.475, 5: 1.22,
              6: 0.925, 7: 1.205, 8: 1.24, 9: 0.67, 10: 3.95}

# Exponent of the transfer from radiance to stored pixel value, measured on the released
# render of a public body against its released shape. It is a property of the renderer's view
# transform and not of any body, so it is the same for every body and every geometry, and it
# is what makes the intensity curve the gamma-th moment of the illumination cosine rather than
# the disk-integrated brightness an astronomical lightcurve would be.
TRANSFER_EXPONENT = 0.475

# The models whose true shape was released.
PUBLIC_MODELS = (1, 2, 3)

# Turntable sense, measured against the real curves -- see the module docstring.
SENSE = -1.0

# Phase of the body at the first frame, in radians, for every body; see the module docstring.
PSI0 = 0.0


def lab_azimuth_deg(azimuth_deg: float) -> float:
    """Tabulated azimuth is measured from the light; the lab frame is 180 deg away."""
    return 180.0 + azimuth_deg


def camera_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector from the body centre toward the camera."""
    phi = np.radians(lab_azimuth_deg(azimuth_deg))
    e = np.radians(elevation_deg)
    return np.array([np.cos(e) * np.cos(phi), np.cos(e) * np.sin(phi), np.sin(e)])


def phase_angle_deg(azimuth_deg: float, elevation_deg: float) -> float:
    """Solar phase angle: cos alpha = cos(elevation) cos(azimuth). Constant in time."""
    c = np.cos(np.radians(elevation_deg)) * np.cos(np.radians(azimuth_deg))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class Camera:
    """One viewing geometry: azimuth from the light, elevation, and which of the four cameras
    at that azimuth it is. hac26.geometry has an older Camera class with the same angles that
    the convex operator uses; the two agree."""
    __slots__ = ("azimuth_deg", "elevation_deg", "kind")

    def __init__(self, azimuth_deg: float, elevation_deg: float, kind: str):
        self.azimuth_deg = float(azimuth_deg)
        self.elevation_deg = float(elevation_deg)
        self.kind = kind

    @property
    def v(self) -> np.ndarray:
        return camera_vector(self.azimuth_deg, self.elevation_deg)

    @property
    def phase_angle_deg(self) -> float:
        return phase_angle_deg(self.azimuth_deg, self.elevation_deg)

    def __repr__(self) -> str:
        return (f"Camera(az={self.azimuth_deg:g}, el={self.elevation_deg:g}, "
                f"{self.kind}, alpha={self.phase_angle_deg:.1f})")


def cameras() -> list:
    """Every geometry, in the released column order: per azimuth (hor_a, hor_b, top, bottom)."""
    out = []
    for az in AZIMUTHS_DEG:
        e = TOP_ELEVATION_DEG[az]
        out.append(Camera(az, 0.0, "hor_a"))
        out.append(Camera(az, 0.0, "hor_b"))
        out.append(Camera(az, +e, "top"))
        out.append(Camera(az, -e, "bottom"))
    return out


def R_z(angle_rad: float | np.ndarray) -> np.ndarray:
    """Rotation about z by `angle_rad`."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def to_body(w: np.ndarray, psi: np.ndarray, psi0: float = 0.0) -> np.ndarray:
    """Carry a lab direction into the body frame at each phase: R_z(-psi - psi0) w, returning
    (len(psi), 3)."""
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    a = -psi - psi0
    c, s = np.cos(a), np.sin(a)
    wx, wy, wz = w
    return np.stack([c * wx - s * wy, s * wx + c * wy, np.full_like(c, wz)], axis=1)


def source_directions(delta_rad: float, k: int = 8) -> np.ndarray:
    """k directions standing in for a source disc of angular radius delta about S_LAB, each
    carrying an equal share of the light. They sit on one ring at radius delta/sqrt(2), which
    gives the ring the same mean squared offset as a uniform disc, so the penumbra has the
    right width."""
    if delta_rad <= 0:
        return S_LAB[None, :].copy()
    r = delta_rad / np.sqrt(2.0)
    # basis orthogonal to s_lab
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])
    ang = 2.0 * np.pi * np.arange(k) / k
    d = (S_LAB[None, :] * np.cos(r)
         + np.sin(r) * (np.cos(ang)[:, None] * e1[None, :]
                        + np.sin(ang)[:, None] * e2[None, :]))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def psi_grid(frames: int = FRAMES, sense: float = SENSE) -> np.ndarray:
    """psi_k = sense * 2 pi k / frames, with sense = -1 measured (see module docstring)."""
    return sense * 2.0 * np.pi * np.arange(frames) / frames

def geometry_digest() -> str:
    """Short digest of everything an instrument is fitted against but does not itself hold.

    A calibration is a fit of the sensor chain against curves rendered through the cameras and
    the transfer in this module. Change a camera's elevation or the transfer's exponent and
    every fitted parameter is the answer to a different question, while the saved file still
    loads. The pipeline skips the calibration when an instrument file is present, so an
    instrument that outlived the geometry it was fitted under would be used in silence; this is
    what Instrument.save records and Instrument.load checks.
    """
    import hashlib
    parts = [repr(sorted(TOP_ELEVATION_DEG.items())), repr(AZIMUTHS_DEG), repr(CAM_KINDS),
             repr(S_LAB.tolist()), repr(SENSE), repr(FRAMES), repr(TRANSFER_EXPONENT)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
