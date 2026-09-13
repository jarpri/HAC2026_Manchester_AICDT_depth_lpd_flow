"""Measurement geometry as the convex operator uses it. hac26.conventions is the newer module
with the same angles, used by the mesh chain; the two agree on every camera direction.

World frame: e3 is the rotation axis (the challenge z-axis). The light comes from -e1, so the
direction from the body to the source is OMEGA0 = -e1, constant (parallel beam).

Camera direction (body -> camera) for azimuth theta, elevation eps, and azimuth handedness
delta in {+1, -1}:

    omega_c = R3(delta*theta) @ (-cos(eps) e1 + sin(eps) e3)
            = (-cos(eps)cos(theta), -delta cos(eps)sin(theta), sin(eps))

theta = 0, eps = 0 gives omega_c = OMEGA0: the camera looks along the beam, phase angle 0.

Rotation: psi = sigma * 2 pi k / m for frame k of an m-frame revolution, with the sense
sigma in {+1, -1}. The values fitted on the public models are in hac26.train.Preset (sigma,
delta) and hac26.conventions.SENSE. The body frame equals the world frame at frame 0.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# --- challenge constants (fips.fi HAC 2026 page) ---------------------------------
AZIMUTHS_DEG: tuple = (0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0)
TOP_ALPHA_DEG: dict = {0.0: 21.0, 45.0: 26.0, 90.0: 26.0, 135.0: 24.0,
                       225.0: 24.0, 270.0: 24.0, 315.0: 24.0}
CAM_KINDS: tuple = ("hor_a", "hor_b", "top", "bottom")  # column order within each azimuth group
OMEGA0 = np.array([-1.0, 0.0, 0.0])


@dataclass(frozen=True)
class Camera:
    """One viewing geometry, with the camera direction for a given azimuth handedness."""
    azimuth_deg: float
    elevation_deg: float
    kind: str  # one of CAM_KINDS

    def omega(self, delta: float = 1.0) -> np.ndarray:
        th = np.deg2rad(self.azimuth_deg)
        ep = np.deg2rad(self.elevation_deg)
        return np.array([-np.cos(ep) * np.cos(th),
                         -delta * np.cos(ep) * np.sin(th),
                         np.sin(ep)])

    @property
    def phase_angle_deg(self) -> float:
        """alpha_c = arccos<omega_c, OMEGA0> = arccos(cos eps cos theta); constant in time."""
        th = np.deg2rad(self.azimuth_deg)
        ep = np.deg2rad(self.elevation_deg)
        return float(np.rad2deg(np.arccos(np.clip(np.cos(ep) * np.cos(th), -1.0, 1.0))))


def build_cameras() -> list:
    """Every geometry in the released column order: per azimuth (hor_a, hor_b, top, bottom)."""
    cams = []
    for az in AZIMUTHS_DEG:
        a = TOP_ALPHA_DEG[az]
        cams.append(Camera(az, 0.0, "hor_a"))
        cams.append(Camera(az, 0.0, "hor_b"))
        cams.append(Camera(az, +a, "top"))
        cams.append(Camera(az, -a, "bottom"))
    return cams


def psi_grid(m: int, sigma: float = 1.0, psi0: float = 0.0) -> np.ndarray:
    """Rotation angles of the m frames of one revolution, sigma * 2 pi k / m + psi0.

    Not the same as hac26.conventions.psi_grid, which has the measured sense built in and no
    phase offset. This one serves the convex operator, which receives sigma explicitly.
    Importing the wrong one silently reverses the rotation.
    """
    return sigma * 2.0 * np.pi * np.arange(m) / m + psi0


def body_frame_dirs(omega_world: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """v_k = R3(-psi_k) @ omega_world, shape (m, 3).

    Identity used: <R3(psi) u, w> = <u, R3(-psi) w>, so photometric cosines of the
    rotating body against a fixed world direction w equal cosines of the *fixed*
    body normals against these rotated directions.
    """
    c, s = np.cos(psi), np.sin(psi)
    wx, wy, wz = omega_world
    return np.stack([c * wx + s * wy, -s * wx + c * wy, np.full_like(c, wz)], axis=1)


# --- EGI normal grid ---------------------------------------------------------------
@dataclass(frozen=True)
class NormalGrid:
    n_theta: int
    n_phi: int
    normals: np.ndarray   # (N, 3), N = n_theta * n_phi, row-major (theta major)
    theta: np.ndarray     # (n_theta,)
    phi: np.ndarray       # (n_phi,)

    @property
    def n(self) -> int:
        return self.n_theta * self.n_phi


def make_grid(n_theta: int = 24, n_phi: int = 48) -> NormalGrid:
    """Equirectangular grid of unit normals; cell centres avoid the exact poles."""
    theta = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    phi = np.arange(n_phi) * 2.0 * np.pi / n_phi
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    normals = np.stack([np.sin(tt) * np.cos(pp),
                        np.sin(tt) * np.sin(pp),
                        np.cos(tt)], axis=-1).reshape(-1, 3)
    return NormalGrid(n_theta, n_phi, normals, theta, phi)


def cell_index(grid: NormalGrid, u: np.ndarray) -> np.ndarray:
    """Flat grid index of the cell containing unit vector(s) u, shape (..., 3)."""
    u = np.asarray(u, dtype=float)
    th = np.arccos(np.clip(u[..., 2], -1.0, 1.0))
    ph = np.mod(np.arctan2(u[..., 1], u[..., 0]), 2.0 * np.pi)
    p = np.clip((th / np.pi * grid.n_theta).astype(int), 0, grid.n_theta - 1)
    q = (ph / (2.0 * np.pi) * grid.n_phi + 0.5).astype(int) % grid.n_phi
    return p * grid.n_phi + q


def project_closure(g: np.ndarray, normals: np.ndarray, iters: int = 200,
                    tol: float = 1e-12) -> np.ndarray:
    """Alternating projections onto {sum_i g_i u_i = 0} and {g >= 0}: the closest facet-area
    vector that can close into a polytope. Both sets are convex, so the alternating
    projections converge to a point in their intersection."""
    U = normals.T                      # (3, N)
    M = U @ U.T                        # (3, 3)
    Minv = np.linalg.inv(M)
    g = np.clip(np.asarray(g, dtype=float).copy(), 0.0, None)
    for _ in range(iters):
        r = U @ g                      # (3,)
        if float(r @ r) <= tol * max(1.0, float(g @ g)):
            break
        g = g - U.T @ (Minv @ r)
        g = np.clip(g, 0.0, None)
    return g
