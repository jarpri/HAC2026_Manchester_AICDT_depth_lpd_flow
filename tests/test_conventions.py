"""Tests of hac26.conventions.

The three published phase angles are the gate: if they do not come out, the camera convention
is wrong and nothing built on it can be trusted. The remaining tests pin what those three
numbers do not touch (camera table, rotation sense, source disc), and the last one reads the
public shapes, since the published radius is a claim about the data.
"""
import numpy as np
import pytest

from hac26.conventions import (AZIMUTHS_DEG, FRAMES, S_LAB, SENSE, TOP_ELEVATION_DEG,
                               TRANSFER_EXPONENT,
                               R_z, camera_vector, cameras, lab_azimuth_deg,
                               phase_angle_deg, psi_grid, source_directions, to_body)

DATA = "dataset/raw"


# ---------------------------------------------------------------- the gate

def test_three_phase_angles():
    """The three phase angles the challenge publishes come out of phase_angle_deg."""
    assert phase_angle_deg(0.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    # The challenge quotes 129.4 for this geometry, the exact value truncated to one decimal,
    # so the tolerance admits that.
    assert phase_angle_deg(135.0, 26.0) == pytest.approx(129.46, abs=0.01)
    assert phase_angle_deg(180.0, 0.0) == pytest.approx(180.0, abs=1e-9)


def test_phase_angle_identity_holds_for_every_geometry():
    """cos alpha = v_c . s_lab must equal cos(e) cos(azimuth) identically."""
    for cam in cameras():
        lhs = float(cam.v @ S_LAB)
        rhs = float(np.cos(np.radians(cam.elevation_deg))
                    * np.cos(np.radians(cam.azimuth_deg)))
        assert lhs == pytest.approx(rhs, abs=1e-12)


def test_azimuth_zero_is_coaxial():
    """At azimuth 0, elevation 0 the camera looks along the light: v_c == s_lab."""
    assert camera_vector(0.0, 0.0) == pytest.approx(S_LAB, abs=1e-12)


# ---------------------------------------------------------------- structure

def test_camera_table():
    """Four cameras per azimuth in the released column order, with the published elevations,
    and no camera at the azimuth that would look into the light."""
    cams = cameras()
    assert len(cams) == 28
    assert lab_azimuth_deg(0.0) == 180.0
    for i, az in enumerate(AZIMUTHS_DEG):
        block = cams[4 * i: 4 * i + 4]
        assert [c.kind for c in block] == ["hor_a", "hor_b", "top", "bottom"]
        assert block[0].elevation_deg == 0.0 and block[1].elevation_deg == 0.0
        assert block[2].elevation_deg == TOP_ELEVATION_DEG[az]
        assert block[3].elevation_deg == -TOP_ELEVATION_DEG[az]
    assert 180.0 not in AZIMUTHS_DEG          # would stare into the beam


def test_camera_vectors_are_unit():
    """Every camera direction has unit length."""
    for cam in cameras():
        assert np.linalg.norm(cam.v) == pytest.approx(1.0, abs=1e-12)


def test_top_and_bottom_are_z_mirrors():
    """The z -> -z mirror maps each top camera onto its bottom counterpart exactly."""
    cams = cameras()
    for i in range(len(AZIMUTHS_DEG)):
        top, bot = cams[4 * i + 2].v, cams[4 * i + 3].v
        assert top * np.array([1, 1, -1]) == pytest.approx(bot, abs=1e-12)


# ---------------------------------------------------------------- rotation

def test_to_body_matches_explicit_rotation():
    """to_body(v, psi, psi0) equals R_z(-psi - psi0) @ v frame by frame."""
    psi = psi_grid(FRAMES)[[0, 1, 90, 359]]
    psi0 = np.radians(-2.0)
    got = to_body(S_LAB, psi, psi0)
    want = np.stack([R_z(-p - psi0) @ S_LAB for p in psi])
    assert got == pytest.approx(want, abs=1e-12)


def test_rotation_preserves_z_and_norm():
    """Carrying a vector into the body frame keeps its length and its z component."""
    v = camera_vector(45.0, 26.0)
    b = to_body(v, psi_grid(37))
    assert np.allclose(np.linalg.norm(b, axis=1), 1.0)
    assert np.allclose(b[:, 2], v[2])          # rotation about z cannot change z


def test_turntable_sense_is_the_measured_one():
    """SENSE is the value measured against the real curves (see hac26.conventions), and
    psi_grid carries it. This guards the constant against being flipped."""
    assert SENSE == -1.0
    assert psi_grid(4)[1] < 0.0


def test_full_revolution_is_identity():
    """Rotating by 2 pi returns the vector to its starting position."""
    v = camera_vector(90.0, 0.0)
    assert to_body(v, np.array([0.0]))[0] == pytest.approx(
        to_body(v, np.array([2 * np.pi]))[0], abs=1e-12)


def test_relative_geometry_is_phase_independent():
    """Rotating light and camera together cannot change the phase angle."""
    psi = psi_grid(24)
    for cam in cameras()[:8]:
        s, v = to_body(S_LAB, psi), to_body(cam.v, psi)
        assert np.allclose((s * v).sum(1), float(cam.v @ S_LAB), atol=1e-12)


# ---------------------------------------------------------------- source disc

def test_source_disc():
    """source_directions returns k unit vectors centred on S_LAB within the given angular
    radius, and a single direction for radius zero."""
    d = source_directions(np.radians(1.5), k=8)
    assert d.shape == (8, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    # centred on s_lab, and within the stated angular radius
    assert d.mean(0) / np.linalg.norm(d.mean(0)) == pytest.approx(S_LAB, abs=1e-9)
    ang = np.degrees(np.arccos(np.clip(d @ S_LAB, -1, 1)))
    assert ang.max() <= 1.5 + 1e-9
    assert source_directions(0.0).shape == (1, 3)


# ---------------------------------------------------------------- against the data

@pytest.mark.parametrize("model", [1, 2, 3])
def test_the_stl_origin_is_the_rotation_axis(model):
    """Posed so that z spans exactly [-1, 1] and left where it is in xy, a public body's
    largest xy radius about the STL origin reproduces its published bounding-cylinder radius
    to within 1%. That is the check that the STL origin IS the rotation axis, and so that
    nothing downstream may translate the body in xy. Skipped when the shapes are absent."""
    import os

    from hac26.conventions import CYLINDER_R
    from hac26.data_io import public_stl
    from hac26.shapes import rescale_touch_z
    from hac26.stl_io import load_stl

    f = public_stl(DATA, model)
    if not os.path.exists(f):
        pytest.skip("public STLs not present")
    v, fc = load_stl(f)
    v = rescale_touch_z(v, fc, centre_xy=False)
    r = float(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).max())
    assert r == pytest.approx(CYLINDER_R[model], rel=0.01), (
        f"model {model}: r_max={r:.4f} vs published R={CYLINDER_R[model]}")
    assert abs(v[:, 2].min() + 1.0) < 1e-6 and abs(v[:, 2].max() - 1.0) < 1e-6


@pytest.mark.parametrize("model", [1, 2, 3])
def test_centring_a_public_body_moves_it_off_the_axis(model):
    """The other half of the same statement: putting a released body's solid centroid on the
    axis moves it off, and makes the published radius fit worse -- for model 2 it pushes the
    body outside the published *minimal* enclosing radius, which the true body cannot be.
    This is why the released shapes and the reconstructions are posed with centre_xy=False."""
    import os

    from hac26.conventions import CYLINDER_R
    from hac26.data_io import public_stl
    from hac26.shapes import rescale_touch_z
    from hac26.stl_io import load_stl

    f = public_stl(DATA, model)
    if not os.path.exists(f):
        pytest.skip("public STLs not present")
    from hac26.shapes import solid_centroid

    v, fc = load_stl(f)
    def r_of(**kw):
        w = rescale_touch_z(v, fc, **kw)
        return float(np.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2).max())
    err_axis = abs(r_of(centre_xy=False) - CYLINDER_R[model])
    err_centred = abs(r_of(centre_xy=True) - CYLINDER_R[model])
    posed = rescale_touch_z(v, fc, centre_xy=False)
    offset = float(np.linalg.norm(solid_centroid(posed, fc)[:2]))
    why = (f"model {model}: centroid {offset:.4f} off the axis, |r - R| about the origin "
           f"{err_axis:.4f}, centred {err_centred:.4f}")

    # R is published to three decimals, so differences below that are not differences.
    assert err_axis <= err_centred + 1e-3, why
    if offset > 0.01:
        # a body whose centroid really is off the axis: centring it is strictly worse, and
        # for model 2 it pushes r_max past the published minimal enclosing radius
        assert err_centred > err_axis, why
        assert r_of(centre_xy=True) > CYLINDER_R[model], why



def test_the_camera_elevations_are_the_ones_the_render_was_made_at():
    """Six of the seven top-camera elevations are the published ones and the seventh is not.

    The elevation of each is measurable against the released render of a body whose shape is
    released, because a two-degree error there costs a factor of three to five in the
    residual of that column. At azimuth 135 the render says 24 degrees where the published
    table says 26, and that azimuth is one of the two at the largest phase angle, where the
    shadows are longest. The table is held here and in hac26.geometry and the two must not
    drift apart."""
    from hac26.geometry import TOP_ALPHA_DEG
    assert TOP_ELEVATION_DEG == TOP_ALPHA_DEG
    assert TOP_ELEVATION_DEG[135.0] == 24.0
    assert [TOP_ELEVATION_DEG[a] for a in (0.0, 45.0, 90.0, 225.0, 270.0, 315.0)] == \
        [21.0, 26.0, 26.0, 24.0, 24.0, 24.0]


def test_the_transfer_exponent_is_a_property_of_the_channel():
    """It is the exponent of the view transform the released render was written through, so
    it is one number for every body and every geometry. Fitting it per body, or per curve,
    would let it absorb a shape error, since a body that is too large in one direction and a
    transfer that is too steep both flatten a curve's peaks."""
    assert 0.4 < TRANSFER_EXPONENT < 0.55
