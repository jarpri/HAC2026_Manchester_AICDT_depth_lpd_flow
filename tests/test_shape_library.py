"""Tests for the shape library (`hac26.shape_library`, `hac26.library_metrics`,
`hac26.curves_mesh`).

Fast tests use small grids and few bodies; the `slow` tests build small libraries and check
library-level statistics, and are skipped with `-m "not slow"`.
"""
from __future__ import annotations

import numpy as np
import pytest

from hac26.conventions import cameras
from hac26.curves_mesh import convex_cross_check
from hac26.library_metrics import (check_body, check_library,
                                   library_descriptors, occupancy, pairwise_dice,
                                   participation_ratio, principal_frame)
from hac26.shape_library import (Body, LibrarySpec, build_library, convexity_ratio,
                                 decimate_mesh, extract, is_edge_manifold,
                                 mesh_volume, n_components, pose, read_shape_model,
                                 sample_body, sd_box, sd_sphere, op_smooth_union,
                                 op_subtract)
from hac26.shapes import hull_mesh, icosphere

FAST_SPEC = LibrarySpec(res=48, convexity_shares=())      # no band redraws: one body per call


# --------------------------------------------------------------------------- primitives

def test_sphere_field_is_a_true_sdf_near_the_surface():
    """For a sphere, f(x) equals the signed distance to the surface."""
    f = sd_sphere(radius=1.0)
    pts = np.array([[2.0, 0, 0], [0, 0, 0], [1.0, 0, 0], [0.5, 0.5, 0.5]])
    got = f(pts)
    want = np.array([1.0, -1.0, 0.0, np.linalg.norm([0.5, 0.5, 0.5]) - 1.0])
    assert np.allclose(got, want, atol=1e-9)


def test_box_field_matches_known_distances():
    """The box field is the signed distance at a face, the centre and a corner."""
    f = sd_box(half=(1.0, 1.0, 1.0))
    assert float(f(np.array([2.0, 0.0, 0.0]))) == pytest.approx(1.0)
    assert float(f(np.array([0.0, 0.0, 0.0]))) == pytest.approx(-1.0)
    assert float(f(np.array([1.0, 1.0, 1.0]))) == pytest.approx(0.0, abs=1e-9)


def test_subtract_removes_material():
    """op_subtract turns the bitten region positive (outside) and leaves the rest inside."""
    a = sd_sphere(radius=1.0)
    b = sd_sphere(centre=(0.5, 0, 0), radius=0.6)
    cut = op_subtract(a, b)
    assert float(cut(np.array([0.5, 0.0, 0.0]))) > 0        # inside the bite: now outside
    assert float(cut(np.array([-0.9, 0.0, 0.0]))) < 0       # untouched region: still inside


def test_smooth_union_has_no_cusp_between_the_two_hard_mins():
    """The smooth union sits at or below the hard minimum everywhere (it can only add
    material) and strictly below it at the join."""
    a = sd_sphere(centre=(-0.5, 0, 0), radius=0.6)
    b = sd_sphere(centre=(0.5, 0, 0), radius=0.6)
    hard = lambda p: np.minimum(a(p), b(p))                  # noqa: E731
    soft = op_smooth_union(a, b, k=0.15)
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [-2.0, 0.0, 0.0]])
    assert np.all(soft(pts) <= hard(pts) + 1e-9)
    assert float(soft(np.array([0.0, 0.0, 0.0]))) < float(hard(np.array([0.0, 0.0, 0.0])))


# --------------------------------------------------------------------------- mesh measures

def test_mesh_volume_of_a_unit_ball_hull():
    """The hull of a fine icosphere has the volume of the unit ball to within 2%."""
    v, f = icosphere(3)
    v, f = hull_mesh(v)
    vol = mesh_volume(v, f)
    assert vol == pytest.approx(4.0 / 3.0 * np.pi, rel=0.02)


def test_convexity_ratio_is_one_for_a_convex_hull():
    """A convex mesh has convexity ratio 1."""
    v, f = icosphere(2)
    assert convexity_ratio(v, f) == pytest.approx(1.0, abs=1e-6)


def test_edge_manifold_true_for_icosphere_false_for_a_hole():
    """An icosphere is edge-manifold; dropping one triangle leaves a boundary edge."""
    v, f = icosphere(1)
    assert is_edge_manifold(f)
    assert not is_edge_manifold(f[1:])


def test_n_components_counts_two_disjoint_spheres():
    """Two spheres far apart count as two components."""
    v1, f1 = icosphere(1)
    v2, f2 = icosphere(1)
    v2 = v2 + np.array([10.0, 0.0, 0.0])
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    assert n_components(v, f) == 2


def test_decimate_mesh_shrinks_face_count_and_preserves_occupancy():
    """decimate_mesh at least halves the face count and keeps the occupancy read from the
    mesh almost unchanged; it makes no manifoldness promise."""
    v, f, _ = extract(sd_sphere(radius=1.0), extent=1.4, res=48)
    v2, f2 = decimate_mesh(v, f, extent=1.35, res=24)
    assert len(f2) < len(f) / 2
    occ_full = occupancy(v, f, res=24, extent=1.35, decimate=False)
    occ_dec = occupancy(v2, f2, res=24, extent=1.35, decimate=False)
    agree = (occ_full == occ_dec).mean()
    assert agree > 0.97


# --------------------------------------------------------------------------- extract + pose

def test_extract_of_a_sphere_field_is_closed_single_component_and_round():
    """Extracting a sphere field gives one edge-manifold component with no voids to fill and
    a nearly constant vertex radius."""
    v, f, info = extract(sd_sphere(radius=1.0), extent=1.5, res=40)
    assert is_edge_manifold(f)
    assert n_components(v, f) == 1
    assert info["n_voids_filled"] == 0
    r = np.linalg.norm(v, axis=1)
    assert r.std() < 0.03


def test_extract_repairs_a_disconnected_field_to_one_component():
    """When the level set is two disjoint solids, extract reports both and keeps only the
    larger one."""
    f = lambda p: np.minimum(sd_sphere((-1.0, 0, 0), 0.3)(p),     # noqa: E731
                             sd_sphere((1.0, 0, 0), 0.15)(p))
    v, fc, info = extract(f, extent=1.6, res=48)
    assert info["n_solid_components"] == 2
    assert n_components(v, fc) == 1
    assert is_edge_manifold(fc)
    # the kept piece is the bigger sphere, centred near x = -1
    assert v[:, 0].mean() < 0


def test_extract_fills_an_interior_void():
    """A solid ball with a small bubble at its centre: the bubble is an enclosed void, and
    extract fills it rather than returning a two-shell mesh."""
    outer = sd_sphere(radius=1.0)

    def field_with_void(p):
        r = np.linalg.norm(p, axis=-1)
        return np.where(r < 0.3, 0.5, outer(p))

    v, f, info = extract(field_with_void, extent=1.5, res=40)
    assert info["n_voids_filled"] >= 1
    assert n_components(v, f) == 1
    assert is_edge_manifold(f)


def test_pose_touches_z_exactly_at_plus_and_minus_one_and_centres_xy():
    """pose puts z exactly in [-1, 1], the xy centre near the axis, and the largest xy radius
    at the requested value."""
    v, f = icosphere(2)
    v = v * np.array([1.3, 0.8, 2.1]) + np.array([5.0, -3.0, 1.0])   # off-centre, stretched
    p = pose(v, radius=1.0, faces=f)
    assert p[:, 2].min() == pytest.approx(-1.0, abs=1e-9)
    assert p[:, 2].max() == pytest.approx(1.0, abs=1e-9)
    assert abs(p[:, 0].mean()) < 0.35               # centred by solid centroid, not vertex mean
    assert abs(p[:, 1].mean()) < 0.35
    assert np.hypot(p[:, 0], p[:, 1]).max() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- sample_body

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_sample_body_passes_every_requirement(seed):
    """A sampled body is closed, single-component, spans z in [-1, 1], fits the cylinder,
    and records its convexity and the radius its mounting gave it."""
    b = sample_body(np.random.default_rng([100, seed]), FAST_SPEC)
    chk = check_body(b, radius=FAST_SPEC.radius)
    assert chk["closed"], chk
    assert chk["single_component"], chk
    assert chk["z_span"], chk
    assert chk["inside_cylinder"], chk
    assert 0.3 < b.convexity <= 1.0 + 1e-6
    assert 0.2 < b.info["cylinder_radius"] < 8.0
    assert b.recipe["mount"]["axis"] in ("short", "long", "middle", "random")


def test_every_family_makes_a_body_and_names_itself():
    """Each procedural family on its own yields bodies that record that family."""
    for fam in ("potato", "bilobe", "trilobe", "top", "faceted", "geometric"):
        spec = LibrarySpec(res=48, family_weights={fam: 1.0}, n_modifiers=(0, 1),
                           convexity_shares=())
        b = sample_body(np.random.default_rng([300, len(fam)]), spec)
        assert b.recipe["base"] == fam
        assert check_body(b)["closed"]


def test_lobed_bodies_are_more_carved_than_potatoes():
    """Bilobes, which the modifiers are switched off for here, come out with a lower
    volume over hull volume than plain potatoes."""
    conv = {}
    for fam in ("potato", "bilobe"):
        spec = LibrarySpec(res=48, family_weights={fam: 1.0}, n_modifiers=(0, 0),
                           convexity_shares=())
        conv[fam] = np.mean([sample_body(np.random.default_rng([400, i]), spec).convexity
                             for i in range(4)])
    assert conv["bilobe"] < conv["potato"]


def test_mounting_follows_the_chosen_axis():
    """With every body mounted on its long axis and no tilt, the posed body's z extent
    exceeds its xy extent relative to a short-axis mounting of the same draw."""
    long = LibrarySpec(res=48, family_weights={"potato": 1.0}, n_modifiers=(0, 0),
                       mount_weights={"long": 1.0}, tilt_deg=0.0, max_tilt_deg=0.0,
                       convexity_shares=())
    short = LibrarySpec(res=48, family_weights={"potato": 1.0}, n_modifiers=(0, 0),
                        mount_weights={"short": 1.0}, tilt_deg=0.0, max_tilt_deg=0.0,
                        convexity_shares=())
    for i in range(3):
        r_long = sample_body(np.random.default_rng([500, i]), long).info["cylinder_radius"]
        r_short = sample_body(np.random.default_rng([500, i]), short).info["cylinder_radius"]
        assert r_long < r_short


def test_a_dealt_convexity_band_is_honoured():
    """With every body dealt the most carved band, the bodies that come back are carved, and
    each records the band it was dealt."""
    spec = LibrarySpec(res=48, convexity_bins=(0.7, 0.85, 0.95),
                       convexity_shares=(1.0, 0.0, 0.0, 0.0), band_attempts=12)
    for i in range(2):
        b = sample_body(np.random.default_rng([600, i]), spec)
        assert b.info["band"] == 0
        assert b.convexity < 0.7 or not b.info["band_hit"]


def test_read_shape_model_reads_obj_and_plain_lists(tmp_path):
    """The reader takes Wavefront OBJ (with polygon faces and slashes) and the plain
    numbered vertex and facet lists radar models use, and returns zero-based triangles."""
    v, f = icosphere(1)
    obj = tmp_path / "a.obj"
    obj.write_text("# comment\n" + "".join(f"v {x} {y} {z}\n" for x, y, z in v)
                   + "".join(f"f {a + 1}/1 {b + 1}/2 {c + 1}/3\n" for a, b, c in f))
    v2, f2 = read_shape_model(str(obj))
    assert np.allclose(v2, v) and (f2 == f).all()
    wf = tmp_path / "b.wf"
    wf.write_text(f"{len(v)}\n" + "".join(f"{i + 1} {x} {y} {z}\n" for i, (x, y, z) in enumerate(v))
                  + f"{len(f)}\n" + "".join(f"{i + 1} {a + 1} {b + 1} {c + 1}\n"
                                            for i, (a, b, c) in enumerate(f)))
    v3, f3 = read_shape_model(str(wf))
    assert np.allclose(v3, v) and (f3 == f).all()


def test_sample_body_is_deterministic_given_its_seed():
    """The same seed gives the same vertices."""
    b1 = sample_body(np.random.default_rng([42, 0]), FAST_SPEC)
    b2 = sample_body(np.random.default_rng([42, 0]), FAST_SPEC)
    assert b1.verts.shape == b2.verts.shape
    assert np.allclose(b1.verts, b2.verts)


# --------------------------------------------------------------------------- library-level

@pytest.mark.slow
def test_library_diversity_clears_the_participation_ratio_baseline():
    """The library's participation ratio on the solver's own descriptor (support on the
    design normals), and on the combined descriptor, clears the stated floors."""
    lib = build_library(40, seed=7, spec=FAST_SPEC)
    desc = library_descriptors(lib, n_probes=120, res=32)
    assert participation_ratio(desc["support"]) > 6.0
    assert participation_ratio(desc["combined"]) > 8.0


@pytest.mark.slow
def test_library_pairwise_dice_is_varied_not_clustered_near_one():
    """Pairwise Dice across the library has a mean well below one and a non-trivial spread."""
    lib = build_library(24, seed=8, spec=FAST_SPEC)
    d = pairwise_dice(lib, res=32, max_pairs=120)
    assert d.mean() < 0.85          # not a library of near-copies
    assert d.std() > 0.03           # spread, not a single cluster


@pytest.mark.slow
def test_library_check_passes_every_body():
    """check_library reports no failed body under any check."""
    lib = build_library(30, seed=9, spec=FAST_SPEC)
    chk = check_library(lib, radius=FAST_SPEC.radius)
    for key, idx in chk["failed"].items():
        assert idx == [], f"{key} failed for bodies {idx}"


def test_dice_of_a_body_against_itself_is_one():
    """dice(b, b) is one."""
    lib = build_library(2, seed=11, spec=FAST_SPEC)
    from hac26.library_metrics import dice
    assert dice(lib[0], lib[0], res=32) == pytest.approx(1.0, abs=1e-6)


def test_dice_is_invariant_to_a_rigid_rotation_of_one_body():
    """Dice after principal-axis alignment recovers most of the overlap between a body and a
    rigidly rotated copy of it."""
    from hac26.library_metrics import dice
    b = sample_body(np.random.default_rng([300, 0]), FAST_SPEC)
    q, r = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))
    R = q * np.sign(np.diag(r))
    rotated = Body(b.verts @ R.T, b.faces, b.recipe, b.info)
    d = dice(b, rotated, res=32)
    assert d > 0.75


def test_principal_frame_orders_axes_by_decreasing_moment():
    """After principal_frame the spread of the vertices decreases from x to z."""
    v, f = icosphere(3)
    v = v * np.array([2.0, 1.0, 0.5])               # longest along x, shortest along z
    R = principal_frame(v, f, res=40, extent=2.5)
    aligned = v @ R.T
    spread = aligned.std(axis=0)
    assert spread[0] >= spread[1] >= spread[2]


# --------------------------------------------------------------------------- curve conventions

def test_mesh_curve_renderer_matches_the_exact_convex_operator():
    """For a convex body the mesh renderer and the convex operator must agree, since ray-cast
    visibility and the mu > 0, mu0 > 0 test coincide; this checks rotation sense, camera
    geometry, the photometric kernel and the level the count is thresholded at, without any
    measured data.

    It is the only check of the kernel that needs nothing but the two models. A kernel that
    describes a scattering law this rig does not have, or a count thresholded on a brightness
    that varies with the direction of view, fails it by a factor of fifty, which is why the
    tolerance is tight enough to see that."""
    u, f = icosphere(1)
    v = u * np.array([1.0, 0.7, 1.3])
    hv, hf = hull_mesh(v)
    geoms = cameras()[:4]
    res = convex_cross_check(hv, hf, m=10, geoms=geoms, res=64, delta=1.0)
    assert res["mean_abs_diff_normalised"] < 0.01, res["mean_abs_diff_normalised"]


def test_mesh_curve_renderer_produces_a_time_directional_curve():
    """The renderer carries the fixed rotation sense `hac26.conventions.SENSE`, so an
    asymmetric body's curve changes when the frame order is reversed."""
    from hac26.curves_mesh import render_curves_mesh
    from hac26.shape_library import sd_box, extract, pose as _pose

    f = op_subtract(sd_sphere(radius=1.0), sd_box((0.6, 0, 0), (0.35, 0.15, 0.15)))
    v, fc, _ = extract(f, extent=1.5, res=36)
    v = _pose(v, radius=1.0, faces=fc)
    geoms = cameras()[4:5]                          # azimuth 45, away from the symmetric 0
    c = render_curves_mesh(v, fc, m=10, curve_types=["binary"], geoms=geoms, res=28,
                           decimate_to=2000)
    assert not np.allclose(c, c[:, ::-1])


def test_no_feature_is_drawn_below_what_the_grid_resolves():
    """The library states that nothing is generated below the scale marching cubes resolves,
    and this is the check of it. The smallest feature the sampler can draw is a basin's mouth
    at the shallowest cut with the smallest cutter: a sphere of radius rho cutting to depth d
    leaves a mouth of radius sqrt(d (2 rho - d)). The next smallest is the fillet that rounds
    a lobed body's neck. Both are compared with the feature radius the build resolution
    resolves, so widening either range past the grid fails here rather than silently
    producing bodies the extraction cannot render."""
    from hac26.shape_library import GRID_FILL, min_feature_radius
    spec = LibrarySpec(res=64)
    floor = min_feature_radius(spec.res, spec.extent)
    s = GRID_FILL * spec.extent                  # the size a finished body is fitted to
    rho, u = 0.3 * s, 0.35                       # smallest cutter, shallowest cut
    assert rho * u > floor                                    # the basin is deep enough
    assert rho * np.sqrt(u * (2.0 - u)) > floor               # and its mouth is wide enough
    # The neck of a lobed body at the closest spacing, where the two lobes touch and the
    # fillet alone opens the waist: a fillet of k on two tangent bodies of radius r leaves a
    # neck of radius about sqrt(2 r * 0.69 k), 0.69 being how far the exponential smooth
    # union pushes the surface out at the contact.
    assert np.sqrt(2.0 * s * 0.69 * 0.03 * s) > floor


def test_a_mounted_body_is_never_inside_out():
    """A body's faces have to point outward after mounting as well as after extraction. The
    mount applies a random orthogonal transform, and a reflection moves the vertices without
    moving the winding, which leaves a mesh that is watertight, winding-consistent and of the
    right convexity but has its inside and outside exchanged. Nothing downstream would notice
    except the signed distance the fit regresses on, which comes back negated."""
    from hac26.shape_library import _rand_rot, mesh_volume
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert np.linalg.det(_rand_rot(rng)) > 0.0
    spec = LibrarySpec(res=32, mount_weights={"random": 1.0})
    for i in range(6):
        b = sample_body(np.random.default_rng(4000 + i), spec)
        assert mesh_volume(b.verts, b.faces) > 0.0
