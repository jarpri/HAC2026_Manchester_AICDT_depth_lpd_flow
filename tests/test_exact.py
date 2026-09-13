"""The exact forward model on the software rasteriser: coverage, shadows, agreement with the
convex operator on a convex body, and the adjoint plumbing."""
import numpy as np
import pytest
import torch

from hac26.conventions import SENSE, TRANSFER_EXPONENT, cameras, psi_grid
from hac26.forward.mesh.exact import (ExactForward, LitCoverage, RenderConfig, normalise,
                                      normalise_vjp)
from hac26.forward.mesh.instrument import Instrument
from hac26.forward.mesh.raster import Rasteriser, flat_faces
from hac26.geometry import build_cameras
from hac26.shapes import hull_mesh, icosphere, mesh_curves_convex

# few patches, so the interreflection runs on a decimated copy of the mesh as it does on a
# real body
SMALL = RenderConfig(height=24, width=40, supersample=1, sun_res=96, phase_chunk=3,
                     radiosity_faces=48)
# fine enough that the missing antialiasing of the software backend is a small effect
MEDIUM = RenderConfig(height=48, width=80, supersample=2, sun_res=96, phase_chunk=4,
                      radiosity_faces=48)


def _two_spheres():
    import trimesh
    a = trimesh.creation.icosphere(subdivisions=1, radius=0.55)
    b = trimesh.creation.icosphere(subdivisions=1, radius=0.55)
    a.apply_translation([-0.375, 0, 0]); b.apply_translation([0.375, 0, 0])
    m = trimesh.util.concatenate([a, b])
    return np.asarray(m.vertices), np.asarray(m.faces)


def _coverage(v, f, direction, res=400):
    vt = torch.tensor(v, dtype=torch.float32); ft = torch.tensor(f)
    fv, ff = flat_faces(vt, ft)
    ras = Rasteriser(res, res, 1, device="cpu", backend="software")
    d = torch.tensor([direction], dtype=torch.float32); d = d / d.norm()
    ext = float(vt.norm(dim=1).max()) * 1.05
    cov = LitCoverage.apply(fv, ff, d, ext, ext, ras)[0].numpy()
    tv = v[f]
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    cos = (n / (2 * area[:, None])) @ (d[0].numpy())
    return cov, area * np.clip(cos, 0, None)


def test_coverage_of_a_convex_body_is_its_projected_area():
    """A convex body casts no shadow on itself, so the lit projected area of every face is
    A cos(theta)+, and their sum is the area of the silhouette."""
    v, f = icosphere(2)
    cov, proj = _coverage(v, f, [0.6, 0.5, 0.3])
    assert np.abs(cov - proj).max() < 0.02 * proj.max()
    assert cov.sum() == pytest.approx(proj.sum(), rel=0.005)


def test_coverage_never_exceeds_the_projected_area_and_shadows_reduce_it():
    """On two overlapping spheres the lit area of each face is at most A cos(theta)+, and the
    faces in the cast shadow of the other sphere get much less."""
    v, f = _two_spheres()
    cov, proj = _coverage(v, f, [1.0, 0.1, 0.0])
    assert (cov <= proj + 1e-3).all()
    shadowed = proj - cov > 0.5 * proj
    assert shadowed.sum() >= 5
    assert cov.sum() < 0.98 * proj.sum()


def test_exact_intensity_matches_the_convex_operator_on_a_convex_body():
    """A convex body has no interreflection, since no two of its faces see each other, so the
    chain that renders an image and the one that sums over facets are computing the same
    integral and must agree. That is the end-to-end check of the rotation sense, the camera
    geometry, the projection and the photometric kernel, and it needs no measured data.

    Both sides are the rendered channel's instrument, whose transfer is the measured power
    law; comparing an image-based model against a facet sum at a different exponent would be
    comparing two different measurements."""
    u, f = icosphere(1)
    v = u * np.array([1.0, 0.7, 1.3])
    hv, hf = hull_mesh(v)
    P, geoms = 8, [0, 2, 9, 15]
    inst = Instrument.blender_start(tau_i=1e-3)
    op = ExactForward(inst, psi_grid(P), MEDIUM, device="cpu", backend="software")
    raw = op.raw_curves(torch.tensor(hv, dtype=torch.float32), torch.tensor(hf), geoms=geoms)
    mine = normalise(raw)[:, 0].numpy()                                   # intensity only
    ref_cams = [build_cameras()[g] for g in geoms]
    ref = mesh_curves_convex(hv, hf, ref_cams, P, ["intensity"] * len(geoms),
                             gamma=TRANSFER_EXPONENT, sigma=SENSE)
    ref = ref / ref.mean(1, keepdims=True)
    assert np.abs(mine - ref).mean() < 0.02, np.abs(mine - ref).mean()


def test_vjp_returns_the_same_curves_and_finite_gradients():
    """The vector-Jacobian product reproduces raw_curves exactly and gives finite gradients for
    the vertices and the instrument parameters; more albedo means more light, so the
    derivative of the total intensity with respect to rho is positive on a body that reflects
    onto itself."""
    v, f = _two_spheres()
    vt = torch.tensor(v, dtype=torch.float32).requires_grad_(True)
    ft = torch.tensor(f)
    inst = Instrument(quantise=False)
    op = ExactForward(inst, psi_grid(6), SMALL, device="cpu", backend="software")
    raw = op.raw_curves(vt.detach(), ft, geoms=[0, 5])
    cot = torch.zeros_like(raw); cot[:, 0] = 1.0                          # d(sum of I)
    raw2, gv, (g_rho, g_tau, g_ped) = op.vjp(vt, ft, cot, geoms=[0, 5],
                                            params=[inst.raw_rho, inst.raw_tau_i, inst.raw_pedestal])
    assert torch.allclose(raw, raw2)
    assert gv.shape == vt.shape and torch.isfinite(gv).all()
    assert float(g_rho) > 0.0
    assert torch.isfinite(g_tau) and torch.isfinite(g_ped).all()


def test_normalise_vjp_matches_autograd():
    """The hand-written adjoint of the per-curve mean normalisation equals autograd's."""
    raw = (torch.rand(3, 2, 7, dtype=torch.float64) + 0.5).requires_grad_(True)
    cot = torch.randn(3, 2, 7, dtype=torch.float64)
    (normalise(raw) * cot).sum().backward()
    assert torch.allclose(raw.grad, normalise_vjp(raw.detach(), cot))


def test_geometry_subset_matches_the_full_set():
    """Curves of a subset of the geometries equal the same rows of the full set."""
    v, f = _two_spheres()
    vt, ft = torch.tensor(v, dtype=torch.float32), torch.tensor(f)
    op = ExactForward(Instrument(quantise=False), psi_grid(4), SMALL, device="cpu",
                      backend="software")
    full = op.raw_curves(vt, ft)
    part = op.raw_curves(vt, ft, geoms=[3, 20])
    assert full.shape == (len(cameras()), 2, 4)
    assert torch.allclose(full[[3, 20]], part)


def test_a_distant_camera_still_sees_the_body():
    """The clip planes follow the camera distance, so a camera far enough away to be nearly
    orthographic still renders the body rather than clipping it."""
    v, f = _two_spheres()
    vt, ft = torch.tensor(v, dtype=torch.float32), torch.tensor(f)
    op = ExactForward(Instrument(eye_distance=400.0, quantise=False), psi_grid(3), SMALL,
                      device="cpu", backend="software")
    raw = op.raw_curves(vt, ft, geoms=[0, 9])
    assert torch.isfinite(raw).all() and (raw > 0).all()


def test_without_interreflection_the_radiance_is_the_direct_term_alone():
    """With the bounce light switched off the radiance of every face is rho E / pi from its
    own lit coverage, and no patches are built. With it on, the faces of two overlapping
    spheres that face each other across the waist receive more when the light comes in
    along the waist, so the direct term is a lower bound that is strict inside the
    concavity."""
    v, f = _two_spheres()
    vt, ft = torch.tensor(v, dtype=torch.float32), torch.tensor(f)
    on = Instrument(rho=0.9, quantise=False)
    off = Instrument(rho=0.9, quantise=False, interreflection=False)
    op_on = ExactForward(on, psi_grid(4), SMALL, device="cpu", backend="software")
    op_off = ExactForward(off, psi_grid(4), SMALL, device="cpu", backend="software")
    quarter = op_off.psi[1:2]                  # the light along the waist of the two spheres
    with torch.no_grad():
        prep_off = op_off._prepare(vt, ft)
        assert prep_off.solver is None and prep_off.patch is None
        _, L_off = op_off._radiance_chunk(prep_off, quarter, 0.0)
        prep_on = op_on._prepare(vt, ft)
        _, L_on = op_on._radiance_chunk(prep_on, quarter, 0.0)
        from hac26.forward.mesh.exact import source_dirs, rotate_z
        dirs = rotate_z(source_dirs(off.delta, SMALL.n_source), -quarter).reshape(-1, 3)
        cov = LitCoverage.apply(prep_off.fv, prep_off.ff, dirs, prep_off.extent,
                                prep_off.extent, op_off.ras_sun)
        e = cov.reshape(1, SMALL.n_source, -1).mean(1) / prep_off.area[None]
    assert torch.allclose(L_off, 0.9 * e / np.pi, rtol=1e-4, atol=1e-6)
    assert (L_on >= L_off - 1e-6).all()
    assert float((L_on - L_off).max()) > 0.1 * float(L_off.max())


def test_the_blender_start_is_a_camera_at_infinity_without_bounce_light(tmp_path):
    """The rendered channel's instrument has the cameras at infinity, the bounce light off
    and the measured power-law transfer, fits nothing at all, and comes back from a saved file
    with both switches and the same sensor chain.

    Nothing is fitted because a render has none of the things a calibration would move: no
    lens whose falloff could be fitted, no photosite whose point spread could, no penumbra and
    no spline transfer. Each of those is a direction a fit would otherwise use to absorb an
    error of shape, and the penumbra is the worst of them, being a blur of the shadow edge,
    which is the feature that carries concavity. The laboratory channel is a different
    instrument and is fitted."""
    from hac26.forward.mesh.sensor import PowerTransfer, SensorModel

    inst = Instrument.blender_start()
    assert not bool(inst.interreflection)
    assert bool(inst.orthographic)
    assert isinstance(inst.sensor, PowerTransfer)
    assert float(inst.sensor.gamma.detach()) == pytest.approx(TRANSFER_EXPONENT, abs=1e-4)
    assert inst.fitted_parameters() == []
    # a parallel beam, not a disc: the parameter is stored unsquashed and floored there, so
    # the source radius is zero to any precision that matters and not exactly zero
    assert float(inst.delta.detach()) < 1e-9
    lab = Instrument()
    assert not bool(lab.orthographic) and isinstance(lab.sensor, SensorModel)
    assert {"raw_rho", "raw_eye", "sensor.raw_vignette"} <= {n for n, _ in
                                                             lab.fitted_parameters()}
    inst.save(tmp_path / "blender.pt")
    back = Instrument.load(tmp_path / "blender.pt")
    assert not bool(back.interreflection) and bool(back.orthographic)
    assert isinstance(back.sensor, PowerTransfer)
    assert float(back.sensor.gamma) == pytest.approx(float(inst.sensor.gamma))
    lab.save(tmp_path / "lab.pt")
    back_lab = Instrument.load(tmp_path / "lab.pt")
    assert isinstance(back_lab.sensor, SensorModel) and not bool(back_lab.orthographic)


def _ortho_and_perspective(dist_list, geoms=(0,), phases=3):
    """Normalised curves of one convex body under the orthographic camera and under
    perspective cameras at the given distances, on the same small configuration."""
    import trimesh
    ico = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    v = np.asarray(ico.vertices, float) * np.array([1.0, 0.7, 0.55])
    f = np.asarray(ico.faces)
    V = torch.tensor(v, dtype=torch.float32)
    F = torch.tensor(f, dtype=torch.long)
    psi = psi_grid(phases)
    out = []
    for inst in [Instrument.blender_start()] + [Instrument.blender_start()
                                                for _ in dist_list]:
        out.append(inst)
    for inst, dist in zip(out[1:], dist_list):
        with torch.no_grad():
            inst.orthographic.fill_(False)
            inst.raw_eye.fill_(float(np.log(np.expm1(dist))))
    return [normalise(ExactForward(i, psi, SMALL, device="cpu", backend="software")
                      .raw_curves(V, F, geoms=list(geoms))).numpy() for i in out]


def test_the_orthographic_camera_is_the_limit_of_a_receding_perspective_one():
    """A camera at infinity is its own projection, not a very distant perspective one, and
    the two agree only in the limit. The intensity curves of a perspective camera must
    approach the orthographic ones as the camera recedes, since the difference between the
    projections is of the order of the body's size over the camera distance. The count
    curves are not checked here: a count is an integer, and on a small image one pixel is
    already a per cent of it."""
    ortho, near, far = _ortho_and_perspective([6.0, 100.0])

    def rms(a, b):
        return float(np.sqrt(((a[:, 0] - b[:, 0]) ** 2).mean()))

    assert rms(ortho, far) < 0.3 * rms(ortho, near)
    assert rms(ortho, far) < 0.02


def test_the_orthographic_projection_frames_the_body_as_the_perspective_one_does():
    """Both projections put the body's own extent at the same fraction of the frame, so the
    rendered intensity of a body is the same size under either, and the batched matrix is
    the single-direction one it is built from."""
    from hac26.forward.mesh.raster import orthographic, orthographic_view

    d = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.3, -0.8, 0.2]])
    batch = orthographic_view(d, 1.7, 5.1, aspect=1.0)
    for i in range(len(d)):
        assert torch.allclose(batch[i], orthographic(d[i], 1.7, 5.1), atol=1e-6)
    wide = orthographic_view(d, 1.7, 5.1, aspect=2.0)
    assert torch.allclose(2.0 * wide[:, 0], batch[:, 0], atol=1e-6)
    assert torch.allclose(wide[:, 1:], batch[:, 1:], atol=1e-6)
    # a point at the top of the frame lands on the clip-space edge
    top = torch.tensor([[0.0, 1.7, 0.0, 1.0]])
    y = torch.einsum("vj,bij->bvi", top, batch)[0, 0, 1]
    assert float(y) == pytest.approx(1.0, abs=1e-5)


def test_a_cpu_run_gets_the_only_rasteriser_that_can_serve_it():
    """nvdiffrast has no CPU path, so on a CPU device there is one backend and not a choice.
    Defaulting to nvdiffrast there turned every --device cpu run into a missing-module
    traceback from inside the forward model, which reads as a broken install rather than as
    the one thing it is. An explicit name still wins, and a CUDA device still gets the fast
    one asked for."""
    from hac26.forward.mesh.raster import _backend
    from hac26.forward.shared import software_raster

    assert _backend(None, "cpu") is software_raster
    assert _backend("software", "cuda") is software_raster
    with pytest.raises(ValueError):
        _backend("something else", "cpu")
def test_a_cached_mesh_gives_the_same_curves_and_gradients():
    """mesh_constants made once and passed back as `mesh` gives the curves and instrument
    gradients of a call that makes its own, which is what lets the calibration make them once
    per body; a cached mesh cannot carry a gradient to the vertices, so asking is refused."""
    v, f = _two_spheres()
    vt = torch.tensor(v, dtype=torch.float32)
    ft = torch.tensor(f)
    inst = Instrument(quantise=False)
    op = ExactForward(inst, psi_grid(6), SMALL, device="cpu", backend="software")
    mesh = op.mesh_constants(vt, ft)
    params = [inst.raw_rho, inst.raw_tau_i, inst.raw_pedestal, inst.sensor.raw_oetf]
    cot = lambda raw: torch.ones_like(raw)                                    # noqa: E731
    raw, _, grads = op.vjp(vt, ft, cot, geoms=[0, 5], params=params)
    raw_c, _, grads_c = op.vjp(vt, ft, cot, geoms=[0, 5], params=params, mesh=mesh)
    assert torch.allclose(raw, raw_c)
    for g, g_c in zip(grads, grads_c):
        assert torch.allclose(g, g_c)
    assert torch.allclose(op.raw_curves(vt, ft, geoms=[0, 5]),
                          op.raw_curves(vt, ft, geoms=[0, 5], mesh=mesh))
    with pytest.raises(ValueError):
        op.vjp(vt.clone().requires_grad_(True), ft, cot, geoms=[0, 5], mesh=mesh)


