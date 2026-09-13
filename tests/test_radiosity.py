"""The radiosity solve on a body that reflects onto itself."""
import numpy as np
import pytest
import torch

from hac26.forward.mesh.radiosity import RadiosityError, RadiositySolver, form_factors


def contact_binary(sep: float = 0.75, r: float = 0.55):
    """Two overlapping spheres: a non-convex body whose lobes see each other."""
    import trimesh
    a = trimesh.creation.icosphere(subdivisions=1, radius=r)
    b = trimesh.creation.icosphere(subdivisions=1, radius=r)
    a.apply_translation([-sep / 2, 0, 0])
    b.apply_translation([+sep / 2, 0, 0])
    m = trimesh.util.concatenate([a, b])
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def test_form_factors_are_reciprocal_and_bounded():
    """A_i F_ij = A_j F_ji, every entry is non-negative, no row sums past one, and faces that
    cannot see each other have a zero entry (the two lobes' outer faces)."""
    v, f = contact_binary()
    F, a, n, c = form_factors(v, f)
    G = torch.as_tensor(a)[:, None] * F
    assert torch.allclose(G, G.T, atol=1e-12)
    assert float(F.min()) >= 0.0
    assert float(F.sum(1).max()) <= 1.0 + 1e-6
    outer = np.nonzero((c[:, 0] < -0.8) | (c[:, 0] > 0.8))[0]
    assert float(F[outer][:, outer].sum()) == 0.0


def test_transport_preserves_the_ordering():
    """More irradiance on every patch gives more radiosity on every patch, since
    (I - rho F)^-1 = I + rho F + rho^2 F^2 + ... has non-negative entries."""
    v, f = contact_binary()
    F, a, n, c = form_factors(v, f)
    solver = RadiositySolver(F, rho=0.85)
    e_more = torch.rand(len(F), dtype=F.dtype) + 0.5
    e_less = e_more * torch.rand(len(F), dtype=F.dtype)
    assert (solver.solve(e_more) >= solver.solve(e_less) - 1e-9).all()


def test_radiosity_is_differentiable_in_rho():
    """With rho a tensor that requires grad, the solution carries a gradient, and more albedo
    means more radiosity everywhere."""
    v, f = contact_binary()
    F, *_ = form_factors(v, f)
    rho = torch.tensor(0.7, dtype=F.dtype, requires_grad=True)
    B = RadiositySolver(F, rho).solve(torch.ones(len(F), dtype=F.dtype))
    (g,) = torch.autograd.grad(B.sum(), rho)
    assert float(g) > 0.0


def test_a_bad_row_raises_or_is_scaled():
    """A row summing past one is refused by default and rescaled on request."""
    v, f = contact_binary()
    F, *_ = form_factors(v, f)
    i = int(F.sum(1).argmax())
    F[i] *= 1.5 / float(F[i].sum())
    with pytest.raises(RadiosityError):
        RadiositySolver(F, rho=0.5)
    s = RadiositySolver(F, rho=0.5, bad_rows="scale")
    assert s.n_bad_rows == 1 and s.max_row_sum <= 1.0 + 1e-6
