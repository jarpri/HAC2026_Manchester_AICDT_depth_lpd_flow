"""The corpus and the training draws: the correction from a convex start to the true hull,
the model-error curves, and the split of bodies."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hac26.field import (CODE_DIM, N_DIR, N_NODES, DepthSphere,               # noqa: E402
                         ImplicitBody)
from hac26.recon import dice, mesh_occupancy                                   # noqa: E402
from hac26.shapes import icosphere, mesh_support                              # noqa: E402
from hac26.solvers.gauss_newton import cap_depths                             # noqa: E402
from build_corpus import correction, correction_matrix, corpus_radius          # noqa: E402
from train_lpd import (Corpus, _quarter_turn_maps, held_out, inv_softplus,     # noqa: E402
                       quarter_turns, smooth_noise_like, support_from_mesh, support_with)


def _support(scale):
    v, f = icosphere(2)
    v = v * np.asarray(scale)
    n = ImplicitBody().core.n.numpy()
    return torch.tensor(mesh_support(v, n), dtype=torch.float32), v, f


def test_inv_softplus_inverts_softplus():
    h = torch.linspace(0.01, 3.0, 50)
    assert torch.allclose(torch.nn.functional.softplus(inv_softplus(h)), h, atol=1e-6)


def test_the_correction_takes_the_start_to_the_true_hull():
    """The dh stored in the corpus, applied to the start the way the flow applies it, gives
    back the true hull up to what a band-limited correction cannot express."""
    h_true, _, _ = _support([1.0, 0.7, 1.2])
    h_start, _, _ = _support([1.0, 0.9, 1.0])
    dh, left = correction(h_true, h_start, correction_matrix())
    assert dh.shape == (N_DIR,)
    before = float((h_start - h_true).abs().mean())
    after = float((support_with(h_start[None], dh[None])[0] - h_true).abs().mean())
    assert after < 0.1 * before
    assert left < 0.1 * float((inv_softplus(h_true) - inv_softplus(h_start)).pow(2).mean().sqrt())


def test_support_from_mesh_is_in_the_canonical_frame():
    """A mesh in the physical frame (xy scaled to a radius) gives the same support as the
    same body at xy radius one."""
    _, v, f = _support([1.0, 0.7, 1.2])
    h1 = support_from_mesh(v, f)
    h2 = support_from_mesh(v * np.array([1.4, 1.4, 1.0]), f)
    assert torch.allclose(h1, h2, atol=1e-5)
    assert (h1 > 0).all()


def test_smooth_noise_has_unit_rms_zero_mean_and_the_curve_s_spectrum():
    P = 64
    x = torch.linspace(0, 2 * np.pi, P + 1)[:-1]
    curves = 1.0 + 0.3 * torch.cos(2 * x) + 0.1 * torch.sin(5 * x)
    curves = curves[None, None].expand(3, 2, P).clone()
    z = smooth_noise_like(curves, generator=torch.Generator().manual_seed(0))
    assert z.shape == curves.shape
    assert torch.allclose(z.mean(-1), torch.zeros(3, 2), atol=1e-5)
    assert torch.allclose(z.pow(2).mean(-1).sqrt(), torch.ones(3, 2), atol=1e-4)
    spec = torch.fft.rfft(z, dim=-1).abs()
    assert (spec[..., [2, 5]] > 1e-3).all()                 # the curve's orders are there
    keep = torch.ones(P // 2 + 1, dtype=torch.bool); keep[[0, 2, 5]] = False
    assert float(spec[..., keep].abs().max()) < 1e-4       # and no others


def test_held_out_is_fixed_and_radii_are_in_range():
    a, b = held_out(40, 8), held_out(40, 8)
    assert a.tolist() == b.tolist() and len(set(a.tolist())) == 8
    assert set(held_out(40, 4).tolist()) <= set(a.tolist())   # a prefix of one permutation
    from hac26.conventions import CYLINDER_R
    r = [corpus_radius(i, float("nan")) for i in range(50)]
    assert 0.9 * min(CYLINDER_R.values()) <= min(r) and max(r) <= 1.1 * max(CYLINDER_R.values())
    assert corpus_radius(3, float("nan")) == corpus_radius(3, float("nan"))
    assert corpus_radius(3, 1.7) == 1.7                       # a recorded radius is used


def _small_operator(P):
    from hac26.conventions import psi_grid
    from hac26.forward.mesh.exact import RenderConfig
    from hac26.forward.mesh.instrument import Instrument
    from hac26.solvers.operator import CodeOperator
    cfg = RenderConfig(height=24, width=40, supersample=1, sun_res=64, phase_chunk=4,
                       radiosity_faces=48)
    return CodeOperator(Instrument(quantise=False), psi_grid(P), res=16, config=cfg,
                        device="cpu", backend="software")


def test_quarter_turns_permute_the_nodes_and_come_back_after_four():
    """A quarter turn about the spin axis is an exact symmetry of the problem, and it is what
    gives the corpus four training pairs per body at no cost in renders. The node set is built
    to be invariant under it, so a turned body's depths are the same numbers in a different
    order: the map has to be a permutation, and an exact one, or the corpus is taught codes no
    fit of those bodies would produce."""
    from hac26.conventions import R_z
    from hac26.field import node_design

    perms, idxs, ws, es = _quarter_turn_maps("cpu")
    one = perms[0]
    assert sorted(one.tolist()) == list(range(len(one)))            # a permutation
    twice = one[one]
    assert torch.equal(twice, perms[1]) and torch.equal(twice[twice], torch.arange(len(one)))
    assert torch.equal(one[perms[1]], perms[2])
    # and it is the turn it claims to be: a turned body has in direction u the value the
    # body has in direction R^T u, so the permuted nodes are the nodes turned back
    u = node_design(N_NODES)
    for q in (1, 2, 3):
        assert np.abs(u[perms[q - 1].numpy()] - u @ R_z(q * np.pi / 2)).max() < 1e-12, q
    # every direction reads weights that sum to one, so a constant support stays constant
    assert torch.allclose(ws.sum(-1), torch.ones_like(ws.sum(-1)), atol=1e-4)
    # dh's turn is exact on the band: four quarter turns give back every band-limited dh,
    # and a band-limited dh is what the flow emits (its expansion drops the rest)
    from hac26.field import dir_design, sh_expand
    band = torch.from_numpy(sh_expand(dir_design(N_DIR), dir_design(N_DIR)))   # the projector
    e = es[0]
    assert torch.allclose(e @ e @ e @ e, band, atol=1e-3)


def test_a_quarter_turn_is_the_pair_the_operator_would_render():
    """Turning a body by quarter turns about its spin axis: the curves quarter_turns makes are
    exactly the curves of the turned mesh, count curves included, in the direction the
    conventions fix; and the turned code describes the turned mesh."""
    from hac26.conventions import PSI0, R_z
    from hac26.forward.mesh.exact import normalise
    P = 8
    op = _small_operator(P)
    h, _, _ = _support([1.0, 0.7, 1.2])                       # not symmetric under a quarter turn
    code = torch.zeros(1, CODE_DIM)
    # one dent, off the spin axis and off the plane that a quarter turn would map to itself,
    # written as a cap of a stated depth so that it is the same dent at any node count
    code[0, N_DIR:] = torch.tensor(cap_depths(DepthSphere(N_NODES).u.numpy(),
                                              [0.80, 0.45, 0.40], 30.0, 0.25),
                                   dtype=torch.float32)
    geoms = [0, 2, 5]
    curves, turned = op.curves_turned(h, code[0], 1.2, geoms=geoms)
    corpus = Corpus(code, curves[None], turned[None], h[None], h[None], torch.tensor([1.2]),
                    torch.tensor([0]))
    v, f = op.mesh(h, code[0])
    for q in (1, 2, 3):
        R = torch.tensor(R_z(q * np.pi / 2), dtype=torch.float32)
        of_turned_mesh = normalise(op.forward.raw_curves(op.physical(v @ R.T, f, 1.2), f,
                                                         geoms=geoms, psi0=PSI0))
        codes_t, curves_t, sup_t, _ = quarter_turns(corpus, torch.tensor([0]), torch.tensor([q]))
        assert torch.allclose(curves_t[0], of_turned_mesh, atol=1e-5), q
        assert not torch.allclose(curves_t[0], curves, atol=1e-2)          # a real shift
        # The turned code has to describe the turned body, and that is a statement about the
        # body and not about where the extraction happens to put a vertex: a quarter turn maps
        # the extraction grid to itself but not its cubes to themselves, so two extractions of
        # the same surface differ by a fraction of a cell wherever they place their vertices.
        # Measured, that fraction is a third of a cell here and the overlap is unaffected.
        v_t, f_t = op.mesh(sup_t[0], codes_t[0])
        d = dice(mesh_occupancy(v_t.numpy(), f_t.numpy(), 96, 1.4),
                 mesh_occupancy((v @ R.T).numpy(), f.numpy(), 96, 1.4))
        assert d > 0.99, (q, d)
