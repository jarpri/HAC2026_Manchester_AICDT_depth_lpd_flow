#!/usr/bin/env python3
"""Reconstruct a challenge model with the trained flow.

The starting support h is the convex stage's reconstruction of the model; the flow supplies
the correction dh and the amplitudes g. The flow is run from several independent draws of
x0, each for --steps steps with the exact operator and its adjoint at every step and noise
added on the way (--churn; hac26.solvers.lpd_flow). The learned steps leave a draw close to
a body that fits the curves but not on it, so each draw is then polished: gradient steps on
the whitened misfit through the exact operator, stopped once the misfit is at the noise level
(polish). Along directions the curves do not constrain the gradient is zero, so the polish
moves nothing the data cannot see. The misfit of every draw before and after is reported.

The curves leave several bodies possible, and the answer has to be one shape scored by voxel
overlap and by the side-view boundary distance. The draws stand in for the bodies that fit,
and every candidate is scored by its mean over the draws under both measures. The candidates
are the draws and the consensus bodies: the level sets of the fraction of draws that contain
each voxel. When the draws are the posterior, a level set is the body with the best expected
voxel score, and it keeps a dent wherever enough draws agree on it; when the draws disagree
on where the dents are, a level set blurs them and a single draw scores better. Which is the
case here is not known in advance, so both kinds stand as candidates and the choice between
them is made by measuring. One of the levels is derived from the draws rather than fixed
(dice_optimal_level). Nothing is averaged in code space.

--hold-out-geoms K keeps K of the measured geometries away from the inversion and reports
the answer's misfit on them beside its misfit on the ones it saw. An answer that fits the
seen cameras and not the held-out ones has fitted curves rather than recovered a shape.

The residual is divided per curve by sqrt(sigma_c^2 + eta_c^2), the measurement noise from
this model's co-located camera pairs and the model error the calibration fitted for that
curve, as in training (train_lpd.flow_loss).

The inversion runs in the canonical frame (z in [-1, 1], xy radius 1), the frame the corpus
was fitted and the flow trained in; the published radius is restored afterwards with
fit_to_cylinder.

Planar snapping is opt-in. It projects near-coplanar vertices onto fitted planes, which
helps flat-faced targets and can turn a smooth surface into terraces; a plane is kept only if
the whitened misfit does not rise by more than SNAP_MAX_RISE.

For a public model the Dice against the truth is reported, both meshes posed by
rescale_touch_z first.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hac26.conventions import CYLINDER_R, PUBLIC_MODELS, psi_grid   # noqa: E402
from hac26.data_io import N_CAMS, load_model_curves, public_stl        # noqa: E402
from hac26.field import CODE_DIM, DESIGN_N, N_DIR, N_NODES   # noqa: E402
from hac26.forward.mesh.exact import normalise                         # noqa: E402
from hac26.forward.mesh.radiosity import RadiosityError                # noqa: E402
from hac26.data_io import native_sigma                                 # noqa: E402
from hac26.solvers.lpd_flow import (CHURN, GUIDANCE, N_MODES, N_STEPS,   # noqa: E402
                                    LPDFlow, flow_inputs, geometry_tags)
from hac26.solvers.operator import CodeOperator                        # noqa: E402
from hac26.solvers.output import (export_stl, metric_medoid, planar_snap,      # noqa: E402
                                  ransac_planes, restore_constraints)
from hac26.recon import dice, fit_to_cylinder, mesh_occupancy          # noqa: E402
from hac26.shapes import rescale_touch_z                              # noqa: E402
from train_lpd import (CALIBRATION, _enable_tf32, add_render_flags,   # noqa: E402
                       check_flow_metadata, cond_channels, load_flow_file, load_instrument,
                       render_from, render_tag, residual_features, support_from_mesh)

SPREAD_MAX = 0.95    # mean Dice of the other draws against the medoid above which the draws
                     # are reported as one body rather than a spread of answers
SNAP_MAX_RISE = 0.05 # a snapped plane is kept if the whitened RMS misfit rises by at most
                     # this many standard deviations
# Levels of the draw fraction that make a consensus body, beside the one dice_optimal_level
# derives from the draws themselves. These two bracket the derived level from below and above
# for any achievable score, so the three together cover the range without a fourth: the
# derivation puts the optimum at half the achievable score, which no reachable score puts
# above a half, and a level of 0.65 measured consistently worse than either of them.
CONSENSUS_LEVELS = (0.35, 0.5)
POLISH_TARGET = 1.0  # the polish stops once the whitened RMS misfit is at the noise level:
                     # below it, it would be fitting noise
POLISH_STEP = 0.1    # first step of the polish, in whitened units per coordinate (RMS)
# Side of the grid the candidates are compared on and the consensus bodies are built from.
# It matters more than a discretisation usually does, because only the consensus bodies pay
# for it twice: a draw is a mesh voxelised once, while a consensus body is built out of the
# voxels and then meshed and voxelised again. Measured on ensembles of eight draws, that
# round trip costs the consensus body 0.08 of voxel overlap at a side of 64 and 0.03 at 128,
# against a real difference between candidates of about 0.02, so at 64 the rule was choosing
# draws over consensus bodies for a reason that had nothing to do with the bodies. Higher is
# not better without more draws: the fraction of draws occupying a voxel takes only as many
# values as there are draws, so a finer grid past this makes the level set jagged rather than
# sharper, and 192 measured worse than 128. It is also no more expensive, since a mesh this
# fine is no longer decimated on the way in.
#
# Both measurements above were taken on ensembles of eight draws, and the draw count has since
# gone up. The tie that held the grid here was the quantisation, and at the present count it no
# longer binds at this side, so a finer grid may now be worth what it costs. Nothing here
# assumes one side over another; what is missing is the same measurement taken again.
OCC_RES = 128


def curve_pairs(curves56: np.ndarray) -> torch.Tensor:
    """(2 * N_CAMS, P) curves in the released order to (N_CAMS, 2, P): per geometry, the
    intensity then the binary curve."""
    return torch.tensor(np.stack([curves56[:N_CAMS], curves56[N_CAMS:]], axis=1),
                        dtype=torch.float32)


def geometry_mask(mask56: np.ndarray) -> torch.Tensor:
    """(1, N_CAMS): a geometry counts as present only if BOTH its curves are.

    AND, and not OR, because of what this mask is for. It is the flow's conditioning channel:
    one flag per geometry, handed to `flow_inputs` beside features that `residual_features`
    computes over both curve columns of every marked geometry, and used to choose the
    geometries the adjoint's cotangent is formed on. Neither path carries `curve_weight`, so
    marking a geometry whose count curve Otsu's threshold collapsed does not recover its
    intensity curve; it shows the network the collapsed curve as data, in the features and in
    the gradient, and a zero residual there reads as a perfect fit. Dropping the geometry
    costs one curve. Marking it corrupts the other.

    The intensity curve of such a geometry is recovered for the solvers instead, by
    `measured_geometries` below: they select curves one at a time through `curve_weight`
    wherever a residual is formed, so for them a geometry with one good curve is a geometry
    with a measurement in it. That is where model 2's eighteen refused count curves stop
    costing anything.
    """
    return torch.tensor((mask56[:N_CAMS] > 0) & (mask56[N_CAMS:] > 0),
                        dtype=torch.float32)[None]


def measured_geometries(mask56: np.ndarray) -> list:
    """The geometries a solver can fit, as indices: those with at least one released curve
    that carries shape.

    This is not `geometry_mask`, and the difference is not an oversight. That mask is a
    conditioning channel for the flow, which has one flag per geometry and no way to say that
    one of a geometry's two curves is missing: a geometry marked present with an absent curve
    would show the network a zero residual there and read as a perfect fit, so it requires
    both. A solver carries `curve_weight` instead, which selects curves one at a time
    wherever the residual is formed, so for it a geometry with one good curve is a geometry
    with a measurement in it. Requiring both would throw that curve away -- which is what it
    would do to the twenty geometries of the sawed-off cube whose count curves Otsu's
    threshold collapses (data_io.count_curve_is_usable), and their intensity curves carry
    shape.
    """
    w = np.asarray(mask56) > 0
    return [i for i in range(N_CAMS) if bool(w[i] or w[i + N_CAMS])]


def curve_weight(mask56: np.ndarray) -> torch.Tensor:
    """(N_CAMS, 2): one for every released curve that carries shape, zero for the rest, laid
    out as the residual is."""
    return torch.tensor(np.asarray(mask56) > 0, dtype=torch.float32).reshape(2, N_CAMS).T


def residual_scale(d: dict, eta56: torch.Tensor) -> torch.Tensor:
    """sqrt(sigma_c^2 + eta_c^2) per curve as (N_CAMS, 2): the measured noise of this model,
    estimated from the high-frequency content of its curves at their native frame rate
    (hac26.noise), and the calibration's per-curve model error. `d` is a
    `data_io.load_model_curves` result, which carries the native-resolution curves the noise
    estimate needs."""
    sigma = torch.tensor(native_sigma(d), dtype=torch.float32)
    return torch.sqrt(sigma ** 2 + eta56.detach().cpu().float() ** 2).reshape(2, N_CAMS).T


def support_from_convex(stl: str) -> torch.Tensor:
    """The base support h for a challenge body from the convex stage's STL for it, made the
    way the corpus starts were (train_lpd.support_from_mesh)."""
    import trimesh
    m = trimesh.load(stl, process=False)
    return support_from_mesh(np.asarray(m.vertices), np.asarray(m.faces))


def make_resid_fn(net, op: CodeOperator, data, scale, geom_mask, M, cond, support, radius):
    """Build resid_fn(z, t) -> FlowInputs for LPDFlow.sample, feeding the network exactly
    what train_lpd.flow_loss feeds it.

    `data` (N_CAMS, 2, P) are the real curves, `scale` (N_CAMS, 2) divides the residual,
    `geom_mask` (1, N_CAMS) says which geometries were measured, `radius` is the published
    xy radius the body is rendered at. `z` is the whitened code and is decoded here;
    `support` is the base h the code's dh block corrects. A draw whose body has no curves is
    masked out for this step.
    """
    sph0, node0 = cond
    C = data.shape[0]
    geoms = torch.nonzero(geom_mask[0] > 0).flatten().tolist()
    gsel = torch.tensor(geoms)
    d_sel, s_sel = data[gsel].to(op.device), scale[gsel].to(op.device)

    def fn(z, t):
        raw = net.codec.decode(z.detach())
        B = len(z)
        pred = torch.zeros(B, C, 2, data.shape[-1])
        adj = torch.zeros(B, CODE_DIM)
        live = torch.ones(B)
        for b in range(B):
            cur, grad = op.adjoint(support, raw[b], radius,
                                   lambda c: (d_sel - c) / s_sel[..., None] ** 2, geoms=geoms)
            if cur is None:
                live[b] = 0.0
                continue
            pred[b, gsel] = cur.cpu()
            adj[b] = grad.cpu()
        m = geom_mask.expand(B, -1) * live[:, None]
        feats = residual_features(data[None].expand(B, -1, -1, -1), pred,
                                  scale[None].expand(B, -1, -1), M, m)
        return flow_inputs(feats, m, sph0, node0, net.codec.pullback(z.detach(), adj))
    return fn


def whitened_misfit(pred, data, scale, geoms, weight=None) -> float:
    """RMS of (data - pred) / scale over the geometries `geoms`, in standard deviations.
    `pred` holds those geometries only, in that order, as the operator returns them; `data`
    and `scale` hold every geometry.

    `weight` (N_CAMS, 2) from `curve_weight`, when given, restricts the average to the curves
    that are measurements. The released set repeats columns and carries count curves, so
    averaging over every column counts one recording twice and mixes the thresholded area
    into the same number as the intensity; a solver that fits only the measurements has to be
    scored on them as well, or its misfit is not the quantity it lowered. Nan when the named
    geometries hold no measurement, which is the honest answer and never wins a comparison.
    """
    r = (data[geoms] - pred) / scale[geoms][..., None]
    if weight is None:
        return float(r.pow(2).mean().sqrt())
    keep = weight[geoms] > 0
    if not bool(keep.any()):
        return float("nan")
    return float(r[keep].pow(2).mean().sqrt())


def mesh_misfit_by_geom(op: CodeOperator, verts, faces, radius, data, scale) -> torch.Tensor:
    """Whitened RMS misfit of a mesh in the canonical frame against the curves, per geometry
    (N_CAMS,), in standard deviations; inf everywhere for a mesh that cannot be rendered, so
    that it is never accepted."""
    f_t = torch.tensor(np.asarray(faces), dtype=torch.long, device=op.device)
    v_t = op.physical(torch.tensor(np.asarray(verts), dtype=torch.float32, device=op.device),
                      f_t, radius)
    try:
        pred = normalise(op.forward.raw_curves(v_t, f_t)).cpu()
    except RadiosityError:
        return torch.full((N_CAMS,), float("inf"))
    return ((pred - data) / scale[..., None]).pow(2).mean((1, 2)).sqrt()


def candidate_diagnostic(kind: str, label: str, verts, faces, misfit_sigma: float) -> dict:
    """Validity and geometry diagnostics for one selectable reconstruction candidate."""
    import trimesh
    from scipy.spatial import ConvexHull

    report = {"kind": kind, "label": label,
              "misfit_sigma": float(misfit_sigma) if np.isfinite(misfit_sigma) else float("inf")}
    try:
        m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=True)
        m.remove_unreferenced_vertices()
        m.merge_vertices()
        m.fix_normals()
        v = np.asarray(m.vertices, dtype=float)
        hull_volume = float(ConvexHull(v).volume) if len(v) >= 4 else float("nan")
        volume = float(m.volume)
        convexity = (abs(volume) / hull_volume
                     if np.isfinite(hull_volume) and hull_volume > 0 else float("nan"))
        report.update({
            "watertight": bool(m.is_watertight),
            "winding_consistent": bool(m.is_winding_consistent),
            "components": int(len(m.split(only_watertight=False))),
            "volume": volume,
            "faces": int(len(m.faces)),
            "convexity": float(convexity),
            "carved_volume_fraction": float(max(0.0, 1.0 - convexity))
                                      if np.isfinite(convexity) else float("nan"),
        })
    except Exception as exc:                         # noqa: BLE001  diagnostic path
        report.update({"watertight": False, "winding_consistent": False,
                       "components": 0, "volume": float("nan"), "faces": 0,
                       "convexity": float("nan"),
                       "carved_volume_fraction": float("nan"), "error": str(exc)})
    report["eligible"] = bool(
        np.isfinite(report["misfit_sigma"])
        and bool(report["watertight"])
        and int(report["components"]) == 1
        and np.isfinite(report["volume"])
        and report["volume"] > 0.0
        and int(report["faces"]) >= 8
    )
    reasons = []
    if not np.isfinite(report["misfit_sigma"]):
        reasons.append("nonfinite_misfit")
    if not report["watertight"]:
        reasons.append("not_watertight")
    if int(report["components"]) != 1:
        reasons.append("not_single_component")
    if not np.isfinite(report["volume"]) or report["volume"] <= 0.0:
        reasons.append("bad_volume")
    if int(report["faces"]) < 8:
        reasons.append("too_few_faces")
    report["ineligible_reasons"] = reasons
    return report


def json_default(obj):
    """Convert NumPy scalar diagnostics to plain JSON values."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def polish(net, op: CodeOperator, z, support, radius, data, scale, geoms, steps: int):
    """Gradient descent on the whitened misfit of one draw, in the whitened code, from where
    the flow left it. Each iteration renders once with the adjoint, steps against the
    gradient by POLISH_STEP per coordinate at first, and halves the step until the misfit
    falls; a step that lowers it grows the next one. It stops when the misfit is at the noise
    level (POLISH_TARGET), after `steps` iterations, or when no step of the allowed sizes
    lowers it. The gradient is zero along the directions the curves do not constrain, so the
    polish moves nothing the data cannot see. Returns (z, misfit before, misfit after,
    iterations taken)."""
    def misfit_of(zz):
        cur = op.curves(support, net.codec.decode(zz), radius, geoms=geoms)
        return float("inf") if cur is None else whitened_misfit(cur.cpu(), data, scale, geoms)

    d_sel, s_sel = data[geoms].to(op.device), scale[geoms].to(op.device)
    z = z.detach().clone()
    alpha, chi0, chi, it = POLISH_STEP, None, None, 0
    while it < steps:
        cur, g = op.adjoint(support, net.codec.decode(z), radius,
                            lambda c: (d_sel - c) / s_sel[..., None] ** 2, geoms=geoms)
        if cur is None:
            break
        chi = whitened_misfit(cur.cpu(), data, scale, geoms)
        chi0 = chi if chi0 is None else chi0
        if chi <= POLISH_TARGET:
            break
        # g is the gradient of -(n/2) chi^2 with respect to the raw code; back to the
        # whitened code, then a unit RMS descent direction
        grad = net.codec.pullback(z, -(2.0 / d_sel.numel()) * g.cpu())
        direction = -grad / grad.pow(2).mean().sqrt().clamp_min(1e-12)
        accepted = False
        for _ in range(5):
            trial = misfit_of(z + alpha * direction)
            if trial < chi:
                z, chi, accepted = z + alpha * direction, trial, True
                alpha = min(alpha * 1.5, 5.0 * POLISH_STEP)
                break
            alpha *= 0.5
        it += 1
        if not accepted:
            break
    if chi0 is None:                      # nothing could be rendered, or no iteration was asked
        chi0 = chi = misfit_of(z)
    return z, chi0, chi, it


def dice_optimal_level(occs: list, extent: float, radius: float, iters: int = 3,
                       lo: float = 0.2, hi: float = 0.7) -> float:
    """The level whose body is the best single answer under the voxel measure, derived rather
    than chosen.

    Write p_v for the fraction of draws occupying voxel v, A for a candidate body and B for
    the truth. The measure is 2|A and B| / (|A| + |B|), so adding voxel v to A raises the
    expected numerator by 2 p_v and the denominator by one. If the body already scores D, the
    change is (2 p_v - D) / (|A| + |B|) to first order, which is positive exactly when
    p_v > D / 2. The level that is right for the body it itself produces is therefore the
    fixed point of t -> D(t) / 2, and D is measured against the draws, which stand in for the
    truth. A ladder of fixed levels cannot do this: the correct level depends on how much the
    draws agree, which is not known before the draws exist. Tight draws score near one and
    want a level near a half; draws that disagree score lower and want a lower level, keeping
    a dent that only some of them have.

    The iteration is started at a half and clamped to [lo, hi], which brackets every value the
    fixed point can take for a score between 0.4 and 1.4 -- the second being unreachable, so
    the upper clamp only guards against a degenerate estimate.
    """
    t = 0.5
    good = None                  # the last level that actually produced a body
    for _ in range(iters):
        made = consensus_bodies(occs, extent, radius, levels=(t,))
        if not made:
            # t is outside the range of the draw fraction, so it would produce nothing
            # downstream either; fall back to the last level that did produce a body.
            return good if good is not None else 0.5
        good = t
        _, v, f = made[0]
        occ = mesh_occupancy(v, f, occs[0].shape[0], extent)
        d = float(np.mean([dice(occ, o) for o in occs]))
        t = float(np.clip(0.5 * d, lo, hi))
    return t


def off_lattice_level(level: float, n_draws: int) -> float:
    """A level strictly between two attainable draw fractions.

    The fraction of draws occupying a voxel takes only the values k / n_draws, so a level
    equal to one of them puts the isosurface exactly through the sampled values. Marching
    cubes then places vertices on grid points and emits zero-area triangles and pinch
    points: the body comes back with a fifth of its faces degenerate and its winding
    inverted, which `export_stl` cannot repair and no voxel measure that relies on
    orientation can read. CONSENSUS_LEVELS holds 0.5 and the draw count is even, so the
    majority level lands on the lattice on every run whatever that count is.

    The level is moved down to the middle of the cell below it, (k - 0.5) / n_draws. That
    keeps exactly the voxels the requested level meant -- those where at least k of the
    n_draws agree -- while passing strictly between attainable values, so every triangle
    has area. A level that is not on the lattice is returned unchanged.
    """
    if n_draws < 1:
        return float(level)
    k = float(level) * n_draws
    kr = round(k)
    return (kr - 0.5) / n_draws if abs(k - kr) < 1e-9 else float(level)


def consensus_bodies(occs: list, extent: float, radius: float, levels=CONSENSUS_LEVELS):
    """Meshes of the level sets of the fraction of draws containing each voxel, for boolean
    grids `occs` on [-extent, extent]^3, posed like the draws. Returns [(level, verts,
    faces)]; a level with no closed surface is left out.

    Each requested level is nudged off the k / len(occs) lattice before the isosurface is
    extracted (see off_lattice_level); the level reported back is the one that was asked
    for, since that is what names the candidate."""
    from skimage import measure
    prob = np.mean([o.astype(np.float32) for o in occs], axis=0)
    n = prob.shape[0]
    spacing = 2.0 * extent / n
    out = []
    for level in levels:
        lv = off_lattice_level(float(level), len(occs))
        if not (prob.min() < lv < prob.max()):
            continue
        v, f, _, _ = measure.marching_cubes(prob, level=lv, spacing=(spacing,) * 3)
        v = v - extent + spacing / 2.0                  # cell centres, not cell corners
        # The radius is capped here rather than set, unlike the draws', which are put at the
        # published radius exactly. A level set is a contour of a probability, not a body: it
        # sits about half a voxel outside or inside the draws' own surface depending on the
        # level, and scaling it to the published radius would correct that offset exactly at
        # the widest point and over-correct everywhere nearer the axis. The offset is a
        # distance, the correction would be a proportion, and measured on real draws the two
        # are the same half per cent, so the cap is left as the smaller distortion.
        out.append((float(level), restore_constraints(v, radius), np.asarray(f, dtype=np.int64)))
    return out


def keep_largest_component(verts, faces, min_fraction: float):
    """Drop every piece but the largest, when the largest is essentially the whole body.

    A level set extracted from a noisy field comes out as the body plus a scatter of specks,
    and a draw with a single speck is thrown away by the same rule that would throw away a
    genuinely bilobed body. Measured over 48 draws of two models, the largest piece held
    100.0% of the volume every time and the rest rounded to nought, and removing them changed
    neither the volume nor the distance of the furthest vertex from the spin axis.

    `min_fraction` is the guard against the case that measurement did not contain. Below it
    the pieces are comparable, the body may really have two lobes, and the draw is left as it
    is -- to be rejected by the usual gate rather than quietly amputated here.

    Returns (verts, faces, n_pieces, largest_fraction); the mesh is unchanged when there is
    one piece or when the largest falls short of min_fraction.
    """
    import trimesh                                   # as elsewhere in this file: a slow import
    # Vertices have to be merged before the split, exactly as the diagnostic above does it.
    # Unmerged, every triangle is its own component -- an STL round trip alone is enough to
    # produce that -- and the split would disagree with the component count this repair is
    # meant to answer to.
    m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=True)
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    pieces = m.split(only_watertight=False)
    if len(pieces) <= 1:
        return verts, faces, len(pieces), 1.0
    vols = [abs(float(q.volume)) for q in pieces]
    total = sum(vols)
    if total <= 0:
        return verts, faces, len(pieces), float("nan")
    k = int(np.argmax(vols))
    frac = vols[k] / total
    if frac < min_fraction:
        return verts, faces, len(pieces), frac
    big = pieces[k]
    return (np.asarray(big.vertices, dtype=float), np.asarray(big.faces),
            len(pieces), frac)


def decode(op: CodeOperator, code, support, res=64, misfit_fn=None, snap: bool = False,
           snap_planes: int = 12, snap_tol: float = 0.02, snap_min_frac: float = 0.02,
           repair_components: float = 0.0):
    """Raw code -> posed mesh in the canonical frame as numpy arrays, with optional planar
    snapping. Returns (None, None, 0) if the extracted mesh is degenerate.

    The pose is the operator's own, so the mesh that leaves here is the mesh whose curves the
    misfit is measured on. The operator poses every iterate it renders, putting the solid
    centroid on the rotation axis, and a mesh posed any other way is a translate of the one
    that was fitted: the misfit would then describe a body other than the one exported. The
    pose is idempotent, so applying it again downstream changes nothing.
    """
    m = op.mesh(support, code, res=res)
    if m is None:
        return None, None, 0
    v, f = m[0], m[1]
    v = CodeOperator.canonical(v, f).cpu().numpy()
    f = f.cpu().numpy()
    if repair_components > 0.0:
        # before the misfit or any diagnostic is taken, so the body that is judged is the
        # body that would be written
        v2, f2, n_pieces, frac = keep_largest_component(v, f, repair_components)
        if n_pieces > 1:
            if v2 is not v:
                import torch as _t
                v2 = CodeOperator.canonical(_t.as_tensor(v2, dtype=_t.float32),
                                            _t.as_tensor(f2)).numpy()
                print(f"    repaired: {n_pieces} pieces, largest {100 * frac:.2f}% of the "
                      f"volume, the rest dropped", flush=True)
                v, f = v2, f2
            else:
                print(f"    left alone: {n_pieces} pieces, largest only "
                      f"{100 * frac:.2f}% of the volume", flush=True)
    kept = 0
    planes = ransac_planes(v, f, n_planes=snap_planes, tol=snap_tol,
                           min_frac=snap_min_frac) if snap else []
    if len(planes):
        v, kept = planar_snap(v, f, planes,
                              misfit_fn=(None if misfit_fn is None
                                         else lambda w: misfit_fn(w, f)),
                              eta=SNAP_MAX_RISE, tol=snap_tol)
        import torch as _t
        v = CodeOperator.canonical(_t.as_tensor(v, dtype=_t.float32),
                                   _t.as_tensor(f)).numpy()
    return v, f, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--ckpt", default="runs/lpd_flow.pt")
    ap.add_argument("--calibration", default=CALIBRATION,
                    help="the Instrument written by scripts/calibrate.py; must be the one "
                         "the flow was trained with")
    ap.add_argument("--data-dir", default="dataset/raw")
    ap.add_argument("--samples", type=int, default=64,
                    help="draws from the flow. They are the candidates and they are also what "
                         "the consensus bodies are built from, and the fraction of draws "
                         "occupying a voxel can only take the values k/samples, so this sets "
                         "how finely a consensus level set can be placed as well as how many "
                         "bodies are scored")
    ap.add_argument("--steps", type=int, default=N_STEPS,
                    help="steps of the sampler from noise to a body; the operator runs at "
                         "each")
    ap.add_argument("--churn", type=float, default=CHURN,
                    help="noise added along the way; 0 is the deterministic flow")
    ap.add_argument("--guidance", type=float, default=GUIDANCE,
                    help="weight on the data part of the velocity (lpd_flow.LPDFlow.velocity). "
                         "One is the model as trained; above one the draws follow the curves "
                         "further from the prior, which is smoother than any single body. "
                         "scripts/decision_check.py measures which weight scores best on "
                         "bodies whose truth is known")
    ap.add_argument("--phases", type=int, default=96)
    ap.add_argument("--operator-res", type=int, default=32,
                    help="FlexiCubes resolution of the operator inside the flow; must match "
                         "the training run")
    ap.add_argument("--res", type=int, default=64,
                    help="FlexiCubes resolution the final mesh is extracted at")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snap", action="store_true",
                    help="enable RANSAC planar snapping postprocess")
    ap.add_argument("--repair-components", type=float, default=0.0, metavar="FRACTION",
                    help="when a draw extracts as several pieces and the largest holds at "
                         "least this fraction of the volume, keep only that piece and drop "
                         "the rest, so the draw is judged as the body it is rather than "
                         "refused for the specks around it. 0 disables it. Below the "
                         "fraction the pieces are comparable, which a bilobed body would "
                         "also look like, so the draw is left alone and the usual gate "
                         "refuses it. 0.98 is a reasonable setting: measured over 48 draws "
                         "the largest piece held 100.0% of the volume every time.")
    ap.add_argument("--snap-planes", type=int, default=12)
    ap.add_argument("--snap-tol", type=float, default=0.02)
    ap.add_argument("--snap-min-frac", type=float, default=0.02)
    ap.add_argument("--medoid-volume-only", action="store_true",
                    help="select the medoid by voxel Dice only")
    ap.add_argument("--medoid-side-points", type=int, default=200000,
                    help="surface samples per draw for side-view medoid selection; use "
                         "1000000 to match hac26/scoring/side_view.py exactly")
    ap.add_argument("--medoid-side-dirs", type=int, default=36)
    ap.add_argument("--medoid-side-res", type=int, default=512)
    ap.add_argument("--medoid-side-mode", choices=["side", "sphere"], default="side")
    ap.add_argument("--support-from", default=None,
                    help="STL whose support function supplies h; defaults to the convex "
                         "stage's reconstruction of this model")
    ap.add_argument("--polish-steps", type=int, default=30,
                    help="most gradient steps of the polish per draw (see polish); 0 skips it")
    ap.add_argument("--hold-out-geoms", type=int, default=0,
                    help="keep this many measured geometries, spread evenly over the list, "
                         "away from the inversion and report the answer's misfit on them; "
                         "0 uses every geometry")
    add_render_flags(ap)
    a = ap.parse_args()
    render = render_from(a, "reconstruction")
    _enable_tf32()
    if not a.medoid_volume_only and a.medoid_side_points <= 0:
        raise SystemExit("--medoid-side-points must be positive unless --medoid-volume-only "
                         "is set")
    torch.manual_seed(a.seed)

    R = CYLINDER_R[a.model]
    psi = psi_grid(a.phases)
    M = min(N_MODES, a.phases // 2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    inst = load_instrument(a.calibration, dev)
    op = CodeOperator(inst, psi, res=a.operator_res, config=render, device=dev)

    sd, flow_meta = load_flow_file(a.ckpt, map_location="cpu")
    if "checkpoint_step" in flow_meta:
        print(f"  {a.ckpt} is a training checkpoint at step {flow_meta['checkpoint_step']}, "
              f"using {'best' if flow_meta.get('loaded_best_state') else 'current'} weights "
              f"from step {flow_meta['loaded_step']}", flush=True)
    check_flow_metadata(flow_meta, calibration=a.calibration, phases=a.phases,
                        operator_res=a.operator_res, context=a.ckpt, render=render_tag(render))
    net = LPDFlow.from_state_dict(sd)
    net.eval()

    sup_stl = a.support_from or f"results/convex/Asteroid{a.model:02d}.stl"
    support = support_from_convex(sup_stl)
    print(f"  h from {sup_stl}: {float(support.min()):.3f}-{float(support.max()):.3f}",
          flush=True)

    d = load_model_curves(a.data_dir, a.model, m=a.phases)
    # A curve file that is absent leaves its block at zero and masked out, which the solver
    # would accept and reconstruct around. An answer built from half the measurement, or none
    # of it, is worse than no answer, so say so instead.
    if set(d["files"]) != {"intensity", "binary"}:
        raise SystemExit(f"model {a.model} needs both measured curve files under "
                         f"{a.data_dir}; found {sorted(d['files'])}")
    data = curve_pairs(d["curves"])                              # (N_CAMS, 2, P)
    mask = geometry_mask(d["mask"])
    scale = residual_scale(d, inst.eta)                          # (N_CAMS, 2)
    tag = geometry_tags()
    present = torch.nonzero(mask[0] > 0).flatten()
    # the geometries the inversion sees, and the ones kept back to test its answer on
    held = present[torch.linspace(0, len(present) - 1, a.hold_out_geoms).round().long()] \
        if a.hold_out_geoms > 0 else present[:0]
    seen = present[~torch.isin(present, held)]
    mask_seen = torch.zeros_like(mask)
    mask_seen[0, seen] = 1.0
    print(f"model {a.model}: R = {R}, {len(present)}/{N_CAMS} geometries present"
          + (f", {len(held)} of them held out: {held.tolist()}" if len(held) else "")
          + f"; residual scale median {float(scale.median()):.4f}", flush=True)

    t0 = time.time()
    cond = cond_channels(support)                    # constant per body: h is fixed here
    codes = net.sample(make_resid_fn(net, op, data, scale, mask_seen, M, cond, support, R),
                       tag.expand(a.samples, -1, -1), mask_seen.expand(a.samples, -1),
                       cond, R, batch=a.samples, n_steps=a.steps, churn=a.churn, guidance=a.guidance)
    print(f"  {a.samples} draws x {a.steps} steps (churn {a.churn:g}) in "
          f"{time.time()-t0:.0f}s", flush=True)

    polished = []
    if a.polish_steps > 0:
        t0 = time.time()
        for i in range(a.samples):
            codes[i], before, after, n_it = polish(net, op, codes[i], support, R, data, scale,
                                                    seen.tolist(), a.polish_steps)
            polished.append([before, after, n_it])
            print(f"  draw {i}: polished from {before:.2f} to {after:.2f} sigma in {n_it} "
                  f"steps", flush=True)
        print(f"  polish in {time.time()-t0:.0f}s", flush=True)

    def misfit_by_geom(w, faces):
        return mesh_misfit_by_geom(op, w, faces, R, data, scale)

    def misfit(w, faces):
        """Whitened RMS misfit of a candidate mesh over the geometries the inversion saw."""
        return float(misfit_by_geom(w, faces)[seen].pow(2).mean().sqrt())

    raw_codes = net.codec.decode(codes)

    # Save the raw codes before anything is decoded. All the operator calls are spent by this
    # line, and the codes are the only record of all the draws: the STL keeps the medoid
    # alone. decode(op, codes[i], support) reproduces draw i exactly.
    codes_path = Path(a.out).with_suffix(".codes.npz")
    try:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(codes_path,
                 codes=raw_codes.detach().cpu().numpy(),
                 support=np.asarray(support, dtype=np.float32),
                 radius=np.float32(R),
                 meta=json.dumps({"model": int(a.model), "code_dim": int(CODE_DIM),
                                  "n_dir": int(N_DIR), "n_nodes": int(N_NODES),
                                  "design_n": int(DESIGN_N), "samples": int(a.samples),
                                  "frame": "canonical; apply fit_to_cylinder(v, radius)"},
                                 sort_keys=True))
        print(f"  codes saved to {codes_path}", flush=True)
    except OSError as exc:                  # a diagnostic must not cost the reconstruction
        print(f"  WARNING: could not save the codes to {codes_path}: {exc}", flush=True)
        codes_path = None

    meshes, snaps, fits, sources = [], [], [], []
    for i in range(a.samples):
        v, f, kept = decode(op, raw_codes[i], support, res=a.res, misfit_fn=misfit,
                            snap=a.snap, snap_planes=a.snap_planes,
                            snap_tol=a.snap_tol, snap_min_frac=a.snap_min_frac,
                            repair_components=a.repair_components)
        if v is None:
            print(f"  draw {i}: degenerate, dropped", flush=True); continue
        chi = misfit(v, f)                   # whitened RMS misfit of this draw, in sigmas
        print(f"  draw {i}: misfit {chi:.2f} sigma", flush=True)
        v = fit_to_cylinder(v, R)            # canonical -> physical: xy only
        meshes.append((v, f)); snaps.append(kept); fits.append(chi)
        sources.append({"kind": "draw", "label": f"draw {i}", "draw": int(i)})
    if not meshes:
        raise SystemExit("every draw was degenerate")
    # one grid for every draw, sized to the widest of them, so the pairwise Dice below
    # compares the same places
    occ_extent = max(float(np.abs(mv).max()) for mv, _ in meshes) * 1.05
    occs = [mesh_occupancy(mv, mf, OCC_RES, occ_extent) for mv, mf in meshes]
    n_draws = len(meshes)
    # the consensus bodies join the draws as candidates; the draws alone are the reference
    # the derived level joins the fixed two; see dice_optimal_level
    if n_draws > 1:
        asked = CONSENSUS_LEVELS + (dice_optimal_level(occs, occ_extent, R),)
        # the derived level can land on one of the fixed ones; asking for it twice builds,
        # samples and scores the same body twice and reports it as two candidates
        levels_used = tuple(dict.fromkeys(round(float(x), 9) for x in asked))
    else:
        levels_used = ()
    extra = consensus_bodies(occs, occ_extent, R, levels=levels_used) if n_draws > 1 else []
    levels = [lv for lv, _, _ in extra]
    candidates = meshes + [(mv, mf) for _, mv, mf in extra]
    sources += [{"kind": "consensus", "label": f"consensus at level {lv:g}",
                 "level": float(lv)} for lv, _, _ in extra]
    occs = occs + [mesh_occupancy(mv, mf, OCC_RES, occ_extent)
                   for _, mv, mf in extra]

    candidate_fits = list(fits)
    for _, mv, mf in extra:
        candidate_fits.append(misfit(mv / np.array([R, R, 1.0]), mf))
    candidate_diagnostics = [
        candidate_diagnostic(src["kind"], src["label"], candidates[i][0], candidates[i][1],
                             candidate_fits[i])
        for i, src in enumerate(sources)
    ]
    eligible = [i for i, row in enumerate(candidate_diagnostics) if row["eligible"]]
    eligible_draws = [i for i in eligible if i < n_draws]
    if not eligible_draws:
        raise SystemExit("no valid draw candidate has finite misfit; refusing to choose an "
                         "answer from consensus bodies alone")
    if not eligible:
        raise SystemExit("no valid LPD candidate has finite misfit")
    invalid = len(candidate_diagnostics) - len(eligible)
    if invalid:
        print(f"  candidate gate: {invalid}/{len(candidate_diagnostics)} candidates ineligible; "
              f"selecting among {len(eligible)} valid candidates", flush=True)
    eligible_occs = [occs[i] for i in eligible]
    eligible_candidates = [candidates[i] for i in eligible]
    n_ref = len(eligible_draws)

    medoid_metric = "volume"
    if a.medoid_volume_only:
        k_local = metric_medoid(eligible_occs, n_ref=n_ref)
    else:
        try:
            from hac26.scoring.side_view import surface_points
        except ImportError as exc:
            raise SystemExit("side-view medoid needs scipy, scikit-image, and trimesh; "
                             "install the project dependencies or pass "
                             "--medoid-volume-only") from exc
        print(f"  side-view medoid: {a.medoid_side_points} surface points/draw, "
              f"{a.medoid_side_dirs} dirs, res {a.medoid_side_res}", flush=True)
        outlines = [surface_points(v, f, n=a.medoid_side_points, seed=a.seed + i)
                    for i, (v, f) in enumerate(eligible_candidates)]
        k_local = metric_medoid(eligible_occs, outlines, side_n_dirs=a.medoid_side_dirs,
                                side_res=a.medoid_side_res, side_mode=a.medoid_side_mode,
                                n_ref=n_ref)
        medoid_metric = "volume+side_view"
        del outlines            # large, and nothing reads it after the medoid
    k = eligible[k_local]

    v, f = candidates[k]
    chosen = sources[k]["label"]
    if a.snap:
        print(f"  planes accepted per draw (allowed rise {SNAP_MAX_RISE}): {snaps}", flush=True)
    else:
        print("  planar snap disabled", flush=True)
    spread = float(np.mean([dice(occs[k], o) for o in occs[:n_draws]]))
    off = ([dice(occs[k], o) for j, o in enumerate(occs[:n_draws]) if j != k] or [0.0])
    spread_off = float(np.mean(off))
    # a consensus body is in the physical frame; the misfit takes canonical vertices
    chosen_fit = float(candidate_diagnostics[k]["misfit_sigma"])
    print(f"  answer = {chosen} of {n_draws} draws and {len(extra)} consensus bodies by "
          f"{medoid_metric}; mean Dice to the draws {spread_off:.4f}; misfit "
          f"{chosen_fit:.2f} sigma (draws {min(fits):.2f}-{max(fits):.2f})", flush=True)
    held_fit = None
    if len(held):
        # the answer in the canonical frame, as misfit_by_geom takes it
        per_geom = misfit_by_geom(v / np.array([R, R, 1.0]), f)
        held_fit = float(per_geom[held].pow(2).mean().sqrt())
        print(f"  held-out geometries {held.tolist()}: misfit {held_fit:.2f} sigma against "
              f"{chosen_fit:.2f} on the seen ones", flush=True)

    collapsed = n_draws > 1 and float(np.mean(
        [dice(occs[i], occs[j]) for i in range(n_draws) for j in range(i + 1, n_draws)])) > SPREAD_MAX
    if collapsed:
        print(f"\n  *** WARNING: mean Dice between the draws is above {SPREAD_MAX}: the draws "
              f"are the same body. Either the correction is dead (rerun scripts/fit_shapes.py "
              f"and check its [check] line) or the flow has collapsed to one answer. The STL "
              f"below is still written.\n", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    info = export_stl(a.out, v, f)
    res = {"model": a.model, "radius": R, "draws": n_draws, "answer": chosen,
           "candidate": int(k), "consensus_levels": [float(x) for x in levels_used],
           # the levels asked for above, and the ones that actually produced a body: a
           # level outside the range of the draw fraction makes none, and it is the second
           # list that indexes the candidates the answer was chosen from
           "consensus_levels_built": [float(x) for x in levels],
           "spread": spread, "spread_off_medoid": spread_off,
           "collapsed": bool(collapsed),
           "medoid_metric": medoid_metric,
           "candidate_diagnostics": candidate_diagnostics,
           "eligible_candidates": [int(i) for i in eligible],
           "medoid_volume_only": bool(a.medoid_volume_only),
           "medoid_side_points": 0 if a.medoid_volume_only else int(a.medoid_side_points),
           "medoid_side_dirs": int(a.medoid_side_dirs),
           "medoid_side_res": int(a.medoid_side_res),
           "medoid_side_mode": a.medoid_side_mode,
           "residual_scale_median": float(scale.median()),
           "steps": int(a.steps), "churn": float(a.churn),
           "polish_steps": int(a.polish_steps), "polish_target": POLISH_TARGET,
           "polish_misfit_sigma": polished,          # per draw: before, after, iterations
           "misfit_sigma": fits, "answer_misfit_sigma": chosen_fit,
           "held_out_geoms": held.tolist(), "seen_geoms": seen.tolist(),
           "held_out_misfit_sigma": held_fit,
           "snap_enabled": bool(a.snap), "snap_max_rise": SNAP_MAX_RISE,
           "snap_planes": int(a.snap_planes), "snap_tol": float(a.snap_tol),
           "snap_min_frac": float(a.snap_min_frac), "planes_accepted": snaps,
           "codes_file": None if codes_path is None else str(codes_path), **info}

    if a.model in PUBLIC_MODELS:
        import trimesh
        t = trimesh.load(public_stl(a.data_dir, a.model), process=False)
        tv = rescale_touch_z(np.asarray(t.vertices), np.asarray(t.faces), centre_xy=False)
        rv = rescale_touch_z(v, f, centre_xy=False)
        e = max(float(np.abs(tv).max()), float(np.abs(rv).max())) * 1.05
        res["dice"] = float(dice(mesh_occupancy(tv, np.asarray(t.faces), 128, e),
                                 mesh_occupancy(rv, f, 128, e)))
        print(f"  DICE vs truth: {res['dice']:.4f}", flush=True)

    print(json.dumps(res, default=json_default))
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=2,
                                                           default=json_default))


if __name__ == "__main__":
    main()
