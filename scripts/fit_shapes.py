#!/usr/bin/env python3
"""Fit the implicit field to every body of the shape library, giving the training corpus.

Per body, the support h is set to the body's own convex hull and frozen, and the depths a are
fitted by regressing the field onto zero at points sampled over the body's own surface. See
DepthFit for why h is not fitted jointly with a, and for why the surface alone is the whole of
the fit. The dh block is stored as zeros: a corpus body's h is exact here, and
scripts/build_corpus.py later sets dh to the correction from the convex stage's start to this h.

With a fixed node set the code is the depth vector itself, so every body is fitted independently
and codes mean the same thing to every reader.

After the fit, a sample of bodies is decoded and its Dice overlap with the mesh it was fitted
to is reported by library family. This is the check that the representation can express the
shapes at all: a family with low fitted Dice (necks, sharp craters) is a family the flow
cannot reconstruct however well it is trained. The per-body values go into the output file.

Public bodies are never in the library.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.shapes import canonicalize_r, rescale_touch_z   # noqa: E402
from hac26.field import (CODE_DIM, DESIGN_N, EXTRACT_EXTENT, KNN, N_DIR,   # noqa: E402
                         N_NODES, NODE_BETA, DepthSphere, ImplicitBody, core_centre,
                         depth_cap, design_sha, dir_design, extract_mesh)
from hac26.recon import dice, mesh_occupancy                # noqa: E402

DICE_RES = 64          # voxel grid of the fitted-Dice check
DICE_EXTENT = 1.35     # half-width of that grid; covers a posed body with its published radius

RIDGE = 1e-3           # relative to the mean diagonal of the normal matrix. Neighbouring node
                       # kernels overlap, so nearby depth vectors describe nearly the same body
                       # and the sample points alone do not choose between them: fitted twice
                       # from different points, one body gets two different codes, and the
                       # difference is a property of the sampling rather than of the body. Both
                       # the ridge and more sample points settle that choice, but the ridge also
                       # costs a little of the shape while points cost only time, so the ridge
                       # is set low and the points carry the work.
SOLVE_CHUNK = 20000    # sample points per block of the solve. The core's support over the
                       # design normals and the normal matrix are both sums or maxima over the
                       # points, so evaluating them a block at a time bounds the memory of the
                       # solve and leaves the point count limited only by the cost of sampling.
POINTS_PER_NODE = 12   # fewest sample points per depth the fit will accept. Measured on the
                       # library: at five per coefficient a carved body's fit overshoots and
                       # decodes to a body unlike itself, at seventeen it reproduces one at
                       # convexity 0.34 to a Dice of 0.97. A smooth body needs far fewer, since
                       # most of its depths are near zero, so this floor is set by the bodies
                       # that matter. See main.
POINTS_PER_NODE_DEFAULT = 18   # sample points per depth when --points is not given. Above the
                       # floor rather than at it: the floor is the fewest the solve tolerates,
                       # and a default sitting on it leaves every run that does not override
                       # it at the edge of being decided by the ridge, where the fitted-Dice
                       # check can refuse the corpus over the sampling rather than the bodies.
                       # Eighteen is just past the seventeen at which a deeply carved body
                       # came back at a Dice of 0.97. Both numbers are per depth, so neither
                       # goes stale when the representation changes how many there are.
DICE_FLOOR = 0.75      # median fitted Dice below which the corpus is refused; see
                       # report_corpus


def sample_arrays(verts, faces, n_pts=6000, seed=0):
    """Points sampled over the body's own surface, by area.

    The whole of the fit lives there. On the surface the field is zero by definition, so the
    depth a direction needs is exactly minus the core's field at that point, and no signed
    distance has to be computed anywhere: what used to be the expensive half of this script is
    not a cheaper calculation but an unnecessary one. Away from the surface the depth field
    cannot follow a signed distance in any case -- it is constant along a ray while the distance
    is not -- so rows taken there ask the representation for something it does not claim and
    pull the fitted depth away from the value the identity gives.

    Sampling by area rather than by vertex because a mesh's vertices crowd where it is curved,
    and a fit weighted that way spends its coefficients on the detail of a rim rather than on
    the surface a camera sees.
    """
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    pts, _ = trimesh.sample.sample_surface(m, int(n_pts), seed=int(seed))
    # The winding decides which side of the surface the body is on, and a mesh that is inside
    # out fits as its own complement with nothing in the residual to show it. The support of an
    # inward-oriented mesh is unchanged, so the volume is the cheapest thing that can tell them
    # apart, and it costs one pass over the faces.
    if m.volume <= 0.0:
        raise ValueError(f"the mesh encloses a volume of {m.volume:.3e}: it is oriented inward, "
                         f"and the body would be fitted as its own complement.")
    return np.asarray(pts, dtype=np.float32)


def _prepare_shape(args):
    """One body's surface samples and hull support, for a worker pool."""
    i, verts, faces, normals, n_pts = args
    pts = sample_arrays(verts, faces, n_pts=n_pts, seed=i)
    h0 = np.maximum((verts @ normals.T).max(axis=0), 1e-3).astype(np.float32)
    return i, pts, h0


class DepthFit:
    """The depths of every body, solved exactly rather than descended to.

    h is pinned to the support of the body's own convex hull and never moves; only the depths
    are fitted. Two reasons:

    1. h and the depths overlap. The same body can be written as a larger core carved more
       deeply or a smaller core carved less. Fitted jointly, one body admits a whole family of
       (h, a) pairs, and a flow trained on that family learns the ambiguity as if it were real.

    2. At reconstruction h does not come from a fit. It comes from the convex stage's estimate
       of the hull. If the corpus's h drifted away from the hull, every corpus depth would have
       been fitted against a core that means something different from the one it is decoded
       against.

    With h fixed the field is linear in the depths, so the fit is a least-squares solve. What it
    is solving is worth stating, because it is an identity and not an approximation: at a point p
    on the body's surface the field has to vanish, so the depth in p's direction is exactly minus
    the core's field there, which is the distance from p out to the core's surface along the
    normal. The weights of a direction sum to one, so

        (Psi^T Psi + lambda I) a = Psi^T (-core),      Psi[i, k] = psi_k(direction of p_i),

    one small symmetric system per body whose right-hand side is the carve itself: it vanishes
    wherever the body agrees with its hull and is largest in the concavities.

    The rows of Psi have KNN non-zeros, so Psi is sparse and the normal matrix is built by one
    sparse product rather than by accumulating dense blocks. Solving reaches whatever depths a
    body needs, however large. A gradient fit does not: each coordinate moves by about the
    learning rate per step, so the depth a body can be carved to is capped by the steps it is
    given, and a deeply carved body comes out shallow with no sign that anything went wrong.

    A body that is not star-shaped about its own centre has directions where the surface is met
    more than once, and no depth satisfies all of them; the least-squares answer is then the
    area-weighted average over the sheets, which is the cost the representation is known to
    carry and not a failure of the solve. Depths beyond the star-shaped bound are refused rather
    than kept, since past it the body no longer contains the centre it is measured from and
    extracts as several pieces.
    """

    def __init__(self, dev, n_nodes=N_NODES):
        ref = ImplicitBody(n_nodes=n_nodes).to(dev)
        self.dev = dev
        self.normals = ref.core.n                                   # (DESIGN_N, 3)
        self.rep = DepthSphere(n_nodes)
        self.n_nodes = n_nodes

    def solve(self, pts, h):
        """One body's depths from its surface samples and its hull support. Returns
        (a, rms before, rms after), the residuals being the root mean square of the field over
        those samples with no depths and with the fitted ones -- which is, by the identity
        above, how far the body is from its own hull before and how far the representation
        leaves it after.

        The core is evaluated a block of points at a time, so the memory the solve needs is set
        by the block size rather than by how many points the body is fitted from."""
        p = torch.as_tensor(pts, dtype=torch.float32, device=self.dev)
        h = torch.as_tensor(h, dtype=torch.float32, device=self.dev)
        core = torch.cat([(p[i:i + SOLVE_CHUNK] @ self.normals.T - h[None, :]).amax(-1)
                          for i in range(0, len(p), SOLVE_CHUNK)]).cpu().numpy()
        o = core_centre(h.detach().cpu().numpy(), self.normals.detach().cpu().numpy())
        psi = self.rep.matrix(np.asarray(pts, dtype=np.float64) - o[None, :])
        A = (psi.T @ psi).toarray().astype(np.float64)
        A[np.diag_indices_from(A)] += RIDGE * max(float(np.diag(A).mean()), 1e-12)
        a = np.linalg.solve(A, psi.T @ (-core.astype(np.float64)))
        a = np.minimum(a, depth_cap(h.detach().cpu().numpy(), o))
        left = psi @ a + core
        return (torch.tensor(a, dtype=torch.float32, device=self.dev),
                float(np.sqrt(np.mean(core ** 2))), float(np.sqrt(np.mean(left ** 2))))


def _load_fit(path: Path, expected: dict, i: int):
    """A body's saved depths as (a, rms before, rms after), or None when the part is missing,
    unreadable or was written under settings that would change the answer."""
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
    except Exception as exc:                           # noqa: BLE001  corrupt: solve it again
        print(f"    ignoring unreadable part {path}: {exc}", flush=True)
        return None
    if int(z["body_index"]) != i or meta != expected:
        print(f"    ignoring stale part {path}", flush=True)
        return None
    return z["a"], float(z["before"]), float(z["after"])


def _save_fit(path: Path, i: int, expected: dict, a: np.ndarray, before: float, after: float):
    """Written under a temporary name and renamed, so a kill mid-write leaves no half part.

    The temporary name has to end in .npz itself: np.savez silently appends .npz to any
    filename that does not already end with it, so a plain "<name>.npz.part" is actually
    written as "<name>.npz.part.npz", and the rename below then fails with FileNotFoundError
    every time, having never found the file it thinks it wrote."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.npz")
    np.savez(tmp, body_index=int(i), meta=json.dumps(expected, sort_keys=True),
             a=a, before=np.float32(before), after=np.float32(after))
    tmp.replace(path)


def fitted_dice(bodies, shape_list, families, n_sample: int, seed: int = 0) -> np.ndarray:
    """Dice of the decoded fitted body against the mesh it was fitted to, for a random sample
    of n_sample bodies (NaN for the rest), printed by family."""
    out = np.full(len(bodies), np.nan)
    idx = np.random.default_rng(seed).permutation(len(bodies))[:n_sample]
    for i in idx:
        b = bodies[i]
        v, f = extract_mesh(lambda y: b(y), EXTRACT_EXTENT, res=DICE_RES,
                            device=str(b.delta.a.device))
        if len(f) < 8:
            out[i] = 0.0
            continue
        sv, sf = shape_list[i]
        out[i] = dice(mesh_occupancy(v, f, DICE_RES, DICE_EXTENT),
                      mesh_occupancy(np.asarray(sv), np.asarray(sf), DICE_RES, DICE_EXTENT))
    print(f"  [dice] fitted body against its mesh, {len(idx)} bodies sampled:")
    for fam in sorted(set(families[i] for i in idx)):
        d = np.array([out[i] for i in idx if families[i] == fam])
        print(f"    {fam:<16} n {len(d):>3}  mean {d.mean():.3f}  min {d.min():.3f}", flush=True)
    return out


def report_corpus(bodies, data, codes, before, after, fit_dice=None) -> bool:
    """Report on the finished corpus. Returns True if it looks usable.

    Called after the corpus is written, never before: a fit that took hours must be flagged,
    not thrown away. The caller turns a False into a non-zero exit.
    """
    with torch.no_grad():
        dmax = max(float(b.delta(data[i].to(b.delta.a.device), centre=b.centre).abs().max())
                   for i, b in enumerate(bodies[:min(8, len(bodies))]))
    gvar = float(codes[:, N_DIR:].var(0).mean())
    gmax = float(np.abs(codes[:, N_DIR:]).max())
    # medians over a stretch of steps, not single steps: each step is one small random
    # minibatch and the per-body loss varies a lot
    r0 = float(np.median(before)) if before else float("nan")
    r1 = float(np.median(after)) if after else float("nan")
    bad = []
    if dmax < 1e-4:
        bad.append(f"the correction is dead: max|Delta| = {dmax:.3e}. Every body in this "
                   f"corpus IS its convex core.")
    if gvar < 1e-12:
        bad.append(f"codes do not vary across bodies: depth variance {gvar:.3e}. "
                   f"Every body was assigned the same code.")
    loss_note = f"weighted residual (median over bodies) {r0:.4f} -> {r1:.4f}"
    print(f"  [check] max|d| {dmax:.4f}, deepest {gmax:.4f}, depth variance {gvar:.5f}, "
          f"{loss_note}", flush=True)
    if before and not (r1 < 0.5 * r0):
        print(f"  WARNING: the depths barely reduced the residual ({r0:.4f} -> {r1:.4f}). "
              f"The corpus was still written, but these bodies are close to their own hulls.",
              flush=True)
    # The decisive check, because it compares the decoded body with the body itself rather
    # than with the sample points it was fitted from. An under-determined solve reproduces
    # its points and not its body: the residual falls, the amplitudes vary, nothing above
    # fires, and the decoded bodies are wrong. Only Dice sees that.
    if fit_dice is not None:
        d = np.asarray(fit_dice, dtype=float)
        d = d[np.isfinite(d)]
        if len(d) and float(np.median(d)) < DICE_FLOOR:
            bad.append(f"the fitted bodies do not reproduce the bodies they were fitted to: "
                       f"median Dice {float(np.median(d)):.3f} over {len(d)} sampled bodies, "
                       f"against a floor of {DICE_FLOOR}. The usual causes are too few sample "
                       f"points for how deeply carved the library is, which --points fixes, and "
                       f"bodies that are not star-shaped about their own centre, which nothing "
                       f"here fixes and which the per-family breakdown above names.")
    for b_ in bad:
        print(f"  ERROR: {b_}", flush=True)
    return not bad


def _out_tmp(out: str) -> str:
    """The stage's product is written here and renamed onto `out` by _out_commit, so a job
    killed mid-write leaves the previous file rather than a truncated one. numpy appends .npz
    to a name without it, so the temporary carries the extension already."""
    return f"{out}.writing.npz"


def _out_commit(out: str) -> None:
    Path(f"{out}.writing.npz").replace(out if out.endswith(".npz") else f"{out}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers for independent SDF/support preprocessing")
    ap.add_argument("--points", type=int, default=POINTS_PER_NODE_DEFAULT * N_NODES,
                    help="surface sample points per body. The depths are solved from these, "
                         "and how far the code is a property of the body rather than of the "
                         "sampling improves as their square root, so this is worth as much as "
                         "can be afforded; they are area samples of a mesh and cost little")
    ap.add_argument("--out", default="runs/corpus_codes.npz")
    ap.add_argument("--device", default=None,
                    help="cuda when available, else cpu; the fit is hours on a CPU and "
                         "minutes on a GPU")
    ap.add_argument("--shapes-dir", required=True,
                    help="directory written by scripts/build_shape_library.py")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed when reading --shapes-dir")
    ap.add_argument("--dice-bodies", type=int, default=64,
                    help="bodies sampled for the fitted-Dice check by family; 0 skips it")
    a = ap.parse_args()
    # The depths are the solution of a system with N_NODES unknowns, and the sample points are
    # its equations. Below a few equations per unknown only the ridge decides the answer, and
    # the fit returns large depths that reproduce the sample points and not the body: the corpus
    # is then quietly wrong, and the flow trains on it for as long as the run takes. A carved
    # body needs the margin more than a smooth one, because more of its depths are doing work.
    # This is a precondition, so it is checked before any body is loaded.
    if a.points < POINTS_PER_NODE * N_NODES:
        raise SystemExit(
            f"--points {a.points} gives {a.points} sample points for {N_NODES} depths, under "
            f"the {POINTS_PER_NODE} per depth the solve needs to be determined by the body "
            f"rather than by the ridge. Use --points {POINTS_PER_NODE * N_NODES} or more.")

    from hac26.library_io import load_library_dir
    print(f"[1] loading {a.bodies} bodies from {a.shapes_dir}", flush=True)
    loaded = load_library_dir(a.shapes_dir, n=a.bodies, seed=a.seed, with_entries=True)
    shape_list = [(v, f) for v, f, _ in loaded]
    families = [str(e.get("base", "unknown")) for _, _, e in loaded]
    # the width over half-height each body was mounted with; a library written before that
    # was recorded carries none, and build_corpus.py then draws a radius instead
    radii = np.array([float(e.get("radius", np.nan)) for _, _, e in loaded])
    if len(shape_list) < a.bodies:
        print(f"  WARNING: only {len(shape_list)} bodies available in {a.shapes_dir}, "
              f"requested {a.bodies}", flush=True)

    # Every body into the canonical frame, whichever source it came from: the nodes are fixed
    # directions in that frame, so node k only means the same direction across bodies if the
    # bodies share it. An already-posed body is untouched.
    n_posed = 0
    posed = []
    for v, f in shape_list:
        v = np.asarray(v, dtype=np.float64)
        # faces passed so the pose centres on the solid centroid, as the library does
        c = canonicalize_r(rescale_touch_z(v, np.asarray(f, dtype=np.int64)))
        if float(np.abs(c - v).max()) > 1e-9:
            n_posed += 1
        posed.append((c, f))
    shape_list = posed
    if n_posed:
        print(f"  posed {n_posed}/{len(shape_list)} bodies into the canonical frame "
              f"(z span 2, xy r_max 1)", flush=True)

    data, h0s = [None] * len(shape_list), [None] * len(shape_list)
    ref = ImplicitBody()
    nrm = ref.core.n.detach().cpu().numpy()
    jobs = [(i, np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64), nrm,
             a.points)
            for i, (v, f) in enumerate(shape_list)]
    workers = max(1, int(a.workers))
    pool = None
    if workers == 1:
        iterator = map(_prepare_shape, jobs)
    else:
        pool = Pool(workers)
        iterator = pool.imap_unordered(_prepare_shape, jobs, chunksize=2)
    failed = False
    try:
        for n_done, (i, pts, h0) in enumerate(iterator, start=1):
            data[i] = torch.tensor(pts)
            h0s[i] = h0
            if n_done % 10 == 0 or n_done == len(jobs):
                print(f"    preprocessed {n_done}/{len(jobs)} bodies", flush=True)
    except Exception:
        failed = True
        if pool is not None:
            pool.terminate()
        raise
    finally:
        if pool is not None:
            if not failed:
                pool.close()
            pool.join()

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    fitter = DepthFit(dev)
    print(f"[2] fit on {dev}: {len(data)} bodies x {N_NODES} depths against "
          f"{DESIGN_N} core normals, solved one body at a time", flush=True)

    # One solve per body, and every solve is independent of every other, so the stage is
    # resumable a body at a time under <out>.parts/ the way the corpus is. Without it a run
    # killed near the end -- a wallclock on a shared machine, most often -- loses every body
    # it fitted, and this stage is hours on a full library.
    part_dir = Path(f"{a.out}.parts")
    expected = {"nodes": int(N_NODES), "points": int(a.points),
                "shapes_dir": str(a.shapes_dir)}
    t0 = time.time()
    bodies, before, after = [], [], []
    n_resumed = 0
    for i, pts in enumerate(data):
        got = _load_fit(part_dir / f"body_{i:05d}.npz", expected, i)
        if got is None:
            g, r0, r1 = fitter.solve(pts, h0s[i])
            _save_fit(part_dir / f"body_{i:05d}.npz", i, expected,
                      np.asarray(g.detach().cpu() if hasattr(g, "detach") else g,
                                 dtype=np.float32), float(r0), float(r1))
        else:
            g, r0, r1 = got
            g = torch.as_tensor(g, dtype=torch.float32, device=dev)
            n_resumed += 1
        b = ImplicitBody().to(dev)
        with torch.no_grad():
            b.set_support(torch.as_tensor(h0s[i]))
            b.delta.a.copy_(g)
        bodies.append(b)
        before.append(r0); after.append(r1)
        if (i + 1) % 25 == 0 or i + 1 == len(data):
            print(f"    fitted {i + 1}/{len(data)} bodies ({n_resumed} resumed), residual "
                  f"median {np.median(before):.4f} -> {np.median(after):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    # the dh block is stored as zeros rather than omitted, so codes.shape[1] is CODE_DIM
    # everywhere
    codes = np.stack([torch.cat([b.dh.detach(), b.delta.a.detach()]).cpu().numpy()
                      for b in bodies])
    sup = np.stack([b.core.h.detach().cpu().numpy() for b in bodies])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": 3,
        "bodies": int(len(bodies)),
        "design_n": int(DESIGN_N),
        "design_sha": design_sha(nrm),
        # the dh directions are a second design, and dh is indexed by them
        "dir_sha": design_sha(dir_design(N_DIR)),
        "code_dim": int(CODE_DIM),
        "n_dir": int(N_DIR),
        "n_nodes": int(N_NODES),
        "node_sha": design_sha(DepthSphere(N_NODES).u.numpy()),
        "knn": int(KNN),
        "node_beta": float(NODE_BETA),
        "points": int(a.points),
        "ridge": float(RIDGE),
        "seed": int(a.seed),
        "shapes_dir": str(Path(a.shapes_dir)),
    }
    fit_d = (fitted_dice(bodies, shape_list, families, a.dice_bodies, seed=a.seed)
             if a.dice_bodies > 0 else np.full(len(bodies), np.nan))
    np.savez(_out_tmp(a.out), codes=codes, support=sup, fit_dice=fit_d, family=np.array(families),
             radius=radii, meta=json.dumps(meta, sort_keys=True))
    _out_commit(a.out)
    print(f"  codes {codes.shape}, depth variance {codes[:, N_DIR:].var(0).mean():.5f}")
    print(f"  wrote {a.out}")
    if not report_corpus(bodies, data, codes, before, after, fit_dice=fit_d):
        raise SystemExit("fit_shapes: the corpus above is degenerate. It was written so the "
                         "fit is not lost, but do not train on it.")


if __name__ == "__main__":
    main()
