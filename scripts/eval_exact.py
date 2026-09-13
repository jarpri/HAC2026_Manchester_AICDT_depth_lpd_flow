#!/usr/bin/env python3
"""Score convex checkpoints, singly or as an ensemble, with the closed-form Dice of
hac26.radial on held-out meshes and on the public models.

Several checkpoints are combined by averaging their support functions after posing. The
mean of support functions is the support function of the Minkowski average of the bodies,
so the ensemble is a convex body with no repair step. Every body is scored in four decode
variants: with and without support smoothing, with and without the cylinder fit.

    python scripts/eval_exact.py --ckpt a.pt b.pt --ensemble --tag ens --heldout <dir>
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hac26.adapter import curves_from_npz  # noqa: E402
from hac26.conventions import CYLINDER_R, PUBLIC_MODELS  # noqa: E402
from hac26.data_io import load_model_curves, public_stl  # noqa: E402
from hac26.radial import (dice_from_radial, fibonacci_sphere, mesh_radial)  # noqa: E402
from hac26.recon import (body_from_support, fit_to_cylinder, smooth_support)  # noqa: E402
from hac26.geometry import project_closure  # noqa: E402
from hac26.solvers.minkowski import solve_minkowski  # noqa: E402
from hac26.shapes import hull_mesh, mesh_support, rescale_touch_z  # noqa: E402
from hac26.stl_io import load_stl  # noqa: E402
from hac26.train import load_net  # noqa: E402


def predict_h(net, grid, d, mask, radius):
    """Support function on the grid normals of the body a checkpoint predicts.

    With a support head that is the head's output. Otherwise the predicted EGI is closed,
    solved by Minkowski and posed, and the support function of that body is taken, so
    every checkpoint lands in the same representation and can join the ensemble.
    """
    with torch.no_grad():
        dd = torch.as_tensor(d, dtype=torch.float32)[None]
        mm = torch.as_tensor(mask, dtype=torch.float32)[None]
        if getattr(net, "r_cond", False):
            out = net(dd, mm, torch.tensor([float(np.log(radius))], dtype=torch.float32))
        else:
            out = net(dd, mm)
    if getattr(net, "support_head", False) and len(out) > 2 and out[2] is not None:
        return out[2][0].numpy().astype(float)
    p = project_closure(out[0][0].numpy().astype(float), grid.normals)
    sol = solve_minkowski(grid.normals, p, drop_tol=2e-4)
    return mesh_support(rescale_touch_z(sol["verts"]), grid.normals)


def decode(h, grid, smooth, radius, fit):
    """Support function -> optional smoothing -> half-space intersection -> challenge pose
    -> optional cylinder fit. Returns (verts, faces) of the hull."""
    if smooth:
        h = smooth_support(h, grid.n_theta, grid.n_phi, k=smooth)
    v, _ = body_from_support(grid.normals, h)
    v, f = hull_mesh(v)
    v = rescale_touch_z(v)
    if fit and radius:
        v = fit_to_cylinder(v, radius)
    return hull_mesh(v)


def ensemble_body(h_list, grids, radius, smooth, fit):
    """Minkowski average of the members: decode and pose each one, average their support
    functions on the first grid's normals, and decode the mean the same way."""
    hs = []
    for h, grid in zip(h_list, grids):
        v, _ = decode(h, grid, smooth, radius, fit)
        hs.append(mesh_support(v, grids[0].normals))
    v, _ = body_from_support(grids[0].normals, np.mean(hs, axis=0))
    v, f = hull_mesh(v)
    v = rescale_touch_z(v)
    if fit and radius:
        v = fit_to_cylinder(v, radius)
    return hull_mesh(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="checkpoint files to score")
    ap.add_argument("--tag", required=True, help="name of the output json")
    ap.add_argument("--ensemble", action="store_true",
                    help="combine all --ckpt by Minkowski average instead of scoring "
                         "the first one")
    # The defaults point outside the repo; pass --heldout explicitly. Results are grouped
    # by Path(root).name, so name the directory holding the meshes, not its parent.
    ap.add_argument("--heldout", nargs="*",
                    default=["../data/team/heldout2/dataset/external",
                             "../data/team/split_test"],
                    help="directories of held-out meshes with stored curves")
    ap.add_argument("--data-dir", default="../data/raw",
                    help="challenge data directory, for the public models")
    ap.add_argument("--rays", type=int, default=4096,
                    help="number of sphere directions for the Dice quadrature")
    ap.add_argument("--out", default="results/eval", help="output directory")
    args = ap.parse_args()

    nets, grids, prs = [], [], []
    for c in args.ckpt:
        n, p, g = load_net(c, device="cpu")
        nets.append(n)
        grids.append(g)
        prs.append(p)
        print(f"loaded {c}  support={getattr(n,'support_head',False)} "
              f"r_cond={getattr(n,'r_cond',False)} canonical={getattr(p,'canonical_r',False)}")
    torch.set_num_threads(2)   # keep the CPU footprint small
    rays = fibonacci_sphere(args.rays)
    pr = prs[0]
    results = {"tag": args.tag, "ckpts": args.ckpt, "ensemble": bool(args.ensemble),
               "rows": []}

    variants = [(s, f) for s in (0, 1) for f in (False, True)]
    n_done = [0]

    def score(name, gt_verts, gt_faces, d, mask, radius, group):
        gv, gf = hull_mesh(rescale_touch_z(gt_verts))
        rho_true = mesh_radial(gv, gf, rays)
        h_list = [predict_h(n, g, d, mask, radius) for n, g in zip(nets, grids)]
        n_done[0] += 1
        if n_done[0] % 10 == 0:
            print(f"    ...{n_done[0]} bodies scored", flush=True)
        for sm, fit in variants:
            try:
                if args.ensemble:
                    v, f = ensemble_body(h_list, grids, radius, sm, fit)
                else:
                    v, f = decode(h_list[0], grids[0], sm, radius, fit)
                sc = dice_from_radial(mesh_radial(v, f, rays), rho_true)
            except Exception as e:
                print(f"  {name} sm{sm} fit={int(fit)}: FAILED {e}", flush=True)
                continue
            results["rows"].append({"group": group, "shape": name, "smooth": sm,
                                    "fit_cylinder": fit, "dice": sc})

    for root in args.heldout:
        if not Path(root).exists():
            continue
        stls = sorted(glob.glob(str(Path(root) / "**" / "*.stl"), recursive=True))
        grp = Path(root).name
        for stl in stls:
            npz = next((c for c in [Path(stl).with_suffix("").as_posix() + "_curves.npz",
                                    Path(stl).with_suffix(".npz").as_posix()]
                        if Path(c).exists()), None)
            if not npz:
                continue
            gv, gf = load_stl(stl)
            d, mask = curves_from_npz(npz, pr.m, eps=pr.eps_norm)
            hv, _ = hull_mesh(rescale_touch_z(gv))
            r_true = float(np.sqrt((hv[:, :2] ** 2).sum(1)).max())
            score(str(Path(stl).relative_to(root)).replace(".stl", ""),
                  gv, gf, d, mask, r_true, grp)
        print(f"  scored {grp}: {len(stls)} meshes", flush=True)

    for Mno in PUBLIC_MODELS:
        dm = load_model_curves(args.data_dir, Mno, m=pr.m)
        if not dm["files"]:
            continue
        ov, of = load_stl(public_stl(args.data_dir, Mno))
        score(f"model{Mno}", ov, of, dm["curves"], dm["mask"], CYLINDER_R[Mno], "public")

    print(f"\n{'group':<14}{'variant':<12}{'n':>4}{'mean dice':>11}{'se':>9}")
    summary = []
    for grp in sorted({r["group"] for r in results["rows"]}):
        for sm, fit in variants:
            vals = [r["dice"] for r in results["rows"]
                    if r["group"] == grp and r["smooth"] == sm and r["fit_cylinder"] == fit]
            if not vals:
                continue
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            summary.append({"group": grp, "smooth": sm, "fit_cylinder": fit,
                            "n": len(vals), "mean": m, "se": se})
            print(f"{grp:<14}{f'sm{sm} fit={int(fit)}':<12}{len(vals):>4}{m:>11.4f}{se:>9.4f}")
    results["summary"] = summary
    p = Path(args.out) / f"exact_{args.tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
