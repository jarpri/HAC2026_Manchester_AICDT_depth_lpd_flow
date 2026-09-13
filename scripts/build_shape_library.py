#!/usr/bin/env python3
"""Build a shape library to disk, in parallel and resumably.

    python scripts/build_shape_library.py --n 5000 --out dataset/generated/shapes \
        --workers 16 --shape-models dataset/shape_models

Body `i` uses its own generator seeded from (seed, i), so generation is parallel across
processes, and each body is saved to its own file as soon as it is made, so a killed run
continues where it stopped on restart. With --shape-models, real shape models (see
scripts/fetch_shape_models.py) are one of the families every body is drawn from.

Output layout:

    {out}/body_00000.npz, body_00001.npz, ...   verts, faces, recipe, info (see library_io)
    {out}/manifest.json                          index -> file, family, convexity, radius
    {out}/report.md                              validity and diversity summary

`scripts/fit_shapes.py --shapes-dir {out}` reads this directly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hac26.library_io import load_body, save_body, write_manifest        # noqa: E402
from hac26.library_metrics import (check_library, library_descriptors,   # noqa: E402
                                   pairwise_dice, participation_ratio)
from hac26.shape_library import (LibrarySpec, body_from_convex_points,   # noqa: E402
                                 load_shape_models, sample_body)


def _entry(i: int, path: Path, b, skipped: bool, seconds: float | None = None) -> dict:
    """A manifest entry: what fit_shapes.py and build_corpus.py need without the mesh."""
    e = {"index": i, "file": path.name, "base": b.recipe["base"],
         "convexity": float(b.info["convexity"]),
         "radius": float(b.info["cylinder_radius"]),
         "n_faces": int(len(b.faces)), "skipped": skipped}
    if seconds is not None:
        e["seconds"] = seconds
    return e


def _worker(args) -> dict:
    """Runs in a subprocess: generate body `i`, save it, and return a small summary rather
    than the mesh. A body whose file already exists and loads is reused, not regenerated."""
    i, seed, out_dir, spec_kw = args
    path = Path(out_dir) / f"body_{i:05d}.npz"
    if path.exists():
        try:
            b = load_body(str(path))
            if "cylinder_radius" in b.info:
                return _entry(i, path, b, True)
        except Exception:                                    # noqa: BLE001  corrupt: redo
            pass
    spec = LibrarySpec(**spec_kw)
    t0 = time.time()
    b = sample_body(np.random.default_rng([seed, i]), spec)
    save_body(str(path), b)
    return _entry(i, path, b, False, time.time() - t0)


def build(n: int, seed: int, out_dir: str, workers: int, spec: LibrarySpec,
          checkpoint_every: int = 100) -> list:
    """Generate `n` bodies with a process pool, rewriting the manifest every
    `checkpoint_every` bodies. Returns the manifest entries in index order."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    spec_kw = {"res": spec.res, "extent": spec.extent, "radius": spec.radius,
               "family_weights": spec.family_weights, "n_modifiers": spec.n_modifiers,
               "mod_weights": spec.mod_weights, "mount_weights": spec.mount_weights,
               "tilt_deg": spec.tilt_deg, "max_tilt_deg": spec.max_tilt_deg,
               "shape_models": tuple(spec.shape_models),
               "object_models": tuple(spec.object_models), "max_attempts": spec.max_attempts,
               "convexity_bins": spec.convexity_bins, "convexity_shares": spec.convexity_shares,
               "band_attempts": spec.band_attempts}
    jobs = [(i, seed, out_dir, spec_kw) for i in range(n)]
    results = [None] * n
    t0 = time.time()
    n_done = 0
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_worker, jobs, chunksize=4):
            results[r["index"]] = r
            n_done += 1
            if n_done % checkpoint_every == 0 or n_done == n:
                done = [r for r in results if r is not None]
                write_manifest(out_dir, done)
                rate = n_done / max(time.time() - t0, 1e-9)
                eta_min = (n - n_done) / max(rate, 1e-9) / 60.0
                n_new = sum(1 for r in done if not r["skipped"])
                print(f"  {n_done:>5}/{n}  ({n_new} generated, {n_done - n_new} resumed)  "
                      f"{rate:.2f} bodies/s  ETA {eta_min:.1f} min", flush=True)
    return [r for r in results if r is not None]


def ingest_damit(out_dir: str, start_index: int, spec: LibrarySpec, seed: int,
                 damit_points: str | None) -> list:
    """Append bodies built on DAMIT convex models after the generated ones, with contiguous
    indices, so `scripts/fit_shapes.py` sees one library. Skipped unless a path is given.
    Returns the new manifest entries."""
    extra = []
    if not damit_points:
        return extra
    print(f"[ingest] DAMIT convex bases from {damit_points}", flush=True)
    z = np.load(damit_points, allow_pickle=True)
    pointsets = z["pointsets"] if "pointsets" in z else [z[k] for k in z.files]
    rng = np.random.default_rng([seed, "damit"])
    i = start_index
    for pts in pointsets:
        try:
            b = body_from_convex_points(np.asarray(pts, float), rng, spec=spec)
        except RuntimeError as e:
            print(f"  skipped one DAMIT shape: {e}", flush=True)
            continue
        path = Path(out_dir) / f"body_{i:05d}.npz"
        save_body(str(path), b)
        extra.append(_entry(i, path, b, False))
        i += 1
    print(f"  ingested {len(extra)} DAMIT-based bodies", flush=True)
    return extra


def write_report(out_dir: str, entries: list, spec: LibrarySpec, sample_n: int = 300,
                 seed: int = 0) -> None:
    """Write `report.md`: validity and diversity on a random sample of `sample_n` bodies,
    plus convexity and base-kind counts over the whole library."""
    from hac26.library_io import load_body

    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(entries), sample_n, replace=False)
          if len(entries) > sample_n else np.arange(len(entries)))
    sample = [load_body(str(Path(out_dir) / entries[i]["file"])) for i in idx]

    chk = check_library(sample, radius=spec.radius)
    desc = library_descriptors(sample, n_probes=150, res=32)
    dice = pairwise_dice(sample, res=32, max_pairs=250, seed=seed)
    bases = {}
    for e in entries:
        bases[e["base"]] = bases.get(e["base"], 0) + 1
    conv = np.array([e["convexity"] for e in entries])
    rad = np.array([e["radius"] for e in entries])
    q = [0.1, 0.25, 0.5, 0.75, 0.9]

    lines = [
        f"# Shape library report: {len(entries)} bodies", "",
        f"Validity, measured on {len(sample)} bodies sampled from the full library:", "",
        "| check | pass |", "|---|---|",
    ]
    for k, v in chk["pass"].items():
        lines.append(f"| {k} | {v}/{len(sample)} |")
    lines += [
        "", f"Convexity (volume / hull volume) over all {len(entries)} bodies, at the 10th, "
            f"25th, 50th, 75th and 90th percentiles: "
            + ", ".join(f"{x:.3f}" for x in np.quantile(conv, q))
            + f"; {(conv < 0.9).mean() * 100:.0f}% below 0.9, "
              f"{(conv >= 0.98).mean() * 100:.0f}% at 0.98 or above",
        "", "Cylinder radius (the published-style width over half-height) at the same "
            "percentiles: " + ", ".join(f"{x:.2f}" for x in np.quantile(rad, q)),
        "", f"Diversity, measured on the same {len(sample)}-body sample:", "",
        f"- PR(support)  = {participation_ratio(desc['support']):.2f}",
        f"- PR(concavity) = {participation_ratio(desc['concavity']):.2f}",
        f"- PR(combined)  = {participation_ratio(desc['combined']):.2f}",
        f"- pairwise Dice: mean {dice.mean():.3f}, std {dice.std():.3f}, "
        f"range [{dice.min():.3f}, {dice.max():.3f}]",
        "", "Base archetype counts over the full library:", "",
    ]
    for k, v in sorted(bases.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}")
    text = "\n".join(lines) + "\n"
    with open(Path(out_dir) / "report.md", "w") as fh:
        fh.write(text)
    print("\n" + text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000, help="number of bodies")
    ap.add_argument("--seed", type=int, default=0, help="base seed; body i uses (seed, i)")
    ap.add_argument("--out", default="dataset/generated/shapes", help="output directory")
    ap.add_argument("--workers", type=int, default=8, help="number of worker processes")
    ap.add_argument("--res", type=int, default=64, help="marching-cubes grid resolution")
    ap.add_argument("--extent", type=float, default=1.6,
                    help="half-width of the sampling grid")
    ap.add_argument("--shape-models", default=None,
                    help="directory of real asteroid shape models (obj, wf, stl, ply); with "
                         "it, the 'real' family draws from them (scripts/fetch_shape_models.py)")
    ap.add_argument("--objects", default=None,
                    help="directory of everyday objects for the 'object' family "
                         "(scripts/fetch_objects.py); defaults to <shape-models>/objects "
                         "when that exists")
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="rewrite the manifest after this many bodies")
    ap.add_argument("--damit-points", default=None,
                    help="npz of point clouds, one array per DAMIT shape")
    ap.add_argument("--report-sample", type=int, default=300,
                    help="number of bodies sampled for the report's validity and diversity")
    a = ap.parse_args()

    models = load_shape_models(a.shape_models) if a.shape_models else []
    if a.shape_models and not models:
        raise SystemExit(f"no readable shape model in {a.shape_models}")
    objects_dir = a.objects or (a.shape_models and str(Path(a.shape_models) / "objects"))
    objects = load_shape_models(objects_dir) if objects_dir and Path(objects_dir).is_dir() else []
    spec = LibrarySpec(res=a.res, extent=a.extent, shape_models=tuple(models),
                       object_models=tuple(objects))
    w = spec.weights()
    print(f"[1/3] bodies: n={a.n}, workers={a.workers}, res={a.res}, out={a.out}; "
          f"{len(models)} real shape models, {len(objects)} objects; families "
          + ", ".join(f"{k} {v / sum(w.values()):.2f}" for k, v in w.items()), flush=True)
    entries = build(a.n, a.seed, a.out, a.workers, spec, a.checkpoint_every)

    entries += ingest_damit(a.out, len(entries), spec, a.seed, a.damit_points)
    write_manifest(a.out, entries)

    print(f"[2/3] wrote {len(entries)} bodies to {a.out}", flush=True)
    print("[3/3] validity + diversity report", flush=True)
    write_report(a.out, entries, spec, sample_n=a.report_sample, seed=a.seed)


if __name__ == "__main__":
    main()
