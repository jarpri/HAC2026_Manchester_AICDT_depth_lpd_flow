#!/usr/bin/env python3
"""Put the flow's bodies into results/submission, and record which body each file is.

The submission is the end of the n1500 pipeline: for each scored model the flow's
reconstruction, and where the flow produced nothing for a model, the convex answer that is
already there. A model can end with no flow body -- reconstruct_lpd refuses to answer when
every draw comes out in several pieces or when none of them render -- and a convex answer is
a worse body but a valid one, so it stands rather than leaving a gap.

Sources are tried in the order given by --from, first match wins, so the newest run should be
named first:

    python scripts/assemble_submission.py --from results/lpd-late results/lpd

Nothing is overwritten until every chosen file has passed scripts/check_submission.py, and
what was replaced is copied to results/submission/convex-backup first. results/submission/
provenance.json then records, per model, which directory its body came from, that file's
sha256, and the misfit the run reported for it.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCORED = range(4, 11)          # the models a submission is made of
SUBMISSION = Path("results/submission")
BACKUP = Path("results/submission/convex-backup")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def misfit_of(stl: Path):
    """What the run recorded for this body, or None when it wrote no sidecar."""
    side = stl.with_suffix(".json")
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text()).get("answer_misfit_sigma")
    except (ValueError, OSError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="sources", nargs="+",
                    default=["results/lpd-late", "results/lpd"],
                    help="directories to take bodies from, best first")
    ap.add_argument("--models", type=int, nargs="+", default=list(SCORED))
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be copied and check it, but copy nothing")
    a = ap.parse_args()

    chosen, kept = {}, []
    for m in a.models:
        nn = f"{m:02d}"
        for src in a.sources:
            p = Path(src) / f"Asteroid{nn}.stl"
            if p.is_file() and p.stat().st_size > 0:
                chosen[m] = p
                break
        else:
            kept.append(m)

    for m in a.models:
        nn = f"{m:02d}"
        if m in chosen:
            mf = misfit_of(chosen[m])
            print(f"  model {m:>2}: {chosen[m]}"
                  + (f"   misfit {mf:.3f} sigma" if isinstance(mf, float) else ""),
                  flush=True)
        else:
            print(f"  model {m:>2}: no flow body; the convex answer already in "
                  f"{SUBMISSION} stands", flush=True)

    if not chosen:
        raise SystemExit("no flow bodies found in " + ", ".join(a.sources))

    # Check the chosen files before anything is replaced: a submission half swapped for one
    # that does not pass is worse than one not swapped at all.
    staged = Path(".submission_staging")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    for m, p in chosen.items():
        shutil.copy2(p, staged / f"Asteroid{m:02d}.stl")
    print("\n  checking the chosen bodies", flush=True)
    r = subprocess.run([sys.executable, "scripts/check_submission.py", str(staged)])
    if r.returncode != 0:
        shutil.rmtree(staged, ignore_errors=True)
        raise SystemExit("a chosen body failed the submission check; nothing was replaced")

    if a.dry_run:
        shutil.rmtree(staged, ignore_errors=True)
        print("\n  dry run: nothing copied")
        return

    BACKUP.mkdir(parents=True, exist_ok=True)
    record = {}
    for m, p in chosen.items():
        dst = SUBMISSION / f"Asteroid{m:02d}.stl"
        if dst.exists() and not (BACKUP / dst.name).exists():
            shutil.copy2(dst, BACKUP / dst.name)
        shutil.copy2(staged / dst.name, dst)
        record[str(m)] = {"from": str(p), "sha256": sha256(dst),
                          "answer_misfit_sigma": misfit_of(p)}
    for m in kept:
        dst = SUBMISSION / f"Asteroid{m:02d}.stl"
        record[str(m)] = {"from": "convex answer (no flow body for this model)",
                          "sha256": sha256(dst) if dst.exists() else None,
                          "answer_misfit_sigma": None}
    shutil.rmtree(staged, ignore_errors=True)

    (SUBMISSION / "provenance.json").write_text(json.dumps(
        {"sources_in_order": a.sources, "models": record}, indent=2, sort_keys=True))
    print(f"\n  wrote {len(chosen)} flow bodies into {SUBMISSION}, "
          f"kept {len(kept)} convex answer(s)")
    print(f"  replaced files are under {BACKUP}, and {SUBMISSION}/provenance.json "
          f"records where each body came from")


if __name__ == "__main__":
    main()
