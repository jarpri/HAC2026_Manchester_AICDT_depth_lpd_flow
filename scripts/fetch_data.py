#!/usr/bin/env python3
"""Fetch the challenge data from the organisers' Dropbox folder into dataset/raw/.

    python scripts/fetch_data.py                    # into dataset/raw
    python scripts/fetch_data.py --force            # re-download over what is there
    HAC_DATA_URL=<link> python scripts/fetch_data.py

A Dropbox shared-folder link with dl=1 returns one .zip of the whole folder, which this
streams down and extracts. The link is the one the challenge page publishes; override it with
--url or the HAC_DATA_URL environment variable if the organisers move it.

Whether the data is already there is decided by dataset/MANIFEST.sha256, not by whether the
directory exists. That distinction is the whole point of this script. The organisers have
re-released these files more than once -- the Blender curves were re-rendered on 29 July 2026
and model 1's real curves realigned on 17/25 August -- and a check that only asks "is there a
directory" answers yes forever, so a re-release is never fetched. That is how this repository
came to pin a snapshot with models 2 and 3 refreshed and model 1 left on the superseded
curves, which no start phase can absorb and which lands in the calibration's residuals looking
like forward-model error. So: a download that verifies clean is skipped, and one that does not
is reported file by file and nothing is downloaded until you ask for it: the archive is
several gigabytes, and a mismatch can equally mean the manifest is the stale half.

The layout also changed between releases. The May zip wrapped everything in a single
HAC_data_May_8/ directory; the current one has the model directories at the top. The manifest
and every reader want them at the top, so a single wrapping directory is lifted out (--no-flatten
to leave it).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_data import entries, is_read, sha256                     # noqa: E402

# The organisers' shared folder, from the challenge page's "Get the data" link.
DROPBOX_URL = ("https://www.dropbox.com/scl/fo/gcqw0ffbt2xa6vfzlhu51/"
               "AESiKo8nIYUKRU_6agUcF1g?rlkey=4o0tiegpeiepxf9ikvnnzfh2s&e=1&dl=0")

# Dropbox refuses urllib's default User-Agent.
USER_AGENT = "Mozilla/5.0 (compatible; hac26 data fetcher)"
CHUNK = 1 << 20


def as_direct_download(url: str) -> str:
    """A Dropbox share link with dl=1, which returns the folder as one zip."""
    if "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    if "dl=1" in url:
        return url
    return f"{url}{'&' if '?' in url else '?'}dl=1"


def download(url: str, dest_zip: Path, timeout: float) -> None:
    """Stream `url` into `dest_zip`, printing progress."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        kind = resp.headers.get("Content-Type", "")
        if kind.startswith("text/html"):
            raise ValueError(f"the server returned a web page, not a zip ({kind}). The link "
                             f"has probably expired or is not a shared-folder link")
        total = resp.length or 0
        tty = sys.stdout.isatty()
        if total:
            print(f"  {total / 1e9:.1f} GB")
        done, next_mark = 0, 0
        with open(dest_zip, "wb") as fh:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if tty:
                    # redrawn in place; only on a terminal, or a log file gets one line per MB
                    print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB" if total
                          else f"\r  {done / 1e6:.0f} MB", end="", flush=True)
                elif total and done * 100 // total >= next_mark:
                    print(f"  {next_mark}%", flush=True)
                    next_mark += 10
    if tty:
        print()


def verify(root: Path, manifest: Path):
    """(matched, missing, changed) of the manifest against what is on disk."""
    ok, missing, changed = 0, [], []
    for rel, digest in entries(manifest):
        path = root / rel
        if not path.exists():
            missing.append(rel)
        elif sha256(path) != digest:
            changed.append(rel)
        else:
            ok += 1
    return ok, missing, changed


def flatten(root: Path) -> str | None:
    """Lift the contents of a single wrapping directory up into `root`.

    Returns the name of the directory that was lifted, or None if there was nothing to do.
    A release that zips its contents directly is left alone, and so is one that unpacks into
    several top-level directories, which is not a wrapper.
    """
    kids = [p for p in root.iterdir() if not p.name.startswith(".")]
    if len(kids) != 1 or not kids[0].is_dir():
        return None
    wrapper = kids[0]
    for item in list(wrapper.iterdir()):
        shutil.move(str(item), str(root / item.name))
    wrapper.rmdir()
    return wrapper.name


def report(root: Path, manifest: Path) -> int:
    """Verify and print; 0 unless a file this repository reads is missing or changed.

    The download carries videos and prose that nothing here opens (check_data.is_read), and a
    mismatch in those is printed and passed over rather than stopping a run that never reads
    them.
    """
    ok, missing, changed = verify(root, manifest)
    print(f"\n{ok} of {ok + len(missing) + len(changed)} files match {manifest}")
    for label, items in (("missing", missing), ("changed", changed)):
        if items:
            print(f"{len(items)} {label}:")
            for rel in items[:20]:
                print(f"   {rel}" + ("" if is_read(rel) else "   (not read here)"))
            if len(items) > 20:
                print(f"   ... and {len(items) - 20} more")
    if changed:
        print("\nA changed file is either something the organisers have replaced since the\n"
              "manifest was written or a manifest entry that was never refreshed. Check the\n"
              "challenge page's News & Updates, then regenerate with\n"
              "  python scripts/check_data.py --write")
    fatal = [rel for rel in missing + changed if is_read(rel)]
    if missing or changed:
        print(f"\n{len(fatal)} of these {'is' if len(fatal) == 1 else 'are'} read by the "
              f"calibration and everything downstream of it."
              if fatal else
              "\nNone of these is read by anything here, so nothing downstream is affected.")
    return 1 if fatal else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("HAC_DATA_URL", DROPBOX_URL),
                    help="Dropbox shared-folder link (default: $HAC_DATA_URL, else the "
                         "link from the challenge page)")
    ap.add_argument("--dest", type=Path, default=Path("dataset/raw"))
    ap.add_argument("--manifest", type=Path, default=Path("dataset/MANIFEST.sha256"))
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--force", action="store_true",
                    help="download even when data is already there, whether or not it "
                         "matches the manifest")
    ap.add_argument("--no-flatten", action="store_true",
                    help="leave a single wrapping directory where the zip put it")
    a = ap.parse_args()

    if not a.url or a.url.startswith("<"):
        print("No download link. Set HAC_DATA_URL, pass --url, or put the challenge page's "
              "link in DROPBOX_URL.", file=sys.stderr)
        return 2

    present = a.dest.is_dir() and any(p for p in a.dest.iterdir()
                                      if not p.name.startswith("."))
    if present and a.manifest.exists() and not a.force:
        ok, missing, changed = verify(a.dest, a.manifest)
        if not [rel for rel in missing + changed if is_read(rel)]:
            print(f"{ok} files in {a.dest} match {a.manifest}, and nothing this repository "
                  f"reads is missing or changed; nothing to do (--force to re-download).")
            return report(a.dest, a.manifest) if (missing or changed) else 0
        # Report and stop. The download is several gigabytes and this is the case where
        # something is already there, so re-fetching it is the caller's decision to make and
        # not a side effect of running a check. It is also not always the right fix: a
        # mismatch can equally mean the manifest is the stale half.
        print(f"{a.dest} does not match {a.manifest}.")
        report(a.dest, a.manifest)
        print(f"\nNothing was downloaded. Re-fetch everything with\n"
              f"  python {Path(sys.argv[0]).name} --force\n"
              f"or, if the files on disk are the ones you want, adopt them with\n"
              f"  python scripts/check_data.py --write")
        return 1

    a.dest.mkdir(parents=True, exist_ok=True)
    # the temporary zip goes beside the destination so the extract is a move within one
    # filesystem rather than a copy across two
    fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=a.dest)
    os.close(fd)
    tmp_zip = Path(tmp_name)
    try:
        print(f"Downloading the challenge data into {a.dest} ...")
        download(as_direct_download(a.url), tmp_zip, a.timeout)
        print("Extracting ...")
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(a.dest)
    except urllib.error.URLError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    except (zipfile.BadZipFile, ValueError) as exc:
        print(f"That was not a usable zip: {exc}", file=sys.stderr)
        return 1
    finally:
        tmp_zip.unlink(missing_ok=True)

    if not a.no_flatten:
        lifted = flatten(a.dest)
        if lifted:
            print(f"lifted the contents of {lifted}/ up into {a.dest} "
                  f"(the manifest and every reader want the model directories at the top)")

    if not a.manifest.exists():
        print(f"\nExtracted to {a.dest}. No {a.manifest} to check it against; write one with\n"
              f"  python scripts/check_data.py --write")
        return 0
    return report(a.dest, a.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
