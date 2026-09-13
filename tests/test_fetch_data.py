"""The data fetch and its integrity check. Nothing here touches the network: what is worth
testing is the logic that decides whether to go near it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_data import entries, sha256                                    # noqa: E402
from fetch_data import as_direct_download, flatten, verify                # noqa: E402


def _tree(root: Path, files: dict):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _manifest(root: Path, path: Path, files):
    path.write_text("".join(f"{sha256(root / f)}  data/raw/{f}\n" for f in files))
    return path


# ---------------------------------------------------------------- the share link

@pytest.mark.parametrize("url,want", [
    ("https://x/y?rlkey=k&dl=0", "https://x/y?rlkey=k&dl=1"),
    ("https://x/y?rlkey=k&dl=1", "https://x/y?rlkey=k&dl=1"),
    ("https://x/y?rlkey=k", "https://x/y?rlkey=k&dl=1"),
    ("https://x/y", "https://x/y?dl=1"),
])
def test_a_share_link_becomes_a_direct_download(url, want):
    assert as_direct_download(url) == want


# ---------------------------------------------------------------- the layout

def test_a_single_wrapping_directory_is_lifted(tmp_path):
    """The May release zipped everything inside HAC_data_May_8/; the manifest and every
    reader want the model directories at the top."""
    _tree(tmp_path, {"HAC_data_May_8/AsteroidModel01_shape_public/a.stl": "x",
                     "HAC_data_May_8/Readme.txt": "y"})
    assert flatten(tmp_path) == "HAC_data_May_8"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["AsteroidModel01_shape_public",
                                                          "Readme.txt"]


def test_an_already_flat_release_is_left_alone(tmp_path):
    _tree(tmp_path, {"AsteroidModel01_shape_public/a.stl": "x", "Readme.txt": "y"})
    assert flatten(tmp_path) is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["AsteroidModel01_shape_public",
                                                          "Readme.txt"]


def test_a_lone_file_is_not_a_wrapper(tmp_path):
    _tree(tmp_path, {"only.txt": "x"})
    assert flatten(tmp_path) is None
    assert [p.name for p in tmp_path.iterdir()] == ["only.txt"]


# ---------------------------------------------------------------- the integrity check

def test_a_clean_tree_verifies(tmp_path):
    root = _tree(tmp_path / "d", {"a.txt": "one", "b/c.txt": "two"})
    man = _manifest(root, tmp_path / "M", ["a.txt", "b/c.txt"])
    assert verify(root, man) == (2, [], [])


def test_a_changed_file_is_reported_not_a_missing_one(tmp_path):
    """The distinction that matters: a file the organisers replaced is *changed*, and the
    fix for it is not the same as for one that never arrived."""
    root = _tree(tmp_path / "d", {"a.txt": "one", "b.txt": "two"})
    man = _manifest(root, tmp_path / "M", ["a.txt", "b.txt"])
    (root / "b.txt").write_text("replaced upstream")
    ok, missing, changed = verify(root, man)
    assert (ok, missing, changed) == (1, [], ["b.txt"])


def test_a_missing_file_is_reported(tmp_path):
    root = _tree(tmp_path / "d", {"a.txt": "one", "b.txt": "two"})
    man = _manifest(root, tmp_path / "M", ["a.txt", "b.txt"])
    (root / "a.txt").unlink()
    ok, missing, changed = verify(root, man)
    assert (ok, missing, changed) == (1, ["a.txt"], [])


def test_the_manifest_prefix_is_stripped(tmp_path):
    """The manifest records paths under data/raw/; the README puts the data in dataset/raw/.
    Entries resolve against whatever root is passed."""
    root = _tree(tmp_path / "anywhere", {"a.txt": "one"})
    man = _manifest(root, tmp_path / "M", ["a.txt"])
    assert [rel for rel, _ in entries(man)] == ["a.txt"]
    assert verify(root, man)[0] == 1
