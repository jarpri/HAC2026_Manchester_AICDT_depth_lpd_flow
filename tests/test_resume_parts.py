"""Every stage that resumes writes its parts atomically, and a saved part is a part the
loader can read.

Both savers write under a temporary name and rename, which is what makes a job killed
mid-write leave the previous part rather than half of one. numpy appends .npz to any filename
that does not already end in it, so a temporary called ".part" is written as ".part.npz" and
the rename then looks for a file that was never created -- a saver that has never been called
looks correct and fails on its first body. These tests call them.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_a_fitted_body_saves_where_its_loader_looks(tmp_path):
    fs = _module("fit_shapes")
    path = tmp_path / "body_00007.npz"
    keys = {"nodes": 2560, "points": 30720, "shapes_dir": "somewhere"}
    fs._save_fit(path, 7, keys, np.arange(5, dtype=np.float32), 0.4, 0.02)
    assert path.exists(), "the part was written under a name the rename did not produce"
    assert [q.name for q in tmp_path.iterdir()] == [path.name], "a temporary was left behind"
    got = fs._load_fit(path, keys, 7)
    assert got is not None
    a, before, after = got
    assert np.allclose(a, np.arange(5)) and before == pytest.approx(0.4)
    assert after == pytest.approx(0.02)
    # a part from other settings, or for another body, is not reused
    assert fs._load_fit(path, {**keys, "points": 1}, 7) is None
    assert fs._load_fit(path, keys, 8) is None


def test_a_corpus_body_saves_every_field_the_corpus_stacks(tmp_path):
    bc = _module("build_corpus")
    path = tmp_path / "body_00003.npz"
    part = {k: np.arange(4, dtype=np.float32) for k in bc.PART_FIELDS}
    expected = {"bodies": 4, "phases": 8, "operator_res": 24}
    bc._save_part(path, 3, part, expected)
    assert path.exists()
    assert [q.name for q in tmp_path.iterdir()] == [path.name]
    got = bc._load_part(path, expected, 3)
    assert got is not None
    # every field, because the corpus stacks each of them at the end: a resumed body that
    # brought back a subset used to raise there, after the stage had done all its rendering
    assert set(got) == set(bc.PART_FIELDS)
    # a part missing a field is refused at save time rather than at the final write
    with pytest.raises(KeyError):
        bc._save_part(tmp_path / "body_00004.npz", 4,
                      {k: v for k, v in part.items() if k != "turned_counts"}, expected)


def test_a_stage_product_is_renamed_onto_the_path_readers_use(tmp_path):
    for name in ("fit_shapes", "build_corpus"):
        m = _module(name)
        for spelling in ("codes.npz", "codes"):
            out = str(tmp_path / spelling)
            np.savez(m._out_tmp(out), x=np.arange(3))
            m._out_commit(out)
            landed = Path(out if out.endswith(".npz") else out + ".npz")
            assert landed.exists(), f"{name}: {spelling} did not land where readers look"
            assert not Path(f"{out}.writing.npz").exists()
            landed.unlink()


def test_a_learned_object_records_the_discretisation_it_was_trained_at():
    """Every stage that renders takes the sensor size as a flag, so a flow trained against
    curves from one discretisation could be used against another and nothing else here would
    notice: the corpus digest, the phase count and the extraction resolution all agree while
    the curves themselves are a different instrument's. The tag is what closes that."""
    from train_lpd import RENDER, check_flow_metadata, render_tag
    from dataclasses import replace
    small = replace(RENDER, height=48, width=80, sun_res=32)
    assert render_tag(RENDER) != render_tag(small)
    # only the sensor and the sun view are in the tag; chunking does not change a curve
    assert render_tag(replace(RENDER, phase_chunk=1, geom_chunk=1)) == render_tag(RENDER)
    meta = {"phases": 96, "operator_res": 32, "render": render_tag(small)}
    check_flow_metadata(meta, phases=96, operator_res=32, render=render_tag(small))
    with pytest.raises(SystemExit):
        check_flow_metadata(meta, phases=96, operator_res=32, render=render_tag(RENDER))


def test_the_corpus_records_the_discretisation_its_curves_were_rendered_at():
    """A corpus built small to check the wiring has to be a different corpus, or a later run
    reuses it and trains the flow on curves from an instrument it will never meet."""
    from dataclasses import replace
    from train_lpd import RENDER
    bc = _module("build_corpus")
    small = replace(RENDER, height=48, width=80, sun_res=32)
    args = (4, 16, 24, "models/lpd_convex.pt", "models/lpd_convex.pt")
    full = bc.corpus_meta(*args, RENDER)
    assert full["render"]["height"] == RENDER.height
    assert bc.corpus_meta(*args, small) != full
