import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_colossus0_mix", REPO / "scripts" / "pipeline" / "build_colossus0_mix.py"
)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def _arrays(n):
    return {
        "ctx": np.zeros((n, 5, 64, 64), np.float16),
        "contact_px": np.zeros((n, 60, 2), np.float32),
        "r_mask": np.ones((n, 60, 5), np.float32),
        "value_target": np.zeros((n, 60, 5), np.float32),
        "value_mask": np.zeros((n, 60, 5), np.float32),
        "ceiling_mask": np.zeros((n, 60, 5), np.float32),
    }


def _write_base(path):
    arrays = _arrays(2)
    arrays["value_target"][:, 0, 0] = 1.0
    arrays["value_mask"][:, 0, 0] = 1.0
    with h5py.File(path, "w") as h5:
        for key, value in arrays.items():
            h5[key] = value
        h5["xml"] = np.array(["old/a.xml", "old/b.xml"], dtype=h5py.string_dtype())
        h5["object_id"] = np.array(["o0", "o1"], dtype=h5py.string_dtype())
        h5["is_root"] = np.array([1, 0], np.int8)
        h5["sample_weight"] = np.ones(2, np.float32)
        h5.attrs["n_samples"] = 2


def _write_new(path):
    arrays = _arrays(5)
    # root setup-positive; finish rank-2 positive; finish rank-1 positive (excluded)
    arrays["value_target"][0:3, 0, 0] = [0.9, 1.0, 1.0]
    arrays["value_mask"][0:3, 0, 0] = 1.0
    # root and finish negative-only ceiling rows
    arrays["value_target"][3:5, 0, 0] = [0.81, 0.9]
    arrays["value_mask"][3:5, 0, 0] = 1.0
    arrays["ceiling_mask"][3:5, 0, 0] = 1.0
    with h5py.File(path, "w") as h5:
        for key, value in arrays.items():
            h5[key] = value
        h5["xml"] = np.array([f"new/{i}.xml" for i in range(5)], dtype=h5py.string_dtype())
        h5["object_id"] = np.array([f"o{i}" for i in range(5)], dtype=h5py.string_dtype())
        h5["node_kind"] = np.array(["root", "depth2", "depth2", "root", "depth2"],
                                         dtype=h5py.string_dtype())
        h5["winner_rank"] = np.array([0, 2, 1, 0, 0], np.int32)


def test_exact_mix_appends_to_d20_base(tmp_path, monkeypatch):
    base = tmp_path / "base.h5"
    new = tmp_path / "new.h5"
    out = tmp_path / "out.h5"
    report = tmp_path / "report.json"
    _write_base(base)
    _write_new(new)
    monkeypatch.setattr(sys, "argv", [
        "build_colossus0_mix.py", "--base-h5", str(base), "--new-h5-glob", str(new),
        "--out", str(out), "--report", str(report), "--new-rows", "4", "--negative-rows", "2",
    ])

    BUILD.main()

    with h5py.File(out, "r") as h5:
        assert h5.attrs["n_samples"] == 6
        assert h5.attrs["base_rows"] == 2
        xmls = BUILD._dec(h5["xml"][:])
        assert "new/2.xml" not in xmls
        assert set(xmls[2:]) == {"new/0.xml", "new/1.xml", "new/3.xml", "new/4.xml"}
        assert np.isclose(h5["sample_weight"][:][h5["is_root"][:] == 1].sum(), 3.0)
        assert np.isclose(h5["sample_weight"][:][h5["is_root"][:] == 0].sum(), 3.0)
