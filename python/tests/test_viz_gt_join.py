import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from viz.build_gt_json import build_episode_gt, green_sets_from_grid  # noqa: E402


def test_green_sets_split_openers_from_setups():
    vt = np.zeros((60, 5), np.float32)
    vt[5, 0] = 1.0
    vt[12, 3] = 0.9
    vt[7, 1] = 0.0
    vt[9, 2] = -1.0
    openers, setups = green_sets_from_grid(vt)
    assert openers == {(5, 0)}
    assert setups == {(12, 3)}


@pytest.fixture
def synthetic_h5(tmp_path):
    p = tmp_path / "gt.h5"
    root_vt = np.zeros((60, 5), np.float32); root_vt[5, 0] = 1.0; root_vt[54, 2] = 0.9
    kid_vt = np.zeros((60, 5), np.float32); kid_vt[12, 1] = 1.0
    other_vt = np.zeros((60, 5), np.float32); other_vt[1, 1] = 1.0
    with h5py.File(p, "w") as f:
        f.create_dataset("value_target", data=np.stack([root_vt, kid_vt, other_vt]))
        f.create_dataset("node_kind", data=np.array([b"root", b"depth2", b"root"]))
        f.create_dataset("xml", data=np.array([b"/x/a.xml", b"/x/a.xml", b"/x/b.xml"]))
        f.create_dataset("object_id", data=np.array([b"obj1", b"obj1", b"obj2"]))
        f.create_dataset("parent_edge", data=np.array([-1, 54, -1], np.int16))
        f.create_dataset("parent_depth", data=np.array([-1, 2, -1], np.int16))
    return p


def test_join_returns_root_and_finish_grids(synthetic_h5):
    with h5py.File(synthetic_h5, "r") as f:
        gt = build_episode_gt(f, "/x/a.xml", "obj1")
    assert gt["root"]["openers"] == [[5, 0]]
    assert gt["root"]["setups"] == [[54, 2]]
    assert gt["finish"]["54_2"]["openers"] == [[12, 1]]


def test_missing_root_returns_none(synthetic_h5):
    with h5py.File(synthetic_h5, "r") as f:
        assert build_episode_gt(f, "/x/missing.xml", "obj1") is None
