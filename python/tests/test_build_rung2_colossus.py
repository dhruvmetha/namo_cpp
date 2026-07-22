import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_rung2_h5", REPO / "scripts" / "pipeline" / "build_rung2_h5.py"
)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_colossus_root_masks_censored_parent_and_uses_ceilings():
    trials = [
        {"edge_idx": 1, "depth": 0, "success": False},
        {"edge_idx": 2, "depth": 1, "success": False},
        {"edge_idx": 3, "depth": 2, "success": False},
    ]
    vt, vm, cm, n_tried, n_win = BUILD._labels_for_node(
        trials,
        reach={1, 2, 3, 4},
        chain_depth=1,
        winning_setups={(1, 0)},
        colossus0=True,
        censored_setups={(2, 1)},
    )

    assert (vt[1, 0], vm[1, 0], cm[1, 0]) == (0.9, 1.0, 0.0)
    assert (vt[2, 1], vm[2, 1], cm[2, 1]) == (0.0, 0.0, 0.0)
    assert np.isclose(vt[3, 2], 0.81)
    assert (vm[3, 2], cm[3, 2]) == (1.0, 1.0)
    assert vm[4, 0] == 0.0
    assert (n_tried, n_win) == (3, 0)


def test_colossus_finish_failures_are_ceiling_not_false_dead():
    trials = [
        {"edge_idx": 5, "depth": 0, "success": False},
        {"edge_idx": 6, "depth": 1, "success": True},
    ]
    vt, vm, cm, n_tried, n_win = BUILD._labels_for_node(
        trials,
        reach={5, 6, 7},
        chain_depth=2,
        winning_setups=set(),
        colossus0=True,
    )

    assert (vt[5, 0], vm[5, 0], cm[5, 0]) == (0.9, 1.0, 1.0)
    assert (vt[6, 1], vm[6, 1], cm[6, 1]) == (1.0, 1.0, 0.0)
    assert vm[7, 0] == 0.0
    assert (n_tried, n_win) == (2, 1)


def test_legacy_labels_remain_exact_zero():
    vt, vm, cm, _, _ = BUILD._labels_for_node(
        [{"edge_idx": 8, "depth": 3, "success": False}],
        reach={8},
        chain_depth=2,
        winning_setups=set(),
    )

    assert (vt[8, 3], vm[8, 3], cm[8, 3]) == (0.0, 1.0, 0.0)
