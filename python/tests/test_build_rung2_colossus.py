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


def test_action_contract_requires_aligned_live_300_slots_and_provenance():
    node = {
        "action_motion": np.zeros((60, 5, 3), np.float32),
        "action_generator_slot_count": 300,
        "action_motion_frame": "world_xy_object_yaw",
        "action_motion_units": "normalized",
        "action_motion_normalization": np.array([0.5, 0.5, np.pi], np.float32),
        "target_object_state": np.array([1.0, 2.0, 0.1, 0.2, 0.3], np.float32),
        "state_observation": {"obj_pose": [1.0, 2.0, 0.1]},
        "primitive_database_id": "db.dat",
        "primitive_database_sha256": "a" * 64,
        "shape_family": "square",
    }
    grid = np.zeros((60, 5), np.float32)

    artifact = BUILD._action_contract(node, "obj", grid, grid, grid)

    assert artifact["action_motion"].shape == (60, 5, 3)
    bad = dict(node, action_generator_slot_count=299)
    try:
        BUILD._action_contract(bad, "obj", grid)
    except AssertionError:
        pass
    else:
        raise AssertionError("299-slot live generator must fail the build")
