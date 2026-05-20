"""Tests for rollout_trace_loader."""

import pickle
import tempfile
from pathlib import Path

import numpy as np


def test_load_attempts_from_pkl_returns_one_record_per_object_neighbor():
    """Loader extracts one record per episode_results entry, surfacing the F-grid."""
    from namo.data_collection.rollout_trace_loader import load_attempts_from_pkl

    # Build a fake pkl matching the worker's output format.
    fake_pkl = {
        "task_id": "rlab7_env_000000",
        "success": True,
        "episodes_collected": 2,
        "episode_results": [
            {
                "episode_id": "ep_0",
                "algorithm": "uniform_rollout_sampler",
                "algorithm_version": "0.1.0",
                "success": True,
                "solution_found": True,
                "solution_depth": 1,
                "xml_file": "/tmp/env.xml",
                "robot_goal": (0.0, 0.0, 0.0),
                "algorithm_stats": {
                    "chosen_object_id": "obj_1",
                    "neighbour_region_label": "neighbor_A",
                    "primitive_trial_log": [
                        {"edge_idx": 0, "depth": 0, "success": True,
                         "wall_collision": False, "movable_collisions": "",
                         "stuck": False, "collision": False, "reachable_after": 1},
                        {"edge_idx": 0, "depth": 1, "success": False,
                         "wall_collision": False, "movable_collisions": "",
                         "stuck": False, "collision": False, "reachable_after": 0},
                    ],
                    "region_goals_sampled": [(0.1, 0.0, 0.0)],
                },
            },
            {
                "episode_id": "ep_1",
                "algorithm": "uniform_rollout_sampler",
                "algorithm_version": "0.1.0",
                "success": False,
                "solution_found": False,
                "solution_depth": 0,
                "xml_file": "/tmp/env.xml",
                "robot_goal": (0.0, 0.0, 0.0),
                "algorithm_stats": {
                    "chosen_object_id": "obj_1",
                    "neighbour_region_label": "neighbor_B",
                    "primitive_trial_log": [
                        {"edge_idx": 0, "depth": 0, "success": False,
                         "wall_collision": True, "movable_collisions": "",
                         "stuck": False, "collision": True, "reachable_after": 0},
                    ],
                    "region_goals_sampled": [(0.0, 0.1, 0.0)],
                },
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = Path(tmp) / "fake.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(fake_pkl, f)

        records = load_attempts_from_pkl(str(pkl_path))

    assert len(records) == 2
    rec0 = records[0]
    assert rec0["xml_file"] == "/tmp/env.xml"
    assert rec0["object_id"] == "obj_1"
    assert rec0["neighbor"] == "neighbor_A"
    assert rec0["F"] == 1
    assert rec0["R"] == 2
    assert rec0["f_ratio"] == 0.5
    # f_grid is (60, 10) with NaN for unevaluated cells, 1.0 for success, 0.0 for fail
    assert rec0["f_grid"].shape == (60, 10)
    assert rec0["f_grid"][0, 0] == 1.0
    assert rec0["f_grid"][0, 1] == 0.0
    assert np.isnan(rec0["f_grid"][1, 0])      # unevaluated

    rec1 = records[1]
    assert rec1["F"] == 0
    assert rec1["R"] == 1
