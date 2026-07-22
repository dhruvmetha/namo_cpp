import json
import pickle
import subprocess
import sys
from pathlib import Path


def test_capped_finish_miss_stays_censored_not_dead(tmp_path):
    pkl_root = tmp_path / "pkls"
    pkl_root.mkdir()
    xml_path = tmp_path / "room.xml"
    xml_path.write_text("<mujoco/>")
    payload = {
        "episode_results": [
            {
                "xml_file": str(xml_path),
                "state_observations": [{"obj_pose": [1.0, 2.0, 0.0]}],
                "algorithm_stats": {
                    "chosen_object_id": "obj",
                    "neighbour_region_label": "goal",
                    "primitive_trial_log": [
                        {"chain_depth": 1, "edge_idx": 0, "depth": 0, "success": False},
                        {
                            "chain_depth": 2,
                            "parent_edge": 0,
                            "parent_depth": 0,
                            "edge_idx": 1,
                            "depth": 0,
                            "success": False,
                            "finish_sweep_censored": True,
                        },
                    ],
                },
            }
        ]
    }
    with (pkl_root / "room_results.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    out = tmp_path / "answer.json"
    script = Path(__file__).parents[2] / "scripts" / "pipeline" / "build_2push_validset.py"
    subprocess.run(
        [sys.executable, str(script), "--pkls-root", str(pkl_root), "--out", str(out), "--workers", "1"],
        check=True,
    )

    episode = next(iter(json.loads(out.read_text()).values()))[0]
    assert episode["depth2_censored"] is True
    assert episode["censored_first_push"] == [[0, 0]]
    assert episode["tried_first_push"] == []
    assert episode["is_dead_within_2push"] is False
