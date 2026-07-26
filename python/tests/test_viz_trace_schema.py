import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from viz.trace_schema import build_trace, episode_filename, make_board, make_pop  # noqa: E402


def test_episode_filename_uses_stem_and_object_id():
    name = episode_filename("/scratch/x/run_0056/env_0056_pair_001.xml", "obstacle_7_movable")
    assert name == "env_0056_pair_001__obstacle_7_movable.json"


def test_root_board_has_sentinel_parent():
    b = make_board(0, 0, -1, -1, [], None, 1.0, 0)
    assert b["board_id"] == 0 and b["depth"] == 0
    assert b["parent_edge"] == -1 and b["parent_depth"] == -1


def test_child_board_records_the_setup_push_that_spawned_it():
    pool = [{"obj": "o", "edge": 12, "depth": 1, "q": 0.4}]
    b = make_board(3, 1, 54, 2, pool, None, 1.0, 1)
    assert (b["parent_edge"], b["parent_depth"]) == (54, 2)
    assert b["n_candidates"] == 1
    assert b["pool"] == pool


def test_pop_carries_the_effective_priority():
    p = make_pop(7, 3, "o", 12, 1, 0.4, 0.5, 0.2, False)
    assert p["t"] == 7 and p["board_id"] == 3
    assert p["se"] == 0.5 * 0.2
    assert p["opened"] is False


def test_build_trace_is_json_serializable_and_versioned():
    import json
    doc = build_trace(
        meta={"xml": "/x/a.xml", "object_id": "o", "model": "ceiling", "strategy": "off"},
        scene={"bounds": [0, 1, 0, 1], "static": [], "movable": [], "robot": [0, 0, 0],
               "goal": [0.5, 0.5, 0.0], "contacts": []},
        boards=[make_board(0, 0, -1, -1, [], None, 1.0, 0)],
        pops=[make_pop(1, 0, "o", 5, 0, 0.9, 0.9, 1.0, True)],
        result={"solved": True, "sims": 1, "plan_len": 1, "end": "solved"},
    )
    assert doc["schema_version"] == 1
    assert doc["result"]["solved"] is True
    json.dumps(doc)
