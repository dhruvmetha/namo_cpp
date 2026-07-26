import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from viz.trace_schema import build_trace, episode_filename, make_board, make_pop  # noqa: E402


def test_episode_filename_uses_stem_and_object_id():
    xml_path = "/scratch/x/run_0056/env_0056_pair_001.xml"
    digest = hashlib.sha1(os.path.realpath(xml_path).encode()).hexdigest()[:8]
    name = episode_filename(xml_path, "obstacle_7_movable")
    assert name == f"env_0056_pair_001__obstacle_7_movable__{digest}.json"


def test_episode_filename_disambiguates_same_basename_across_dirs():
    name_a = episode_filename("/scratch/x/run_0056/env_0056_pair_001.xml", "obstacle_7_movable")
    name_b = episode_filename("/scratch/y/run_0099/env_0056_pair_001.xml", "obstacle_7_movable")
    assert name_a != name_b


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


SEARCH_PARAMS = {"hmax": 2, "sim_budget": 30, "prior": "model", "agg": "mean5", "combine": "blend",
                 "discount": "conf", "gamma": 0.65, "tau": 0.15, "eps": 1e-3, "w0_mode": "one",
                 "free_strike_q": 2.0, "dive_bonus": 0.0, "raw": False, "gtable": None}


def test_build_trace_is_json_serializable_and_versioned():
    import json
    doc = build_trace(
        meta={"xml": "/x/a.xml", "object_id": "o", "model": "ceiling", "strategy": "off",
              "search": SEARCH_PARAMS},
        scene={"bounds": [0, 1, 0, 1], "static": [], "movable": [], "robot": [0, 0, 0],
               "goal": [0.5, 0.5, 0.0], "contacts": []},
        boards=[make_board(0, 0, -1, -1, [], None, 1.0, 0)],
        pops=[make_pop(1, 0, "o", 5, 0, 0.9, 0.9, 1.0, True)],
        result={"solved": True, "sims": 1, "plan_len": 1, "end": "solved"},
    )
    assert doc["schema_version"] == 2
    assert doc["result"]["solved"] is True
    json.dumps(doc)


def test_meta_carries_every_order_affecting_search_parameter():
    """v2 contract: the viz can only reproduce the queue order if meta records the knobs that set it --
    the priority formula (combine/agg/prior/raw/dive_bonus) and the w demotion (discount/gamma/tau/eps/
    w0_mode/free_strike_q/gtable), plus the search bounds. Missing one = a silently mis-ordered page."""
    doc = build_trace(meta={"xml": "/x/a.xml", "object_id": "o", "search": SEARCH_PARAMS},
                      scene={}, boards=[], pops=[], result={})
    required = {"hmax", "sim_budget", "prior", "agg", "combine", "discount", "gamma", "tau", "eps",
                "w0_mode", "free_strike_q", "dive_bonus", "raw", "gtable"}
    assert required <= set(doc["meta"]["search"])


def test_generator_writes_the_full_parameter_set():
    """Guard the writer, not just the schema: eval_bestfirst.py's search_params must stay in sync with the
    argparse flags that affect ordering (read as source -- importing it pulls in the simulator bindings)."""
    src = (REPO_ROOT / "scripts/sandbox/eval_bestfirst.py").read_text()
    block = src.split("search_params = {", 1)[1].split("}\n", 1)[0]
    for flag in ("hmax", "sim_budget", "prior", "agg", "combine", "discount", "gamma", "tau", "eps",
                 "w0_mode", "free_strike_q", "dive_bonus", "raw", "gtable"):
        assert f'"{flag}"' in block, flag
    assert '"search": search_params' in src
