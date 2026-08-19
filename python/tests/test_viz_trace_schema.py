import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from viz.trace_schema import (build_trace, episode_filename, make_board, make_pop,  # noqa: E402
                              rle_decode, rle_encode)


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


def test_board_geometry_is_optional_and_defaults_to_none():
    """v3 is ADDITIVE: a caller that does not record geometry still gets the keys, set to None, so
    the page can branch on presence instead of on schema_version arithmetic."""
    b = make_board(0, 0, -1, -1, [], None, 1.0, 0)
    assert b["geom"] is None and b["regions"] is None


def test_board_carries_per_state_geometry_and_regions():
    geom = {"movable": {"obj_1": [0.1, 0.2, 0.3]}, "robot": [0.0, 0.0, 1.0],
            "contacts": [[0.0, 0.0]] * 60}
    regions = {"nx": 2, "ny": 3, "res": 0.005, "origin": [-0.4, -0.4],
               "labels": {"1": "robot", "2": "goal"}, "rle": [1, 3, 0, 1, 2, 2]}
    b = make_board(1, 1, 5, 0, [], None, 1.0, 0, geom=geom, regions=regions)
    assert b["geom"]["robot"] == [0.0, 0.0, 1.0]
    assert len(b["geom"]["contacts"]) == 60
    assert b["regions"]["labels"]["2"] == "goal"


def test_rle_roundtrips_and_never_crosses_a_row():
    grid = [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [2, 0, 2, 0, 2]]
    flat = rle_encode(grid)
    # runs are per row: row 0 -> (0,2),(1,3); row 1 -> (1,5); row 2 -> five singletons.
    assert flat == [0, 2, 1, 3, 1, 5, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1]
    assert sum(flat[1::2]) == 3 * 5
    assert rle_decode(flat, 3, 5) == grid


def test_rle_roundtrips_a_uniform_grid():
    grid = [[7] * 4 for _ in range(3)]
    assert rle_encode(grid) == [7, 4, 7, 4, 7, 4]     # one run per row, not one run overall
    assert rle_decode(rle_encode(grid), 3, 4) == grid


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


def test_pop_outcome_geometry_is_optional_and_defaults_to_none():
    """v4 is ADDITIVE the same way v3 was: a caller that records no geometry still gets the keys, set
    to None, so the page branches on presence rather than on schema_version arithmetic."""
    p = make_pop(7, 3, "o", 12, 1, 0.4, 0.5, 0.2, False)
    assert p["geom"] is None and p["regions"] is None


def test_pop_carries_the_state_its_push_reached_in_the_board_shape():
    """v4: the outcome of EVERY pop, in exactly make_board's geometry shape -- one format, one decoder."""
    geom = {"movable": {"obj_1": [0.4, 0.2, 0.3]}, "robot": [0.1, 0.0, 1.0],
            "contacts": [[0.0, 0.0]] * 60}
    regions = {"nx": 2, "ny": 3, "res": 0.005, "origin": [-0.4, -0.4],
               "labels": {"1": "robot_goal"}, "rle": [1, 3, 0, 1, 1, 2]}
    p = make_pop(7, 3, "o", 12, 1, 0.4, 0.5, 0.2, True, geom=geom, regions=regions)
    assert p["geom"]["movable"]["obj_1"] == [0.4, 0.2, 0.3]
    assert len(p["geom"]["contacts"]) == 60
    assert p["regions"]["labels"]["1"] == "robot_goal"
    assert set(make_board(0, 0, -1, -1, [], None, 1.0, 0, geom=geom, regions=regions)) >= {"geom", "regions"}


def test_pop_keeps_every_pre_v4_field_unchanged():
    """Purely additive: the v3 key set must survive verbatim, since the page and every analysis script
    read these by name."""
    p = make_pop(7, 3, "o", 12, 1, 0.4, 0.5, 0.2, False)
    assert set(p) == {"t", "board_id", "obj", "edge", "depth", "q", "bp", "w", "se", "opened",
                      "geom", "regions"}


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
    assert doc["schema_version"] == 4
    assert doc["result"]["solved"] is True
    json.dumps(doc)


def test_meta_carries_every_order_affecting_search_parameter():
    """v2 contract: the viz can only reproduce the queue order if meta records the knobs that set it --
    the priority formula (combine/agg/prior/raw/dive_bonus) and the w demotion (discount/gamma/tau/eps/
    w0_mode/free_strike_q/gtable), plus the search bounds. Missing one = a silently mis-ordered page."""
    doc = build_trace(meta={"xml": "/x/a.xml", "object_id": "o", "search": SEARCH_PARAMS},
                      scene={}, boards=[], pops=[], result={})
    required = {"hmax", "sim_budget", "prior", "agg", "combine", "discount", "gamma", "tau", "eps",
                "w0_mode", "free_strike_q", "dive_bonus", "raw", "dedupe_noop", "prune_jam_depth", "gtable"}
    assert required <= set(doc["meta"]["search"])


def test_generator_writes_the_full_parameter_set():
    """Guard the writer, not just the schema: eval_bestfirst.py's search_params must stay in sync with the
    argparse flags that affect ordering (read as source -- importing it pulls in the simulator bindings)."""
    src = (REPO_ROOT / "scripts/sandbox/eval_bestfirst.py").read_text()
    block = src.split("search_params = {", 1)[1].split("}\n", 1)[0]
    for flag in ("hmax", "sim_budget", "prior", "agg", "combine", "discount", "gamma", "tau", "eps",
                 "w0_mode", "free_strike_q", "dive_bonus", "raw", "dedupe_noop", "prune_jam_depth", "gtable"):
        assert f'"{flag}"' in block, flag
    assert '"search": search_params' in src


def test_generator_defaults_to_and_records_dedupe_and_jam_pruning():
    src = (REPO_ROOT / "scripts/sandbox/eval_bestfirst.py").read_text()
    assert "ap.set_defaults(dedupe_noop=True)" in src
    assert "ap.set_defaults(prune_jam_depth=True)" in src
    assert '"dedupe_noop": bool(a.dedupe_noop)' in src
    assert '"prune_jam_depth": bool(a.prune_jam_depth)' in src


def test_generator_records_geometry_per_board_and_only_under_trace_out():
    """v3 contract on the writer: boards are built with geom/regions, the capture is created only on
    the --trace-out path (so the flag-off run stays byte-identical), and it restores the sim state
    before reading, since the scorer moves it."""
    src = (REPO_ROOT / "scripts/sandbox/eval_bestfirst.py").read_text()
    assert 'geom=b["geom"], regions=b["regions"]' in src
    body = src.split("def _make_capture(", 1)[1].split("\ndef main(", 1)[0]
    assert body.count("env.set_full_state(state)") == 2      # restore on entry AND after the snapshot
    assert "rle_encode(rm.tolist())" in body
    assert "capture = _make_capture(" in src.split("if a.trace_out:", 2)[-1]


def test_generator_captures_every_pop_before_the_early_return_on_success():
    """v4 contract on the writer, and the whole point of the change: the post-push state is read for EVERY
    pop -- while the sim still stands in it, and ABOVE the `if opened: return` -- so the winning push of a
    solved episode (which spawns no board, so v3 recorded nothing) is captured too."""
    # The search loop moved into the package; eval_bestfirst.py keeps only the
    # CLI, the capture helpers and the reporting that the sibling tests read.
    src = (REPO_ROOT / "python/namo/planners/opening/best_first_search.py").read_text()
    loop = src.split("opened = bool(is_open(env))", 1)[1].split("if ndone < hmax:", 1)[0]
    cap = loop.index("pop_geom, pop_regions = capture(s_after)")
    assert cap < loop.index("geom=pop_geom, regions=pop_regions")   # captured before it is written to the pop
    assert cap < loop.index('if opened:')                           # ... and before the search returns
    assert "s_new = env.get_full_state() if s_after is None else s_after" in src
