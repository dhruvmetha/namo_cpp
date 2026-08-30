"""The region graph must record a doorway plugged by two movables, and must not disturb one-movable scenes.

`build_region_connectivity_graph` used to ask only "would removing THIS ONE object join two regions".
A door held shut by two touching blocks answers no for each block alone, so nothing was written and
the regions read as unconnected. That fed straight through `sample_region_goals`, which walks the
adjacency out from the robot, so the goal region got zero sampled poses and `exhaustive_hmax2.py`
dropped the scene with no log line. 67 of 300 scenes in the v1 two-movable pool went that way.

The movable-blob pass adds edges for clumps of two or more objects. The control test is the other
half: single-movable scenes must come out exactly as they did before, byte for byte, because every
difficulty label in the project rests on them.
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "python", "tests", "data")
CFG = os.path.join(REPO, "config", "namo_config_complete_skill15_car_1x.yaml")

pytest.importorskip("namo_rl")
import namo_rl  # noqa: E402


def _snapshot(xml):
    """Every field the snapshot exposes, poses included.

    Counting sampled poses is not enough for a parity check: a regression that moves all 100 goal
    poses somewhere else keeps the count at 100 and slips through. The coordinates are what the
    success bar is measured against, so they are the thing worth pinning.
    """
    env = namo_rl.RLEnvironment(xml, CFG, False)
    env.get_reachable_objects()
    s = env.get_region_snapshot(100, -1.0, False, 42, True)
    return {
        "region_labels": {str(k): v for k, v in dict(s.get("region_labels", {})).items()},
        "adjacency": {a: sorted(b) for a, b in dict(s.get("adjacency", {})).items()},
        "edge_objects": {a: {c: sorted(d) for c, d in dict(b).items()}
                         for a, b in dict(s.get("edge_objects", {})).items()},
        "multi_object_edges": {a: sorted(b)
                               for a, b in dict(s.get("multi_object_edges", {})).items()},
        "robot_label": s.get("robot_label"),
        "goal_label": s.get("goal_label"),
        "goal_reachable": bool(s.get("goal_reachable")),
        "goal_in_free_space": bool(s.get("goal_in_free_space")),
        "region_goals": {
            k: {"blocking_objects": sorted(v.blocking_objects),
                "goals": [[round(g.x, 9), round(g.y, 9), round(g.theta, 9)] for g in v.goals]}
            for k, v in dict(s.get("region_goals", {})).items()
        },
    }


def test_two_movable_doorway_gets_an_edge_naming_both_blocks():
    snap = _snapshot(os.path.join(DATA, "two_movable_doorway_fixture.xml"))

    assert "goal" in snap["adjacency"].get("robot", []), (
        "robot and goal must be adjacent; an empty adjacency here is the bug this pass fixes"
    )
    assert "robot" in snap["adjacency"].get("goal", [])

    both = {"obstacle_0_movable", "obstacle_1_movable"}
    assert set(snap["edge_objects"]["robot"]["goal"]) == both, (
        "the clump plugs the door, so both blocks are candidates on that edge"
    )
    assert set(snap["edge_objects"]["goal"]["robot"]) == both, "edges must be symmetric"

    # The whole point downstream: with the edge present, the pose sampler reaches the goal region,
    # so exhaustive_hmax2.goal_region_points gets targets instead of None and the scene is labelled.
    assert len(snap["region_goals"].get("goal", {}).get("goals", [])) > 0, (
        "no sampled goal poses means the labeller silently drops this scene"
    )

    # Neither block opens this door alone, so the edge must be marked as needing the whole plug.
    # Both directions: the writer records both, and a reader that only ever looks one way would not
    # notice the reverse going missing.
    assert "goal" in snap["multi_object_edges"].get("robot", []), (
        "a door no single object opens must be distinguishable from one where either object works"
    )
    assert "robot" in snap["multi_object_edges"].get("goal", []), "the marker must be symmetric"


def test_marker_reaches_the_project_python_api():
    """The wrapper in namo.planners rebuilds the snapshot key by key.

    A new binding field stays invisible there until it is named, and this test exists because that
    is exactly how it went missing the first time. Asserting through the raw binding alone would not
    have caught it.
    """
    from namo.planners import get_region_snapshot

    env = namo_rl.RLEnvironment(os.path.join(DATA, "two_movable_doorway_fixture.xml"), CFG, False)
    env.get_reachable_objects()
    snap = get_region_snapshot(env, goals_per_region=100, seed=42, use_xml_goal=True)
    assert "multi_object_edges" in snap, "the wrapper dropped the field"
    assert "goal" in snap["multi_object_edges"].get("robot", set())


def test_single_movable_scene_is_unchanged():
    """Parity gate. If this fails, every existing difficulty label is suspect.

    Semantic parity rather than byte parity: container order is canonicalised and poses are rounded
    to 9 decimals, which is sub-nanometre against a 5 mm grid. Everything the snapshot exposes is
    compared, coordinates included.
    """
    golden = json.load(open(os.path.join(DATA, "single_movable_control_golden.json")))
    assert _snapshot(os.path.join(DATA, "single_movable_control_fixture.xml")) == golden


def _snapshot_with_switch_off(xml):
    """Same binary, blob pass disabled, whole snapshot back as JSON."""
    # Load this module by path rather than as a package: python/tests has no __init__.py, and the
    # subprocess must reuse THIS _snapshot so both sides serialise identically.
    script = (
        "import json,importlib.util as u;"
        "sp=u.spec_from_file_location('t', %r); m=u.module_from_spec(sp); sp.loader.exec_module(m);"
        "print(json.dumps(m._snapshot(%r),sort_keys=True))"
        % (os.path.abspath(__file__), xml)
    )
    env = dict(os.environ, NAMO_DISABLE_MOVABLE_BLOB_EDGES="1")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env=env, cwd=REPO)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip())


def test_blob_pass_changes_nothing_on_the_control_scene():
    """Whole snapshot, switch on versus off, on one binary.

    Proves the parity above comes from the blob pass writing nothing on a one-movable scene, rather
    than from two code paths happening to agree for some other reason.
    """
    xml = os.path.join(DATA, "single_movable_control_fixture.xml")
    assert _snapshot_with_switch_off(xml) == _snapshot(xml)


def test_blob_pass_is_what_creates_the_two_movable_edge():
    """The converse: on the two-movable scene the switch must make a difference.

    Without this the parity tests could all pass on a build where the new pass never runs.
    """
    xml = os.path.join(DATA, "two_movable_doorway_fixture.xml")
    off = _snapshot_with_switch_off(xml)
    assert off["adjacency"].get("robot", []) == [], (
        "the fixture must reproduce the original bug when the pass is disabled"
    )
    assert _snapshot(xml)["adjacency"].get("robot", []) == ["goal"]
