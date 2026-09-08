"""Pure contract tests for the static boundary-group census (no MuJoCo / no pushes)."""
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "scripts" / "pipeline" / "probe_static_topology.py"


@pytest.fixture(scope="module")
def census_module():
    spec = importlib.util.spec_from_file_location("boundary_group_census_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_kind_requires_explicit_multi_object_marker(census_module):
    assert census_module.group_kind_from_snapshot(
        {"multi_object_edges": {}}, "robot", "goal", ["obstacle_0_movable"]
    ) == "singleton"
    assert census_module.group_kind_from_snapshot(
        {"multi_object_edges": {}}, "robot", "goal",
        ["obstacle_0_movable", "obstacle_1_movable"],
    ) == "alternatives"
    assert census_module.group_kind_from_snapshot(
        {"multi_object_edges": {"robot": {"goal"}, "goal": {"robot"}}},
        "robot", "goal", ["obstacle_0_movable", "obstacle_1_movable"],
    ) == "joint_blockage"
    with pytest.raises(AssertionError, match="multi_object_edges"):
        census_module.group_kind_from_snapshot({}, "robot", "goal", ["a", "b"])


def test_manifest_sources_keep_sibling_episode_identities(tmp_path, census_module, monkeypatch):
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    divisions = tmp_path / "divisions.json"
    room = str(tmp_path / "room.xml")
    legacy_room = "/legacy/room.xml"
    monkeypatch.setattr(census_module, "resolve_namo_path",
                        lambda path: room if path == legacy_room else path)
    one.write_text(json.dumps({legacy_room: [
        {"object_id": "obstacle_a_movable", "region": "goal", "object_center": [0.1, 0.2],
         "solve_rate": 0.025, "tried": [[0, 0]] * 40, "valid": [[0, 0]]},
        {"object_id": "obstacle_b_movable", "region": "goal", "object_center": [0.3, 0.4],
         "solve_rate": 0.4, "tried": [[0, 0]] * 10, "valid": [[0, 0]] * 4},
    ]}))
    two.write_text(json.dumps({legacy_room: [
        {"object_id": "obstacle_a_movable", "region": "goal", "object_center": [0.1, 0.2],
         "valid_first_push": [[0, 0], [2, 0], [4, 0]]},
    ]}))
    # A legacy-looking manifest key and current path must meet through resolve+realpath, not by
    # basename. The supplied division is the sole source of the 2-push tier.
    divisions.write_text(json.dumps({room: [
        {"object_id": "obstacle_a_movable", "region": "goal", "division": "medium"},
    ]}))

    by_room = census_module.load_manifest_sources(str(one), str(two), str(divisions))
    sources = by_room[str((tmp_path / "room.xml").resolve())]
    assert [(s["leg"], s["object_id"], s["tier"]) for s in sources] == [
        ("1push", "obstacle_a_movable", "hard"),
        ("1push", "obstacle_b_movable", "easy"),
        ("2push", "obstacle_a_movable", "medium"),
    ]
    assert sources[0]["object_center"] == [0.1, 0.2]


def test_off_boundary_source_is_explicitly_excluded(census_module, monkeypatch):
    class Goal:
        x = 0.123456789
        y = -0.987654321

    class Bundle:
        goals = [Goal()]

    class FakeEnv:
        def set_robot_goal(self, *_goal):
            pass

        def get_reachable_objects(self):
            return ["obstacle_a_movable"]

        def get_reachable_edges(self, _obj):
            return [object()]

        def count_reachable_points(self, _points):
            return 0, -1

        def get_observation(self):
            return {"robot_pose": [0, 0, 0]}

    snapshot = {
        "robot_label": "robot", "goal_label": "goal", "goal_reachable": False,
        "adjacency": {"robot": {"goal"}},
        "edge_objects": {"robot": {"goal": {"obstacle_a_movable", "obstacle_c_movable"}}},
        "multi_object_edges": {"robot": set(), "goal": set()},
        "region_goals": {"goal": Bundle()},
    }
    monkeypatch.setattr(census_module.namo_rl, "RLEnvironment", lambda *_args: FakeEnv())
    monkeypatch.setattr(census_module, "get_region_snapshot", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(census_module, "extract_goal_from_xml", lambda _room: (0.0, 0.0))
    rows = census_module.census_room(("/room.xml", [{
        "leg": "1push", "xml_path": "/room.xml", "room_realpath": "/room.xml",
        "object_id": "obstacle_b_movable", "region": "goal", "tier": "hard", "object_center": [0, 0],
    }], "unused.yaml"))
    assert len(rows) == 1
    assert rows[0]["exclusion_reason"] == "source_object_not_on_boundary"
    assert rows[0]["boundary_objects"] == ["obstacle_a_movable", "obstacle_c_movable"]
    assert rows[0]["target_points"] == [[0.123456789, -0.987654321]]

    monkeypatch.setattr(FakeEnv, "count_reachable_points", lambda *_args: (1, 0))
    rows = census_module.census_room(("/room.xml", [{
        "leg": "1push", "xml_path": "/room.xml", "room_realpath": "/room.xml",
        "object_id": "obstacle_a_movable", "region": "goal", "tier": "hard", "object_center": [0, 0],
    }], "unused.yaml"))
    assert rows[0]["eligible"] is False
    assert rows[0]["initial_target_reachable_fraction"] == 1.0
    assert rows[0]["exclusion_reason"] == "initial_target_fraction_at_least_0_2"


def test_pilot_is_bounded_deterministic_and_source_stratified(census_module):
    rows = [
        {"group_id": "bg_c", "eligible": True, "source_strata": ["1push:hard"]},
        {"group_id": "bg_b", "eligible": True, "source_strata": ["1push:hard"]},
        {"group_id": "bg_a", "eligible": True, "source_strata": ["2push:medium"]},
        {"group_id": "excluded", "eligible": False, "source_strata": ["1push:easy"],
         "exclusion_reason": "singleton_boundary", "group_kind": "singleton"},
    ]
    assert census_module.select_pilot_groups(rows, limit=24) == ["bg_b", "bg_a", "bg_c"]
    summary = census_module.summarize_census(rows, "one.json", "two.json", "divisions.json")
    assert summary["n_eligible_multi_object_groups"] == 3
    assert summary["pilot_group_ids"] == ["bg_b", "bg_a", "bg_c"]
    assert "source strata are provenance" in summary["pilot_selection"]
    # Excluded rooms can have no boundary classification. All summary map keys must
    # remain strings so the production writer can sort them alongside real kinds.
    encoded = json.dumps(summary, sort_keys=True)
    assert json.loads(encoded)["group_kind_counts"]["unclassified"] == 3


def test_pilot_keeps_rare_joint_boundaries(census_module):
    rows = [{"group_id": f"bg_{i:02}", "eligible": True,
             "source_strata": ["1push:easy"], "group_kind": "alternatives"}
            for i in range(30)]
    rows.append({"group_id": "bg_rare_joint", "eligible": True,
                 "source_strata": ["1push:easy"], "group_kind": "joint_blockage"})
    selected = census_module.select_pilot_groups(rows, limit=24)
    assert len(selected) == 24
    assert "bg_rare_joint" in selected
    assert selected == census_module.select_pilot_groups(list(reversed(rows)), limit=24)


def test_census_requires_the_config_sibling_5mm_margin(tmp_path, census_module):
    config = tmp_path / "margin_5mm" / "namo.yaml"
    config.parent.mkdir()
    config.write_text("planning: {}\n")
    sidecar = config.parent / "wavefront_inflation.yaml"
    sidecar.write_text("tier1:\n  base_inflation_margin_m: 0.005\n")
    census_module._require_census_margin(str(config))

    sidecar.write_text("tier1:\n  base_inflation_margin_m: 0.001\n")
    with pytest.raises(ValueError, match="5 mm"):
        census_module._require_census_margin(str(config))
