import copy
import sys
import types
from types import SimpleNamespace


if "namo_rl" not in sys.modules:
    namo_rl_stub = types.ModuleType("namo_rl")
    namo_rl_stub.Action = type("Action", (), {})
    namo_rl_stub.RLEnvironment = object
    namo_rl_stub.RLState = object
    sys.modules["namo_rl"] = namo_rl_stub

from namo.core import PlannerConfig
from namo.planners.opening.region_opening import AttemptResult, RegionOpeningPlanner


class FakeEnv:
    def __init__(self):
        self.state = {"name": "baseline", "values": [0]}
        self.collision_checking_calls = []

    def get_full_state(self):
        return copy.deepcopy(self.state)

    def set_full_state(self, state):
        self.state = copy.deepcopy(state)

    def set_collision_checking(self, enabled):
        self.collision_checking_calls.append(enabled)

    def get_xml_path(self):
        return "dummy.xml"

    def get_config_path(self):
        return "dummy.yaml"

    def get_observation(self):
        return {}

    def get_reachable_objects(self):
        return []


def make_planner(monkeypatch, env):
    def fake_initialize(self):
        self.goal_strategy = SimpleNamespace()

    monkeypatch.setattr(RegionOpeningPlanner, "_initialize_algorithm", fake_initialize)
    return RegionOpeningPlanner(env, PlannerConfig())


def test_get_boundary_objects_is_symmetric_and_order_insensitive(monkeypatch):
    planner = make_planner(monkeypatch, FakeEnv())
    edge_objects = {
        "robot": {"a": ["box_b", "box_a"]},
        "a": {"robot": ["box_a", "box_b"]},
    }

    objects, error = planner._get_boundary_objects(edge_objects, "robot", "a")

    assert error is None
    assert objects == ["box_a", "box_b"]


def test_get_boundary_objects_detects_inconsistency(monkeypatch):
    planner = make_planner(monkeypatch, FakeEnv())
    edge_objects = {
        "robot": {"a": ["box_a"]},
        "a": {"robot": ["box_b"]},
    }

    objects, error = planner._get_boundary_objects(edge_objects, "robot", "a")

    assert objects is None
    assert error == "boundary_object_map_inconsistent"


def test_search_restores_baseline_and_preserves_resulting_state(monkeypatch):
    env = FakeEnv()
    planner = make_planner(monkeypatch, env)

    def fake_explore(self, state, level=0, target_neighbor=None):
        self._last_explore_context = {
            "local_robot_label": "robot",
            "local_neighbors": ["a"],
            "target_neighbor": target_neighbor,
            "target_is_immediate_neighbor": True,
        }
        self.env.state = {"name": "opened", "values": [1, 2, 3]}
        return [
            AttemptResult(
                success=True,
                neighbour_region_label=target_neighbor or "a",
                chosen_object_id="box",
                chosen_goal=(1.0, 2.0, 0.0),
                resulting_state=self.env.get_full_state(),
                failure_reason="success",
            )
        ]

    monkeypatch.setattr(RegionOpeningPlanner, "_explore_from_state", fake_explore)

    result = planner.search((0.0, 0.0, 0.0), target_neighbor="a")

    assert result.success is True
    assert env.state == {"name": "baseline", "values": [0]}
    attempt = result.algorithm_stats["attempt_results"][0]
    assert attempt.resulting_state == {"name": "opened", "values": [1, 2, 3]}
    assert result.algorithm_stats["target_summary"]["failure_reason"] == "success"
    assert result.algorithm_stats["target_summary"]["boundary_exhausted"] is False


def test_targeted_non_neighbor_returns_explicit_failure_reason(monkeypatch):
    env = FakeEnv()
    planner = make_planner(monkeypatch, env)

    monkeypatch.setattr(
        "namo.planners.get_region_snapshot",
        lambda *args, **kwargs: {
            "adjacency": {"robot": {"a"}, "a": {"robot"}},
            "edge_objects": {},
            "region_labels": {1: "robot", 2: "a"},
            "region_goals": {},
            "robot_label": "robot",
            "goal_label": "",
            "goal_in_free_space": False,
        },
    )

    result = planner.search((0.0, 0.0, 0.0), target_neighbor="b")

    assert result.success is False
    target_summary = result.algorithm_stats["target_summary"]
    assert target_summary["failure_reason"] == "target_not_immediate_neighbor"
    assert target_summary["local_neighbors"] == ["a"]
    assert target_summary["target_is_immediate_neighbor"] is False
    assert env.state == {"name": "baseline", "values": [0]}


def test_target_summary_boundary_exhaustion_is_conservative(monkeypatch):
    planner = make_planner(monkeypatch, FakeEnv())
    planner._last_explore_context = {
        "local_robot_label": "robot",
        "local_neighbors": ["a"],
        "target_neighbor": "a",
        "target_is_immediate_neighbor": True,
    }

    planner.attempt_results = [
        AttemptResult(success=False, neighbour_region_label="a", failure_reason="no_reachable_objects"),
        AttemptResult(success=False, neighbour_region_label="a", failure_reason="all_pushes_failed"),
    ]
    exhausted = planner._build_target_summary("a")
    assert exhausted["boundary_exhausted"] is True

    planner.attempt_results = [
        AttemptResult(success=False, neighbour_region_label="a", failure_reason="no_reachable_objects"),
        AttemptResult(success=False, neighbour_region_label="a", failure_reason="timeout"),
    ]
    not_exhausted = planner._build_target_summary("a")
    assert not_exhausted["boundary_exhausted"] is False
