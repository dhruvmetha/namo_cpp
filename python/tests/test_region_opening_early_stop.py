import time

import pytest

from namo.core.base_planner import PlannerConfig
from namo.planners.opening.region_opening import RegionOpeningPlanner
from namo.strategies.goal_selection_strategy import Goal


class _DummyEnv:
    def __init__(self, reachable_objects):
        self._reachable_objects = list(reachable_objects)

    def get_reachable_objects(self):
        return self._reachable_objects

    def set_full_state(self, _state):
        return None


def test_attempt_opening_stops_after_max_solutions(monkeypatch):
    # Avoid strategy initialization (primitive DB, ML models, etc.).
    monkeypatch.setattr(RegionOpeningPlanner, "_setup_constraints", lambda self: None)
    monkeypatch.setattr(RegionOpeningPlanner, "_initialize_algorithm", lambda self: setattr(self, "goal_strategy", object()))

    env = _DummyEnv(reachable_objects=["obj1", "obj2"])
    config = PlannerConfig(
        verbose=False,
        algorithm_params={
            "region_max_solutions_per_neighbor": 1,
            "region_stop_after_max_solutions": True,
        },
    )
    planner = RegionOpeningPlanner(env, config)

    # Skip reachability validation details for this unit test.
    monkeypatch.setattr(planner, "_validate_opening", lambda *_args, **_kwargs: (False, 0, None, []))

    calls = []

    def _fake_search(object_id, *_args, **_kwargs):
        calls.append(object_id)
        goal = Goal(x=0.0, y=0.0, theta=0.0, score=1.0, edge_idx=0, depth=0)
        successful_goals = [
            (
                [goal],          # goal_chain
                [],              # state_obs
                [],              # post_state_obs
                None,            # resulting_state
                (1.0, 2.0, 0.0), # region_goal_used
                [(1.0, 2.0, 0.0)],  # region_goals_sampled
                None,            # reachable_before
                None,            # reachable_after
                1,               # total_cost
                0,               # skill_calls_before_success
                time.time(),     # success_timestamp
                False,           # any_wall_collision
                0,               # unique_movable_collision_count
            )
        ]
        return successful_goals, 0, {"ML-only": 1}, "ML-only", False, 0

    monkeypatch.setattr(planner, "_search_with_chaining_bfs", _fake_search)

    attempts = planner._attempt_opening_to_neighbour(
        robot_label="region_0",
        neighbour_label="region_1",
        adjacency={"region_0": {"region_1"}},
        edge_objects={"region_0": {"region_1": {"obj1", "obj2"}}},
        region_goals={},
        max_solutions=1,
        exploration_state=object(),
        exploration_level=0,
    )

    # Should stop after the first success instead of searching other objects.
    assert len(calls) == 1
    assert any(a.success for a in attempts)


def test_explore_stops_after_first_neighbour_success(monkeypatch):
    # Avoid strategy initialization (primitive DB, ML models, etc.).
    monkeypatch.setattr(RegionOpeningPlanner, "_setup_constraints", lambda self: None)
    monkeypatch.setattr(RegionOpeningPlanner, "_initialize_algorithm", lambda self: setattr(self, "goal_strategy", object()))

    env = _DummyEnv(reachable_objects=["obj1"])
    config = PlannerConfig(
        verbose=False,
        algorithm_params={
            "region_stop_after_first_success": True,
        },
    )
    planner = RegionOpeningPlanner(env, config)

    # Snapshot generation is heavy; stub it out at the module level.
    import namo.planners.opening.region_opening as region_opening_mod

    def _fake_snapshot(*_args, **_kwargs):
        adjacency = {"region_0": {"region_1", "region_2"}}
        edge_objects = {"region_0": {"region_1": {"obj1"}, "region_2": {"obj1"}}}
        region_labels = {"robot": "region_0"}
        region_goals = {}
        return adjacency, edge_objects, region_labels, region_goals, None

    monkeypatch.setattr(region_opening_mod, "snapshot_region_connectivity", _fake_snapshot)
    monkeypatch.setattr(region_opening_mod, "find_robot_label", lambda _labels: "region_0")

    # Only the first neighbour returns a success.
    calls = []

    def _fake_attempt(robot_label, neighbour_label, *_args, **_kwargs):
        calls.append(neighbour_label)
        return [
            region_opening_mod.AttemptResult(
                success=(neighbour_label == "region_1"),
                neighbour_region_label=neighbour_label,
                failure_reason="success" if neighbour_label == "region_1" else "no_solution",
                timing_ms=0.0,
                candidate_objects_count=1,
            )
        ]

    monkeypatch.setattr(planner, "_attempt_opening_to_neighbour", _fake_attempt)
    monkeypatch.setattr(env, "get_xml_path", lambda: "dummy.xml", raising=False)
    monkeypatch.setattr(env, "get_config_path", lambda: "dummy.yaml", raising=False)

    attempts = planner._explore_from_state(state=object(), level=0, target_neighbor=None)

    assert calls == ["region_1"]
    assert any(a.success for a in attempts)
