import time
from types import SimpleNamespace

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
            "region_stop_after_root_opener": True,
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
        return successful_goals, 0, {"all": 1}, "all", False, 0, [], []

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
    assert attempts[0].root_opener_rejected is True


def test_root_opener_rejection_skips_depth_two_and_replay(monkeypatch):
    monkeypatch.setattr(RegionOpeningPlanner, "_setup_constraints", lambda self: None)
    monkeypatch.setattr(
        RegionOpeningPlanner,
        "_initialize_algorithm",
        lambda self: setattr(self, "goal_strategy", SimpleNamespace()),
    )

    class _SearchEnv(_DummyEnv):
        def __init__(self):
            super().__init__(["obj1"])
            self.step_calls = 0

        def get_observation(self):
            return {}

        def get_reachable_edges(self, _object_id):
            return [0]

        def step(self, _action):
            self.step_calls += 1
            raise AssertionError("rejection mode must not replay the verified root opener")

    env = _SearchEnv()
    planner = RegionOpeningPlanner(
        env,
        PlannerConfig(
            verbose=False,
            algorithm_params={
                "region_max_chain_depth": 2,
                "region_exhaustive_mode": True,
                "region_label_mode": True,
                "region_stop_after_root_opener": True,
                "region_selection_strategy": "ml_first",
            },
        ),
    )
    goal = Goal(x=0.0, y=0.0, theta=0.0, score=1.0, edge_idx=0, depth=0)
    planner.goal_strategy.generate_goals = lambda *_args, **_kwargs: [[goal]]
    searched_depths = []

    def _fake_bfs(*_args, **kwargs):
        searched_depths.append(kwargs["current_chain_depth"])
        success_node = SimpleNamespace(step_cost=1, skill_calls_before_success=1)
        success = (goal, [{}], [{}], object(), None, [], success_node, time.time())
        trial = {"chain_depth": 1, "edge_idx": 0, "depth": 0, "success": True}
        return [success], 1, [], False, set(), [trial]

    monkeypatch.setattr(planner, "_search_bfs", _fake_bfs)

    result = planner._search_with_chaining_bfs(
        "obj1",
        object(),
        "region_1",
        {},
    )

    assert searched_depths == [1]
    assert env.step_calls == 0
    assert result[1] == 1
    assert result[6][0]["success"] is True


def test_finish_miss_audit_assignment_is_stable_per_episode(monkeypatch):
    monkeypatch.setattr(RegionOpeningPlanner, "_setup_constraints", lambda self: None)
    monkeypatch.setattr(
        RegionOpeningPlanner,
        "_initialize_algorithm",
        lambda self: setattr(self, "goal_strategy", SimpleNamespace()),
    )
    env = _DummyEnv(["obj1"])
    planner = RegionOpeningPlanner(
        env,
        PlannerConfig(
            algorithm_params={
                "xml_file": "/tmp/room.xml",
                "region_finish_topk_cap": 20,
                "region_finish_miss_audit_fraction": 1.0,
                "region_finish_miss_audit_seed": 42,
            }
        ),
    )

    assert planner._select_finish_miss_audit("obj1", "goal") is True
    assert planner._select_finish_miss_audit("obj1", "goal") is True
    planner.finish_miss_audit_fraction = 0.0
    assert planner._select_finish_miss_audit("obj1", "goal") is False


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

    # Snapshot generation is heavy; stub it out at the planner API boundary.
    import namo.planners
    import namo.planners.opening.region_opening as region_opening_mod

    def _fake_snapshot(*_args, **_kwargs):
        return {
            "adjacency": {"region_0": {"region_1", "region_2"}},
            "edge_objects": {"region_0": {"region_1": {"obj1"}, "region_2": {"obj1"}}},
            "region_labels": {1: "robot", 2: "region_1", 3: "region_2"},
            "region_goals": {},
            "robot_label": "region_0",
            "goal_label": "",
            "goal_in_free_space": False,
        }

    monkeypatch.setattr(namo.planners, "get_region_snapshot", _fake_snapshot)

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
