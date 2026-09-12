"""A local opening can be executable before the whole simulated task is solved."""

from types import SimpleNamespace

import pytest

from namo.core import PlannerConfig
from namo.planners.full_namo.full_namo_planner import FullNAMOPlanner
from test_full_namo_strict_bfs import FakeEnv, make_snapshot, make_success_result


POINTS = [(0.2 + i * 0.01, 0.5, 0.0) for i in range(10)]


def two_keyholes(monkeypatch, horizon="full_goal", *, second_fails=False):
    env = FakeEnv()
    calls = []

    class Opener:
        def search(self, goal, target_neighbor=None, **kwargs):
            calls.append(target_neighbor)
            if len(calls) == 2 and second_fails:
                raise AssertionError("searched the second keyhole")
            result = make_success_result(
                target_neighbor, "first_open" if len(calls) == 1 else "opened2"
            )
            if len(calls) == 1:
                result.action_sequence *= 2
            result.algorithm_stats["attempt_results"][0].region_goals_sampled = POINTS
            return result

        def _minimum_needed(self, n):
            return 2

    monkeypatch.setattr(
        FullNAMOPlanner, "_initialize_algorithm",
        lambda self: setattr(self, "region_opener", Opener()),
    )
    planner = FullNAMOPlanner(env, PlannerConfig(algorithm_params={
        "full_namo_local_search": "best_first",
        "full_namo_planning_horizon": horizon,
    }))
    initial = make_snapshot({
        "robot": {"middle"}, "middle": {"robot", "goal"}, "goal": {"middle"},
    }, goal_label="goal")
    after_first = make_snapshot({"robot": {"goal"}, "goal": {"robot"}}, goal_label="goal")
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda:
                        initial if env.current_state == "baseline" else after_first)
    return planner, env, calls


def test_full_goal_still_searches_both_keyholes(monkeypatch):
    planner, env, calls = two_keyholes(monkeypatch)
    result = planner.search((0.5, 0.7, 0.0))
    assert result.success
    assert calls == ["middle", "goal"]
    assert len(result.action_sequence) == 3
    assert env.is_robot_goal_reachable()


def test_first_keyhole_returns_complete_local_chain_without_searching_next(monkeypatch):
    planner, env, calls = two_keyholes(monkeypatch, "first_keyhole", second_fails=True)
    result = planner.search((0.5, 0.7, 0.0))
    assert result.success
    assert calls == ["middle"]
    assert len(result.action_sequence) == 2
    assert not env.is_robot_goal_reachable()
    assert result.algorithm_stats["plan_outcome"] == "keyhole_ready"
    assert result.algorithm_stats["goal_reachable"] is False
    target = result.algorithm_stats["keyhole_target"]
    assert target["kind"] == "region"
    assert target["blocking_objects"] == ["box"]
    assert target["target_points"] == [list(p[:2]) for p in POINTS]
    assert target["min_reachable"] == 2


def test_invalid_horizon_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="planning_horizon"):
        two_keyholes(monkeypatch, "one_push")


def test_first_keyhole_already_reachable_goal_needs_no_push(monkeypatch):
    planner, env, calls = two_keyholes(monkeypatch, "first_keyhole")
    env.current_state = "opened2"
    result = planner.search((0.5, 0.7, 0.0))
    assert result.success
    assert result.action_sequence == []
    assert calls == []
    assert result.algorithm_stats["plan_outcome"] == "goal_reachable"


def test_replan_keeps_original_blocker_and_frozen_opening_bar(monkeypatch):
    planner, env, _ = two_keyholes(monkeypatch, "first_keyhole")
    target = {
        "kind": "region", "blocking_objects": ["box"],
        "target_points": [list(p[:2]) for p in POINTS], "min_reachable": 2,
    }
    planner.active_keyhole = target
    snapshot = make_snapshot({
        "robot": {"a", "b"}, "a": {"robot", "goal"},
        "b": {"robot", "goal"}, "goal": {"a", "b"},
    }, goal_label="goal")
    snapshot["edge_objects"]["robot"]["a"] = ["other"]
    snapshot["edge_objects"]["a"]["robot"] = ["other"]
    routes = planner._route_choices(snapshot, "robot", "goal", {})
    assert {choice.object_id for choice in routes} == {"box"}


def test_clearance_opening_preserves_its_own_success_criterion():
    from namo.planners.full_namo.keyhole_target import keyhole_is_open
    env = SimpleNamespace(
        is_robot_goal_reachable=lambda: False,
        object_occupies_point=lambda object_id, point: object_id == "box" and point == (0.4, 0.7),
    )
    target = {"kind": "goal_clearance", "blocking_objects": ["box"], "witness_xy": [0.4, 0.7]}
    assert not keyhole_is_open(env, target)
    env.object_occupies_point = lambda *_: False
    assert keyhole_is_open(env, target)
