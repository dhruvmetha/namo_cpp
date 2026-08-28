"""Full NAMO greedy DFS commits one child and rebuilds the global graph."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


if "namo_rl" not in sys.modules:
    namo_rl_stub = types.ModuleType("namo_rl")
    namo_rl_stub.Action = type("Action", (), {})
    namo_rl_stub.RLEnvironment = object
    namo_rl_stub.RLState = object
    sys.modules["namo_rl"] = namo_rl_stub

from namo.core import PlannerConfig, PlannerResult
from namo.planners.full_namo.full_namo_planner import FullNAMOPlanner
from namo.planners.utils import PushAttemptBudget


GOAL = (1.0, 2.0, 0.0)


class FakeEnv:
    def __init__(self):
        self.goal = None
        self.current_state = "baseline"
        self.state_history = []

    def set_robot_goal(self, x, y, theta):
        self.goal = (x, y, theta)

    def is_robot_goal_reachable(self):
        return self.current_state == "opened"

    def set_full_state(self, state):
        self.current_state = state
        self.state_history.append(state)

    def get_xml_path(self):
        return "dummy.xml"

    def get_config_path(self):
        return "dummy.yaml"


def _snapshot(targets, goal_label):
    adjacency = {"robot": set(targets)}
    for target in targets:
        adjacency[target] = {"robot"}
    return {
        "adjacency": adjacency,
        "robot_label": "robot",
        "goal_label": goal_label,
        "goal_in_free_space": True,
    }


def _alternate_snapshot():
    return {
        "adjacency": {
            "robot": {"a", "b"},
            "a": {"robot", "goal"},
            "b": {"robot", "goal"},
            "goal": {"a", "b"},
        },
        "robot_label": "robot",
        "goal_label": "goal",
        "goal_in_free_space": True,
    }


def _action(edge):
    return SimpleNamespace(
        object_id="box", edge_idx=edge, depth=0,
        x=0.0, y=0.0, theta=0.0,
    )


def _result(target, *, state=None, edge=None, opened=False, exhausted=False, sims=1):
    committed = state is not None and edge is not None
    reason = "success" if opened else "greedy_step_committed" if committed else "all_pushes_failed"
    attempt = SimpleNamespace(
        success=opened,
        resulting_state=state,
        failure_reason=reason,
        chosen_object_id="box" if committed else None,
        push_exec_count=sims,
        pushes_total_for_neighbour=sims,
    )
    actions = [_action(edge)] if committed else []
    return PlannerResult(
        success=opened,
        solution_found=opened,
        action_sequence=actions,
        algorithm_stats={
            "attempt_results": [attempt],
            "target_summary": {
                "target_neighbor": target,
                "local_robot_label": "robot",
                "local_neighbors": [target],
                "target_is_immediate_neighbor": True,
                "failure_reason": reason,
                "attempt_count": 1,
                "detail_reasons": [reason],
                "boundary_exhausted": exhausted,
            },
            "rejection_breakdown": {},
            "total_primitives_attempted": sims,
            "simulation_budget_used": sims,
            "greedy_commit": {
                "end": "opened" if opened else "committed" if committed else "exhausted",
                "rejections": [],
            },
        },
    )


class FakeOpener:
    def __init__(self, results):
        self.results = list(results)
        self.targets = []
        self.greedy_calls = 0
        self.search_calls = 0

    def reset(self):
        return None

    def greedy_commit(self, robot_goal, target_neighbor=None, **_kwargs):
        self.greedy_calls += 1
        self.targets.append(target_neighbor)
        return self.results.pop(0)

    def search(self, robot_goal, target_neighbor=None, **_kwargs):
        self.search_calls += 1
        self.targets.append(target_neighbor)
        return self.results.pop(0)


def _planner(monkeypatch, env, opener, *, mode="greedy_dfs", max_pushes=2):
    monkeypatch.setattr(
        FullNAMOPlanner,
        "_initialize_algorithm",
        lambda self: setattr(self, "region_opener", opener),
    )
    config = PlannerConfig(
        algorithm_params={
            "full_namo_local_search": "best_first",
            "full_namo_exec_mode": mode,
            "best_first_hmax": max_pushes,
            "push_budget": PushAttemptBudget(limit=20),
        }
    )
    return FullNAMOPlanner(env, config)


def test_rebuilds_the_global_graph_after_each_committed_push(monkeypatch):
    env = FakeEnv()
    opener = FakeOpener([
        _result("first", state="state-1", edge=3),
        _result("second", state="opened", edge=7, opened=True),
    ])
    planner = _planner(monkeypatch, env, opener)
    snapshots = iter([_snapshot(["first"], "first"), _snapshot(["second"], "second")])
    snapshot_calls = []

    def next_snapshot():
        snapshot_calls.append(env.current_state)
        return next(snapshots)

    monkeypatch.setattr(planner, "_compute_region_snapshot", next_snapshot)

    result = planner.search(GOAL)

    assert result.success is True
    assert snapshot_calls == ["baseline", "state-1"]
    assert opener.targets == ["first", "second"]
    assert opener.search_calls == 0
    assert [action.edge_idx for action in result.action_sequence] == [3, 7]
    assert result.algorithm_stats["greedy_committed_pushes"] == 2


def test_greedy_dfs_is_not_capped_by_candidate_hmax(monkeypatch):
    env = FakeEnv()
    opener = FakeOpener([
        _result("goal", state="state-1", edge=3),
        _result("goal", state="state-2", edge=7),
        _result("goal", state="opened", edge=11, opened=True),
    ])
    planner = _planner(monkeypatch, env, opener, max_pushes=2)
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: _snapshot(["goal"], "goal"))

    result = planner.search(GOAL)

    assert result.success is True
    assert [action.edge_idx for action in result.action_sequence] == [3, 7, 11]
    assert result.algorithm_stats["greedy_committed_pushes"] == 3


def test_greedy_policy_returns_one_moving_step_before_goal_opens(monkeypatch):
    env = FakeEnv()
    opener = FakeOpener([_result("goal", state="state-1", edge=3)])
    planner = _planner(monkeypatch, env, opener, mode="greedy_policy")
    monkeypatch.setattr(
        planner,
        "_compute_region_snapshot",
        lambda: _snapshot(["goal"], "goal"),
    )

    result = planner.search(GOAL)

    assert result.success is True
    assert [action.edge_idx for action in result.action_sequence] == [3]
    assert result.algorithm_stats["exec_mode"] == "greedy_policy"
    assert result.algorithm_stats["policy_outcome"] == "policy_step_ready"


def test_reselects_at_same_state_when_a_boundary_has_no_moving_candidate(monkeypatch):
    env = FakeEnv()
    opener = FakeOpener([
        _result("a", exhausted=True, sims=3),
        _result("b", state="opened", edge=4, opened=True),
    ])
    planner = _planner(monkeypatch, env, opener, max_pushes=1)
    snapshot = _alternate_snapshot()
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: snapshot)

    result = planner.search(GOAL)

    assert result.success is True
    assert opener.targets == ["a", "b"]
    assert env.state_history == ["opened"]


def test_requires_best_first():
    with pytest.raises(ValueError, match="best_first"):
        FullNAMOPlanner(
            FakeEnv(),
            PlannerConfig(algorithm_params={"full_namo_exec_mode": "greedy_dfs"}),
        )


def test_ordinary_search_does_not_enter_greedy_commit(monkeypatch):
    env = FakeEnv()
    opener = FakeOpener([_result("goal", state="opened", edge=2, opened=True)])
    planner = _planner(monkeypatch, env, opener, mode="search")
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: _snapshot(["goal"], "goal"))

    result = planner.search(GOAL)

    assert result.success is True
    assert opener.greedy_calls == 0
    assert opener.search_calls == 1
