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
from namo.planners.full_namo.full_namo_planner import (
    FullNAMOPlanner,
    choose_region_route,
)


class FakeEnv:
    def __init__(self):
        self.goal = None
        self.current_state = "baseline"

    def set_robot_goal(self, x, y, theta):
        self.goal = (x, y, theta)

    def is_robot_goal_reachable(self):
        return self.current_state == "opened" or self.current_state == "opened2"

    def set_full_state(self, state):
        self.current_state = state

    def get_xml_path(self):
        return "dummy.xml"

    def get_config_path(self):
        return "dummy.yaml"


def make_planner(monkeypatch, env, opener):
    def fake_initialize(self):
        self.region_opener = opener

    monkeypatch.setattr(FullNAMOPlanner, "_initialize_algorithm", fake_initialize)
    return FullNAMOPlanner(env, PlannerConfig())


def make_success_result(target, resulting_state, object_id="box"):
    action = SimpleNamespace(object_id=object_id, x=0.0, y=0.0, theta=0.0)
    attempt = SimpleNamespace(
        success=True,
        resulting_state=resulting_state,
        failure_reason="success",
    )
    return PlannerResult(
        success=True,
        solution_found=True,
        action_sequence=[action],
        algorithm_stats={
            "attempt_results": [attempt],
            "target_summary": {
                "target_neighbor": target,
                "local_robot_label": "robot",
                "local_neighbors": [target],
                "target_is_immediate_neighbor": True,
                "failure_reason": "success",
                "attempt_count": 1,
                "detail_reasons": ["success"],
                "boundary_exhausted": False,
            },
            "rejection_breakdown": {},
            "total_primitives_attempted": 0,
        },
    )


def make_failure_result(target, reason, *, boundary_exhausted, local_neighbors=None):
    attempt = SimpleNamespace(
        success=False,
        resulting_state=None,
        failure_reason=reason,
    )
    return PlannerResult(
        success=False,
        solution_found=False,
        action_sequence=[],
        algorithm_stats={
            "attempt_results": [attempt],
            "target_summary": {
                "target_neighbor": target,
                "local_robot_label": "robot",
                "local_neighbors": local_neighbors or [target],
                "target_is_immediate_neighbor": reason != "target_not_immediate_neighbor",
                "failure_reason": reason,
                "attempt_count": 1,
                "detail_reasons": [reason],
                "boundary_exhausted": boundary_exhausted,
            },
            "rejection_breakdown": {},
            "total_primitives_attempted": 0,
        },
    )


def make_snapshot(adjacency, *, goal_label, robot_label="robot", goal_in_free_space=True):
    edge_objects = {robot_label: {}}
    for neighbor in adjacency.get(robot_label, set()):
        edge_objects[robot_label][neighbor] = ["box"]
        edge_objects.setdefault(neighbor, {})[robot_label] = ["box"]
    return {
        "adjacency": adjacency,
        "edge_objects": edge_objects,
        "robot_label": robot_label,
        "goal_label": goal_label,
        "goal_in_free_space": goal_in_free_space,
    }


def test_full_namo_executes_only_first_hop_of_longer_path(monkeypatch):
    env = FakeEnv()
    calls = []

    class FakeOpener:
        def reset(self):
            pass

        def search(self, robot_goal, target_neighbor=None, **_kwargs):
            calls.append(target_neighbor)
            return make_success_result(target_neighbor, "opened")

    planner = make_planner(monkeypatch, env, FakeOpener())
    snapshot = make_snapshot(
        {
            "robot": {"a"},
            "a": {"robot", "b"},
            "b": {"a", "goal"},
            "goal": {"b"},
        },
        goal_label="goal",
    )
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: snapshot)

    result = planner.search((1.0, 2.0, 0.0))

    assert result.success is True
    assert calls == ["a"]
    assert result.algorithm_stats["successful_region_steps"] == 1
    opened_trace = next(
        entry for entry in result.algorithm_stats["iteration_trace"] if entry.get("outcome") == "opened_target"
    )
    assert opened_trace["chosen_path"] == ["robot", "a", "b", "goal"]
    assert opened_trace["chosen_target_region"] == "a"


def test_full_namo_region_path_exhausted_after_boundary_exhaustions(monkeypatch):
    env = FakeEnv()
    calls = []

    class FakeOpener:
        def reset(self):
            pass

        def search(self, robot_goal, target_neighbor=None, **_kwargs):
            calls.append(target_neighbor)
            return make_failure_result(
                target_neighbor,
                "all_pushes_failed",
                boundary_exhausted=True,
            )

    planner = make_planner(monkeypatch, env, FakeOpener())
    snapshot = make_snapshot(
        {
            "robot": {"a", "b"},
            "a": {"robot", "goal"},
            "b": {"robot", "goal"},
            "goal": {"a", "b"},
        },
        goal_label="goal",
    )
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: snapshot)

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is False
    assert result.algorithm_stats["failure_kind"] == "region_path_exhausted"
    assert calls == ["a", "b"]
    assert result.algorithm_stats["boundary_exhaustions"] == 2
    blocked = {tuple(edge) for edge in result.algorithm_stats["failure_context"]["blocked_boundaries"]}
    assert blocked == {("a", "robot"), ("b", "robot")}


def test_full_namo_non_exhaustive_failure_does_not_block_boundary(monkeypatch):
    env = FakeEnv()

    class FakeOpener:
        def reset(self):
            pass

        def search(self, robot_goal, target_neighbor=None, **_kwargs):
            return make_failure_result(target_neighbor, "timeout", boundary_exhausted=False)

    planner = make_planner(monkeypatch, env, FakeOpener())
    snapshot = make_snapshot(
        {
            "robot": {"a"},
            "a": {"robot", "goal"},
            "goal": {"a"},
        },
        goal_label="goal",
    )
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: snapshot)

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is False
    assert result.algorithm_stats["failure_kind"] == "opener_failure_not_boundary_exhausted"
    assert result.algorithm_stats["failure_context"]["blocked_boundaries"] == []


def test_full_namo_snapshot_mismatch_is_explicit_invariant(monkeypatch):
    env = FakeEnv()

    class FakeOpener:
        def reset(self):
            pass

        def search(self, robot_goal, target_neighbor=None, **_kwargs):
            return make_failure_result(
                target_neighbor,
                "target_not_immediate_neighbor",
                boundary_exhausted=False,
                local_neighbors=["b"],
            )

    planner = make_planner(monkeypatch, env, FakeOpener())
    snapshot = make_snapshot(
        {
            "robot": {"a"},
            "a": {"robot", "goal"},
            "goal": {"a"},
        },
        goal_label="goal",
    )
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: snapshot)

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is False
    assert result.algorithm_stats["failure_kind"] == "planner_invariant_violation"
    assert result.algorithm_stats["failure_subkind"] == "target_not_immediate_neighbor"
    assert result.algorithm_stats["invariant_context"]["robot_neighbors"] == ["a"]
    assert result.algorithm_stats["invariant_context"]["target_summary"]["local_neighbors"] == ["b"]


def test_full_namo_recomputes_goal_region_each_iteration(monkeypatch):
    env = FakeEnv()
    calls = []

    class FakeOpener:
        def reset(self):
            pass

        def search(self, robot_goal, target_neighbor=None, **_kwargs):
            calls.append(target_neighbor)
            if target_neighbor == "a":
                return make_success_result(target_neighbor, "opened1")
            return make_success_result(target_neighbor, "opened2")

    planner = make_planner(monkeypatch, env, FakeOpener())

    def compute_snapshot():
        if env.current_state == "opened1":
            return make_snapshot(
                {
                    "robot": {"c"},
                    "c": {"robot", "goal_right"},
                    "goal_right": {"c"},
                },
                goal_label="goal_right",
            )
        return make_snapshot(
            {
                "robot": {"a"},
                "a": {"robot", "goal_left"},
                "goal_left": {"a"},
            },
            goal_label="goal_left",
        )

    monkeypatch.setattr(planner, "_compute_region_snapshot", compute_snapshot)
    monkeypatch.setattr(env, "is_robot_goal_reachable", lambda: env.current_state == "opened2")

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is True
    assert calls == ["a", "c"]


def test_validate_region_path_rejects_blocked_or_non_adjacent_hops(monkeypatch):
    env = FakeEnv()

    class FakeOpener:
        def reset(self):
            pass

    planner = make_planner(monkeypatch, env, FakeOpener())
    adjacency = {
        "robot": {"a"},
        "a": {"robot", "goal"},
        "goal": {"a"},
    }

    assert planner._validate_region_path(
        path=["a", "robot", "goal"],
        robot_region="robot",
        goal_region="goal",
        adjacency=adjacency,
        blocked_boundaries=set(),
    ) == "path_does_not_start_at_robot_region"
    assert planner._validate_region_path(
        path=["robot", "a", "robot"],
        robot_region="robot",
        goal_region="goal",
        adjacency=adjacency,
        blocked_boundaries=set(),
    ) == "path_does_not_end_at_goal_region"
    assert planner._validate_region_path(
        path=["robot", "a", "robot", "goal"],
        robot_region="robot",
        goal_region="goal",
        adjacency=adjacency,
        blocked_boundaries=set(),
    ) == "path_contains_repeated_region"
    assert planner._validate_region_path(
        path=["robot", "b", "goal"],
        robot_region="robot",
        goal_region="goal",
        adjacency=adjacency,
        blocked_boundaries=set(),
    ) == "path_contains_non_adjacent_hop"
    assert planner._validate_region_path(
        path=["robot", "a", "goal"],
        robot_region="robot",
        goal_region="goal",
        adjacency=adjacency,
        blocked_boundaries={planner._boundary_key("robot", "a")},
    ) == "path_uses_blocked_boundary"


def test_route_attempt_cost_can_defer_a_short_route_to_an_untried_route():
    snapshot = {
        "adjacency": {
            "robot": {"a", "b"},
            "a": {"robot", "goal"},
            "b": {"robot", "c"},
            "c": {"b", "goal"},
            "goal": {"a", "c"},
        },
        "edge_objects": {
            "robot": {"a": ["box_a"], "b": ["box_b"]},
            "a": {"robot": ["box_a"]},
            "b": {"robot": ["box_b"]},
        },
    }

    first = choose_region_route(snapshot, "robot", "goal", opening_attempts={})
    deferred = choose_region_route(
        snapshot,
        "robot",
        "goal",
        opening_attempts={"box_a": 1},
    )

    assert first is not None
    assert (first.object_id, first.hops, first.cost) == ("box_a", 2, 2)
    assert deferred is not None
    assert (deferred.object_id, deferred.hops, deferred.cost) == ("box_b", 3, 3)


@pytest.mark.parametrize("goal_clearance_enabled", [False, True])
def test_full_namo_searches_every_blocker_on_the_boundary_in_one_call(
    monkeypatch, goal_clearance_enabled,
):
    """Two blockers on one boundary reach the opener together, and a chain on
    the second-named one is accepted without the first being exhausted."""
    env = FakeEnv()
    calls = []

    class FakeOpener:
        def reset(self):
            pass

        def search(
            self,
            robot_goal,
            target_neighbor=None,
            target_object_id=None,
            require_push=False,
            **_kwargs,
        ):
            calls.append((target_neighbor, target_object_id, require_push))
            return make_success_result(
                target_neighbor,
                "opened",
                object_id="box_b",
            )

    opener = FakeOpener()
    monkeypatch.setattr(
        FullNAMOPlanner,
        "_initialize_algorithm",
        lambda self: setattr(self, "region_opener", opener),
    )
    planner = FullNAMOPlanner(
        env,
        PlannerConfig(algorithm_params={
            "full_namo_local_search": "best_first",
            "full_namo_goal_clearance": goal_clearance_enabled,
        }),
    )
    snapshot = {
        **make_snapshot(
            {
                "robot": {"a"},
                "a": {"robot", "goal"},
                "goal": {"a"},
            },
            goal_label="goal",
        ),
        "edge_objects": {
            "robot": {"a": ["box_b", "box_a"]},
            "a": {"robot": ["box_a", "box_b"]},
        },
        "goal_clearance": {
            "goal_xy": [0.0, 0.0],
            "resolution": 0.01,
            "cells": [{
                "xy": [0.0, 0.0],
                "grid": [0, 0],
                "static_blocked": False,
                "objects": [],
                "region": "goal",
            }],
            "access_regions": {},
            "reachable_objects": ["box_a", "box_b"],
        },
    }
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: snapshot)

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is True
    assert calls == [("a", ("box_a", "box_b"), True)]
    assert result.algorithm_stats["boundary_exhaustions"] == 0
    opened = [
        row
        for row in result.algorithm_stats["iteration_trace"]
        if row.get("outcome") == "opened_target"
    ]
    assert [row["candidate_blockers"] for row in opened] == [["box_a", "box_b"]]


def test_unchanged_hop_count_defers_a_tried_short_route_to_an_untried_route(
    monkeypatch,
):
    env = FakeEnv()
    calls = []

    class FakeOpener:
        def reset(self):
            pass

        def search(
            self,
            robot_goal,
            target_neighbor=None,
            target_object_id=None,
            require_push=False,
        ):
            calls.append((target_neighbor, target_object_id, require_push))
            state = "changed" if target_object_id == "box_a" else "opened"
            return make_success_result(
                target_neighbor,
                state,
                object_id=target_object_id,
            )

    planner = make_planner(monkeypatch, env, FakeOpener())
    snapshot = {
        **make_snapshot(
            {
                "robot": {"a", "b"},
                "a": {"robot", "goal"},
                "b": {"robot", "c"},
                "c": {"b", "goal"},
                "goal": {"a", "c"},
            },
            goal_label="goal",
        ),
        "edge_objects": {
            "robot": {"a": ["box_a"], "b": ["box_b"]},
            "a": {"robot": ["box_a"]},
            "b": {"robot": ["box_b"]},
        },
    }
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: snapshot)

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is True
    assert calls == [
        ("a", "box_a", True),
        ("b", "box_b", True),
    ]
    opened = [
        row
        for row in result.algorithm_stats["iteration_trace"]
        if row.get("outcome") == "opened_target"
    ]
    assert [(row["route_hops"], row["route_cost"]) for row in opened] == [
        (2, 2),
        (3, 3),
    ]


def test_shorter_recomputed_route_resets_attempt_history(monkeypatch):
    env = FakeEnv()

    class FakeOpener:
        def reset(self):
            pass

        def search(
            self,
            robot_goal,
            target_neighbor=None,
            target_object_id=None,
            require_push=False,
        ):
            next_state = {
                "box_a": "changed",
                "box_b": "shorter",
                "box_c": "opened",
            }[target_object_id]
            return make_success_result(
                target_neighbor,
                next_state,
                object_id=target_object_id,
            )

    planner = make_planner(monkeypatch, env, FakeOpener())
    long_snapshot = {
        **make_snapshot(
            {
                "robot": {"a", "b"},
                "a": {"robot", "goal"},
                "b": {"robot", "c"},
                "c": {"b", "goal"},
                "goal": {"a", "c"},
            },
            goal_label="goal",
        ),
        "edge_objects": {
            "robot": {"a": ["box_a"], "b": ["box_b"]},
            "a": {"robot": ["box_a"]},
            "b": {"robot": ["box_b"]},
        },
    }
    shorter_snapshot = {
        **make_snapshot(
            {
                "robot": {"goal", "a"},
                "a": {"robot", "goal"},
                "goal": {"robot", "a"},
            },
            goal_label="goal",
        ),
        "edge_objects": {
            "robot": {"goal": ["box_c"], "a": ["box_a"]},
            "goal": {"robot": ["box_c"]},
            "a": {"robot": ["box_a"]},
        },
    }
    monkeypatch.setattr(
        planner,
        "_compute_region_snapshot",
        lambda: shorter_snapshot if env.current_state == "shorter" else long_snapshot,
    )

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is True
    opened = [
        row
        for row in result.algorithm_stats["iteration_trace"]
        if row.get("outcome") == "opened_target"
    ]
    assert [row["chosen_initial_blocker"] for row in opened] == [
        "box_a",
        "box_b",
        "box_c",
    ]
    assert opened[-1]["opening_attempts_by_object"] == {}


def _two_blocker_snapshot():
    """robot -> a -> goal, with box_a and box_b both on the robot/a boundary."""
    return {
        **make_snapshot(
            {
                "robot": {"a"},
                "a": {"robot", "goal"},
                "goal": {"a"},
            },
            goal_label="goal",
        ),
        "edge_objects": {
            "robot": {"a": ["box_b", "box_a"]},
            "a": {"robot": ["box_a", "box_b"]},
        },
    }


def make_policy_planner(monkeypatch, env, opener):
    def fake_initialize(self):
        self.region_opener = opener

    monkeypatch.setattr(FullNAMOPlanner, "_initialize_algorithm", fake_initialize)
    config = PlannerConfig(
        algorithm_params={
            "full_namo_local_search": "best_first",
            "full_namo_exec_mode": "greedy_policy",
        }
    )
    return FullNAMOPlanner(env, config)


def make_greedy_result(target, resulting_state, object_id):
    """A greedy_commit result: one committed action, boundary not yet open."""
    result = make_success_result(target, resulting_state, object_id=object_id)
    result.success = False
    result.solution_found = False
    result.action_sequence[0].edge_idx = 7
    result.action_sequence[0].depth = 0
    attempt = result.algorithm_stats["attempt_results"][0]
    attempt.success = False
    attempt.failure_reason = "greedy_step_committed"
    attempt.chosen_object_id = object_id
    summary = result.algorithm_stats["target_summary"]
    summary["failure_reason"] = "greedy_step_committed"
    summary["detail_reasons"] = ["greedy_step_committed"]
    summary["boundary_exhausted"] = False
    result.algorithm_stats["greedy_commit"] = {"end": "committed", "rejections": []}
    return result


def test_greedy_policy_hands_every_open_blocker_on_the_boundary_to_the_opener(
    monkeypatch,
):
    """The policy returns after one push, so it must rank across both blockers at once."""
    env = FakeEnv()
    calls = []

    class FakeOpener:
        def reset(self):
            pass

        def greedy_commit(self, robot_goal, target_neighbor=None, target_object_id=None, **_kw):
            calls.append((target_neighbor, target_object_id))
            # The arg-max lands on the second blocker; the planner must accept it.
            return make_greedy_result(target_neighbor, "pushed", object_id="box_b")

    planner = make_policy_planner(monkeypatch, env, FakeOpener())
    monkeypatch.setattr(planner, "_compute_region_snapshot", _two_blocker_snapshot)

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is True
    assert calls == [("a", ("box_a", "box_b"))]
    assert [str(a.object_id) for a in result.action_sequence] == ["box_b"]
    trace = result.algorithm_stats["iteration_trace"][-1]
    assert trace["candidate_blockers"] == ["box_a", "box_b"]
    assert trace["greedy_action"]["object_id"] == "box_b"


def test_search_exhaustion_of_the_pooled_blockers_blocks_both_and_reroutes(monkeypatch):
    """Search-mode twin of the policy exhaustion test: when the simulator finds no
    chain on either blocker, both count as tried and the next boundary is used."""
    env = FakeEnv()
    calls = []

    class FakeOpener:
        def reset(self):
            pass

        def search(self, robot_goal, target_neighbor=None, target_object_id=None, **_kw):
            calls.append((target_neighbor, target_object_id))
            if target_neighbor == "a":
                return make_failure_result(
                    target_neighbor, "all_pushes_failed", boundary_exhausted=True
                )
            return make_success_result(target_neighbor, "opened", object_id="box_c")

    planner = make_planner(monkeypatch, env, FakeOpener())
    snapshot = {
        **make_snapshot(
            {
                "robot": {"a", "b"},
                "a": {"robot", "goal"},
                "b": {"robot", "goal"},
                "goal": {"a", "b"},
            },
            goal_label="goal",
        ),
        "edge_objects": {
            "robot": {"a": ["box_a", "box_b"], "b": ["box_c"]},
            "a": {"robot": ["box_a", "box_b"]},
            "b": {"robot": ["box_c"]},
        },
    }
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: snapshot)

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is True
    assert calls == [("a", ("box_a", "box_b")), ("b", "box_c")]
    assert result.algorithm_stats["boundary_exhaustions"] == 1
    opened = [
        entry
        for entry in result.algorithm_stats["iteration_trace"]
        if entry.get("outcome") == "opened_target"
    ]
    assert opened[-1]["opening_attempts_by_object"] == {"box_a": 1, "box_b": 1}


def test_greedy_policy_exhaustion_blocks_every_candidate_and_reroutes(monkeypatch):
    """When the pooled candidates are all dead, both blockers count as tried and the
    planner moves to the next boundary in the same call instead of retrying one of them."""
    env = FakeEnv()
    calls = []

    class FakeOpener:
        def reset(self):
            pass

        def greedy_commit(self, robot_goal, target_neighbor=None, target_object_id=None, **_kw):
            calls.append((target_neighbor, target_object_id))
            if target_neighbor == "a":
                return make_failure_result(
                    target_neighbor, "all_pushes_failed", boundary_exhausted=True
                )
            return make_greedy_result(target_neighbor, "pushed", object_id="box_c")

    planner = make_policy_planner(monkeypatch, env, FakeOpener())
    snapshot = {
        **make_snapshot(
            {
                "robot": {"a", "b"},
                "a": {"robot", "goal"},
                "b": {"robot", "goal"},
                "goal": {"a", "b"},
            },
            goal_label="goal",
        ),
        "edge_objects": {
            "robot": {"a": ["box_a", "box_b"], "b": ["box_c"]},
            "a": {"robot": ["box_a", "box_b"]},
            "b": {"robot": ["box_c"]},
        },
    }
    monkeypatch.setattr(planner, "_compute_region_snapshot", lambda: snapshot)

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is True
    assert calls == [("a", ("box_a", "box_b")), ("b", "box_c")]
    assert result.algorithm_stats["boundary_exhaustions"] == 1
    exhausted_trace = result.algorithm_stats["iteration_trace"][0]
    assert exhausted_trace["outcome"] == "boundary_exhausted"
    assert exhausted_trace["candidate_blockers"] == ["box_a", "box_b"]
    committed_trace = result.algorithm_stats["iteration_trace"][1]
    assert committed_trace["opening_attempts_by_object"] == {"box_a": 1, "box_b": 1}
    assert sorted(
        entry["object_id"] for entry in committed_trace["blocked_boundary_objects"]
    ) == ["box_a", "box_b"]
