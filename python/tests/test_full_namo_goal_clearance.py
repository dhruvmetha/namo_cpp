"""Focused occupied-goal regressions; physics coverage runs on Amarel."""

from types import SimpleNamespace

import numpy as np
import pytest

from namo.core import PlannerConfig
from namo.planners.full_namo.full_namo_planner import FullNAMOPlanner
from test_full_namo_strict_bfs import make_failure_result, make_success_result


def scene_snapshot(objects=("A", "B"), *, access="robot", static=False):
    adjacency = {"robot": set()}
    edges = {"robot": {}}
    if access != "robot":
        adjacency = {"robot": {access}, access: {"robot"}}
        edges = {"robot": {access: {"K"}}, access: {"robot": {"K"}}}
    return {
        "adjacency": adjacency,
        "edge_objects": edges,
        "region_labels": {1: "robot", 2: access},
        "robot_label": "robot",
        "goal_label": "" if objects or static else "robot",
        "goal_in_free_space": not objects and not static,
        "region_goals": {},
        "goal_clearance": {
            "goal_xy": [0.205, 0.105],
            "resolution": 0.01,
            "cells": [{"xy": [0.205, 0.105], "grid": [20, 10],
                       "static_blocked": static, "objects": list(objects),
                       "region": "" if objects or static else "robot"}],
            "access_regions": {obj: [access] for obj in objects},
            "reachable_objects": list(objects) if access == "robot" else [],
        },
    }


class ClearanceEnv:
    def __init__(self):
        self.objects = frozenset({"A", "B"})

    def set_robot_goal(self, *goal):
        self.goal = goal

    def is_robot_goal_reachable(self):
        return not self.objects

    def object_occupies_point(self, object_id, xy):
        return object_id in self.objects

    def get_full_state(self):
        return self.objects

    def set_full_state(self, state):
        self.objects = state

    def get_xml_path(self):
        return "scene.xml"

    def get_config_path(self):
        return "config.yaml"


def planner_for(monkeypatch, env, opener):
    monkeypatch.setattr(FullNAMOPlanner, "_initialize_algorithm",
                        lambda self: setattr(self, "region_opener", opener))
    planner = FullNAMOPlanner(env, PlannerConfig(algorithm_params={
        "full_namo_local_search": "best_first", "full_namo_goal_clearance": True,
    }))
    monkeypatch.setattr(planner, "_compute_region_snapshot",
                        lambda: scene_snapshot(sorted(env.objects)))
    return planner


def test_clearance_task_can_finish_with_second_blocker_remaining():
    from namo.planners.full_namo.goal_clearance import clearance_targets

    targets = clearance_targets(scene_snapshot())
    env = ClearanceEnv()
    assert [target.object_id for target in targets] == ["A", "B"]
    assert not targets[0].is_open(env)
    env.objects = frozenset({"B"})
    assert targets[0].is_open(env)
    assert not env.is_robot_goal_reachable()
    assert not targets[1].is_open(env)
    assert clearance_targets(scene_snapshot(static=True)) == []


def test_full_namo_clears_two_objects_without_neighbor_regions(monkeypatch):
    env = ClearanceEnv()
    calls = []

    class Opener:
        def reset(self):
            pass

        def search(self, goal, target_neighbor=None, **kwargs):
            target = kwargs["clearance_target"]
            calls.append(target.object_id)
            return make_success_result(target_neighbor,
                                       env.objects - {target.object_id}, target.object_id)

    planner = planner_for(monkeypatch, env, Opener())
    result = planner.search((0.205, 0.105, 0.0))
    assert result.success, result.error_message
    assert calls == ["A", "B"]
    assert env.is_robot_goal_reachable()
    assert len(result.action_sequence) == 2


def test_unreachable_goal_blocker_routes_to_actual_approach_region(monkeypatch):
    env = ClearanceEnv()
    planner = planner_for(monkeypatch, env, object())
    snapshot = scene_snapshot(access="middle")
    choices = planner._route_choices(snapshot, "robot", "", {})
    assert choices
    assert {(choice.object_id, choice.target_region) for choice in choices} == {("K", "middle")}
    assert all(choice.hops == 2 and choice.graph_hops == 1 for choice in choices)
    assert all(not choice.reaches_goal for choice in choices)


def test_exhausted_clearance_tries_other_object_in_same_state(monkeypatch):
    env = ClearanceEnv()
    calls = []

    class Opener:
        def reset(self):
            pass

        def search(self, goal, target_neighbor=None, **kwargs):
            obj = kwargs["target_object_id"]
            calls.append(obj)
            if obj == "A":
                return make_failure_result(target_neighbor, "all_pushes_failed", boundary_exhausted=True)
            # A collision with another movable can displace it too; final goal decides success.
            return make_success_result(target_neighbor, frozenset(), obj)

    result = planner_for(monkeypatch, env, Opener()).search((0.205, 0.105, 0.0))
    assert result.success, result.error_message
    assert calls == ["A", "B"]


def test_explicit_goal_mask_and_diagnostics_preserve_overlap():
    from namo.planners.full_namo.goal_clearance import (
        GoalClearanceScorer, clearance_targets, goal_diagnostics, goal_mask,
    )

    mask = goal_mask([(0.205, 0.105)], 0.01, (0.0, 0.4, 0.0, 0.4), 40)
    assert mask.dtype == np.float32
    assert mask.sum() == pytest.approx(1.0)
    assert mask[10, 20] == pytest.approx(1.0)
    diag = goal_diagnostics(scene_snapshot())
    assert diag["goal_occupied_by_movables"] is True
    assert diag["movable_ids_covering_goal_cells"] == ["A", "B"]
    assert diag["reachable_goal_blockers"] == ["A", "B"]
    assert goal_diagnostics(scene_snapshot(static=True))["goal_has_static_free_cells"] is False

    # The adapter changes only the goal channel; weights, contacts and score mode stay fixed.
    ctx = np.arange(5 * 40 * 40, dtype=np.float32).reshape(5, 40, 40)
    original = ctx.copy()
    calls = []
    scorer = SimpleNamespace(
        render_ctx=lambda *args: (ctx, {"local_bounds": (0.0, 0.4, 0.0, 0.4)}),
        contact_px_live=lambda *args: [3, 4],
        score_ctx=lambda ctx, contacts, **kwargs: calls.append((ctx.copy(), contacts, kwargs)),
    )
    GoalClearanceScorer(scorer, clearance_targets(scene_snapshot())[0]).score_state(
        object(), "A", (0.205, 0.105, 0.0), "scene.xml", h=2, raw=True)
    np.testing.assert_array_equal(calls[0][0][:-1], original[:-1])
    np.testing.assert_array_equal(calls[0][0][-1], mask)
    assert calls[0][1:] == ([3, 4], {"h": 2, "raw": True})


def test_real_opener_marks_non_neighbor_clearance_as_exhausted(monkeypatch):
    from namo.planners.full_namo.goal_clearance import clearance_targets
    from namo.planners.opening.best_first_region_opening import BestFirstRegionOpeningPlanner
    import namo.planners.opening.best_first_region_opening as opening

    env = ClearanceEnv()
    monkeypatch.setattr("namo.planners.get_region_snapshot", lambda *args, **kwargs: scene_snapshot())
    monkeypatch.setattr(opening, "solve_scene", lambda *args, **kwargs: (False, 0, None, [], "exhausted"))
    opener = BestFirstRegionOpeningPlanner(env, PlannerConfig(algorithm_params={"best_first_prior": "uniform"}))
    target = clearance_targets(scene_snapshot())[0]
    result = opener.search((0.205, 0.105, 0.0), target_neighbor=target.label,
                           target_object_id=target.object_id, clearance_target=target)
    assert not result.success
    assert result.algorithm_stats["target_summary"]["boundary_exhausted"]
    assert not result.algorithm_stats["target_summary"]["target_is_immediate_neighbor"]
    assert env.objects == frozenset({"A", "B"})


def test_backend_goal_ownership_is_read_only():
    from conftest import REAL_NAMO_RL, CONFIG_PATH, TEST_SCENE
    if not REAL_NAMO_RL:
        pytest.skip("Run with real bindings on Amarel")
    import namo_rl

    env = namo_rl.RLEnvironment(str(TEST_SCENE), str(CONFIG_PATH), False, True)
    observation = env.get_observation()
    object_id = next(key[:-5] for key in observation if key.endswith("_pose") and key != "robot_pose")
    xy = observation[object_id + "_pose"][:2]
    env.set_robot_goal(*xy, 0.0)
    before = env.get_full_state()
    snapshot = env.get_region_snapshot(0, -1.0, False, 42, False, True)
    assert any(object_id in cell["objects"] for cell in snapshot["goal_clearance"]["cells"])
    assert any(not cell["static_blocked"] for cell in snapshot["goal_clearance"]["cells"])
    assert env.object_occupies_point(object_id, xy)
    assert env.get_full_state().qpos == before.qpos
    assert env.get_full_state().qvel == before.qvel


def test_failed_runner_retains_committed_actions_and_terminal_state(monkeypatch):
    import namo.solvability_runner as runner
    from namo.core import PlannerResult

    action = SimpleNamespace(object_id="A", edge_idx=2, depth=0, x=0.0, y=0.1, theta=0.0)
    result = PlannerResult(success=False, solution_found=False, action_sequence=[action], algorithm_stats={
        "failure_kind": "simulation_budget_exhausted", "simulation_budget_used": 20000,
        "goal_diagnostics": {"goal_occupied_by_movables": True},
    })
    env = SimpleNamespace(get_full_state=lambda: SimpleNamespace(qpos=[1.0, 2.0], qvel=[0.0]),
                          is_robot_goal_reachable=lambda: False)
    monkeypatch.setattr(runner.namo_rl, "RLEnvironment", lambda *args: env)
    monkeypatch.setattr(runner, "FullNAMOPlanner", lambda *args: SimpleNamespace(search=lambda goal: result))
    monkeypatch.setattr(runner, "extract_goal_from_xml", lambda path: (0.0, 0.0, 0.0))
    task = runner.SolveTask("scene.xml", 2, "config.yaml", "random_rollout", 2, "data",
                            "1x_car_d5_", None, None, 20, 100, 42, True, 20000,
                            local_search="best_first", best_first_prior="uniform", goal_clearance=True)
    output = runner.solve_environment_task(task)
    assert output["kind"] == "unsolved"
    row = output["row"]
    assert row["simulation_budget_used"] == 20000
    assert len(row["committed_actions"]) == 1
    assert row["terminal_state"] == {"qpos": [1.0, 2.0], "qvel": [0.0]}
    assert row["goal_diagnostics"]["goal_occupied_by_movables"]
