"""Solving ONE pinned region boundary through the service facade.

plan_from_xml solves the whole problem and re-chooses its own next boundary on
every call. An executor running one physical push at a time cannot use that:
the choice is remade after each push, so a setup push can strand itself against
a boundary the next call no longer targets.

solve_boundary_from_xml takes the choice as input instead. These tests cover
the parts that do not need a simulator: resolving which boundary the caller
means in a fresh snapshot, and flattening an opener's result into something an
executor can act on and persist.
"""

from types import SimpleNamespace

import pytest

from namo.services import BoundaryOpeningResult, NAMOPlanningService
from namo.services.planning_service import (
    _boundary_object_set,
    _durable_state,
    _resolve_boundary_target,
)

ROBOT = "robot_goal"
NEAR = "region_3"
FAR = "region_7"
POINTS = [(0.3, 0.4), (0.31, 0.4), (0.3, 0.41)]


def _snapshot(adjacency=None, edge_objects=None, robot_label=ROBOT):
    return {
        "robot_label": robot_label,
        "adjacency": adjacency if adjacency is not None else {ROBOT: {NEAR, FAR}},
        "edge_objects": edge_objects or {},
    }


# --- resolving which boundary the caller means -------------------------------

def test_blocking_objects_identify_the_boundary():
    snapshot = _snapshot(edge_objects={ROBOT: {NEAR: ["box_a"], FAR: ["box_b"]}})

    label, err = _resolve_boundary_target(snapshot, ["box_b"], None)

    assert (label, err) == (FAR, "")


def test_blocking_objects_beat_a_stale_label_hint():
    """Labels renumber after a push; the object set is the durable handle."""
    snapshot = _snapshot(edge_objects={ROBOT: {NEAR: ["box_a"], FAR: ["box_b"]}})

    label, _err = _resolve_boundary_target(snapshot, ["box_b"], target_hint=NEAR)

    assert label == FAR


def test_best_overlap_wins_when_the_boundary_partially_changed():
    snapshot = _snapshot(
        edge_objects={ROBOT: {NEAR: ["box_a"], FAR: ["box_b", "box_c"]}}
    )

    label, _err = _resolve_boundary_target(snapshot, ["box_b", "box_c"], None)

    assert label == FAR


def test_label_hint_is_used_when_no_objects_are_supplied():
    label, err = _resolve_boundary_target(_snapshot(), None, target_hint=NEAR)

    assert (label, err) == (NEAR, "")


def test_boundary_that_merged_away_is_a_typed_failure_not_an_exception():
    label, err = _resolve_boundary_target(_snapshot(), ["gone"], target_hint="region_99")

    assert label is None
    assert err == "target_not_immediate_neighbor"


def test_no_neighbours_is_reported_distinctly():
    label, err = _resolve_boundary_target(_snapshot(adjacency={ROBOT: set()}), None, NEAR)

    assert (label, err) == (None, "no_immediate_neighbors")


def test_boundary_objects_are_read_in_either_direction():
    edge_objects = {ROBOT: {NEAR: ["box_a"]}, NEAR: {ROBOT: ["box_b"]}}

    assert _boundary_object_set(edge_objects, ROBOT, NEAR) == {"box_a", "box_b"}


# --- flattening the opener's result ------------------------------------------

def _planner_result(success, attempt, actions=(), stats_extra=None):
    stats = {"attempt_results": [attempt] if attempt else []}
    stats.update(stats_extra or {})
    return SimpleNamespace(
        success=success,
        action_sequence=list(actions),
        algorithm_stats=stats,
        error_message="",
    )


def _flatten(result):
    return NAMOPlanningService._boundary_result(result, FAR, ["box_b"], POINTS, 1.0)


def test_already_open_is_a_success_with_nothing_to_execute():
    """plan_from_xml reports this as failure because it needs a non-empty plan."""
    attempt = SimpleNamespace(failure_reason="already_accessible", resulting_state=None)

    flat = _flatten(_planner_result(True, attempt))

    assert flat.success is True
    assert flat.already_open is True
    assert flat.actions == []


def test_actions_are_flattened_and_sentinels_dropped():
    attempt = SimpleNamespace(failure_reason="success", resulting_state=None)
    actions = [
        SimpleNamespace(object_id="box_b", edge_idx=17, depth=1),
        SimpleNamespace(object_id="box_b", edge_idx=-1, depth=0),
    ]

    flat = _flatten(_planner_result(True, attempt, actions))

    assert [(a.object_id, a.edge_idx, a.depth) for a in flat.actions] == [("box_b", 17, 1)]


def test_boundary_exhausted_is_surfaced():
    attempt = SimpleNamespace(failure_reason="all_pushes_failed", resulting_state=None)
    stats = {"target_summary": {"boundary_exhausted": True}}

    flat = _flatten(_planner_result(False, attempt, stats_extra=stats))

    assert flat.boundary_exhausted is True
    assert flat.failure_reason == "all_pushes_failed"


def test_resulting_state_is_stored_as_plain_lists():
    """RLState is not picklable, so the executor gets qpos/qvel it can persist."""
    attempt = SimpleNamespace(
        failure_reason="success", resulting_state=SimpleNamespace(qpos=[1, 2], qvel=[0, 0])
    )

    flat = _flatten(_planner_result(True, attempt))

    assert flat.resulting_state == {"qpos": [1.0, 2.0], "qvel": [0.0, 0.0]}


def test_durable_state_of_nothing_is_none():
    assert _durable_state(None) is None


def test_graded_points_are_echoed_for_the_run_log():
    attempt = SimpleNamespace(failure_reason="success", resulting_state=None)

    assert _flatten(_planner_result(True, attempt)).graded_points == POINTS


# --- input validation --------------------------------------------------------

def _service():
    return NAMOPlanningService.__new__(NAMOPlanningService)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"target_points": []}, "target_points"),
        ({"target_points": POINTS, "local_search": "beam"}, "local_search"),
    ],
)
def test_bad_inputs_raise_before_any_simulator_work(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NAMOPlanningService.solve_boundary_from_xml(
            _service(), "scene.xml", (0.0, 0.0, 0.0), **kwargs
        )


def test_result_defaults_are_a_clean_failure():
    empty = BoundaryOpeningResult()

    assert empty.success is False
    assert empty.already_open is False
    assert empty.resulting_state is None
