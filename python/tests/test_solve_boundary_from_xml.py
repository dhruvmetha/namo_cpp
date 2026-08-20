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
    _region_search_params,
    _reporting_attempt,
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


def test_equally_matching_neighbours_are_refused_not_guessed():
    """Both boundaries carry box_a and nothing else, so the objects name neither."""
    snapshot = _snapshot(edge_objects={ROBOT: {NEAR: ["box_a"], FAR: ["box_a"]}})

    label, err = _resolve_boundary_target(snapshot, ["box_a"], None)

    assert label is None
    assert err == "ambiguous_boundary"


def test_the_closer_match_wins_over_a_boundary_carrying_extras():
    """Equal overlap, but FAR is blocked by exactly what the caller pinned."""
    snapshot = _snapshot(
        edge_objects={ROBOT: {NEAR: ["box_a", "box_x", "box_y"], FAR: ["box_a"]}}
    )

    label, err = _resolve_boundary_target(snapshot, ["box_a"], None)

    assert (label, err) == (FAR, "")


def test_a_label_hint_does_not_rescue_an_ambiguous_object_set():
    """The hint is for the first call only; here the caller must re-choose."""
    snapshot = _snapshot(edge_objects={ROBOT: {NEAR: ["box_a"], FAR: ["box_a"]}})

    label, err = _resolve_boundary_target(snapshot, ["box_a"], target_hint=NEAR)

    assert (label, err) == (None, "ambiguous_boundary")


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

def _attempt(failure_reason, success=False, resulting_state=None):
    """One AttemptResult as the opener records it, one per candidate object."""
    return SimpleNamespace(
        success=success, failure_reason=failure_reason, resulting_state=resulting_state
    )


def _planner_result(success, attempts, actions=(), stats_extra=None):
    stats = {"attempt_results": [a for a in attempts if a is not None]}
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
    attempt = _attempt("already_accessible")

    flat = _flatten(_planner_result(True, [attempt]))

    assert flat.success is True
    assert flat.already_open is True
    assert flat.actions == []


def test_actions_are_flattened_and_sentinels_dropped():
    attempt = _attempt("success", success=True)
    actions = [
        SimpleNamespace(object_id="box_b", edge_idx=17, depth=1),
        SimpleNamespace(object_id="box_b", edge_idx=-1, depth=0),
    ]

    flat = _flatten(_planner_result(True, [attempt], actions))

    assert [(a.object_id, a.edge_idx, a.depth) for a in flat.actions] == [("box_b", 17, 1)]


def test_boundary_exhausted_is_surfaced():
    attempt = _attempt("all_pushes_failed")
    stats = {"target_summary": {"boundary_exhausted": True}}

    flat = _flatten(_planner_result(False, [attempt], stats_extra=stats))

    assert flat.boundary_exhausted is True
    assert flat.failure_reason == "all_pushes_failed"


def test_resulting_state_is_stored_as_plain_lists():
    """RLState is not picklable, so the executor gets qpos/qvel it can persist."""
    attempt = _attempt(
        "success", success=True, resulting_state=SimpleNamespace(qpos=[1, 2], qvel=[0, 0])
    )

    flat = _flatten(_planner_result(True, [attempt]))

    assert flat.resulting_state == {"qpos": [1.0, 2.0], "qvel": [0.0, 0.0]}


def test_a_sweep_that_failed_then_succeeded_reports_the_successful_attempt():
    """The opener tries every candidate object and keeps the failures.

    Regression: reading attempt_results[0] returned success=True carrying the
    first candidate's all_pushes_failed and no resulting state, so an executor
    logged a solved boundary as a failure and had no state to continue from.
    """
    state = SimpleNamespace(qpos=[1, 2], qvel=[0, 0])
    attempts = [
        _attempt("all_pushes_failed"),
        _attempt("no_reachable_objects"),
        _attempt("success", success=True, resulting_state=state),
    ]
    actions = [SimpleNamespace(object_id="box_c", edge_idx=3, depth=0)]

    flat = _flatten(_planner_result(True, attempts, actions))

    assert flat.success is True
    assert flat.failure_reason == "success"
    assert flat.already_open is False
    assert flat.resulting_state == {"qpos": [1.0, 2.0], "qvel": [0.0, 0.0]}


def test_the_openers_aggregate_verdict_beats_a_single_attempt():
    """target_summary already accounts for the whole sweep, so trust it."""
    attempts = [_attempt("all_pushes_failed"), _attempt("success", success=True)]
    stats = {"target_summary": {"failure_reason": "success"}}

    flat = _flatten(_planner_result(True, attempts, stats_extra=stats))

    assert flat.failure_reason == "success"


def test_every_attempt_failing_still_reports_the_first_reason():
    attempts = [_attempt("all_pushes_failed"), _attempt("no_reachable_objects")]

    flat = _flatten(_planner_result(False, attempts))

    assert flat.success is False
    assert flat.failure_reason == "all_pushes_failed"


def test_reporting_attempt_of_nothing_is_none():
    assert _reporting_attempt([]) is None


def test_durable_state_of_nothing_is_none():
    assert _durable_state(None) is None


def test_graded_points_are_echoed_for_the_run_log():
    attempt = _attempt("success", success=True)

    assert _flatten(_planner_result(True, [attempt])).graded_points == POINTS


# --- the search options, shared with plan_from_xml ---------------------------
#
# solve_boundary_from_xml used to name none of these, so a caller holding a
# boundary across pushes got whatever the openers default to. region_bfs
# defaults region_max_chain_depth to 1, at which no setup-then-finish chain
# exists -- the only reason to hold a boundary. Both entry points now build the
# region_* keys from one table so they cannot diverge again.

def _search_params(**over):
    kwargs = dict(
        goal_strategy="primitive", max_chain_depth=1, max_solutions_per_neighbor=1,
        allow_collisions=True, frontier_beam_width=10000, chain_link_cost=11,
        selection_strategy="cost_first",
    )
    kwargs.update(over)
    return _region_search_params(**kwargs)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_chain_depth_becomes_the_key_the_opener_reads(depth):
    assert _search_params(max_chain_depth=depth)["region_max_chain_depth"] == depth


@pytest.mark.parametrize(
    "caller_name,region_key,value",
    [
        ("allow_collisions", "region_allow_collisions", False),
        ("frontier_beam_width", "region_frontier_beam_width", 25),
        ("chain_link_cost", "region_chain_link_cost", 7),
        ("selection_strategy", "region_selection_strategy", "depth_first"),
        ("max_solutions_per_neighbor", "region_max_solutions_per_neighbor", 4),
    ],
)
def test_each_option_maps_to_the_key_the_opener_reads(caller_name, region_key, value):
    assert _search_params(**{caller_name: value})[region_key] == value


def test_recorded_solutions_follow_the_solution_cap():
    """One caller-facing name drives both keys, which used to be set twice."""
    params = _search_params(max_solutions_per_neighbor=3)

    assert params["region_max_recorded_solutions_per_neighbor"] == 3


def test_an_absent_timeout_stays_absent():
    """The opener has its own default; a None must not overwrite it."""
    assert "region_timeout_per_neighbour_sec" not in _search_params()
    assert _search_params(timeout_per_neighbour_sec=2.5)[
        "region_timeout_per_neighbour_sec"
    ] == 2.5


def test_both_entry_points_name_the_same_options():
    """The regression: holding a boundary must not search differently."""
    import inspect

    whole = inspect.signature(NAMOPlanningService.plan_from_xml).parameters
    held = inspect.signature(NAMOPlanningService.solve_boundary_from_xml).parameters
    shared = set(inspect.signature(_region_search_params).parameters)

    assert shared <= set(whole)
    assert shared <= set(held)
    for name in shared:
        assert whole[name].default == held[name].default, name


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
