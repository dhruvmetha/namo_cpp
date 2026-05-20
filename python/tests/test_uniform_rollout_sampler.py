"""Tests for UniformRolloutSampler."""

from dataclasses import asdict

import namo.planners.sampling.uniform_rollout_sampler  # noqa: F401 — registers on import
from namo.core import PlannerFactory
from namo.planners.sampling.uniform_rollout_sampler import (
    EnvMetadata,
    SamplerAttemptResult,
    TransitionRecord,
)


def test_uniform_rollout_sampler_is_registered():
    available = PlannerFactory.list_available_planners()
    assert "uniform_rollout_sampler" in available


def test_transition_record_roundtrip():
    rec = TransitionRecord(
        transition_id=0,
        parent_id=None,
        depth=0,
        object_id="obj_1",
        edge_idx=5,
        push_depth_idx=3,
        target_pose=(0.1, 0.2, 0.3),
        r=1,
        per_neighbor_opening={"neighbor_A": True, "neighbor_B": False},
        wall_collision=False,
        movable_collisions=[],
        push_terminated_early=False,
        sim_failure=False,
        sim_time_ms=12.3,
        state_after_se2={"obj_1": (0.1, 0.2, 0.3), "robot": (0.0, 0.0, 0.0)},
    )
    d = asdict(rec)
    assert d["transition_id"] == 0
    assert d["depth"] == 0
    assert d["r"] == 1
    assert d["per_neighbor_opening"]["neighbor_A"] is True


def test_env_metadata_fields():
    md = EnvMetadata(
        xml_file="/tmp/env.xml",
        robot_goal=(0.0, 0.0, 0.0),
        initial_state_se2={"obj_1": (0.0, 0.0, 0.0)},
        per_neighbor_region_goals={"neighbor_A": [(0.1, 0.2, 0.0)]},
        neighbor_labels=["robot_region_0", "neighbor_A"],
        static_object_info={"obj_1": {"size_x": 0.05, "size_y": 0.05}},
        collection_timestamp_utc="2026-05-20T00:00:00Z",
        sampler_version="0.1.0",
    )
    d = asdict(md)
    assert d["xml_file"] == "/tmp/env.xml"
    assert d["per_neighbor_region_goals"]["neighbor_A"] == [(0.1, 0.2, 0.0)]


def test_sampler_attempt_result_mirrors_region_opening():
    """SamplerAttemptResult must expose the fields the existing worker branch reads.

    See modular_parallel_collection.py lines 432-515 for the consumed fields.
    """
    attempt = SamplerAttemptResult(
        success=True,
        neighbour_region_label="neighbor_A",
        chosen_object_id="obj_1",
        chosen_goal=(0.1, 0.2, 0.0),
        region_goals_sampled=[(0.1, 0.2, 0.0), (0.15, 0.2, 0.0)],
        region_goal_used=(0.1, 0.2, 0.0),
        primitive_trial_log=[{"edge_idx": 0, "depth": 0, "success": True,
                              "wall_collision": False, "movable_collisions": "",
                              "stuck": False, "collision": False,
                              "reachable_after": 1}],
        chain_depth=1,
        timing_ms=42.0,
        state_observations=[{"obj_1": [0.0, 0.0, 0.0]}],
        post_action_state_observations=[{"obj_1": [0.1, 0.2, 0.0]}],
        reachable_objects_before_action=[["obj_1"]],
        reachable_objects_after_action=[["obj_1"]],
    )
    d = asdict(attempt)
    # Worker reads these in modular_parallel_collection.py line 432-515:
    for required in ("success", "neighbour_region_label", "chosen_object_id",
                     "chosen_goal", "region_goals_sampled", "region_goal_used",
                     "primitive_trial_log", "chain_depth", "timing_ms",
                     "state_observations", "post_action_state_observations",
                     "reachable_objects_before_action", "reachable_objects_after_action"):
        assert required in d, f"missing required field: {required}"


from unittest.mock import MagicMock


def test_enumerate_reachable_primitives_combines_objects_edges_depths():
    """Enumeration is the Cartesian product of (reachable objects) × (their reachable edges) × (depths 0..9)."""
    from namo.planners.sampling.uniform_rollout_sampler import enumerate_reachable_primitives

    env = MagicMock()
    env.get_reachable_objects.return_value = ["obj_1", "obj_2"]
    env.get_reachable_edges.side_effect = lambda name: {
        "obj_1": [0, 1, 2],
        "obj_2": [10],
    }[name]

    NUM_DEPTHS = 10
    prims = enumerate_reachable_primitives(env, num_depths=NUM_DEPTHS)

    # obj_1: 3 edges × 10 depths = 30
    # obj_2: 1 edge × 10 depths = 10
    assert len(prims) == 40

    # Deterministic ordering: sorted by (object_id, edge_idx, depth_idx)
    assert prims[0] == ("obj_1", 0, 0)
    assert prims[1] == ("obj_1", 0, 1)
    assert prims[9] == ("obj_1", 0, 9)
    assert prims[10] == ("obj_1", 1, 0)
    assert prims[30] == ("obj_2", 10, 0)
    assert prims[-1] == ("obj_2", 10, 9)


def test_enumerate_reachable_primitives_excludes_robot():
    """Robot is in get_reachable_objects but should not be a pushable object."""
    from namo.planners.sampling.uniform_rollout_sampler import enumerate_reachable_primitives

    env = MagicMock()
    env.get_reachable_objects.return_value = ["robot", "obj_1"]
    env.get_reachable_edges.side_effect = lambda name: {"obj_1": [0]}[name]

    prims = enumerate_reachable_primitives(env, num_depths=10)

    assert all(p[0] != "robot" for p in prims)
    assert len(prims) == 10


def test_enumerate_reachable_primitives_handles_no_reachable_edges():
    """Object with empty reachable_edges contributes nothing."""
    from namo.planners.sampling.uniform_rollout_sampler import enumerate_reachable_primitives

    env = MagicMock()
    env.get_reachable_objects.return_value = ["obj_1", "obj_2"]
    env.get_reachable_edges.side_effect = lambda name: {"obj_1": [], "obj_2": [0]}[name]

    prims = enumerate_reachable_primitives(env, num_depths=10)

    assert len(prims) == 10
    assert all(p[0] == "obj_2" for p in prims)


from typing import Tuple


def _make_action(object_id: str, target: Tuple[float, float, float],
                 edge_idx: int, depth: int):
    """Helper: build a namo_rl.Action shaped object for tests."""
    a = MagicMock()
    a.object_id = object_id
    a.x = target[0]
    a.y = target[1]
    a.theta = target[2]
    a.edge_idx = edge_idx
    a.depth = depth
    return a


def test_execute_primitive_returns_partial_record_with_outcome():
    """execute_primitive runs env.step, captures wall_collision/stuck/movable_collisions from info."""
    from namo.planners.sampling.uniform_rollout_sampler import execute_primitive

    env = MagicMock()
    initial_state = MagicMock()
    # set_full_state returns nothing; env.step returns a StepResult-like with info dict
    step_result = MagicMock()
    step_result.info = {
        "wall_collision": "true",
        "stuck": "false",
        "movable_collisions": "obj_2",
        "robot_goal_reached": "true",
    }
    env.step.return_value = step_result
    env.is_robot_goal_reachable.return_value = True
    env.get_observation.return_value = {"obj_1": [0.5, 0.5, 0.1]}

    partial = execute_primitive(
        env=env,
        initial_state=initial_state,
        object_id="obj_1",
        edge_idx=3,
        push_depth_idx=5,
        target_pose=(0.5, 0.5, 0.1),
    )

    assert partial["object_id"] == "obj_1"
    assert partial["edge_idx"] == 3
    assert partial["push_depth_idx"] == 5
    assert partial["r"] == 1
    assert partial["wall_collision"] is True
    assert partial["push_terminated_early"] is False
    assert partial["movable_collisions"] == ["obj_2"]
    assert partial["sim_failure"] is False
    assert partial["state_after_se2"] == {"obj_1": (0.5, 0.5, 0.1)}
    env.set_full_state.assert_called_with(initial_state)
    env.step.assert_called_once()


def test_execute_primitive_catches_sim_failure():
    """If env.step raises, partial record has sim_failure=True and r=0."""
    from namo.planners.sampling.uniform_rollout_sampler import execute_primitive

    env = MagicMock()
    env.step.side_effect = RuntimeError("contact resolver failed")

    partial = execute_primitive(
        env=env,
        initial_state=MagicMock(),
        object_id="obj_1",
        edge_idx=0,
        push_depth_idx=0,
        target_pose=(0.0, 0.0, 0.0),
    )
    assert partial["sim_failure"] is True
    assert partial["r"] == 0
    assert partial["wall_collision"] is False
