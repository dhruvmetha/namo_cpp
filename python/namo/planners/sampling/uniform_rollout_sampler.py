"""Uniform Rollout Sampler — fresh 1-push F-characterization with chain-extendable schema.

v0 collects depth-0 exhaustive data only. The schema (TransitionRecord, EnvMetadata,
SamplerAttemptResult) is designed so a follow-up spec can append depth-1 / depth-2
records without breaking existing readers. See docs/superpowers/specs/
2026-05-19-uniform-rollout-sampler-design.md for the design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import namo_rl
from namo.core import BasePlanner, PlannerConfig, PlannerFactory, PlannerResult

# The robot appears in env.get_reachable_objects() but is never a valid push target.
# Match the convention used in region_opening's object selection.
_ROBOT_NAME_PATTERNS = ("robot", "car")


def _is_pushable_object(object_id: str) -> bool:
    """True iff this object is a candidate to be pushed (not the robot itself)."""
    lower = object_id.lower()
    return not any(pattern in lower for pattern in _ROBOT_NAME_PATTERNS)


def enumerate_reachable_primitives(
    env: namo_rl.RLEnvironment,
    num_depths: int = 10,
) -> List[Tuple[str, int, int]]:
    """Return every (object_id, edge_idx, push_depth_idx) the robot can physically attempt.

    Args:
        env: NAMO RL environment positioned at the state to enumerate from.
        num_depths: Number of push depths per edge (matches motion-primitive resolution; 10 in the
            existing F-char data).

    Returns:
        Sorted list of (object_id, edge_idx, depth_idx) tuples. Sorting is deterministic
        for reproducibility.

    Notes:
        - The robot itself (object name containing "robot" or "car") is excluded.
        - Objects with empty reachable_edges contribute nothing.
    """
    reachable_objects = [o for o in env.get_reachable_objects() if _is_pushable_object(o)]

    prims: List[Tuple[str, int, int]] = []
    for obj in sorted(reachable_objects):
        edges = env.get_reachable_edges(obj)
        for edge_idx in sorted(edges):
            for depth_idx in range(num_depths):
                prims.append((obj, edge_idx, depth_idx))
    return prims


@dataclass
class TransitionRecord:
    """One push attempt with its outcome.

    At v0 (depth-0 only), all records have depth=0 and parent_id=None.
    Fields are kept for chain-extendability — when the follow-up spec adds
    depth-1/2, the same record shape carries the chain bookkeeping.
    """

    transition_id: int  # unique within env, dense from 0
    parent_id: Optional[int]  # always None at v0
    depth: int  # always 0 at v0
    object_id: str
    edge_idx: int  # 0..59
    push_depth_idx: int  # 0..9
    target_pose: Tuple[float, float, float]  # (x, y, θ) the push aimed for
    r: int  # 0 or 1; is_robot_goal_reachable(state_after)
    per_neighbor_opening: Dict[str, bool]  # {neighbor_label: opened?}
    wall_collision: bool
    movable_collisions: List[str]  # object_ids hit
    push_terminated_early: bool
    sim_failure: bool  # True iff env.step raised
    sim_time_ms: float
    state_after_se2: Dict[str, Tuple[float, float, float]]


@dataclass
class EnvMetadata:
    """Per-env context shared across all transitions in that env.

    At v0 all transitions in an env share state_before = initial scene, so
    initial_state_se2 + per_neighbor_region_goals live here (env-level).
    """

    xml_file: str
    robot_goal: Tuple[float, float, float]
    initial_state_se2: Dict[str, Tuple[float, float, float]]
    per_neighbor_region_goals: Dict[str, List[Tuple[float, float, float]]]
    neighbor_labels: List[str]  # all neighbors of the robot's region
    static_object_info: Dict[str, Dict[str, Any]]
    collection_timestamp_utc: str
    sampler_version: str


@dataclass
class SamplerAttemptResult:
    """Per-(object, neighbor) result shaped to match region_opening's AttemptResult.

    The worker code at modular_parallel_collection.py:432-515 reads these fields
    to build one ModularEpisodeResult per attempt. By exposing the same field
    set, the sampler reuses the existing worker branch and the existing
    batch_collection_classifier.py post-processing unchanged.
    """

    success: bool
    neighbour_region_label: str
    chosen_object_id: Optional[str] = None
    chosen_goal: Optional[Tuple[float, float, float]] = None
    region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None
    region_goal_used: Optional[Tuple[float, float, float]] = None
    primitive_trial_log: Optional[List[Dict[str, Any]]] = None
    chain_depth: int = 1
    timing_ms: Optional[float] = None
    state_observations: Optional[List[Dict[str, List[float]]]] = None
    post_action_state_observations: Optional[List[Dict[str, List[float]]]] = None
    reachable_objects_before_action: Optional[List[List[str]]] = None
    reachable_objects_after_action: Optional[List[List[str]]] = None
    error_message: Optional[str] = None
    # Mirrored from region_opening AttemptResult for cross-compat;
    # populated where applicable, None otherwise.
    validation_method: str = "connectivity"
    connectivity_before: Optional[Dict[str, Any]] = None
    connectivity_after: Optional[Dict[str, Any]] = None
    goal_chain: Optional[List[Any]] = None  # always None at v0
    any_wall_collision: bool = False
    unique_movable_collision_count: int = 0


class UniformRolloutSampler(BasePlanner):
    """Exhaustively executes every reachable push primitive at the initial scene.

    Does no search. Logs every outcome. Output flows through the existing
    region_opening-style worker branch by emitting one AttemptResult per
    (object, neighbor) pair.
    """

    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        super().__init__(env, config)

    def _setup_constraints(self) -> None:
        # No constraints needed: every reachable primitive is enumerated.
        pass

    def _initialize_algorithm(self) -> None:
        # v0 has no internal algorithm state — every search() call is independent.
        pass

    def reset(self) -> None:
        pass

    @property
    def algorithm_name(self) -> str:
        return "uniform_rollout_sampler"

    @property
    def algorithm_version(self) -> str:
        return "0.1.0"

    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        # Skeleton: implemented in later tasks. Returns empty result for now.
        return PlannerResult(
            success=False,
            solution_found=False,
            action_sequence=None,
            algorithm_stats={"attempt_results": []},
        )


PlannerFactory.register_planner("uniform_rollout_sampler", UniformRolloutSampler)
