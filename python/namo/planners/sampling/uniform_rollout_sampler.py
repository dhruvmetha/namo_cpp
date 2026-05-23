"""Uniform Rollout Sampler — fresh 1-push F-characterization with chain-extendable schema.

v0 collects depth-0 exhaustive data only. The schema (TransitionRecord, EnvMetadata,
SamplerAttemptResult) is designed so a follow-up spec can append depth-1 / depth-2
records without breaking existing readers. See docs/superpowers/specs/
2026-05-19-uniform-rollout-sampler-design.md for the design.
"""

from __future__ import annotations

import datetime as _datetime
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import namo_rl
from namo.core import BasePlanner, PlannerConfig, PlannerFactory, PlannerResult
from namo.planners.connectivity_snapshot import find_robot_label, snapshot_region_connectivity

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


def _parse_movable_collisions(raw: str) -> List[str]:
    """info['movable_collisions'] is a comma-separated string of object IDs (or empty)."""
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _se2_from_observation(obs: Dict[str, List[float]]) -> Dict[str, Tuple[float, float, float]]:
    """Convert env.get_observation() output to per-object SE(2) tuples."""
    out: Dict[str, Tuple[float, float, float]] = {}
    for name, pose in obs.items():
        if pose is None or len(pose) < 3:
            continue
        out[name] = (float(pose[0]), float(pose[1]), float(pose[2]))
    return out


def execute_primitive(
    env: namo_rl.RLEnvironment,
    initial_state: "namo_rl.RLState",
    object_id: str,
    edge_idx: int,
    push_depth_idx: int,
    target_pose: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Execute one primitive from `initial_state` and return a partial transition record.

    The partial record is missing `transition_id`, `per_neighbor_opening`, and the
    canonical dataclass wrap — the caller composes the final `TransitionRecord`.

    Restores env to `initial_state` before stepping. Captures wall_collision,
    movable_collisions, stuck, and the per-object SE(2) after the push. If the
    underlying sim raises, returns a record with sim_failure=True and r=0.
    """
    env.set_full_state(initial_state)

    action = namo_rl.Action()
    action.object_id = object_id
    action.x = float(target_pose[0])
    action.y = float(target_pose[1])
    action.theta = float(target_pose[2])
    action.edge_idx = int(edge_idx)
    action.depth = int(push_depth_idx)

    t0 = _time.perf_counter()
    sim_failure = False
    info: Dict[str, str] = {}
    try:
        step_result = env.step(action)
        info = dict(step_result.info or {})
    except Exception:
        sim_failure = True

    sim_time_ms = (_time.perf_counter() - t0) * 1000.0

    if sim_failure:
        return {
            "object_id": object_id,
            "edge_idx": edge_idx,
            "push_depth_idx": push_depth_idx,
            "target_pose": tuple(target_pose),
            "r": 0,
            "wall_collision": False,
            "movable_collisions": [],
            "push_terminated_early": False,
            "sim_failure": True,
            "sim_time_ms": sim_time_ms,
            "state_after_se2": {},
        }

    r = 1 if env.is_robot_goal_reachable() else 0
    wall_collision = info.get("wall_collision", "false").lower() == "true"
    push_terminated_early = info.get("stuck", "false").lower() == "true"
    movable_collisions = _parse_movable_collisions(info.get("movable_collisions", ""))
    state_after_se2 = _se2_from_observation(env.get_observation())

    return {
        "object_id": object_id,
        "edge_idx": edge_idx,
        "push_depth_idx": push_depth_idx,
        "target_pose": tuple(target_pose),
        "r": r,
        "wall_collision": wall_collision,
        "movable_collisions": movable_collisions,
        "push_terminated_early": push_terminated_early,
        "sim_failure": False,
        "sim_time_ms": sim_time_ms,
        "state_after_se2": state_after_se2,
    }


def _evaluate_opening_from_snapshots(
    before_labels: Dict[int, str],
    before_adjacency: Dict[str, Any],
    after_labels: Dict[int, str],
    after_adjacency: Dict[str, Any],
) -> Dict[str, bool]:
    """Diff two connectivity snapshots to determine per-neighbor opening.

    A neighbor X of the robot's region at state_before is 'opened' iff X no longer
    appears as a distinct region label at state_after — it merged into the robot's
    region (the passage between robot and X became open).

    Returns a dict mapping each neighbor label seen at state_before to a bool.
    Returns {} if robot label is missing at state_before (degenerate env).
    """
    robot_label = find_robot_label(before_labels)
    if robot_label is None:
        return {}

    neighbors_before = set(before_adjacency.get(robot_label, set()))
    if not neighbors_before:
        return {}

    after_label_set = set(after_labels.values())
    return {n: (n not in after_label_set) for n in neighbors_before}


def group_transitions_into_attempts(
    transitions: List[Dict[str, Any]],
    neighbor_labels: List[str],
    region_goals: Dict[str, List[Tuple[float, float, float]]],
    initial_observation: Dict[str, List[float]],
    reachable_objects_before: List[str],
) -> List["SamplerAttemptResult"]:
    """Build one SamplerAttemptResult per (object, neighbor) pair.

    Each attempt's primitive_trial_log contains the subset of transitions involving
    that object, with success labeled per-neighbor (true iff the push opened
    THIS neighbor).

    Args:
        transitions: Flat list of dicts as produced by execute_primitive() +
            per_neighbor_opening dict added by the caller.
        neighbor_labels: All neighbor labels for the robot region at s₀.
        region_goals: Per-neighbor sampled goal points (for the region_goals_sampled
            field consumed by NAMODataVisualizer).
        initial_observation: env.get_observation() result at s₀, used to populate
            the single-entry state_observations list (matches how exhaustive-mode
            region_opening fills it).
        reachable_objects_before: env.get_reachable_objects() at s₀.

    Returns:
        List of SamplerAttemptResult, one per (object, neighbor). The order is
        sorted by (object_id, neighbor_label) for reproducibility.
    """
    objects = sorted({t["object_id"] for t in transitions})
    attempts: List["SamplerAttemptResult"] = []

    for obj in objects:
        obj_transitions = [t for t in transitions if t["object_id"] == obj]
        for neighbor in sorted(neighbor_labels):
            # Build the trial_log for this (obj, neighbor) using the existing F-char schema.
            trial_log: List[Dict[str, Any]] = []
            success_target: Optional[Tuple[float, float, float]] = None
            any_wall = False
            unique_movable: set = set()

            for t in obj_transitions:
                opened = bool(t["per_neighbor_opening"].get(neighbor, False))
                entry = {
                    "edge_idx": int(t["edge_idx"]),
                    "depth": int(t["push_depth_idx"]),
                    "success": opened,
                    "wall_collision": bool(t["wall_collision"]),
                    "movable_collisions": ",".join(t["movable_collisions"]),
                    "stuck": bool(t["push_terminated_early"]),
                    "collision": bool(t["wall_collision"] or t["movable_collisions"]),
                    "reachable_after": int(t["r"]),
                }
                trial_log.append(entry)
                if opened and success_target is None:
                    success_target = tuple(t["target_pose"])
                if t["wall_collision"]:
                    any_wall = True
                for mc in t["movable_collisions"]:
                    unique_movable.add(mc)

            sampled_goals = region_goals.get(neighbor)
            attempts.append(SamplerAttemptResult(
                success=success_target is not None,
                neighbour_region_label=neighbor,
                chosen_object_id=obj,
                chosen_goal=success_target,
                region_goals_sampled=sampled_goals,
                region_goal_used=(sampled_goals[0] if sampled_goals else None),
                primitive_trial_log=trial_log,
                chain_depth=1,
                state_observations=[initial_observation],
                post_action_state_observations=None,
                reachable_objects_before_action=[reachable_objects_before],
                reachable_objects_after_action=None,
                any_wall_collision=any_wall,
                unique_movable_collision_count=len(unique_movable),
            ))

    return attempts


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

    See module docstring + docs/superpowers/specs/2026-05-19-uniform-rollout-sampler-design.md.
    """

    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        super().__init__(env, config)

    def _setup_constraints(self) -> None:
        pass

    def _initialize_algorithm(self) -> None:
        algo_params = self.config.algorithm_params or {}
        self.max_chain_depth: int = int(algo_params.get("max_chain_depth", 1))
        if self.max_chain_depth < 1 or self.max_chain_depth > 10:
            raise ValueError(
                f"max_chain_depth must be in [1, 10]; got {self.max_chain_depth}"
            )
        # Chain-mode knobs (used when max_chain_depth > 1).
        self.num_rollouts_per_env: int = int(algo_params.get("num_rollouts_per_env", 10000))
        self.max_solutions_per_env: int = int(algo_params.get("max_solutions_per_env", 1))
        if self.num_rollouts_per_env < 1:
            raise ValueError(f"num_rollouts_per_env must be >= 1; got {self.num_rollouts_per_env}")
        if self.max_solutions_per_env < 1:
            raise ValueError(f"max_solutions_per_env must be >= 1; got {self.max_solutions_per_env}")
        self.region_goal_samples_per_neighbor: int = int(
            algo_params.get("region_goal_samples_per_neighbor", 5)
        )
        self.num_depths: int = int(algo_params.get("num_depths", 10))
        self.xml_file: Optional[str] = algo_params.get("xml_file")
        self.config_file: Optional[str] = algo_params.get("config_file_path")
        # Per-env determinism: seed = base + env_idx. The collection script injects
        # env_idx per-task (mirror of how xml_file is injected). Without env_idx,
        # all envs in a worker share the same seed (legacy v0 behaviour).
        seed_base = int(algo_params.get("seed", 42))
        env_idx = int(algo_params.get("env_idx", 0))
        self.seed: int = seed_base + env_idx

    def reset(self) -> None:
        pass

    @property
    def algorithm_name(self) -> str:
        return "uniform_rollout_sampler"

    @property
    def algorithm_version(self) -> str:
        # 0.1.0 = v0 (depth-0 exhaustive)
        # 1.0.0 = v1 (chain mode added; n=1 still gives v0 behaviour)
        return "1.0.0"

    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        import numpy as np
        t_start = _time.perf_counter()

        # 1. Set the robot goal and snapshot the initial state.
        self.env.set_robot_goal(float(robot_goal[0]), float(robot_goal[1]), float(robot_goal[2]))
        initial_state = self.env.get_full_state()
        initial_observation = self.env.get_observation()
        initial_se2 = _se2_from_observation(initial_observation)
        reachable_objects_initial = list(self.env.get_reachable_objects())

        # 2. Compute connectivity at s₀, sampling K region goals per neighbor.
        rng = np.random.default_rng(self.seed)
        try:
            before_adj, _edge_objs, before_labels, before_region_goals, _snap = snapshot_region_connectivity(
                self.env,
                xml_path=self.xml_file or "",
                config_path=self.config_file or "",
                goals_per_region=self.region_goal_samples_per_neighbor,
                generate_training_data=True,
                local_info_only=False,
                use_current_state=True,
                rng=rng,
            )
        except Exception as exc:
            return PlannerResult(
                success=False, solution_found=False, action_sequence=None,
                algorithm_stats={"attempt_results": []},
                error_message=f"connectivity snapshot at s₀ failed: {exc}",
                search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
            )

        robot_label = find_robot_label(before_labels)
        if robot_label is None:
            return PlannerResult(
                success=False, solution_found=False, action_sequence=None,
                algorithm_stats={"attempt_results": []},
                error_message="no robot region label at s₀",
                search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
            )
        neighbor_labels = sorted(before_adj.get(robot_label, set()))

        # Convert region_goals to plain tuples (per-neighbor, only non-robot labels).
        region_goals: Dict[str, List[Tuple[float, float, float]]] = {}
        for nbr in neighbor_labels:
            bundle = before_region_goals.get(nbr)
            if bundle is None:
                region_goals[nbr] = []
                continue
            region_goals[nbr] = [(g.x, g.y, g.theta) for g in bundle.goals]

        # 3a. Chain mode (max_chain_depth > 1): N random rollouts per blocking object
        # on the R₀ ↔ "goal" boundary. Dispatches to its own code path; v0
        # exhaustive logic continues below for max_chain_depth == 1.
        if self.max_chain_depth > 1:
            return self._run_chain_mode(
                robot_goal=robot_goal,
                initial_state=initial_state,
                initial_observation=initial_observation,
                initial_se2=initial_se2,
                reachable_objects_initial=reachable_objects_initial,
                before_adj=before_adj,
                before_labels=before_labels,
                edge_objects=_edge_objs,
                robot_label=robot_label,
                neighbor_labels=neighbor_labels,
                region_goals=region_goals,
                t_start=t_start,
            )

        # 3. Enumerate every reachable primitive at s₀.
        primitives = enumerate_reachable_primitives(self.env, num_depths=self.num_depths)
        if not primitives:
            return PlannerResult(
                success=False, solution_found=False, action_sequence=None,
                algorithm_stats={"attempt_results": []},
                error_message="no reachable primitives at s₀",
                search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
            )

        # 4. For each primitive: execute, snapshot state_after, diff for per-neighbor opening.
        transitions: List[Dict[str, Any]] = []
        for tid, (obj, edge_idx, depth_idx) in enumerate(primitives):
            # Target pose is the primitive's calibrated landing pose; we don't have
            # it without running the goal strategy. The C++ skill resolves the target
            # from (object, edge_idx, depth) internally — we pass placeholder zeros and
            # set edge_idx/depth on the Action so the skill uses the database.
            target_pose = (0.0, 0.0, 0.0)
            partial = execute_primitive(
                env=self.env,
                initial_state=initial_state,
                object_id=obj,
                edge_idx=edge_idx,
                push_depth_idx=depth_idx,
                target_pose=target_pose,
            )

            # Snapshot state_after for per-neighbor opening.
            try:
                after_adj, _eo, after_labels, _rg, _snap = snapshot_region_connectivity(
                    self.env,
                    xml_path=self.xml_file or "",
                    config_path=self.config_file or "",
                    goals_per_region=0,
                    generate_training_data=False,
                    local_info_only=False,
                    use_current_state=True,
                )
                per_neighbor = _evaluate_opening_from_snapshots(
                    before_labels=before_labels, before_adjacency=before_adj,
                    after_labels=after_labels, after_adjacency=after_adj,
                )
            except Exception:
                per_neighbor = {n: False for n in neighbor_labels}

            partial["per_neighbor_opening"] = per_neighbor
            partial["transition_id"] = tid
            transitions.append(partial)

        # Restore env to s₀ once at the end so downstream code sees a stable state.
        self.env.set_full_state(initial_state)

        # 5. Group into per-(object, neighbor) AttemptResults.
        attempts = group_transitions_into_attempts(
            transitions=transitions,
            neighbor_labels=neighbor_labels,
            region_goals=region_goals,
            initial_observation=initial_observation,
            reachable_objects_before=reachable_objects_initial,
        )

        # 6. Assemble env metadata.
        env_meta = EnvMetadata(
            xml_file=self.xml_file or "",
            robot_goal=tuple(robot_goal),
            initial_state_se2=initial_se2,
            per_neighbor_region_goals=region_goals,
            neighbor_labels=neighbor_labels,
            static_object_info=dict(self.env.get_object_info()),
            collection_timestamp_utc=_datetime.datetime.utcnow().isoformat() + "Z",
            sampler_version=self.algorithm_version,
        )

        # 7. Wrap into a PlannerResult.
        from dataclasses import asdict as _asdict
        return PlannerResult(
            success=any(a.success for a in attempts),
            solution_found=False,                         # sampler doesn't 'solve' anything
            action_sequence=None,
            algorithm_stats={
                "attempt_results": attempts,             # consumed by the worker branch
                "env_metadata": _asdict(env_meta),
                "sampler_summary_stats": {
                    "n_transitions": len(transitions),
                    "n_attempts": len(attempts),
                    "n_r1": sum(1 for t in transitions if t["r"] == 1),
                    "n_sim_failures": sum(1 for t in transitions if t["sim_failure"]),
                },
            },
            search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
        )

    # ------------------------------------------------------------------
    # Chain mode (v1): N random rollouts per blocking object on R₀↔goal.
    # ------------------------------------------------------------------

    def _run_chain_mode(
        self,
        *,
        robot_goal: Tuple[float, float, float],
        initial_state: "namo_rl.RLState",
        initial_observation: Dict[str, List[float]],
        initial_se2: Dict[str, Tuple[float, float, float]],
        reachable_objects_initial: List[str],
        before_adj: Dict[str, Any],
        before_labels: Dict[int, str],
        edge_objects: Dict[str, Dict[str, Any]],
        robot_label: str,
        neighbor_labels: List[str],
        region_goals: Dict[str, List[Tuple[float, float, float]]],
        t_start: float,
    ) -> PlannerResult:
        """v1 chain mode: random n-push rollouts per (blocking object, 'goal').

        Per env:
        - Identify the "goal" region (must be immediate neighbor of R₀).
        - Identify blocking objects on the R₀↔goal boundary.
        - For each blocking object: run up to N rollouts of up to n random pushes.
        - Per rollout: stop on goal opened, sim_failure, or no reachable primitives.
          Stuck does NOT stop.
        - Per (object, goal): stop after S solutions or N rollouts.
        """
        import numpy as np

        # 1. Goal region must be an immediate neighbor.
        if "goal" not in neighbor_labels:
            attempt = SamplerAttemptResult(
                success=False,
                neighbour_region_label="goal",
                chosen_object_id=None,
                primitive_trial_log=[],
                state_observations=[initial_observation],
                reachable_objects_before_action=[reachable_objects_initial],
                error_message="goal region not an immediate neighbor of robot region at s_0",
            )
            attempt.timing_ms = (_time.perf_counter() - t_start) * 1000.0
            attempt.any_wall_collision = False
            attempt.unique_movable_collision_count = 0
            # Attach failure reason via setattr (field exists on region_opening's
            # AttemptResult but not on SamplerAttemptResult; getattr in worker is safe).
            try:
                attempt.failure_reason = "goal_not_immediate_neighbor"
            except Exception:
                pass
            return PlannerResult(
                success=False, solution_found=False, action_sequence=None,
                algorithm_stats={
                    "attempt_results": [attempt],
                    "sampler_summary_stats": {
                        "mode": "chain", "n_rollouts": 0, "n_solutions": 0,
                        "n_pushes": 0, "n_blocking_objects": 0,
                    },
                },
                search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
            )

        # 2. Boundary objects on R₀ ↔ "goal" edge.
        # edge_objects shape: Dict[str, Dict[str, Set[str]]]
        forward = edge_objects.get(robot_label, {}).get("goal", set()) or set()
        reverse = edge_objects.get("goal", {}).get(robot_label, set()) or set()
        boundary_objects = sorted({str(o) for o in (forward | reverse)})

        if not boundary_objects:
            attempt = SamplerAttemptResult(
                success=False,
                neighbour_region_label="goal",
                chosen_object_id=None,
                primitive_trial_log=[],
                state_observations=[initial_observation],
                reachable_objects_before_action=[reachable_objects_initial],
                error_message="no blocking objects on robot↔goal boundary",
            )
            attempt.timing_ms = (_time.perf_counter() - t_start) * 1000.0
            attempt.any_wall_collision = False
            attempt.unique_movable_collision_count = 0
            try:
                attempt.failure_reason = "no_blocking_objects"
            except Exception:
                pass
            return PlannerResult(
                success=False, solution_found=False, action_sequence=None,
                algorithm_stats={
                    "attempt_results": [attempt],
                    "sampler_summary_stats": {
                        "mode": "chain", "n_rollouts": 0, "n_solutions": 0,
                        "n_pushes": 0, "n_blocking_objects": 0,
                    },
                },
                search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
            )

        # 3. RNG and outer per-object loop.
        rng = np.random.default_rng(self.seed)
        attempts: List[SamplerAttemptResult] = []
        total_rollouts = 0
        total_pushes = 0
        total_solutions = 0
        global_transition_id = 0

        goal_region_goals = region_goals.get("goal", [])
        goal_region_goal_used: Optional[Tuple[float, float, float]] = (
            goal_region_goals[0] if goal_region_goals else None
        )

        for committed_object in boundary_objects:
            attempt_start = _time.perf_counter()
            (
                attempt,
                rollouts_for_obj,
                pushes_for_obj,
                solutions_for_obj,
                global_transition_id,
            ) = self._run_rollouts_for_object(
                committed_object=committed_object,
                initial_state=initial_state,
                initial_observation=initial_observation,
                reachable_objects_initial=reachable_objects_initial,
                before_adj=before_adj,
                before_labels=before_labels,
                neighbor_labels=neighbor_labels,
                rng=rng,
                global_transition_id=global_transition_id,
                goal_region_goals=goal_region_goals,
                goal_region_goal_used=goal_region_goal_used,
            )
            attempt.timing_ms = (_time.perf_counter() - attempt_start) * 1000.0
            attempts.append(attempt)
            total_rollouts += rollouts_for_obj
            total_pushes += pushes_for_obj
            total_solutions += solutions_for_obj

        # 4. Env metadata.
        env_meta = EnvMetadata(
            xml_file=self.xml_file or "",
            robot_goal=tuple(robot_goal),
            initial_state_se2=initial_se2,
            per_neighbor_region_goals=region_goals,
            neighbor_labels=neighbor_labels,
            static_object_info=dict(self.env.get_object_info()),
            collection_timestamp_utc=_datetime.datetime.utcnow().isoformat() + "Z",
            sampler_version=self.algorithm_version,
        )

        from dataclasses import asdict as _asdict
        return PlannerResult(
            success=any(a.success for a in attempts),
            solution_found=False,
            action_sequence=None,
            algorithm_stats={
                "attempt_results": attempts,
                "env_metadata": _asdict(env_meta),
                "sampler_summary_stats": {
                    "mode": "chain",
                    "max_chain_depth": self.max_chain_depth,
                    "num_rollouts_per_env": self.num_rollouts_per_env,
                    "max_solutions_per_env": self.max_solutions_per_env,
                    "n_blocking_objects": len(boundary_objects),
                    "n_rollouts": total_rollouts,
                    "n_solutions": total_solutions,
                    "n_pushes": total_pushes,
                },
            },
            search_time_ms=(_time.perf_counter() - t_start) * 1000.0,
        )

    def _run_rollouts_for_object(
        self,
        *,
        committed_object: str,
        initial_state: "namo_rl.RLState",
        initial_observation: Dict[str, List[float]],
        reachable_objects_initial: List[str],
        before_adj: Dict[str, Any],
        before_labels: Dict[int, str],
        neighbor_labels: List[str],
        rng,
        global_transition_id: int,
        goal_region_goals: List[Tuple[float, float, float]],
        goal_region_goal_used: Optional[Tuple[float, float, float]],
    ) -> Tuple["SamplerAttemptResult", int, int, int, int]:
        """Run up to N random rollouts on `committed_object`. Returns
        (attempt, rollouts_executed, pushes_executed, solutions_found, new_global_tid)."""
        rollouts_executed = 0
        pushes_executed = 0
        solutions_found = 0
        primitive_trial_log: List[Dict[str, Any]] = []
        rollout_trace: List[Dict[str, Any]] = []
        solving_chains: List[List[Dict[str, Any]]] = []
        # State observation streams for the FIRST solving chain (worker reads these).
        solving_state_obs: List[Dict[str, List[float]]] = []
        solving_post_state_obs: List[Dict[str, List[float]]] = []
        solving_reachable_before: List[List[str]] = []
        solving_reachable_after: List[List[str]] = []
        any_wall_collision = False
        unique_movable_collisions: set = set()

        for rollout_idx in range(self.num_rollouts_per_env):
            rollouts_executed += 1
            # Restore env to s₀ for fresh start.
            self.env.set_full_state(initial_state)

            chain_records: List[Dict[str, Any]] = []
            chain_state_obs: List[Dict[str, List[float]]] = []
            chain_post_state_obs: List[Dict[str, List[float]]] = []
            chain_reachable_before: List[List[str]] = []
            chain_reachable_after: List[List[str]] = []
            chain_solved = False
            parent_tid: Optional[int] = None

            for chain_step in range(self.max_chain_depth):
                # Enumerate reachable primitives for committed_object only.
                all_reach = enumerate_reachable_primitives(self.env, num_depths=self.num_depths)
                candidates = [
                    (o, e, d) for (o, e, d) in all_reach if o == committed_object
                ]
                if not candidates:
                    # Object wedged / no longer reachable. End this rollout.
                    break

                idx = int(rng.integers(len(candidates)))
                obj, edge_idx, push_depth_idx = candidates[idx]

                # Capture pre-push observation.
                pre_state_obs = self.env.get_observation()
                pre_reachable = list(self.env.get_reachable_objects())

                # Execute primitive in place (no state restore — chain continues).
                current_state = self.env.get_full_state()
                partial = execute_primitive(
                    env=self.env,
                    initial_state=current_state,
                    object_id=obj,
                    edge_idx=edge_idx,
                    push_depth_idx=push_depth_idx,
                    target_pose=(0.0, 0.0, 0.0),
                )
                pushes_executed += 1

                # Cheap stop signal: partial["r"] is is_robot_goal_reachable() result
                # from execute_primitive. When "goal" is an immediate neighbor of R₀
                # (precondition for this code path), r=1 iff goal region is now open.
                # Avoids expensive per-push connectivity snapshots.
                goal_now_reachable = bool(partial["r"] == 1)
                per_neighbor = {n: False for n in neighbor_labels}
                per_neighbor["goal"] = goal_now_reachable

                # Capture post-push observation.
                post_state_obs = self.env.get_observation()
                post_reachable = list(self.env.get_reachable_objects())

                # Build transition record.
                record = {
                    "transition_id": global_transition_id,
                    "parent_id": parent_tid,
                    "chain_depth": chain_step,
                    "rollout_id": rollout_idx,
                    "object_id": obj,
                    "edge_idx": int(edge_idx),
                    "push_depth_idx": int(push_depth_idx),
                    "target_pose": tuple(partial["target_pose"]),
                    "r": int(partial["r"]),
                    "per_neighbor_opening": dict(per_neighbor),
                    "wall_collision": bool(partial["wall_collision"]),
                    "movable_collisions": list(partial["movable_collisions"]),
                    "push_terminated_early": bool(partial["push_terminated_early"]),
                    "sim_failure": bool(partial["sim_failure"]),
                    "sim_time_ms": float(partial["sim_time_ms"]),
                    "state_after_se2": dict(partial["state_after_se2"]),
                }
                chain_records.append(record)
                chain_state_obs.append(pre_state_obs)
                chain_post_state_obs.append(post_state_obs)
                chain_reachable_before.append(pre_reachable)
                chain_reachable_after.append(post_reachable)

                if partial["wall_collision"]:
                    any_wall_collision = True
                for mc in partial["movable_collisions"]:
                    unique_movable_collisions.add(mc)

                parent_tid = global_transition_id
                global_transition_id += 1

                # Stop conditions: goal opened, sim_failure. Stuck continues.
                goal_opened = bool(per_neighbor.get("goal", False))
                if goal_opened:
                    chain_solved = True
                    break
                if partial["sim_failure"]:
                    break

            # Record this rollout's transitions into the global trace.
            rollout_trace.extend(chain_records)

            # primitive_trial_log: chain_depth==0 entry per rollout (the first push).
            # This is what downstream f_grid-filling readers expect — single-push
            # outcomes from s₀ for the committed object.
            if chain_records:
                first = chain_records[0]
                # The legacy `depth` key in primitive_trial_log means cookbook push_depth_idx.
                primitive_trial_log.append({
                    "edge_idx": first["edge_idx"],
                    "depth": first["push_depth_idx"],
                    "chain_depth": 0,
                    "success": bool(first["per_neighbor_opening"].get("goal", False)),
                    "wall_collision": first["wall_collision"],
                    "movable_collisions": ",".join(first["movable_collisions"]),
                    "stuck": first["push_terminated_early"],
                    "collision": bool(first["wall_collision"] or first["movable_collisions"]),
                    "reachable_after": int(first["r"]),
                })

            if chain_solved:
                solutions_found += 1
                solving_chains.append(chain_records)
                if not solving_state_obs:
                    # Record observations for the FIRST solving chain.
                    solving_state_obs = chain_state_obs
                    solving_post_state_obs = chain_post_state_obs
                    solving_reachable_before = chain_reachable_before
                    solving_reachable_after = chain_reachable_after

            if solutions_found >= self.max_solutions_per_env:
                break

        # Restore env at the end so downstream code sees a stable state.
        self.env.set_full_state(initial_state)

        # Build AttemptResult.
        solved = solutions_found > 0
        if solved:
            chosen_chain = solving_chains[0]
            last_record = chosen_chain[-1]
            chosen_goal = tuple(last_record["target_pose"])
            chain_depth = len(chosen_chain)
            goal_chain = [_ChainGoal.from_record(r) for r in chosen_chain]
            failure_reason = "success"
        else:
            chosen_goal = None
            chain_depth = 0
            goal_chain = []
            failure_reason = "all_pushes_failed"

        attempt = SamplerAttemptResult(
            success=solved,
            neighbour_region_label="goal",
            chosen_object_id=committed_object,
            chosen_goal=chosen_goal,
            region_goals_sampled=list(goal_region_goals) if goal_region_goals else None,
            region_goal_used=goal_region_goal_used,
            primitive_trial_log=primitive_trial_log,
            chain_depth=chain_depth,
            state_observations=solving_state_obs or [initial_observation],
            post_action_state_observations=solving_post_state_obs or None,
            reachable_objects_before_action=solving_reachable_before or [reachable_objects_initial],
            reachable_objects_after_action=solving_reachable_after or None,
            any_wall_collision=any_wall_collision,
            unique_movable_collision_count=len(unique_movable_collisions),
        )
        # Per-attempt extras (read by worker via getattr).
        attempt.goal_chain = goal_chain if goal_chain else None
        try:
            attempt.failure_reason = failure_reason
            attempt.solutions_found_for_neighbour = solutions_found
            attempt.solutions_total_for_neighbour = solutions_found
            attempt.pushes_total_for_neighbour = pushes_executed
            attempt.rollouts_total_for_neighbour = rollouts_executed
            attempt.rollout_trace = rollout_trace
        except Exception:
            pass

        return attempt, rollouts_executed, pushes_executed, solutions_found, global_transition_id


@dataclass
class _ChainGoal:
    """Goal-shaped record consumed by the worker when building action_sequence.

    Mirrors the fields the worker reads from `goal_chain[i]`:
    `.x, .y, .theta, .edge_idx, .depth`. `depth` here is cookbook push_depth_idx
    (matches region_opening's Goal convention).
    """
    x: float
    y: float
    theta: float
    edge_idx: int
    depth: int  # cookbook push_depth_idx

    @classmethod
    def from_record(cls, rec: Dict[str, Any]) -> "_ChainGoal":
        tp = rec["target_pose"]
        return cls(
            x=float(tp[0]), y=float(tp[1]), theta=float(tp[2]),
            edge_idx=int(rec["edge_idx"]),
            depth=int(rec["push_depth_idx"]),
        )


PlannerFactory.register_planner("uniform_rollout_sampler", UniformRolloutSampler)
