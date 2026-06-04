"""Full NAMO Planner - Hierarchical planning using region opening as sub-problem.

This planner solves the full NAMO problem (reaching a specific robot goal) by:
1. Computing the current region connectivity graph
2. Finding a shortest path through regions from robot to goal
3. Opening only the first robot-adjacent boundary on that path
4. Recomputing after each successful opening until the robot goal is reachable
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import namo_rl

from namo.core import BasePlanner, PlannerConfig, PlannerResult
from namo.planners.connectivity_snapshot import find_robot_label
from namo.planners.opening.region_opening import RegionOpeningPlanner


@dataclass
class FullNAMOStats:
    """Statistics for full NAMO planning."""

    iterations: int = 0
    total_pushes: int = 0
    # Includes every simulated push attempt (successful and rejected) made by
    # nested RegionOpeningPlanner calls while solving the full NAMO problem.
    total_attempted_pushes: int = 0
    successful_region_steps: int = 0
    boundary_exhaustions: int = 0
    regions_opened: List[str] = field(default_factory=list)


@dataclass
class RegionOpeningResult:
    """Result from one region opening in the Full NAMO sequence."""

    target_region: str
    object_id: str
    actions: List[namo_rl.Action] = field(default_factory=list)
    resulting_state: Optional[namo_rl.RLState] = None


class FullNAMOPlanner(BasePlanner):
    """Full NAMO planner using region opening as a local sub-problem solver."""

    _INVARIANT_TARGET_FAILURES = {
        "already_accessible",
        "boundary_object_map_inconsistent",
        "missing_robot_region",
        "no_attempt_results",
        "no_blocking_objects",
        "target_not_immediate_neighbor",
    }

    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        algo_params = config.algorithm_params or {}
        raw_max_iterations = algo_params.get("full_namo_max_iterations")
        self.max_iterations = None if raw_max_iterations is None else int(raw_max_iterations)
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError(
                f"Invalid full_namo_max_iterations: {self.max_iterations}. Must be at least 1."
            )
        self.use_cpp_unified_wavefront = algo_params.get(
            "full_namo_use_cpp_unified_wavefront",
            algo_params.get("region_use_cpp_unified_wavefront", True),
        )
        self.region_snapshot_seed = int(algo_params.get("region_snapshot_seed", 42))
        region_goal_radius = algo_params.get("region_goal_radius_m", None)
        self.region_goal_radius_m = float(region_goal_radius) if region_goal_radius is not None else None
        self._config = config
        self._env = env
        self.region_opener: Optional[RegionOpeningPlanner] = None
        self.stats = FullNAMOStats()
        self._aggregated_rejections: Dict[str, int] = {}
        self._aggregated_primitives: int = 0
        self._iteration_trace: List[Dict[str, Any]] = []
        self.push_budget = algo_params.get("push_budget")
        super().__init__(env, config)

    def _setup_constraints(self):
        """Setup action constraints from environment."""
        pass

    def _initialize_algorithm(self):
        """Initialize algorithm-specific components."""
        self.region_opener = RegionOpeningPlanner(self._env, self._config)

    @property
    def algorithm_name(self) -> str:
        return "Full NAMO Planner"

    @property
    def algorithm_version(self) -> str:
        return "v1.0-hierarchical"

    def reset(self):
        self.stats = FullNAMOStats()
        self._aggregated_rejections = {}
        self._aggregated_primitives = 0
        self._iteration_trace = []
        if self.region_opener:
            self.region_opener.reset()

    def _debug(self, message: str):
        if getattr(self.config, "verbose", False):
            print(message)

    def _current_budget_stats(self) -> Dict[str, int]:
        if self.push_budget is None:
            return {}
        return {
            "simulation_budget_limit": int(self.push_budget.limit),
            "simulation_budget_used": int(self.push_budget.used),
            "simulation_budget_remaining": int(self.push_budget.remaining),
        }

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except Exception:
            return None

    def _extract_attempted_pushes_from_region_result(self, result: PlannerResult) -> int:
        """Estimate total simulated pushes attempted in one region-opening call.

        RegionOpeningPlanner returns per-object AttemptResult rows. We aggregate
        one count per object (max when duplicates exist) to avoid double counting
        when multiple recorded solutions exist for the same object.
        """
        if not result.algorithm_stats:
            return 0

        attempt_results = result.algorithm_stats.get("attempt_results", [])
        if not attempt_results:
            return 0

        pushes_by_key: Dict[str, int] = {}
        for attempt in attempt_results:
            obj_id = getattr(attempt, "chosen_object_id", None)
            neighbour = getattr(attempt, "neighbour_region_label", "")
            if obj_id is None:
                # Fallback rows can have no object id; keep them distinct.
                err = getattr(attempt, "error_message", "") or ""
                key = f"none:{neighbour}:{err}"
            else:
                key = f"obj:{obj_id}"

            # push_exec_count tracks actual env.step calls for this object search.
            pushes = self._as_int(getattr(attempt, "push_exec_count", None))
            if pushes is None:
                # Older rows may only populate pushes_total_for_neighbour.
                pushes = self._as_int(getattr(attempt, "pushes_total_for_neighbour", None))
            if pushes is None:
                continue

            if key in pushes_by_key:
                pushes_by_key[key] = max(pushes_by_key[key], pushes)
            else:
                pushes_by_key[key] = max(0, pushes)

        return int(sum(pushes_by_key.values()))

    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        start_time = time.time()
        self.stats = FullNAMOStats()
        self._aggregated_rejections = {}
        self._aggregated_primitives = 0
        self._iteration_trace = []

        self.env.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])

        self._debug(f"\n{'=' * 60}")
        self._debug(
            f"Full NAMO Planner - Target: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f}, {robot_goal[2]:.2f})"
        )
        self._debug(f"{'=' * 60}\n")
        actions: List[namo_rl.Action] = []
        region_openings: List[RegionOpeningResult] = []
        blocked_boundaries: Set[Tuple[str, str]] = set()

        if self.env.is_robot_goal_reachable():
            self._record_iteration_trace({"iteration": 0, "outcome": "goal_reachable_immediately"})
            return self._success_result(start_time, actions, region_openings)

        iteration = 1
        while True:
            if self.max_iterations is not None and iteration > self.max_iterations:
                return self._failure_result(
                    "Max iterations exceeded",
                    start_time,
                    actions,
                    failure_kind="max_iterations_exceeded",
                )
            self.stats.iterations = iteration
            self._debug(f"\n--- Iteration {iteration} ---")

            self.env.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])
            if self.env.is_robot_goal_reachable():
                self._record_iteration_trace(
                    {
                        "iteration": iteration,
                        "outcome": "goal_reachable",
                        "blocked_boundaries": self._serialize_blocked_boundaries(blocked_boundaries),
                    }
                )
                return self._success_result(start_time, actions, region_openings)

            snapshot = self._compute_region_snapshot()
            if snapshot is None:
                return self._failure_result(
                    "Failed to compute region snapshot",
                    start_time,
                    actions,
                    failure_kind="snapshot_compute_failed",
                )

            goal_region = snapshot.get("goal_label")
            goal_in_free_space = bool(snapshot.get("goal_in_free_space", False))
            if not goal_region or not goal_in_free_space:
                return self._failure_result(
                    "Goal position is in obstacle or out of bounds",
                    start_time,
                    actions,
                    failure_kind="goal_region_invalid",
                )

            robot_region = self._get_robot_region_label(snapshot)
            if robot_region is None:
                return self._failure_result(
                    "Can't find robot region",
                    start_time,
                    actions,
                    failure_kind="robot_region_invalid",
                )

            path = self._find_region_path_avoiding_edges(
                snapshot_data=snapshot,
                start_label=robot_region,
                goal_label=goal_region,
                failed_edges=blocked_boundaries,
            )

            base_context = self._build_iteration_context(
                iteration=iteration,
                snapshot_data=snapshot,
                robot_region=robot_region,
                goal_region=goal_region,
                path=path,
                blocked_boundaries=blocked_boundaries,
            )

            if path is None:
                self._record_iteration_trace({**base_context, "outcome": "region_path_exhausted"})
                return self._failure_result(
                    "No admissible region path to goal after blocked-boundary retries",
                    start_time,
                    actions,
                    failure_kind="region_path_exhausted",
                    context=base_context,
                )

            validation_error = self._validate_region_path(
                path=path,
                robot_region=robot_region,
                goal_region=goal_region,
                adjacency=snapshot["adjacency"],
                blocked_boundaries=blocked_boundaries,
            )
            if validation_error is not None:
                self._record_iteration_trace({**base_context, "outcome": validation_error})
                return self._invariant_failure(
                    validation_error,
                    start_time,
                    actions,
                    context=base_context,
                )

            if len(path) == 1:
                context = {
                    **base_context,
                    "possible_root_cause": "reachability_model_mismatch",
                }
                self._record_iteration_trace({**context, "outcome": "same_region_but_goal_unreachable"})
                return self._invariant_failure(
                    "same_region_but_goal_unreachable",
                    start_time,
                    actions,
                    context=context,
                )

            target = path[1]
            result = self.region_opener.search(robot_goal, target_neighbor=target)
            self.stats.total_attempted_pushes += self._extract_attempted_pushes_from_region_result(result)
            budget_used = self._as_int((result.algorithm_stats or {}).get("simulation_budget_used"))
            if budget_used is not None:
                self.stats.total_attempted_pushes = max(self.stats.total_attempted_pushes, budget_used)
            self._aggregate_region_result(result)
            target_summary = self._get_target_summary(result)
            context = self._build_iteration_context(
                iteration=iteration,
                snapshot_data=snapshot,
                robot_region=robot_region,
                goal_region=goal_region,
                path=path,
                blocked_boundaries=blocked_boundaries,
                target_region=target,
                target_summary=target_summary,
            )

            inner_failure_kind = str((result.algorithm_stats or {}).get("failure_kind") or "")
            if inner_failure_kind == "simulation_budget_exhausted":
                self._record_iteration_trace({**context, "outcome": "simulation_budget_exhausted"})
                return self._failure_result(
                    result.error_message or "Simulation budget exhausted",
                    start_time,
                    actions,
                    failure_kind="simulation_budget_exhausted",
                    context=context,
                )

            invariant_subkind = self._classify_target_invariant(result, target_summary)
            if invariant_subkind is not None:
                self._record_iteration_trace({**context, "outcome": invariant_subkind})
                return self._invariant_failure(
                    invariant_subkind,
                    start_time,
                    actions,
                    context=context,
                )

            if result.success:
                resulting_state = self._get_resulting_state_from_result(result)
                if resulting_state is None:
                    self._record_iteration_trace(
                        {**context, "outcome": "opener_contract_violation_missing_resulting_state"}
                    )
                    return self._invariant_failure(
                        "opener_contract_violation_missing_resulting_state",
                        start_time,
                        actions,
                        context=context,
                    )

                self.env.set_full_state(resulting_state)
                if self._compute_region_snapshot() is None:
                    self._record_iteration_trace({**context, "outcome": "post_open_snapshot_failed"})
                    return self._failure_result(
                        "Failed to recompute region snapshot after successful opening",
                        start_time,
                        actions,
                        failure_kind="post_open_snapshot_failed",
                        context=context,
                    )

                if result.action_sequence:
                    self.stats.total_pushes += len(result.action_sequence)
                    actions.extend(result.action_sequence)
                self.stats.successful_region_steps += 1
                self.stats.regions_opened.append(target)
                region_openings.append(
                    RegionOpeningResult(
                        target_region=target,
                        object_id=result.action_sequence[0].object_id if result.action_sequence else "unknown",
                        actions=list(result.action_sequence) if result.action_sequence else [],
                        resulting_state=resulting_state,
                    )
                )
                blocked_boundaries = set()
                self._record_iteration_trace({**context, "outcome": "opened_target"})
                self._debug(f"Opened {target}")
                iteration += 1
                continue

            if bool(target_summary.get("boundary_exhausted", False)):
                blocked_boundaries.add(self._boundary_key(robot_region, target))
                self.stats.boundary_exhaustions += 1
                self._record_iteration_trace({**context, "outcome": "boundary_exhausted"})
                self._debug(f"Boundary exhausted for {target}; retrying with blocked boundary")
                iteration += 1
                continue

            self._record_iteration_trace({**context, "outcome": "opener_failure_not_boundary_exhausted"})
            return self._failure_result(
                "Region opener failed without exhausting the targeted boundary",
                start_time,
                actions,
                failure_kind="opener_failure_not_boundary_exhausted",
                context=context,
            )

    def _success_result(
        self,
        start_time: float,
        actions: List[namo_rl.Action],
        region_openings: List[RegionOpeningResult],
    ) -> PlannerResult:
        algorithm_stats = {
            "full_namo_stats": self.stats,
            "iterations": self.stats.iterations,
            "total_pushes": self.stats.total_pushes,
            "total_attempted_pushes": self.stats.total_attempted_pushes,
            "successful_region_steps": self.stats.successful_region_steps,
            "boundary_exhaustions": self.stats.boundary_exhaustions,
            "regions_opened": list(self.stats.regions_opened),
            "region_opening_sequence": region_openings,
            "rejection_breakdown": dict(self._aggregated_rejections),
            "total_primitives_attempted": self._aggregated_primitives,
            "iteration_trace": list(self._iteration_trace),
        }
        algorithm_stats.update(self._current_budget_stats())
        return PlannerResult(
            success=True,
            solution_found=True,
            action_sequence=actions,
            solution_depth=len(actions),
            search_time_ms=(time.time() - start_time) * 1000,
            algorithm_stats=algorithm_stats,
        )
    def _failure_result(
        self,
        error_message: str,
        start_time: float,
        actions: List[namo_rl.Action],
        *,
        failure_kind: str,
        failure_subkind: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PlannerResult:
        total_time = (time.time() - start_time) * 1000
        self._debug(f"FAILURE: {error_message}")

        algorithm_stats: Dict[str, Any] = {
            "full_namo_stats": self.stats,
            "iterations": self.stats.iterations,
            "total_pushes": self.stats.total_pushes,
            "total_attempted_pushes": self.stats.total_attempted_pushes,
            "successful_region_steps": self.stats.successful_region_steps,
            "boundary_exhaustions": self.stats.boundary_exhaustions,
            "regions_opened": list(self.stats.regions_opened),
            "rejection_breakdown": dict(self._aggregated_rejections),
            "total_primitives_attempted": self._aggregated_primitives,
            "iteration_trace": list(self._iteration_trace),
            "failure_kind": failure_kind,
        }
        if failure_subkind is not None:
            algorithm_stats["failure_subkind"] = failure_subkind
        if context is not None:
            key = "invariant_context" if failure_kind == "planner_invariant_violation" else "failure_context"
            algorithm_stats[key] = context
        algorithm_stats.update(self._current_budget_stats())

        return PlannerResult(
            success=False,
            solution_found=False,
            action_sequence=actions if actions else None,
            solution_depth=len(actions) if actions else None,
            search_time_ms=total_time,
            error_message=error_message,
            algorithm_stats=algorithm_stats,
        )

    def _invariant_failure(
        self,
        failure_subkind: str,
        start_time: float,
        actions: List[namo_rl.Action],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> PlannerResult:
        return self._failure_result(
            f"Planner invariant violation: {failure_subkind}",
            start_time,
            actions,
            failure_kind="planner_invariant_violation",
            failure_subkind=failure_subkind,
            context=context,
        )

    def _compute_region_snapshot(self) -> Optional[Dict[str, Any]]:
        try:
            from namo.planners import get_region_snapshot as _get_region_snapshot

            snapshot = _get_region_snapshot(
                self.env,
                goals_per_region=self.config.goals_per_region,
                goal_radius=self.region_goal_radius_m,
                local_info_only=False,
                seed=self.region_snapshot_seed,
                use_cpp_unified=self.use_cpp_unified_wavefront,
                use_xml_goal=True,
            )
            return snapshot
        except Exception as e:
            self._debug(f"Error computing region snapshot: {e}")
            return None
    def _get_robot_region_label(self, snapshot_data: Dict[str, Any]) -> Optional[str]:
        robot_label = snapshot_data.get("robot_label")
        if robot_label:
            return robot_label
        region_labels = snapshot_data["region_labels"]
        label = find_robot_label(region_labels)
        return label if isinstance(label, str) else None

    def _get_resulting_state_from_result(self, result: PlannerResult) -> Optional[namo_rl.RLState]:
        attempt_results = (result.algorithm_stats or {}).get("attempt_results", [])
        for attempt in attempt_results:
            if getattr(attempt, "success", False) and getattr(attempt, "resulting_state", None) is not None:
                return attempt.resulting_state
        return None

    def _aggregate_region_result(self, result: PlannerResult):
        inner_stats = result.algorithm_stats or {}
        inner_breakdown = inner_stats.get("rejection_breakdown") or {}
        for key, value in inner_breakdown.items():
            self._aggregated_rejections[key] = self._aggregated_rejections.get(key, 0) + int(value)
        self._aggregated_primitives += int(inner_stats.get("total_primitives_attempted", 0))

    def _get_target_summary(self, result: PlannerResult) -> Optional[Dict[str, Any]]:
        algorithm_stats = result.algorithm_stats or {}
        summary = algorithm_stats.get("target_summary")
        return summary if isinstance(summary, dict) else None

    def _classify_target_invariant(
        self,
        result: PlannerResult,
        target_summary: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if target_summary is None:
            return "opener_contract_violation_missing_target_summary"

        failure_reason = target_summary.get("failure_reason")
        if failure_reason in self._INVARIANT_TARGET_FAILURES:
            return str(failure_reason)

        if result.success and failure_reason not in {"success", "already_accessible"}:
            return "opener_contract_violation_success_reason_mismatch"
        if (not result.success) and failure_reason == "success":
            return "opener_contract_violation_failure_reason_mismatch"
        return None

    def _build_iteration_context(
        self,
        *,
        iteration: int,
        snapshot_data: Dict[str, Any],
        robot_region: str,
        goal_region: str,
        path: Optional[List[str]],
        blocked_boundaries: Set[Tuple[str, str]],
        target_region: Optional[str] = None,
        target_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        robot_neighbors = sorted(snapshot_data["adjacency"].get(robot_region, set()))
        return {
            "iteration": iteration,
            "robot_region": robot_region,
            "goal_region": goal_region,
            "chosen_path": list(path) if path is not None else None,
            "chosen_target_region": target_region,
            "robot_neighbors": robot_neighbors,
            "blocked_boundaries": self._serialize_blocked_boundaries(blocked_boundaries),
            "target_summary": target_summary,
            "rejection_breakdown": dict(self._aggregated_rejections),
            "total_primitives_attempted": self._aggregated_primitives,
        }

    def _record_iteration_trace(self, trace: Dict[str, Any]):
        self._iteration_trace.append(dict(trace))

    def _serialize_blocked_boundaries(
        self,
        blocked_boundaries: Set[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        return sorted(blocked_boundaries)

    def _boundary_key(self, a: str, b: str) -> Tuple[str, str]:
        if not isinstance(a, str) or not isinstance(b, str):
            raise TypeError(f"Boundary endpoints must be strings, got {type(a).__name__}, {type(b).__name__}")
        if a == b:
            raise ValueError("Boundary endpoints must be distinct")
        return tuple(sorted((a, b)))

    def _validate_region_path(
        self,
        *,
        path: List[str],
        robot_region: str,
        goal_region: str,
        adjacency: Dict[str, Set[str]],
        blocked_boundaries: Set[Tuple[str, str]],
    ) -> Optional[str]:
        if not path:
            return "empty_region_path"
        if path[0] != robot_region:
            return "path_does_not_start_at_robot_region"
        if path[-1] != goal_region:
            return "path_does_not_end_at_goal_region"
        if len(path) != len(set(path)):
            return "path_contains_repeated_region"

        for label in path:
            if not isinstance(label, str):
                return "path_contains_non_string_region_label"

        for current, nxt in zip(path, path[1:]):
            if nxt not in adjacency.get(current, set()):
                return "path_contains_non_adjacent_hop"
            if self._boundary_key(current, nxt) in blocked_boundaries:
                return "path_uses_blocked_boundary"
        return None

    def _find_region_path_avoiding_edges(
        self,
        snapshot_data: Dict[str, Any],
        start_label: str,
        goal_label: str,
        failed_edges: Optional[Set[Tuple[str, str]]] = None,
    ) -> Optional[List[str]]:
        adjacency = snapshot_data["adjacency"]
        blocked = failed_edges or set()

        if start_label not in adjacency:
            self._debug(f"Start label {start_label} not in adjacency graph")
            return None
        if start_label == goal_label:
            return [start_label]

        queue = deque([(start_label, [start_label])])
        visited: Set[str] = {start_label}

        while queue:
            current, path = queue.popleft()
            for neighbor in sorted(adjacency.get(current, set())):
                edge = self._boundary_key(current, neighbor)
                if edge in blocked:
                    continue
                if neighbor == goal_label:
                    return path + [neighbor]
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

        return None


from namo.core import PlannerFactory

PlannerFactory.register_planner("full_namo", FullNAMOPlanner)
