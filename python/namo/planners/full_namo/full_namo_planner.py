"""Full NAMO Planner - Hierarchical planning using region opening as sub-problem.

This planner solves the full NAMO problem (reaching a specific robot goal) by:
1. Computing the current region connectivity graph
2. Finding a shortest path through regions from robot to goal
3. Opening only the first robot-adjacent boundary on that path
4. Recomputing after each successful opening until the robot goal is reachable
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set, Tuple

import namo_rl

from namo.core import BasePlanner, PlannerConfig, PlannerResult
from namo.planners.connectivity_snapshot import find_robot_label
from namo.planners.opening.best_first_region_opening import (
    BestFirstRegionOpeningPlanner,
)
from namo.planners.opening.region_opening import RegionOpeningPlanner
from namo.planners.utils import PushAttemptBudget


FULL_NAMO_EXEC_MODES = ("search", "greedy_dfs", "greedy_policy")
GREEDY_COMMIT_EXEC_MODES = frozenset({"greedy_dfs", "greedy_policy"})

# The opener's word for "I ran out of simulations", written by
# best_first_region_opening.py at both the result level (line 257) and the
# per-attempt summary (line 521). Named here so the outer loop compares against
# one spelling instead of a literal in three places.
_BUDGET_STOP_REASON = "simulation_budget_exhausted"
DEFAULT_FULL_NAMO_EXEC_MODE = "search"


def boundary_key(a: str, b: str) -> Tuple[str, str]:
    """Order-independent identity for the boundary between two regions."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError(
            f"Boundary endpoints must be strings, got {type(a).__name__}, {type(b).__name__}"
        )
    if a == b:
        raise ValueError("Boundary endpoints must be distinct")
    return tuple(sorted((a, b)))


def find_region_path(
    adjacency: Dict[str, Set[str]],
    start_label: str,
    goal_label: str,
    failed_edges: Optional[Set[Tuple[str, str]]] = None,
) -> Optional[List[str]]:
    """Shortest region path avoiding blocked boundaries, or None.

    Module-level because the rule for choosing which boundary to open next --
    path[1] of this BFS -- must be identical for the planner and for any
    external caller that pins a boundary. Neighbours are iterated in sorted
    order so the choice is deterministic.
    """
    blocked = failed_edges or set()
    if start_label not in adjacency:
        return None
    if start_label == goal_label:
        return [start_label]

    queue = deque([(start_label, [start_label])])
    visited: Set[str] = {start_label}

    while queue:
        current, path = queue.popleft()
        for neighbor in sorted(adjacency.get(current, set())):
            if boundary_key(current, neighbor) in blocked:
                continue
            if neighbor == goal_label:
                return path + [neighbor]
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))

    return None


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
    # Boundaries abandoned because they consumed the simulation budget without
    # opening. Distinct from boundary_exhaustions, which counts boundaries whose
    # candidate pool ran dry; a budget stop may still be openable given more.
    boundary_budget_stops: int = 0
    regions_opened: List[str] = field(default_factory=list)
    greedy_committed_pushes: int = 0
    greedy_rejected_simulations: int = 0


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
        self.local_search = str(algo_params.get("full_namo_local_search", "region_bfs"))
        if self.local_search not in {"region_bfs", "best_first"}:
            raise ValueError("full_namo_local_search must be 'region_bfs' or 'best_first'")
        self.exec_mode = str(
            algo_params.get("full_namo_exec_mode", DEFAULT_FULL_NAMO_EXEC_MODE)
        )
        if self.exec_mode not in FULL_NAMO_EXEC_MODES:
            raise ValueError(
                f"Unknown full_namo_exec_mode {self.exec_mode!r}. "
                f"Valid: {list(FULL_NAMO_EXEC_MODES)}"
            )
        if (
            self.exec_mode in GREEDY_COMMIT_EXEC_MODES
            and self.local_search != "best_first"
        ):
            raise ValueError(
                f"full_namo_exec_mode={self.exec_mode!r} requires "
                "full_namo_local_search='best_first'"
            )
        self.region_opener: Optional[Any] = None
        self.stats = FullNAMOStats()
        self._aggregated_rejections: Dict[str, int] = {}
        self._aggregated_primitives: int = 0
        self._iteration_trace: List[Dict[str, Any]] = []
        self.push_budget = algo_params.get("push_budget")
        self.budget_scope = str(algo_params.get("full_namo_budget_scope", "full_problem"))
        if self.budget_scope not in {"full_problem", "keyhole"}:
            raise ValueError(
                f"Invalid full_namo_budget_scope: {self.budget_scope}. "
                "Must be 'full_problem' or 'keyhole'."
            )
        raw_keyhole_limit = algo_params.get("full_namo_keyhole_simulation_budget")
        if raw_keyhole_limit is None and self.push_budget is not None:
            raw_keyhole_limit = self.push_budget.limit
        self.keyhole_budget_limit = None if raw_keyhole_limit is None else int(raw_keyhole_limit)
        if self.budget_scope == "keyhole" and self.keyhole_budget_limit is None:
            raise ValueError("Keyhole budget scope requires full_namo_keyhole_simulation_budget")
        self._keyhole_budget_usage: List[Dict[str, Any]] = []
        self.audit_next_keyhole_reachability = bool(
            algo_params.get("full_namo_audit_next_keyhole_reachability", False)
        )
        self.preserve_next_keyhole_access = bool(
            algo_params.get("full_namo_preserve_next_keyhole_access", False)
        )
        if self.preserve_next_keyhole_access and self.local_search != "best_first":
            raise ValueError(
                "full_namo_preserve_next_keyhole_access requires full_namo_local_search='best_first'"
            )
        super().__init__(env, config)

    def _setup_constraints(self):
        """Setup action constraints from environment."""
        pass

    def _initialize_algorithm(self):
        """Initialize algorithm-specific components."""
        self.region_opener = self._make_region_opener(self._config)

    def _make_region_opener(self, config: PlannerConfig):
        if self.local_search == "best_first":
            return BestFirstRegionOpeningPlanner(self._env, config)
        return RegionOpeningPlanner(self._env, config)

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
        self._keyhole_budget_usage = []
        if self.region_opener:
            self.region_opener.reset()

    def _debug(self, message: str):
        if getattr(self.config, "verbose", False):
            print(message)

    def _current_budget_stats(self) -> Dict[str, Any]:
        if self.budget_scope == "keyhole":
            used_by_keyhole = [int(row["used"]) for row in self._keyhole_budget_usage]
            return {
                "simulation_budget_scope": "keyhole",
                "simulation_budget_limit_per_keyhole": int(self.keyhole_budget_limit),
                "simulation_budget_used_total": int(sum(used_by_keyhole)),
                "simulation_budget_used_by_keyhole": used_by_keyhole,
                "simulation_budget_keyholes_attempted": len(used_by_keyhole),
            }
        if self.push_budget is None:
            # No counters to report, but the scope is still known and still
            # decides what a result means. This is the shape robot_control
            # produces, since it forwards a per-keyhole limit and never a
            # budget object, so returning {} left the one configuration that
            # runs on hardware as the only one whose budget rule went
            # unrecorded. The constant is named CANONICAL_KEYHOLE_SIMULATION_
            # BUDGET and this default is not per keyhole, so a reader with no
            # scope in the record will assume the wrong one.
            return {"simulation_budget_scope": self.budget_scope}
        return {
            "simulation_budget_scope": "full_problem",
            "simulation_budget_limit": int(self.push_budget.limit),
            "simulation_budget_used": int(self.push_budget.used),
            "simulation_budget_remaining": int(self.push_budget.remaining),
        }

    def _prepare_region_opener_for_keyhole(self):
        if self.budget_scope != "keyhole":
            return self.region_opener

        local_budget = PushAttemptBudget(limit=self.keyhole_budget_limit)
        local_params = dict(self._config.algorithm_params or {})
        local_params["push_budget"] = local_budget
        local_config = replace(self._config, algorithm_params=local_params)
        self.region_opener = self._make_region_opener(local_config)
        return self.region_opener

    @staticmethod
    def _budget_stopped(result: PlannerResult) -> bool:
        """True when the opener stopped because it ran out of simulations.

        The opener reports this two ways. `failure_kind` is set from
        `end == "budget"` (best_first_region_opening.py:257) and the
        per-attempt `failure_reason` carries the same string, so accept
        either rather than depending on which layer answered.
        """
        stats = result.algorithm_stats or {}
        if str(stats.get("failure_kind", "")) == _BUDGET_STOP_REASON:
            return True
        summary = stats.get("target_summary") or {}
        return str(summary.get("failure_reason", "")) == _BUDGET_STOP_REASON

    def _simulation_budget_remains(self, result: PlannerResult) -> bool:
        """Is there budget left to attempt a different boundary with?

        Keyhole scope rebuilds the opener with a fresh allowance per boundary
        (`_prepare_region_opener_for_keyhole`), so the answer is always yes.
        Full-problem scope shares one mutable budget across every boundary, so
        a boundary that consumed it leaves nothing for a reroute.

        The opener reports `simulation_budget_remaining` in its own stats and
        that is the authority, since it is the thing spending the budget. The
        live counter is only a fallback for an opener that reports no figure.
        Assuming budget remains when nothing says so would reroute into a
        boundary there is nothing left to attempt.
        """
        if self.budget_scope == "keyhole":
            return True
        reported = (result.algorithm_stats or {}).get("simulation_budget_remaining")
        if reported is not None:
            return int(reported) > 0
        budget = getattr(self.region_opener, "push_budget", None)
        if budget is None:
            return False
        return int(budget.remaining) > 0

    def _record_keyhole_budget(self, iteration: int, target: str, result: PlannerResult):
        if self.budget_scope != "keyhole":
            return
        used = self._as_int((result.algorithm_stats or {}).get("simulation_budget_used"))
        if used is None:
            used = int(self.region_opener.push_budget.used)
        self._keyhole_budget_usage.append(
            {
                "iteration": int(iteration),
                "target_region": target,
                "limit": int(self.keyhole_budget_limit),
                "used": int(used),
            }
        )
        self.stats.total_attempted_pushes = int(
            sum(row["used"] for row in self._keyhole_budget_usage)
        )

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
        self._keyhole_budget_usage = []
        zero_push_opened: Set[Tuple[str, str]] = set()

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
            next_keyhole_profile = None
            if (
                self.audit_next_keyhole_reachability or self.preserve_next_keyhole_access
            ) and len(path) >= 3:
                next_keyhole_profile = self._profile_next_keyhole_before_open(
                    snapshot_data=snapshot,
                    path=path,
                    robot_goal=robot_goal,
                )
            if (
                self.preserve_next_keyhole_access
                and next_keyhole_profile is not None
                and next_keyhole_profile.get("status") != "ok"
            ):
                context = {
                    **base_context,
                    "chosen_target_region": target,
                    "next_keyhole_profile": next_keyhole_profile,
                }
                self._record_iteration_trace(
                    {**context, "outcome": "next_keyhole_profile_unavailable"}
                )
                return self._invariant_failure(
                    "next_keyhole_profile_unavailable",
                    start_time,
                    actions,
                    context=context,
                )
            opener = self._prepare_region_opener_for_keyhole()
            opener_kwargs: Dict[str, Any] = {}
            if self.local_search == "best_first" and len(path) == 2:
                opener_kwargs["opening_predicate"] = (
                    lambda candidate_env: candidate_env.is_robot_goal_reachable()
                )
            if self.preserve_next_keyhole_access and next_keyhole_profile is not None:
                opener_kwargs["candidate_acceptor"] = (
                    lambda candidate_env: self._check_next_keyhole_access_candidate(
                        env=candidate_env,
                        profile=next_keyhole_profile,
                    )
                )
            if self.exec_mode in GREEDY_COMMIT_EXEC_MODES:
                result = opener.greedy_commit(
                    robot_goal,
                    target_neighbor=target,
                    # The policy mode is simulator-free by contract: the ranked
                    # arg-max goes to the robot untried and the camera judges
                    # it, because a push the sim calls inert may move the real
                    # block. greedy_dfs keeps the simulator; its rollout needs
                    # the resulting state to take the next step from.
                    simulate=self.exec_mode != "greedy_policy",
                    **opener_kwargs,
                )
            else:
                result = opener.search(
                    robot_goal, target_neighbor=target, **opener_kwargs
                )
            self._record_keyhole_budget(iteration, target, result)
            if self.budget_scope == "full_problem":
                self.stats.total_attempted_pushes += self._extract_attempted_pushes_from_region_result(result)
            budget_used = self._as_int((result.algorithm_stats or {}).get("simulation_budget_used"))
            if self.budget_scope == "full_problem" and budget_used is not None:
                self.stats.total_attempted_pushes = max(self.stats.total_attempted_pushes, budget_used)
            self._aggregate_region_result(result)
            target_summary = self._get_target_summary(result)
            greedy_commit_stats = (result.algorithm_stats or {}).get("greedy_commit") or {}
            if self.exec_mode in GREEDY_COMMIT_EXEC_MODES:
                self.stats.greedy_rejected_simulations += len(
                    greedy_commit_stats.get("rejections") or []
                )
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

            # A boundary that ate its simulation budget without opening is a
            # reason to try another route, not to abandon the problem. This
            # used to return unconditionally, which meant best_first could
            # never reroute: it spends its budget where region_bfs runs its
            # candidate pool dry, and only the dry-pool reasons reach the
            # `boundary_exhausted` reroute below (best_first_region_opening.py:221).
            # Measured on real_exp/twohop_00013, a captured scene whose
            # robot-to-goal doorway needs both movables: region_bfs gave up
            # after 132 attempts, rerouted through a cheaper doorway, and
            # solved it in two pushes, while best_first burned all 900 sims on
            # that one doorway and returned failure without ever looking at the
            # two beside it.
            #
            # Rerouting only helps if there is budget left to reroute WITH.
            # Keyhole scope rebuilds the opener with a fresh allowance per
            # boundary, so there always is. Full-problem scope shares one
            # budget, so a boundary that consumed it leaves nothing and failure
            # is still the honest answer.
            if self._budget_stopped(result):
                if self._simulation_budget_remains(result):
                    blocked_boundaries.add(self._boundary_key(robot_region, target))
                    self.stats.boundary_budget_stops += 1
                    self._record_iteration_trace(
                        {**context, "outcome": "boundary_budget_exhausted"}
                    )
                    self._debug(
                        f"Budget spent on {target} without opening it; rerouting "
                        f"with the boundary blocked"
                    )
                    iteration += 1
                    continue
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

            if self.exec_mode in GREEDY_COMMIT_EXEC_MODES and result.action_sequence:
                if len(result.action_sequence) != 1:
                    return self._invariant_failure(
                        "greedy_commit_returned_multiple_actions",
                        start_time,
                        actions,
                        context=context,
                    )
                resulting_state = self._get_resulting_state_from_result(result)
                if resulting_state is None:
                    return self._invariant_failure(
                        "greedy_commit_missing_resulting_state",
                        start_time,
                        actions,
                        context=context,
                    )
                action = result.action_sequence[0]
                self.env.set_full_state(resulting_state)
                actions.append(action)
                self.stats.total_pushes += 1
                self.stats.greedy_committed_pushes += 1
                blocked_boundaries = set()
                opened = bool(result.success)
                if opened:
                    self.stats.successful_region_steps += 1
                    self.stats.regions_opened.append(target)
                    region_openings.append(
                        RegionOpeningResult(
                            target_region=target,
                            object_id=action.object_id,
                            actions=[action],
                            resulting_state=resulting_state,
                        )
                    )
                if self.exec_mode == "greedy_policy":
                    step_outcome = "policy_step_ready"
                elif opened:
                    step_outcome = "greedy_step_opened"
                else:
                    step_outcome = "greedy_step_committed"
                self._record_iteration_trace({
                    **context,
                    "outcome": step_outcome,
                    "greedy_action": {
                        "object_id": str(action.object_id),
                        "edge_idx": int(action.edge_idx),
                        "depth": int(action.depth),
                    },
                    "greedy_commit": dict(greedy_commit_stats),
                })
                if self.exec_mode == "greedy_policy":
                    return self._success_result(
                        start_time,
                        actions,
                        region_openings,
                        extra_stats={"policy_outcome": "policy_step_ready"},
                    )
                iteration += 1
                continue

            if result.success:
                # A zero-push opening means the opener already counted the target region
                # reachable while the region graph still called the boundary blocked. The
                # scene is fine, but nothing moved, so the next iteration would rebuild an
                # identical snapshot. Allow it once, then treat a repeat as a genuine
                # graph/opener mismatch and fall back to the blocked-boundary reroute.
                zero_push = not result.action_sequence
                if zero_push:
                    zero_push_key = self._boundary_key(robot_region, target)
                    if zero_push_key in zero_push_opened:
                        blocked_boundaries.add(zero_push_key)
                        self.stats.boundary_exhaustions += 1
                        self._record_iteration_trace({**context, "outcome": "already_accessible_repeat"})
                        iteration += 1
                        continue
                    zero_push_opened.add(zero_push_key)

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
                post_open_snapshot = self._compute_region_snapshot()
                if post_open_snapshot is None:
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
                if not zero_push:
                    # Physical state changed, so previously exhausted boundaries may now be
                    # openable. A zero-push opening changes nothing, so keep the blacklist.
                    blocked_boundaries = set()
                independence_audit = None
                if next_keyhole_profile is not None:
                    independence_audit = self._audit_next_keyhole_after_open(
                        profile=next_keyhole_profile,
                        post_snapshot=post_open_snapshot,
                    )
                trace = {**context, "outcome": "opened_target"}
                if independence_audit is not None:
                    trace["next_keyhole_reachability"] = independence_audit
                self._record_iteration_trace(trace)
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
        *,
        extra_stats: Optional[Dict[str, Any]] = None,
    ) -> PlannerResult:
        """Return an executable planner result with JSON-safe summary stats.

        For ``greedy_policy``, executable success means one physical policy
        step is ready; final goal success remains camera-validated by the
        robot runtime after subsequent observation and navigation.
        """
        algorithm_stats = {
            "full_namo_stats": self.stats,
            "iterations": self.stats.iterations,
            "total_pushes": self.stats.total_pushes,
            "total_attempted_pushes": self.stats.total_attempted_pushes,
            "successful_region_steps": self.stats.successful_region_steps,
            "boundary_exhaustions": self.stats.boundary_exhaustions,
            "boundary_budget_stops": self.stats.boundary_budget_stops,
            "regions_opened": list(self.stats.regions_opened),
            "exec_mode": self.exec_mode,
            "greedy_committed_pushes": self.stats.greedy_committed_pushes,
            "greedy_rejected_simulations": self.stats.greedy_rejected_simulations,
            "region_opening_sequence": region_openings,
            "rejection_breakdown": dict(self._aggregated_rejections),
            "total_primitives_attempted": self._aggregated_primitives,
            "iteration_trace": list(self._iteration_trace),
        }
        algorithm_stats.update(self._current_budget_stats())
        if extra_stats:
            algorithm_stats.update(extra_stats)
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
            "boundary_budget_stops": self.stats.boundary_budget_stops,
            "regions_opened": list(self.stats.regions_opened),
            "exec_mode": self.exec_mode,
            "greedy_committed_pushes": self.stats.greedy_committed_pushes,
            "greedy_rejected_simulations": self.stats.greedy_rejected_simulations,
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

    @staticmethod
    def _boundary_objects_for_snapshot(
        snapshot_data: Dict[str, Any], source: str, target: str
    ) -> Tuple[List[str], Optional[str]]:
        edge_objects = snapshot_data["edge_objects"]
        forward = edge_objects.get(source, {}).get(target)
        reverse = edge_objects.get(target, {}).get(source)
        if forward is not None and reverse is not None and set(forward) != set(reverse):
            return [], "boundary_object_map_inconsistent"
        return sorted(set(forward if forward is not None else reverse or [])), None

    @staticmethod
    def _pose_for_object(observation: Dict[str, Any], object_id: str) -> Optional[List[float]]:
        pose = observation.get(f"{object_id}_pose")
        if pose is None:
            return None
        return [float(pose[0]), float(pose[1]), float(pose[2])]

    def _profile_next_keyhole_before_open(
        self,
        *,
        snapshot_data: Dict[str, Any],
        path: List[str],
        robot_goal: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        """Measure keyhole k+1 from inside its source region without touching the live search env."""
        source_region, target_region = path[1], path[2]
        objects, boundary_error = self._boundary_objects_for_snapshot(
            snapshot_data, source_region, target_region
        )
        profile: Dict[str, Any] = {
            "status": "ok",
            "source_region": source_region,
            "target_region": target_region,
            "path_hops_before": len(path) - 1,
            "objects_before": objects,
        }
        if boundary_error is not None:
            profile.update(status=boundary_error, objects={})
            return profile
        if not objects:
            profile.update(status="no_next_boundary_objects", objects={})
            return profile

        bundle = snapshot_data.get("region_goals", {}).get(source_region)
        goals = list(bundle.goals) if bundle is not None else []
        if not goals:
            profile.update(status="no_middle_region_seed", objects={})
            return profile

        shadow = namo_rl.RLEnvironment(
            self.env.get_xml_path(), self.env.get_config_path(), False
        )
        shadow.set_full_state(self.env.get_full_state())
        shadow.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])
        observation = shadow.get_observation()
        seed = goals[0]
        shadow.set_robot_pose(float(seed.x), float(seed.y), float(seed.theta))
        profile["middle_region_seed"] = [float(seed.x), float(seed.y), float(seed.theta)]
        profile["objects"] = {
            object_id: {
                "pose_before": self._pose_for_object(observation, object_id),
                "reachable_edges_before": sorted(
                    int(edge) for edge in shadow.get_reachable_edges(object_id)
                ),
            }
            for object_id in objects
        }
        return profile

    def _check_next_keyhole_access_candidate(
        self,
        *,
        env: namo_rl.RLEnvironment,
        profile: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Accept a local opening only if every original next-keyhole action remains available."""
        if env.is_robot_goal_reachable():
            return True, {
                "accepted": True,
                "goal_reachable": True,
                "failure_reasons": [],
                "objects": {},
            }

        observation = env.get_observation()
        object_rows: Dict[str, Any] = {}
        failure_reasons: Set[str] = set()
        for object_id, before in profile["objects"].items():
            before_edges = set(int(edge) for edge in before["reachable_edges_before"])
            after_edges = set(int(edge) for edge in env.get_reachable_edges(object_id))
            pose_before = before.get("pose_before")
            pose_after = self._pose_for_object(observation, object_id)
            dxy_mm = None
            dtheta_deg = None
            pose_unchanged = False
            if pose_before is not None and pose_after is not None:
                dxy_mm = 1000.0 * math.hypot(
                    pose_after[0] - pose_before[0], pose_after[1] - pose_before[1]
                )
                dtheta = abs(
                    (pose_after[2] - pose_before[2] + math.pi) % (2.0 * math.pi) - math.pi
                )
                dtheta_deg = math.degrees(dtheta)
                pose_unchanged = dxy_mm <= 0.1 and dtheta_deg <= 0.1
            lost_edges = before_edges - after_edges
            if lost_edges:
                failure_reasons.add("next_contact_edges_lost")
            if not pose_unchanged:
                failure_reasons.add("next_blocker_moved")
            object_rows[object_id] = {
                "reachable_edges_before": sorted(before_edges),
                "reachable_edges_after": sorted(after_edges),
                "lost_edges": sorted(lost_edges),
                "gained_edges": sorted(after_edges - before_edges),
                "all_original_edges_reachable": not lost_edges,
                "pose_dxy_mm": round(dxy_mm, 4) if dxy_mm is not None else None,
                "pose_dtheta_deg": round(dtheta_deg, 4) if dtheta_deg is not None else None,
                "pose_unchanged": pose_unchanged,
            }

        accepted = not failure_reasons
        return accepted, {
            "accepted": accepted,
            "goal_reachable": False,
            "failure_reasons": sorted(failure_reasons),
            "objects": object_rows,
        }

    def _audit_next_keyhole_after_open(
        self,
        *,
        profile: Dict[str, Any],
        post_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare the exact next-keyhole contact set before and after a committed opening."""
        audit: Dict[str, Any] = {
            "status": profile.get("status"),
            "path_hops_before": profile.get("path_hops_before"),
            "objects_before": list(profile.get("objects_before") or []),
            "preserved": False,
        }
        if profile.get("status") != "ok":
            return audit

        robot_region = self._get_robot_region_label(post_snapshot)
        goal_region = post_snapshot.get("goal_label")
        post_path = None
        if robot_region and goal_region:
            post_path = self._find_region_path_avoiding_edges(
                snapshot_data=post_snapshot,
                start_label=robot_region,
                goal_label=goal_region,
                failed_edges=set(),
            )
        audit["path_after"] = list(post_path) if post_path is not None else None
        audit["path_hops_after"] = len(post_path) - 1 if post_path else None
        audit["hop_reduced_by_one"] = bool(
            post_path
            and len(post_path) - 1 == int(profile["path_hops_before"]) - 1
        )

        objects_after: List[str] = []
        boundary_error = None
        if post_path and len(post_path) >= 2:
            objects_after, boundary_error = self._boundary_objects_for_snapshot(
                post_snapshot, post_path[0], post_path[1]
            )
        audit["objects_after"] = objects_after
        audit["object_identity_unchanged"] = (
            boundary_error is None and objects_after == audit["objects_before"]
        )
        if boundary_error is not None:
            audit["status"] = boundary_error

        observation = self.env.get_observation()
        object_rows: Dict[str, Any] = {}
        all_edges_same = True
        all_poses_same = True
        for object_id, before in profile["objects"].items():
            before_edges = set(int(edge) for edge in before["reachable_edges_before"])
            after_edges = set(int(edge) for edge in self.env.get_reachable_edges(object_id))
            pose_before = before.get("pose_before")
            pose_after = self._pose_for_object(observation, object_id)
            dxy_mm = None
            dtheta_deg = None
            pose_unchanged = False
            if pose_before is not None and pose_after is not None:
                dxy_mm = 1000.0 * math.hypot(
                    pose_after[0] - pose_before[0], pose_after[1] - pose_before[1]
                )
                dtheta = abs(
                    (pose_after[2] - pose_before[2] + math.pi) % (2.0 * math.pi) - math.pi
                )
                dtheta_deg = math.degrees(dtheta)
                pose_unchanged = dxy_mm <= 0.1 and dtheta_deg <= 0.1
            edges_same = before_edges == after_edges
            all_edges_same = all_edges_same and edges_same
            all_poses_same = all_poses_same and pose_unchanged
            object_rows[object_id] = {
                "reachable_edges_before": sorted(before_edges),
                "reachable_edges_after": sorted(after_edges),
                "lost_edges": sorted(before_edges - after_edges),
                "gained_edges": sorted(after_edges - before_edges),
                "exact_edge_set_unchanged": edges_same,
                "pose_dxy_mm": round(dxy_mm, 4) if dxy_mm is not None else None,
                "pose_dtheta_deg": round(dtheta_deg, 4) if dtheta_deg is not None else None,
                "pose_unchanged": pose_unchanged,
            }
        audit["objects"] = object_rows
        audit["exact_edge_sets_unchanged"] = all_edges_same
        audit["object_poses_unchanged"] = all_poses_same
        audit["preserved"] = bool(
            audit["hop_reduced_by_one"]
            and audit["object_identity_unchanged"]
            and all_edges_same
            and all_poses_same
        )
        return audit
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
            resulting_state = getattr(attempt, "resulting_state", None)
            committed = getattr(attempt, "failure_reason", None) == "greedy_step_committed"
            if resulting_state is not None and (
                getattr(attempt, "success", False) or committed
            ):
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
        return boundary_key(a, b)

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
        if start_label not in adjacency:
            self._debug(f"Start label {start_label} not in adjacency graph")
            return None
        return find_region_path(adjacency, start_label, goal_label, failed_edges)


from namo.core import PlannerFactory

PlannerFactory.register_planner("full_namo", FullNAMOPlanner)
