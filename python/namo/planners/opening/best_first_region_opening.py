"""Region-opening adapter for the canonical best-first search loop."""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import namo_rl
import numpy as np

from namo.core import PlannerConfig, PlannerResult
from namo.planners.connectivity_snapshot import find_robot_label
from namo.planners.utils import PushAttemptBudget
from namo.strategies import PrimitiveGoalStrategy
from namo.strategies.scorer_goal_strategy import _get_scorer
from namo.runtime_profile import CANONICAL_NUM_DEPTHS, CANONICAL_PRIMITIVE_PREFIX

from .best_first_search import make_action, solve_scene
from .region_opening import (
    AttemptResult,
    CANONICAL_MIN_REACHABLE_FRACTION,
    RegionOpeningPlanner,
)


# Canonical search protocol, from docs/experiments/horizon_q_model_registry.md:
# every registered best-first evaluation runs hmax=2 with a 900-simulation
# budget (combine=q, discount off, dedupe+jam on). A config that omits these
# must land on the evaluated protocol -- a shallower search or a smaller budget
# produces numbers that cannot be compared to anything in the registry.
CANONICAL_BEST_FIRST_HMAX = 2
CANONICAL_KEYHOLE_SIMULATION_BUDGET = 900


class _BlacklistedEdgeFilter:
    """Hides blacklisted (object, edge) pairs from candidate enumeration.

    eval_bestfirst reaches every candidate through ``planner.prim.generate_goals``,
    so dropping goals here keeps a push the caller has already seen fail out of
    the pool entirely: never scored, never simulated, never charged to the
    budget. region_bfs gets the same effect by seeding its per-node
    ``edge_min_stuck_depth`` map; this is the equivalent for a search whose
    enumeration lives in shared evaluation code that must not be modified.

    Blacklist format matches region_bfs: ``{object_id: {edge_idx, ...}}`` in
    simulator naming, skipping those edges at every depth.
    """

    def __init__(self, inner: Any, blacklist: Dict[str, Set[int]]):
        self._inner = inner
        self._blacklist = blacklist

    def generate_goals(self, object_id, state, env, max_goals=0):
        goals_per_edge = self._inner.generate_goals(
            object_id, state, env, max_goals=max_goals
        )
        banned = self._blacklist.get(str(object_id))
        if not banned:
            return goals_per_edge
        return [
            [goal for goal in edge_goals if int(goal.edge_idx) not in banned]
            for edge_goals in goals_per_edge
        ]

    def __getattr__(self, name):
        return getattr(self._inner, name)


class BestFirstRegionOpeningPlanner:
    """Solve one robot-adjacent boundary with eval_bestfirst's single global heap."""

    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        params = config.algorithm_params or {}
        self.env = env
        self.config = config
        self.push_budget: PushAttemptBudget = params.get("push_budget") or PushAttemptBudget(
            int(
                params.get(
                    "full_namo_keyhole_simulation_budget",
                    CANONICAL_KEYHOLE_SIMULATION_BUDGET,
                )
            )
        )
        self.prior = str(params.get("best_first_prior", "model"))
        if self.prior not in {"model", "uniform"}:
            raise ValueError("best_first_prior must be 'model' or 'uniform'")
        self.hmax = int(params.get("best_first_hmax", CANONICAL_BEST_FIRST_HMAX))
        if self.hmax < 1:
            raise ValueError(f"Invalid best_first_hmax: {self.hmax}. Must be at least 1")
        self.agg = str(params.get("best_first_agg", "mean5"))
        self.combine = str(params.get("best_first_combine", "q"))
        self.raw = bool(params.get("best_first_raw", True))
        self.seed = int(params.get("shuffle_seed", config.random_seed or 42))
        self.snapshot_seed = int(params.get("region_snapshot_seed", 42))
        self.use_cpp_snapshot = bool(params.get("region_use_cpp_unified_wavefront", True))
        self.goal_radius = params.get("region_goal_radius_m")
        self.goal_radius = float(self.goal_radius) if self.goal_radius is not None else None
        self.min_reachable = int(params.get("region_success_min_reachable", 1))
        # Must match region_bfs: an omitted key lands on the canonical bar, so the
        # two local searches are graded identically. Set explicitly to 0.0 to fall
        # back to the absolute region_success_min_reachable count.
        self.min_fraction = float(
            params.get("region_min_reachable_fraction", CANONICAL_MIN_REACHABLE_FRACTION)
        )
        if not (0.0 <= self.min_fraction <= 1.0):
            raise ValueError(
                f"Invalid region_min_reachable_fraction: {self.min_fraction}. "
                "Must be in [0, 1]"
            )
        # Caller-pinned target points, in simulator metres. Same contract as
        # region_bfs: when supplied, every opening check grades against exactly
        # these instead of points re-sampled from the current snapshot, so a
        # boundary survives the re-partitioning a physical push causes.
        self._pinned_target_points = RegionOpeningPlanner._parse_target_points(
            params.get("region_target_points")
        )
        self.xml_path = str(params.get("xml_file") or "")

        primitive_data_dir = str(params.get("primitive_data_dir", "data"))
        primitive_prefix = str(
            params.get("primitive_prefix", CANONICAL_PRIMITIVE_PREFIX)
        )
        max_push_steps = params.get("max_push_steps", CANONICAL_NUM_DEPTHS)
        prim = PrimitiveGoalStrategy(
            data_dir=primitive_data_dir,
            primitive_prefix=primitive_prefix,
            max_push_steps=max_push_steps,
        )
        scorer = None
        if self.prior == "model":
            ckpt = params.get("scorer_ckpt")
            if not ckpt:
                raise ValueError("best-first model search requires scorer_ckpt")
            scorer = _get_scorer(
                str(ckpt), params.get("namo_config_path"), params.get("ml_device", "cpu")
            )
        # Edges the caller has already seen fail -- on the real robot, pushes that
        # were attempted and did not move the object. region_bfs honours this key;
        # ignoring it here meant the search happily re-proposed a push the robot
        # had just physically failed, and the replan loop could cycle on it.
        external_blacklist = params.get("external_edge_blacklist", None) or {}
        self.external_edge_blacklist: Dict[str, Set[int]] = {
            str(obj): {int(edge) for edge in edges}
            for obj, edges in external_blacklist.items()
        }
        if self.external_edge_blacklist:
            prim = _BlacklistedEdgeFilter(prim, self.external_edge_blacklist)
            if config.verbose:
                print(f"\U0001f6ab External edge blacklist: {dict(self.external_edge_blacklist)}")

        self._search_planner = SimpleNamespace(prim=prim, scorer=scorer)

    def reset(self):
        pass

    @staticmethod
    def _boundary_objects(
        edge_objects: Dict[str, Dict[str, Sequence[str]]], source: str, target: str
    ) -> Tuple[List[str], Optional[str]]:
        forward = edge_objects.get(source, {}).get(target)
        reverse = edge_objects.get(target, {}).get(source)
        if forward is not None and reverse is not None and set(forward) != set(reverse):
            return [], "boundary_object_map_inconsistent"
        return sorted(set(forward if forward is not None else reverse or [])), None

    def _minimum_needed(self, total: int) -> int:
        if self.min_fraction > 0.0:
            return max(1, math.ceil(self.min_fraction * total))
        return self.min_reachable

    @staticmethod
    def _target_summary(
        target: str,
        robot_label: Optional[str],
        neighbors: Sequence[str],
        failure_reason: str,
        attempt_count: int,
        success: bool,
    ) -> Dict[str, Any]:
        immediate = target in neighbors
        return {
            "target_neighbor": target,
            "local_robot_label": robot_label,
            "local_neighbors": sorted(neighbors),
            "target_is_immediate_neighbor": immediate,
            "failure_reason": failure_reason,
            "attempt_count": int(attempt_count),
            "detail_reasons": [failure_reason],
            "boundary_exhausted": immediate and not success and failure_reason in {
                "all_pushes_failed",
                "no_reachable_objects",
            },
        }

    def _result(
        self,
        *,
        target: str,
        robot_label: Optional[str],
        neighbors: Sequence[str],
        attempt: AttemptResult,
        start_time: float,
        sims: int,
        end: str,
        actions: Optional[List[namo_rl.Action]] = None,
    ) -> PlannerResult:
        success = bool(attempt.success)
        failure_kind = "simulation_budget_exhausted" if end == "budget" and not success else None
        stats: Dict[str, Any] = {
            "attempt_results": [attempt],
            "all_solutions": [],
            "successful_openings": int(success),
            "total_attempts": 1,
            "rejection_breakdown": {},
            "total_primitives_attempted": int(sims),
            "best_first_prior": self.prior,
            "best_first_hmax": self.hmax,
            "best_first_end": end,
            "simulation_budget_limit": int(self.push_budget.limit),
            "simulation_budget_used": int(self.push_budget.used),
            "simulation_budget_remaining": int(self.push_budget.remaining),
            "target_summary": self._target_summary(
                target,
                robot_label,
                neighbors,
                str(attempt.failure_reason),
                1,
                success,
            ),
        }
        if success:
            stats["all_solutions"] = [{
                "actions": list(actions or []),
                "neighbor": target,
                "object": attempt.chosen_object_id,
            }]
        if failure_kind:
            stats["failure_kind"] = failure_kind
        return PlannerResult(
            success=success,
            solution_found=success,
            action_sequence=list(actions or []) if success else None,
            solution_depth=len(actions or []) if success else None,
            search_time_ms=(time.time() - start_time) * 1000.0,
            error_message=(
                f"Simulation budget exhausted after {self.push_budget.used}/{self.push_budget.limit} env.step calls"
                if failure_kind
                else ""
            ),
            algorithm_stats=stats,
        )

    def search(
        self,
        robot_goal: Tuple[float, float, float],
        target_neighbor: Optional[str] = None,
    ) -> PlannerResult:
        if target_neighbor is None:
            raise ValueError("best-first region opening requires target_neighbor")
        start_time = time.time()
        baseline = self.env.get_full_state()
        try:
            from namo.planners import get_region_snapshot

            snapshot = get_region_snapshot(
                self.env,
                goals_per_region=self.config.goals_per_region,
                goal_radius=self.goal_radius,
                local_info_only=True,
                seed=self.snapshot_seed,
                use_cpp_unified=self.use_cpp_snapshot,
                use_xml_goal=True,
            )
            robot_label = snapshot.get("robot_label") or find_robot_label(snapshot["region_labels"])
            neighbors = sorted(snapshot["adjacency"].get(robot_label, set())) if robot_label else []
            if not robot_label:
                attempt = AttemptResult(False, target_neighbor, failure_reason="missing_robot_region")
                return self._result(
                    target=target_neighbor, robot_label=None, neighbors=[], attempt=attempt,
                    start_time=start_time, sims=0, end="exhausted",
                )
            if target_neighbor not in neighbors:
                attempt = AttemptResult(False, target_neighbor, failure_reason="target_not_immediate_neighbor")
                return self._result(
                    target=target_neighbor, robot_label=robot_label, neighbors=neighbors, attempt=attempt,
                    start_time=start_time, sims=0, end="exhausted",
                )

            if self._pinned_target_points is not None:
                # theta is unused by count_reachable_points and by the scorer's
                # region channel, so a pinned (x, y) is the whole criterion.
                region_samples = [(x, y, 0.0) for x, y in self._pinned_target_points]
            else:
                bundle = snapshot["region_goals"].get(target_neighbor)
                region_samples = [
                    (float(g.x), float(g.y), float(g.theta)) for g in (bundle.goals if bundle else [])
                ]
            # region_samples feeds BOTH the is_open bar and solve_scene's
            # region_samples argument, which conditions the ranker's
            # goal_sample_region channel -- so the model is scored against the
            # same target the search is graded against.
            xy_samples = [(p[0], p[1]) for p in region_samples]
            before_count, _ = self.env.count_reachable_points(xy_samples) if xy_samples else (0, -1)
            if xy_samples and before_count >= self._minimum_needed(len(xy_samples)):
                attempt = AttemptResult(
                    True,
                    target_neighbor,
                    goal_chain=[],
                    actions_executed=[],
                    resulting_state=baseline,
                    region_goals_sampled=region_samples,
                    failure_reason="already_accessible",
                    push_exec_count=0,
                )
                return self._result(
                    target=target_neighbor, robot_label=robot_label, neighbors=neighbors, attempt=attempt,
                    start_time=start_time, sims=0, end="solved", actions=[],
                )

            boundary_objects, boundary_error = self._boundary_objects(
                snapshot["edge_objects"], robot_label, target_neighbor
            )
            if boundary_error or not boundary_objects:
                reason = boundary_error or "no_blocking_objects"
                attempt = AttemptResult(
                    False,
                    target_neighbor,
                    failure_reason=reason,
                    candidate_objects_count=len(boundary_objects),
                    region_goals_sampled=region_samples,
                )
                return self._result(
                    target=target_neighbor, robot_label=robot_label, neighbors=neighbors, attempt=attempt,
                    start_time=start_time, sims=0, end="exhausted",
                )

            remaining = self.push_budget.remaining
            solution: Dict[str, Any] = {}

            def is_open(env):
                if not xy_samples:
                    return False
                count, _ = env.count_reachable_points(xy_samples)
                return count >= self._minimum_needed(len(xy_samples))

            solved, sims, plan_len, _boards, end = solve_scene(
                self._search_planner,
                self.env,
                robot_goal,
                self.xml_path,
                baseline,
                self.hmax,
                remaining,
                self.prior,
                self.agg,
                self.combine,
                np.random.default_rng(self.seed),
                restrict_obj=boundary_objects,
                is_open=is_open,
                raw=self.raw,
                discount="off",
                dedupe_noop=True,
                prune_jam_depth=True,
                region_samples=region_samples,
                solution_out=solution,
            )
            self.push_budget.used += int(sims)

            plan = list(solution.get("plan", []))
            actions = [make_action(obj, goal) for obj, goal in plan]
            failure_reason = "success" if solved else (
                "no_reachable_objects" if sims == 0 else "all_pushes_failed"
            )
            attempt = AttemptResult(
                solved,
                target_neighbor,
                chosen_object_id=(plan[0][0] if plan else None),
                chosen_goal=(
                    (float(plan[0][1].x), float(plan[0][1].y), float(plan[0][1].theta))
                    if plan else None
                ),
                goal_chain=[goal for _, goal in plan] if plan else None,
                chain_depth=int(plan_len or 0),
                region_goals_sampled=region_samples,
                actions_executed=actions,
                resulting_state=solution.get("state"),
                failure_reason=failure_reason,
                candidate_objects_count=len(boundary_objects),
                pushes_total_for_neighbour=int(sims),
                push_exec_count=int(sims),
            )
            return self._result(
                target=target_neighbor,
                robot_label=robot_label,
                neighbors=neighbors,
                attempt=attempt,
                start_time=start_time,
                sims=sims,
                end=end,
                actions=actions,
            )
        finally:
            self.env.set_full_state(baseline)
