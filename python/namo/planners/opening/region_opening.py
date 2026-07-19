"""Region Opening Planner for NAMO.

This planner creates an opening from the robot's region to each immediate neighbour
region. For each neighbour, it picks a blocking object, samples push goals, executes,
validates the opening, logs an episode, then restores the baseline and proceeds to
the next neighbour.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set, Union

import namo_rl

# Default camera settings for top-down view
DEFAULT_CAMERA_DISTANCE = 15.0
DEFAULT_CAMERA_AZIMUTH = 0.0
DEFAULT_CAMERA_ELEVATION = -90.0
from namo.core import BasePlanner, PlannerConfig, PlannerResult
from namo.planners.connectivity_snapshot import find_robot_label
from namo.strategies import (
    PrimitiveGoalStrategy,
    RandomRolloutGoalStrategy,
    Goal,
    MLPrimitiveGoalStrategy,
    MLPrimitiveFallbackStrategy,
    MLPrimitiveAsyncStrategy,
    AsyncGoalResult,
    GeometricTransportStrategy,
    ScorerGoalStrategy,
)
from namo.planners.opening.ml_driven_search import MLDrivenAsyncSearch
from namo.planners.utils import PushBudgetExceeded


def _sort_candidates_sync(
    candidates: List[List[object]],
    *,
    depth_first: bool,
) -> None:
    """Sort in-place the candidate list [edge_idx, depth_idx, goal]."""

    if depth_first:
        # Depth-first, then score (descending), then edge index for determinism.
        candidates.sort(key=lambda x: (x[1], -float(getattr(x[2], "score", 0.0)), x[0]))
    else:
        # Score-first (descending), then depth, then edge index for determinism.
        candidates.sort(key=lambda x: (-float(getattr(x[2], "score", 0.0)), x[1], x[0]))


@dataclass
class ChainNode:
    """Node in the skill chaining search tree."""
    state: namo_rl.RLState  # Environment state after this push
    goal: Goal  # Goal that led to this state
    edge_idx: int  # Edge index used
    depth: int  # Chain depth (1, 2, or 3)
    parent: Optional['ChainNode'] = None  # Parent node in chain
    collided_edges: Set[int] = field(default_factory=set)  # Edges that collided at this state
    # Cost of this step within its inner BFS call (primitive depth, 1-based)
    # For root node (no action), this remains 0
    step_cost: int = 0
    skill_calls_before_success: int = 0


@dataclass
class AttemptResult:
    """Result from attempting to open a path to a neighbour region."""

    success: bool
    neighbour_region_label: str
    chosen_object_id: Optional[str] = None
    chosen_goal: Optional[Tuple[float, float, float]] = None
    goal_chain: Optional[List[Goal]] = None  # Chain of goals that led to success
    chain_depth: int = 1  # Number of pushes in the successful chain
    validation_method: str = "connectivity"
    connectivity_before: Optional[Dict] = None
    connectivity_after: Optional[Dict] = None
    region_goal_used: Optional[Tuple[float, float, float]] = None  # First reachable goal (for validation)
    region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None  # All goal samples for the neighbor region
    error_message: Optional[str] = None
    actions_executed: List[namo_rl.Action] = field(default_factory=list)
    state_observations: Optional[List[Dict[str, List[float]]]] = None  # State before each action
    post_action_state_observations: Optional[List[Dict[str, List[float]]]] = None  # State after each action
    reachable_objects_before_action: Optional[List[List[str]]] = None  # Reachable objects before each action
    reachable_objects_after_action: Optional[List[List[str]]] = None  # Reachable objects after each action
    exploration_state: Optional['namo_rl.RLState'] = None  # State we were exploring from when this opening was found
    resulting_state: Optional['namo_rl.RLState'] = None  # Full state after executing this opening (for multi-level exploration)
    exploration_level: int = 0  # Which exploration level this opening was found at (0 = initial state)
    timing_ms: Optional[float] = None
    # Optional goal-strategy profiling payload (e.g., geometric transport timing breakdown)
    goal_strategy_profile: Optional[Dict[str, Any]] = None
    # Total additive cost of the chain (sum of inner primitive depths)
    total_cost: int = 0
    # Neighbour-level solution accounting
    solutions_found_for_neighbour: int = 0
    solutions_cap_for_neighbour: int = 0
    skill_calls_before_success: int = 0
    # Total successful solutions found for this neighbour during search
    solutions_total_for_neighbour: int = 0
    # Total pushes executed (env.step calls) for this neighbour during search
    pushes_total_for_neighbour: int = 0
    # Failure tracking: categorize WHY there were 0 solutions/pushes
    # Values:
    #   "success"              - Found a valid opening (not a failure)
    #   "already_accessible"   - Neighbor was already reachable (0 pushes needed)
    #   "no_blocking_objects"  - No objects identified on the region edge
    #   "no_reachable_objects" - Blocking objects exist but robot can't reach them
    #   "ml_no_goals_extracted"- ML model produced 0 goals (mask extraction failed)
    #   "ml_goals_not_aligned" - ML produced goals but none matched primitive slots
    #   "no_reachable_edges"   - Goals aligned but none on edges robot can reach
    #   "no_valid_goals"       - Fallback: goals existed but weren't tried
    #   "all_pushes_failed"    - N pushes executed but none created opening
    #   "timeout"              - Search timed out
    failure_reason: Optional[str] = None
    # ML goal generation stats (for debugging ML model quality)
    ml_goals_generated: int = 0  # Raw ML goals before primitive alignment
    ml_goals_aligned: int = 0    # ML goals that matched a primitive slot
    # Number of diffusion model calls (GoalInferenceModel.infer invocations) used while
    # searching this (neighbour, object) attempt. Purely additive; helps measure volatility.
    ml_diffusion_calls: int = 0
    # Time spent attaching ML mask samples to primitive vote bins.
    ml_mask_vote_attach_calls: int = 0
    ml_mask_vote_attach_ms_total: float = 0.0
    ml_mask_vote_attach_ms_avg: float = 0.0
    reachable_edges_count: int = 0  # Number of reachable edges for the object
    candidate_objects_count: int = 0  # Number of candidate blocking objects
    # Detailed ML goal info (for analysis/visualization)
    # Each entry: {'edge_idx': int, 'depth_idx': int, 'x': float, 'y': float, 'theta': float, 'votes': int}
    aligned_primitives: Optional[List[Dict]] = None
    # Raw ML goals before alignment: [{'x': float, 'y': float, 'theta': float}, ...]
    ml_goals_raw: Optional[List[Dict]] = None
    # List of reachable edge indices
    reachable_edges: Optional[List[int]] = None
    # Collision tracking for hardness metrics (aggregated across all pushes in chain)
    any_wall_collision: bool = False  # Did any push hit a wall?
    unique_movable_collision_count: int = 0  # Number of unique movable objects hit across all pushes
    # Phase tracking for hybrid decomposition analysis
    # Tracks pushes per search phase: {"ML-only": X, "primitives": Y}
    phase_push_counts: Optional[Dict[str, int]] = None
    # Which phase found the solution: "ML-only", "primitives", or "" if not found
    solved_in_phase: str = ""
    # Primitive ranking/sorting timings in BFS candidate ordering.
    primitive_ranking_calls: int = 0
    primitive_ranking_ms_total: float = 0.0
    primitive_ranking_ms_avg: float = 0.0
    primitive_ranking_candidates_total: int = 0
    primitive_ranking_candidates_avg: float = 0.0
    # Single-push execution timings (env.step only), with depth breakdown.
    push_exec_count: int = 0
    push_exec_ms_total: float = 0.0
    push_exec_ms_avg: float = 0.0
    push_exec_ms_by_depth: Optional[Dict[str, Dict[str, float]]] = None
    # Additional runtime timing buckets to reduce unattributed wall-clock time.
    goal_generation_calls: int = 0
    goal_generation_ms_total: float = 0.0
    goal_generation_ms_avg: float = 0.0
    opening_validation_calls: int = 0
    opening_validation_ms_total: float = 0.0
    opening_validation_ms_avg: float = 0.0
    opening_validation_goal_checks_total: int = 0
    opening_validation_goal_checks_avg_per_call: float = 0.0
    opening_validation_reachability_calls: int = 0
    opening_validation_reachability_ms_total: float = 0.0
    opening_validation_reachability_ms_avg: float = 0.0
    chain_observation_replay_calls: int = 0
    chain_observation_replay_ms_total: float = 0.0
    chain_observation_replay_ms_avg: float = 0.0
    # Per-primitive trial log for F characterization (exhaustive mode)
    # Each entry: {'edge_idx': int, 'depth': int, 'success': bool,
    #              'wall_collision': bool, 'movable_collisions': str,
    #              'stuck': bool, 'collision': bool, 'reachable_after': int}
    primitive_trial_log: Optional[List[Dict]] = None
    # Per-EXPANDED-NODE reachable edge set ([USER 2026-06-12]: record reachability going forward): one
    # entry per node whose candidates were generated: chain_depth (node.depth+1; root=1), parent_edge/
    # parent_depth (the action that created the node; None for root), reachable_edges (sorted indices
    # from env.get_reachable_edges at THAT node's state). Joins 1:1 with primitive_trial_log's level/
    # parent tags; level>=2 entries capture POST-PUSH-state reachability for free. Under sampled
    # restarts the same node identity may repeat (identical set) — dedupe by identity downstream.
    reachability_log: Optional[List[Dict]] = None


class RegionOpeningPlanner(BasePlanner):
    """Region opening planner for creating paths to neighbour regions.

    For the current scene, this planner creates an opening from the robot's region
    to each immediate neighbour region (one neighbour per attempt). For each neighbour,
    it picks a blocking object, samples push goals, executes, validates the opening,
    and logs an episode.
    """

    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        """Initialize region opening planner.

        Args:
            env: NAMO RL environment
            config: Planner configuration (uses algorithm_params for region_search_strategy)
        """
        self.attempt_results: List[AttemptResult] = []

        algo_params = config.algorithm_params or {}
        self.algorithm_params = algo_params
        self.push_budget = algo_params.get("push_budget")

        # Get collision termination flag from config.algorithm_params
        # region_allow_collisions=True means ALLOW collisions (don't terminate)
        # We invert it: terminate_on_collision=True means TERMINATE on collision
        allow_collisions = algo_params.get("region_allow_collisions", False)
        self.terminate_on_collision = not allow_collisions

        # Get max chain depth from config.algorithm_params (default: 1, no chaining)
        self.max_chain_depth = algo_params.get("region_max_chain_depth", 1)
        if self.max_chain_depth < 1 or self.max_chain_depth > 10:
            raise ValueError(f"Invalid max_chain_depth: {self.max_chain_depth}. Must be between 1 and 10")

        # Get max solutions per neighbor from config.algorithm_params (default: 10)
        self.max_solutions_per_neighbor = algo_params.get("region_max_solutions_per_neighbor", 10)
        if self.max_solutions_per_neighbor < 1:
            raise ValueError(f"Invalid max_solutions_per_neighbor: {self.max_solutions_per_neighbor}. Must be at least 1")

        # Exhaustive mode: disable search pruning, log all primitive outcomes for F characterization
        self.exhaustive_mode = algo_params.get("region_exhaustive_mode", False)

        # SAMPLED collection (horizon-Q, [USER 2026-06-12]: "sample all levels at 30; keep the data and
        # know the mask during training"): cap the per-node candidate list at a UNIFORM random subset of
        # k (edge, depth) cells. Applies at EVERY chain level (each node's expansion passes through the
        # same candidate build). Labels stay exact per tried cell; training uses the tried set as the
        # loss mask (B30). Uses the worker RNG unseeded-per-scene, so REPEATING an instance draws a fresh
        # subset (randomized-repeats lever). 0 = off (legacy exhaustive-over-reachable).
        self.sample_k = int(algo_params.get("region_sample_k", 0))
        # LABEL mode (Beast 2-push): exhaustive setups + scorer-ordered finish with EARLY-STOP at the first
        # opening finish; setups stay exhaustive; log each finish's score+rank for recall-miss recycling;
        # disable cost-optimality pruning so setups in direct-opener scenes still get a finish sweep. See
        # card EXP-2026-07-14 "Beast (2-push)". False = off (rung-1/legacy behaviour unchanged).
        self.label_mode = bool(algo_params.get("region_label_mode", False))
        # label-mode top-k cap on the FINISH sweep (deepest level only): stop after k failed finish
        # candidates instead of exhausting (~60). Misses become honest ceilings, never false labels.
        # Pilot-validated (2026-07-19, n=2.4M): antman-5c k=15 retains 97.7% of successes at ~30% cost.
        # 0 = exhaustive (round-0 behavior). Setup enumeration (depth-1) is NEVER capped by this.
        self.label_topk = int(algo_params.get("region_label_topk", 0) or 0)
        # Restart-on-failure ([USER]: "run the same instance up to 3 times with different seeds, only if
        # we don't find a solution"): total search attempts per (object, neighbour); each retry draws
        # fresh random subsets at every level; trial logs are MERGED (union of tried cells = the training
        # mask); stops at the first attempt that finds any chain. 1 = off.
        self.sample_restarts = int(algo_params.get("region_sample_restarts", 1))

        # Early-exit the candidate-object loop in _attempt_opening_to_neighbour after
        # the first object yields a successful opening. This skips evaluating remaining
        # candidate objects for the same neighbor — fine for execution (we only need
        # ONE valid push per region transition; outer FullNAMOPlanner picks the next
        # region after that). For data collection we want per-object outcomes, so the
        # default is False to preserve eval coverage.
        self.early_exit_on_first_success = algo_params.get(
            "region_early_exit_on_first_success", False
        )

        # Minimum number of sampled region-goal points that must be reachable
        # for a region to count as "opened" (success). Paired with
        # goals_per_region (how many points are sampled): e.g. 100 sampled / 20
        # reachable is a stricter "meaningfully open" bar than the legacy
        # 10 sampled / >=1. Default 1 preserves the historical criterion so
        # other callers (data collection, eval) are unaffected.
        self._success_min_reachable = int(
            algo_params.get("region_success_min_reachable", 1)
        )
        if self._success_min_reachable < 1:
            raise ValueError(
                f"Invalid region_success_min_reachable: "
                f"{self._success_min_reachable}. Must be at least 1"
            )

        # Fractional success bar (OPT-IN): a region counts as "opened" only when at least
        # ``fraction × (#sampled goal points in that region)`` are reachable — a stricter,
        # area-aware criterion than the absolute count above (rejects flake openings where a
        # sliver of the goal disc becomes reachable). 0.0 (default) = disabled → fall back to
        # the absolute ``region_success_min_reachable`` count (backward-compatible).
        self._min_reachable_fraction = float(
            algo_params.get("region_min_reachable_fraction", 0.0)
        )
        if not (0.0 <= self._min_reachable_fraction <= 1.0):
            raise ValueError(
                f"Invalid region_min_reachable_fraction: "
                f"{self._min_reachable_fraction}. Must be in [0, 1]"
            )

        # Get max recorded solutions per neighbor (subset of found solutions to keep), default: 2
        self.max_recorded_solutions_per_neighbor = algo_params.get(
            "region_max_recorded_solutions_per_neighbor", 2
        )
        if self.max_recorded_solutions_per_neighbor < 1:
            raise ValueError(
                f"Invalid region_max_recorded_solutions_per_neighbor: {self.max_recorded_solutions_per_neighbor}. Must be at least 1"
            )

        # When True, stop attempting additional objects for a neighbour once the
        # configured max_solutions has been collected for that neighbour.
        # Default is False to preserve legacy behaviour (useful for evaluation/triplet logging).
        self.stop_after_max_solutions = algo_params.get("region_stop_after_max_solutions", False)

        # When True, stop exploring additional neighbour regions as soon as any
        # successful opening is found from the current exploration state.
        # Default is False to preserve legacy behaviour (collect per-neighbour data).
        self.stop_after_first_success = algo_params.get("region_stop_after_first_success", False)

        # When True, auto-target the region containing the XML's <site name="goal">
        # (which the snapshot labels as "goal") instead of opening all unreachable
        # neighbours. If the robot is already in the goal region, no episodes are
        # recorded. If the goal region is not an immediate neighbour, one fail
        # AttemptResult is recorded with failure_reason="target_not_immediate_neighbor"
        # (signal for phase-2 deeper-chain retries).
        self.target_goal_region = bool(algo_params.get("target_goal_region", False))

        # Optional: cap number of frontier nodes per chain level (beam width)
        # None or 0 => unbounded frontier (complete)
        beam_width = algo_params.get("region_frontier_beam_width", None)
        if isinstance(beam_width, int) and beam_width <= 0:
            beam_width = None
        self.frontier_beam_width = beam_width

        # Timeout per neighbour in seconds (default: None = no timeout)
        # e.g., region_timeout_per_neighbour_sec=1200 for 20 minutes
        timeout_sec = algo_params.get("region_timeout_per_neighbour_sec", None)
        if timeout_sec is not None and timeout_sec <= 0:
            timeout_sec = None
        self.timeout_per_neighbour_sec = timeout_sec

        # Unified wavefront snapshot backend:
        # - True: prefer C++ `env.get_region_snapshot(...)`
        # - False: fallback to Python snapshot exporter
        self.use_cpp_unified_wavefront = algo_params.get("region_use_cpp_unified_wavefront", True)
        self.region_snapshot_seed = int(algo_params.get("region_snapshot_seed", 42))
        region_goal_radius = algo_params.get("region_goal_radius_m", None)
        self.region_goal_radius_m = float(region_goal_radius) if region_goal_radius is not None else None

        # ML blacklist override: when True, ML-scored primitives bypass the
        # edge blacklist built during pre-ML exhaustive phase. This allows
        # ML suggestions to be tried even if earlier depth-first exploration
        # caused collisions on that edge. (default: False for backward compat)
        self.ml_ignore_blacklist = algo_params.get("region_ml_ignore_blacklist", False)

        # Chain link cost: additional cost added per chain link beyond the first push.
        # With chain_link_cost=0 (default), a 2-push chain at depths [0,0] costs 2.
        # With chain_link_cost=5, the same chain costs 2 + 5 = 7.
        self.chain_link_cost = algo_params.get("region_chain_link_cost", 0)

        # Selection strategy for multi-push frontier prioritization:
        # - "ml_first": (-score, chain_cost, step_cost) - trust ML votes, cost as tiebreaker
        # - "cost_first": (chain_cost, step_cost, -score) - minimize disruption, ML as tiebreaker
        # Default: "ml_first" to prioritize ML-derived states
        self.selection_strategy = algo_params.get("region_selection_strategy", "ml_first")
        if self.selection_strategy not in {"cost_first", "ml_first"}:
            raise ValueError(f"Invalid selection_strategy: {self.selection_strategy}. Must be 'cost_first' or 'ml_first'")

        # Region/object skip dict: skip specific (region, object) pairs during neighbor exploration.
        # Format: Dict[region_label, List[object_id]] where empty list = skip entire region
        # Example: {"goal": [], "region_2": ["box_1", "box_2"]}
        #   - "goal" with empty list: skip entire goal region
        #   - "region_2" with ["box_1", "box_2"]: only skip those objects when opening region_2
        region_object_skip = algo_params.get("region_object_skip", None)
        if region_object_skip is None:
            self.region_object_skip = {}
        elif isinstance(region_object_skip, dict):
            self.region_object_skip = region_object_skip
        else:
            # Legacy: list of region names (backward compatible)
            self.region_object_skip = {r: [] for r in region_object_skip}

        # Log skip config at planner init (only if verbose)
        if self.region_object_skip and config.verbose:
            skip_regions = [r for r, objs in self.region_object_skip.items() if not objs]
            skip_objects = [(r, objs) for r, objs in self.region_object_skip.items() if objs]
            parts = []
            if skip_regions:
                parts.append(f"regions={skip_regions}")
            if skip_objects:
                parts.append(f"objects={skip_objects}")
            print(f"🔧 Skip config: {', '.join(parts)}")

        # Externally-provided per-object edge blacklist, fed in from the caller
        # (e.g. the real-robot executor reports edges that failed at runtime).
        # Format: Dict[object_id, Iterable[int]] — edges to skip at ALL depths
        # for that object. Seeds the per-node `edge_min_stuck_depth` map with
        # depth=0 so the BFS treats these edges as already-stuck.
        external_blacklist = algo_params.get("external_edge_blacklist", None) or {}
        self.external_edge_blacklist: Dict[str, Set[int]] = {
            str(obj): {int(e) for e in edges}
            for obj, edges in external_blacklist.items()
        }
        if self.external_edge_blacklist and config.verbose:
            print(f"🚫 External edge blacklist: {dict(self.external_edge_blacklist)}")

        # Visualization settings (can be set after init, like IDFS planners)
        self.visualize_search = False
        self.search_delay = 0.5
        self.step_mode = False

        # ML-driven async search flag (set in _initialize_algorithm)
        self._use_ml_driven_async = False
        self._primitive_strategy = None
        self._ml_async_strategy = None
        self._runtime_timing_stats = self._new_runtime_timing_stats()

        # Progress reporter state — prints [Progress] line every progress_interval_sec
        # while the BFS is grinding through primitives. Reset each search() call.
        self._progress_total_primitives = 0
        self._progress_last_print_time = 0.0
        self._progress_last_print_count = 0
        self._progress_interval_sec = 2.0

        # Rejection-reason tally — surfaced in algorithm_stats so callers can
        # render a diagnostic breakdown when planning fails to find a plan.
        self._rejection_stats: Dict[str, int] = {}
        self._last_explore_context: Optional[Dict[str, Any]] = None

        super().__init__(env, config)

    @staticmethod
    def _new_runtime_timing_stats() -> Dict[str, Any]:
        return {
            "primitive_ranking_calls": 0,
            "primitive_ranking_ms_total": 0.0,
            "primitive_ranking_candidates_total": 0,
            "push_exec_count": 0,
            "push_exec_ms_total": 0.0,
            "push_exec_ms_by_depth": {},
            "goal_generation_calls": 0,
            "goal_generation_ms_total": 0.0,
            "opening_validation_calls": 0,
            "opening_validation_ms_total": 0.0,
            "opening_validation_goal_checks_total": 0,
            "opening_validation_reachability_calls": 0,
            "opening_validation_reachability_ms_total": 0.0,
            "chain_observation_replay_calls": 0,
            "chain_observation_replay_ms_total": 0.0,
        }

    def _reset_runtime_timing_stats(self) -> None:
        self._runtime_timing_stats = self._new_runtime_timing_stats()

    def _record_primitive_ranking_timing(self, elapsed_ms: float, candidate_count: int) -> None:
        stats = self._runtime_timing_stats
        stats["primitive_ranking_calls"] += 1
        stats["primitive_ranking_ms_total"] += max(0.0, float(elapsed_ms))
        stats["primitive_ranking_candidates_total"] += max(0, int(candidate_count))

    def _record_push_exec_timing(self, elapsed_ms: float, primitive_depth_1_indexed: int) -> None:
        stats = self._runtime_timing_stats
        stats["push_exec_count"] += 1
        stats["push_exec_ms_total"] += max(0.0, float(elapsed_ms))

        depth_key = str(int(primitive_depth_1_indexed))
        depth_map = stats["push_exec_ms_by_depth"]
        if depth_key not in depth_map:
            depth_map[depth_key] = {"count": 0, "ms_total": 0.0}
        depth_map[depth_key]["count"] += 1
        depth_map[depth_key]["ms_total"] += max(0.0, float(elapsed_ms))

    def _record_goal_generation_timing(self, elapsed_ms: float) -> None:
        stats = self._runtime_timing_stats
        stats["goal_generation_calls"] += 1
        stats["goal_generation_ms_total"] += max(0.0, float(elapsed_ms))

    def _record_opening_validation_timing(
        self,
        elapsed_ms: float,
        goal_checks: int = 0,
        reachability_calls: int = 0,
        reachability_ms: float = 0.0,
    ) -> None:
        stats = self._runtime_timing_stats
        stats["opening_validation_calls"] += 1
        stats["opening_validation_ms_total"] += max(0.0, float(elapsed_ms))
        stats["opening_validation_goal_checks_total"] += max(0, int(goal_checks))
        stats["opening_validation_reachability_calls"] += max(0, int(reachability_calls))
        stats["opening_validation_reachability_ms_total"] += max(0.0, float(reachability_ms))

    def _record_chain_observation_replay_timing(self, elapsed_ms: float) -> None:
        stats = self._runtime_timing_stats
        stats["chain_observation_replay_calls"] += 1
        stats["chain_observation_replay_ms_total"] += max(0.0, float(elapsed_ms))

    def _get_runtime_timing_summary(self) -> Dict[str, Any]:
        stats = self._runtime_timing_stats
        rank_calls = int(stats.get("primitive_ranking_calls", 0))
        rank_ms_total = float(stats.get("primitive_ranking_ms_total", 0.0))
        rank_candidates_total = int(stats.get("primitive_ranking_candidates_total", 0))

        push_count = int(stats.get("push_exec_count", 0))
        push_ms_total = float(stats.get("push_exec_ms_total", 0.0))
        goal_gen_calls = int(stats.get("goal_generation_calls", 0))
        goal_gen_ms_total = float(stats.get("goal_generation_ms_total", 0.0))
        validation_calls = int(stats.get("opening_validation_calls", 0))
        validation_ms_total = float(stats.get("opening_validation_ms_total", 0.0))
        validation_goal_checks_total = int(stats.get("opening_validation_goal_checks_total", 0))
        validation_reachability_calls = int(stats.get("opening_validation_reachability_calls", 0))
        validation_reachability_ms_total = float(stats.get("opening_validation_reachability_ms_total", 0.0))
        replay_calls = int(stats.get("chain_observation_replay_calls", 0))
        replay_ms_total = float(stats.get("chain_observation_replay_ms_total", 0.0))

        push_by_depth_raw = stats.get("push_exec_ms_by_depth", {}) or {}
        push_by_depth_summary: Dict[str, Dict[str, float]] = {}
        for depth_key, depth_stats in push_by_depth_raw.items():
            d_count = int(depth_stats.get("count", 0))
            d_total = float(depth_stats.get("ms_total", 0.0))
            push_by_depth_summary[str(depth_key)] = {
                "count": d_count,
                "ms_total": d_total,
                "ms_avg": (d_total / d_count) if d_count > 0 else 0.0,
            }

        return {
            "primitive_ranking_calls": rank_calls,
            "primitive_ranking_ms_total": rank_ms_total,
            "primitive_ranking_ms_avg": (rank_ms_total / rank_calls) if rank_calls > 0 else 0.0,
            "primitive_ranking_candidates_total": rank_candidates_total,
            "primitive_ranking_candidates_avg": (rank_candidates_total / rank_calls) if rank_calls > 0 else 0.0,
            "push_exec_count": push_count,
            "push_exec_ms_total": push_ms_total,
            "push_exec_ms_avg": (push_ms_total / push_count) if push_count > 0 else 0.0,
            "push_exec_ms_by_depth": push_by_depth_summary,
            "goal_generation_calls": goal_gen_calls,
            "goal_generation_ms_total": goal_gen_ms_total,
            "goal_generation_ms_avg": (goal_gen_ms_total / goal_gen_calls) if goal_gen_calls > 0 else 0.0,
            "opening_validation_calls": validation_calls,
            "opening_validation_ms_total": validation_ms_total,
            "opening_validation_ms_avg": (validation_ms_total / validation_calls) if validation_calls > 0 else 0.0,
            "opening_validation_goal_checks_total": validation_goal_checks_total,
            "opening_validation_goal_checks_avg_per_call": (
                validation_goal_checks_total / validation_calls
            ) if validation_calls > 0 else 0.0,
            "opening_validation_reachability_calls": validation_reachability_calls,
            "opening_validation_reachability_ms_total": validation_reachability_ms_total,
            "opening_validation_reachability_ms_avg": (
                validation_reachability_ms_total / validation_reachability_calls
            ) if validation_reachability_calls > 0 else 0.0,
            "chain_observation_replay_calls": replay_calls,
            "chain_observation_replay_ms_total": replay_ms_total,
            "chain_observation_replay_ms_avg": (replay_ms_total / replay_calls) if replay_calls > 0 else 0.0,
        }

    def _setup_constraints(self):
        """Setup action constraints from environment."""
        # No constraints needed for primitive strategy
        pass

    def _initialize_algorithm(self):
        """Initialize algorithm-specific components."""
        # Random seed
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

        algo_params = getattr(self, "algorithm_params", {}) or {}
        primitive_data_dir = algo_params.get("primitive_data_dir", "data")
        strategy_name = algo_params.get("goal_strategy")
        max_push_steps = algo_params.get("max_push_steps", None)

        if strategy_name and strategy_name.lower() in {"ml", "ml_primitive"}:
            ml_path = algo_params.get("ml_goal_model_path")
            if not ml_path:
                raise ValueError("ML primitive goal sampler requires 'ml_goal_model_path'")

            self.goal_strategy = MLPrimitiveGoalStrategy(
                goal_model_path=ml_path,
                primitive_data_dir=primitive_data_dir,
                samples=algo_params.get("ml_samples", 32),
                device=algo_params.get("ml_device", "cuda"),
                match_position_tolerance=algo_params.get("ml_match_position_tolerance", 0.1),
                match_angle_tolerance=algo_params.get("ml_match_angle_tolerance", 0.1),
                angle_weight=algo_params.get("ml_match_angle_weight", 0.5),
                max_matches=algo_params.get("ml_match_max_per_call", 8),
                verbose=self.config.verbose,
                min_goals_threshold=algo_params.get("ml_min_goals", 1),
                xml_path=algo_params.get("xml_file"),
                preview_mask_count=algo_params.get("preview_ml_goal_masks", 0),
                preloaded_model=algo_params.get("preloaded_goal_model"),
                preview_aligned_primitives=algo_params.get("preview_aligned_primitives", False),
                k_nearest=algo_params.get("ml_k_nearest", 1),
                seed=algo_params.get("ml_seed"),
                sampler_method=algo_params.get("ml_sampler_method"),
                num_steps=algo_params.get("ml_num_steps"),
                primitive_prefix=algo_params.get("primitive_prefix", ""),
                max_push_steps=max_push_steps,
                namo_config_path=algo_params.get("namo_config_path"),
            )
            self._debug("▶ Using ML-aligned primitive goal strategy")
        elif strategy_name and strategy_name.lower() in {"ml_fallback", "ml_primitive_fallback"}:
            ml_path = algo_params.get("ml_goal_model_path")
            if not ml_path:
                raise ValueError("ML fallback goal sampler requires 'ml_goal_model_path'")

            self.goal_strategy = MLPrimitiveFallbackStrategy(
                goal_model_path=ml_path,
                primitive_data_dir=primitive_data_dir,
                samples=algo_params.get("ml_samples", 32),
                device=algo_params.get("ml_device", "cuda"),
                match_position_tolerance=algo_params.get("ml_match_position_tolerance", 0.1),
                match_angle_tolerance=algo_params.get("ml_match_angle_tolerance", 0.1),
                angle_weight=algo_params.get("ml_match_angle_weight", 0.5),
                max_matches=algo_params.get("ml_match_max_per_call", 8),
                verbose=self.config.verbose,
                min_goals_threshold=algo_params.get("ml_min_goals", 1),
                xml_path=algo_params.get("xml_file"),
                preview_mask_count=algo_params.get("preview_ml_goal_masks", 0),
                preloaded_model=algo_params.get("preloaded_goal_model"),
                preview_aligned_primitives=algo_params.get("preview_aligned_primitives", False),
                k_nearest=algo_params.get("ml_k_nearest", 1),
                seed=algo_params.get("ml_seed"),
                sampler_method=algo_params.get("ml_sampler_method"),
                num_steps=algo_params.get("ml_num_steps"),
            )
            self._debug("▶ Using ML-first with primitive fallback goal strategy")
        elif strategy_name and strategy_name.lower() in {"ml_async", "ml_primitive_async"}:
            ml_path = algo_params.get("ml_goal_model_path")
            if not ml_path:
                raise ValueError("ML async goal sampler requires 'ml_goal_model_path'")

            self.goal_strategy = MLPrimitiveAsyncStrategy(
                goal_model_path=ml_path,
                primitive_data_dir=primitive_data_dir,
                samples=algo_params.get("ml_samples", 32),
                device=algo_params.get("ml_device", "cuda"),
                match_position_tolerance=algo_params.get("ml_match_position_tolerance", 0.1),
                match_angle_tolerance=algo_params.get("ml_match_angle_tolerance", 0.1),
                angle_weight=algo_params.get("ml_match_angle_weight", 0.5),
                verbose=self.config.verbose,
                min_goals_threshold=algo_params.get("ml_min_goals", 1),
                xml_path=algo_params.get("xml_file"),
                preloaded_model=algo_params.get("preloaded_goal_model"),
                k_nearest=algo_params.get("ml_k_nearest", 1),
                max_workers=algo_params.get("ml_async_workers", 1),
                seed=algo_params.get("ml_seed"),
                sampler_method=algo_params.get("ml_sampler_method"),
                num_steps=algo_params.get("ml_num_steps"),
            )
            self._debug("▶ Using async ML with primitive pre-execution goal strategy")
        elif strategy_name and strategy_name.lower() in {"ml_driven_async"}:
            # ML-Driven Async: uses MLDrivenAsyncSearch with zero idle time guarantee
            ml_path = algo_params.get("ml_goal_model_path")
            if not ml_path:
                raise ValueError("ML driven async requires 'ml_goal_model_path'")

            # Store strategies for MLDrivenAsyncSearch
            self._primitive_strategy = PrimitiveGoalStrategy(
                data_dir=primitive_data_dir,
                verbose=self.config.verbose,
                primitive_prefix=algo_params.get("primitive_prefix", ""),
            )
            self._ml_async_strategy = MLPrimitiveAsyncStrategy(
                goal_model_path=ml_path,
                primitive_data_dir=primitive_data_dir,
                samples=algo_params.get("ml_samples", 32),
                device=algo_params.get("ml_device", "cuda"),
                match_position_tolerance=algo_params.get("ml_match_position_tolerance", 0.1),
                match_angle_tolerance=algo_params.get("ml_match_angle_tolerance", 0.1),
                angle_weight=algo_params.get("ml_match_angle_weight", 0.5),
                verbose=self.config.verbose,
                min_goals_threshold=algo_params.get("ml_min_goals", 1),
                xml_path=algo_params.get("xml_file"),
                preloaded_model=algo_params.get("preloaded_goal_model"),
                k_nearest=algo_params.get("ml_k_nearest", 1),
                max_workers=1,  # Always 1 - GPU runs 1 ML inference at a time
                seed=algo_params.get("ml_seed"),
                sampler_method=algo_params.get("ml_sampler_method"),
                num_steps=algo_params.get("ml_num_steps"),
            )
            # Set goal_strategy to primitive for compatibility (MLDrivenAsyncSearch handles ML internally)
            self.goal_strategy = self._primitive_strategy
            self._use_ml_driven_async = True
            self._debug("▶ Using ML-driven async search (zero idle time, ML priority)")
        elif strategy_name and strategy_name.lower() in {"scorer", "f_scorer"}:
            # Champion 1-push F-scorer ranks candidate pushes by P(opens neighbour region).
            # Same primitive enumeration as the default; only goal.score changes -> RO tries
            # high-P pushes first -> fewer forward sims to find the opening.
            scorer_kwargs = dict(
                namo_config_path=algo_params.get("namo_config_path"),
                xml_path=algo_params.get("xml_file"),
                device=algo_params.get("ml_device", "cuda"),
                data_dir=primitive_data_dir,
                primitive_prefix=algo_params.get("primitive_prefix", ""),
                verbose=self.config.verbose,
                max_push_steps=max_push_steps,
            )
            scorer_ckpt = algo_params.get("scorer_ckpt")
            if scorer_ckpt:
                scorer_kwargs["ckpt"] = scorer_ckpt
            self.goal_strategy = ScorerGoalStrategy(**scorer_kwargs)
            self._debug("▶ Using scorer-guided goal strategy (F-scorer ranks pushes)")
        elif strategy_name and strategy_name.lower() in {"geometric", "geometric_transport"}:
            # Use geometric transport heuristic for goal prioritization
            self.goal_strategy = GeometricTransportStrategy(
                primitive_data_dir=primitive_data_dir,
                verbose=self.config.verbose,
                profile=bool(algo_params.get("profile_geometric", False)),
            )
            self._debug("▶ Using geometric transport goal strategy")
        elif strategy_name and strategy_name.lower() in {"random_rollout", "random"}:
            # Random rollouts: same primitive enumeration, but random scores
            # (uniform per-state ordering) and optional cap on candidates per
            # state. Combined with max_chain_depth, gives trial-style search.
            self.goal_strategy = RandomRolloutGoalStrategy(
                data_dir=primitive_data_dir,
                verbose=self.config.verbose,
                samples_per_state=algo_params.get("rollout_samples_per_state", None),
                seed=algo_params.get("shuffle_seed", None),
                primitive_prefix=algo_params.get("primitive_prefix", ""),
                max_push_steps=max_push_steps,
            )
            self._debug(
                "▶ Using random-rollout goal strategy "
                f"(samples_per_state={algo_params.get('rollout_samples_per_state', None)})"
            )
        else:
            # Use primitive goal strategy for push goals
            self.goal_strategy = PrimitiveGoalStrategy(
                data_dir=primitive_data_dir,
                verbose=self.config.verbose,
                shuffle_edges=algo_params.get("shuffle_edges", False),
                seed=algo_params.get("shuffle_seed", None),
                primitive_prefix=algo_params.get("primitive_prefix", ""),
                max_push_steps=max_push_steps,
            )

    @property
    def algorithm_name(self) -> str:
        """Return human-readable algorithm name."""
        return "Region Opening Planner"

    @property
    def algorithm_version(self) -> str:
        """Return algorithm version/variant identifier."""
        return "v1.0-reachability"

    def reset(self):
        """Reset internal algorithm state for new planning episode."""
        self.attempt_results = []
        self._last_explore_context = None

    def _debug(self, message: str):
        if getattr(self.config, "verbose", False):
            print(message)

    def _current_budget_stats(self) -> Dict[str, Any]:
        if self.push_budget is None:
            return {}
        return {
            "simulation_budget_limit": int(self.push_budget.limit),
            "simulation_budget_used": int(self.push_budget.used),
            "simulation_budget_remaining": int(self.push_budget.remaining),
        }

    def _consume_push_budget(self):
        if self.push_budget is None:
            return
        self.push_budget.consume_or_raise()

    @staticmethod
    def _normalize_boundary_object_ids(objects: Optional[Set[str]]) -> Set[str]:
        if not objects:
            return set()
        return {str(obj) for obj in objects}

    def _get_boundary_objects(
        self,
        edge_objects: Dict[str, Dict[str, Set[str]]],
        source_label: str,
        neighbour_label: str,
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        forward = edge_objects.get(source_label, {}).get(neighbour_label)
        reverse = edge_objects.get(neighbour_label, {}).get(source_label)

        if forward is not None and reverse is not None:
            forward_ids = self._normalize_boundary_object_ids(forward)
            reverse_ids = self._normalize_boundary_object_ids(reverse)
            if forward_ids != reverse_ids:
                return None, "boundary_object_map_inconsistent"
            return sorted(forward_ids), None

        if forward is not None:
            return sorted(self._normalize_boundary_object_ids(forward)), None
        if reverse is not None:
            return sorted(self._normalize_boundary_object_ids(reverse)), None
        return [], None

    def _build_target_summary(self, target_neighbor: Optional[str]) -> Optional[Dict[str, Any]]:
        if target_neighbor is None:
            return None

        context = self._last_explore_context or {}
        local_robot_label = context.get("local_robot_label")
        local_neighbors = sorted(context.get("local_neighbors", []))
        target_is_immediate = bool(context.get("target_is_immediate_neighbor", False))
        attempts = list(self.attempt_results or [])
        detail_reasons = sorted(
            {
                str(getattr(attempt, "failure_reason", "") or "unknown")
                for attempt in attempts
            }
        )
        success_found = any(getattr(attempt, "success", False) for attempt in attempts)

        if not attempts:
            failure_reason = "no_attempt_results"
        elif not target_is_immediate:
            failure_reason = "target_not_immediate_neighbor"
        elif any(reason == "boundary_object_map_inconsistent" for reason in detail_reasons):
            failure_reason = "boundary_object_map_inconsistent"
        elif any(reason == "already_accessible" for reason in detail_reasons):
            failure_reason = "already_accessible"
        elif any(reason == "no_blocking_objects" for reason in detail_reasons):
            failure_reason = "no_blocking_objects"
        elif success_found:
            failure_reason = "success"
        elif len(detail_reasons) == 1:
            failure_reason = detail_reasons[0]
        else:
            failure_reason = "mixed_failure_reasons"

        boundary_exhausted = (
            target_is_immediate
            and not success_found
            and bool(detail_reasons)
            and set(detail_reasons).issubset({"all_pushes_failed", "no_reachable_objects"})
        )

        return {
            "target_neighbor": target_neighbor,
            "local_robot_label": local_robot_label,
            "local_neighbors": local_neighbors,
            "target_is_immediate_neighbor": target_is_immediate,
            "failure_reason": failure_reason,
            "attempt_count": len(attempts),
            "detail_reasons": detail_reasons,
            "boundary_exhausted": boundary_exhausted,
        }

    def _focus_camera_on_object(self, object_id: str):
        """Focus camera on the specified object from above and render.

        Only does something if visualize_search is enabled.
        """
        if not self.visualize_search:
            return

        try:
            obs = self.env.get_observation()
            if object_id in obs:
                pos = obs[object_id]
                self.env.set_camera_lookat(pos[0], pos[1], 0.0)
                self.env.set_camera_position(
                    DEFAULT_CAMERA_DISTANCE,
                    DEFAULT_CAMERA_AZIMUTH,
                    DEFAULT_CAMERA_ELEVATION
                )
            self.env.render()

            if self.step_mode:
                input(f"[Step mode] Focused on {object_id}. Press Enter to continue...")
            elif self.search_delay > 0:
                time.sleep(self.search_delay)
        except Exception as e:
            # Don't let visualization errors break the search
            if self.config.verbose:
                self._debug(f"Camera focus error: {e}")

    def search(
        self,
        robot_goal: Tuple[float, float, float],
        target_neighbor: Optional[str] = None,
    ) -> PlannerResult:
        """Execute region opening planner (single-level exploration from initial state only).

        This method explores region openings from the initial state only:
        - Find all possible openings from the initial environment configuration
        - No multi-level exploration (no queueing of resulting states)

        Args:
            robot_goal: Target robot position (x, y, theta) - stored but not directly used
            target_neighbor: If set, only attempt to open path to this specific neighbor.
                           If None, attempt to open paths to ALL neighbors (default behavior).

        Returns:
            PlannerResult with all attempt results from initial state
        """
        start_time = time.time()
        self.attempt_results = []
        self._last_explore_context = None

        # Reset progress reporter for this search
        self._progress_total_primitives = 0
        self._progress_last_print_time = time.time()
        self._progress_last_print_count = 0

        # Reset rejection tally for this search
        self._rejection_stats = {}

        # Configure collision checking based on region_allow_collisions setting
        collision_checking_enabled = self.terminate_on_collision
        self.env.set_collision_checking(collision_checking_enabled)

        # Save baseline state
        baseline = self.env.get_full_state()
        try:
            if self.config.verbose:
                self._debug(f"\n{'='*60}")
                target_info = f" (target: {target_neighbor})" if target_neighbor else ""
                self._debug(f"Region Opening Planner - Single-Level Exploration{target_info}")
                self._debug(
                    f"Max chain depth: {self.max_chain_depth} | Collision checking: {'ON' if collision_checking_enabled else 'OFF'}"
                )
                self._debug(f"{'='*60}\n")

            # Explore from initial state only (Level 0)
            try:
                self.attempt_results = self._explore_from_state(
                    baseline,
                    level=0,
                    target_neighbor=target_neighbor,
                )
            except PushBudgetExceeded as exc:
                total_time = (time.time() - start_time) * 1000
                algorithm_stats: Dict[str, Any] = {
                    "attempt_results": self.attempt_results,
                    "all_solutions": [],
                    "successful_openings": 0,
                    "total_attempts": len(self.attempt_results),
                    "rejection_breakdown": dict(self._rejection_stats),
                    "total_primitives_attempted": self._progress_total_primitives,
                    "failure_kind": "simulation_budget_exhausted",
                }
                algorithm_stats.update(self._current_budget_stats())
                target_summary = self._build_target_summary(target_neighbor)
                if target_summary is not None:
                    algorithm_stats["target_summary"] = target_summary
                return PlannerResult(
                    success=False,
                    solution_found=False,
                    search_time_ms=total_time,
                    error_message=str(exc),
                    algorithm_stats=algorithm_stats,
                )

            if self.config.verbose:
                successful_attempts = sum(1 for a in self.attempt_results if a.success)
                self._debug(f"\n{'='*60}")
                self._debug(
                    f"Exploration Complete: {successful_attempts}/{len(self.attempt_results)} successful openings"
                )
                self._debug(f"{'='*60}\n")

            total_time = (time.time() - start_time) * 1000
            successful_attempts = sum(1 for a in self.attempt_results if a.success)

            action_sequence = []
            all_solutions = []
            for attempt in self.attempt_results:
                if not attempt.success:
                    continue

                attempt_actions = []
                if attempt.goal_chain:
                    for goal in attempt.goal_chain:
                        action = namo_rl.Action()
                        action.object_id = attempt.chosen_object_id
                        action.x = goal.x
                        action.y = goal.y
                        action.theta = goal.theta
                        action.edge_idx = getattr(goal, "edge_idx", -1)
                        action.depth = getattr(goal, "depth", -1)
                        attempt_actions.append(action)
                elif attempt.chosen_goal:
                    action = namo_rl.Action()
                    action.object_id = attempt.chosen_object_id
                    action.x = attempt.chosen_goal[0]
                    action.y = attempt.chosen_goal[1]
                    action.theta = attempt.chosen_goal[2]
                    attempt_actions.append(action)

                all_solutions.append(
                    {
                        "actions": attempt_actions,
                        "neighbor": attempt.neighbour_region_label,
                        "object": attempt.chosen_object_id,
                    }
                )
                if not action_sequence:
                    action_sequence = attempt_actions

            algorithm_stats: Dict[str, Any] = {
                "attempt_results": self.attempt_results,
                "all_solutions": all_solutions,
                "successful_openings": successful_attempts,
                "total_attempts": len(self.attempt_results),
                "rejection_breakdown": dict(self._rejection_stats),
                "total_primitives_attempted": self._progress_total_primitives,
            }
            algorithm_stats.update(self._current_budget_stats())
            target_summary = self._build_target_summary(target_neighbor)
            if target_summary is not None:
                algorithm_stats["target_summary"] = target_summary

            return PlannerResult(
                success=successful_attempts > 0,
                solution_found=successful_attempts > 0,
                action_sequence=action_sequence,
                solution_depth=len(action_sequence) if action_sequence else None,
                search_time_ms=total_time,
                algorithm_stats=algorithm_stats,
            )
        finally:
            self.env.set_full_state(baseline)

    def _explore_from_state(
        self,
        state: 'namo_rl.RLState',
        level: int = 0,
        target_neighbor: Optional[str] = None,
    ) -> List[AttemptResult]:
        """Explore region openings from a given state.

        This helper method:
        1. Sets environment to the given state
        2. Computes region connectivity snapshot
        3. Identifies robot region and neighbors
        4. Attempts to create openings to each neighbor (up to max_solutions_per_neighbor solutions)

        Args:
            state: Full environment state to explore from
            level: Exploration level (0 = initial state, 1+ = subsequent explorations)
            target_neighbor: If set, only attempt to open path to this specific neighbor.
                           If None, attempt to open paths to ALL neighbors.

        Returns:
            List of AttemptResults from exploring this state
        """
        print = self._debug

        # Set environment to exploration state
        self.env.set_full_state(state)

        # Get unified region connectivity + sampled goals from one wavefront source.
        from namo.planners import get_region_snapshot as _get_region_snapshot

        snapshot = _get_region_snapshot(
            self.env,
            goals_per_region=self.config.goals_per_region,
            goal_radius=self.region_goal_radius_m,
            local_info_only=True,
            seed=self.region_snapshot_seed,
            use_cpp_unified=self.use_cpp_unified_wavefront,
            use_xml_goal=True,
        )
        adjacency = snapshot["adjacency"]
        edge_objects = snapshot["edge_objects"]
        region_labels = snapshot["region_labels"]
        region_goals = snapshot["region_goals"]

        # Identify robot region
        robot_label = snapshot.get("robot_label") or find_robot_label(region_labels)
        if not robot_label:
            self._last_explore_context = {
                "local_robot_label": None,
                "local_neighbors": [],
                "target_neighbor": target_neighbor,
                "target_is_immediate_neighbor": False,
            }
            if self.config.verbose:
                print(f"  ⚠ Could not identify robot region")
            if target_neighbor is not None:
                return [AttemptResult(
                    success=False,
                    neighbour_region_label=target_neighbor,
                    error_message="Could not identify robot region",
                    failure_reason="missing_robot_region",
                    timing_ms=0.0,
                )]
            return []

        raw_neighbours = sorted(list(adjacency.get(robot_label, set())))
        neighbours = list(raw_neighbours)

        # Auto-target the XML goal region when the planner is configured for it.
        # Done here (vs. in search()) so it runs after the snapshot is computed,
        # avoiding a duplicate snapshot call.
        #
        # The snapshot uses these region-label conventions:
        #   "robot"       — region containing the robot only
        #   "goal"        — region containing the XML goal only
        #   "robot_goal"  — robot's region also contains the goal (no opening needed)
        #   "region_N"    — any other region
        if target_neighbor is None and getattr(self, "target_goal_region", False):
            robot_already_at_goal = (
                robot_label == "robot_goal" or robot_label == "goal"
            )
            if robot_already_at_goal:
                if self.config.verbose:
                    print(f"  ✓ target_goal_region: robot region '{robot_label}' already contains the goal, no episodes to record")
                self._last_explore_context = {
                    "local_robot_label": robot_label,
                    "local_neighbors": list(raw_neighbours),
                    "target_neighbor": "goal",
                    "target_is_immediate_neighbor": False,
                }
                return []
            elif "goal" in region_labels.values():
                target_neighbor = "goal"
                if self.config.verbose:
                    print("  ▶ target_goal_region: auto-targeting 'goal' region")
            else:
                # Goal region exists in the env but is not wavefront-reachable from
                # the robot — i.e. completely walled off + blocked by obstacles such
                # that no immediate-neighbour transition leads toward it. Record one
                # failure AttemptResult so phase-2 (deeper chain) picks it up.
                msg = (
                    f"target_goal_region: no 'goal' label in snapshot "
                    f"(labels={list(region_labels.values())}) — goal region not "
                    f"wavefront-reachable from robot, phase-2 candidate"
                )
                if self.config.verbose:
                    print(f"  ⚠ {msg}")
                self._last_explore_context = {
                    "local_robot_label": robot_label,
                    "local_neighbors": list(raw_neighbours),
                    "target_neighbor": "goal",
                    "target_is_immediate_neighbor": False,
                }
                return [AttemptResult(
                    success=False,
                    neighbour_region_label="goal",
                    error_message=msg,
                    failure_reason="goal_region_not_in_snapshot",
                    timing_ms=0.0,
                )]

        # Apply region skip filter (blacklist)
        # Skip entire regions that have empty object lists in region_object_skip
        if self.region_object_skip:
            # Regions with empty list = skip entire region
            regions_to_skip = [r for r, objs in self.region_object_skip.items() if not objs]
            skipped = [n for n in neighbours if n in regions_to_skip]
            neighbours = [n for n in neighbours if n not in regions_to_skip]
            # Print skipped regions
            if self.config.verbose and skipped:
                for region in skipped:
                    print(f"   ⏭ Neighbor '{region}': SKIPPED (from manifest)")
            # Regions with specific objects will be filtered later in _attempt_opening_to_neighbour

        self._last_explore_context = {
            "local_robot_label": robot_label,
            "local_neighbors": list(raw_neighbours),
            "target_neighbor": target_neighbor,
            "target_is_immediate_neighbor": target_neighbor in raw_neighbours if target_neighbor is not None else None,
        }

        # Filter to target_neighbor if specified (for FullNAMOPlanner)
        if target_neighbor is not None:
            if target_neighbor in neighbours:
                neighbours = [target_neighbor]
                if self.config.verbose:
                    print(f"  Targeting specific neighbor: {target_neighbor}")
            else:
                if self.config.verbose:
                    print(f"  ⚠ Target neighbor '{target_neighbor}' not in immediate neighbors: {neighbours}")
                return [AttemptResult(
                    success=False,
                    neighbour_region_label=target_neighbor,
                    error_message=f"Target neighbor '{target_neighbor}' is not an immediate neighbor",
                    failure_reason="target_not_immediate_neighbor",
                    timing_ms=0.0,
                )]

        if self.config.verbose:
            # Print region snapshot details
            total_regions = len(region_labels)
            total_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
            has_goal_region = "goal" in region_labels.values()
            goal_info = " (includes GOAL region)" if has_goal_region else ""
            print(f"  Region snapshot: robot={robot_label} | regions={total_regions}{goal_info} | edges={total_edges} | neighbors={len(neighbours)}")
            print(f"  All regions: {list(region_labels.values())}")
            print(f"  Adjacency: {dict(adjacency)}")
            print(f"  Edge objects: {dict(edge_objects)}")
            print(f"  Neighbors to explore: {neighbours}")

        # Collect attempts from this state
        state_attempts = []

        # Process each neighbour (up to max_solutions=2 per neighbor)
        for neighbour_label in neighbours:
            # Restore state before trying this neighbour
            self.env.set_full_state(state)

            print(f"\n🌟 [_explore_from_state] Attempting to open path to neighbour: '{neighbour_label}'")

            # Attempt to open path to this neighbour
            attempts = self._attempt_opening_to_neighbour(
                robot_label,
                neighbour_label,
                adjacency,
                edge_objects,
                region_goals,
                max_solutions=self.max_solutions_per_neighbor,
                exploration_state=state,
                exploration_level=level
            )

            # Collect results (simplified - detailed info printed in _attempt_opening_to_neighbour)
            if isinstance(attempts, list):
                state_attempts.extend(attempts)
            else:
                state_attempts.append(attempts)

            if not self.exhaustive_mode and self.stop_after_first_success:
                # Stop exploring other neighbours once any successful opening exists.
                if any(a.success for a in (attempts if isinstance(attempts, list) else [attempts])):
                    if self.config.verbose:
                        print(f"  🛑 Stopping neighbour exploration after first success (neighbour='{neighbour_label}')")
                    break

        return state_attempts

    def _attempt_opening_to_neighbour(
        self,
        robot_label: str,
        neighbour_label: str,
        adjacency: Dict[str, Set[str]],
        edge_objects: Dict[str, Dict[str, Set[str]]],
        region_goals: Dict[str, Any],
        max_solutions: int = 2,
        exploration_state: Optional['namo_rl.RLState'] = None,
        exploration_level: int = 0
    ) -> List[AttemptResult]:
        """Attempt to open a path to a specific neighbour region.

        Args:
            robot_label: Robot's current region
            neighbour_label: Target neighbour region
            adjacency: Region adjacency graph
            edge_objects: Blocking objects between regions
            region_goals: Sampled goals for each region (for reachability validation)
            max_solutions: Maximum number of solutions to find for this neighbour
            exploration_state: State we're exploring from (for visualization context)
            exploration_level: Exploration level (0 = initial, 1+ = subsequent)

        Returns:
            List of AttemptResults (one per successful push variation, up to max_solutions)
        """
        attempt_start = time.time()

        print = self._debug
        conn_before = {"adjacency": dict(adjacency), "robot_label": robot_label}
        neighbour_push_counter = {"count": 0}

        # Ensure environment is in correct state before pre-check
        self.env.set_full_state(exploration_state)

        # For visualization/debugging: set a stable "target" goal marker for this neighbour
        # (rather than leaving whatever goal was used for the previous neighbour).
        try:
            seed_bundle = region_goals.get(neighbour_label)
            if seed_bundle and getattr(seed_bundle, "goals", None):
                seed_goal = seed_bundle.goals[0]
                self.env.set_robot_goal(seed_goal.x, seed_goal.y, seed_goal.theta)
        except Exception:
            pass

        # Pre-check: Is this neighbor already accessible?
        is_already_accessible, reachable_count_before, precheck_region_goal, all_region_goals = self._validate_opening(
            neighbour_label,
            region_goals
        )

        if is_already_accessible:
            # Neighbor is already accessible - no need to push anything!
            if self.config.verbose:
                total_goals = len(region_goals[neighbour_label].goals) if neighbour_label in region_goals else 0
                region_type = "(GOAL REGION)" if neighbour_label == "goal" else ""
                print(f"    ⊙ '{neighbour_label}' already accessible {region_type} ({reachable_count_before}/{total_goals} reachable) - skipping")

            already_result = AttemptResult(
                success=True,
                neighbour_region_label=neighbour_label,
                chosen_object_id=None,
                chosen_goal=None,
                goal_chain=[],
                chain_depth=0,
                validation_method="already_accessible",
                connectivity_before=conn_before,
                connectivity_after=conn_before,
                region_goal_used=precheck_region_goal,
                region_goals_sampled=all_region_goals,
                actions_executed=[],
                state_observations=[],
                post_action_state_observations=[],
                reachable_objects_before_action=None,
                reachable_objects_after_action=None,
                exploration_state=exploration_state,
                resulting_state=exploration_state,
                exploration_level=exploration_level,
                timing_ms=(time.time() - attempt_start) * 1000,
                total_cost=0,
                skill_calls_before_success=0,
                solutions_found_for_neighbour=0,
                solutions_cap_for_neighbour=self.max_solutions_per_neighbor,
                failure_reason="already_accessible",
            )
            return [already_result]

        # Get candidate objects blocking the boundary between the robot and neighbour region.
        candidates, boundary_error = self._get_boundary_objects(edge_objects, robot_label, neighbour_label)
        if boundary_error is not None:
            if self.config.verbose:
                print(f"    ✗ '{neighbour_label}' - {boundary_error}")
            return [AttemptResult(
                success=False,
                neighbour_region_label=neighbour_label,
                error_message="Boundary object map is inconsistent across directions",
                timing_ms=(time.time() - attempt_start) * 1000,
                failure_reason=boundary_error,
                candidate_objects_count=0,
            )]

        # Filter out objects specified in region_object_skip for this neighbour
        if self.region_object_skip and neighbour_label in self.region_object_skip:
            objects_to_skip = set(self.region_object_skip[neighbour_label])
            if objects_to_skip:  # Only filter if there are specific objects (not empty list)
                skipped_objects = [c for c in candidates if c in objects_to_skip]
                candidates = [c for c in candidates if c not in objects_to_skip]
                # Print skipped objects
                if self.config.verbose and skipped_objects:
                    for obj in skipped_objects:
                        print(f"   ⏭ Neighbor '{neighbour_label}' Object '{obj}': SKIPPED (from manifest)")

        if not candidates:
            if self.config.verbose:
                print(f"    ✗ '{neighbour_label}' - no blocking objects found")
            return [AttemptResult(
                success=False,
                neighbour_region_label=neighbour_label,
                error_message="No blocking objects found",
                timing_ms=(time.time() - attempt_start) * 1000,
                region_goal_used=precheck_region_goal,
                region_goals_sampled=all_region_goals,
                failure_reason="no_blocking_objects",
                candidate_objects_count=0,
            )]

        # Intersect with reachable objects
        reachable = set(self.env.get_reachable_objects())
        original_candidates_count = len(candidates)
        candidates = [obj for obj in candidates if obj in reachable]

        if not candidates:
            if self.config.verbose:
                print(f"    ✗ '{neighbour_label}' - no reachable blocking objects (had {original_candidates_count} blocking objects)")
            return [AttemptResult(
                success=False,
                neighbour_region_label=neighbour_label,
                error_message=f"No reachable blocking objects (had {original_candidates_count} blocking)",
                timing_ms=(time.time() - attempt_start) * 1000,
                region_goal_used=precheck_region_goal,
                region_goals_sampled=all_region_goals,
                failure_reason="no_reachable_objects",
                candidate_objects_count=original_candidates_count,
            )]

        # Print what we're attempting
        if self.config.verbose:
            total_goals = len(region_goals[neighbour_label].goals) if neighbour_label in region_goals else 0
            print(f"    → '{neighbour_label}' ({reachable_count_before}/{total_goals} reachable) - trying {len(candidates)} objects: {candidates}")

        # Collect attempts from candidate objects (per-object limits applied)
        all_goal_attempts = []
        total_solutions_collected = 0

        # Track ML goal stats across all objects for failure analysis
        total_ml_goals_generated = 0
        total_ml_goals_aligned = 0
        total_reachable_edges = 0
        # Detailed info for analysis (accumulated across all objects)
        all_aligned_primitives = []
        all_ml_goals_raw = []
        all_reachable_edges = set()

        # Try each candidate object with BFS search (already filtered for reachability)
        # NOTE: We try ALL objects (no early termination) to record per-object triplets for eval
        timed_out = False
        for obj_idx, object_id in enumerate(candidates, 1):
            if not self.exhaustive_mode and self.stop_after_max_solutions and total_solutions_collected >= max_solutions:
                if self.config.verbose:
                    print(
                        f"    🛑 Collected {total_solutions_collected}/{max_solutions} solutions for '{neighbour_label}', "
                        f"stopping before trying remaining objects"
                    )
                break

            # Check timeout before trying next object
            if self.timeout_per_neighbour_sec is not None:
                elapsed_sec = time.time() - attempt_start
                if elapsed_sec >= self.timeout_per_neighbour_sec:
                    if self.config.verbose:
                        print(f"    ⏱ TIMEOUT after {elapsed_sec:.1f}s (limit: {self.timeout_per_neighbour_sec}s) - stopping search for '{neighbour_label}'")
                    timed_out = True
                    break

            # CRITICAL: Reset to exploration_state before trying each object
            # This ensures each object is tried from the same starting configuration
            self.env.set_full_state(exploration_state)

            print(f"  🎯 [_attempt_opening_to_neighbour] Trying object {obj_idx}/{len(candidates)}: {object_id} for neighbour '{neighbour_label}'")

            # Reset diffusion call counter so per-object stats reflect only this object's search.
            if hasattr(self.goal_strategy, "reset_diffusion_call_counter"):
                try:
                    self.goal_strategy.reset_diffusion_call_counter()
                except Exception:
                    pass
            # Reset optional goal-strategy profiler so per-object stats reflect only this object's search.
            if hasattr(self.goal_strategy, "reset_profile"):
                try:
                    self.goal_strategy.reset_profile()
                except Exception:
                    pass
            self._reset_runtime_timing_stats()

            # BFS search with chaining (or ML-driven async if enabled)
            object_attempt_start = time.time()
            pushes_before_object = neighbour_push_counter.get("count", 0)

            # Use per-object solution limit (not global remaining) unless explicitly stopping early.
            if not self.exhaustive_mode and self.stop_after_max_solutions:
                remaining = max(0, max_solutions - total_solutions_collected)
                max_solutions_for_object = max(1, remaining) if remaining > 0 else 1
            else:
                max_solutions_for_object = self.max_solutions_per_neighbor

            if self._use_ml_driven_async:
                # Use ML-driven async search
                successful_goals, min_depth = self._search_with_ml_driven_async(
                    object_id,
                    exploration_state,
                    neighbour_label,
                    region_goals,
                    max_solutions_to_collect=max_solutions_for_object,
                    push_counter=neighbour_push_counter,
                )
                # Async search doesn't have phase tracking or trial log yet
                phase_push_counts = None
                solved_in_phase = ""
                search_any_wall_collision = False
                search_unique_movable_collision_count = 0
                object_trial_log = []
                object_reachability_log = []
            else:
                # Use standard BFS search. With sampled collection (sample_k>0), restart with FRESH
                # random subsets up to sample_restarts times, ONLY while no chain has been found;
                # merge trial logs across attempts (union of tried cells = the training loss mask).
                n_attempts = self.sample_restarts if self.sample_k > 0 else 1
                object_trial_log = []
                object_reachability_log = []
                for _attempt in range(max(1, n_attempts)):
                    successful_goals, min_depth, phase_push_counts, solved_in_phase, search_any_wall_collision, search_unique_movable_collision_count, attempt_trial_log, attempt_reach_log = self._search_with_chaining_bfs(
                        object_id,
                        exploration_state,
                        neighbour_label,
                        region_goals,
                        max_solutions_to_collect=max_solutions_for_object,
                        push_counter=neighbour_push_counter,
                    )
                    object_trial_log.extend(attempt_trial_log or [])
                    object_reachability_log.extend(attempt_reach_log or [])
                    if successful_goals:
                        break

            pushes_for_this_object = neighbour_push_counter.get("count", 0) - pushes_before_object
            obj_goal_strategy_profile = None
            if hasattr(self.goal_strategy, "get_profile"):
                try:
                    obj_goal_strategy_profile = self.goal_strategy.get_profile()
                except Exception:
                    obj_goal_strategy_profile = None
            obj_runtime_timing = self._get_runtime_timing_summary()

            obj_ml_mask_vote_attach_calls = 0
            obj_ml_mask_vote_attach_ms_total = 0.0
            obj_ml_mask_vote_attach_ms_avg = 0.0
            if obj_goal_strategy_profile:
                try:
                    obj_ml_mask_vote_attach_calls = int(
                        obj_goal_strategy_profile.get("ml_mask_vote_attach_calls", 0) or 0
                    )
                except Exception:
                    obj_ml_mask_vote_attach_calls = 0
                try:
                    obj_ml_mask_vote_attach_ms_total = float(
                        obj_goal_strategy_profile.get("ml_mask_vote_attach_ms_total", 0.0) or 0.0
                    )
                except Exception:
                    obj_ml_mask_vote_attach_ms_total = 0.0
                if obj_ml_mask_vote_attach_calls > 0:
                    obj_ml_mask_vote_attach_ms_avg = (
                        obj_ml_mask_vote_attach_ms_total / obj_ml_mask_vote_attach_calls
                    )

            # Get per-object ML goal stats from goal_strategy (if it supports get_last_goal_stats)
            # Capture per-object values BEFORE accumulating into totals
            obj_ml_goals = 0
            obj_ml_aligned = 0
            obj_ml_diffusion_calls = 0
            obj_reachable_edges = 0
            obj_aligned_primitives = []
            obj_ml_goals_raw = []
            obj_reachable_edges_set = set()

            if hasattr(self.goal_strategy, 'get_last_goal_stats'):
                stats = self.goal_strategy.get_last_goal_stats()
                obj_ml_goals = stats.get('ml_goals_generated', 0)
                obj_ml_aligned = stats.get('ml_goals_aligned', 0)
                obj_ml_diffusion_calls = stats.get('ml_diffusion_calls', 0)
                obj_reachable_edges = stats.get('reachable_edges_count', 0)
                obj_aligned_primitives = stats.get('aligned_primitives', [])
                obj_ml_goals_raw = stats.get('ml_goals_raw', [])
                obj_reachable_edges_set = set(stats.get('reachable_edges', []))

                # Accumulate into totals (for neighbor-level stats if needed)
                total_ml_goals_generated += obj_ml_goals
                total_ml_goals_aligned += obj_ml_aligned
                total_reachable_edges += obj_reachable_edges
                # Accumulate detailed info (tag with object_id for multi-object analysis)
                for p in obj_aligned_primitives:
                    p_with_obj = dict(p)
                    p_with_obj['object_id'] = object_id
                    all_aligned_primitives.append(p_with_obj)
                for g in obj_ml_goals_raw:
                    g_with_obj = dict(g)
                    g_with_obj['object_id'] = object_id
                    all_ml_goals_raw.append(g_with_obj)
                all_reachable_edges.update(obj_reachable_edges_set)

            if successful_goals:
                if self.config.verbose:
                    print(f"      ✓ {object_id}: Found {len(successful_goals[:max_solutions])} solutions (depth={min_depth})")

                # Create AttemptResults directly from successful goal chains
                # State observations were already captured during BFS search

                per_object_limit = max_solutions
                if not self.exhaustive_mode and self.stop_after_max_solutions:
                    per_object_limit = max(0, max_solutions - total_solutions_collected)
                for goal_idx, (goal_chain, state_obs, post_state_obs, resulting_state, region_goal_used, region_goals_sampled, reachable_before, reachable_after, total_cost, skill_calls_before_success, success_timestamp, any_wall_collision, unique_movable_collision_count) in enumerate(successful_goals[:per_object_limit]):

                    per_goal_timing_ms = max(0.0, (success_timestamp - object_attempt_start) * 1000.0)

                    # Create AttemptResult
                    if len(goal_chain) == 1:
                        # Single push - also set goal_chain to preserve edge_idx/depth
                        goal = goal_chain[0]
                        total_solutions_collected += 1
                        all_goal_attempts.append(AttemptResult(
                            success=True,
                            neighbour_region_label=neighbour_label,
                            chosen_object_id=object_id,
                            chosen_goal=(goal.x, goal.y, goal.theta),
                            goal_chain=goal_chain,  # Preserve Goal objects with edge_idx/depth
                            chain_depth=1,
                            validation_method="reachability_validated",
                            connectivity_before=conn_before,
                            connectivity_after=None,
                            region_goal_used=region_goal_used,
                            region_goals_sampled=region_goals_sampled,
                            actions_executed=[],
                            state_observations=state_obs,
                            post_action_state_observations=post_state_obs,
                            reachable_objects_before_action=reachable_before,
                            reachable_objects_after_action=reachable_after,
                            exploration_state=exploration_state,
                            resulting_state=resulting_state,
                            exploration_level=exploration_level,
                            timing_ms=per_goal_timing_ms,
                            goal_strategy_profile=obj_goal_strategy_profile,
                            total_cost=total_cost,
                            skill_calls_before_success=skill_calls_before_success,
                            solutions_found_for_neighbour=total_solutions_collected,
                            solutions_cap_for_neighbour=self.max_solutions_per_neighbor,
                            pushes_total_for_neighbour=pushes_for_this_object,  # Per-object pushes
                            failure_reason="success",
                            candidate_objects_count=len(candidates),
                            ml_goals_generated=obj_ml_goals,  # Per-object
                            ml_goals_aligned=obj_ml_aligned,  # Per-object
                            ml_diffusion_calls=obj_ml_diffusion_calls,  # Per-object
                            ml_mask_vote_attach_calls=obj_ml_mask_vote_attach_calls,
                            ml_mask_vote_attach_ms_total=obj_ml_mask_vote_attach_ms_total,
                            ml_mask_vote_attach_ms_avg=obj_ml_mask_vote_attach_ms_avg,
                            reachable_edges_count=obj_reachable_edges,  # Per-object
                            aligned_primitives=obj_aligned_primitives if obj_aligned_primitives else None,
                            ml_goals_raw=obj_ml_goals_raw if obj_ml_goals_raw else None,
                            reachable_edges=sorted(list(obj_reachable_edges_set)) if obj_reachable_edges_set else None,
                            primitive_ranking_calls=int(obj_runtime_timing.get("primitive_ranking_calls", 0)),
                            primitive_ranking_ms_total=float(obj_runtime_timing.get("primitive_ranking_ms_total", 0.0)),
                            primitive_ranking_ms_avg=float(obj_runtime_timing.get("primitive_ranking_ms_avg", 0.0)),
                            primitive_ranking_candidates_total=int(obj_runtime_timing.get("primitive_ranking_candidates_total", 0)),
                            primitive_ranking_candidates_avg=float(obj_runtime_timing.get("primitive_ranking_candidates_avg", 0.0)),
                            push_exec_count=int(obj_runtime_timing.get("push_exec_count", 0)),
                            push_exec_ms_total=float(obj_runtime_timing.get("push_exec_ms_total", 0.0)),
                            push_exec_ms_avg=float(obj_runtime_timing.get("push_exec_ms_avg", 0.0)),
                            push_exec_ms_by_depth=obj_runtime_timing.get("push_exec_ms_by_depth", {}),
                            goal_generation_calls=int(obj_runtime_timing.get("goal_generation_calls", 0)),
                            goal_generation_ms_total=float(obj_runtime_timing.get("goal_generation_ms_total", 0.0)),
                            goal_generation_ms_avg=float(obj_runtime_timing.get("goal_generation_ms_avg", 0.0)),
                            opening_validation_calls=int(obj_runtime_timing.get("opening_validation_calls", 0)),
                            opening_validation_ms_total=float(obj_runtime_timing.get("opening_validation_ms_total", 0.0)),
                            opening_validation_ms_avg=float(obj_runtime_timing.get("opening_validation_ms_avg", 0.0)),
                            opening_validation_goal_checks_total=int(obj_runtime_timing.get("opening_validation_goal_checks_total", 0)),
                            opening_validation_goal_checks_avg_per_call=float(obj_runtime_timing.get("opening_validation_goal_checks_avg_per_call", 0.0)),
                            opening_validation_reachability_calls=int(obj_runtime_timing.get("opening_validation_reachability_calls", 0)),
                            opening_validation_reachability_ms_total=float(obj_runtime_timing.get("opening_validation_reachability_ms_total", 0.0)),
                            opening_validation_reachability_ms_avg=float(obj_runtime_timing.get("opening_validation_reachability_ms_avg", 0.0)),
                            chain_observation_replay_calls=int(obj_runtime_timing.get("chain_observation_replay_calls", 0)),
                            chain_observation_replay_ms_total=float(obj_runtime_timing.get("chain_observation_replay_ms_total", 0.0)),
                            chain_observation_replay_ms_avg=float(obj_runtime_timing.get("chain_observation_replay_ms_avg", 0.0)),
                            any_wall_collision=any_wall_collision,
                            unique_movable_collision_count=unique_movable_collision_count,
                            phase_push_counts=phase_push_counts,
                            solved_in_phase=solved_in_phase,
                            primitive_trial_log=object_trial_log if self.exhaustive_mode else None,
                            reachability_log=object_reachability_log if self.exhaustive_mode else None,
                        ))
                        # Verbose: print running count of solutions for this object
                        if self.config.verbose:
                            print(f"        → Object {object_id} solutions: {goal_idx + 1}/{per_object_limit}")

                        if not self.exhaustive_mode and self.stop_after_max_solutions and total_solutions_collected >= max_solutions:
                            if self.config.verbose:
                                print(
                                    f"        🛑 Collected {total_solutions_collected}/{max_solutions} solutions for '{neighbour_label}', "
                                    f"stopping search for this neighbour"
                                )
                            break
                    else:
                        # Multi-push chain
                        total_solutions_collected += 1
                        all_goal_attempts.append(AttemptResult(
                            success=True,
                            neighbour_region_label=neighbour_label,
                            chosen_object_id=object_id,
                            chosen_goal=None,
                            goal_chain=goal_chain,
                            chain_depth=len(goal_chain),
                            validation_method="reachability_validated",
                            connectivity_before=conn_before,
                            connectivity_after=None,
                            region_goal_used=region_goal_used,
                            region_goals_sampled=region_goals_sampled,
                            actions_executed=[],
                            state_observations=state_obs,
                            post_action_state_observations=post_state_obs,
                            reachable_objects_before_action=reachable_before,
                            reachable_objects_after_action=reachable_after,
                            exploration_state=exploration_state,
                            resulting_state=resulting_state,
                            exploration_level=exploration_level,
                            timing_ms=per_goal_timing_ms,
                            goal_strategy_profile=obj_goal_strategy_profile,
                            total_cost=total_cost,
                            skill_calls_before_success=skill_calls_before_success,
                            solutions_found_for_neighbour=total_solutions_collected,
                            solutions_cap_for_neighbour=self.max_solutions_per_neighbor,
                            pushes_total_for_neighbour=pushes_for_this_object,  # Per-object pushes
                            failure_reason="success",
                            candidate_objects_count=len(candidates),
                            ml_goals_generated=obj_ml_goals,  # Per-object
                            ml_goals_aligned=obj_ml_aligned,  # Per-object
                            ml_diffusion_calls=obj_ml_diffusion_calls,  # Per-object
                            ml_mask_vote_attach_calls=obj_ml_mask_vote_attach_calls,
                            ml_mask_vote_attach_ms_total=obj_ml_mask_vote_attach_ms_total,
                            ml_mask_vote_attach_ms_avg=obj_ml_mask_vote_attach_ms_avg,
                            reachable_edges_count=obj_reachable_edges,  # Per-object
                            aligned_primitives=obj_aligned_primitives if obj_aligned_primitives else None,
                            ml_goals_raw=obj_ml_goals_raw if obj_ml_goals_raw else None,
                            reachable_edges=sorted(list(obj_reachable_edges_set)) if obj_reachable_edges_set else None,
                            primitive_ranking_calls=int(obj_runtime_timing.get("primitive_ranking_calls", 0)),
                            primitive_ranking_ms_total=float(obj_runtime_timing.get("primitive_ranking_ms_total", 0.0)),
                            primitive_ranking_ms_avg=float(obj_runtime_timing.get("primitive_ranking_ms_avg", 0.0)),
                            primitive_ranking_candidates_total=int(obj_runtime_timing.get("primitive_ranking_candidates_total", 0)),
                            primitive_ranking_candidates_avg=float(obj_runtime_timing.get("primitive_ranking_candidates_avg", 0.0)),
                            push_exec_count=int(obj_runtime_timing.get("push_exec_count", 0)),
                            push_exec_ms_total=float(obj_runtime_timing.get("push_exec_ms_total", 0.0)),
                            push_exec_ms_avg=float(obj_runtime_timing.get("push_exec_ms_avg", 0.0)),
                            push_exec_ms_by_depth=obj_runtime_timing.get("push_exec_ms_by_depth", {}),
                            goal_generation_calls=int(obj_runtime_timing.get("goal_generation_calls", 0)),
                            goal_generation_ms_total=float(obj_runtime_timing.get("goal_generation_ms_total", 0.0)),
                            goal_generation_ms_avg=float(obj_runtime_timing.get("goal_generation_ms_avg", 0.0)),
                            opening_validation_calls=int(obj_runtime_timing.get("opening_validation_calls", 0)),
                            opening_validation_ms_total=float(obj_runtime_timing.get("opening_validation_ms_total", 0.0)),
                            opening_validation_ms_avg=float(obj_runtime_timing.get("opening_validation_ms_avg", 0.0)),
                            opening_validation_goal_checks_total=int(obj_runtime_timing.get("opening_validation_goal_checks_total", 0)),
                            opening_validation_goal_checks_avg_per_call=float(obj_runtime_timing.get("opening_validation_goal_checks_avg_per_call", 0.0)),
                            opening_validation_reachability_calls=int(obj_runtime_timing.get("opening_validation_reachability_calls", 0)),
                            opening_validation_reachability_ms_total=float(obj_runtime_timing.get("opening_validation_reachability_ms_total", 0.0)),
                            opening_validation_reachability_ms_avg=float(obj_runtime_timing.get("opening_validation_reachability_ms_avg", 0.0)),
                            chain_observation_replay_calls=int(obj_runtime_timing.get("chain_observation_replay_calls", 0)),
                            chain_observation_replay_ms_total=float(obj_runtime_timing.get("chain_observation_replay_ms_total", 0.0)),
                            chain_observation_replay_ms_avg=float(obj_runtime_timing.get("chain_observation_replay_ms_avg", 0.0)),
                            any_wall_collision=any_wall_collision,
                            unique_movable_collision_count=unique_movable_collision_count,
                            phase_push_counts=phase_push_counts,
                            solved_in_phase=solved_in_phase,
                            primitive_trial_log=object_trial_log if self.exhaustive_mode else None,
                            reachability_log=object_reachability_log if self.exhaustive_mode else None,
                        ))
                        # Verbose: print running count of solutions for this object
                        if self.config.verbose:
                            print(f"        → Object {object_id} solutions: {goal_idx + 1}/{per_object_limit}")

                        if not self.exhaustive_mode and self.stop_after_max_solutions and total_solutions_collected >= max_solutions:
                            if self.config.verbose:
                                print(
                                    f"        🛑 Collected {total_solutions_collected}/{max_solutions} solutions for '{neighbour_label}', "
                                    f"stopping search for this neighbour"
                                )
                            break
            else:
                # Record per-object failure for eval triplet tracking
                # This ensures we can measure success rates per (env, region, object) triplet
                object_timing_ms = (time.time() - object_attempt_start) * 1000.0

                # Determine per-object failure reason
                if hasattr(self.goal_strategy, 'get_last_goal_stats'):
                    obj_stats = self.goal_strategy.get_last_goal_stats()
                    obj_ml_goals = obj_stats.get('ml_goals_generated', 0)
                    obj_ml_aligned = obj_stats.get('ml_goals_aligned', 0)
                    obj_ml_diffusion_calls = obj_stats.get('ml_diffusion_calls', 0)
                    obj_reachable_edges = obj_stats.get('reachable_edges_count', 0)
                else:
                    obj_ml_goals = 0
                    obj_ml_aligned = 0
                    obj_ml_diffusion_calls = 0
                    obj_reachable_edges = 0

                if pushes_for_this_object > 0:
                    obj_failure_reason = "all_pushes_failed"
                elif obj_ml_goals == 0:
                    obj_failure_reason = "ml_no_goals_extracted"
                elif obj_ml_aligned == 0:
                    obj_failure_reason = "ml_goals_not_aligned"
                elif obj_reachable_edges == 0:
                    obj_failure_reason = "no_reachable_edges"
                else:
                    obj_failure_reason = "no_valid_goals"

                # Build a one-line human-readable summary surfacing the most actionable
                # numbers for post-hoc analysis (PKL inspection / failure mining).
                obj_error_msg = (
                    f"{obj_failure_reason}: object={object_id}, "
                    f"neighbour={neighbour_label}, "
                    f"pushes={pushes_for_this_object}, "
                    f"candidates={len(candidates)}, "
                    f"ml_goals_generated={obj_ml_goals}, "
                    f"ml_goals_aligned={obj_ml_aligned}, "
                    f"reachable_edges={obj_reachable_edges}, "
                    f"timing_ms={object_timing_ms:.0f}"
                )

                if self.config.verbose:
                    print(f"      ✗ {object_id}: No solutions found ({obj_failure_reason}, {pushes_for_this_object} pushes)")

                all_goal_attempts.append(AttemptResult(
                    success=False,
                    neighbour_region_label=neighbour_label,
                    chosen_object_id=object_id,  # KEY: Include object_id for triplet tracking
                    chosen_goal=None,
                    error_message=obj_error_msg,
                    validation_method="failed",
                    connectivity_before=conn_before,
                    connectivity_after=None,
                    region_goal_used=precheck_region_goal,
                    region_goals_sampled=all_region_goals,
                    exploration_state=exploration_state,
                    exploration_level=exploration_level,
                    timing_ms=object_timing_ms,
                    goal_strategy_profile=obj_goal_strategy_profile,
                    solutions_found_for_neighbour=0,
                    solutions_cap_for_neighbour=self.max_solutions_per_neighbor,
                    pushes_total_for_neighbour=pushes_for_this_object,
                    failure_reason=obj_failure_reason,
                    candidate_objects_count=len(candidates),
                    ml_goals_generated=obj_ml_goals,
                    ml_goals_aligned=obj_ml_aligned,
                    ml_diffusion_calls=obj_ml_diffusion_calls,
                    ml_mask_vote_attach_calls=obj_ml_mask_vote_attach_calls,
                    ml_mask_vote_attach_ms_total=obj_ml_mask_vote_attach_ms_total,
                    ml_mask_vote_attach_ms_avg=obj_ml_mask_vote_attach_ms_avg,
                    reachable_edges_count=obj_reachable_edges,
                    primitive_ranking_calls=int(obj_runtime_timing.get("primitive_ranking_calls", 0)),
                    primitive_ranking_ms_total=float(obj_runtime_timing.get("primitive_ranking_ms_total", 0.0)),
                    primitive_ranking_ms_avg=float(obj_runtime_timing.get("primitive_ranking_ms_avg", 0.0)),
                    primitive_ranking_candidates_total=int(obj_runtime_timing.get("primitive_ranking_candidates_total", 0)),
                    primitive_ranking_candidates_avg=float(obj_runtime_timing.get("primitive_ranking_candidates_avg", 0.0)),
                    push_exec_count=int(obj_runtime_timing.get("push_exec_count", 0)),
                    push_exec_ms_total=float(obj_runtime_timing.get("push_exec_ms_total", 0.0)),
                    push_exec_ms_avg=float(obj_runtime_timing.get("push_exec_ms_avg", 0.0)),
                    push_exec_ms_by_depth=obj_runtime_timing.get("push_exec_ms_by_depth", {}),
                    goal_generation_calls=int(obj_runtime_timing.get("goal_generation_calls", 0)),
                    goal_generation_ms_total=float(obj_runtime_timing.get("goal_generation_ms_total", 0.0)),
                    goal_generation_ms_avg=float(obj_runtime_timing.get("goal_generation_ms_avg", 0.0)),
                    opening_validation_calls=int(obj_runtime_timing.get("opening_validation_calls", 0)),
                    opening_validation_ms_total=float(obj_runtime_timing.get("opening_validation_ms_total", 0.0)),
                    opening_validation_ms_avg=float(obj_runtime_timing.get("opening_validation_ms_avg", 0.0)),
                    opening_validation_goal_checks_total=int(obj_runtime_timing.get("opening_validation_goal_checks_total", 0)),
                    opening_validation_goal_checks_avg_per_call=float(obj_runtime_timing.get("opening_validation_goal_checks_avg_per_call", 0.0)),
                    opening_validation_reachability_calls=int(obj_runtime_timing.get("opening_validation_reachability_calls", 0)),
                    opening_validation_reachability_ms_total=float(obj_runtime_timing.get("opening_validation_reachability_ms_total", 0.0)),
                    opening_validation_reachability_ms_avg=float(obj_runtime_timing.get("opening_validation_reachability_ms_avg", 0.0)),
                    chain_observation_replay_calls=int(obj_runtime_timing.get("chain_observation_replay_calls", 0)),
                    chain_observation_replay_ms_total=float(obj_runtime_timing.get("chain_observation_replay_ms_total", 0.0)),
                    chain_observation_replay_ms_avg=float(obj_runtime_timing.get("chain_observation_replay_ms_avg", 0.0)),
                    any_wall_collision=search_any_wall_collision,
                    unique_movable_collision_count=search_unique_movable_collision_count,
                    primitive_trial_log=object_trial_log if self.exhaustive_mode else None,
                            reachability_log=object_reachability_log if self.exhaustive_mode else None,
                ))

            # Execution-mode early exit: once we have at least one successful opening
            # for this neighbor, skip the remaining candidate objects. FullNAMOPlanner
            # picks the next region on the path after this, so additional openings to
            # the same neighbor are wasted compute in execution. Disabled by default
            # so data-collection runs still gather per-object outcomes for every
            # candidate.
            if not self.exhaustive_mode and self.early_exit_on_first_success and total_solutions_collected >= 1:
                remaining = len(candidates) - obj_idx
                if remaining > 0 and self.config.verbose:
                    print(
                        f"      ⤴ Early exit: {object_id} opened '{neighbour_label}' "
                        f"({total_solutions_collected} solution(s)); "
                        f"skipping remaining {remaining} candidate object(s)"
                    )
                break

        # After trying all objects, return results
        if all_goal_attempts:
            # Separate successes and failures
            successes = [a for a in all_goal_attempts if a.success]
            failures = [a for a in all_goal_attempts if not a.success]

            # Set solutions_total for all attempts (successes and failures)
            # pushes_total_for_neighbour is already set per-object during creation
            for attempt in successes + failures:
                attempt.solutions_total_for_neighbour = total_solutions_collected

            # Truncate successes PER OBJECT (not globally)
            # Group by object_id and keep max_recorded per object
            from collections import defaultdict
            successes_by_object = defaultdict(list)
            for s in successes:
                successes_by_object[s.chosen_object_id].append(s)

            truncated_successes = []
            for obj_id, obj_successes in successes_by_object.items():
                # Truncate to max_recorded_solutions_per_neighbor per object
                truncated_successes.extend(obj_successes[:self.max_recorded_solutions_per_neighbor])

            # Combine: successes first, then failures (both per-object)
            all_goal_attempts = truncated_successes + failures

            # Update solutions_found_for_neighbour per object
            solutions_by_object = defaultdict(int)
            for s in truncated_successes:
                solutions_by_object[s.chosen_object_id] += 1

            for attempt in all_goal_attempts:
                attempt.solutions_found_for_neighbour = solutions_by_object.get(attempt.chosen_object_id, 0)

            return all_goal_attempts
        else:
            # No successful opening found from any object
            pushes_executed = neighbour_push_counter.get("count", 0)
            if timed_out:
                error_msg = f"Timeout after {self.timeout_per_neighbour_sec}s"
                failure_reason = "timeout"
            elif pushes_executed > 0:
                # Pushes were executed but all failed to create opening
                error_msg = f"Tried {len(candidates)} objects, {pushes_executed} pushes, none succeeded"
                failure_reason = "all_pushes_failed"
            elif total_ml_goals_generated == 0:
                # ML model produced no goals at all (extraction failed)
                error_msg = f"Tried {len(candidates)} objects, ML produced 0 goals"
                failure_reason = "ml_no_goals_extracted"
            elif total_ml_goals_aligned == 0:
                # ML produced goals but none aligned to any primitives
                error_msg = f"Tried {len(candidates)} objects, ML produced {total_ml_goals_generated} goals but 0 aligned to primitives"
                failure_reason = "ml_goals_not_aligned"
            elif total_reachable_edges == 0:
                # Goals aligned but none on reachable edges
                error_msg = f"Tried {len(candidates)} objects, {total_ml_goals_aligned} aligned but 0 reachable edges"
                failure_reason = "no_reachable_edges"
            else:
                # Fallback: goals existed but weren't tried for some other reason
                error_msg = f"Tried {len(candidates)} objects, no valid goals found (0 pushes, {total_ml_goals_generated} ML, {total_ml_goals_aligned} aligned)"
                failure_reason = "no_valid_goals"
            if self.config.verbose:
                print(f"      ✗ {error_msg}")
            return [AttemptResult(
                success=False,
                neighbour_region_label=neighbour_label,
                error_message=error_msg,
                connectivity_before=conn_before,
                timing_ms=(time.time() - attempt_start) * 1000,
                region_goal_used=precheck_region_goal,
                region_goals_sampled=all_region_goals,
                solutions_total_for_neighbour=total_solutions_collected,
                pushes_total_for_neighbour=pushes_executed,
                failure_reason=failure_reason,
                candidate_objects_count=len(candidates),
                ml_goals_generated=total_ml_goals_generated,
                ml_goals_aligned=total_ml_goals_aligned,
                reachable_edges_count=total_reachable_edges,
                aligned_primitives=all_aligned_primitives if all_aligned_primitives else None,
                ml_goals_raw=all_ml_goals_raw if all_ml_goals_raw else None,
                reachable_edges=sorted(list(all_reachable_edges)) if all_reachable_edges else None,
            )]

    def _collect_chain_observations(
        self,
        object_id: str,
        goal_chain: List[Goal],
        baseline_state: namo_rl.RLState
    ) -> Tuple[List, List, List, List, bool, int]:
        """Execute a goal chain and collect state observations for each push.

        Args:
            object_id: Object being pushed
            goal_chain: List of goals to execute in sequence
            baseline_state: Starting state

        Returns:
            Tuple of (state_observations, post_action_state_observations,
                     reachable_before, reachable_after, any_wall_collision,
                     unique_movable_collision_count)
        """
        self.env.set_full_state(baseline_state)
        state_obs = []
        post_state_obs = []
        reachable_before = []
        reachable_after = []

        # Collision tracking - accumulate across all pushes
        any_wall_collision = False
        all_movable_collisions: Set[str] = set()

        for goal in goal_chain:
            # Capture state and reachable objects before action
            pre_obs = self.env.get_observation()
            pre_reachable = self.env.get_reachable_objects()
            state_obs.append(pre_obs)
            reachable_before.append(pre_reachable)

            # Execute action — MUST include edge_idx/depth or the C++ skill rejects
            # the push with "edge_idx and depth must both be >= 0; this skill no
            # longer supports the MPC search fallback", leaving the object frozen
            # at baseline. Same fix as the single-push branch at line ~2105.
            action = namo_rl.Action()
            action.object_id = object_id
            action.x = goal.x
            action.y = goal.y
            action.theta = goal.theta
            action.edge_idx = goal.edge_idx
            action.depth = goal.depth
            self._consume_push_budget()
            step_result = self.env.step(action)

            # Extract collision info from step result
            if step_result.info.get("wall_collision", "false") == "true":
                any_wall_collision = True
            movable_str = step_result.info.get("movable_collisions", "")
            if movable_str:
                for obj_name in movable_str.split(","):
                    if obj_name:
                        all_movable_collisions.add(obj_name)

            # Capture state and reachable objects after action
            post_obs = self.env.get_observation()
            post_reachable = self.env.get_reachable_objects()
            post_state_obs.append(post_obs)
            reachable_after.append(post_reachable)

        unique_movable_collision_count = len(all_movable_collisions)
        return state_obs, post_state_obs, reachable_before, reachable_after, any_wall_collision, unique_movable_collision_count

    def _search_with_chaining_bfs(
        self,
        object_id: str,
        baseline_state: namo_rl.RLState,
        neighbour_label: str,
        region_goals: Dict[str, Any],
        max_solutions_to_collect: Optional[int] = None,
        push_counter: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[Tuple[List[Goal], List, List, 'namo_rl.RLState', Optional[Tuple], Optional[List[Tuple]], List, List, int, Optional[int], float]], int, Dict[str, int], str, bool, int]:
        """Outer BFS over chain depth: Try single pushes, then 2-push chains, then 3-push chains.

        Collects ALL successful chains across all depths instead of stopping early.

        Returns:
            Tuple of (all_chains, min_depth) where all_chains is a list of tuples:
            (goal_chain, state_obs, post_state_obs, resulting_state, region_goal_used,
             region_goals_sampled, reachable_before, reachable_after, total_cost,
             skill_calls_before_success, success_time). Returns ([], 0) if no solution found.
        """
        print = self._debug

        # Extract region_goals_sampled for the target neighbor (for ML inference mask generation)
        neighbour_region_goals = None
        if neighbour_label in region_goals:
            bundle = region_goals[neighbour_label]
            if bundle.goals:
                neighbour_region_goals = [(g.x, g.y, g.theta) for g in bundle.goals]

        # Initial frontier for chain depth 1
        root_node = ChainNode(
            state=baseline_state,
            goal=None,
            edge_idx=-1,
            depth=0,
            parent=None,
            step_cost=0
        )
        # Track frontier nodes by depth level - persists across phases
        # Key: depth level (0=root), Value: list of frontier nodes at that depth
        frontiers_by_depth: Dict[int, List[ChainNode]] = {0: [root_node]}

        # Collect all successful chains across all depths
        all_chains_across_depths = []
        min_chain_depth_found = None

        # Track best cumulative cost found so far; use for pruning
        best_total_cost = None
        skill_call_counter = {"count": 0}

        # Accumulated trial logs across all chain depths (for F characterization)
        all_trial_logs = []
        # Per-node reachable-edge recording (see RegionOpeningStats.reachability_log)
        reachability_log = []

        # Two-phase search for ml_first with ml_fallback:
        # Phase 1: Try ONLY ML goals (score > 0) across ALL depths
        # Phase 2: If Phase 1 fails, try ONLY primitives (score = 0) across ALL depths
        # This ensures ML predictions get global priority before falling back to primitives
        use_two_phase = self.selection_strategy == "ml_first"
        # phases: (stop_at_score_zero, primitives_only, phase_name)
        phases = [
            (True, False, "ML-only"),      # Phase 1: ML goals only
            (False, True, "primitives"),   # Phase 2: primitives only (skip ML)
        ] if use_two_phase else [(False, False, "all")]

        # Track global phase state across all depths
        global_phase_push_counts = {}
        # Track which phase found the first solution
        solved_in_phase = ""
        any_wall_collision_during_search = False
        movable_collisions_during_search: Set[str] = set()

        # Cache goals per node to avoid redundant ML inference across phases and depths
        # Key: node id, Value: (goals_per_edge, reachable_edge_indices)
        node_goals_cache: Dict[int, Tuple[List[List[Goal]], Set[int]]] = {}

        # Persist blacklists per node across phases and depths
        # Key: node id, Value: {edge_idx: min_stuck_depth}
        node_blacklists: Dict[int, Dict[int, int]] = {}

        # Ensure ML inference uses a goal seed consistent with region sampling.
        # Region-opening validation calls `set_robot_goal()` while iterating sampled goals and
        # can leave the env goal at an arbitrary (often last-sampled) value. Stack A SE(2)
        # inference uses `robot_goal` as the fallback seed for `goal_sample_region`, so we
        # explicitly set the env goal to the first sampled goal for the neighbor region when
        # generating ML goals to match training behavior.
        goal_seed = None
        try:
            seed_bundle = region_goals.get(neighbour_label)
            if seed_bundle and getattr(seed_bundle, "goals", None):
                seed_goal = seed_bundle.goals[0]
                goal_seed = (float(seed_goal.x), float(seed_goal.y), float(seed_goal.theta))
        except Exception:
            goal_seed = None

        for stop_at_zero, prims_only, phase_name in phases:
            # Skip primitives phase if ML phase found a solution
            if prims_only and len(all_chains_across_depths) > 0:
                if self.config.verbose:
                    print(f"    ⏭️ Skipping '{phase_name}' phase (ML found solution)")
                break

            # Reset blacklists at start of each phase for completeness
            # Phase 1 stuck edges shouldn't block Phase 2 exploration
            # This ensures each phase can fully explore its candidate space
            node_blacklists = {}

            if self.config.verbose:
                print(f"    🎯 Phase: {phase_name}")
            phase_start_pushes = skill_call_counter.get("count", 0)

            # Try chain depths 1, 2, 3, ...
            for chain_depth in range(1, self.max_chain_depth + 1):
                # Get frontier for this depth level (persists across phases)
                frontier = frontiers_by_depth.get(chain_depth - 1, [])

                next_frontier = []
                processed_frontiers = 0
                total_frontier_time_ms = 0.0
                orig_frontier_len = len(frontier)
                reached_cap = False

                # Verbose: indicate which chain-depth search level we are at
                if self.config.verbose:
                    chain_label = f"{chain_depth}-chain"
                    print(f"    ▶ Searching {chain_label} (frontier={len(frontier)})")

                # Ensure blacklists exist for frontier nodes (reuse if already created).
                # Seed each new blacklist with externally-reported failed edges for
                # this object_id (depth=0 means "skip at all depths" because the
                # check is `depth >= edge_min_stuck_depth[edge_idx]`).
                external_for_object = self.external_edge_blacklist.get(object_id, ())
                for node in frontier:
                    if id(node) not in node_blacklists:
                        node_blacklists[id(node)] = {
                            edge_idx: 0 for edge_idx in external_for_object
                        }

                depth_start_pushes = skill_call_counter.get("count", 0)

                node_idx = 0
                for node in frontier:
                    node_idx += 1
                    node_start_time = time.time()
                    # Cost-based node prune: if cost so far already meets/exceeds best, skip.
                    # DISABLED in label_mode: it would skip depth-2 setups in direct-opener scenes
                    # (cost>=best_total_cost=~1), which are exactly the 178k solvable scenes we relabel.
                    if best_total_cost is not None and not self.label_mode:
                        chain_cost_so_far = self._compute_chain_cost(node)
                        if chain_cost_so_far >= best_total_cost:
                            continue
                    # Restore to this node's state
                    self.env.set_full_state(node.state)

                    # Generate goals for this node (cached to avoid redundant ML inference across phases)
                    if id(node) not in node_goals_cache:
                        original_robot_goal = None
                        if goal_seed is not None:
                            try:
                                original_robot_goal = self.env.get_robot_goal()
                            except Exception:
                                original_robot_goal = None
                            try:
                                self.env.set_robot_goal(*goal_seed)
                            except Exception:
                                goal_seed = None
                        goal_gen_start = time.perf_counter()
                        goals_per_edge = self.goal_strategy.generate_goals(
                            object_id,
                            node.state,
                            self.env,
                            max_goals=0,
                            region_goals_sampled=neighbour_region_goals
                        )
                        self._record_goal_generation_timing((time.perf_counter() - goal_gen_start) * 1000.0)
                        if original_robot_goal is not None:
                            try:
                                self.env.set_robot_goal(*original_robot_goal)
                            except Exception:
                                pass
                        reachable_edge_indices = set(self.env.get_reachable_edges(object_id)) if goals_per_edge else set()
                        node_goals_cache[id(node)] = (goals_per_edge, reachable_edge_indices)
                        _ng = getattr(node, "goal", None)
                        reachability_log.append({
                            'chain_depth': node.depth + 1,
                            'parent_edge': getattr(_ng, "edge_idx", None) if _ng is not None else None,
                            'parent_depth': getattr(_ng, "depth", None) if _ng is not None else None,
                            'reachable_edges': sorted(reachable_edge_indices),
                            # SAVE THE NODE STATE ([USER 2026-06-13]: re-collect everything incl. post-push).
                            # env is AT node.state here (set_full_state above) -> for chain_depth>=2 this IS
                            # the post-push state s1. Persisting it makes dead post-push states first-class
                            # training rows (correct by construction — the actual s1 the a2 labels came from;
                            # the replay route DIVERGED on collisions, 86fd9b2). object_id tags whose pose
                            # is the pushed object so the renderer can anchor the crop.
                            'state_observation': self.env.get_observation(),
                            'object_id': object_id,
                        })

                        # Verbose: show reachable edges and object state
                        if self.config.verbose:
                            obs = self.env.get_observation()
                            obj_pose = obs.get(f"{object_id}_pose", [0, 0, 0])
                            print(f"    📍 Object {object_id} at ({obj_pose[0]:.3f}, {obj_pose[1]:.3f}, θ={obj_pose[2]:.3f})")
                            print(f"    🔗 Reachable edges ({len(reachable_edge_indices)}/{len(goals_per_edge)}): {sorted(reachable_edge_indices)}")
                    else:
                        goals_per_edge, reachable_edge_indices = node_goals_cache[id(node)]

                    if not goals_per_edge:
                        continue

                    if not reachable_edge_indices:
                        continue

                    # Run inner BFS (single-skill search)
                    collect_frontier = (chain_depth < self.max_chain_depth)
                    # Compute remaining budget if we already have a best cost
                    if 'best_total_cost' in locals() and best_total_cost is not None and not self.label_mode:
                        chain_cost_so_far = self._compute_chain_cost(node)
                        remaining_budget = max(0, best_total_cost - chain_cost_so_far)
                    else:
                        remaining_budget = None

                    # Calculate how many more solutions we need to collect
                    # If we have already collected some solutions in previous iterations (best_total_cost is set),
                    # we need to account for them.
                    # However, all_chains_across_depths is defined outside this loop and contains all valid solutions found so far.
                    current_solutions_count = len(all_chains_across_depths)

                    inner_max_solutions = None
                    if not self.exhaustive_mode and max_solutions_to_collect is not None:
                        if current_solutions_count >= max_solutions_to_collect:
                            reached_cap = True
                            break
                        inner_max_solutions = max_solutions_to_collect - current_solutions_count

                    successful_results, primitive_depth, new_frontier_nodes, bfs_any_wall_collision, bfs_movable_collisions, bfs_trial_log = self._search_bfs(
                        goals_per_edge,
                        reachable_edge_indices,
                        node.state,
                        neighbour_label,
                        region_goals,
                        object_id,
                        parent_node=node,
                        current_chain_depth=chain_depth,
                        collect_frontier=collect_frontier,
                        remaining_budget=remaining_budget,
                        skill_call_counter=skill_call_counter,
                        push_counter=push_counter,
                        max_solutions_to_collect=inner_max_solutions,
                        stop_at_score_zero=stop_at_zero,
                        primitives_only=prims_only,
                        shared_blacklist=node_blacklists.get(id(node)),
                    )
                    any_wall_collision_during_search = any_wall_collision_during_search or bfs_any_wall_collision
                    movable_collisions_during_search.update(bfs_movable_collisions)
                    all_trial_logs.extend(bfs_trial_log)

                    # If we found success, reconstruct ALL goal chains with their state observations
                    if successful_results:
                        for (final_goal, final_state_obs, final_post_state_obs, resulting_state, region_goal_used, all_region_goals, success_node, success_time) in successful_results:
                            # For multi-push chains, reconstruct full chain with observations
                            if chain_depth > 1:
                                goal_chain, state_obs, post_state_obs, reachable_before, reachable_after, total_cost, any_wall_collision, unique_movable_collision_count = self._reconstruct_chain_with_observations(
                                    success_node, object_id, baseline_state
                                )
                            else:
                                # Single push - use observations captured during search
                                goal_chain = [final_goal]
                                state_obs = final_state_obs
                                post_state_obs = final_post_state_obs
                                # For single push, we don't have reachable objects captured during BFS
                                # So collect them now with collision tracking
                                replay_start = time.perf_counter()
                                self.env.set_full_state(baseline_state)
                                reachable_before = [self.env.get_reachable_objects()]
                                # Execute the action to get reachable after and collision info.
                                # CRITICAL: include edge_idx + depth so the C++ env re-runs
                                # the *same* primitive the search just declared a success.
                                # Without these, the env falls back to picking a primitive
                                # from (x, y, theta), which routes to a different edge/depth
                                # — visible in --viewer as a different last push than what
                                # gets recorded in the chain. See chain JSON vs viewer
                                # discrepancy reported 2026-05-20.
                                action = namo_rl.Action()
                                action.object_id = object_id
                                action.x = final_goal.x
                                action.y = final_goal.y
                                action.theta = final_goal.theta
                                action.edge_idx = final_goal.edge_idx
                                action.depth = final_goal.depth
                                self._consume_push_budget()
                                step_result = self.env.step(action)
                                reachable_after = [self.env.get_reachable_objects()]
                                # Extract collision info from step result
                                any_wall_collision = step_result.info.get("wall_collision", "false") == "true"
                                movable_str = step_result.info.get("movable_collisions", "")
                                unique_movable_collision_count = len([s for s in movable_str.split(",") if s]) if movable_str else 0
                                self._record_chain_observation_replay_timing((time.perf_counter() - replay_start) * 1000.0)
                                # For single push, total_cost equals the primitive depth at which success occurred
                                total_cost = max(1, getattr(success_node, "step_cost", 1))

                            skill_calls_before_success = getattr(success_node, "skill_calls_before_success", None)

                            # Verbose: print each solution found at this chain depth
                            if self.config.verbose:
                                print(f"      ✓ Found solution at {chain_depth}-chain (total_cost={total_cost})")

                            # Entry layout: (goal_chain, state_obs, post_state_obs, resulting_state, region_goal_used, region_goals_sampled, reachable_before, reachable_after, total_cost, skill_calls, success_time, any_wall_collision, unique_movable_collision_count)
                            # Maintain only min-cost solutions so far; reset when a new lower cost is found
                            if best_total_cost is None or total_cost <= best_total_cost:
                                # If strictly better cost, reset collection to only keep new best-cost solutions
                                if best_total_cost is None or total_cost < best_total_cost:
                                    best_total_cost = total_cost
                                    all_chains_across_depths = [entry for entry in all_chains_across_depths if entry[8] == best_total_cost]

                                all_chains_across_depths.append(
                                    (
                                        goal_chain,
                                        state_obs,
                                        post_state_obs,
                                        resulting_state,
                                        region_goal_used,
                                        all_region_goals,
                                        reachable_before,
                                        reachable_after,
                                        total_cost,
                                        skill_calls_before_success,
                                        success_time,
                                        any_wall_collision,
                                        unique_movable_collision_count,
                                    )
                                )
                                if self.config.verbose:
                                    # Running count of min-cost solutions so far (object scope)
                                    print(f"        → Solutions so far (object, best_cost={best_total_cost}): {len(all_chains_across_depths)}")

                                # Early stop if we reached the per-object cap
                                if not self.exhaustive_mode and max_solutions_to_collect is not None and len(all_chains_across_depths) >= max_solutions_to_collect:
                                    reached_cap = True
                                    break

                        # Track minimum chain depth where we found a solution
                        if min_chain_depth_found is None:
                            min_chain_depth_found = chain_depth
                            # Record which phase found the first solution
                            if not solved_in_phase:
                                solved_in_phase = phase_name

                    # Add new frontier nodes for next chain level
                    next_frontier.extend(new_frontier_nodes)

                    if reached_cap:
                        break

                    # Per-frontier timing
                    processed_frontiers += 1
                    node_elapsed_ms = (time.time() - node_start_time) * 1000.0
                    total_frontier_time_ms += node_elapsed_ms
                    if self.config.verbose:
                        print(f"      • Frontier {processed_frontiers}/{orig_frontier_len} took {node_elapsed_ms:.1f} ms")

                # Sort frontier for next chain depth (based on selection strategy)
                if self.selection_strategy == "ml_first":
                    # (-score, chain_cost, step_cost) - trust ML votes, cost as tiebreaker
                    next_frontier.sort(key=lambda n: (
                        -getattr(n.goal, "score", 0.0) if n.goal else 0.0,
                        self._compute_chain_cost(n),
                        getattr(n, "step_cost", 0)
                    ))
                else:
                    # "cost_first": (chain_cost, step_cost, -score) - minimize disruption, ML as tiebreaker
                    next_frontier.sort(key=lambda n: (
                        self._compute_chain_cost(n),
                        getattr(n, "step_cost", 0),
                        -getattr(n.goal, "score", 0.0) if n.goal else 0.0
                    ))

                # Store new frontier nodes at this depth level (extend if already exists from previous phase)
                if next_frontier:
                    # Apply beam width pruning if configured
                    if self.frontier_beam_width is not None:
                        next_frontier = next_frontier[: self.frontier_beam_width]

                    if chain_depth not in frontiers_by_depth:
                        frontiers_by_depth[chain_depth] = next_frontier
                    else:
                        # Extend existing frontier (Phase 2 may add more nodes)
                        frontiers_by_depth[chain_depth].extend(next_frontier)

                # Log depth completion stats
                if self.config.verbose and processed_frontiers > 0:
                    avg_ms = total_frontier_time_ms / processed_frontiers
                    print(f"    ◼ Completed {processed_frontiers}/{orig_frontier_len} frontiers | avg {avg_ms:.1f} ms | total {total_frontier_time_ms:.1f} ms")

                # Check early termination conditions
                if reached_cap:
                    break

            # End of phase - log push count (outside chain_depth loop)
            phase_end_pushes = skill_call_counter.get("count", 0)
            phase_pushes = phase_end_pushes - phase_start_pushes
            global_phase_push_counts[phase_name] = phase_pushes
            if self.config.verbose:
                found_in_phase = len(all_chains_across_depths) > 0
                print(f"      📈 Phase '{phase_name}': {phase_pushes} pushes, found={found_in_phase}")

        # Log total phase breakdown (outside phases loop)
        if self.config.verbose and len(global_phase_push_counts) > 1:
            total_phase_pushes = sum(global_phase_push_counts.values())
            print(f"      📊 Phase breakdown: {global_phase_push_counts}, total={total_phase_pushes}")

        # If we found any chains, filter to keep only minimum-cost ones
        if all_chains_across_depths:
            best_cost = min(entry[8] for entry in all_chains_across_depths)
            min_cost_chains = [entry for entry in all_chains_across_depths if entry[8] == best_cost]
            if self.config.verbose:
                print(f"    ✔ Returning {len(min_cost_chains)} min-cost solution(s) with cost={best_cost}")
            return min_cost_chains, min_chain_depth_found if min_chain_depth_found else 0, global_phase_push_counts, solved_in_phase, any_wall_collision_during_search, len(movable_collisions_during_search), all_trial_logs, reachability_log
        else:
            return all_chains_across_depths, 0, global_phase_push_counts, solved_in_phase, any_wall_collision_during_search, len(movable_collisions_during_search), all_trial_logs, reachability_log

    def _reconstruct_chain(self, final_node: ChainNode, final_goal: Goal) -> List[Goal]:
        """Reconstruct the chain of goals from root to final goal."""
        chain = []
        node = final_node

        # Walk back to root, collecting goals
        while node.parent is not None:
            chain.append(node.goal)
            node = node.parent

        # Reverse to get root-to-leaf order
        chain.reverse()

        # Add the final goal
        chain.append(final_goal)

        return chain

    def _reconstruct_chain_with_observations(
        self,
        success_node: ChainNode,
        object_id: str,
        baseline_state: namo_rl.RLState
    ) -> Tuple[List[Goal], List, List, List, List, int, bool, int]:
        """Reconstruct goal chain and collect observations by re-executing.

        This is only called for multi-push chains (chain_depth > 1).

        Args:
            success_node: Final ChainNode containing parent chain
            object_id: Object being pushed
            baseline_state: Starting state for re-execution

        Returns:
            Tuple of (goal_chain, state_obs, post_state_obs, reachable_before, reachable_after,
                     total_cost, any_wall_collision, unique_movable_collision_count)
        """
        # Reconstruct goal chain from parent nodes
        goal_chain = []
        node = success_node
        while node.parent is not None:
            goal_chain.append(node.goal)
            node = node.parent
        goal_chain.reverse()

        # Re-execute chain to collect observations
        replay_start = time.perf_counter()
        state_obs, post_state_obs, reachable_before, reachable_after, any_wall_collision, unique_movable_collision_count = self._collect_chain_observations(
            object_id, goal_chain, baseline_state
        )
        self._record_chain_observation_replay_timing((time.perf_counter() - replay_start) * 1000.0)

        # Compute cumulative cost along the reconstructed chain
        total_cost = 0
        num_pushes = 0
        node = success_node
        while node.parent is not None:
            total_cost += max(0, getattr(node, "step_cost", 0))
            num_pushes += 1
            node = node.parent
        # Add chain link cost for multi-push chains (flat cost, not per-link)
        if num_pushes > 1:
            total_cost += self.chain_link_cost

        return goal_chain, state_obs, post_state_obs, reachable_before, reachable_after, total_cost, any_wall_collision, unique_movable_collision_count

    def _compute_chain_cost(self, node: ChainNode) -> int:
        """Compute cumulative additive cost from root to the given node.

        Root node has cost 0. Each edge contributes its inner primitive depth (1-based).
        Additionally, chain_link_cost is added once if this is a multi-push chain.

        Total cost = sum(step_costs) + chain_link_cost (if num_pushes > 1)
        """
        total = 0
        num_pushes = 0
        cursor = node
        while cursor is not None and cursor.parent is not None:
            total += max(0, getattr(cursor, "step_cost", 0))
            num_pushes += 1
            cursor = cursor.parent
        # Add chain link cost for multi-push chains (flat cost, not per-link)
        if num_pushes > 1:
            total += self.chain_link_cost
        return total

    def _search_bfs(
        self,
        goals_or_async: Union[List[List[Goal]], AsyncGoalResult],
        reachable_edge_indices: Set[int],
        baseline_state: namo_rl.RLState,
        neighbour_label: str,
        region_goals: Dict[str, Any],
        object_id: str,
        parent_node: Optional[ChainNode] = None,
        current_chain_depth: int = 1,
        collect_frontier: bool = False,
        remaining_budget: Optional[int] = None,
        skill_call_counter: Optional[Dict[str, int]] = None,
        push_counter: Optional[Dict[str, int]] = None,
        max_solutions_to_collect: Optional[int] = None,
        stop_at_score_zero: bool = False,
        primitives_only: bool = False,
        shared_blacklist: Optional[Dict[int, int]] = None,
    ) -> Tuple[List[Tuple[Goal, List, List, 'namo_rl.RLState', Optional[Tuple], ChainNode, float]], int, List[ChainNode], bool, Set[str]]:
        """BFS: Try all edges at ALL depths to collect all possible solutions.

        Supports async ML inference: if goals_or_async is an AsyncGoalResult,
        primitives start executing immediately while ML runs in background.
        When ML completes, remaining candidates are re-sorted by ML scores.

        Args:
            goals_or_async: Either List[List[Goal]] (sync) or AsyncGoalResult (async).
            collect_frontier: If True, collect valid but unsuccessful states as frontier nodes
            max_solutions_to_collect: If provided, stop searching once this many successful solutions are found.
            stop_at_score_zero: If True, stop trying candidates when score becomes 0 (ML-first phase).
                               Used for two-phase chain search: ML goals first across all nodes,
                               then primitives. Only affects ml_first selection strategy.
            primitives_only: If True, skip ML goals (score > 0) and only try primitives (score = 0).
                            Used for Phase 2 of two-phase search to avoid re-trying ML goals.
            shared_blacklist: Optional dict to share edge blacklist across phases. If provided,
                             this dict is used and updated in-place. Same (edge, depth) = same target.

        Returns:
            Tuple of (all_successful_results, min_depth, frontier_nodes) where all_successful_results
            is a list of (goal, state_obs, post_state_obs, resulting_state, region_goal_used, chain_node, success_time) tuples
            from ALL depths. min_depth is the minimum depth at which a solution was found.
            The chain_node contains the full parent chain for observation reconstruction.
        """
        print = self._debug

        # Handle both sync and async goal results
        if isinstance(goals_or_async, AsyncGoalResult):
            async_result = goals_or_async
            goals_per_edge = async_result.primitive_goals
        else:
            async_result = None
            goals_per_edge = goals_or_async

        max_depth = len(goals_per_edge[0]) if goals_per_edge else 10

        # Track minimum depth at which each edge got stuck/collided during THIS skill execution
        # Only skip depths >= the stuck depth (shallower depths might still work)
        # Use shared_blacklist if provided (for two-phase search), otherwise create fresh
        edge_min_stuck_depth: Dict[int, int] = shared_blacklist if shared_blacklist is not None else {}

        # Track edges that have already yielded a successful opening in THIS skill execution
        # Once an edge succeeds at any primitive depth, we do not explore deeper depths on that edge
        solved_edges_this_skill = set()

        # Frontier nodes for chaining
        frontier_nodes = []

        # Collect ALL successful results across all depths
        all_successful_results = []
        min_depth_found = None

        # Per-primitive trial log for F characterization
        trial_log = []

        # Track async ML merge state
        ml_merged = False
        ml_scores: Dict[Tuple[int, int], float] = {}
        ml_scored_slots: Set[Tuple[int, int]] = set()  # Track which (edge, depth) have ML scores
        any_wall_collision_during_search = False
        movable_collisions_during_search: Set[str] = set()


        # Flatten goals into candidates for prioritized iteration.
        #
        # IMPORTANT: `goals_per_edge` may be returned in a shuffled edge ordering (see
        # PrimitiveGoalStrategy.shuffle_edges). The outer list index is NOT guaranteed to be
        # the true primitive edge index, so we must use `goal.edge_idx` when filtering against
        # `reachable_edge_indices` (which are true edge indices from the env).
        #
        # Use list of lists to allow mutation during re-sort.
        candidates = []
        for edge_goals in goals_per_edge:
            # Determine the true edge index for this group.
            true_edge_idx = None
            for g in edge_goals:
                if g is not None:
                    true_edge_idx = int(getattr(g, "edge_idx", -1))
                    break
            if true_edge_idx is None or true_edge_idx < 0:
                continue

            # Filter: only try reachable edges (true indices).
            if true_edge_idx not in reachable_edge_indices:
                continue

            for depth, goal in enumerate(edge_goals):
                if goal is not None:
                    candidates.append([true_edge_idx, depth, goal])  # Use list for mutability

        # Sampled collection: uniform k-subset of the reachable candidates (every chain level hits this).
        if self.sample_k > 0 and len(candidates) > self.sample_k:
            candidates = random.sample(candidates, self.sample_k)

        # Initial sort depends on whether we have async ML
        if async_result is not None and async_result.ml_future is not None:
            # Async mode: sort by (depth, edge) initially - shortest pushes first while waiting for ML
            sort_start = time.perf_counter()
            candidates.sort(key=lambda x: (x[1], x[0]))
            self._record_primitive_ranking_timing((time.perf_counter() - sort_start) * 1000.0, len(candidates))
            if self.config.verbose:
                print(f"      📋 Async mode: {len(candidates)} candidates sorted by depth (ML running in background)")
        else:
            # Sync mode:
            # - ML strategies want score-first ordering (ML goals before primitives).
            # - GeometricTransportStrategy wants depth-first ordering, then priority within depth.
            depth_first = isinstance(self.goal_strategy, GeometricTransportStrategy)
            sort_start = time.perf_counter()
            _sort_candidates_sync(candidates, depth_first=depth_first)
            self._record_primitive_ranking_timing((time.perf_counter() - sort_start) * 1000.0, len(candidates))

        # Track position for re-sorting remaining candidates
        candidate_idx = 0

        while candidate_idx < len(candidates):
            # label-mode k-cap: at the finish level, stop after label_topk tried candidates (success
            # still early-stops first via the label_mode break below; this bounds the FAILURE cost).
            if (self.label_mode and self.label_topk > 0
                    and current_chain_depth >= self.max_chain_depth
                    and candidate_idx >= self.label_topk):
                break
            # ══════════════════════════════════════════════════════════════════
            # ASYNC ML POLLING: Check if ML inference is ready (non-blocking)
            # ══════════════════════════════════════════════════════════════════
            if async_result is not None and not ml_merged and async_result.poll_ml_ready():
                ml_scores = async_result.get_ml_scores()
                ml_merged = True
                ml_scored_slots = set(ml_scores.keys())  # Track all ML-scored slots

                if self.config.verbose:
                    print(f"      🎯 ML ready! {len(ml_scores)} slots with votes (after {candidate_idx} primitive pushes)")

                # Update scores for remaining candidates
                for i in range(candidate_idx, len(candidates)):
                    edge_idx_i, depth_i, goal_i = candidates[i]
                    key = (edge_idx_i, depth_i)
                    if key in ml_scores:
                        # Update goal with ML score
                        candidates[i][2] = Goal(
                            x=goal_i.x,
                            y=goal_i.y,
                            theta=goal_i.theta,
                            score=ml_scores[key]
                        )

                # Re-sort remaining candidates: score DESC, depth ASC, edge_idx ASC
                remaining = candidates[candidate_idx:]
                # ML merge always uses score-first re-sorting, regardless of depth-first
                # geometric ordering (geometric strategy does not use async ML).
                sort_start = time.perf_counter()
                _sort_candidates_sync(remaining, depth_first=False)
                self._record_primitive_ranking_timing((time.perf_counter() - sort_start) * 1000.0, len(remaining))
                candidates[candidate_idx:] = remaining

                if self.config.verbose:
                    # Show top candidates after re-sort
                    top_5 = candidates[candidate_idx:candidate_idx+5]
                    top_info = [(c[0], c[1], getattr(c[2], 'score', 0.0)) for c in top_5]
                    print(f"      📊 Re-sorted remaining {len(remaining)} candidates. Top 5: {top_info}")

            # Get current candidate
            edge_idx, depth, goal = candidates[candidate_idx]
            candidate_idx += 1

            # Two-phase chain search: stop at score=0 candidates (primitives) during ML-first phase
            # Since candidates are sorted by -score, once we hit score=0, all remaining are primitives
            if stop_at_score_zero and getattr(goal, 'score', 0.0) == 0:
                if self.config.verbose:
                    print(f"        ⏸️ ML-first phase: stopping at score=0 (primitive), {len(candidates) - candidate_idx + 1} candidates remaining")
                break

            # Two-phase chain search: skip ML goals (score > 0) during primitives-only phase
            # This avoids re-trying ML goals that already failed in Phase 1
            if primitives_only and getattr(goal, 'score', 0.0) > 0:
                continue

            # Stop if we've reached the solution cap (disabled in exhaustive mode)
            if not self.exhaustive_mode and max_solutions_to_collect is not None and len(all_successful_results) >= max_solutions_to_collect:
                if self.config.verbose:
                    print(f"        🛑 Reached max solutions ({len(all_successful_results)}/{max_solutions_to_collect}), stopping search")
                break
            # Global prune: once any success is found at depth D, skip candidates with depth > D
            # (disabled in exhaustive mode — evaluate all depths)
            if not self.exhaustive_mode and min_depth_found is not None and depth > min_depth_found:
                continue

            # Budget prune
            if remaining_budget is not None and (depth + 1) > remaining_budget:
                continue

            # Filter: skip if this edge got stuck/collided at a shallower or equal depth
            # (shallower depths than the stuck depth are still worth trying)
            # NOTE: blacklist is NOT disabled in exhaustive mode — stuck/collision is physical, not search pruning
            # Exception: if ml_ignore_blacklist is enabled, we disable blacklist during pre-ML phase
            # entirely, and bypass for ML-scored slots after ML merges
            is_blacklisted = edge_idx in edge_min_stuck_depth and depth >= edge_min_stuck_depth[edge_idx]
            if is_blacklisted:
                # During pre-ML phase: disable blacklist entirely if ml_ignore_blacklist is enabled
                if not ml_merged and self.ml_ignore_blacklist:
                    if self.config.verbose:
                        print(f"        🔓 Ignoring blacklist during pre-ML phase (edge {edge_idx}, depth {depth+1})")
                # After ML merge: bypass only for ML-scored slots
                elif ml_merged and self.ml_ignore_blacklist and (edge_idx, depth) in ml_scored_slots:
                    if self.config.verbose:
                        print(f"        🔓 Bypassing blacklist for ML-scored slot (edge {edge_idx}, depth {depth+1})")
                else:
                    self._rejection_stats["skipped_edge_blacklisted_deeper"] = self._rejection_stats.get("skipped_edge_blacklisted_deeper", 0) + 1
                    continue

            # Filter: skip edges that have already produced a successful opening
            # (disabled in exhaustive mode — evaluate all depths on all edges)
            if not self.exhaustive_mode and edge_idx in solved_edges_this_skill:
                self._rejection_stats["skipped_edge_already_solved"] = self._rejection_stats.get("skipped_edge_already_solved", 0) + 1
                continue

            self.env.set_full_state(baseline_state)

            # Check reachability BEFORE push
            is_accessible_before, reachable_count_before, _, _ = self._validate_opening(neighbour_label, region_goals)
            if self.config.verbose:
                print(f"        🔍 BEFORE push edge {edge_idx} depth {depth+1}: is_accessible={is_accessible_before}, reachable={reachable_count_before}")

            # Capture state observation before action
            pre_state_obs = self.env.get_observation()

            # Check if this slot has an ML-aligned goal
            if depth == 0 and self.config.verbose:  # Only print for first depth to reduce noise, and only in verbose
                total_region_goals = len(region_goals[neighbour_label].goals) if neighbour_label in region_goals else 0
                goal_type = f"score={getattr(goal, 'score', 0.0):.1f}" if goal is not None else "empty"
                print(f"      Testing edge {edge_idx} depth {depth+1} ({goal_type}): {neighbour_label} ({reachable_count_before}/{total_region_goals} reachable before)")

            # Execute push
            action = namo_rl.Action()
            action.object_id = object_id
            action.x = goal.x
            action.y = goal.y
            action.theta = goal.theta
            action.edge_idx = goal.edge_idx  # Pass actual edge index for direct C++ execution
            action.depth = goal.depth        # Pass depth (0-indexed) for direct C++ execution

            if self.config.verbose:
                # Show object current position vs goal position
                obs = self.env.get_observation()
                obj_pose = obs.get(f"{object_id}_pose", [0, 0, 0])
                dx = goal.x - obj_pose[0]
                dy = goal.y - obj_pose[1]
                dtheta = goal.theta - obj_pose[2]
                print(f"        🚀 EXECUTING PUSH edge {edge_idx} depth {depth+1}:")
                print(f"           object_id={object_id}")
                print(f"           current=({obj_pose[0]:.3f}, {obj_pose[1]:.3f}, θ={obj_pose[2]:.3f})")
                print(f"           goal=({goal.x:.3f}, {goal.y:.3f}, θ={goal.theta:.3f})")
                print(f"           delta=({dx:.3f}, {dy:.3f}, dθ={dtheta:.3f})")

            if skill_call_counter is not None:
                skill_call_counter["count"] += 1
            if push_counter is not None:
                push_counter["count"] += 1

            # Progress reporter: print a one-liner every progress_interval_sec
            # so the user can see the BFS is alive and tracking its throughput.
            # NOTE: `print` is rebound to self._debug at top of this function, so
            # we use sys.stdout.write+flush to bypass and always emit (even when
            # verbose is off).
            self._progress_total_primitives += 1
            _now = time.time()
            if _now - self._progress_last_print_time >= self._progress_interval_sec:
                _delta = self._progress_total_primitives - self._progress_last_print_count
                _elapsed = _now - self._progress_last_print_time
                _rate = _delta / _elapsed if _elapsed > 0 else 0.0
                import sys
                sys.stdout.write(
                    f"  [Progress] {self._progress_total_primitives} primitives tried "
                    f"({_rate:.1f}/sec) — current: obj={object_id} "
                    f"edge={goal.edge_idx} depth={goal.depth + 1} "
                    f"neighbour='{neighbour_label}'\n"
                )
                sys.stdout.flush()
                self._progress_last_print_time = _now
                self._progress_last_print_count = self._progress_total_primitives

            try:
                if self.config.verbose:
                    print(f"        DEBUG: goal.edge_idx={goal.edge_idx}, goal.depth={goal.depth}")
                    print(f"        ⏳ Calling env.step()...")
                self._consume_push_budget()
                step_start = time.perf_counter()
                step_result = self.env.step(action)
                step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0
                self._record_push_exec_timing(step_elapsed_ms, depth + 1)
                if self.config.verbose:
                    print(f"        ✓ env.step() returned: done={step_result.done}, reward={step_result.reward}")
                    print(f"        ✓ step_result.info={step_result.info}")
                    # Check object position after push
                    post_obs = self.env.get_observation()
                    post_pose = post_obs.get(f"{object_id}_pose", [0, 0, 0])
                    print(f"        ✓ AFTER push: object at ({post_pose[0]:.3f}, {post_pose[1]:.3f}, θ={post_pose[2]:.3f})")

                # Visualize after action if enabled
                self._focus_camera_on_object(object_id)

            except PushBudgetExceeded:
                raise
            except Exception as e:
                if self.config.verbose:
                    print(f"        ❌ EXCEPTION during env.step(): {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self._rejection_stats["env_step_exception"] = self._rejection_stats.get("env_step_exception", 0) + 1
                continue

            wall_collision = step_result.info.get("wall_collision", "false")
            if isinstance(wall_collision, str):
                any_wall_collision_during_search = any_wall_collision_during_search or (wall_collision.lower() == "true")
            else:
                any_wall_collision_during_search = any_wall_collision_during_search or bool(wall_collision)

            movable_raw = step_result.info.get("movable_collisions", "")
            if isinstance(movable_raw, str):
                if movable_raw:
                    for obj_name in movable_raw.split(","):
                        obj_name = obj_name.strip()
                        if obj_name:
                            movable_collisions_during_search.add(obj_name)
            elif isinstance(movable_raw, (list, tuple, set)):
                for obj_name in movable_raw:
                    obj_str = str(obj_name).strip()
                    if obj_str:
                        movable_collisions_during_search.add(obj_str)

            # Categorize the outcome from step_result.info for diagnostic stats.
            # Each push gets exactly ONE outcome bucket; the final assignment
            # is decided below after we've also seen the post-step region check.
            # `_sim_outcome` holds the sim-side categorization (if any); the
            # post-step block decides between push_opened_region /
            # push_did_not_open_region for pushes with no sim-side failure.
            self._rejection_stats["executed_in_sim"] = self._rejection_stats.get("executed_in_sim", 0) + 1
            _info = step_result.info or {}
            _ftype = str(_info.get("failure_type", ""))
            _sim_outcome: Optional[str] = None
            if _ftype == "4":
                _sim_outcome = "edge_unreachable"
            elif _info.get("stuck") == "true" or _ftype == "3":
                _sim_outcome = "controller_stuck"
            elif _info.get("wall_collision") == "true" or _ftype == "2":
                _sim_outcome = "push_collided_with_wall"
            elif _ftype and _ftype != "0":
                _sim_outcome = f"failure_type_{_ftype}"
            if _sim_outcome is not None:
                self._rejection_stats[_sim_outcome] = self._rejection_stats.get(_sim_outcome, 0) + 1

            # We have a post-action state - ALWAYS capture observation and check goal condition

            post_state_obs = self.env.get_observation()

            # Check reachability AFTER push (ALWAYS - this is the goal check for post-action state)
            is_accessible_after, reachable_count_after, region_goal_used, all_region_goals = self._validate_opening(neighbour_label, region_goals)
                
            if self.config.verbose:
                print(f"        🔍 AFTER push edge {edge_idx} depth {depth+1}: is_accessible={is_accessible_after}, reachable={reachable_count_after}")

            # Detect error conditions (but don't skip goal check - already done above)
            collision_detected = False
            if self.terminate_on_collision and "collision_object" in step_result.info:
                if self.config.verbose:
                    print(f"        ⚠️  COLLISION detected: {step_result.info.get('collision_object', 'unknown')}")
                collision_detected = True
                # Record this depth as stuck - shallower depths might still work
                if edge_idx not in edge_min_stuck_depth or depth < edge_min_stuck_depth[edge_idx]:
                    edge_min_stuck_depth[edge_idx] = depth
                    if self.config.verbose:
                        print(f"        📍 Edge {edge_idx} stuck at depth {depth+1}, depths 1-{depth} still valid")

            stuck_detected = False
            if "stuck" in step_result.info and step_result.info["stuck"] == "true":
                if self.config.verbose:
                    print(f"        ⚠️  STUCK condition detected")
                stuck_detected = True
                # Record this depth as stuck - shallower depths might still work
                if edge_idx not in edge_min_stuck_depth or depth < edge_min_stuck_depth[edge_idx]:
                    edge_min_stuck_depth[edge_idx] = depth
                    if self.config.verbose:
                        print(f"        📍 Edge {edge_idx} stuck at depth {depth+1}, depths 1-{depth} still valid")

            # Log this primitive trial for F characterization.
            # chain_depth + parent_{edge,depth} make the EXHAUSTIVE trial log self-describing:
            #   depth-1 successes -> F (1-push solving cells);
            #   depth-2 successes -> their parent first-push enabled a 2-push solve -> F1'.
            # (The recorded episode_results are only a SAMPLE of solutions; the trial log is exhaustive.)
            _parent_goal = getattr(parent_node, "goal", None) if parent_node is not None else None
            # RUNG-2: persist the post-push full state (qpos/qvel) on setup pushes (chain_depth <
            # max_chain_depth) so build_rung2_h5 can render a depth-2 node's ctx from the exact
            # post-shove state its second pushes were searched from. The env is AT the post-push
            # state here (post-_validate_opening, same point as the frontier ChainNode.state at ~2970
            # and the state re-applied via set_full_state at ~2129), so this is faithful by
            # construction. RLState is not picklable -> store plain qpos/qvel lists (asdict/pickle
            # safe). Gated on exhaustive_mode + non-leaf depth (terminal states are never re-rendered).
            _resulting_state = None
            if self.exhaustive_mode and current_chain_depth < self.max_chain_depth:
                _rs = self.env.get_full_state()
                _resulting_state = {'qpos': list(_rs.qpos), 'qvel': list(_rs.qvel)}
            trial_log.append({
                'edge_idx': edge_idx,
                'depth': depth,
                'success': is_accessible_after and not is_accessible_before,
                'wall_collision': step_result.info.get("wall_collision", "false") == "true",
                'movable_collisions': step_result.info.get("movable_collisions", ""),
                'stuck': stuck_detected,
                'collision': collision_detected,
                'reachable_after': reachable_count_after,
                'chain_depth': current_chain_depth,
                'parent_edge': getattr(_parent_goal, "edge_idx", None) if _parent_goal is not None else None,
                'parent_depth': getattr(_parent_goal, "depth", None) if _parent_goal is not None else None,
                'resulting_state': _resulting_state,
                # LABEL: scorer value + 0-based rank of this push in the -score sorted order at its node,
                # so build_rung2 can flag recall-misses (winning finish rank > k) for recycling.
                'score': float(getattr(goal, "score", 0.0)),
                'rank': candidate_idx - 1,
            })

            total_region_goals = len(region_goals[neighbour_label].goals) if neighbour_label in region_goals else 0
            if is_accessible_after and not is_accessible_before:
                # Successful opening — but only count if there was no sim-side
                # failure (otherwise this push was already bucketed by _sim_outcome).
                if _sim_outcome is None:
                    self._rejection_stats["push_opened_region"] = self._rejection_stats.get("push_opened_region", 0) + 1
                print(f"      ✅ SUCCESS! {object_id} edge {edge_idx} depth {depth+1}: {reachable_count_before}/{total_region_goals} → {reachable_count_after}/{total_region_goals} reachable")
            else:
                # Push ran without any sim-side failure but didn't change region
                # accessibility. Only bucket here if not already attributed to
                # a sim-side outcome — ensures each push is counted exactly once.
                if _sim_outcome is None:
                    self._rejection_stats["push_did_not_open_region"] = self._rejection_stats.get("push_did_not_open_region", 0) + 1
                if depth == 0 and goal is not None and self.config.verbose:
                    print(f"        ✗ Failed edge {edge_idx} depth {depth+1}: {reachable_count_before}/{total_region_goals} → {reachable_count_after}/{total_region_goals}")

            # Check if we IMPROVED accessibility (goal condition for opening creation)
            if is_accessible_after and not is_accessible_before:
                # Created NEW opening! ✓ (even if stuck/collision - object moved enough)
                success_timestamp = time.time()
                resulting_state = self.env.get_full_state()

                # Create a ChainNode for this successful goal (stores observations)
                success_node = ChainNode(
                    state=resulting_state,
                    goal=goal,
                    edge_idx=edge_idx,
                    depth=current_chain_depth,
                    parent=parent_node,
                    collided_edges=set(),
                    step_cost=depth + 1
                )
                if skill_call_counter is not None:
                    success_node.skill_calls_before_success = skill_call_counter["count"]

                # Add to all results instead of returning early
                all_successful_results.append(
                    (
                        goal,
                        [pre_state_obs],
                        [post_state_obs],
                        resulting_state,
                        region_goal_used,
                        all_region_goals,
                        success_node,
                        success_timestamp,
                    )
                )

                # Track minimum depth where we found a solution
                if min_depth_found is None:
                    min_depth_found = depth + 1

                # Prevent exploring deeper depths for this edge in this BFS call
                solved_edges_this_skill.add(edge_idx)

                # LABEL mode: at the FINISH level (deepest chain) stop at the first opening finish.
                # Candidates are scorer-sorted, so this is R's best-ranked finish that works; its rank
                # (in trial_log) says if it was within top-k. Setups (chain_depth < max) stay exhaustive —
                # this break does NOT fire for them.
                if self.label_mode and current_chain_depth >= self.max_chain_depth:
                    break

                # If we're only collecting a fixed number of solutions, stop immediately
                # once we reach the cap (don’t wait for the next candidate iteration).
                if not self.exhaustive_mode and max_solutions_to_collect is not None and len(all_successful_results) >= max_solutions_to_collect:
                    if self.config.verbose:
                        print(
                            f"        🛑 Reached max solutions ({len(all_successful_results)}/{max_solutions_to_collect}), stopping search"
                        )
                    break

            elif not (collision_detected or stuck_detected) and collect_frontier:
                # Valid push but didn't create opening - add to frontier
                # (Don't add stuck/collision states to frontier - they're already blacklisted)
                if remaining_budget is None or (depth + 1) <= remaining_budget:
                    new_node = ChainNode(
                        state=self.env.get_full_state(),
                        goal=goal,
                        edge_idx=edge_idx,
                        depth=current_chain_depth,
                        parent=parent_node,
                        collided_edges=set(),
                        step_cost=depth + 1
                    )
                    frontier_nodes.append(new_node)

        # ══════════════════════════════════════════════════════════════════
        # CLEANUP: Cancel pending ML inference if not yet merged
        # ══════════════════════════════════════════════════════════════════
        if async_result is not None and not ml_merged:
            cancelled = async_result.cancel_if_pending()
            if self.config.verbose:
                if cancelled:
                    print(f"      🚫 Cancelled pending ML inference (solution found before ML ready)")
                else:
                    print(f"      ⏳ ML inference still running (will complete in background)")

        # Return all successful results found across all depths
        return all_successful_results, min_depth_found if min_depth_found else 0, frontier_nodes, any_wall_collision_during_search, movable_collisions_during_search, trial_log

    def _validate_opening(
        self,
        neighbour_label: str,
        region_goals: Dict[str, Any]
    ) -> Tuple[bool, int, Optional[Tuple[float, float, float]], Optional[List[Tuple[float, float, float]]]]:
        """Validate that opening to neighbour was created using reachability.

        Success criterion: at least ``self._success_min_reachable`` of the
        sampled region goals must be reachable (default 1; configurable via
        algo_params["region_success_min_reachable"]).

        Args:
            neighbour_label: Target neighbour region
            region_goals: Region goal samples from snapshot

        Returns:
            Tuple of (success, reachable_count, first_reachable_goal, all_region_goals):
                - success: True if >= self._success_min_reachable goals reachable
                - reachable_count: Number of reachable goals
                - first_reachable_goal: First reachable goal found (for validation)
                - all_region_goals: All goal samples for this region (for visualization)
        """
        validation_start = time.perf_counter()
        goal_checks = 0
        reachability_calls = 0
        reachability_ms_total = 0.0
        try:
            # Get region goals for this neighbour
            if neighbour_label not in region_goals:
                return False, 0, None, None

            bundle = region_goals[neighbour_label]
            if not bundle.goals:
                return False, 0, None, None

            # Collect all goal samples for visualization / return value.
            all_goals = [(g.x, g.y, g.theta) for g in bundle.goals]

            # Single C++ call: updates the wavefront once, then performs cheap
            # grid lookups for every (x, y) point. Does NOT mutate the env's
            # robot goal, so the user's task goal is preserved across this call.
            xy_points = [(g.x, g.y) for g in bundle.goals]
            reachability_start = time.perf_counter()
            reachable_count, first_idx = self.env.count_reachable_points(xy_points)
            reachability_ms_total = (time.perf_counter() - reachability_start) * 1000.0
            reachability_calls = 1
            goal_checks = len(bundle.goals)

            first_reachable_goal = all_goals[first_idx] if first_idx >= 0 else None

            # Success bar: fractional (area-aware) if region_min_reachable_fraction > 0,
            # else the absolute region_success_min_reachable count. Fraction is computed
            # against the points ACTUALLY sampled in this region, floored at 1.
            if self._min_reachable_fraction > 0.0:
                min_needed = max(1, math.ceil(self._min_reachable_fraction * len(bundle.goals)))
            else:
                min_needed = self._success_min_reachable

            if reachable_count >= min_needed:
                return True, reachable_count, first_reachable_goal, all_goals
            else:
                return False, reachable_count, None, all_goals
        finally:
            self._record_opening_validation_timing(
                (time.perf_counter() - validation_start) * 1000.0,
                goal_checks=goal_checks,
                reachability_calls=reachability_calls,
                reachability_ms=reachability_ms_total,
            )

    def _search_with_ml_driven_async(
        self,
        object_id: str,
        baseline_state: namo_rl.RLState,
        neighbour_label: str,
        region_goals: Dict[str, Any],
        max_solutions_to_collect: Optional[int] = None,
        push_counter: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[Tuple[List[Goal], List, List, 'namo_rl.RLState', Optional[Tuple], Optional[List[Tuple]], List, List, int, Optional[int], float]], int]:
        """ML-driven async search: zero idle time, ML priority, N-push support.

        Uses MLDrivenAsyncSearch to find openings with:
        - Immediate fallback execution while ML runs in background
        - ML results jump the queue when ready
        - CPU never idles waiting for GPU

        Returns same format as _search_with_chaining_bfs for compatibility.
        """
        # Create search instance
        search = MLDrivenAsyncSearch(
            env=self.env,
            primitive_strategy=self._primitive_strategy,
            ml_strategy=self._ml_async_strategy,
            max_chain_depth=self.max_chain_depth,
            max_solutions=max_solutions_to_collect or self.max_solutions_per_neighbor,
            verbose=self.config.verbose,
            terminate_on_collision=self.terminate_on_collision,
        )

        # Create validation function (env param unused - uses self.env internally)
        def validate_fn(_env):
            return self._validate_opening(neighbour_label, region_goals)

        # Run search
        solutions = search.search(
            object_id=object_id,
            baseline_state=baseline_state,
            neighbor_label=neighbour_label,
            validate_opening_fn=validate_fn,
        )

        # Update push counter
        if push_counter is not None:
            push_counter["count"] += search.total_pushes

        # Convert solutions to expected format
        # Format: (goal_chain, state_obs, post_state_obs, resulting_state,
        #          region_goal_used, region_goals_sampled, reachable_before,
        #          reachable_after, total_cost, skill_calls, success_time,
        #          any_wall_collision, unique_movable_collision_count)
        results = []
        min_depth = None

        for sol in solutions:
            # Get validation info
            self.env.set_full_state(sol.resulting_state)
            is_open, reachable_count, region_goal, all_goals = self._validate_opening(
                neighbour_label, region_goals
            )

            # Build result tuple (13 elements to match _search_with_chaining_bfs)
            result = (
                sol.chain,                                  # goal_chain
                sol.state_observations,                     # state_obs
                sol.post_action_observations,               # post_state_obs
                sol.resulting_state,                        # resulting_state
                region_goal,                                # region_goal_used
                all_goals,                                  # region_goals_sampled
                [],                                         # reachable_before (not tracked)
                [],                                         # reachable_after (not tracked)
                sol.num_pushes,                             # total_cost
                None,                                       # skill_calls_before_success
                time.time(),                                # success_timestamp
                sol.any_wall_collision,                     # any_wall_collision
                sol.unique_movable_collision_count,         # unique_movable_collision_count
            )
            results.append(result)

            if min_depth is None or sol.num_pushes < min_depth:
                min_depth = sol.num_pushes

        return results, min_depth if min_depth else 0


# Register the planner with the factory
from namo.core import PlannerFactory
PlannerFactory.register_planner("region_opening", RegionOpeningPlanner)
