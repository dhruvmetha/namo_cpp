"""Region Opening Planner for NAMO.

This planner creates an opening from the robot's region to each immediate neighbour
region. For each neighbour, it picks a blocking object, samples push goals, executes,
validates the opening, logs an episode, then restores the baseline and proceeds to
the next neighbour.
"""

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
from namo.planners import snapshot_region_connectivity, find_robot_label
from namo.strategies import (
    PrimitiveGoalStrategy,
    Goal,
    MLPrimitiveGoalStrategy,
    MLPrimitiveFallbackStrategy,
    MLPrimitiveAsyncStrategy,
    AsyncGoalResult
)
from namo.planners.opening.ml_driven_search import MLDrivenAsyncSearch


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
    reachable_edges_count: int = 0  # Number of reachable edges for the object
    candidate_objects_count: int = 0  # Number of candidate blocking objects
    # Detailed ML goal info (for analysis/visualization)
    # Each entry: {'edge_idx': int, 'depth_idx': int, 'x': float, 'y': float, 'theta': float, 'votes': int}
    aligned_primitives: Optional[List[Dict]] = None
    # Raw ML goals before alignment: [{'x': float, 'y': float, 'theta': float}, ...]
    ml_goals_raw: Optional[List[Dict]] = None
    # List of reachable edge indices
    reachable_edges: Optional[List[int]] = None


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

        # Get max recorded solutions per neighbor (subset of found solutions to keep), default: 2
        self.max_recorded_solutions_per_neighbor = algo_params.get(
            "region_max_recorded_solutions_per_neighbor", 2
        )
        if self.max_recorded_solutions_per_neighbor < 1:
            raise ValueError(
                f"Invalid region_max_recorded_solutions_per_neighbor: {self.max_recorded_solutions_per_neighbor}. Must be at least 1"
            )

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

        # ML blacklist override: when True, ML-scored primitives bypass the
        # edge blacklist built during pre-ML exhaustive phase. This allows
        # ML suggestions to be tried even if earlier depth-first exploration
        # caused collisions on that edge. (default: False for backward compat)
        self.ml_ignore_blacklist = algo_params.get("region_ml_ignore_blacklist", False)

        # Chain link cost: additional cost added per chain link beyond the first push.
        # With chain_link_cost=0 (default), a 2-push chain at depths [0,0] costs 2.
        # With chain_link_cost=5, the same chain costs 2 + 5 = 7.
        self.chain_link_cost = algo_params.get("region_chain_link_cost", 0)

        # Visualization settings (can be set after init, like IDFS planners)
        self.visualize_search = False
        self.search_delay = 0.5
        self.step_mode = False

        # ML-driven async search flag (set in _initialize_algorithm)
        self._use_ml_driven_async = False
        self._primitive_strategy = None
        self._ml_async_strategy = None

        super().__init__(env, config)

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
        sampler_name = (
            algo_params.get("goal_sampler")
            or algo_params.get("goal_selection_strategy")
        )

        if sampler_name and sampler_name.lower() in {"ml", "ml_primitive"}:
            ml_path = algo_params.get("ml_goal_model_path")
            if not ml_path:
                raise ValueError("ML primitive goal sampler requires 'ml_goal_model_path'")

            self.goal_sampler = MLPrimitiveGoalStrategy(
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
            )
            self._debug("▶ Using ML-aligned primitive goal sampler")
        elif sampler_name and sampler_name.lower() in {"ml_fallback", "ml_primitive_fallback"}:
            ml_path = algo_params.get("ml_goal_model_path")
            if not ml_path:
                raise ValueError("ML fallback goal sampler requires 'ml_goal_model_path'")

            self.goal_sampler = MLPrimitiveFallbackStrategy(
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
            )
            self._debug("▶ Using ML-first with primitive fallback goal sampler")
        elif sampler_name and sampler_name.lower() in {"ml_async", "ml_primitive_async"}:
            ml_path = algo_params.get("ml_goal_model_path")
            if not ml_path:
                raise ValueError("ML async goal sampler requires 'ml_goal_model_path'")

            self.goal_sampler = MLPrimitiveAsyncStrategy(
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
            )
            self._debug("▶ Using async ML with primitive pre-execution goal sampler")
        elif sampler_name and sampler_name.lower() in {"ml_driven_async"}:
            # ML-Driven Async: uses MLDrivenAsyncSearch with zero idle time guarantee
            ml_path = algo_params.get("ml_goal_model_path")
            if not ml_path:
                raise ValueError("ML driven async requires 'ml_goal_model_path'")

            # Store strategies for MLDrivenAsyncSearch
            self._primitive_strategy = PrimitiveGoalStrategy(
                data_dir=primitive_data_dir,
                verbose=self.config.verbose
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
                max_workers=algo_params.get("ml_async_workers", 1),
            )
            # Set goal_sampler to primitive for compatibility (MLDrivenAsyncSearch handles ML internally)
            self.goal_sampler = self._primitive_strategy
            self._use_ml_driven_async = True
            self._debug("▶ Using ML-driven async search (zero idle time, ML priority)")
        else:
            # Use primitive goal strategy for push goals
            self.goal_sampler = PrimitiveGoalStrategy(
                data_dir=primitive_data_dir,
                verbose=self.config.verbose
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

    def _debug(self, message: str):
        if getattr(self.config, "verbose", False):
            print(message)

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

    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        """Execute region opening planner (single-level exploration from initial state only).

        This method explores region openings from the initial state only:
        - Find all possible openings from the initial environment configuration
        - No multi-level exploration (no queueing of resulting states)

        Args:
            robot_goal: Target robot position (x, y, theta) - stored but not directly used

        Returns:
            PlannerResult with all attempt results from initial state
        """
        start_time = time.time()
        self.attempt_results = []

        # Configure collision checking based on region_allow_collisions setting
        collision_checking_enabled = self.terminate_on_collision
        self.env.set_collision_checking(collision_checking_enabled)

        # Save baseline state
        baseline = self.env.get_full_state()

        if self.config.verbose:
            self._debug(f"\n{'='*60}")
            self._debug("Region Opening Planner - Single-Level Exploration")
            self._debug(f"Max chain depth: {self.max_chain_depth} | Collision checking: {'ON' if collision_checking_enabled else 'OFF'}")
            self._debug(f"{'='*60}\n")

        # Explore from initial state only (Level 0)
        self.attempt_results = self._explore_from_state(baseline, level=0)

        if self.config.verbose:
            successful_attempts = sum(1 for a in self.attempt_results if a.success)
            self._debug(f"\n{'='*60}")
            self._debug(f"Exploration Complete: {successful_attempts}/{len(self.attempt_results)} successful openings")
            self._debug(f"{'='*60}\n")

        # Calculate statistics
        total_time = (time.time() - start_time) * 1000  # ms
        successful_attempts = sum(1 for a in self.attempt_results if a.success)

        # Build action_sequence from the first successful attempt for visualization
        action_sequence = []
        all_solutions = []  # Store ALL successful attempts as separate action sequences

        for attempt in self.attempt_results:
            if not attempt.success:
                continue

            # Build action sequence for this attempt
            attempt_actions = []

            # Handle both single goal and goal chain
            if attempt.goal_chain and len(attempt.goal_chain) > 1:
                # Multi-push chain
                for goal in attempt.goal_chain:
                    action = namo_rl.Action()
                    action.object_id = attempt.chosen_object_id
                    action.x = goal.x
                    action.y = goal.y
                    action.theta = goal.theta
                    attempt_actions.append(action)
            elif attempt.chosen_goal:
                # Single push
                action = namo_rl.Action()
                action.object_id = attempt.chosen_object_id
                action.x = attempt.chosen_goal[0]
                action.y = attempt.chosen_goal[1]
                action.theta = attempt.chosen_goal[2]
                attempt_actions.append(action)

            all_solutions.append({
                "actions": attempt_actions,
                "neighbor": attempt.neighbour_region_label,
                "object": attempt.chosen_object_id
            })

            # Keep first success as primary action_sequence for backward compatibility
            if not action_sequence:
                action_sequence = attempt_actions

        # Return PlannerResult with action_sequence for visualization
        return PlannerResult(
            success=successful_attempts > 0,
            solution_found=successful_attempts > 0,
            action_sequence=action_sequence,
            solution_depth=len(action_sequence) if action_sequence else None,
            search_time_ms=total_time,
            algorithm_stats={
                "attempt_results": self.attempt_results,
                "all_solutions": all_solutions,  # ALL successful openings from initial state
                "successful_openings": successful_attempts,
                "total_attempts": len(self.attempt_results)
            }
        )

    def _explore_from_state(self, state: 'namo_rl.RLState', level: int = 0) -> List[AttemptResult]:
        """Explore region openings from a given state.

        This helper method:
        1. Sets environment to the given state
        2. Computes region connectivity snapshot
        3. Identifies robot region and neighbors
        4. Attempts to create openings to each neighbor (up to max_solutions_per_neighbor solutions)

        Args:
            state: Full environment state to explore from
            level: Exploration level (0 = initial state, 1+ = subsequent explorations)

        Returns:
            List of AttemptResults from exploring this state
        """
        print = self._debug

        # Set environment to exploration state
        self.env.set_full_state(state)

        # Get region connectivity and goals from snapshot
        # use_current_state=True ensures snapshot uses current object positions (not initial XML state)
        # and always uses XML goal (not whatever was last set via set_robot_goal during validation)
        xml_path = self.env.get_xml_path()
        config_path = self.env.get_config_path()
        adjacency, edge_objects, region_labels, region_goals, _ = snapshot_region_connectivity(
            self.env,
            xml_path,
            config_path,
            include_snapshot=False,
            local_info_only=True,
            goals_per_region=self.config.goals_per_region,
            generate_training_data=True,
            use_current_state=True,
        )

        # Identify robot region
        robot_label = find_robot_label(region_labels)
        if robot_label is None:
            if self.config.verbose:
                print(f"  ⚠ Could not identify robot region")
            return []

        # Get neighbours
        neighbours = sorted(list(adjacency.get(robot_label, set())))

        if self.config.verbose:
            # Print region snapshot details
            total_regions = len(region_labels)
            total_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
            has_goal_region = "goal" in region_labels.values()
            goal_info = " (includes GOAL region)" if has_goal_region else ""
            print(f"  Region snapshot: robot={robot_label} | regions={total_regions}{goal_info} | edges={total_edges} | neighbors={len(neighbours)}")
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

        # Get candidate objects blocking the edge
        candidates = list(edge_objects.get(robot_label, {}).get(neighbour_label, set()))

        if not candidates:
            if self.config.verbose:
                print(f"    ✗ '{neighbour_label}' - no blocking objects found")
            return [AttemptResult(
                success=False,
                neighbour_region_label=neighbour_label,
                error_message="No blocking objects found",
                timing_ms=(time.time() - attempt_start) * 1000,
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
                failure_reason="no_reachable_objects",
                candidate_objects_count=original_candidates_count,
            )]

        # Print what we're attempting
        if self.config.verbose:
            total_goals = len(region_goals[neighbour_label].goals) if neighbour_label in region_goals else 0
            print(f"    → '{neighbour_label}' ({reachable_count_before}/{total_goals} reachable) - trying {len(candidates)} objects: {candidates}")

        # Collect attempts from candidate objects, capped per-neighbour
        all_goal_attempts = []
        solutions_remaining = self.max_solutions_per_neighbor
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
        timed_out = False
        for obj_idx, object_id in enumerate(candidates, 1):
            if solutions_remaining <= 0:
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

            # BFS search with chaining (or ML-driven async if enabled)
            object_attempt_start = time.time()

            if self._use_ml_driven_async:
                # Use ML-driven async search
                successful_goals, min_depth = self._search_with_ml_driven_async(
                    object_id,
                    exploration_state,
                    neighbour_label,
                    region_goals,
                    max_solutions_to_collect=solutions_remaining,
                    push_counter=neighbour_push_counter,
                )
            else:
                # Use standard BFS search
                successful_goals, min_depth = self._search_with_chaining_bfs(
                    object_id,
                    exploration_state,
                    neighbour_label,
                    region_goals,
                    max_solutions_to_collect=solutions_remaining,
                    push_counter=neighbour_push_counter,
                )

            # Accumulate ML goal stats from goal_sampler (if it supports get_last_goal_stats)
            if hasattr(self.goal_sampler, 'get_last_goal_stats'):
                stats = self.goal_sampler.get_last_goal_stats()
                total_ml_goals_generated += stats.get('ml_goals_generated', 0)
                total_ml_goals_aligned += stats.get('ml_goals_aligned', 0)
                total_reachable_edges += stats.get('reachable_edges_count', 0)
                # Accumulate detailed info (tag with object_id for multi-object analysis)
                for p in stats.get('aligned_primitives', []):
                    p_with_obj = dict(p)
                    p_with_obj['object_id'] = object_id
                    all_aligned_primitives.append(p_with_obj)
                for g in stats.get('ml_goals_raw', []):
                    g_with_obj = dict(g)
                    g_with_obj['object_id'] = object_id
                    all_ml_goals_raw.append(g_with_obj)
                all_reachable_edges.update(stats.get('reachable_edges', []))

            if successful_goals:
                if self.config.verbose:
                    print(f"      ✓ {object_id}: Found {len(successful_goals[:max_solutions])} solutions (depth={min_depth})")

                # Create AttemptResults directly from successful goal chains
                # State observations were already captured during BFS search

                # Limit to max_solutions per object
                # Respect global per-neighbour cap
                per_object_limit = min(max_solutions, solutions_remaining)
                for goal_idx, (goal_chain, state_obs, post_state_obs, resulting_state, region_goal_used, region_goals_sampled, reachable_before, reachable_after, total_cost, skill_calls_before_success, success_timestamp) in enumerate(successful_goals[:per_object_limit]):

                    per_goal_timing_ms = max(0.0, (success_timestamp - object_attempt_start) * 1000.0)

                    # Create AttemptResult
                    if len(goal_chain) == 1:
                        # Single push
                        goal = goal_chain[0]
                        total_solutions_collected += 1
                        all_goal_attempts.append(AttemptResult(
                            success=True,
                            neighbour_region_label=neighbour_label,
                            chosen_object_id=object_id,
                            chosen_goal=(goal.x, goal.y, goal.theta),
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
                            total_cost=total_cost,
                            skill_calls_before_success=skill_calls_before_success,
                            solutions_found_for_neighbour=total_solutions_collected,
                            solutions_cap_for_neighbour=self.max_solutions_per_neighbor,
                            failure_reason="success",
                            candidate_objects_count=len(candidates),
                            ml_goals_generated=total_ml_goals_generated,
                            ml_goals_aligned=total_ml_goals_aligned,
                            reachable_edges_count=total_reachable_edges,
                            aligned_primitives=all_aligned_primitives if all_aligned_primitives else None,
                            ml_goals_raw=all_ml_goals_raw if all_ml_goals_raw else None,
                            reachable_edges=sorted(list(all_reachable_edges)) if all_reachable_edges else None,
                        ))
                        # Verbose: print running count of solutions for this neighbour
                        if self.config.verbose:
                            print(f"        → Solutions so far: {len(all_goal_attempts)}/{self.max_solutions_per_neighbor}")
                        solutions_remaining -= 1
                        if solutions_remaining <= 0:
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
                            total_cost=total_cost,
                            skill_calls_before_success=skill_calls_before_success,
                            solutions_found_for_neighbour=total_solutions_collected,
                            solutions_cap_for_neighbour=self.max_solutions_per_neighbor,
                            failure_reason="success",
                            candidate_objects_count=len(candidates),
                            ml_goals_generated=total_ml_goals_generated,
                            ml_goals_aligned=total_ml_goals_aligned,
                            reachable_edges_count=total_reachable_edges,
                            aligned_primitives=all_aligned_primitives if all_aligned_primitives else None,
                            ml_goals_raw=all_ml_goals_raw if all_ml_goals_raw else None,
                            reachable_edges=sorted(list(all_reachable_edges)) if all_reachable_edges else None,
                        ))
                        # Verbose: print running count of solutions for this neighbour
                        if self.config.verbose:
                            print(f"        → Solutions so far: {len(all_goal_attempts)}/{self.max_solutions_per_neighbor}")
                        solutions_remaining -= 1
                        if solutions_remaining <= 0:
                            break

        # After trying all objects, return results
        if all_goal_attempts:
            # Record the total number of successful solutions and pushes for this neighbour
            for attempt in all_goal_attempts:
                attempt.solutions_total_for_neighbour = total_solutions_collected
                attempt.pushes_total_for_neighbour = neighbour_push_counter.get("count", 0)

            # Keep at most configured number of solutions for this neighbour (min-cost by design)
            if len(all_goal_attempts) > self.max_recorded_solutions_per_neighbor:
                all_goal_attempts = all_goal_attempts[: self.max_recorded_solutions_per_neighbor]

            # Update solutions_found_for_neighbour to reflect final recorded count (after truncation)
            final_recorded_count = len(all_goal_attempts)
            for attempt in all_goal_attempts:
                attempt.solutions_found_for_neighbour = final_recorded_count

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
    ) -> Tuple[List, List, List, List]:
        """Execute a goal chain and collect state observations for each push.

        Args:
            object_id: Object being pushed
            goal_chain: List of goals to execute in sequence
            baseline_state: Starting state

        Returns:
            Tuple of (state_observations, post_action_state_observations,
                     reachable_before, reachable_after)
        """
        self.env.set_full_state(baseline_state)
        state_obs = []
        post_state_obs = []
        reachable_before = []
        reachable_after = []

        for goal in goal_chain:
            # Capture state and reachable objects before action
            pre_obs = self.env.get_observation()
            pre_reachable = self.env.get_reachable_objects()
            state_obs.append(pre_obs)
            reachable_before.append(pre_reachable)

            # Execute action
            action = namo_rl.Action()
            action.object_id = object_id
            action.x = goal.x
            action.y = goal.y
            action.theta = goal.theta
            self.env.step(action)

            # Capture state and reachable objects after action
            post_obs = self.env.get_observation()
            post_reachable = self.env.get_reachable_objects()
            post_state_obs.append(post_obs)
            reachable_after.append(post_reachable)

        return state_obs, post_state_obs, reachable_before, reachable_after

    def _search_with_chaining_bfs(
        self,
        object_id: str,
        baseline_state: namo_rl.RLState,
        neighbour_label: str,
        region_goals: Dict[str, Any],
        max_solutions_to_collect: Optional[int] = None,
        push_counter: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[Tuple[List[Goal], List, List, 'namo_rl.RLState', Optional[Tuple], Optional[List[Tuple]], List, List, int, Optional[int], float]], int]:
        """Outer BFS over chain depth: Try single pushes, then 2-push chains, then 3-push chains.

        Collects ALL successful chains across all depths instead of stopping early.

        Returns:
            Tuple of (all_chains, min_depth) where all_chains is a list of tuples:
            (goal_chain, state_obs, post_state_obs, resulting_state, region_goal_used,
             region_goals_sampled, reachable_before, reachable_after, total_cost,
             skill_calls_before_success, success_time). Returns ([], 0) if no solution found.
        """
        print = self._debug
        # Initial frontier for chain depth 1
        root_node = ChainNode(
            state=baseline_state,
            goal=None,
            edge_idx=-1,
            depth=0,
            parent=None,
            step_cost=0
        )
        frontier = [root_node]

        # Collect all successful chains across all depths
        all_chains_across_depths = []
        min_chain_depth_found = None

        # Track best cumulative cost found so far; use for pruning
        best_total_cost = None
        skill_call_counter = {"count": 0}

        # Try chain depths 1, 2, 3, ...
        for chain_depth in range(1, self.max_chain_depth + 1):
            next_frontier = []
            processed_frontiers = 0
            total_frontier_time_ms = 0.0
            orig_frontier_len = len(frontier)
            reached_cap = False

            # Verbose: indicate which chain-depth search level we are at
            if self.config.verbose:
                chain_label = f"{chain_depth}-chain"
                print(f"    ▶ Searching {chain_label} (frontier={len(frontier)})")

            for node in frontier:
                node_start_time = time.time()
                # Cost-based node prune: if cost so far already meets/exceeds best, skip
                if best_total_cost is not None:
                    chain_cost_so_far = self._compute_chain_cost(node)
                    if chain_cost_so_far >= best_total_cost:
                        continue
                # Restore to this node's state
                self.env.set_full_state(node.state)

                # Generate goals for this node's state
                print(f"      🔮 [_search_with_chaining_bfs] Chain depth {chain_depth}, node {frontier.index(node)+1}/{len(frontier)}: Generating goals for {object_id}")
                goals_per_edge = self.goal_sampler.generate_goals(
                    object_id,
                    node.state,
                    self.env,
                    max_goals=0
                )

                if not goals_per_edge:
                    print(f"      ⚠️ No goals generated, skipping node")
                    continue

                # Get reachable edges from this state
                reachable_edge_indices = set(self.env.get_reachable_edges(object_id))

                print(f"      📍 Reachable edges: {sorted(list(reachable_edge_indices))} (total: {len(reachable_edge_indices)})")

                if not reachable_edge_indices:
                    print(f"      ⚠️ No reachable edges, skipping node")
                    continue

                # Run inner BFS (single-skill search)
                collect_frontier = (chain_depth < self.max_chain_depth)
                # Compute remaining budget if we already have a best cost
                if 'best_total_cost' in locals() and best_total_cost is not None:
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
                if max_solutions_to_collect is not None:
                    if current_solutions_count >= max_solutions_to_collect:
                        reached_cap = True
                        break
                    inner_max_solutions = max_solutions_to_collect - current_solutions_count

                successful_results, primitive_depth, new_frontier_nodes = self._search_bfs(
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
                    max_solutions_to_collect=inner_max_solutions
                )

                # If we found success, reconstruct ALL goal chains with their state observations
                if successful_results:
                    for (final_goal, final_state_obs, final_post_state_obs, resulting_state, region_goal_used, all_region_goals, success_node, success_time) in successful_results:
                        # For multi-push chains, reconstruct full chain with observations
                        if chain_depth > 1:
                            goal_chain, state_obs, post_state_obs, reachable_before, reachable_after, total_cost = self._reconstruct_chain_with_observations(
                                success_node, object_id, baseline_state
                            )
                        else:
                            # Single push - use observations captured during search
                            goal_chain = [final_goal]
                            state_obs = final_state_obs
                            post_state_obs = final_post_state_obs
                            # For single push, we don't have reachable objects captured during BFS
                            # So collect them now
                            self.env.set_full_state(baseline_state)
                            reachable_before = [self.env.get_reachable_objects()]
                            # Execute the action to get reachable after
                            action = namo_rl.Action()
                            action.object_id = object_id
                            action.x = final_goal.x
                            action.y = final_goal.y
                            action.theta = final_goal.theta
                            self.env.step(action)
                            reachable_after = [self.env.get_reachable_objects()]
                            # For single push, total_cost equals the primitive depth at which success occurred
                            total_cost = max(1, getattr(success_node, "step_cost", 1))

                        skill_calls_before_success = getattr(success_node, "skill_calls_before_success", None)

                        # Verbose: print each solution found at this chain depth
                        if self.config.verbose:
                            print(f"      ✓ Found solution at {chain_depth}-chain (total_cost={total_cost})")

                        # Entry layout: (goal_chain, state_obs, post_state_obs, resulting_state, region_goal_used, region_goals_sampled, reachable_before, reachable_after, total_cost, skill_calls, success_time)
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
                                )
                            )
                            if self.config.verbose:
                                # Running count of min-cost solutions so far (object scope)
                                print(f"        → Solutions so far (object, best_cost={best_total_cost}): {len(all_chains_across_depths)}")

                            # Early stop if we reached the per-object cap
                            if max_solutions_to_collect is not None and len(all_chains_across_depths) >= max_solutions_to_collect:
                                reached_cap = True
                                break

                    # Track minimum chain depth where we found a solution
                    if min_chain_depth_found is None:
                        min_chain_depth_found = chain_depth

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

            # Apply beam width pruning on next_frontier if configured
            if self.frontier_beam_width is not None:
                # Sort by cumulative chain cost (ascending), then by step_cost (ascending)
                next_frontier.sort(key=lambda n: (self._compute_chain_cost(n), getattr(n, "step_cost", 0)))
                # Keep top-K
                frontier = next_frontier[: self.frontier_beam_width]
            else:
                # Move to next chain depth
                frontier = next_frontier

            if reached_cap:
                break

            # After this chain depth, report completion stats
            if self.config.verbose and processed_frontiers > 0:
                avg_ms = total_frontier_time_ms / processed_frontiers
                print(f"    ◼ Completed {processed_frontiers}/{orig_frontier_len} frontiers | avg {avg_ms:.1f} ms | total {total_frontier_time_ms:.1f} ms")

            if not frontier:
                break

        # If we found any chains, filter to keep only minimum-cost ones
        if all_chains_across_depths:
            best_cost = min(entry[8] for entry in all_chains_across_depths)
            min_cost_chains = [entry for entry in all_chains_across_depths if entry[8] == best_cost]
            if self.config.verbose:
                print(f"    ✔ Returning {len(min_cost_chains)} min-cost solution(s) with cost={best_cost}")
            return min_cost_chains, min_chain_depth_found if min_chain_depth_found else 0
        else:
            return all_chains_across_depths, 0

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
    ) -> Tuple[List[Goal], List, List, List, List, int]:
        """Reconstruct goal chain and collect observations by re-executing.

        This is only called for multi-push chains (chain_depth > 1).

        Args:
            success_node: Final ChainNode containing parent chain
            object_id: Object being pushed
            baseline_state: Starting state for re-execution

        Returns:
            Tuple of (goal_chain, state_obs, post_state_obs, reachable_before, reachable_after)
        """
        # Reconstruct goal chain from parent nodes
        goal_chain = []
        node = success_node
        while node.parent is not None:
            goal_chain.append(node.goal)
            node = node.parent
        goal_chain.reverse()

        # Re-execute chain to collect observations
        state_obs, post_state_obs, reachable_before, reachable_after = self._collect_chain_observations(
            object_id, goal_chain, baseline_state
        )

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

        return goal_chain, state_obs, post_state_obs, reachable_before, reachable_after, total_cost

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
    ) -> Tuple[List[Tuple[Goal, List, List, 'namo_rl.RLState', Optional[Tuple], ChainNode, float]], int, List[ChainNode]]:
        """BFS: Try all edges at ALL depths to collect all possible solutions.

        Supports async ML inference: if goals_or_async is an AsyncGoalResult,
        primitives start executing immediately while ML runs in background.
        When ML completes, remaining candidates are re-sorted by ML scores.

        Args:
            goals_or_async: Either List[List[Goal]] (sync) or AsyncGoalResult (async).
            collect_frontier: If True, collect valid but unsuccessful states as frontier nodes
            max_solutions_to_collect: If provided, stop searching once this many successful solutions are found.

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
        edge_min_stuck_depth: Dict[int, int] = {}  # edge_idx -> min depth that got stuck

        # Track edges that have already yielded a successful opening in THIS skill execution
        # Once an edge succeeds at any primitive depth, we do not explore deeper depths on that edge
        solved_edges_this_skill = set()

        # Frontier nodes for chaining
        frontier_nodes = []

        # Collect ALL successful results across all depths
        all_successful_results = []
        min_depth_found = None

        # Track async ML merge state
        ml_merged = False
        ml_scores: Dict[Tuple[int, int], float] = {}
        ml_scored_slots: Set[Tuple[int, int]] = set()  # Track which (edge, depth) have ML scores


        # Flatten goals into candidates for prioritized iteration
        # Use list of lists to allow mutation during re-sort
        candidates = []
        for edge_idx, edge_goals in enumerate(goals_per_edge):
            # Filter: only try reachable edges
            if edge_idx not in reachable_edge_indices:
                continue

            for depth, goal in enumerate(edge_goals):
                if goal is not None:
                    candidates.append([edge_idx, depth, goal])  # Use list for mutability

        # Initial sort depends on whether we have async ML
        if async_result is not None and async_result.ml_future is not None:
            # Async mode: sort by (depth, edge) initially - shortest pushes first while waiting for ML
            candidates.sort(key=lambda x: (x[1], x[0]))
            if self.config.verbose:
                print(f"      📋 Async mode: {len(candidates)} candidates sorted by depth (ML running in background)")
        else:
            # Sync mode: sort by (-score, depth) - high votes first, then shortest
            candidates.sort(key=lambda x: (-getattr(x[2], 'score', 0.0), x[1]))

        # Track position for re-sorting remaining candidates
        candidate_idx = 0

        while candidate_idx < len(candidates):
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

                # Re-sort remaining candidates: score DESC, depth ASC
                remaining = candidates[candidate_idx:]
                remaining.sort(key=lambda x: (-getattr(x[2], 'score', 0.0), x[1]))
                candidates[candidate_idx:] = remaining

                if self.config.verbose:
                    # Show top candidates after re-sort
                    top_5 = candidates[candidate_idx:candidate_idx+5]
                    top_info = [(c[0], c[1], getattr(c[2], 'score', 0.0)) for c in top_5]
                    print(f"      📊 Re-sorted remaining {len(remaining)} candidates. Top 5: {top_info}")

            # Get current candidate
            edge_idx, depth, goal = candidates[candidate_idx]
            candidate_idx += 1

            # Stop if we've reached the solution cap
            if max_solutions_to_collect is not None and len(all_successful_results) >= max_solutions_to_collect:
                if self.config.verbose:
                    print(f"        🛑 Reached max solutions ({len(all_successful_results)}/{max_solutions_to_collect}), stopping search")
                break
            # Global prune: once any success is found at depth D, skip candidates with depth > D
            if min_depth_found is not None and depth > min_depth_found:
                continue

            # Budget prune
            if remaining_budget is not None and (depth + 1) > remaining_budget:
                continue

            # Filter: skip if this edge got stuck/collided at a shallower or equal depth
            # (shallower depths than the stuck depth are still worth trying)
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
                    continue

            # Filter: skip edges that have already produced a successful opening
            if edge_idx in solved_edges_this_skill:
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
                goal_type = "ML-aligned" if goal is not None else "empty"
                print(f"      Testing edge {edge_idx} depth {depth+1} ({goal_type}): {neighbour_label} ({reachable_count_before}/{total_region_goals} reachable before)")

            # Execute push
            action = namo_rl.Action()
            action.object_id = object_id
            action.x = goal.x
            action.y = goal.y
            action.theta = goal.theta

            if self.config.verbose:
                print(f"        🚀 EXECUTING PUSH edge {edge_idx} depth {depth+1}:")
                print(f"           object_id={object_id}, goal=({goal.x:.3f}, {goal.y:.3f}, {goal.theta:.3f})")

            if skill_call_counter is not None:
                skill_call_counter["count"] += 1
            if push_counter is not None:
                push_counter["count"] += 1

            try:
                if self.config.verbose:
                    print(f"        ⏳ Calling env.step()...")
                step_result = self.env.step(action)
                if self.config.verbose:
                    print(f"        ✓ env.step() returned successfully")

                # Visualize after action if enabled
                self._focus_camera_on_object(object_id)

            except Exception as e:
                if self.config.verbose:
                    print(f"        ❌ EXCEPTION during env.step(): {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                continue

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

            total_region_goals = len(region_goals[neighbour_label].goals) if neighbour_label in region_goals else 0
            if is_accessible_after and not is_accessible_before:
                print(f"      ✅ SUCCESS! {object_id} edge {edge_idx} depth {depth+1}: {reachable_count_before}/{total_region_goals} → {reachable_count_after}/{total_region_goals} reachable")
            elif depth == 0 and goal is not None:  # Show failures only for first depth and only ML-aligned goals
                if self.config.verbose:
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
        return all_successful_results, min_depth_found if min_depth_found else 0, frontier_nodes

    def _validate_opening(
        self,
        neighbour_label: str,
        region_goals: Dict[str, Any]
    ) -> Tuple[bool, int, Optional[Tuple[float, float, float]], Optional[List[Tuple[float, float, float]]]]:
        """Validate that opening to neighbour was created using reachability.

        Success criterion: At least half of the region goals must be reachable.

        Args:
            neighbour_label: Target neighbour region
            region_goals: Region goal samples from snapshot

        Returns:
            Tuple of (success, reachable_count, first_reachable_goal, all_region_goals):
                - success: True if at least half of region goals are reachable
                - reachable_count: Number of reachable goals
                - first_reachable_goal: First reachable goal found (for validation)
                - all_region_goals: All goal samples for this region (for visualization)
        """
        # Get region goals for this neighbour
        if neighbour_label not in region_goals:
            return False, 0, None, None

        bundle = region_goals[neighbour_label]
        if not bundle.goals:
            return False, 0, None, None

        total_goals = len(bundle.goals)
        reachable_count = 0
        first_reachable_goal = None

        # Collect all goal samples for visualization
        all_goals = [(g.x, g.y, g.theta) for g in bundle.goals]

        # Check ALL goals and count how many are reachable
        for goal_sample in bundle.goals:
            self.env.set_robot_goal(goal_sample.x, goal_sample.y, goal_sample.theta)
            if self.env.is_robot_goal_reachable():
                reachable_count += 1
                if first_reachable_goal is None:
                    first_reachable_goal = (goal_sample.x, goal_sample.y, goal_sample.theta)

        # Success if at least half of the goals are reachable
        required_count = (total_goals + 1) // 2  # Ceiling division
        if reachable_count >= required_count:
            return True, reachable_count, first_reachable_goal, all_goals
        else:
            return False, reachable_count, None, all_goals

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
        #          reachable_after, total_cost, skill_calls, success_time)
        results = []
        min_depth = None

        for sol in solutions:
            # Get validation info
            self.env.set_full_state(sol.resulting_state)
            is_open, reachable_count, region_goal, all_goals = self._validate_opening(
                neighbour_label, region_goals
            )

            # Build result tuple
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
            )
            results.append(result)

            if min_depth is None or sol.num_pushes < min_depth:
                min_depth = sol.num_pushes

        return results, min_depth if min_depth else 0


# Register the planner with the factory
from namo.core import PlannerFactory
PlannerFactory.register_planner("region_opening", RegionOpeningPlanner)
