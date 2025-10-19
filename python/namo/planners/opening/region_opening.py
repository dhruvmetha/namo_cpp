"""Region Opening Planner for NAMO.

This planner creates an opening from the robot's region to each immediate neighbour
region. For each neighbour, it picks a blocking object, samples push goals, executes,
validates the opening, logs an episode, then restores the baseline and proceeds to
the next neighbour.
"""

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Set

import namo_rl
from namo.core import BasePlanner, PlannerConfig, PlannerResult
from namo.planners import snapshot_region_connectivity, find_robot_label
from namo.strategies import PrimitiveGoalStrategy, Goal


@dataclass
class ChainNode:
    """Node in the skill chaining search tree."""
    state: namo_rl.RLState  # Environment state after this push
    goal: Goal  # Goal that led to this state
    edge_idx: int  # Edge index used
    depth: int  # Chain depth (1, 2, or 3)
    parent: Optional['ChainNode'] = None  # Parent node in chain
    collided_edges: Set[int] = field(default_factory=set)  # Edges that collided at this state


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
    region_goal_used: Optional[Tuple[float, float, float]] = None
    error_message: Optional[str] = None
    actions_executed: List[namo_rl.Action] = field(default_factory=list)
    state_observations: Optional[List[Dict[str, List[float]]]] = None  # State before each action
    post_action_state_observations: Optional[List[Dict[str, List[float]]]] = None  # State after each action
    exploration_state: Optional['namo_rl.RLState'] = None  # State we were exploring from when this opening was found
    resulting_state: Optional['namo_rl.RLState'] = None  # Full state after executing this opening (for multi-level exploration)
    exploration_level: int = 0  # Which exploration level this opening was found at (0 = initial state)
    timing_ms: Optional[float] = None


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

        # Get collision termination flag from config.algorithm_params
        # region_allow_collisions=True means ALLOW collisions (don't terminate)
        # We invert it: terminate_on_collision=True means TERMINATE on collision
        allow_collisions = config.algorithm_params.get("region_allow_collisions", False)
        self.terminate_on_collision = not allow_collisions

        # Get max chain depth from config.algorithm_params (default: 1, no chaining)
        self.max_chain_depth = config.algorithm_params.get("region_max_chain_depth", 1)
        if self.max_chain_depth < 1 or self.max_chain_depth > 10:
            raise ValueError(f"Invalid max_chain_depth: {self.max_chain_depth}. Must be between 1 and 10")

        # Get max solutions per neighbor from config.algorithm_params (default: 2)
        self.max_solutions_per_neighbor = config.algorithm_params.get("region_max_solutions_per_neighbor", 2)
        if self.max_solutions_per_neighbor < 1:
            raise ValueError(f"Invalid max_solutions_per_neighbor: {self.max_solutions_per_neighbor}. Must be at least 1")

        # Get max explorations (total states queued for multi-level exploration) (default: 20)
        self.max_explorations = config.algorithm_params.get("region_max_explorations", 20)
        if self.max_explorations < 1:
            raise ValueError(f"Invalid max_explorations: {self.max_explorations}. Must be at least 1")

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

        # Use primitive goal strategy for push goals
        self.goal_sampler = PrimitiveGoalStrategy(
            data_dir="data",
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

    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        """Execute multi-level region opening planner with BFS exploration.

        This method explores region openings across multiple levels:
        - Level 0: Find openings from initial state
        - Level 1+: For each successful opening, explore further openings from resulting state
        - Continue until no new openings are found

        Args:
            robot_goal: Target robot position (x, y, theta) - stored but not directly used

        Returns:
            PlannerResult with all attempt results across all exploration levels
        """
        start_time = time.time()
        self.attempt_results = []

        # Configure collision checking based on region_allow_collisions setting
        collision_checking_enabled = self.terminate_on_collision
        self.env.set_collision_checking(collision_checking_enabled)

        # Save baseline state
        baseline = self.env.get_full_state()

        # Multi-level BFS exploration queue: list of (state, level) tuples
        exploration_queue = [(baseline, 0)]
        level_statistics = {}  # Track statistics per level
        total_explorations_queued = 0  # Track total states added to queue (excluding initial state)

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Region Opening Planner - Multi-Level BFS Exploration")
            print(f"Max explorations: {self.max_explorations} | Max chain depth: {self.max_chain_depth} | Collision checking: {'ON' if collision_checking_enabled else 'OFF'}")
            print(f"{'='*60}\n")

        # BFS loop: explore states level-by-level
        while exploration_queue:
            current_state, current_level = exploration_queue.pop(0)

            if self.config.verbose:
                print(f"\n[Level {current_level}] Exploring state (queue: {len(exploration_queue)} remaining)")

            # Explore from this state
            level_attempts = self._explore_from_state(current_state, current_level)

            # Track statistics for this level
            if current_level not in level_statistics:
                level_statistics[current_level] = {'total_attempts': 0, 'successful_openings': 0}
            level_statistics[current_level]['total_attempts'] += len(level_attempts)

            # Process results and queue successful openings for next level
            for attempt in level_attempts:
                self.attempt_results.append(attempt)

                if attempt.success and attempt.resulting_state is not None:
                    # Check if we've reached the exploration limit
                    if total_explorations_queued < self.max_explorations:
                        # Queue this resulting state for exploration at next level
                        exploration_queue.append((attempt.resulting_state, current_level + 1))
                        total_explorations_queued += 1
                        level_statistics[current_level]['successful_openings'] += 1

                        if self.config.verbose:
                            chain_info = f"chain={len(attempt.goal_chain)}" if attempt.goal_chain else "single"
                            print(f"  ✓ Queued '{attempt.neighbour_region_label}' opening (obj={attempt.chosen_object_id}, {chain_info}) → Level {current_level + 1} [Total queued: {total_explorations_queued}/{self.max_explorations}]")
                    else:
                        if self.config.verbose:
                            print(f"  ⊗ Skipped '{attempt.neighbour_region_label}' (exploration limit reached)")

            if self.config.verbose:
                successful = level_statistics[current_level]['successful_openings']
                total = level_statistics[current_level]['total_attempts']
                print(f"  Level {current_level} complete: {successful}/{total} successful")

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Exploration Complete | Levels: {len(level_statistics)} | Queued states: {total_explorations_queued}/{self.max_explorations}")
            total_successful = sum(stats['successful_openings'] for stats in level_statistics.values())
            total_attempts = sum(stats['total_attempts'] for stats in level_statistics.values())
            print(f"Overall: {total_successful} successful openings from {total_attempts} attempts")
            print(f"{'='*60}\n")

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
                "all_solutions": all_solutions,  # ALL successful openings across all levels
                "successful_openings": successful_attempts,
                "total_levels_explored": len(level_statistics),
                "level_statistics": level_statistics,
                "total_explorations_queued": total_explorations_queued,
                "max_explorations": self.max_explorations,
                "exploration_limit_reached": total_explorations_queued >= self.max_explorations
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
            goals_per_region=8,
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

        # Ensure environment is in correct state before pre-check
        self.env.set_full_state(exploration_state)

        # Pre-check: Is this neighbor already accessible?
        is_already_accessible, reachable_count_before, _ = self._validate_opening(
            neighbour_label,
            region_goals
        )

        if is_already_accessible:
            # Neighbor is already accessible - no need to push anything!
            if self.config.verbose:
                total_goals = len(region_goals[neighbour_label].goals) if neighbour_label in region_goals else 0
                region_type = "(GOAL REGION)" if neighbour_label == "goal" else ""
                print(f"    ⊙ '{neighbour_label}' already accessible {region_type} ({reachable_count_before}/{total_goals} reachable) - skipping")

            # Return empty list - NO attempt recorded
            return []

        # Get candidate objects blocking the edge
        candidates = list(edge_objects.get(robot_label, {}).get(neighbour_label, set()))

        if not candidates:
            if self.config.verbose:
                print(f"    ✗ '{neighbour_label}' - no blocking objects found")
            return [AttemptResult(
                success=False,
                neighbour_region_label=neighbour_label,
                error_message="No blocking objects found",
                timing_ms=(time.time() - attempt_start) * 1000
            )]

        # Intersect with reachable objects (try up to 5 objects)
        reachable = set(self.env.get_reachable_objects())
        candidates = [obj for obj in candidates if obj in reachable][:5]

        if not candidates:
            if self.config.verbose:
                print(f"    ✗ '{neighbour_label}' - no reachable blocking objects")
            return [AttemptResult(
                success=False,
                neighbour_region_label=neighbour_label,
                error_message="No reachable blocking objects",
                timing_ms=(time.time() - attempt_start) * 1000
            )]

        # Print what we're attempting
        if self.config.verbose:
            total_goals = len(region_goals[neighbour_label].goals) if neighbour_label in region_goals else 0
            print(f"    → '{neighbour_label}' ({reachable_count_before}/{total_goals} reachable) - trying {len(candidates)} objects: {candidates}")

        # Snapshot connectivity before (for validation)
        conn_before = {"adjacency": dict(adjacency), "robot_label": robot_label}

        # Collect attempts from ALL candidate objects (not just first successful one)
        all_goal_attempts = []

        # Try each candidate object with BFS search (already filtered for reachability)
        for obj_idx, object_id in enumerate(candidates, 1):
            # CRITICAL: Reset to exploration_state before trying each object
            # This ensures each object is tried from the same starting configuration
            self.env.set_full_state(exploration_state)

            # BFS search for minimum-depth goals (pass exploration_state directly)
            successful_goals, min_depth = self._search_minimum_depth_goals(
                object_id,
                exploration_state,
                neighbour_label,
                region_goals
            )

            if successful_goals:
                if self.config.verbose:
                    print(f"      ✓ {object_id}: Found {len(successful_goals[:max_solutions])} solutions (depth={min_depth})")

                # Create AttemptResults directly from successful goal chains
                # State observations were already captured during BFS search

                # Limit to max_solutions per object
                for goal_idx, (goal_chain, state_obs, post_state_obs, resulting_state, region_goal_used) in enumerate(successful_goals[:max_solutions]):

                    # Create AttemptResult
                    if len(goal_chain) == 1:
                        # Single push
                        goal = goal_chain[0]
                        all_goal_attempts.append(AttemptResult(
                            success=True,
                            neighbour_region_label=neighbour_label,
                            chosen_object_id=object_id,
                            chosen_goal=(goal.x, goal.y, goal.theta),
                            validation_method="reachability_validated",
                            connectivity_before=conn_before,
                            connectivity_after=None,
                            region_goal_used=region_goal_used,
                            actions_executed=[],
                            state_observations=state_obs,
                            post_action_state_observations=post_state_obs,
                            exploration_state=exploration_state,
                            resulting_state=resulting_state,
                            exploration_level=exploration_level,
                            timing_ms=(time.time() - attempt_start) * 1000
                        ))
                    else:
                        # Multi-push chain
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
                            actions_executed=[],
                            state_observations=state_obs,
                            post_action_state_observations=post_state_obs,
                            exploration_state=exploration_state,
                            resulting_state=resulting_state,
                            exploration_level=exploration_level,
                            timing_ms=(time.time() - attempt_start) * 1000
                        ))

        # After trying all objects, return results
        if all_goal_attempts:
            return all_goal_attempts
        else:
            # No successful opening found from any object
            if self.config.verbose:
                print(f"      ✗ No solutions found from {len(candidates)} objects")
            return [AttemptResult(
                success=False,
                neighbour_region_label=neighbour_label,
                error_message=f"Tried {len(candidates)} objects, none succeeded",
                connectivity_before=conn_before,
                timing_ms=(time.time() - attempt_start) * 1000
            )]

    def _collect_chain_observations(
        self,
        object_id: str,
        goal_chain: List[Goal],
        baseline_state: namo_rl.RLState
    ) -> Tuple[List, List]:
        """Execute a goal chain and collect state observations for each push.

        Args:
            object_id: Object being pushed
            goal_chain: List of goals to execute in sequence
            baseline_state: Starting state

        Returns:
            Tuple of (state_observations, post_action_state_observations)
        """
        self.env.set_full_state(baseline_state)
        state_obs = []
        post_state_obs = []

        for goal in goal_chain:
            # Capture state before action
            pre_obs = self.env.get_observation()
            state_obs.append(pre_obs)

            # Execute action
            action = namo_rl.Action()
            action.object_id = object_id
            action.x = goal.x
            action.y = goal.y
            action.theta = goal.theta
            self.env.step(action)

            # Capture state after action
            post_obs = self.env.get_observation()
            post_state_obs.append(post_obs)

        return state_obs, post_state_obs

    def _search_minimum_depth_goals(
        self,
        object_id: str,
        baseline_state: namo_rl.RLState,
        neighbour_label: str,
        region_goals: Dict[str, Any]
    ) -> Tuple[List[Tuple[List[Goal], List, List, 'namo_rl.RLState', Optional[Tuple]]], int]:
        """Search for minimum-depth goals using BFS through push steps.

        Args:
            object_id: Object to push
            baseline_state: State to reset to before each push
            neighbour_label: Neighbour region to validate opening to
            region_goals: Region goals for validation

        Returns:
            Tuple of (successful_chains, depth) where successful_chains is a list of tuples:
            (goal_chain, state_obs, post_state_obs, resulting_state, region_goal_used).
            Returns ([], 0) if no goals succeed.
        """
        # Generate all primitive goals (60 edges × 10 steps)
        goals_per_edge = self.goal_sampler.generate_goals(
            object_id,
            baseline_state,
            self.env,
            max_goals=0  # Ignored by primitive strategy
        )

        if not goals_per_edge:
            return [], 0

        # Get reachable edge indices using wavefront analysis
        self.env.set_full_state(baseline_state)
        reachable_edge_indices = set(self.env.get_reachable_edges(object_id))

        if not reachable_edge_indices:
            return [], 0

        # Search with chaining (outer BFS over chain depth)
        return self._search_with_chaining_bfs(goals_per_edge, reachable_edge_indices, baseline_state,
                                             neighbour_label, region_goals, object_id)

    def _search_with_chaining_bfs(
        self,
        initial_goals_per_edge: List[List[Goal]],
        initial_reachable_edges: Set[int],
        baseline_state: namo_rl.RLState,
        neighbour_label: str,
        region_goals: Dict[str, Any],
        object_id: str
    ) -> Tuple[List[Tuple[List[Goal], List, List, 'namo_rl.RLState', Optional[Tuple]]], int]:
        """Outer BFS over chain depth: Try single pushes, then 2-push chains, then 3-push chains.

        Returns:
            Tuple of (all_chains, depth) where all_chains is a list of tuples:
            (goal_chain, state_obs, post_state_obs, resulting_state, region_goal_used).
            Returns ([], 0) if no solution found.
        """

        # Initial frontier for chain depth 1
        root_node = ChainNode(
            state=baseline_state,
            goal=None,
            edge_idx=-1,
            depth=0,
            parent=None
        )
        frontier = [root_node]

        # Try chain depths 1, 2, 3, ...
        for chain_depth in range(1, self.max_chain_depth + 1):
            next_frontier = []

            for node in frontier:
                # Restore to this node's state
                self.env.set_full_state(node.state)

                # Generate primitive goals for object at THIS state
                goals_per_edge = self.goal_sampler.generate_goals(
                    object_id,
                    node.state,
                    self.env,
                    max_goals=0
                )

                if not goals_per_edge:
                    continue

                # Get reachable edges from this state (state already set above)
                reachable_edge_indices = set(self.env.get_reachable_edges(object_id))

                if not reachable_edge_indices:
                    continue

                # Run inner BFS (single-skill search)
                collect_frontier = (chain_depth < self.max_chain_depth)
                successful_results, primitive_depth, new_frontier_nodes = self._search_bfs(
                    goals_per_edge,
                    reachable_edge_indices,
                    node.state,
                    neighbour_label,
                    region_goals,
                    object_id,
                    parent_node=node,
                    current_chain_depth=chain_depth,
                    collect_frontier=collect_frontier
                )

                # If we found success, reconstruct ALL goal chains with their state observations
                if successful_results:
                    all_chains = []
                    for (final_goal, final_state_obs, final_post_state_obs, resulting_state, region_goal_used, success_node) in successful_results:
                        # For multi-push chains, reconstruct full chain with observations
                        if chain_depth > 1:
                            goal_chain, state_obs, post_state_obs = self._reconstruct_chain_with_observations(
                                success_node, object_id, baseline_state
                            )
                        else:
                            # Single push - use observations captured during search
                            goal_chain = [final_goal]
                            state_obs = final_state_obs
                            post_state_obs = final_post_state_obs

                        all_chains.append((goal_chain, state_obs, post_state_obs, resulting_state, region_goal_used))

                    # Return all chains at this minimum depth
                    return all_chains, chain_depth

                # Add new frontier nodes for next chain level
                next_frontier.extend(new_frontier_nodes)

            # Move to next chain depth
            frontier = next_frontier

            if not frontier:
                break

        # No solution found at any chain depth
        return [], 0

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
    ) -> Tuple[List[Goal], List, List]:
        """Reconstruct goal chain and collect observations by re-executing.

        This is only called for multi-push chains (chain_depth > 1).

        Args:
            success_node: Final ChainNode containing parent chain
            object_id: Object being pushed
            baseline_state: Starting state for re-execution

        Returns:
            Tuple of (goal_chain, state_obs, post_state_obs)
        """
        # Reconstruct goal chain from parent nodes
        goal_chain = []
        node = success_node
        while node.parent is not None:
            goal_chain.append(node.goal)
            node = node.parent
        goal_chain.reverse()

        # Re-execute chain to collect observations
        state_obs, post_state_obs = self._collect_chain_observations(
            object_id, goal_chain, baseline_state
        )

        return goal_chain, state_obs, post_state_obs

    def _search_bfs(
        self,
        goals_per_edge: List[List[Goal]],
        reachable_edge_indices: Set[int],
        baseline_state: namo_rl.RLState,
        neighbour_label: str,
        region_goals: Dict[str, Any],
        object_id: str,
        parent_node: Optional[ChainNode] = None,
        current_chain_depth: int = 1,
        collect_frontier: bool = False
    ) -> Tuple[List[Tuple[Goal, List, List, 'namo_rl.RLState', Optional[Tuple], ChainNode]], int, List[ChainNode]]:
        """BFS: Try all edges at depth 1, then all at depth 2, etc.

        Args:
            collect_frontier: If True, collect valid but unsuccessful states as frontier nodes

        Returns:
            Tuple of (successful_results, primitive_depth, frontier_nodes) where successful_results
            is a list of (goal, state_obs, post_state_obs, resulting_state, region_goal_used, chain_node) tuples.
            The chain_node contains the full parent chain for observation reconstruction.
        """
        max_depth = len(goals_per_edge[0]) if goals_per_edge else 10

        # Track edges that have collided or gotten stuck during THIS skill execution
        # Once an edge collides or gets stuck, we blacklist it for all remaining primitive depths
        blacklisted_edges_this_skill = set()

        # Frontier nodes for chaining
        frontier_nodes = []

        for depth in range(max_depth):  # depth 0 = step 1, depth 1 = step 2, etc.
            successful_results = []

            # Try only reachable edge points at this depth
            for edge_idx, edge_goals in enumerate(goals_per_edge):
                # Filter: only try reachable edges
                if edge_idx not in reachable_edge_indices:
                    continue

                # Filter: skip blacklisted edges (collided or stuck earlier in this skill execution)
                if edge_idx in blacklisted_edges_this_skill:
                    continue

                if depth >= len(edge_goals):
                    continue  # This edge doesn't have this many steps

                goal = edge_goals[depth]

                # Reset to baseline before each attempt
                self.env.set_full_state(baseline_state)

                # Check reachability BEFORE push
                is_accessible_before, reachable_count_before, _ = self._validate_opening(neighbour_label, region_goals)

                # Capture state observation before action
                pre_state_obs = self.env.get_observation()

                # Execute push
                action = namo_rl.Action()
                action.object_id = object_id
                action.x = goal.x
                action.y = goal.y
                action.theta = goal.theta

                try:
                    step_result = self.env.step(action)

                    # Capture state observation after action
                    post_state_obs = self.env.get_observation()

                    # Check for collision (only if we should terminate on collision)
                    if self.terminate_on_collision and "collision_object" in step_result.info:
                        # Blacklist this edge for all remaining depths in this skill execution
                        blacklisted_edges_this_skill.add(edge_idx)
                        continue

                    # Check for stuck condition (propagated from C++ skill execution)
                    if "stuck" in step_result.info and step_result.info["stuck"] == "true":
                        # Blacklist this edge for all remaining depths
                        blacklisted_edges_this_skill.add(edge_idx)
                        continue

                except Exception as e:
                    continue

                # Check reachability AFTER push
                is_accessible_after, reachable_count_after, region_goal_used = self._validate_opening(neighbour_label, region_goals)

                # Only count as success if we IMPROVED accessibility
                if is_accessible_after and not is_accessible_before:
                    # Created NEW opening! ✓
                    resulting_state = self.env.get_full_state()

                    # Create a ChainNode for this successful goal (stores observations)
                    success_node = ChainNode(
                        state=resulting_state,
                        goal=goal,
                        edge_idx=edge_idx,
                        depth=current_chain_depth,
                        parent=parent_node,
                        collided_edges=set()
                    )

                    # Return the node along with single-step observations
                    successful_results.append((goal, [pre_state_obs], [post_state_obs], resulting_state, region_goal_used, success_node))
                elif collect_frontier:
                    # Valid push but didn't create opening - add to frontier for next chain level
                    new_node = ChainNode(
                        state=self.env.get_full_state(),
                        goal=goal,
                        edge_idx=edge_idx,
                        depth=current_chain_depth,
                        parent=parent_node,
                        collided_edges=set()  # Reset blacklist for next chain level
                    )
                    frontier_nodes.append(new_node)

            # If we found any successes at this depth, return them all
            if successful_results:
                return successful_results, depth + 1, frontier_nodes

        # No successful goals found at any depth
        return [], 0, frontier_nodes

    def _validate_opening(
        self,
        neighbour_label: str,
        region_goals: Dict[str, Any]
    ) -> Tuple[bool, int, Optional[Tuple[float, float, float]]]:
        """Validate that opening to neighbour was created using reachability.

        Success criterion: At least half of the region goals must be reachable.

        Args:
            neighbour_label: Target neighbour region
            region_goals: Region goal samples from snapshot

        Returns:
            Tuple of (success, reachable_count, region_goal_used):
                - success: True if at least half of region goals are reachable
                - reachable_count: Number of reachable goals
                - region_goal_used: The first reachable goal found
        """
        # Get region goals for this neighbour
        if neighbour_label not in region_goals:
            return False, 0, None

        bundle = region_goals[neighbour_label]
        if not bundle.goals:
            return False, 0, None

        total_goals = len(bundle.goals)
        reachable_count = 0
        first_reachable_goal = None

        # Check ALL goals and count how many are reachable
        for goal_sample in bundle.goals:
            self.env.set_robot_goal(goal_sample.x, goal_sample.y, goal_sample.theta)
            if self.env.is_robot_goal_reachable():
                reachable_count += 1
                # Store the first reachable goal
                if first_reachable_goal is None:
                    first_reachable_goal = (goal_sample.x, goal_sample.y, goal_sample.theta)

        # Success if at least half of the goals are reachable
        required_count = (total_goals + 1) // 2  # Ceiling division
        if reachable_count >= required_count:
            return True, reachable_count, first_reachable_goal
        else:
            return False, reachable_count, None


# Register the planner with the factory
from namo.core import PlannerFactory
PlannerFactory.register_planner("region_opening", RegionOpeningPlanner)
