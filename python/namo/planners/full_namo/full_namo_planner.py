"""Full NAMO Planner - Hierarchical planning using region opening as sub-problem.

This planner solves the full NAMO problem (reaching a specific robot goal) by:
1. Computing the region connectivity graph
2. Finding the path through regions from robot to goal
3. Iteratively opening regions along that path using RegionOpeningPlanner
4. Terminating when the robot goal becomes reachable
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import namo_rl

from namo.core import BasePlanner, PlannerConfig, PlannerResult
from namo.planners.connectivity_snapshot import snapshot_region_connectivity, find_robot_label
from namo.planners.opening.region_opening import RegionOpeningPlanner


@dataclass
class FullNAMOStats:
    """Statistics for full NAMO planning."""
    iterations: int = 0
    total_pushes: int = 0
    regions_opened: List[str] = field(default_factory=list)
    region_paths: List[List[str]] = field(default_factory=list)
    per_iteration_stats: List[Dict[str, Any]] = field(default_factory=list)


class FullNAMOPlanner(BasePlanner):
    """Full NAMO planner using region opening as sub-problem solver.

    This planner iteratively opens regions along the path from the robot's
    current region to the goal region until the robot goal becomes reachable.
    """

    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        """Initialize full NAMO planner.

        Args:
            env: NAMO RL environment
            config: Planner configuration
        """
        algo_params = config.algorithm_params or {}

        # Full NAMO specific settings
        self.max_iterations = algo_params.get("full_namo_max_iterations", 20)

        # Store config for creating region opener
        self._config = config
        self._env = env

        # Will be initialized in _initialize_algorithm
        self.region_opener: Optional[RegionOpeningPlanner] = None

        # Statistics
        self.stats = FullNAMOStats()

        super().__init__(env, config)

    def _setup_constraints(self):
        """Setup action constraints from environment."""
        pass

    def _initialize_algorithm(self):
        """Initialize algorithm-specific components."""
        # Create the region opening planner as sub-solver
        self.region_opener = RegionOpeningPlanner(self._env, self._config)

    @property
    def algorithm_name(self) -> str:
        """Return human-readable algorithm name."""
        return "Full NAMO Planner"

    @property
    def algorithm_version(self) -> str:
        """Return algorithm version/variant identifier."""
        return "v1.0-hierarchical"

    def reset(self):
        """Reset internal algorithm state for new planning episode."""
        self.stats = FullNAMOStats()
        if self.region_opener:
            self.region_opener.reset()

    def _debug(self, message: str):
        """Print debug message if verbose mode is enabled."""
        if getattr(self.config, "verbose", False):
            print(message)

    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        """Execute full NAMO planning to reach robot goal.

        Args:
            robot_goal: Target robot position (x, y, theta)

        Returns:
            PlannerResult with action sequence to reach goal
        """
        start_time = time.time()
        self.stats = FullNAMOStats()

        # Set the robot goal in environment
        self.env.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])

        all_actions: List[namo_rl.Action] = []
        # Track resulting state after each action for visualization replay
        action_resulting_states: List[namo_rl.RLState] = []

        self._debug(f"\n{'='*60}")
        self._debug(f"Full NAMO Planner - Target: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f}, {robot_goal[2]:.2f})")
        self._debug(f"Max iterations: {self.max_iterations}")
        self._debug(f"{'='*60}\n")

        for iteration in range(self.max_iterations):
            self.stats.iterations = iteration + 1
            iter_start = time.time()

            self._debug(f"\n--- Iteration {iteration + 1} ---")

            # Re-set robot goal each iteration (region opening may have changed it during validation)
            self.env.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])

            # Check if goal is already reachable
            if self.env.is_robot_goal_reachable():
                self._debug(f"Goal is reachable! Success after {iteration} region openings.")
                total_time = (time.time() - start_time) * 1000

                return PlannerResult(
                    success=True,
                    solution_found=True,
                    action_sequence=all_actions,
                    solution_depth=len(all_actions),
                    search_time_ms=total_time,
                    algorithm_stats={
                        "full_namo_stats": self.stats,
                        "iterations": self.stats.iterations,
                        "total_pushes": self.stats.total_pushes,
                        "regions_opened": self.stats.regions_opened,
                    }
                )

            # Compute region snapshot (full graph, not local)
            snapshot_data = self._compute_region_snapshot()

            if snapshot_data is None:
                return self._failure_result(
                    "Failed to compute region snapshot",
                    start_time, all_actions
                )

            # Find goal's region
            goal_region_label = self._get_region_label_at_position(
                snapshot_data, robot_goal[0], robot_goal[1]
            )

            if goal_region_label is None:
                return self._failure_result(
                    f"Goal position ({robot_goal[0]:.2f}, {robot_goal[1]:.2f}) is in obstacle or out of bounds",
                    start_time, all_actions
                )

            # Find robot's region
            robot_region_label = self._get_robot_region_label(snapshot_data)

            if robot_region_label is None:
                return self._failure_result(
                    "Could not identify robot region",
                    start_time, all_actions
                )

            self._debug(f"Robot region: {robot_region_label}")
            self._debug(f"Goal region: {goal_region_label}")

            # Check if goal is in robot's region (but not reachable - edge case)
            if goal_region_label == robot_region_label:
                # This shouldn't happen for valid goals if wavefront is correct
                # But could happen if goal is in a tight corner
                return self._failure_result(
                    f"Goal is in robot region ({robot_region_label}) but not reachable - may be blocked locally",
                    start_time, all_actions
                )

            # Find path through regions
            region_path = self._find_region_path(
                snapshot_data, robot_region_label, goal_region_label
            )

            if region_path is None:
                return self._failure_result(
                    f"No path from {robot_region_label} to {goal_region_label} in region graph",
                    start_time, all_actions
                )

            self._debug(f"Region path: {' -> '.join(region_path)}")
            self.stats.region_paths.append(region_path)

            # Open the next region on the path
            next_region = region_path[1]  # path[0] is robot_region
            self._debug(f"Opening path to: {next_region}")

            # Call region opener with target_neighbor
            result = self.region_opener.search(robot_goal, target_neighbor=next_region)

            iter_stats = {
                "iteration": iteration + 1,
                "target_region": next_region,
                "region_path": region_path,
                "success": result.success,
                "pushes": len(result.action_sequence) if result.action_sequence else 0,
                "time_ms": (time.time() - iter_start) * 1000,
            }
            self.stats.per_iteration_stats.append(iter_stats)

            if not result.success:
                self._debug(f"Failed to open path to {next_region}")
                return self._failure_result(
                    f"Failed to open path to region {next_region}",
                    start_time, all_actions
                )

            # Collect actions
            if result.action_sequence:
                all_actions.extend(result.action_sequence)
                self.stats.total_pushes += len(result.action_sequence)

            # CRITICAL: Set environment to post-push state for next iteration
            # The region opener restores state before each attempt, so we must
            # explicitly apply the resulting state from the successful opening
            resulting_state = self._get_resulting_state_from_result(result)
            if resulting_state is not None:
                self.env.set_full_state(resulting_state)
            else:
                # Fallback: re-execute the actions to get correct state
                # This shouldn't happen if attempt_results are properly populated
                self._debug("Warning: No resulting_state found, re-executing actions")
                if result.action_sequence:
                    for action in result.action_sequence:
                        self.env.step(action)

            self.stats.regions_opened.append(next_region)
            self._debug(f"Opened {next_region} with {len(result.action_sequence or [])} pushes")

        # Max iterations exceeded
        return self._failure_result(
            f"Max iterations ({self.max_iterations}) exceeded",
            start_time, all_actions
        )

    def _get_resulting_state_from_result(self, result: PlannerResult) -> Optional[namo_rl.RLState]:
        """Extract the resulting state from a successful region opening result.

        The RegionOpeningPlanner stores attempt results in algorithm_stats.
        Each successful AttemptResult has a resulting_state field.

        Args:
            result: PlannerResult from region_opener.search()

        Returns:
            The resulting state after the successful push, or None if not found
        """
        if not result.algorithm_stats:
            return None

        attempt_results = result.algorithm_stats.get("attempt_results", [])
        if not attempt_results:
            return None

        # Find the first successful attempt (should match the action_sequence)
        for attempt in attempt_results:
            if attempt.success and attempt.resulting_state is not None:
                return attempt.resulting_state

        return None

    def _failure_result(
        self,
        error_message: str,
        start_time: float,
        actions: List[namo_rl.Action]
    ) -> PlannerResult:
        """Create a failure PlannerResult."""
        total_time = (time.time() - start_time) * 1000
        self._debug(f"FAILURE: {error_message}")

        return PlannerResult(
            success=False,
            solution_found=False,
            action_sequence=actions if actions else None,
            solution_depth=len(actions) if actions else None,
            search_time_ms=total_time,
            error_message=error_message,
            algorithm_stats={
                "full_namo_stats": self.stats,
                "iterations": self.stats.iterations,
                "total_pushes": self.stats.total_pushes,
                "regions_opened": self.stats.regions_opened,
            }
        )

    def _compute_region_snapshot(self) -> Optional[Dict[str, Any]]:
        """Compute full region connectivity snapshot.

        Returns:
            Dict with adjacency, region_labels, and snapshot (for region_map lookup)
        """
        try:
            xml_path = self.env.get_xml_path()
            config_path = self.env.get_config_path()

            adjacency, edge_objects, region_labels, region_goals, snapshot = snapshot_region_connectivity(
                self.env,
                xml_path,
                config_path,
                include_snapshot=True,  # Need region_map for position lookup
                local_info_only=False,  # Full graph, not just neighbors
                goals_per_region=self.config.goals_per_region,
                generate_training_data=True,
                use_current_state=True,
            )

            return {
                'adjacency': adjacency,
                'edge_objects': edge_objects,
                'region_labels': region_labels,
                'region_goals': region_goals,
                'snapshot': snapshot,
            }
        except Exception as e:
            self._debug(f"Error computing region snapshot: {e}")
            return None

    def _get_region_label_at_position(
        self,
        snapshot_data: Dict[str, Any],
        x: float,
        y: float
    ) -> Optional[str]:
        """Get the region label at world position (x, y).

        Args:
            snapshot_data: Data from _compute_region_snapshot
            x: World x coordinate
            y: World y coordinate

        Returns:
            Region label (e.g., "robot", "goal", "region_2") or None if invalid
        """
        snapshot = snapshot_data['snapshot']
        region_labels = snapshot_data['region_labels']

        bounds = snapshot.bounds  # (xmin, xmax, ymin, ymax)
        resolution = snapshot.resolution
        region_map = snapshot.region_map  # 2D numpy array, indexed [gx, gy]

        # Convert world to grid coordinates
        gx = int((x - bounds[0]) / resolution)
        gy = int((y - bounds[2]) / resolution)

        # Check bounds
        if not (0 <= gx < region_map.shape[0] and 0 <= gy < region_map.shape[1]):
            self._debug(f"Position ({x}, {y}) -> grid ({gx}, {gy}) out of bounds")
            return None

        region_id = int(region_map[gx, gy])

        # region_id == 0 means unassigned/obstacle in some cases
        # region_id < 0 also indicates obstacle
        if region_id <= 0:
            self._debug(f"Position ({x}, {y}) has region_id {region_id} (obstacle/unassigned)")
            return None

        # Look up label
        label = region_labels.get(region_id)
        if label is None:
            self._debug(f"No label for region_id {region_id}")
            return None

        return label

    def _get_robot_region_label(self, snapshot_data: Dict[str, Any]) -> Optional[str]:
        """Get the robot's current region label.

        Args:
            snapshot_data: Data from _compute_region_snapshot

        Returns:
            Robot's region label (typically "robot" or "robot_goal")
        """
        region_labels = snapshot_data['region_labels']
        return find_robot_label(region_labels)

    def _find_region_path(
        self,
        snapshot_data: Dict[str, Any],
        start_label: str,
        goal_label: str
    ) -> Optional[List[str]]:
        """Find shortest path through regions using BFS.

        Args:
            snapshot_data: Data from _compute_region_snapshot
            start_label: Starting region label (robot's region)
            goal_label: Goal region label

        Returns:
            List of region labels forming path, or None if no path exists
        """
        adjacency = snapshot_data['adjacency']

        if start_label not in adjacency:
            self._debug(f"Start label {start_label} not in adjacency graph")
            return None

        if start_label == goal_label:
            return [start_label]

        # BFS
        queue = deque([(start_label, [start_label])])
        visited: Set[str] = {start_label}

        while queue:
            current, path = queue.popleft()

            for neighbor in adjacency.get(current, set()):
                if neighbor == goal_label:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found


# Register with PlannerFactory
from namo.core import PlannerFactory
PlannerFactory.register_planner("full_namo", FullNAMOPlanner)
