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


@dataclass
class RegionOpeningResult:
    """Result from one region opening in the Full NAMO sequence.

    Each RegionOpeningResult represents opening a path to one neighbor region
    by pushing one object (possibly multiple times via skill chaining).
    """
    target_region: str  # The neighbor region we opened path to
    object_id: str      # Object that was pushed
    actions: List[namo_rl.Action] = field(default_factory=list)  # Skill chain actions (with edge_idx/depth)
    resulting_state: Optional[namo_rl.RLState] = None  # State after this opening


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
        """Execute full NAMO planning as online execution.

        Greedy path-following with edge-blocking for recovery.
        No backtracking - if no path exists, we fail.

        Args:
            robot_goal: Target robot position (x, y, theta)

        Returns:
            PlannerResult with action sequence to reach goal
        """
        start_time = time.time()
        self.stats = FullNAMOStats()

        self.env.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])

        self._debug(f"\n{'='*60}")
        self._debug(f"Full NAMO Planner - Target: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f}, {robot_goal[2]:.2f})")
        self._debug(f"{'='*60}\n")

        # Check if goal is already reachable
        if self.env.is_robot_goal_reachable():
            self._debug("Goal is already reachable!")
            return self._success_result(start_time, [], [], [])

        # Get goal region
        snapshot = self._compute_region_snapshot()
        if snapshot is None:
            return self._failure_result("Failed to compute region snapshot", start_time, [])

        goal_region = self._get_region_label_at_position(snapshot, robot_goal[0], robot_goal[1])
        if goal_region is None:
            return self._failure_result("Goal position is in obstacle or out of bounds", start_time, [])

        # State for online execution
        actions: List[namo_rl.Action] = []
        region_openings: List[RegionOpeningResult] = []
        accessible_regions: Set[str] = set()
        failed_edges: Set[Tuple[str, str]] = set()

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            self.stats.iterations = iteration

            self._debug(f"\n--- Iteration {iteration} ---")

            self.env.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])

            # Check if goal reachable
            if self.env.is_robot_goal_reachable():
                self._debug(f"Goal reachable! {len(region_openings)} region openings.")
                return self._success_result(start_time, actions, region_openings,
                                           list(accessible_regions))

            # Compute snapshot
            snapshot = self._compute_region_snapshot()
            if snapshot is None:
                return self._failure_result("Failed to compute region snapshot", start_time, actions)

            robot_region = self._get_robot_region_label(snapshot)
            if robot_region is None:
                return self._failure_result("Can't find robot region", start_time, actions)

            self._debug(f"Robot: {robot_region}, Goal: {goal_region}")
            self._debug(f"Failed edges: {sorted(failed_edges)}")

            # Find path avoiding failed edges
            path = self._find_region_path_avoiding_edges(
                snapshot, robot_region, goal_region,
                accessible_regions, failed_edges
            )

            if path is None:
                return self._failure_result("No path to goal", start_time, actions)

            self._debug(f"Path: {' -> '.join(path)}")

            if len(path) < 2:
                return self._failure_result("At goal region but goal not reachable", start_time, actions)

            target = path[1]
            edge = (robot_region, target)

            # Check if target is direct neighbor
            robot_neighbors = set(snapshot['adjacency'].get(robot_region, set()))
            if target not in robot_neighbors:
                # Target via accessible region - block that edge
                self._debug(f"Target {target} via accessible region")
                for acc in accessible_regions:
                    if target in snapshot['adjacency'].get(acc, set()):
                        failed_edges.add((acc, target))
                        break
                else:
                    failed_edges.add(edge)
                continue

            self._debug(f"Opening {target}")

            # Try to open
            result = self.region_opener.search(robot_goal, target_neighbor=target)

            if not result.success:
                self._debug(f"Failed, marking edge {edge}")
                failed_edges.add(edge)
                continue

            if self._check_already_accessible(result):
                self._debug(f"{target} already accessible")
                accessible_regions.add(target)
                continue

            resulting_state = self._get_resulting_state_from_result(result)
            if resulting_state is None:
                self._debug("No resulting state, marking edge failed")
                failed_edges.add(edge)
                continue

            self.env.set_full_state(resulting_state)
            if self._compute_region_snapshot() is None:
                self._debug("Snapshot after opening failed")
                failed_edges.add(edge)
                continue

            # Success - update state
            if result.action_sequence:
                self.stats.total_pushes += len(result.action_sequence)
                actions.extend(result.action_sequence)
            self.stats.regions_opened.append(target)

            region_openings.append(RegionOpeningResult(
                target_region=target,
                object_id=result.action_sequence[0].object_id if result.action_sequence else "unknown",
                actions=list(result.action_sequence) if result.action_sequence else [],
                resulting_state=resulting_state,
            ))

            # Reset failed edges for new state
            failed_edges = set()

            self._debug(f"Opened {target}")

        return self._failure_result("Max iterations exceeded", start_time, actions)

    def _success_result(self, start_time: float, actions: List[namo_rl.Action],
                        region_openings: List[RegionOpeningResult],
                        accessible_regions: List[str]) -> PlannerResult:
        """Create a success PlannerResult."""
        return PlannerResult(
            success=True,
            solution_found=True,
            action_sequence=actions,
            solution_depth=len(actions),
            search_time_ms=(time.time() - start_time) * 1000,
            algorithm_stats={
                "full_namo_stats": self.stats,
                "iterations": self.stats.iterations,
                "total_pushes": self.stats.total_pushes,
                "regions_opened": self.stats.regions_opened,
                "accessible_regions": accessible_regions,
                "region_opening_sequence": region_openings,
            }
        )

    def _check_already_accessible(self, result: PlannerResult) -> bool:
        """Check if region opener result indicates "already_accessible".

        This happens when the target neighbor is already reachable from the
        robot's current position without any pushing needed.

        Args:
            result: PlannerResult from region_opener.search()

        Returns:
            True if the neighbor was already accessible (no push needed)
        """
        if not result.algorithm_stats:
            return False

        attempt_results = result.algorithm_stats.get("attempt_results", [])
        if not attempt_results:
            return False

        # Check if any attempt has failure_reason="already_accessible"
        for attempt in attempt_results:
            if getattr(attempt, 'failure_reason', None) == "already_accessible":
                return True

        return False

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

    def _find_region_path_avoiding_edges(
        self,
        snapshot_data: Dict[str, Any],
        start_label: str,
        goal_label: str,
        accessible_regions: Optional[Set[str]] = None,
        failed_edges: Optional[Set[Tuple[str, str]]] = None
    ) -> Optional[List[str]]:
        """Find shortest path through regions using BFS, avoiding failed edges.

        Args:
            snapshot_data: Data from _compute_region_snapshot
            start_label: Starting region label (robot's region)
            goal_label: Goal region label
            accessible_regions: Set of regions already accessible from robot's position.
            failed_edges: Set of (from, to) edges that should not be traversed.

        Returns:
            List of region labels forming path, or None if no path exists.
        """
        adjacency = snapshot_data['adjacency']
        accessible = accessible_regions or set()
        blocked = failed_edges or set()

        if start_label not in adjacency:
            self._debug(f"Start label {start_label} not in adjacency graph")
            return None

        if start_label == goal_label:
            return [start_label]

        # Check if goal is in accessible regions
        if goal_label in accessible:
            return [start_label, goal_label]

        # Build expanded start set: start + all accessible regions
        expanded_start = {start_label} | accessible

        # BFS from expanded start
        queue = deque()
        visited: Set[str] = set(expanded_start)

        # Initialize: find all neighbors of expanded_start that need opening
        sorted_start = sorted(expanded_start, key=lambda x: (x != start_label, x))
        for start_node in sorted_start:
            for neighbor in sorted(adjacency.get(start_node, set())):
                # Check if this edge is blocked
                edge = (start_node, neighbor)
                if edge in blocked:
                    continue

                if neighbor == goal_label:
                    return [start_label, goal_label]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [start_label, neighbor], start_node))

        # Continue BFS
        while queue:
            current, path, prev = queue.popleft()

            for neighbor in sorted(adjacency.get(current, set())):
                # Check if this edge is blocked
                edge = (current, neighbor)
                if edge in blocked:
                    continue

                if neighbor == goal_label:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], current))

        return None  # No path found


# Register with PlannerFactory
from namo.core import PlannerFactory
PlannerFactory.register_planner("full_namo", FullNAMOPlanner)
