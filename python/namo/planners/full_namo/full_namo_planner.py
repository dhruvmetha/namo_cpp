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
        self.use_cpp_unified_wavefront = algo_params.get(
            "full_namo_use_cpp_unified_wavefront",
            algo_params.get("region_use_cpp_unified_wavefront", True),
        )
        self.region_snapshot_seed = int(algo_params.get("region_snapshot_seed", 42))
        region_goal_radius = algo_params.get("region_goal_radius_m", None)
        self.region_goal_radius_m = float(region_goal_radius) if region_goal_radius is not None else None

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

        goal_region = snapshot.get("goal_label")
        goal_in_free_space = bool(snapshot.get("goal_in_free_space", False))
        if not goal_region or not goal_in_free_space:
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
            self.stats.total_attempted_pushes += self._extract_attempted_pushes_from_region_result(result)

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
        serialized_region_openings = self._serialize_region_opening_sequence(region_openings)
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
                "total_attempted_pushes": self.stats.total_attempted_pushes,
                "regions_opened": self.stats.regions_opened,
                "accessible_regions": accessible_regions,
                # Keep this pickle-safe for collection serialization.
                "region_opening_sequence": serialized_region_openings,
            }
        )

    def _serialize_action(self, action: namo_rl.Action) -> Dict[str, Any]:
        """Convert namo_rl.Action to a plain dict for safe persistence."""
        payload: Dict[str, Any] = {
            "object_id": getattr(action, "object_id", None),
            "x": None,
            "y": None,
            "theta": None,
            "edge_idx": None,
            "depth": None,
        }
        try:
            payload["x"] = float(getattr(action, "x"))
        except Exception:
            pass
        try:
            payload["y"] = float(getattr(action, "y"))
        except Exception:
            pass
        try:
            payload["theta"] = float(getattr(action, "theta"))
        except Exception:
            pass
        try:
            payload["edge_idx"] = int(getattr(action, "edge_idx"))
        except Exception:
            pass
        try:
            payload["depth"] = int(getattr(action, "depth"))
        except Exception:
            pass
        return payload

    def _serialize_region_opening_sequence(
        self,
        region_openings: List[RegionOpeningResult],
    ) -> List[Dict[str, Any]]:
        """Serialize region-opening details into plain Python objects."""
        out: List[Dict[str, Any]] = []
        for opening in region_openings or []:
            out.append(
                {
                    "target_region": opening.target_region,
                    "object_id": opening.object_id,
                    "actions": [self._serialize_action(a) for a in (opening.actions or [])],
                    # RLState is not pickle-safe; only keep a flag.
                    "has_resulting_state": opening.resulting_state is not None,
                }
            )
        return out

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
                "total_attempted_pushes": self.stats.total_attempted_pushes,
                "regions_opened": self.stats.regions_opened,
            }
        )

    def _compute_region_snapshot(self) -> Optional[Dict[str, Any]]:
        """Compute full region connectivity snapshot.

        Returns:
            Dict with adjacency/edge_objects/region_labels and goal-region metadata.
        """
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
        """Get the robot's current region label.

        Args:
            snapshot_data: Data from _compute_region_snapshot

        Returns:
            Robot's region label (typically "robot" or "robot_goal")
        """
        robot_label = snapshot_data.get("robot_label")
        if robot_label:
            return robot_label
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
