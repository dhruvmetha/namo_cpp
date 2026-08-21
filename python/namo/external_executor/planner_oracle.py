"""Planner oracle that returns push goal_chains.

This module wraps the NAMO planner to provide a simple interface for
the external executor. Given a tmp_planner.xml and config, it:

1. Spawns NAMO planning environment
2. Computes region connectivity
3. Finds path to goal region
4. Calls RegionOpeningPlanner to get a goal_chain for the next region step
5. Returns the chain as (object_id, edge_idx, depth) tuples

The oracle is stateless - it spawns a fresh planner env each call.
This matches the "generate tmp_planner.xml → plan → execute" workflow.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import namo_rl

from namo.core import PlannerConfig
from namo.planners.connectivity_snapshot import snapshot_region_connectivity, find_robot_label
from namo.planners.opening.region_opening import RegionOpeningPlanner, AttemptResult


@dataclass
class ChainLink:
    """Single link in a push goal_chain."""
    object_id: str
    edge_idx: int
    depth: int  # depth=0 means 1 push step, depth=1 means 2 push steps, etc.
    
    # Optional: contact info computed by planner (if available)
    contact_xy: Optional[Tuple[float, float]] = None
    push_dir: Optional[Tuple[float, float]] = None


@dataclass
class OracleResult:
    """Result from planner oracle."""
    success: bool
    goal_chain: List[ChainLink]  # Empty if not success
    target_region: Optional[str] = None
    error_message: Optional[str] = None
    
    # Debugging info
    robot_region: Optional[str] = None
    goal_region: Optional[str] = None
    region_path: Optional[List[str]] = None
    attempt_result: Optional[AttemptResult] = None


class PlannerOracle:
    """Oracle that spawns NAMO planner and returns push goal_chains.
    
    Usage:
        oracle = PlannerOracle(config)
        result = oracle.get_next_push_chain(
            tmp_planner_xml,
            robot_goal=(x, y, theta)
        )
        
        if result.success:
            for link in result.goal_chain:
                # Execute push: navigate to contact, push depth+1 times
                execute_push(link.object_id, link.edge_idx, link.depth)
    """
    
    def __init__(self, config_yaml: Path, verbose: bool = False):
        """Initialize oracle.
        
        Args:
            config_yaml: Path to the canonical car 1x d5 NAMO config YAML
            verbose: Enable verbose output
        """
        self.config_yaml = Path(config_yaml)
        self.verbose = verbose
    
    def get_next_push_chain(
        self,
        tmp_planner_xml: Path,
        robot_goal: Tuple[float, float, float],
        max_chain_depth: int = 3
    ) -> OracleResult:
        """Get the next push goal_chain to progress toward goal.
        
        This spawns a fresh NAMO planner environment, computes regions,
        and calls RegionOpeningPlanner for the next step.
        
        Args:
            tmp_planner_xml: Path to temporary planner XML
            robot_goal: Target robot position (x, y, theta)
            max_chain_depth: Maximum chain depth for region opening
            
        Returns:
            OracleResult with goal_chain or error
        """
        tmp_planner_xml = Path(tmp_planner_xml)
        
        # Create planner environment
        try:
            env = namo_rl.RLEnvironment(
                str(tmp_planner_xml),
                str(self.config_yaml)
            )
        except Exception as e:
            return OracleResult(
                success=False,
                goal_chain=[],
                error_message=f"Failed to create planner environment: {e}"
            )
        
        # Set robot goal for reachability checks
        env.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])
        
        # Check if goal is already reachable
        if env.is_robot_goal_reachable():
            return OracleResult(
                success=True,
                goal_chain=[],  # Empty chain = goal already reachable
                error_message="Goal already reachable"
            )
        
        # Compute region connectivity
        try:
            region_snapshot = snapshot_region_connectivity(
                env,
                str(tmp_planner_xml),
                str(self.config_yaml)
            )
        except Exception as e:
            return OracleResult(
                success=False,
                goal_chain=[],
                error_message=f"Failed to compute region connectivity: {e}"
            )
        
        # Find robot and goal regions
        robot_region = find_robot_label(region_snapshot)
        if robot_region is None:
            return OracleResult(
                success=False,
                goal_chain=[],
                error_message="Robot not in any region"
            )
        
        goal_region = self._get_region_at_position(region_snapshot, robot_goal[0], robot_goal[1])
        if goal_region is None:
            return OracleResult(
                success=False,
                goal_chain=[],
                robot_region=robot_region,
                error_message="Goal not in any region"
            )
        
        # Find path through regions
        region_path = self._find_region_path(
            region_snapshot["adjacency"],
            robot_region,
            goal_region
        )
        
        if region_path is None or len(region_path) < 2:
            return OracleResult(
                success=False,
                goal_chain=[],
                robot_region=robot_region,
                goal_region=goal_region,
                error_message="No region path to goal"
            )
        
        # Get next neighbor to open
        target_neighbor = region_path[1]  # First step after current region
        
        if self.verbose:
            print(f"[Oracle] Robot in {robot_region}, goal in {goal_region}")
            print(f"[Oracle] Region path: {' -> '.join(region_path)}")
            print(f"[Oracle] Opening path to: {target_neighbor}")
        
        # Create region opening planner
        planner_config = PlannerConfig(
            algorithm="region_opening",
            max_depth=max_chain_depth,
            verbose=self.verbose,
            algorithm_params={
                "max_chain_depth": max_chain_depth,
                "solutions_per_neighbor": 1,
                "use_geometric_strategy": True,
            }
        )
        
        region_opener = RegionOpeningPlanner(env, planner_config)
        
        # Run region opening for target neighbor
        try:
            attempt_results = region_opener.search(
                robot_goal,
                target_neighbor=target_neighbor
            )
        except Exception as e:
            return OracleResult(
                success=False,
                goal_chain=[],
                robot_region=robot_region,
                goal_region=goal_region,
                region_path=region_path,
                error_message=f"Region opening failed: {e}"
            )
        
        # Find successful attempt
        successful_attempt = None
        if hasattr(attempt_results, 'attempt_results'):
            # PlannerResult with attempt_results attribute
            for attempt in attempt_results.attempt_results:
                if attempt.success:
                    successful_attempt = attempt
                    break
        elif isinstance(attempt_results, AttemptResult) and attempt_results.success:
            successful_attempt = attempt_results
        
        if successful_attempt is None:
            return OracleResult(
                success=False,
                goal_chain=[],
                robot_region=robot_region,
                goal_region=goal_region,
                region_path=region_path,
                target_region=target_neighbor,
                error_message="No successful opening found"
            )
        
        # Extract goal_chain from attempt
        goal_chain = self._extract_goal_chain(successful_attempt)
        
        return OracleResult(
            success=True,
            goal_chain=goal_chain,
            robot_region=robot_region,
            goal_region=goal_region,
            region_path=region_path,
            target_region=target_neighbor,
            attempt_result=successful_attempt
        )
    
    def _get_region_at_position(
        self,
        snapshot: Dict[str, Any],
        x: float,
        y: float
    ) -> Optional[str]:
        """Get region label at a world position."""
        resolution = snapshot["resolution"]
        bounds = snapshot["bounds"]
        region_map = snapshot["region_map"]
        region_labels = snapshot["region_labels"]
        
        # Convert to grid coordinates
        gx = int((x - bounds[0]) / resolution)
        gy = int((y - bounds[2]) / resolution)
        
        if 0 <= gx < region_map.shape[1] and 0 <= gy < region_map.shape[0]:
            label_id = region_map[gy, gx]
            return region_labels.get(label_id)
        
        return None
    
    def _find_region_path(
        self,
        adjacency: Dict[str, set],
        start: str,
        goal: str
    ) -> Optional[List[str]]:
        """Find shortest path through region graph using BFS."""
        from collections import deque
        
        if start == goal:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in adjacency.get(current, set()):
                if neighbor == goal:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def _extract_goal_chain(self, attempt: AttemptResult) -> List[ChainLink]:
        """Extract goal_chain from AttemptResult.
        
        The AttemptResult has:
        - goal_chain: List[Goal] where Goal has object_id, edge_idx, depth
        - actions_executed: List[namo_rl.Action]
        """
        chain = []
        
        if attempt.goal_chain is not None:
            for goal in attempt.goal_chain:
                # Goal object has: object_id, edge_idx, depth (depth_idx), x, y, theta
                link = ChainLink(
                    object_id=goal.object_id,
                    edge_idx=goal.edge_idx,
                    depth=goal.depth_idx if hasattr(goal, 'depth_idx') else 0,
                    contact_xy=(goal.x, goal.y) if hasattr(goal, 'x') else None
                )
                chain.append(link)
        elif attempt.actions_executed:
            # Fallback: extract from actions
            for action in attempt.actions_executed:
                # namo_rl.Action has: object_id, edge_idx, depth
                link = ChainLink(
                    object_id=action.object_id,
                    edge_idx=action.edge_idx,
                    depth=action.depth
                )
                chain.append(link)
        
        return chain


def check_goal_reachable(
    tmp_planner_xml: Path,
    config_yaml: Path,
    robot_goal: Tuple[float, float, float]
) -> bool:
    """Quick check if robot goal is reachable.
    
    Args:
        tmp_planner_xml: Path to planner XML
        config_yaml: Path to config YAML
        robot_goal: Target robot position
        
    Returns:
        True if goal is currently reachable
    """
    env = namo_rl.RLEnvironment(str(tmp_planner_xml), str(config_yaml))
    env.set_robot_goal(robot_goal[0], robot_goal[1], robot_goal[2])
    return env.is_robot_goal_reachable()
