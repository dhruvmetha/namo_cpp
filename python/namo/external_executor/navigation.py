"""Navigation module: wavefront grid + BFS pathfinding + pure pursuit control.

This module provides executor-side navigation to arbitrary (x, y) targets.
It builds an inflated occupancy grid from the executor state, computes
BFS paths, and follows them with pure pursuit control.

Reuses patterns from python/namo/visualization/wavefront_snapshot.py
but adapts for the external executor context.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import ExecutorConfig, SKILL15_DEFAULTS
from .executor import MuJoCoExecutor, SE2Pose, ExecutorSnapshot


GridArray = NDArray[np.int_]


@dataclass
class OccupancyGrid:
    """2D occupancy grid for navigation."""
    grid: GridArray  # 0 = free, 1 = occupied
    resolution: float
    origin: Tuple[float, float]  # (x, y) of grid cell (0, 0)
    width: int
    height: int
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices."""
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return (gx, gy)
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid cell indices to world coordinates (cell center)."""
        x = self.origin[0] + (gx + 0.5) * self.resolution
        y = self.origin[1] + (gy + 0.5) * self.resolution
        return (x, y)
    
    def is_valid(self, gx: int, gy: int) -> bool:
        """Check if grid cell is within bounds."""
        return 0 <= gx < self.width and 0 <= gy < self.height
    
    def is_free(self, gx: int, gy: int) -> bool:
        """Check if grid cell is free (not occupied)."""
        return self.is_valid(gx, gy) and self.grid[gy, gx] == 0


@dataclass
class NavigationResult:
    """Result of a navigation attempt."""
    success: bool
    final_pose: SE2Pose
    path_length: float
    steps_taken: int
    reason: str = ""


class WavefrontNavigator:
    """Navigator using wavefront planning + pure pursuit.
    
    Usage:
        nav = WavefrontNavigator(executor, config)
        result = nav.navigate_to(target_x, target_y)
    """
    
    def __init__(self, executor: MuJoCoExecutor, config: ExecutorConfig):
        """Initialize navigator.
        
        Args:
            executor: MuJoCo executor for simulation
            config: Configuration with grid resolution, inflation, etc.
        """
        self.executor = executor
        self.config = config
        
        # Navigation parameters
        self.resolution = config.grid_resolution
        self.robot_inflation = config.robot_inflation
        self.goal_tolerance = config.nav_goal_tolerance
        self.lookahead = config.lookahead_distance
        self.max_linear_vel = SKILL15_DEFAULTS["max_linear_velocity"]
        self.max_angular_vel = SKILL15_DEFAULTS["max_angular_velocity"]
        
        # Cached grid (rebuilt when snapshot changes)
        self._cached_grid: Optional[OccupancyGrid] = None
    
    def build_occupancy_grid(
        self,
        snapshot: Optional[ExecutorSnapshot] = None,
        exclude_object: Optional[str] = None
    ) -> OccupancyGrid:
        """Build inflated occupancy grid from executor state.
        
        Args:
            snapshot: Executor snapshot (uses current if None)
            exclude_object: Object name to exclude from obstacles (for pushing)
            
        Returns:
            Occupancy grid with inflated obstacles
        """
        if snapshot is None:
            snapshot = self.executor.get_snapshot()
        
        bounds = snapshot.world_bounds
        xmin, xmax, ymin, ymax = bounds
        
        # Grid dimensions
        width = int((xmax - xmin) / self.resolution) + 1
        height = int((ymax - ymin) / self.resolution) + 1
        
        # Initialize grid as free
        grid = np.zeros((height, width), dtype=np.int_)
        
        # Inflation radius in cells
        inflate_cells = int(math.ceil((self.robot_inflation + snapshot.robot_radius) / self.resolution))
        
        # Mark walls/static obstacles
        # For simplicity, we mark cells based on object positions
        # (In production, would parse XML geometry more carefully)
        
        # Mark movable obstacles
        for name, pose in snapshot.movable_poses.items():
            if name == exclude_object:
                continue
            
            info = snapshot.object_info.get(name)
            if info is None:
                continue
            
            # Mark cells occupied by this object (with inflation)
            self._mark_box_obstacle(
                grid,
                pose.x, pose.y, pose.theta,
                info.half_extent_x, info.half_extent_y,
                inflate_cells,
                xmin, ymin,
                self.resolution
            )
        
        # Mark boundary walls (from bounds)
        self._mark_boundary_walls(grid, bounds, inflate_cells, self.resolution)
        
        return OccupancyGrid(
            grid=grid,
            resolution=self.resolution,
            origin=(xmin, ymin),
            width=width,
            height=height
        )
    
    def _mark_box_obstacle(
        self,
        grid: GridArray,
        cx: float, cy: float, theta: float,
        hw: float, hd: float,
        inflate: int,
        origin_x: float, origin_y: float,
        resolution: float
    ):
        """Mark cells occupied by a rotated box obstacle."""
        # Get corners of rotated box
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        corners_local = [
            (-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)
        ]
        
        corners_world = []
        for lx, ly in corners_local:
            wx = cx + lx * cos_t - ly * sin_t
            wy = cy + lx * sin_t + ly * cos_t
            corners_world.append((wx, wy))
        
        # Find bounding box in grid coordinates
        xs = [c[0] for c in corners_world]
        ys = [c[1] for c in corners_world]
        
        min_gx = int((min(xs) - origin_x) / resolution) - inflate - 1
        max_gx = int((max(xs) - origin_x) / resolution) + inflate + 1
        min_gy = int((min(ys) - origin_y) / resolution) - inflate - 1
        max_gy = int((max(ys) - origin_y) / resolution) + inflate + 1
        
        height, width = grid.shape
        
        # Mark cells inside the inflated box
        for gy in range(max(0, min_gy), min(height, max_gy + 1)):
            for gx in range(max(0, min_gx), min(width, max_gx + 1)):
                # Cell center in world coordinates
                px = origin_x + (gx + 0.5) * resolution
                py = origin_y + (gy + 0.5) * resolution
                
                # Check distance to box (approximate with point-to-rotated-box)
                if self._point_near_box(px, py, cx, cy, theta, hw, hd, inflate * resolution):
                    grid[gy, gx] = 1
    
    def _point_near_box(
        self,
        px: float, py: float,
        cx: float, cy: float, theta: float,
        hw: float, hd: float,
        margin: float
    ) -> bool:
        """Check if point is within margin of rotated box."""
        # Transform point to box-local coordinates
        cos_t = math.cos(-theta)
        sin_t = math.sin(-theta)
        
        dx = px - cx
        dy = py - cy
        
        lx = dx * cos_t - dy * sin_t
        ly = dx * sin_t + dy * cos_t
        
        # Check if inside inflated box
        return abs(lx) <= hw + margin and abs(ly) <= hd + margin
    
    def _mark_boundary_walls(
        self,
        grid: GridArray,
        bounds: Tuple[float, float, float, float],
        inflate: int,
        resolution: float
    ):
        """Mark boundary walls in grid."""
        xmin, xmax, ymin, ymax = bounds
        height, width = grid.shape
        
        # Mark cells outside bounds as occupied
        for gy in range(height):
            for gx in range(width):
                px = xmin + (gx + 0.5) * resolution
                py = ymin + (gy + 0.5) * resolution
                
                # Distance to boundary
                dist_to_boundary = min(
                    px - xmin, xmax - px,
                    py - ymin, ymax - py
                )
                
                if dist_to_boundary < inflate * resolution:
                    grid[gy, gx] = 1
    
    def find_path_bfs(
        self,
        grid: OccupancyGrid,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """Find path using BFS on occupancy grid.
        
        Args:
            grid: Occupancy grid
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            List of waypoints (x, y), or None if no path exists
        """
        start_cell = grid.world_to_grid(start[0], start[1])
        goal_cell = grid.world_to_grid(goal[0], goal[1])
        
        if not grid.is_valid(*start_cell) or not grid.is_valid(*goal_cell):
            return None
        
        if not grid.is_free(*goal_cell):
            # Try to find nearest free cell to goal
            goal_cell = self._find_nearest_free(grid, goal_cell)
            if goal_cell is None:
                return None
        
        # BFS
        queue: Deque[Tuple[int, int]] = deque([start_cell])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_cell: None}
        
        # 8-connected neighbors
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1)
        ]
        
        while queue:
            current = queue.popleft()
            
            if current == goal_cell:
                # Reconstruct path
                path_cells = []
                cell = current
                while cell is not None:
                    path_cells.append(cell)
                    cell = came_from[cell]
                path_cells.reverse()
                
                # Convert to world coordinates
                return [grid.grid_to_world(gx, gy) for gx, gy in path_cells]
            
            for dx, dy in neighbors:
                next_cell = (current[0] + dx, current[1] + dy)
                
                if next_cell in came_from:
                    continue
                
                if not grid.is_free(*next_cell):
                    continue
                
                came_from[next_cell] = current
                queue.append(next_cell)
        
        return None  # No path found
    
    def _find_nearest_free(
        self,
        grid: OccupancyGrid,
        cell: Tuple[int, int],
        max_radius: int = 10
    ) -> Optional[Tuple[int, int]]:
        """Find nearest free cell to given cell."""
        gx, gy = cell
        
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:  # Only check boundary
                        nx, ny = gx + dx, gy + dy
                        if grid.is_free(nx, ny):
                            return (nx, ny)
        
        return None
    
    def pure_pursuit_control(
        self,
        robot_pose: SE2Pose,
        path: List[Tuple[float, float]],
        path_idx: int
    ) -> Tuple[float, float, int]:
        """Compute pure pursuit control.
        
        Args:
            robot_pose: Current robot pose
            path: List of waypoints
            path_idx: Current waypoint index
            
        Returns:
            (vx, vy, new_path_idx) control and updated path index
        """
        # Find lookahead point
        lookahead_point = None
        new_idx = path_idx
        
        for i in range(path_idx, len(path)):
            px, py = path[i]
            dist = math.sqrt((px - robot_pose.x)**2 + (py - robot_pose.y)**2)
            
            if dist >= self.lookahead:
                lookahead_point = (px, py)
                new_idx = i
                break
        
        if lookahead_point is None and path:
            # Use last point
            lookahead_point = path[-1]
            new_idx = len(path) - 1
        
        if lookahead_point is None:
            return (0.0, 0.0, path_idx)
        
        # Compute velocity toward lookahead point
        dx = lookahead_point[0] - robot_pose.x
        dy = lookahead_point[1] - robot_pose.y
        dist = math.sqrt(dx * dx + dy * dy)
        
        if dist < 0.001:
            return (0.0, 0.0, new_idx)
        
        # Normalize and scale
        vx = (dx / dist) * self.max_linear_vel
        vy = (dy / dist) * self.max_linear_vel
        
        return (vx, vy, new_idx)
    
    def navigate_to(
        self,
        target_x: float,
        target_y: float,
        max_steps: int = 10000,
        exclude_object: Optional[str] = None
    ) -> NavigationResult:
        """Navigate robot to target position.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            max_steps: Maximum simulation steps
            exclude_object: Object to exclude from obstacles (for pushing)
            
        Returns:
            NavigationResult with success status and final pose
        """
        # Build grid
        snapshot = self.executor.get_snapshot()
        grid = self.build_occupancy_grid(snapshot, exclude_object)
        
        # Get current pose
        start_pose = self.executor.get_robot_pose()
        start = (start_pose.x, start_pose.y)
        goal = (target_x, target_y)
        
        # Check if already at goal
        dist_to_goal = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        if dist_to_goal < self.goal_tolerance:
            return NavigationResult(
                success=True,
                final_pose=start_pose,
                path_length=0.0,
                steps_taken=0,
                reason="already_at_goal"
            )
        
        # Find path
        path = self.find_path_bfs(grid, start, goal)
        if path is None:
            return NavigationResult(
                success=False,
                final_pose=start_pose,
                path_length=0.0,
                steps_taken=0,
                reason="no_path_found"
            )
        
        # Follow path with pure pursuit
        path_idx = 0
        steps = 0
        total_distance = 0.0
        prev_pose = start_pose
        
        while steps < max_steps:
            current_pose = self.executor.get_robot_pose()
            
            # Check if reached goal
            dist_to_goal = math.sqrt(
                (goal[0] - current_pose.x)**2 + (goal[1] - current_pose.y)**2
            )
            if dist_to_goal < self.goal_tolerance:
                # Stop robot
                self.executor.set_robot_control(0.0, 0.0)
                return NavigationResult(
                    success=True,
                    final_pose=current_pose,
                    path_length=total_distance,
                    steps_taken=steps,
                    reason="goal_reached"
                )
            
            # Compute control
            vx, vy, path_idx = self.pure_pursuit_control(current_pose, path, path_idx)
            
            # Apply control and step
            self.executor.set_robot_control(vx, vy)
            self.executor.step()
            steps += 1
            
            # Track distance
            new_pose = self.executor.get_robot_pose()
            total_distance += math.sqrt(
                (new_pose.x - prev_pose.x)**2 + (new_pose.y - prev_pose.y)**2
            )
            prev_pose = new_pose
        
        # Timeout
        final_pose = self.executor.get_robot_pose()
        return NavigationResult(
            success=False,
            final_pose=final_pose,
            path_length=total_distance,
            steps_taken=steps,
            reason="max_steps_reached"
        )
    
    def is_position_reachable(
        self,
        target_x: float,
        target_y: float,
        exclude_object: Optional[str] = None
    ) -> bool:
        """Check if a position is reachable (path exists).
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            exclude_object: Object to exclude from obstacles
            
        Returns:
            True if path exists
        """
        snapshot = self.executor.get_snapshot()
        grid = self.build_occupancy_grid(snapshot, exclude_object)
        
        start_pose = self.executor.get_robot_pose()
        start = (start_pose.x, start_pose.y)
        goal = (target_x, target_y)
        
        path = self.find_path_bfs(grid, start, goal)
        return path is not None
