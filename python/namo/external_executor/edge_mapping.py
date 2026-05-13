"""Edge/contact point mapping for push execution.

This module provides a Python port of the C++ edge enumeration algorithm
from src/planning/namo_push_controller.cpp:generate_rectangular_edge_points.

The algorithm generates 4 * points_per_face edge points around a rectangular
object in a specific order (top/bottom pairs, then right/left pairs),
along with paired midpoints for push direction computation.

Edge indexing (with points_per_face=15, total=60):
- Indices 0-29: top/bottom face pairs
  - Even indices (0,2,4,...,28): top face, from left to right
  - Odd indices (1,3,5,...,29): bottom face, from left to right  
- Indices 30-59: right/left face pairs
  - Even indices (30,32,34,...,58): right face, from bottom to top
  - Odd indices (31,33,35,...,59): left face, from bottom to top

Push direction is computed from edge point toward its paired midpoint:
- For even edge_idx i: midpoint is average of edge_points[i] and edge_points[i+1]
- For odd edge_idx i: midpoint is average of edge_points[i] and edge_points[i-1]
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

from .config import ExecutorConfig, SKILL15_DEFAULTS
from .executor import MuJoCoExecutor, SE2Pose, ObjectInfo


@dataclass
class EdgeContact:
    """Contact information for a push edge."""
    edge_idx: int
    contact_xy: Tuple[float, float]  # World coordinates of contact point
    midpoint_xy: Tuple[float, float]  # World coordinates of paired midpoint
    push_dir: Tuple[float, float]    # Unit vector from contact toward object center
    
    @property
    def push_dir_angle(self) -> float:
        """Angle of push direction in radians."""
        return math.atan2(self.push_dir[1], self.push_dir[0])


class EdgeContactMapper:
    """Maps edge_idx to contact points and push directions.
    
    This is a Python port of NAMOPushController::generate_rectangular_edge_points
    from src/planning/namo_push_controller.cpp.
    
    Usage:
        mapper = EdgeContactMapper(config)
        contact = mapper.compute_contact(
            object_info,
            object_pose,
            edge_idx,
            robot_radius
        )
    """
    
    def __init__(self, config: ExecutorConfig):
        """Initialize mapper.
        
        Args:
            config: Configuration with points_per_face setting
        """
        self.config = config
        self.points_per_face = config.points_per_face
        self.total_edge_points = 4 * self.points_per_face
        self.contact_offset = config.robot_contact_offset
    
    def compute_all_edges(
        self,
        obj_info: ObjectInfo,
        obj_pose: SE2Pose,
        robot_radius: float
    ) -> List[EdgeContact]:
        """Compute all edge contacts for an object.
        
        Args:
            obj_info: Object geometry info (half extents)
            obj_pose: Object pose (x, y, theta)
            robot_radius: Robot radius for offset
            
        Returns:
            List of EdgeContact for all edge indices
        """
        # Generate local edge points (same algorithm as C++)
        local_edge_points, local_mid_points = self._generate_local_points(
            obj_info.half_extent_x,
            obj_info.half_extent_y,
            robot_radius
        )
        
        # Transform to world coordinates
        yaw = obj_pose.theta
        cx, cy = obj_pose.x, obj_pose.y
        
        world_edge_points = [
            self._transform_point(p, cx, cy, yaw) for p in local_edge_points
        ]
        world_mid_points = [
            self._transform_point(p, cx, cy, yaw) for p in local_mid_points
        ]
        
        # Build EdgeContact list
        edges = []
        for i in range(len(world_edge_points)):
            contact = world_edge_points[i]
            midpoint = world_mid_points[i]
            
            # Push direction: from contact toward midpoint (toward object center)
            dx = midpoint[0] - contact[0]
            dy = midpoint[1] - contact[1]
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist > 1e-6:
                push_dir = (dx / dist, dy / dist)
            else:
                push_dir = (0.0, 1.0)  # fallback
            
            edges.append(EdgeContact(
                edge_idx=i,
                contact_xy=contact,
                midpoint_xy=midpoint,
                push_dir=push_dir
            ))
        
        return edges
    
    def compute_contact(
        self,
        obj_info: ObjectInfo,
        obj_pose: SE2Pose,
        edge_idx: int,
        robot_radius: float
    ) -> EdgeContact:
        """Compute contact information for a specific edge index.
        
        Args:
            obj_info: Object geometry info
            obj_pose: Object pose
            edge_idx: Edge index (0 to total_edge_points-1)
            robot_radius: Robot radius
            
        Returns:
            EdgeContact for the specified edge
        """
        if edge_idx < 0 or edge_idx >= self.total_edge_points:
            raise ValueError(f"edge_idx {edge_idx} out of range [0, {self.total_edge_points})")
        
        all_edges = self.compute_all_edges(obj_info, obj_pose, robot_radius)
        return all_edges[edge_idx]
    
    def _generate_local_points(
        self,
        half_w: float,
        half_d: float,
        robot_radius: float
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Generate edge points and midpoints in local (object-centered) coordinates.
        
        This mirrors the C++ implementation exactly.
        
        Args:
            half_w: Half-width of object (x direction)
            half_d: Half-depth of object (y direction)
            robot_radius: Robot radius
            
        Returns:
            (edge_points, mid_points) both in local coordinates
        """
        n = self.points_per_face
        offset = robot_radius + self.contact_offset  # robot_size + 0.02
        
        # Linear sampling helper
        def sample_lin(a: float, b: float, n: int, i: int) -> float:
            if n <= 1:
                return (a + b) * 0.5
            return a + (b - a) * (float(i) / float(n - 1))
        
        edge_points: List[Tuple[float, float]] = []
        
        # Top/Bottom pairs: sample along x-direction
        for j in range(n):
            u = sample_lin(-half_w, half_w, n, j)
            edge_points.append((u, half_d + offset))   # Top(j) - even index
            edge_points.append((u, -half_d - offset))  # Bottom(j) - odd index
        
        # Right/Left pairs: sample along y-direction
        for k in range(n):
            v = sample_lin(-half_d, half_d, n, k)
            edge_points.append((half_w + offset, v))   # Right(k) - even index
            edge_points.append((-half_w - offset, v))  # Left(k) - odd index
        
        # Calculate midpoints using consecutive pairing
        mid_points: List[Tuple[float, float]] = []
        for i in range(len(edge_points)):
            mate = (i + 1) if (i % 2 == 0) else (i - 1)
            mid = (
                0.5 * (edge_points[i][0] + edge_points[mate][0]),
                0.5 * (edge_points[i][1] + edge_points[mate][1])
            )
            mid_points.append(mid)
        
        return edge_points, mid_points
    
    def _transform_point(
        self,
        local: Tuple[float, float],
        cx: float,
        cy: float,
        yaw: float
    ) -> Tuple[float, float]:
        """Transform point from local to world coordinates.
        
        Args:
            local: Point in local (object) coordinates
            cx, cy: Object center in world coordinates
            yaw: Object rotation (radians)
            
        Returns:
            Point in world coordinates
        """
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        
        world_x = cx + local[0] * cos_y - local[1] * sin_y
        world_y = cy + local[0] * sin_y + local[1] * cos_y
        
        return (world_x, world_y)
    
    def get_precontact_position(
        self,
        contact: EdgeContact,
        approach_distance: float
    ) -> Tuple[float, float]:
        """Get pre-contact position (backed off from contact).
        
        Args:
            contact: Edge contact information
            approach_distance: Distance to back off from contact
            
        Returns:
            (x, y) pre-contact position
        """
        # Back off from contact in opposite direction of push
        x = contact.contact_xy[0] - approach_distance * contact.push_dir[0]
        y = contact.contact_xy[1] - approach_distance * contact.push_dir[1]
        return (x, y)


def compute_push_direction_from_poses(
    robot_pose: SE2Pose,
    object_pose: SE2Pose
) -> Tuple[float, float]:
    """Compute push direction from robot toward object.
    
    This is used during push execution to dynamically update push direction.
    
    Args:
        robot_pose: Current robot pose
        object_pose: Current object pose
        
    Returns:
        Unit vector (dx, dy) from robot toward object
    """
    dx = object_pose.x - robot_pose.x
    dy = object_pose.y - robot_pose.y
    dist = math.sqrt(dx * dx + dy * dy)
    
    if dist < 1e-6:
        return (1.0, 0.0)  # fallback
    
    return (dx / dist, dy / dist)
