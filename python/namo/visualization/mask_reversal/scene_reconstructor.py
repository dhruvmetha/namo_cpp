"""
Scene Reconstructor - Convert pixel coordinates back to world coordinates.

This module transforms detected objects from pixel space back to world space
using the inverse of the coordinate transform used in mask generation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

from .mask_analyzer import DetectedObject, DetectedWall


@dataclass
class WorldObject:
    """Object in world coordinates."""
    
    # Position in world coordinates
    x: float
    y: float
    theta: float  # Rotation in radians
    
    # Size in world units (half-extents)
    half_width: float
    half_height: float
    
    # Object type
    obj_type: str
    
    # Optional: radius for circular objects
    radius: Optional[float] = None


@dataclass
class ReconstructedScene:
    """Complete reconstructed scene in world coordinates."""
    
    robot_position: Tuple[float, float, float]  # (x, y, theta)
    goal_position: Tuple[float, float, float]   # (x, y, theta)
    movable_objects: List[WorldObject]
    static_objects: List[WorldObject]
    
    # World bounds (estimated or provided)
    world_bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)


class SceneReconstructor:
    """Reconstructs world-space scene from pixel-space detections."""
    
    IMG_SIZE = 224
    
    def __init__(self, world_bounds: Optional[Tuple[float, float, float, float]] = None,
                 default_world_size: float = 11.0):
        """Initialize reconstructor.
        
        Args:
            world_bounds: Known world bounds (x_min, x_max, y_min, y_max).
                         If None, will estimate from object positions.
            default_world_size: Default world size to use if bounds unknown (default 11.0)
        """
        self.world_bounds = world_bounds
        self.default_world_size = default_world_size
        
        if world_bounds is not None:
            x_min, x_max, y_min, y_max = world_bounds
            self.world_width = x_max - x_min
            self.world_height = y_max - y_min
            self.world_size = max(self.world_width, self.world_height)
            self.scale = self.IMG_SIZE / self.world_size
            self.world_center_x = (x_min + x_max) / 2
            self.world_center_y = (y_min + y_max) / 2
        else:
            # Use default symmetric bounds
            self.world_size = default_world_size
            self.scale = self.IMG_SIZE / self.world_size
            self.world_center_x = 0.0
            self.world_center_y = 0.0
            half_size = self.world_size / 2
            self.world_bounds = (-half_size, half_size, -half_size, half_size)
    
    def pixel_to_world(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates.
        
        This is the inverse of the _world_to_pixel transform in NAMOImageConverter.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            
        Returns:
            (world_x, world_y) in world coordinates
        """
        pixel_center = self.IMG_SIZE // 2
        
        # Inverse transform (note Y axis flip)
        world_x = self.world_center_x + (pixel_x - pixel_center) / self.scale
        world_y = self.world_center_y - (pixel_y - pixel_center) / self.scale  # Flip Y axis
        
        return world_x, world_y
    
    def pixel_size_to_world_size(self, pixel_size: float) -> float:
        """Convert pixel size to world size.
        
        Args:
            pixel_size: Size in pixels
            
        Returns:
            Size in world units
        """
        return pixel_size / self.scale
    
    def reconstruct_circle(self, detected: DetectedObject) -> WorldObject:
        """Reconstruct circular object (robot, goal).
        
        Args:
            detected: Detected circular object
            
        Returns:
            WorldObject in world coordinates
        """
        # Convert center
        world_x, world_y = self.pixel_to_world(detected.center_px[0], detected.center_px[1])
        
        # Convert radius
        radius_px = detected.width_px / 2  # diameter -> radius
        radius = self.pixel_size_to_world_size(radius_px)
        
        return WorldObject(
            x=world_x,
            y=world_y,
            theta=0.0,  # Circles have no orientation
            half_width=radius,
            half_height=radius,
            obj_type=detected.obj_type,
            radius=radius
        )
    
    def reconstruct_rectangle(self, detected: DetectedObject) -> WorldObject:
        """Reconstruct rectangular object (movable, static).
        
        Args:
            detected: Detected rectangular object
            
        Returns:
            WorldObject in world coordinates
        """
        # Convert center
        world_x, world_y = self.pixel_to_world(detected.center_px[0], detected.center_px[1])
        
        # Convert size (OpenCV gives full width/height, we want half-extents)
        half_width = self.pixel_size_to_world_size(detected.width_px / 2)
        half_height = self.pixel_size_to_world_size(detected.height_px / 2)
        
        # Convert angle (OpenCV gives degrees, we want radians)
        # Note: OpenCV angle is in range [-90, 0], need to convert properly
        theta_rad = np.deg2rad(detected.angle_deg)
        
        return WorldObject(
            x=world_x,
            y=world_y,
            theta=theta_rad,
            half_width=half_width,
            half_height=half_height,
            obj_type=detected.obj_type
        )
    
    def reconstruct_wall(self, detected: DetectedWall) -> WorldObject:
        """Reconstruct wall from line segment detection.
        
        Args:
            detected: DetectedWall from line detection
            
        Returns:
            WorldObject representing the wall as a box
        """
        # Convert center position
        world_x, world_y = self.pixel_to_world(detected.center_px[0], detected.center_px[1])
        
        # Convert length and thickness
        length_world = self.pixel_size_to_world_size(detected.length_px)
        thickness_world = self.pixel_size_to_world_size(detected.thickness_px)
        
        # For walls, we model them as thin rectangles
        # Horizontal wall: length along x-axis, thickness along y-axis
        # Vertical wall: thickness along x-axis, length along y-axis
        if detected.wall_type == 'horizontal':
            half_width = length_world / 2
            half_height = thickness_world / 2
            theta = 0.0
        else:  # vertical
            half_width = thickness_world / 2
            half_height = length_world / 2
            theta = np.pi / 2  # 90 degrees
        
        return WorldObject(
            x=world_x,
            y=world_y,
            theta=theta,
            half_width=half_width,
            half_height=half_height,
            obj_type='static'
        )
    
    def reconstruct_scene(self, detections: Dict[str, List[Union[DetectedObject, DetectedWall]]],
                         robot_goal_metadata: Optional[np.ndarray] = None) -> ReconstructedScene:
        """Reconstruct complete scene from detections.
        
        Args:
            detections: Dictionary of detected objects by type
            robot_goal_metadata: Optional robot goal from NPZ metadata [x, y, theta]
            
        Returns:
            ReconstructedScene with all objects in world coordinates
        """
        # Reconstruct robot
        robot_objects = [self.reconstruct_circle(det) for det in detections.get('robot', [])]
        if not robot_objects:
            robot_position = (0.0, 0.0, 0.0)
            print("Warning: No robot detected, using default position (0, 0, 0)")
        else:
            robot = robot_objects[0]  # Take first if multiple
            robot_position = (robot.x, robot.y, robot.theta)
        
        # Reconstruct goal
        goal_objects = [self.reconstruct_circle(det) for det in detections.get('goal', [])]
        if not goal_objects:
            # Try to use metadata if available
            if robot_goal_metadata is not None and len(robot_goal_metadata) >= 2:
                goal_position = (float(robot_goal_metadata[0]), 
                               float(robot_goal_metadata[1]),
                               float(robot_goal_metadata[2]) if len(robot_goal_metadata) >= 3 else 0.0)
                print(f"Using goal from metadata: {goal_position}")
            else:
                goal_position = (2.0, 2.0, 0.0)
                print("Warning: No goal detected, using default position (2, 2, 0)")
        else:
            goal = goal_objects[0]
            goal_position = (goal.x, goal.y, goal.theta)
        
        # Reconstruct movable objects
        movable_objects = [self.reconstruct_rectangle(det) for det in detections.get('movable', [])]
        
        # Reconstruct static objects (walls)
        static_detections = detections.get('static', [])
        static_objects = []
        for det in static_detections:
            if isinstance(det, DetectedWall):
                static_objects.append(self.reconstruct_wall(det))
            else:
                static_objects.append(self.reconstruct_rectangle(det))
        
        return ReconstructedScene(
            robot_position=robot_position,
            goal_position=goal_position,
            movable_objects=movable_objects,
            static_objects=static_objects,
            world_bounds=self.world_bounds
        )
    
    def estimate_world_bounds(self, scene: ReconstructedScene, 
                            padding: float = 0.5) -> Tuple[float, float, float, float]:
        """Estimate world bounds from reconstructed scene objects.
        
        Args:
            scene: Reconstructed scene
            padding: Padding to add around objects (in world units)
            
        Returns:
            (x_min, x_max, y_min, y_max)
        """
        # Collect all object positions
        x_coords = [scene.robot_position[0], scene.goal_position[0]]
        y_coords = [scene.robot_position[1], scene.goal_position[1]]
        
        for obj in scene.movable_objects + scene.static_objects:
            x_coords.append(obj.x)
            y_coords.append(obj.y)
        
        if not x_coords:
            return self.world_bounds
        
        x_min = min(x_coords) - padding
        x_max = max(x_coords) + padding
        y_min = min(y_coords) - padding
        y_max = max(y_coords) + padding
        
        return (x_min, x_max, y_min, y_max)
