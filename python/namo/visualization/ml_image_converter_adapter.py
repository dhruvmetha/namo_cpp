#!/usr/bin/env python3
"""
ML Image Converter Adapter

This adapter makes the UnifiedImageConverter compatible with the existing ML inference
models (ObjectInferenceModel and GoalInferenceModel) without changing their interface.

The adapter translates between the JSON message format expected by ML models and the
ObjectInfo format used by the UnifiedImageConverter.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from unified_image_converter import UnifiedImageConverter, ObjectInfo, create_converter_from_xml
from scipy.spatial.transform import Rotation as R


class MLImageConverterAdapter:
    """Adapter that makes UnifiedImageConverter compatible with ML inference models.
    
    This class provides the same interface as the original json2img.ImageConverter
    but uses the unified converter internally for consistency.
    """
    
    IMG_SIZE = 224  # Match the original interface
    
    def __init__(self, xml_path: str):
        """Initialize the ML adapter.
        
        Args:
            xml_path: Path to MuJoCo XML file (relative paths will be resolved)
        """
        self.xml_path = xml_path
        self.converter = create_converter_from_xml(xml_path)
        
        # Load object sizes from XML for compatibility
        self.object_sizes = self._load_object_sizes_from_xml(xml_path)
        
        # Store world bounds for compatibility
        self.world_bounds = self.converter.world_bounds
        
        # Expose bounds as individual attributes (for compatibility)
        self.x_min, self.x_max, self.y_min, self.y_max = self.world_bounds
        self.WORLD_WIDTH = self.x_max - self.x_min
        self.WORLD_HEIGHT = self.y_max - self.y_min
        self.WORLD_SIZE = self.converter.world_size
        self.SCALE = self.converter.scale
        
        # Store data_point for compatibility with create_object_mask
        self.data_point = None
    
    def _load_object_sizes_from_xml(self, xml_path: str) -> Dict[str, np.ndarray]:
        """Load object sizes from XML file for compatibility."""
        import mujoco
        import os
        
        # Handle relative paths
        if not os.path.isabs(xml_path):
            # Check if path starts with ../ and can be resolved directly first
            if xml_path.startswith("../ml4kp_ktamp"):
                if os.path.exists(xml_path):
                    full_xml_path = os.path.abspath(xml_path)
                else:
                    # Fallback to hardcoded resource path
                    full_xml_path = os.path.join("/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models", xml_path)
            elif os.path.exists(xml_path):
                full_xml_path = os.path.abspath(xml_path)
            else:
                full_xml_path = os.path.join("/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models", xml_path)
        else:
            full_xml_path = xml_path
        
        model = mujoco.MjModel.from_xml_path(full_xml_path)
        geom_sizes = {}
        
        for i in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name is None:
                geom_name = f"geom_{i}"
            geom_sizes[geom_name] = model.geom_size[i]
        
        del model
        return geom_sizes
    
    def print_bounds_info(self):
        """Print debugging information about bounds (for compatibility)."""
        print(f"Environment bounds: x=[{self.x_min:.2f}, {self.x_max:.2f}], y=[{self.y_min:.2f}, {self.y_max:.2f}]")
        print(f"World size: {self.WORLD_WIDTH:.2f} x {self.WORLD_HEIGHT:.2f}")
        print(f"Using square world size: {self.WORLD_SIZE:.2f}")
        print(f"Scale: {self.SCALE:.2f} pixels per world unit")
        print(f"Image center at pixel: ({self.IMG_SIZE//2}, {self.IMG_SIZE//2})")
        
        # Test corner conversions
        corners = [
            (self.x_min, self.y_min, "bottom-left"),
            (self.x_max, self.y_min, "bottom-right"), 
            (self.x_min, self.y_max, "top-left"),
            (self.x_max, self.y_max, "top-right")
        ]
        print("Corner pixel positions:")
        for x, y, name in corners:
            px, py = self.converter.world_to_pixel(x, y)
            print(f"  {name}: world({x:.2f}, {y:.2f}) -> pixel({px}, {py})")
    
    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world to pixel coordinates (for compatibility)."""
        return self.converter.world_to_pixel(x, y)
    
    def pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel to world coordinates (for compatibility)."""
        return self.converter.pixel_to_world(px, py)
    
    def _json_to_object_info_list(self, data_point: Dict[str, Any]) -> List[ObjectInfo]:
        """Convert JSON message format to ObjectInfo list."""
        objects = []
        
        for obj_name, obj_data in data_point['objects'].items():
            # Get position and rotation
            pos = obj_data['position']
            quat = obj_data['quaternion']  # [w, x, y, z] scalar-first format
            
            # Convert quaternion to Euler angle (Z rotation only)
            rotation = R.from_quat(quat, scalar_first=True).as_euler('xyz', degrees=False)[2]  # radians
            
            # Get object sizes (note: NOT doubled like in original ML converter)
            if obj_name in self.object_sizes:
                size_x = self.object_sizes[obj_name][0]  # Half-extent
                size_y = self.object_sizes[obj_name][1]  # Half-extent
            else:
                size_x = size_y = 0.1  # Default
            
            # Determine if static or movable
            is_static = "movable" not in obj_name
            is_reachable = obj_name in data_point.get('reachable_objects', [])
            
            obj_info = ObjectInfo(
                name=obj_name,
                x=pos[0],
                y=pos[1], 
                theta=rotation,
                size_x=size_x,
                size_y=size_y,
                is_static=is_static,
                is_reachable=is_reachable
            )
            objects.append(obj_info)
        
        return objects
    
    def process_datapoint(self, data_point: Dict[str, Any], robot_goal_pos: Tuple[float, float]) -> Dict[str, Any]:
        """Process data point to create multi-channel images (main ML interface).
        
        This method provides the same interface as the original ImageConverter
        but uses the unified converter internally.
        
        Args:
            data_point: JSON message from planning system
            robot_goal_pos: Robot goal position [x, y]
            
        Returns:
            Dictionary with image channels and object center pixels
        """
        # Store data_point for compatibility with create_object_mask
        self.data_point = data_point
        
        # Extract robot position
        robot_pos = data_point['robot']['position']
        robot_position = (robot_pos[0], robot_pos[1], 0.0)  # Assume theta=0 if not provided
        robot_goal = (robot_goal_pos[0], robot_goal_pos[1], 0.0)
        
        # Convert to ObjectInfo list
        objects = self._json_to_object_info_list(data_point)
        
        # Create multi-channel image
        channels = self.converter.create_multi_channel_image(robot_position, robot_goal, objects)
        
        # Get object center pixels (for ML model compatibility)
        obj2center_px = self.converter.get_object_center_pixels(objects)
        
        # Convert quaternions to angles for compatibility
        obj2angle = {}
        for obj in objects:
            obj2angle[obj.name] = np.degrees(obj.theta)  # Convert to degrees for compatibility
        
        # Return in expected format (channels are already float32 in [0,1] range)
        return {
            'robot_image': channels[0:1].transpose(1, 2, 0),  # (H, W, 1)
            'goal_image': channels[1:2].transpose(1, 2, 0),   # (H, W, 1)
            'movable_objects_image': channels[2:3].transpose(1, 2, 0),  # (H, W, 1)
            'static_objects_image': channels[3:4].transpose(1, 2, 0),   # (H, W, 1)
            'reachable_objects_image': channels[4:5].transpose(1, 2, 0), # (H, W, 1)
            'obj2center_px': obj2center_px,
            'obj2angle': obj2angle
        }
    
    def create_object_mask(self, object_name: str) -> np.ndarray:
        """Create a binary mask for a specific object (for compatibility).
        
        Args:
            object_name: Name of the object to create mask for
            
        Returns:
            Binary mask with shape (IMG_SIZE, IMG_SIZE, 1)
        """
        if self.data_point is None:
            raise ValueError("data_point is not set. Call process_datapoint first.")
            
        if object_name not in self.data_point['objects']:
            raise ValueError(f"Object '{object_name}' not found in data_point")
        
        # Get object data
        obj_data = self.data_point['objects'][object_name]
        pos = obj_data['position']
        quat = obj_data['quaternion']
        
        # Convert quaternion to angle
        rotation = R.from_quat(quat, scalar_first=True).as_euler('xyz', degrees=False)[2]
        
        # Get object sizes (note: NOT doubled)
        if object_name in self.object_sizes:
            size_x = self.object_sizes[object_name][0]
            size_y = self.object_sizes[object_name][1]
        else:
            size_x = size_y = 0.1
        
        # Create mask
        mask = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
        self.converter.draw_rotated_box(mask, pos[0], pos[1], size_x, size_y, rotation, 1.0)
        
        # Return in expected format (H, W, 1)
        return mask[:, :, np.newaxis]
    
    def rotate_relative_to_world(self, obj_name: str, angle: float) -> np.ndarray:
        """Rotate object relative to world (for compatibility).
        
        Args:
            obj_name: Object name
            angle: Angle to add in degrees
            
        Returns:
            New quaternion [w, x, y, z] in scalar-first format
        """
        if self.data_point is None:
            raise ValueError("data_point is not set")
            
        obj_angle = self.data_point['objects'][obj_name]['quaternion']
        current_angle = R.from_quat(obj_angle, scalar_first=True).as_euler('xyz', degrees=True)[2]
        final_quat = R.from_euler('xyz', [0, 0, current_angle + angle], degrees=True).as_quat(scalar_first=True)
        return final_quat


# For backward compatibility, create an alias
ImageConverter = MLImageConverterAdapter