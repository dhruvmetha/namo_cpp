"""
NAMO Image Converter - Adapted for NAMO environment state representation.

This converter takes NAMO environment state and converts it to image format
compatible with the diffusion models from the learning system.
"""

# mujoco import removed - using only environment interface
import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Any
import namo_rl


class NAMOImageConverter:
    """Converts NAMO environment state to image format for diffusion models."""
    
    IMG_SIZE = 224
    
    def __init__(self, env: namo_rl.RLEnvironment):
        """Initialize converter using only environment interface.
        
        Args:
            env: NAMO environment instance
        """
        self.env = env
        
        # Get all object info from environment (includes movable objects, static objects, and robot)
        self.object_info = env.get_object_info()
        
        # Get world bounds from environment
        world_bounds = env.get_world_bounds()
        self.x_min, self.x_max, self.y_min, self.y_max = world_bounds
        
        self.WORLD_WIDTH = self.x_max - self.x_min
        self.WORLD_HEIGHT = self.y_max - self.y_min
        
        # Use larger dimension to maintain square images
        self.WORLD_SIZE = max(self.WORLD_WIDTH, self.WORLD_HEIGHT)
        self.SCALE = self.IMG_SIZE / self.WORLD_SIZE  # pixels per world unit
        
        print(f"NAMOImageConverter initialized:")
        print(f"  World bounds: x=[{self.x_min:.2f}, {self.x_max:.2f}], y=[{self.y_min:.2f}, {self.y_max:.2f}]")
        print(f"  World size: {self.WORLD_WIDTH:.2f} x {self.WORLD_HEIGHT:.2f}")
        print(f"  Using square world size: {self.WORLD_SIZE:.2f}")
        print(f"  Scale: {self.SCALE:.2f} pixels per world unit")
    
    def _is_static_object(self, obj_name: str) -> bool:
        """Check if an object is static based on its name and available data.
        
        Args:
            obj_name: Name of the object
            
        Returns:
            True if object is static (has position/orientation data in object_info)
        """
        obj_data = self.object_info.get(obj_name, {})
        # Static objects have position and orientation data, movable objects only have sizes
        return 'pos_x' in obj_data and 'quat_w' in obj_data
    
    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates.
        
        Args:
            x, y: World coordinates
            
        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        # Center the world bounds in the image
        world_center_x = (self.x_min + self.x_max) / 2
        world_center_y = (self.y_min + self.y_max) / 2
        
        # For rectangular environments, center properly within square image
        pixel_center = self.IMG_SIZE // 2
        
        # Convert to pixel coordinates (Y axis flipped for image coordinates)
        pixel_x = int(pixel_center + (x - world_center_x) * self.SCALE)
        pixel_y = int(pixel_center - (y - world_center_y) * self.SCALE)  # Flip Y axis
        
        # Clamp to image bounds
        pixel_x = max(0, min(self.IMG_SIZE - 1, pixel_x))
        pixel_y = max(0, min(self.IMG_SIZE - 1, pixel_y))
        
        return pixel_x, pixel_y
    
    def _draw_circle(self, image: np.ndarray, center_x: float, center_y: float, 
                    radius: float, value: float = 1.0) -> None:
        """Draw a filled circle on the image.
        
        Args:
            image: Image array to draw on
            center_x, center_y: World coordinates of circle center
            radius: Circle radius in world units
            value: Pixel value to set (0.0 to 1.0)
        """
        pixel_x, pixel_y = self._world_to_pixel(center_x, center_y)
        pixel_radius = max(1, int(radius * self.SCALE))
        
        cv2.circle(image, (pixel_x, pixel_y), pixel_radius, value, -1)
    
    def _draw_box(self, image: np.ndarray, center_x: float, center_y: float,
                 half_width: float, half_height: float, value: float = 1.0) -> None:
        """Draw a filled rectangle on the image.
        
        Args:
            image: Image array to draw on
            center_x, center_y: World coordinates of box center
            half_width, half_height: Half-extents in world units
            value: Pixel value to set (0.0 to 1.0)
        """
        # Convert corners to pixel coordinates
        x1, y1 = self._world_to_pixel(center_x - half_width, center_y - half_height)
        x2, y2 = self._world_to_pixel(center_x + half_width, center_y + half_height)
        
        # Ensure proper ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), value, -1)
    
    def _draw_rotated_box(self, image: np.ndarray, center_x: float, center_y: float,
                         half_width: float, half_height: float, angle_rad: float, 
                         value: float = 1.0) -> None:
        """Draw a filled rotated rectangle on the image.
        
        Args:
            image: Image array to draw on
            center_x, center_y: World coordinates of box center
            half_width, half_height: Half-extents in world units
            angle_rad: Rotation angle in radians
            value: Pixel value to set (0.0 to 1.0)
        """
        # Convert center to pixel coordinates
        center_px, center_py = self._world_to_pixel(center_x, center_y)
        
        # Convert sizes to pixels
        half_width_px = half_width * self.SCALE
        half_height_px = half_height * self.SCALE
        
        # Create box corners in local coordinates (before rotation)
        corners = np.array([
            [-half_width_px, -half_height_px],
            [half_width_px, -half_height_px], 
            [half_width_px, half_height_px],
            [-half_width_px, half_height_px]
        ], dtype=np.float32)
        
        # Rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Rotate corners
        rotated_corners = corners @ rotation_matrix.T
        
        # Translate to center position
        rotated_corners[:, 0] += center_px
        rotated_corners[:, 1] += center_py
        
        # Convert to integer coordinates
        points = rotated_corners.astype(np.int32)
        
        # Draw filled polygon
        cv2.fillPoly(image, [points], value)
    
    def convert_state_to_image(self, env: namo_rl.RLEnvironment, 
                              robot_goal: Tuple[float, float, float],
                              reachable_objects: List[str] = None) -> np.ndarray:
        """Convert NAMO environment state to multi-channel image.
        
        Args:
            env: NAMO RL environment
            robot_goal: Robot goal position (x, y, theta)
            reachable_objects: List of reachable object names (optional)
            
        Returns:
            Multi-channel image array with shape (channels, height, width)
            Channels: [robot, goal, movable, static, reachable_objects]
        """
        # Get current observations
        obs = env.get_observation()
        
        # Get reachable objects if not provided
        if reachable_objects is None:
            reachable_objects = env.get_reachable_objects()
        
        # Initialize channels
        # Channel 0: Robot position
        # Channel 1: Robot goal position  
        # Channel 2: Movable objects
        # Channel 3: Static walls
        # Channel 4: Reachable objects mask
        channels = np.zeros((5, self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
        
        # Channel 0: Robot position
        if 'robot_pose' in obs:
            robot_x, robot_y = obs['robot_pose'][0], obs['robot_pose'][1]
            self._draw_circle(channels[0], robot_x, robot_y, 0.15, 1.0)  # Robot radius ~0.15
        
        # Channel 1: Robot goal
        goal_x, goal_y = robot_goal[0], robot_goal[1]
        self._draw_circle(channels[1], goal_x, goal_y, 0.1, 1.0)  # Goal marker
        
        # Channel 2: Movable objects
        for key, pose in obs.items():
            if key.endswith('_pose') and key != 'robot_pose':
                obj_name = key[:-5]  # Remove '_pose' suffix
                obj_x, obj_y, obj_theta = pose[0], pose[1], pose[2]
                
                # Get object geometry from environment interface
                if obj_name in self.object_info:
                    obj_info = self.object_info[obj_name]
                    half_width = obj_info['size_x']   # X half-extent
                    half_height = obj_info['size_y']  # Y half-extent
                    
                    # Draw rotated rectangle with actual dimensions
                    self._draw_rotated_box(channels[2], obj_x, obj_y, 
                                         half_width, half_height, obj_theta, 1.0)
                else:
                    # Fallback to simple box if geometry not found
                    self._draw_box(channels[2], obj_x, obj_y, 0.1, 0.1, 1.0)
        
        # Channel 3: Static walls (from environment interface)
        for obj_name, obj_info in self.object_info.items():
            if self._is_static_object(obj_name):
                # Static objects have position and orientation data
                pos_x = obj_info['pos_x']
                pos_y = obj_info['pos_y']
                half_width = obj_info['size_x']
                half_height = obj_info['size_y']
                
                # Extract rotation from quaternion
                qw, qx, qy, qz = obj_info['quat_w'], obj_info['quat_x'], obj_info['quat_y'], obj_info['quat_z']
                # Convert quaternion to yaw (rotation around z-axis)
                theta = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                
                # Draw rotated rectangle for static objects
                self._draw_rotated_box(channels[3], pos_x, pos_y, 
                                     half_width, half_height, theta, 1.0)
        
        # Channel 4: Reachable objects
        reachable_set = set(reachable_objects)
        for key, pose in obs.items():
            if key.endswith('_pose') and key != 'robot_pose':
                obj_name = key[:-5]
                if obj_name in reachable_set:
                    obj_x, obj_y, obj_theta = pose[0], pose[1], pose[2]
                    
                    # Get object geometry from environment interface
                    if obj_name in self.object_info:
                        obj_info = self.object_info[obj_name]
                        half_width = obj_info['size_x']   # X half-extent
                        half_height = obj_info['size_y']  # Y half-extent
                        
                        # Draw rotated rectangle with actual dimensions
                        self._draw_rotated_box(channels[4], obj_x, obj_y, 
                                             half_width, half_height, obj_theta, 1.0)
                    else:
                        # Fallback to simple box if geometry not found
                        self._draw_box(channels[4], obj_x, obj_y, 0.1, 0.1, 1.0)
        
        return channels
    
    def save_image_visualization(self, image_channels: np.ndarray, 
                                output_path: str) -> None:
        """Save visualization of multi-channel image.
        
        Args:
            image_channels: Multi-channel image array
            output_path: Path to save visualization
        """
        # Create visualization with all channels
        fig_height = 2
        fig_width = image_channels.shape[0]  # Number of channels
        
        combined = np.zeros((fig_height * self.IMG_SIZE, 
                           fig_width * self.IMG_SIZE), dtype=np.uint8)
        
        channel_names = ['Robot', 'Goal', 'Movable', 'Static', 'Reachable']
        
        # Place individual channels
        for i, (channel, name) in enumerate(zip(image_channels, channel_names)):
            # Convert to 0-255 range
            channel_viz = (channel * 255).astype(np.uint8)
            
            x_start = i * self.IMG_SIZE
            combined[0:self.IMG_SIZE, x_start:x_start + self.IMG_SIZE] = channel_viz
        
        # Create composite view (all channels overlaid with different colors)
        composite = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
        colors = [(255, 0, 0),    # Robot: Red
                 (0, 255, 0),     # Goal: Green  
                 (0, 0, 255),     # Movable: Blue
                 (128, 128, 128), # Static: Gray
                 (255, 255, 0)]   # Reachable: Yellow
        
        for i, (channel, color) in enumerate(zip(image_channels, colors)):
            mask = channel > 0.5
            for c in range(3):
                composite[:, :, c][mask] = color[c]
        
        # Place composite at bottom
        composite_gray = cv2.cvtColor(composite, cv2.COLOR_RGB2GRAY)
        combined[self.IMG_SIZE:2*self.IMG_SIZE, 0:self.IMG_SIZE] = composite_gray
        
        # Save
        cv2.imwrite(output_path, combined)
        print(f"Image visualization saved to: {output_path}")


def test_image_conversion():
    """Quick test function to verify image conversion works."""
    print("Testing NAMO Image Converter...")
    
    # Use absolute paths
    xml_path = "/common/home/dm1487/robotics_research/ktamp/namo/data/test_scene.xml"
    config_path = "/common/home/dm1487/robotics_research/ktamp/namo/config/namo_config_complete.yaml"
    
    # Initialize environment
    env = namo_rl.RLEnvironment(xml_path, config_path)
    
    # Create converter with environment instance
    converter = NAMOImageConverter(xml_path, env)
    
    # Set a robot goal
    robot_goal = (3.0, 3.0, 0.0)
    
    # Debug: Print observation format
    obs = env.get_observation()
    print(f"\nDEBUG: env.get_observation() format:")
    print(f"Type: {type(obs)}")
    print(f"Keys: {list(obs.keys()) if hasattr(obs, 'keys') else 'Not a dict'}")
    for key, value in obs.items():
        print(f"  {key}: {value} (type: {type(value)})")
    
    # Test reachable objects
    reachable = env.get_reachable_objects()
    print(f"\nReachable objects: {reachable}")
    print(f"Reachable objects type: {type(reachable)}")
    
    # Convert current state to image
    image_channels = converter.convert_state_to_image(env, robot_goal)
    
    print(f"\nGenerated image with shape: {image_channels.shape}")
    print(f"Channel value ranges:")
    for i, channel in enumerate(image_channels):
        print(f"  Channel {i}: min={channel.min():.3f}, max={channel.max():.3f}, "
              f"nonzero={np.count_nonzero(channel)}")
    
    # Save visualization
    converter.save_image_visualization(image_channels, "test_namo_image.png")
    
    return image_channels


def test_ml4kp_environment():
    """Test with ML4KP environment that has multiple objects."""
    print("Testing NAMO Image Converter with ML4KP environment...")
    
    # Use ML4KP environment
    xml_path = "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100a.xml"
    config_path = "/common/home/dm1487/robotics_research/ktamp/namo/config/namo_config_complete.yaml"
    
    print(f"Loading XML: {xml_path}")
    
    # Initialize environment
    env = namo_rl.RLEnvironment(xml_path, config_path)
    
    # Create converter with environment instance
    converter = NAMOImageConverter(xml_path, env)
    
    # Set robot goal (read from the site in XML or set manually)
    robot_goal = (2.372330825787018, 2.758425751277106, 0.0)  # From XML goal site
    
    # Debug: Print observation format
    obs = env.get_observation()
    print(f"\nDEBUG: env.get_observation() format:")
    print(f"Type: {type(obs)}")
    print(f"Keys: {list(obs.keys()) if hasattr(obs, 'keys') else 'Not a dict'}")
    for key, value in obs.items():
        print(f"  {key}: {value} (type: {type(value)})")
    
    # Test reachable objects
    reachable = env.get_reachable_objects()
    print(f"\nReachable objects: {reachable}")
    print(f"Reachable objects type: {type(reachable)}")
    print(f"Number of reachable objects: {len(reachable)}")
    
    # Test object info interface
    object_info = env.get_object_info()
    print(f"\nObject info from environment:")
    for obj_name, info in object_info.items():
        print(f"  {obj_name}: {info}")
    
    # Convert current state to image
    image_channels = converter.convert_state_to_image(env, robot_goal)
    
    print(f"\nGenerated image with shape: {image_channels.shape}")
    print(f"Channel value ranges:")
    for i, channel in enumerate(image_channels):
        print(f"  Channel {i}: min={channel.min():.3f}, max={channel.max():.3f}, "
              f"nonzero={np.count_nonzero(channel)}")
    
    # Save visualization
    converter.save_image_visualization(image_channels, "test_ml4kp_namo_image.png")
    
    return image_channels


if __name__ == "__main__":
    # Test with simple environment first
    print("=== Testing with simple NAMO environment ===")
    test_image_conversion()
    
    print("\n" + "="*60)
    print("=== Testing with ML4KP environment ===")
    test_ml4kp_environment()