"""
MCTS Mask Generator for Neural Network Training

This module converts MCTS/AlphaZero data into mask-based datasets
suitable for training neural networks on NAMO planning tasks.

Key datasets generated:
1. Goal Proposal: P(goal|object,state) as spatial heatmaps
2. Value Networks: V(s) and Q(s,a) with mask input features
3. TODO: Object Selection P(o|s) - mask representation TBD

Uses existing NAMODataVisualizer for all mask generation.
"""

import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Use existing mask generation infrastructure  
sys.path.insert(0, '/common/home/dm1487/robotics_research/ktamp/learning/src')
from mask_generation.visualizer import NAMODataVisualizer


class MCTSMaskGenerator:
    """Generates mask-based training data from MCTS episodes."""
    
    def __init__(self, xml_file: str, config_file: str):
        """Initialize with environment configuration."""
        self.xml_file = xml_file
        self.config_file = config_file
        
        # Initialize existing visualizer (reuse all infrastructure)
        self.visualizer = NAMODataVisualizer()
        
        # Get image dimensions 
        self.image_width = 224  # Standard NAMO image size
        self.image_height = 224
        
    def generate_goal_proposal_data(self, 
                                  scene_observation: Dict[str, Any],
                                  robot_goal: Tuple[float, float, float],
                                  object_id: str,
                                  goal_proposals: List[Dict],
                                  post_action_poses: Optional[Dict[str, List[float]]] = None,
                                  static_object_info: Optional[Dict] = None) -> List[Dict[str, np.ndarray]]:
        """
        Generate goal proposal data for diffusion model training.
        
        Args:
            scene_observation: Current scene state (object poses)
            robot_goal: Target robot position
            object_id: Object being pushed
            goal_proposals: List of goal proposals with probabilities and visit counts
            post_action_poses: Optional post-action SE(2) poses from executed actions
            
        Returns:
            List of training samples, one per goal proposal
        """
        training_samples = []
        
        for i, goal_data in enumerate(goal_proposals):
            goal_pos = goal_data['goal_position']  # (x, y, theta)
            probability = goal_data['probability']
            visit_count = goal_data['visit_count'] 
            q_value = goal_data['q_value']
            
            # Create fake episode data compatible with existing visualizer
            fake_episode = {
                'xml_file': self.xml_file,
                'robot_goal': robot_goal,
                'action_sequence': [{'object_id': object_id, 'target': goal_pos}],
                'state_observations': [scene_observation],  # Initial state
                'static_object_info': static_object_info or {}  # CRITICAL: Include object sizes
            }
            
            # Use existing visualizer to generate all masks
            masks = self.visualizer.generate_episode_masks(fake_episode)
            
            # Generate post-action masks if we have post-action poses
            post_action_masks = {}
            if post_action_poses is not None:
                post_action_masks = self._generate_post_action_masks(
                    post_action_poses, robot_goal, static_object_info
                )
            
            # Combine masks with MCTS statistics
            sample = {
                # All standard masks from existing visualizer
                **masks,
                
                # Post-action masks (NEW!)
                **post_action_masks,
                
                # MCTS quality signals for conditional diffusion
                'goal_probability': np.float32(probability),
                'goal_q_value': np.float32(q_value),
                'goal_visit_count': np.int32(visit_count),
                
                # Metadata
                'object_id': object_id,
                'proposal_index': i,
                'goal_coordinates': np.array(goal_pos, dtype=np.float32)
            }
            training_samples.append(sample)
        
        return training_samples
    
    def _generate_post_action_masks(self, 
                                  post_action_poses: Dict[str, List[float]],
                                  robot_goal: Tuple[float, float, float],
                                  static_object_info: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Generate masks showing where objects ended up after the action."""
        
        # FIXED: Use existing visualizer but with post-action data
        fake_post_episode = {
            'xml_file': self.xml_file,
            'robot_goal': robot_goal,
            'action_sequence': [],  # No actions needed, just final state
            'state_observations': [post_action_poses],  # Direct pose data
            'static_object_info': static_object_info or {}  # CRITICAL: Include object sizes
        }
        
        # Generate masks for post-action state using existing visualizer
        post_masks = self.visualizer.generate_episode_masks(fake_post_episode)
        
        # Rename masks to indicate they're post-action
        post_action_masks = {}
        for key, mask in post_masks.items():
            if key in ['robot', 'movable', 'robot_distance']:
                post_action_masks[f'post_action_{key}'] = mask
        
        print(f"✅ Generated post-action masks: robot_sum={post_action_masks.get('post_action_robot', np.zeros((224,224))).sum():.1f}, movable_sum={post_action_masks.get('post_action_movable', np.zeros((224,224))).sum():.1f}")
        return post_action_masks
    
    def generate_value_network_data(self,
                                  scene_observation: Dict[str, Any], 
                                  robot_goal: Tuple[float, float, float],
                                  state_value: float,
                                  object_q_values: Dict[str, float],
                                  selected_object_id: Optional[str] = None,
                                  selected_goal: Optional[Tuple[float, float, float]] = None) -> Dict[str, np.ndarray]:
        """
        Generate input features and targets for value networks V(s) and Q(s,a).
        
        Args:
            scene_observation: Current scene state  
            robot_goal: Target robot position
            state_value: V(s) target from MCTS root node
            object_q_values: Q(s,object) targets from MCTS ObjectNodes
            selected_object_id: For Q(s,a) - which object was selected
            selected_goal: For Q(s,a) - which goal was selected
            
        Returns:
            Dictionary with mask features and value targets
        """
        # Create fake episode for existing visualizer 
        action = [{'object_id': selected_object_id, 'target': selected_goal}] if selected_object_id else []
        fake_episode = {
            'xml_file': self.xml_file,
            'robot_goal': robot_goal,
            'action_sequence': action,
            'state_observations': [scene_observation]
        }
        
        # Use existing visualizer to generate masks
        masks = self.visualizer.generate_episode_masks(fake_episode)
        
        data = {
            # All standard masks from existing visualizer
            **masks,
            
            # Value targets
            'state_value': np.float32(state_value),  # V(s) target
            'object_q_values': object_q_values,      # Q(s,object) targets
            
            # Metadata
            'selected_object_id': selected_object_id,
            'selected_goal': np.array(selected_goal, dtype=np.float32) if selected_goal else None
        }
        
        return data
    
    def generate_object_selection_data(self,
                                     scene_observation: Dict[str, Any],
                                     robot_goal: Tuple[float, float, float], 
                                     object_proposals: List[Dict]) -> Dict[str, np.ndarray]:
        """
        TODO: Generate training data for object selection P(o|s) network.
        
        This needs more thought on how to represent object selection with masks.
        Possible approaches:
        1. Binary masks per object with selection probabilities  
        2. Object-centric feature extraction from masks
        3. Attention mechanism over object regions
        4. Graph neural network with object nodes
        
        For now, placeholder that stores raw data for future design.
        """
        base_masks = self.base_converter.generate_masks_from_observation(
            observation=scene_observation,
            robot_goal=robot_goal
        )
        
        return {
            # Input features
            'scene_mask': base_masks['combined_mask'],
            'robot_mask': base_masks['robot_mask'],
            'object_masks': base_masks['object_masks'],
            'robot_goal_mask': base_masks['robot_goal_mask'],
            
            # Raw object proposals (not mask-based yet)
            'object_proposals': object_proposals,  # TODO: Convert to mask representation
            
            # Metadata
            'robot_goal': np.array(robot_goal, dtype=np.float32),
            'world_bounds': np.array(self.base_converter.world_bounds, dtype=np.float32),
            'TODO': 'Object selection mask representation needs design'
        }
    
    def _create_target_object_mask(self, scene_observation: Dict[str, Any], object_id: str) -> np.ndarray:
        """Create target_object mask showing object at current position."""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        
        # Get object pose from scene observation
        pose_key = f"{object_id}_pose"
        if pose_key in scene_observation:
            obj_pose = scene_observation[pose_key]
            x, y, theta = obj_pose[0], obj_pose[1], obj_pose[2]
            
            # Get object size from base converter's static info
            static_info = self.base_converter.get_object_info()
            if object_id in static_info and 'size_x' in static_info[object_id]:
                size_x = static_info[object_id]['size_x']
                size_y = static_info[object_id]['size_y']
                
                # Draw rotated box at current position
                self._draw_rotated_box_mask(mask, x, y, size_x, size_y, theta, 1.0)
        
        return mask
    
    def _create_target_goal_mask(self, object_id: str, goal_pos: Tuple[float, float, float]) -> np.ndarray:
        """Create target_goal mask showing object at proposed goal position."""
        mask = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        
        goal_x, goal_y, goal_theta = goal_pos
        
        # Get object size from base converter's static info
        static_info = self.base_converter.get_object_info()
        if object_id in static_info and 'size_x' in static_info[object_id]:
            size_x = static_info[object_id]['size_x']
            size_y = static_info[object_id]['size_y']
            
            # Draw rotated box at goal position
            self._draw_rotated_box_mask(mask, goal_x, goal_y, size_x, size_y, goal_theta, 1.0)
        
        return mask
    
    def _draw_rotated_box_mask(self, mask: np.ndarray, x: float, y: float, 
                              size_x: float, size_y: float, theta: float, intensity: float):
        """Draw rotated rectangle in mask (similar to existing visualizer)."""
        # Convert world coordinates to image coordinates
        center_img_x, center_img_y = self.base_converter.world_to_image(x, y)
        
        # Convert world size to image size
        world_bounds = self.base_converter.world_bounds
        world_width = world_bounds[2] - world_bounds[0] 
        world_height = world_bounds[3] - world_bounds[1]
        
        size_img_x = int(size_x * self.image_width / world_width)
        size_img_y = int(size_y * self.image_height / world_height)
        
        # Create rotated rectangle points
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Half sizes
        half_x = size_img_x / 2
        half_y = size_img_y / 2
        
        # Corner points (relative to center)
        corners = [
            [-half_x, -half_y],
            [half_x, -half_y], 
            [half_x, half_y],
            [-half_x, half_y]
        ]
        
        # Rotate and translate corners
        rotated_corners = []
        for corner_x, corner_y in corners:
            rot_x = corner_x * cos_theta - corner_y * sin_theta + center_img_x
            rot_y = corner_x * sin_theta + corner_y * cos_theta + center_img_y
            rotated_corners.append([int(rot_x), int(rot_y)])
        
        # Fill polygon
        import cv2
        cv2.fillPoly(mask, [np.array(rotated_corners)], intensity)
    
    def _add_gaussian_blob(self, image: np.ndarray, x: int, y: int, 
                          intensity: float, sigma: int):
        """Add Gaussian blob to image at specified location."""
        h, w = image.shape
        
        # Create coordinate grids
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Gaussian function
        gaussian = intensity * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        
        # Add to image
        image += gaussian
    
    def _generate_scene_masks(self, scene_observation: Dict[str, Any], 
                            robot_goal: Tuple[float, float, float]) -> Dict[str, np.ndarray]:
        """Generate scene masks using NAMOImageConverter."""
        
        # Set environment state to match the scene observation
        if hasattr(self.env, 'set_observation_state'):
            self.env.set_observation_state(scene_observation)
        else:
            # Fallback: set individual object poses
            for key, value in scene_observation.items():
                if key.endswith('_pose') and len(value) >= 3:
                    obj_name = key.replace('_pose', '')
                    # This is a basic fallback - may need adjustment based on actual API
                    try:
                        self.env.set_object_pose(obj_name, value[0], value[1], value[2])
                    except:
                        pass  # Skip if method doesn't exist or object not found
        
        # Get multi-channel image from converter
        multi_channel = self.image_converter.convert_state_to_image(self.env, robot_goal)
        
        # Extract individual channel masks to match existing interface
        masks = {
            'robot_mask': multi_channel[0],      # Channel 0: Robot position
            'robot_goal_mask': multi_channel[1], # Channel 1: Robot goal position  
            'movable': multi_channel[2],         # Channel 2: Movable objects
            'static': multi_channel[3],          # Channel 3: Static walls
            'reachable': multi_channel[4],       # Channel 4: Reachable objects mask
        }
        
        # Draw robot position from scene observation
        if 'robot_pose' in scene_observation:
            robot_pose = scene_observation['robot_pose']
            self._draw_circle_mask(masks['robot_mask'], robot_pose[0], robot_pose[1], 0.2, world_bounds, 1.0)
        
        # Draw objects from scene observation (same logic as existing visualizer)
        for obj_name, pose in scene_observation.items():
            if obj_name != 'robot_pose' and obj_name.endswith('_pose'):
                obj_base_name = obj_name.replace('_pose', '')
                obj_info = static_object_info.get(obj_base_name, {})
                
                if 'size_x' in obj_info and 'size_y' in obj_info:
                    x, y, theta = pose[0], pose[1], pose[2]
                    size_x = obj_info['size_x']
                    size_y = obj_info['size_y']
                    
                    # Draw in movable objects mask (same as existing)
                    self._draw_rotated_box_mask_existing_style(
                        masks['movable'], x, y, size_x, size_y, theta, world_bounds, 1.0
                    )
        
        # Generate reachable objects mask (same as existing)
        masks['reachable'] = masks['movable'].copy()
        
        # Generate distance fields (same algorithms as existing visualizer)
        if 'robot_pose' in scene_observation:
            robot_pose = scene_observation['robot_pose']
            masks['robot_distance'] = self._compute_distance_field_existing_style(
                robot_pose[0], robot_pose[1], masks['movable'], world_bounds
            )
        else:
            masks['robot_distance'] = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            
        masks['goal_distance'] = self._compute_distance_field_existing_style(
            robot_goal[0], robot_goal[1], masks['movable'], world_bounds
        )
        
        return masks
    
    def _draw_circle_mask(self, mask: np.ndarray, center_x: float, center_y: float, 
                         radius: float, world_bounds: Tuple[float, float, float, float], value: float = 1.0) -> None:
        """Draw a filled circle on mask (EXACT COPY from existing visualizer)."""
        # Convert center to pixel coordinates
        center_px, center_py = self._world_to_pixel(center_x, center_y, world_bounds)
        
        # Convert radius to pixel coordinates
        scale = self._get_pixel_scale(world_bounds)
        radius_px = int(radius * scale)
        
        # Convert mask to uint8 for cv2, draw, then convert back
        import cv2
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.circle(mask_uint8, (center_px, center_py), radius_px, int(value * 255), -1)
        
        # Convert back to float and update original mask
        mask[:] = mask_uint8.astype(np.float32) / 255.0
    
    def _draw_rotated_box_mask_existing_style(self, mask: np.ndarray, center_x: float, center_y: float,
                                            size_x: float, size_y: float, theta: float, 
                                            world_bounds: Tuple[float, float, float, float], value: float = 1.0) -> None:
        """Draw rotated box (EXACT COPY from existing visualizer)."""
        # Convert center to pixel coordinates
        center_px, center_py = self._world_to_pixel(center_x, center_y, world_bounds)
        
        # Convert size to pixel coordinates
        scale = self._get_pixel_scale(world_bounds)
        size_px_x = int(size_x * scale)
        size_px_y = int(size_y * scale)
        
        # Create rectangle points
        rect = ((center_px, center_py), (size_px_x, size_px_y), np.degrees(theta))
        box = cv2.boxPoints(rect).astype(int)
        
        # Draw filled polygon
        import cv2
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.fillPoly(mask_uint8, [box], int(value * 255))
        
        # Convert back to float
        mask[:] = mask_uint8.astype(np.float32) / 255.0
    
    def _world_to_pixel(self, x: float, y: float, world_bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates (EXACT COPY from existing visualizer)."""
        x_min, y_min, x_max, y_max = world_bounds
        
        # Normalize to [0, 1]
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
        
        # Convert to pixel coordinates (flip y-axis for image coordinates)
        pixel_x = int(x_norm * 224)
        pixel_y = int((1.0 - y_norm) * 224)  # Flip y-axis
        
        # Clamp to valid range
        pixel_x = max(0, min(223, pixel_x))
        pixel_y = max(0, min(223, pixel_y))
        
        return pixel_x, pixel_y
    
    def _get_pixel_scale(self, world_bounds: Tuple[float, float, float, float]) -> float:
        """Calculate pixels per world unit (EXACT COPY from existing visualizer)."""
        x_min, y_min, x_max, y_max = world_bounds
        world_width = x_max - x_min
        world_height = y_max - y_min
        
        # Use minimum dimension for uniform scaling
        scale_x = 224 / world_width
        scale_y = 224 / world_height
        
        return min(scale_x, scale_y)
    
    def _compute_distance_field_existing_style(self, start_x: float, start_y: float,
                                             obstacles_mask: np.ndarray, 
                                             world_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Compute distance field (SIMPLIFIED version of existing visualizer)."""
        # For now, return zeros - full implementation would require BFS like existing visualizer
        # This would need the full algorithm from the existing visualizer's _compute_distance_field
        return np.zeros_like(obstacles_mask)