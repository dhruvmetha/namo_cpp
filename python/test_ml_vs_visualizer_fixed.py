#!/usr/bin/env python3
"""
Fixed ML Inference vs Visualizer Comparison Test

This version properly handles:
- Static wall rendering from XML
- Object rotations from XML euler angles
- Correct robot and goal positions/sizes
- Proper coordinate systems
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

# Import torch for CUDA check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Add necessary paths
sys.path.append('/common/home/dm1487/robotics_research/ktamp/namo/python')
sys.path.append('/common/home/dm1487/robotics_research/ktamp/learning')

# Import NAMO RL environment
import namo_rl

# Import our unified converter and ML strategies
from unified_image_converter import UnifiedImageConverter, ObjectInfo, create_converter_from_xml
from ml_image_converter_adapter import MLImageConverterAdapter
from mask_generation.visualizer import NAMODataVisualizer

# Import IDFS strategies
from idfs.ml_strategies import MLGoalSelectionStrategy
from idfs.object_selection_strategy import NoHeuristicStrategy


def parse_xml_properly(xml_path):
    """Parse XML file to extract all object information properly."""
    print("=== Parsing XML File ===")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    objects = {}
    robot_info = None
    goal_info = None
    
    # Parse worldbody for all objects
    worldbody = root.find('.//worldbody')
    if worldbody is not None:
        # Find walls (in walls body)
        walls_body = worldbody.find('.//body[@name="walls"]')
        if walls_body is not None:
            for geom in walls_body.findall('geom'):
                name = geom.get('name')
                pos_str = geom.get('pos', '0 0 0')
                size_str = geom.get('size', '0.1 0.1 0.1')
                
                pos = [float(x) for x in pos_str.split()]
                size = [float(x) for x in size_str.split()]
                
                objects[name] = {
                    "position": [pos[0], pos[1], pos[2]],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],  # Static walls not rotated
                    "size": size,
                    "is_static": True
                }
                print(f"  Wall {name}: pos={pos}, size={size}")
        
        # Find robot
        robot_body = worldbody.find('.//body[@name="robot"]')
        if robot_body is not None:
            robot_geom = robot_body.find('geom')
            if robot_geom is not None:
                pos_str = robot_geom.get('pos', '0 0 0')
                size_str = robot_geom.get('size', '0.15 0.15 0.15')
                
                pos = [float(x) for x in pos_str.split()]
                size = [float(x) for x in size_str.split()]
                
                robot_info = {
                    "position": [pos[0], pos[1], pos[2]],
                    "size": size[0]  # Radius for sphere
                }
                print(f"  Robot: pos={pos}, radius={size[0]}")
        
        # Find goal site
        goal_site = worldbody.find('.//site[@name="goal"]')
        if goal_site is not None:
            pos_str = goal_site.get('pos', '0 0 0')
            size_str = goal_site.get('size', '0.1 0.1 0.1')
            
            pos = [float(x) for x in pos_str.split()]
            size = [float(x) for x in size_str.split()]
            
            goal_info = {
                "position": [pos[0], pos[1], pos[2]],
                "size": size[0]  # Radius for sphere
            }
            print(f"  Goal: pos={pos}, radius={size[0]}")
        
        # Find movable objects
        for body in worldbody.findall('body'):
            name = body.get('name')
            if name and 'movable' in name:
                geom = body.find('geom')
                if geom is not None:
                    pos_str = geom.get('pos', '0 0 0')
                    size_str = geom.get('size', '0.1 0.1 0.1')
                    euler_str = geom.get('euler', '0 0 0')
                    
                    pos = [float(x) for x in pos_str.split()]
                    size = [float(x) for x in size_str.split()]
                    euler = [float(x) for x in euler_str.split()]
                    
                    # Convert euler angles to quaternion (degrees to radians)
                    rotation = R.from_euler('xyz', euler, degrees=True)
                    quat = rotation.as_quat(scalar_first=True)  # [w, x, y, z]
                    
                    objects[name] = {
                        "position": [pos[0], pos[1], pos[2]],
                        "quaternion": [quat[0], quat[1], quat[2], quat[3]],
                        "size": size,
                        "is_static": False,
                        "euler_deg": euler
                    }
                    print(f"  {name}: pos={pos}, size={size}, euler={euler}°")
    
    return objects, robot_info, goal_info


def setup_environment_fixed():
    """Set up environment with proper XML parsing."""
    print("=== Setting up Environment (Fixed) ===")
    
    xml_path = "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml"
    config_path = "/common/home/dm1487/robotics_research/ktamp/namo/config/namo_config_complete.yaml"
    
    # Parse XML properly
    all_objects, robot_info, goal_info = parse_xml_properly(xml_path)
    
    # Initialize environment
    env = namo_rl.RLEnvironment(xml_path, config_path)
    env.reset()
    initial_state = env.get_full_state()
    obs = env.get_observation()
    
    # Get reachable objects
    reachable_objects = env.get_reachable_objects()
    
    return env, initial_state, obs, all_objects, robot_info, goal_info, reachable_objects


def create_proper_ml_json(all_objects, robot_info, goal_info, reachable_objects):
    """Create proper JSON message with all objects including static walls."""
    print("=== Creating Proper ML JSON ===")
    
    json_objects = {}
    
    # Add all objects from XML
    for obj_name, obj_data in all_objects.items():
        json_objects[obj_name] = {
            "position": obj_data["position"],
            "quaternion": obj_data["quaternion"]
        }
        print(f"  Added {obj_name}: {obj_data['is_static'] and 'static' or 'movable'}")
    
    json_message = {
        "xml_path": "custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml",
        "robot_goal": [goal_info["position"][0], goal_info["position"][1]],
        "reachable_objects": reachable_objects,
        "robot": {
            "position": robot_info["position"]
        },
        "objects": json_objects
    }
    
    print(f"  Robot position: {robot_info['position']}")
    print(f"  Goal position: {goal_info['position']}")
    print(f"  Total objects: {len(json_objects)} (including walls)")
    print(f"  Reachable objects: {reachable_objects}")
    
    return json_message


def create_proper_episode_data(all_objects, robot_info, goal_info, obs, reachable_objects):
    """Create proper episode data for visualizer."""
    print("=== Creating Proper Episode Data ===")
    
    # Create static object info with actual XML data
    static_object_info = {}
    
    # Add static walls
    for obj_name, obj_data in all_objects.items():
        if obj_data["is_static"]:
            static_object_info[obj_name] = {
                'pos_x': obj_data["position"][0],
                'pos_y': obj_data["position"][1], 
                'pos_z': obj_data["position"][2],
                'size_x': obj_data["size"][0],
                'size_y': obj_data["size"][1],
                'size_z': obj_data["size"][2],
                'quat_w': obj_data["quaternion"][0],
                'quat_x': obj_data["quaternion"][1],
                'quat_y': obj_data["quaternion"][2],
                'quat_z': obj_data["quaternion"][3]
            }
            print(f"  Static {obj_name}: pos=({obj_data['position'][0]:.2f}, {obj_data['position'][1]:.2f})")
    
    # Add movable objects (size info only)
    for obj_name, obj_data in all_objects.items():
        if not obj_data["is_static"]:
            static_object_info[obj_name] = {
                'size_x': obj_data["size"][0],
                'size_y': obj_data["size"][1],
                'size_z': obj_data["size"][2]
            }
            print(f"  Movable {obj_name}: size=({obj_data['size'][0]:.2f}, {obj_data['size'][1]:.2f})")
    
    # Create modified observations with XML positions for consistency
    modified_obs = obs.copy()
    
    # Update movable object poses with XML positions and rotations
    for obj_name, obj_data in all_objects.items():
        if not obj_data["is_static"]:
            pose_key = f"{obj_name}_pose"
            if pose_key in modified_obs:
                # Use XML position and rotation
                quat = obj_data["quaternion"]
                rotation = R.from_quat(quat, scalar_first=True)
                euler_rad = rotation.as_euler('xyz')[2]  # Z rotation in radians
                
                modified_obs[pose_key] = [
                    obj_data["position"][0],  # X from XML
                    obj_data["position"][1],  # Y from XML  
                    euler_rad  # Z rotation in radians
                ]
                print(f"  Updated {pose_key}: pos=({obj_data['position'][0]:.2f}, {obj_data['position'][1]:.2f}), rot={np.degrees(euler_rad):.1f}°")
    
    # Update robot pose with XML position
    if 'robot_pose' in modified_obs:
        modified_obs['robot_pose'] = [
            robot_info["position"][0],
            robot_info["position"][1], 
            0.0  # Robot doesn't rotate
        ]
        print(f"  Updated robot_pose: pos=({robot_info['position'][0]:.2f}, {robot_info['position'][1]:.2f})")
    
    episode_data = {
        'episode_id': 'fixed_comparison',
        'solution_found': True,
        'robot_goal': goal_info["position"],
        'static_object_info': static_object_info,
        'state_observations': [modified_obs],
        'action_sequence': []
    }
    
    print(f"  Episode data created with {len(static_object_info)} objects")
    return episode_data


def main():
    """Main test function with all fixes."""
    print("Fixed ML Inference vs Visualizer Comparison")
    print("=" * 60)
    
    # Step 1: Set up environment with proper XML parsing
    env, initial_state, obs, all_objects, robot_info, goal_info, reachable_objects = setup_environment_fixed()
    
    # Step 2: Create proper ML JSON message
    json_message = create_proper_ml_json(all_objects, robot_info, goal_info, reachable_objects)
    
    # Step 3: Run ML adapter with proper JSON
    print("\n=== Running ML Adapter (Fixed) ===")
    xml_rel_path = "custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml"
    ml_adapter = MLImageConverterAdapter(xml_rel_path)
    
    ml_result = ml_adapter.process_datapoint(json_message, [goal_info["position"][0], goal_info["position"][1]])
    
    ml_channels = {
        'robot': ml_result['robot_image'][:, :, 0],
        'goal': ml_result['goal_image'][:, :, 0],
        'movable': ml_result['movable_objects_image'][:, :, 0],
        'static': ml_result['static_objects_image'][:, :, 0],
        'reachable': ml_result['reachable_objects_image'][:, :, 0]
    }
    
    print("ML Results:")
    for name, channel in ml_channels.items():
        nonzero = np.count_nonzero(channel)
        print(f"  {name}: {nonzero} nonzero pixels")
    
    # Step 4: Create proper episode data and run visualizer
    print("\n=== Running Visualizer (Fixed) ===")
    episode_data = create_proper_episode_data(all_objects, robot_info, goal_info, obs, reachable_objects)
    
    visualizer = NAMODataVisualizer()
    vis_masks = visualizer.generate_episode_masks_batch(episode_data)
    
    vis_channels = {}
    for mask_name, mask in vis_masks.items():
        if mask_name in ['robot', 'goal', 'movable', 'static', 'reachable']:
            vis_channels[mask_name] = mask
    
    print("Visualizer Results:")
    for name, channel in vis_channels.items():
        nonzero = np.count_nonzero(channel)
        print(f"  {name}: {nonzero} nonzero pixels")
    
    # Step 5: Create comparison
    print("\n=== Creating Fixed Comparison ===")
    channel_names = ['robot', 'goal', 'movable', 'static', 'reachable']
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for col, channel_name in enumerate(channel_names):
        if channel_name in ml_channels and channel_name in vis_channels:
            ml_img = ml_channels[channel_name]
            vis_img = vis_channels[channel_name]
            
            # ML image (top row)
            axes[0, col].imshow(ml_img, cmap='gray', vmin=0, vmax=1)
            axes[0, col].set_title(f'ML: {channel_name.title()}')
            axes[0, col].axis('off')
            
            # Visualizer image (middle row)
            axes[1, col].imshow(vis_img, cmap='gray', vmin=0, vmax=1)
            axes[1, col].set_title(f'Visualizer: {channel_name.title()}')
            axes[1, col].axis('off')
            
            # Difference image (bottom row)
            diff_img = np.abs(ml_img - vis_img)
            max_diff = diff_img.max()
            axes[2, col].imshow(diff_img, cmap='hot', vmin=0, vmax=max(max_diff, 0.1))
            axes[2, col].set_title(f'Diff: {max_diff:.3f}')
            axes[2, col].axis('off')
            
            # Print detailed comparison
            ml_nonzero = np.count_nonzero(ml_img)
            vis_nonzero = np.count_nonzero(vis_img)
            print(f"  {channel_name}:")
            print(f"    ML: {ml_nonzero} pixels, Visualizer: {vis_nonzero} pixels")
            print(f"    Max diff: {max_diff:.6f}, Mean diff: {diff_img.mean():.6f}")
    
    plt.suptitle('FIXED: ML Inference vs Unified Visualizer Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = "ml_vs_visualizer_comparison_FIXED.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Fixed comparison saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())