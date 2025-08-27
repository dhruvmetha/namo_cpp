#!/usr/bin/env python3
"""
FINAL ML Inference vs Visualizer Comparison Test

Correct approach:
- Robot/goal sizes: 0.2 and 0.25 (fixed in unified converter)
- Object positions/rotations: From environment observations (JSON)
- Object sizes: From XML parsing
- Static objects: From XML parsing with proper positions
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

# Add necessary paths
sys.path.append('/common/home/dm1487/robotics_research/ktamp/namo/python')
sys.path.append('/common/home/dm1487/robotics_research/ktamp/learning')

import namo_rl
from unified_image_converter import UnifiedImageConverter, ObjectInfo, create_converter_from_xml
from ml_image_converter_adapter import MLImageConverterAdapter
from mask_generation.visualizer import NAMODataVisualizer
from idfs.object_selection_strategy import NoHeuristicStrategy


def setup_environment():
    """Set up environment and get all necessary data."""
    print("=== Setting up Environment (Final) ===")
    
    xml_path = "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml"
    config_path = "/common/home/dm1487/robotics_research/ktamp/namo/config/namo_config_complete.yaml"
    
    # Initialize environment
    env = namo_rl.RLEnvironment(xml_path, config_path)
    env.reset()
    initial_state = env.get_full_state()
    obs = env.get_observation()
    reachable_objects = env.get_reachable_objects()
    
    print(f"Environment observations: {list(obs.keys())}")
    print(f"Reachable objects: {reachable_objects}")
    
    return env, initial_state, obs, reachable_objects, xml_path


def create_proper_json_message(obs, reachable_objects, xml_path):
    """Create JSON message with positions/rotations from observations, static objects from XML."""
    print("=== Creating Proper JSON Message ===")
    
    # Parse XML for static objects and robot/goal info
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    json_objects = {}
    robot_pos = None
    goal_pos = None
    
    # Add static walls from XML
    worldbody = root.find('.//worldbody')
    walls_body = worldbody.find('.//body[@name="walls"]')
    if walls_body is not None:
        for geom in walls_body.findall('geom'):
            name = geom.get('name')
            pos_str = geom.get('pos', '0 0 0')
            pos = [float(x) for x in pos_str.split()]
            
            json_objects[name] = {
                "position": [pos[0], pos[1], pos[2]],
                "quaternion": [1.0, 0.0, 0.0, 0.0]  # Static walls not rotated
            }
    
    # Get robot position from XML (for goal, since observations might not match)
    robot_body = worldbody.find('.//body[@name="robot"]')
    if robot_body is not None:
        robot_geom = robot_body.find('geom')
        if robot_geom is not None:
            pos_str = robot_geom.get('pos', '0 0 0')
            robot_pos = [float(x) for x in pos_str.split()]
    
    # Get goal position from XML
    goal_site = worldbody.find('.//site[@name="goal"]')
    if goal_site is not None:
        pos_str = goal_site.get('pos', '0 0 0')
        goal_pos = [float(x) for x in pos_str.split()]
    
    # Add movable objects from observations (positions and rotations)
    for key, pose in obs.items():
        if key.endswith('_pose') and key != 'robot_pose':
            obj_name = key.replace('_pose', '')
            if len(pose) >= 3:
                # Convert angle to quaternion (pose[2] is rotation in radians)
                rotation = R.from_euler('z', pose[2])  # Z-axis rotation
                quat = rotation.as_quat(scalar_first=True)  # [w, x, y, z]
                
                json_objects[obj_name] = {
                    "position": [float(pose[0]), float(pose[1]), float(pose[2])],
                    "quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                }
                print(f"  {obj_name}: pos=({pose[0]:.2f}, {pose[1]:.2f}), rot={np.degrees(pose[2]):.1f}°")
    
    json_message = {
        "xml_path": "custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml",
        "robot_goal": [goal_pos[0], goal_pos[1]],
        "reachable_objects": reachable_objects,
        "robot": {
            "position": robot_pos
        },
        "objects": json_objects
    }
    
    print(f"  Robot position: {robot_pos}")
    print(f"  Goal position: {goal_pos}")
    print(f"  Total objects in JSON: {len(json_objects)}")
    
    return json_message


def create_episode_data(obs, reachable_objects, goal_pos, xml_path):
    """Create episode data for visualizer using environment observations."""
    print("=== Creating Episode Data for Visualizer ===")
    
    # Parse XML for object sizes
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    static_object_info = {}
    
    # Add static walls from XML
    worldbody = root.find('.//worldbody')
    walls_body = worldbody.find('.//body[@name="walls"]')
    if walls_body is not None:
        for geom in walls_body.findall('geom'):
            name = geom.get('name')
            pos_str = geom.get('pos', '0 0 0')
            size_str = geom.get('size', '0.1 0.1 0.1')
            
            pos = [float(x) for x in pos_str.split()]
            size = [float(x) for x in size_str.split()]
            
            static_object_info[name] = {
                'pos_x': pos[0], 'pos_y': pos[1], 'pos_z': pos[2],
                'size_x': size[0], 'size_y': size[1], 'size_z': size[2],
                'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0
            }
    
    # Add movable object sizes from XML
    for body in worldbody.findall('body'):
        name = body.get('name')
        if name and 'movable' in name:
            geom = body.find('geom')
            if geom is not None:
                size_str = geom.get('size', '0.1 0.1 0.1')
                size = [float(x) for x in size_str.split()]
                
                static_object_info[name] = {
                    'size_x': size[0], 'size_y': size[1], 'size_z': size[2]
                }
    
    episode_data = {
        'episode_id': 'final_comparison',
        'solution_found': True,
        'robot_goal': goal_pos,
        'static_object_info': static_object_info,
        'state_observations': [obs],
        'action_sequence': []
    }
    
    print(f"  Created episode data with {len(static_object_info)} objects")
    return episode_data


def main():
    """Main test function."""
    print("FINAL ML Inference vs Visualizer Comparison")
    print("=" * 60)
    
    # Step 1: Setup
    env, initial_state, obs, reachable_objects, xml_path = setup_environment()
    
    # Step 2: Create JSON message (positions/rotations from obs, sizes from XML)
    json_message = create_proper_json_message(obs, reachable_objects, xml_path)
    
    # Step 3: Run ML adapter
    print("\n=== Running ML Adapter (Final) ===")
    xml_rel_path = "custom_walled_envs/aug9/easy/set1/benchmark_2/env_config_1275a.xml"
    ml_adapter = MLImageConverterAdapter(xml_rel_path)
    
    # Use goal position from JSON message
    goal_pos = json_message["robot_goal"]
    ml_result = ml_adapter.process_datapoint(json_message, goal_pos)
    
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
    
    # Step 4: Run visualizer
    print("\n=== Running Visualizer (Final) ===")
    episode_data = create_episode_data(obs, reachable_objects, [goal_pos[0], goal_pos[1], 0.0], xml_path)
    
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
    
    # Step 5: Compare
    print("\n=== Final Comparison ===")
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
            
            # Print comparison
            ml_nonzero = np.count_nonzero(ml_img)
            vis_nonzero = np.count_nonzero(vis_img)
            print(f"  {channel_name}: ML={ml_nonzero}, Vis={vis_nonzero}, MaxDiff={max_diff:.3f}")
    
    plt.suptitle('FINAL: ML Inference vs Unified Visualizer (Fixed Robot/Goal Sizes)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = "ml_vs_visualizer_comparison_FINAL.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Final comparison saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())