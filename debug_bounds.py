#!/usr/bin/env python3
"""
Debug bounds calculation and coordinate transformation.
"""

import sys
import os
import numpy as np
sys.path.append('python')

from mask_generation.visualizer import NAMODataVisualizer

def debug_bounds_calculation():
    """Debug the bounds calculation and coordinate transformation."""
    print("🔍 Debugging bounds calculation...")
    
    # Create simple test case with known bounds
    episode_data = {
        'episode_id': 'debug_episode_0',
        'solution_found': True,
        'robot_goal': (0.0, 0.0, 0.0),
        'action_sequence': [],
        'state_observations': [
            {
                'robot_pose': [0.0, 0.0, 0.0]
            }
        ],
        'static_object_info': {
            # Single wall exactly at x=2.0, should appear at right edge
            'wall_right': {
                'pos_x': 2.0, 'pos_y': 0.0, 'pos_z': 0.0,
                'size_x': 0.1, 'size_y': 1.0, 'size_z': 0.3,
                'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0
            },
            'robot': {
                'size_x': 0.15, 'size_y': 0.15, 'size_z': 0.3
            }
        }
    }
    
    # Create visualizer and extract environment info
    visualizer = NAMODataVisualizer()
    env_info = visualizer._extract_env_info_from_episode(episode_data)
    
    print(f"📐 Calculated world bounds: {env_info.world_bounds}")
    x_min, x_max, y_min, y_max = env_info.world_bounds
    
    # Expected bounds: wall from 1.9 to 2.1, robot at 0.0, so roughly [-0.15, 2.1] + padding
    print(f"   X range: {x_min:.3f} to {x_max:.3f} (width: {x_max - x_min:.3f})")
    print(f"   Y range: {y_min:.3f} to {y_max:.3f} (height: {y_max - y_min:.3f})")
    
    # Test coordinate transformation
    print(f"\n🎯 Testing coordinate transformation:")
    
    # Test corner coordinates
    test_coords = [
        (x_min, y_min, "bottom-left"),
        (x_max, y_min, "bottom-right"),
        (x_min, y_max, "top-left"),
        (x_max, y_max, "top-right"),
        (2.0, 0.0, "wall center")  # The wall center
    ]
    
    for x, y, label in test_coords:
        px, py = visualizer._world_to_pixel(x, y, env_info.world_bounds)
        print(f"   {label}: world({x:.2f}, {y:.2f}) -> pixel({px}, {py})")
    
    # Check if wall center is near the edge
    wall_px, wall_py = visualizer._world_to_pixel(2.0, 0.0, env_info.world_bounds)
    print(f"\n🏗️  Wall center at pixel ({wall_px}, {wall_py})")
    print(f"   Distance from right edge: {224 - wall_px} pixels")
    print(f"   Distance from image center: {abs(wall_px - 112)} pixels from x-center")
    
    return True

if __name__ == "__main__":
    debug_bounds_calculation()