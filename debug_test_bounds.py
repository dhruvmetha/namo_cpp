#!/usr/bin/env python3
"""
Debug script to understand exactly what bounds are being calculated.
"""

import sys
import os
import numpy as np
sys.path.append('python')

from mask_generation.visualizer import NAMODataVisualizer

def debug_test_bounds():
    """Debug the exact bounds calculation for our test case."""
    print("🔍 Debugging test case bounds calculation...")
    
    # Create test episode with boundary walls at the edges
    episode_data = {
        'episode_id': 'bounds_test_episode_0',
        'solution_found': True,
        'robot_goal': (0.0, 0.0, 0.0),
        'action_sequence': [
            {'object_id': 'obstacle_1_movable', 'target': (0.5, 0.5, 0.0)}
        ],
        'state_observations': [
            {
                'robot_pose': [0.0, 0.0, 0.0],
                'obstacle_1_movable_pose': [0.5, 0.5, 0.0]
            }
        ],
        'static_object_info': {
            # Boundary walls at the edges - should be visible without padding
            'wall_left': {
                'pos_x': -2.0, 'pos_y': 0.0, 'pos_z': 0.0,
                'size_x': 0.1, 'size_y': 2.0, 'size_z': 0.3,
                'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0
            },
            'wall_right': {
                'pos_x': 2.0, 'pos_y': 0.0, 'pos_z': 0.0,
                'size_x': 0.1, 'size_y': 2.0, 'size_z': 0.3,
                'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0
            },
            'wall_bottom': {
                'pos_x': 0.0, 'pos_y': -2.0, 'pos_z': 0.0,
                'size_x': 2.0, 'size_y': 0.1, 'size_z': 0.3,
                'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0
            },
            'wall_top': {
                'pos_x': 0.0, 'pos_y': 2.0, 'pos_z': 0.0,
                'size_x': 2.0, 'size_y': 0.1, 'size_z': 0.3,
                'quat_w': 1.0, 'quat_x': 0.0, 'quat_y': 0.0, 'quat_z': 0.0
            },
            'obstacle_1_movable': {
                'size_x': 0.2, 'size_y': 0.2, 'size_z': 0.3
            },
            'robot': {
                'size_x': 0.15, 'size_y': 0.15, 'size_z': 0.3
            }
        }
    }
    
    print("📊 Wall positions and extents:")
    for name, wall in episode_data['static_object_info'].items():
        if name.startswith('wall'):
            x, y = wall['pos_x'], wall['pos_y']
            size_x, size_y = wall['size_x'], wall['size_y']
            print(f"   {name}: center=({x}, {y}), size=({size_x}, {size_y})")
            print(f"      -> x_extent: [{x-size_x}, {x+size_x}], y_extent: [{y-size_y}, {y+size_y}]")
    
    # Create visualizer and extract environment info
    visualizer = NAMODataVisualizer()
    env_info = visualizer._extract_env_info_from_episode(episode_data)
    
    x_min, x_max, y_min, y_max = env_info.world_bounds
    print(f"\n📐 Calculated world bounds: ({x_min}, {x_max}, {y_min}, {y_max})")
    print(f"   X range: {x_min:.3f} to {x_max:.3f} (width: {x_max - x_min:.3f})")
    print(f"   Y range: {y_min:.3f} to {y_max:.3f} (height: {y_max - y_min:.3f})")
    
    # Calculate what the padding should be
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    padding_x_calc = x_range * 0.1
    padding_y_calc = y_range * 0.1
    padding_x_actual = min(padding_x_calc, 0.2)
    padding_y_actual = min(padding_y_calc, 0.2)
    
    print(f"\n🧮 Padding calculation:")
    print(f"   X: 10% of {x_range:.3f} = {padding_x_calc:.3f}, min(that, 0.2) = {padding_x_actual:.3f}")
    print(f"   Y: 10% of {y_range:.3f} = {padding_y_calc:.3f}, min(that, 0.2) = {padding_y_actual:.3f}")
    
    # Test coordinate transformation for wall edges
    print(f"\n🎯 Wall positions in pixel coordinates:")
    
    wall_positions = [
        (-2.1, 0, "left wall left edge"),
        (-1.9, 0, "left wall right edge"),
        (1.9, 0, "right wall left edge"),
        (2.1, 0, "right wall right edge"),
        (0, -2.1, "bottom wall bottom edge"),
        (0, -1.9, "bottom wall top edge"),
        (0, 1.9, "top wall bottom edge"),
        (0, 2.1, "top wall top edge"),
    ]
    
    for x, y, label in wall_positions:
        px, py = visualizer._world_to_pixel(x, y, env_info.world_bounds)
        print(f"   {label}: world({x:.1f}, {y:.1f}) -> pixel({px}, {py})")
    
    print(f"\n📏 Expected vs actual:")
    print(f"   Left wall should extend from x=-2.1 to x=-1.9")
    print(f"   Right wall should extend from x=1.9 to x=2.1")  
    print(f"   If bounds are properly tight, walls should be near image edges")
    
    return True

if __name__ == "__main__":
    debug_test_bounds()