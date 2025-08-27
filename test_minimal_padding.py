#!/usr/bin/env python3
"""
Test with minimal padding to match the reference image.
"""

import sys
import os
import numpy as np
sys.path.append('python')

from mask_generation.visualizer import NAMODataVisualizer

def test_minimal_padding():
    """Test with much smaller padding to match reference image."""
    print("🧪 Testing with minimal padding...")
    
    # Create test episode with boundary walls
    episode_data = {
        'episode_id': 'minimal_padding_test',
        'solution_found': True,
        'robot_goal': (0.0, 0.0, 0.0),
        'action_sequence': [],
        'state_observations': [
            {
                'robot_pose': [0.0, 0.0, 0.0]
            }
        ],
        'static_object_info': {
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
            'robot': {
                'size_x': 0.15, 'size_y': 0.15, 'size_z': 0.3
            }
        }
    }
    
    # Test different padding values
    padding_tests = [
        (0.02, "Very small padding (0.02)"),
        (0.05, "Small padding (0.05)"), 
        (0.1, "Medium padding (0.1)"),
        (0.2, "Current padding (0.2)")
    ]
    
    for padding_val, description in padding_tests:
        print(f"\n📊 {description}:")
        
        # Manually calculate bounds with this padding
        # Wall extents: x=[-2.1, 2.1], y=[-2.1, 2.1]
        x_min, x_max = -2.1, 2.1
        y_min, y_max = -2.1, 2.1
        
        # Add the test padding
        test_bounds = (x_min - padding_val, x_max + padding_val, 
                      y_min - padding_val, y_max + padding_val)
        
        print(f"   Bounds: {test_bounds}")
        
        # Calculate where walls would appear
        visualizer = NAMODataVisualizer()
        left_wall_px = visualizer._world_to_pixel(-2.1, 0, test_bounds)[0]
        right_wall_px = visualizer._world_to_pixel(2.1, 0, test_bounds)[0]
        
        left_distance = left_wall_px
        right_distance = 223 - right_wall_px
        
        print(f"   Left wall at pixel {left_wall_px} (distance from edge: {left_distance})")
        print(f"   Right wall at pixel {right_wall_px} (distance from edge: {right_distance})")
        
        # Check if this matches the reference image (walls very close to edges)
        if left_distance <= 2 and right_distance <= 2:
            print(f"   ✅ This matches reference image (walls within 2 pixels of edges)")
        else:
            print(f"   ❌ Too much padding (walls more than 2 pixels from edges)")

if __name__ == "__main__":
    test_minimal_padding()