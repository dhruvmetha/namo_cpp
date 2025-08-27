#!/usr/bin/env python3
"""
Test script to verify proper bounds handling without excessive padding.
"""

import sys
import os
import numpy as np
sys.path.append('python')

from mask_generation.visualizer import NAMODataVisualizer

def test_bounds_padding():
    """Test that masks use actual world bounds without excessive padding."""
    print("🧪 Testing bounds handling without excessive padding...")
    
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
    
    print("📊 Created test episode with boundary walls")
    
    try:
        # Create visualizer
        visualizer = NAMODataVisualizer()
        
        # Generate masks
        print("🎨 Generating masks to check bounds...")
        masks = visualizer.generate_episode_masks_batch(episode_data)
        
        # Check static mask to see if boundary walls are at the edges
        static_mask = masks['static']
        
        print(f"✅ Generated static mask with shape {static_mask.shape}")
        
        # Check edge pixels - boundary walls should be visible at edges
        top_edge = static_mask[0, :]  # Top row
        bottom_edge = static_mask[-1, :]  # Bottom row
        left_edge = static_mask[:, 0]  # Left column
        right_edge = static_mask[:, -1]  # Right column
        
        # Count non-zero pixels at edges (should have wall pixels)
        top_pixels = np.sum(top_edge > 0)
        bottom_pixels = np.sum(bottom_edge > 0)
        left_pixels = np.sum(left_edge > 0)
        right_pixels = np.sum(right_edge > 0)
        
        print(f"📏 Edge wall pixels:")
        print(f"   Top edge: {top_pixels} pixels")
        print(f"   Bottom edge: {bottom_pixels} pixels") 
        print(f"   Left edge: {left_pixels} pixels")
        print(f"   Right edge: {right_pixels} pixels")
        
        # Check center area - should have less wall coverage than edges
        center_region = static_mask[75:149, 75:149]  # Central region
        center_wall_pixels = np.sum(center_region > 0)
        total_center_pixels = center_region.size
        center_wall_ratio = center_wall_pixels / total_center_pixels
        
        print(f"📐 Center region wall coverage: {center_wall_ratio:.3f} ({center_wall_pixels}/{total_center_pixels} pixels)")
        
        # Check that boundary walls are actually at the edges
        edge_wall_count = top_pixels + bottom_pixels + left_pixels + right_pixels
        print(f"🔍 Total edge wall pixels: {edge_wall_count}")
        
        if edge_wall_count > 0:
            print("✅ Boundary walls are visible at edges - bounds appear correct")
        else:
            print("⚠️  No boundary walls at edges - may have excessive padding")
        
        # Check overall static mask coverage
        total_static_pixels = np.sum(static_mask > 0)
        total_pixels = static_mask.size
        static_coverage = total_static_pixels / total_pixels
        
        print(f"📊 Overall static coverage: {static_coverage:.3f} ({total_static_pixels}/{total_pixels} pixels)")
        
        # Save a test mask for visual inspection
        test_output_dir = "/tmp/bounds_test"
        os.makedirs(test_output_dir, exist_ok=True)
        
        import cv2
        # Save static mask as image for inspection
        static_img = (static_mask * 255).astype(np.uint8)
        cv2.imwrite(f"{test_output_dir}/static_mask_bounds_test.png", static_img)
        
        print(f"💾 Saved test mask to {test_output_dir}/static_mask_bounds_test.png for visual inspection")
        print("🎉 Bounds padding test completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bounds_padding()
    sys.exit(0 if success else 1)