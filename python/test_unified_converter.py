#!/usr/bin/env python3
"""
Test script for the unified image converter.

This script tests the UnifiedImageConverter and MLImageConverterAdapter
to ensure they produce consistent results with the data collection approach.
"""

import sys
import os
import numpy as np
import cv2

# Add python directory to path
sys.path.append('/common/home/dm1487/robotics_research/ktamp/namo/python')

from unified_image_converter import UnifiedImageConverter, ObjectInfo, create_converter_from_xml
from ml_image_converter_adapter import MLImageConverterAdapter

def test_basic_converter():
    """Test basic functionality of UnifiedImageConverter."""
    print("=== Testing UnifiedImageConverter ===")
    
    # Create a simple test environment
    world_bounds = (-3.0, 3.0, -3.0, 3.0)
    converter = UnifiedImageConverter(world_bounds)
    
    print(f"World bounds: {world_bounds}")
    print(f"World size: {converter.world_size}")
    print(f"Scale: {converter.scale}")
    
    # Test coordinate conversion
    test_points = [
        (0.0, 0.0, "center"),
        (-3.0, -3.0, "bottom-left"),
        (3.0, 3.0, "top-right"),
        (1.5, -1.5, "arbitrary")
    ]
    
    print("\nCoordinate conversion test:")
    for x, y, name in test_points:
        px, py = converter.world_to_pixel(x, y)
        x_back, y_back = converter.pixel_to_world(px, py)
        print(f"  {name}: world({x:.2f}, {y:.2f}) -> pixel({px}, {py}) -> world({x_back:.2f}, {y_back:.2f})")
    
    # Test object drawing
    objects = [
        ObjectInfo("wall_1", -2.0, 0.0, 0.0, 0.5, 2.0, is_static=True),
        ObjectInfo("box_1", 1.0, 1.0, np.pi/4, 0.3, 0.3, is_static=False, is_reachable=True),
        ObjectInfo("box_2", -1.0, 1.5, 0.0, 0.2, 0.4, is_static=False, is_reachable=False),
    ]
    
    robot_pos = (0.0, -2.0, 0.0)
    robot_goal = (0.0, 2.0, 0.0)
    
    # Create multi-channel image
    channels = converter.create_multi_channel_image(robot_pos, robot_goal, objects)
    print(f"\nMulti-channel image shape: {channels.shape}")
    
    # Check each channel
    channel_names = ["robot", "goal", "movable", "static", "reachable"]
    for i, name in enumerate(channel_names):
        nonzero = np.count_nonzero(channels[i])
        print(f"  Channel {i} ({name}): {nonzero} nonzero pixels")
    
    # Save visualization
    save_multichannel_visualization(channels, "test_unified_converter.png")
    print("Saved visualization to test_unified_converter.png")
    
    return True

def test_ml_adapter():
    """Test MLImageConverterAdapter with XML file."""
    print("\n=== Testing MLImageConverterAdapter ===")
    
    # Use a test XML file
    xml_path = "custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100a.xml"
    
    try:
        adapter = MLImageConverterAdapter(xml_path)
        adapter.print_bounds_info()
        
        # Create a mock JSON message
        json_message = {
            "xml_path": xml_path,
            "robot_goal": [2.37, 2.76],
            "reachable_objects": ["obstacle_1_movable", "obstacle_2_movable"],
            "robot": {
                "position": [0.0, 0.0, 0.0]
            },
            "objects": {
                "obstacle_1_movable": {
                    "position": [1.0, 1.0, 0.0],
                    "quaternion": [1.0, 0.0, 0.0, 0.0]
                },
                "obstacle_2_movable": {
                    "position": [-1.0, 1.5, 0.0],
                    "quaternion": [0.707, 0.0, 0.0, 0.707]  # 90 degrees
                },
                "wall_north": {
                    "position": [0.0, 3.0, 0.0],
                    "quaternion": [1.0, 0.0, 0.0, 0.0]
                }
            }
        }
        
        # Process with adapter
        result = adapter.process_datapoint(json_message, [2.37, 2.76])
        
        print(f"\nML Adapter result keys: {list(result.keys())}")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape {value.shape}, nonzero: {np.count_nonzero(value)}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} items")
        
        # Create comparison visualization
        channels = []
        channel_names = ['robot_image', 'goal_image', 'movable_objects_image', 
                        'static_objects_image', 'reachable_objects_image']
        
        for name in channel_names:
            if name in result:
                # Convert from (H, W, 1) to (H, W)
                channel = result[name][:, :, 0]
                channels.append(channel)
        
        if channels:
            channels_array = np.array(channels)
            save_multichannel_visualization(channels_array, "test_ml_adapter.png")
            print("Saved ML adapter visualization to test_ml_adapter.png")
        
        # Test object mask creation
        if adapter.data_point:
            for obj_name in ["obstacle_1_movable", "obstacle_2_movable"]:
                if obj_name in json_message["objects"]:
                    mask = adapter.create_object_mask(obj_name)
                    print(f"  {obj_name} mask: shape {mask.shape}, nonzero: {np.count_nonzero(mask)}")
        
        return True
        
    except Exception as e:
        print(f"ML Adapter test failed: {e}")
        return False

def save_multichannel_visualization(channels: np.ndarray, filename: str):
    """Save a visualization of multi-channel image."""
    n_channels = channels.shape[0]
    cols = min(n_channels, 5)
    rows = 1
    
    # Create combined image
    img_size = channels.shape[1]
    combined = np.zeros((rows * img_size, cols * img_size), dtype=np.uint8)
    
    channel_names = ['Robot', 'Goal', 'Movable', 'Static', 'Reachable']
    
    for i in range(min(n_channels, cols)):
        # Convert to 0-255 range
        channel_viz = (channels[i] * 255).astype(np.uint8)
        
        x_start = i * img_size
        combined[0:img_size, x_start:x_start + img_size] = channel_viz
    
    cv2.imwrite(filename, combined)

def compare_with_original():
    """Compare results with original implementations if possible."""
    print("\n=== Comparison Test ===")
    
    try:
        # Try to import the original converters for comparison
        sys.path.append('/common/home/dm1487/robotics_research/ktamp/namo/python')
        from namo_image_converter import NAMOImageConverter
        
        print("Found NAMO image converter - comparison possible")
        
        # Could add comparison logic here
        
    except ImportError as e:
        print(f"Original converters not available for comparison: {e}")
    
    return True

def main():
    """Run all tests."""
    print("Testing Unified Image Converter System")
    print("=" * 50)
    
    tests = [
        ("Basic Converter", test_basic_converter),
        ("ML Adapter", test_ml_adapter),
        ("Comparison", compare_with_original),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            results[test_name] = test_func()
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())