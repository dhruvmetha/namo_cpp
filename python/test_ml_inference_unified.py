#!/usr/bin/env python3
"""
Test ML inference with unified image converter.

This script tests that the ML models work with our unified image converter
and produce consistent results.
"""

import sys
import os
import numpy as np

# Add paths for imports
sys.path.append('/common/home/dm1487/robotics_research/ktamp/namo/python')
sys.path.append('/common/home/dm1487/robotics_research/ktamp/learning')

from ml_image_converter_adapter import MLImageConverterAdapter

def test_ml_adapter_interface():
    """Test that the ML adapter provides the same interface as the original."""
    print("=== Testing ML Adapter Interface ===")
    
    try:
        # Test with a known XML file
        xml_path = "custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100a.xml"
        adapter = MLImageConverterAdapter(xml_path)
        
        print(f"✓ MLImageConverterAdapter created successfully")
        print(f"  IMG_SIZE: {adapter.IMG_SIZE}")
        print(f"  World bounds: {adapter.world_bounds}")
        print(f"  Scale: {adapter.SCALE:.2f}")
        
        # Test interface methods
        test_point = (1.0, 2.0)
        px, py = adapter._world_to_pixel(*test_point)
        x_back, y_back = adapter.pixel_to_world(px, py)
        print(f"✓ Coordinate conversion: world{test_point} -> pixel({px},{py}) -> world({x_back:.2f},{y_back:.2f})")
        
        # Test with mock JSON data
        json_message = {
            "xml_path": xml_path,
            "robot_goal": [2.37, 2.76],
            "reachable_objects": ["obstacle_1_movable"],
            "robot": {
                "position": [0.0, 0.0, 0.0]
            },
            "objects": {
                "obstacle_1_movable": {
                    "position": [1.0, 1.0, 0.0],
                    "quaternion": [1.0, 0.0, 0.0, 0.0]
                },
                "wall_north": {
                    "position": [0.0, 3.0, 0.0],
                    "quaternion": [1.0, 0.0, 0.0, 0.0]
                }
            }
        }
        
        result = adapter.process_datapoint(json_message, [2.37, 2.76])
        print(f"✓ process_datapoint completed")
        print(f"  Result keys: {list(result.keys())}")
        
        # Verify expected output format
        expected_keys = ['robot_image', 'goal_image', 'movable_objects_image', 
                        'static_objects_image', 'reachable_objects_image', 
                        'obj2center_px', 'obj2angle']
        
        for key in expected_keys:
            if key in result:
                if isinstance(result[key], np.ndarray):
                    print(f"  {key}: shape {result[key].shape}, dtype {result[key].dtype}")
                else:
                    print(f"  {key}: {type(result[key])} with {len(result[key])} items")
            else:
                print(f"  ✗ Missing key: {key}")
        
        # Test object mask creation
        if adapter.data_point:
            mask = adapter.create_object_mask("obstacle_1_movable")
            print(f"✓ Object mask created: shape {mask.shape}, nonzero: {np.count_nonzero(mask)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_models_import():
    """Test that ML models can import and use the unified converter."""
    print("\n=== Testing ML Models with Unified Converter ===")
    
    try:
        # Test importing the updated ML models
        from ktamp_learning.object_inference_model import ObjectInferenceModel
        print("✓ ObjectInferenceModel imported successfully")
        
        from ktamp_learning.goal_inference_model import GoalInferenceModel  
        print("✓ GoalInferenceModel imported successfully")
        
        # Note: We can't actually load models without trained checkpoints,
        # but at least we verified the imports work with our unified converter
        
        return True
        
    except Exception as e:
        print(f"✗ ML model import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consistency_check():
    """Test that unified converter produces consistent results."""
    print("\n=== Testing Consistency ===")
    
    try:
        xml_path = "custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100a.xml"
        
        # Create two adapter instances
        adapter1 = MLImageConverterAdapter(xml_path)
        adapter2 = MLImageConverterAdapter(xml_path)
        
        # Test with same input
        json_message = {
            "xml_path": xml_path,
            "robot_goal": [2.37, 2.76],
            "reachable_objects": ["obstacle_1_movable"],
            "robot": {"position": [0.0, 0.0, 0.0]},
            "objects": {
                "obstacle_1_movable": {
                    "position": [1.0, 1.0, 0.0],
                    "quaternion": [1.0, 0.0, 0.0, 0.0]
                }
            }
        }
        
        result1 = adapter1.process_datapoint(json_message, [2.37, 2.76])
        result2 = adapter2.process_datapoint(json_message, [2.37, 2.76])
        
        # Check that results are identical
        for key in ['robot_image', 'goal_image', 'movable_objects_image']:
            if key in result1 and key in result2:
                diff = np.abs(result1[key] - result2[key]).max()
                if diff < 1e-6:
                    print(f"✓ {key}: consistent (max diff: {diff})")
                else:
                    print(f"✗ {key}: inconsistent (max diff: {diff})")
                    return False
        
        print("✓ Consistency check passed")
        return True
        
    except Exception as e:
        print(f"✗ Consistency test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing ML Inference with Unified Converter")
    print("=" * 50)
    
    tests = [
        ("ML Adapter Interface", test_ml_adapter_interface),
        ("ML Models Import", test_ml_models_import),
        ("Consistency Check", test_consistency_check),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
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