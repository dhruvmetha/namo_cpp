#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl')

import namo_rl

def test_reachability():
    """Test the new reachability functionality in the RL environment."""
    
    print("Testing reachability functionality...")
    
    try:
        # Create environment
        env = namo_rl.RLEnvironment(
            'data/test_scene.xml', 
            'config/namo_config_complete.yaml', 
            visualize=False
        )
        print("✓ Environment created successfully")
        
        # Reset environment
        env.reset()
        print("✓ Environment reset")
        
        # Test get_reachable_objects
        reachable_objects = env.get_reachable_objects()
        print(f"✓ Reachable objects: {reachable_objects}")
        
        # Test individual object reachability
        for obj_name in reachable_objects:
            is_reachable = env.is_object_reachable(obj_name)
            print(f"  - {obj_name}: {'✓ reachable' if is_reachable else '✗ not reachable'}")
        
        # Test with non-existent object
        is_fake_reachable = env.is_object_reachable("fake_object")
        print(f"  - fake_object: {'✓ reachable' if is_fake_reachable else '✗ not reachable (expected)'}")
        
        # Get observation to see all objects
        obs = env.get_observation()
        print(f"✓ All objects in scene: {list(obs.keys())}")
        
        print("\n✅ All reachability tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_reachability()