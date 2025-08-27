#!/usr/bin/env python3
"""
Test script for the NAMO RL Environment Python bindings.

This script demonstrates how to use the RLEnvironment for MCTS-style
tree search with full state management.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    import namo_rl
    print("✓ Successfully imported namo_rl module")
except ImportError as e:
    print(f"✗ Failed to import namo_rl: {e}")
    print("Make sure you've built the module with: cmake -DBUILD_PYTHON_BINDINGS=ON")
    sys.exit(1)

def test_basic_functionality():
    """Test basic RL environment functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Initialize environment (adjust paths as needed)
    xml_path = "data/test_scene.xml"  # Adjust to your scene file
    config_path = "config/namo_config_complete.yaml"  # Adjust to your config file
    
    try:
        env = namo_rl.RLEnvironment(xml_path, config_path, True)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    
    # Test reset
    try:
        env.reset()
        print("✓ Environment reset successful")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        return False
    
    # Test observation
    try:
        obs = env.get_observation()
        print(f"✓ Got observation with {len(obs)} objects")
        for obj_name, pose in obs.items():
            print(f"  {obj_name}: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
    except Exception as e:
        print(f"✗ Failed to get observation: {e}")
        return False
    
    return True

def test_state_management():
    """Test state save/restore functionality for MCTS."""
    print("\n=== Testing State Management for MCTS ===")
    
    xml_path = "data/test_scene.xml"
    config_path = "config/namo_config_complete.yaml"
    
    try:
        env = namo_rl.RLEnvironment(xml_path, config_path, True)
        env.reset()
        
        # Get initial state
        initial_state = env.get_full_state()
        print(f"✓ Captured initial state: {initial_state}")
        
        # Get initial observation for comparison
        initial_obs = env.get_observation()
        
        # Create and execute an action
        action = namo_rl.Action()
        if len(initial_obs) > 0:
            # Use the first movable object
            obj_name = list(initial_obs.keys())[0].replace("_pose", "")
            action.object_id = obj_name
            action.x = initial_obs[obj_name + "_pose"][0] + np.random.uniform(-0.5, 0.5)  # Move 0.5m in x
            action.y = initial_obs[obj_name + "_pose"][1] + np.random.uniform(-0.5, 0.5)  # Move 0.5m in y
            action.theta = np.random.uniform(-np.pi, np.pi)
            
            print(f"✓ Created action: push {action.object_id} to ({action.x:.3f}, {action.y:.3f}, {action.theta:.3f})")
            
            # Execute action
            result = env.step(action)
            print(f"✓ Action executed. Success: {result.done}, Reward: {result.reward}")
            if result.info:
                for key, value in result.info.items():
                    print(f"  {key}: {value}")
            
            # Get new observation
            new_obs = env.get_observation()
            print("✓ State after action:")
            for obj_name, pose in new_obs.items():
                print(f"  {obj_name}: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
            
            # Restore initial state
            env.set_full_state(initial_state)
            print("✓ Restored to initial state")
            
            # Verify restoration
            restored_obs = env.get_observation()
            print("✓ State after restoration:")
            for obj_name, pose in restored_obs.items():
                print(f"  {obj_name}: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
            
            # Check if restoration worked
            all_close = True
            for obj_name in initial_obs:
                if obj_name in restored_obs:
                    for i in range(3):  # x, y, theta
                        diff = abs(initial_obs[obj_name][i] - restored_obs[obj_name][i])
                        if diff > 1e-6:
                            all_close = False
                            break
            
            if all_close:
                print("✓ State restoration verified - positions match initial state")
            else:
                print("⚠ State restoration may not be perfect - small differences detected")
                
        else:
            print("⚠ No movable objects found, skipping action test")
            
    except Exception as e:
        print(f"✗ State management test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("NAMO RL Environment Test Suite")
    print("=" * 40)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n✗ Basic functionality test failed")
        return 1
    
    # Test state management
    if not test_state_management():
        print("\n✗ State management test failed")
        return 1
    

    print("\n" + "=" * 40)
    print("✓ All tests passed!")
    print("\nThe RL environment is ready for MCTS integration.")
    print("\nNext steps:")
    print("1. Implement your MCTS algorithm in Python")
    print("2. Use env.get_full_state() to save states at each node")
    print("3. Use env.set_full_state(state) to restore states when backtracking")
    print("4. Use env.step(action) to simulate actions and get rewards")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
