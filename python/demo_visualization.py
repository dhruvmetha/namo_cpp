#!/usr/bin/env python3
"""
Visualization demo for NAMO RL Environment.

This script demonstrates how to use the visualization features
of the NAMO RL environment Python bindings.
"""

import sys
import os
import time

# Add the build directory to Python path
# sys.path.insert(0, "../build_python")

try:
    import namo_rl
    print("✓ Successfully imported namo_rl module")
except ImportError as e:
    print(f"✗ Failed to import namo_rl: {e}")
    print("Make sure you've built the module with visualization support")
    sys.exit(1)

def demo_with_visualization():
    """Demonstrate visualization capabilities."""
    print("\n=== NAMO Visualization Demo ===")
    
    # Create environment with visualization enabled
    print("Creating environment with visualization...")
    try:
        env = namo_rl.RLEnvironment(
            "data/test_scene.xml", 
            "config/namo_config_complete.yaml", 
            visualize=True  # Enable visualization
        )
        print("✓ Environment created with visualization")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    
    goals = [[1.0, 0.2], [0.2, 1.0]]
    for goal in goals:
        # Reset environment
        env.reset()
        print("✓ Environment reset")
        
        # Get initial observation
        obs = env.get_observation()
        print(f"✓ Initial observation: {len(obs)} objects")
        for obj_name, pose in obs.items():
            print(f"  {obj_name}: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
        
        # Render initial state
        print("\nRendering initial state...")
        env.render()
        print("✓ Initial state rendered")
    
        # Create and execute an action
        if len(obs) > 0:
            obj_name = list(obs.keys())[0].replace("_pose", "")
            
            print(f"\nExecuting push action on {obj_name}...")
            action = namo_rl.Action()
            action.object_id = obj_name
            action.x = obs[obj_name + "_pose"][0] + goal[0]  # Move 1m in x
            action.y = obs[obj_name + "_pose"][1] + goal[1]  # Move 0.2m in y  
            action.theta = 0.0
            
            print(f"Action: push {action.object_id} to ({action.x:.3f}, {action.y:.3f}, {action.theta:.3f})")
            
            # Execute action with periodic rendering
            result = env.step(action)
            print(f"✓ Action completed: success={result.done}, reward={result.reward}")
            
            # Render final state
            print("Rendering final state...")
            env.render()
            print("✓ Final state rendered")
            
            # Show final positions
            final_obs = env.get_observation()
            print("Final object positions:")
            for obj_name, pose in final_obs.items():
                print(f"  {obj_name}: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
        
    return True

def demo_without_visualization():
    """Demonstrate non-visualization mode for comparison."""
    print("\n=== Non-Visualization Mode Demo ===")
    
    # Create environment without visualization (default)
    print("Creating environment without visualization...")
    env = namo_rl.RLEnvironment("data/test_scene.xml", "config/namo_config_complete.yaml")
    print("✓ Environment created without visualization")
    
    # Reset and observe
    env.reset()
    obs = env.get_observation()
    print(f"✓ Observation: {len(obs)} objects (no visualization)")
    
    # Calling render() in non-visualization mode is safe (no-op)
    env.render()  # This will do nothing but won't error
    print("✓ Render call in non-visualization mode (no-op)")
    
    return True

def main():
    """Run visualization demos."""
    print("NAMO RL Environment Visualization Demo")
    print("=" * 50)
    
    # Demo without visualization first
    if not demo_without_visualization():
        return 1
    
    # Demo with visualization
    if not demo_with_visualization():
        print("\n⚠ Visualization demo failed (likely due to headless environment)")
        print("This is expected on systems without display/X11.")
        print("On a system with display, you would see:")
        print("  - Real-time MuJoCo simulation window")
        print("  - 3D visualization of robot and objects")
        print("  - Interactive camera controls")
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nUsage Summary:")
    print("1. Enable visualization: env = namo_rl.RLEnvironment(xml, config, visualize=True)")
    print("2. Render states: env.render()")
    print("3. Works with all standard RL methods (step, reset, etc.)")
    print("4. Perfect for debugging and demonstrations")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())