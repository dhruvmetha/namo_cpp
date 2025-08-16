#!/usr/bin/env python3
"""
Demonstration of using reachability queries in an RL environment.

This shows how an RL agent or planner can query which objects are reachable
through push actions before deciding on actions.
"""

import sys
import os
sys.path.insert(0, '/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl')

import namo_rl
import random

def demo_rl_with_reachability():
    """
    Demonstrate how RL agents can use reachability information.
    """
    
    print("🤖 RL Environment with Reachability Demo")
    print("=" * 50)
    
    # Create environment
    env = namo_rl.RLEnvironment(
        'data/test_scene.xml', 
        'config/namo_config_complete.yaml', 
        visualize=False
    )
    
    # Reset to initial state
    env.reset()
    
    # Get initial observation and reachability
    obs = env.get_observation()
    reachable_objects = env.get_reachable_objects()
    
    print(f"📍 Initial state:")
    for obj_name, pose in obs.items():
        if obj_name.endswith('_pose'):
            print(f"  {obj_name}: [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]")
    
    print(f"\n🎯 Objects reachable through push: {reachable_objects}")
    
    # Simulate RL agent decision making
    print(f"\n🧠 RL Agent Decision Process:")
    
    if not reachable_objects:
        print("  ❌ No objects are reachable - agent cannot take any actions")
        return
    
    # Agent selects from reachable objects only
    selected_object = random.choice(reachable_objects)
    print(f"  ✅ Agent selected object: {selected_object}")
    
    # Verify selection is valid
    if env.is_object_reachable(selected_object):
        print(f"  ✓ Confirmed {selected_object} is reachable")
    else:
        print(f"  ❌ Error: {selected_object} is not reachable!")
        return
    
    # Create action (random target pose for demo)
    action = namo_rl.Action()
    action.object_id = selected_object
    action.x = 2.0
    action.y = 1.0  
    action.theta = 0.0
    
    print(f"  🎯 Action: push {selected_object} to [{action.x}, {action.y}, {action.theta}]")
    
    # Execute action
    result = env.step(action)
    
    print(f"\n📊 Action Result:")
    print(f"  Success: {result.done}")
    print(f"  Reward: {result.reward}")
    print(f"  Info: {result.info}")
    
    # Check reachability after action
    new_reachable = env.get_reachable_objects()
    print(f"\n🎯 Objects reachable after action: {new_reachable}")
    
    if set(new_reachable) != set(reachable_objects):
        print("  ℹ️  Reachability changed after action!")
        added = set(new_reachable) - set(reachable_objects)
        removed = set(reachable_objects) - set(new_reachable)
        if added:
            print(f"    + Now reachable: {list(added)}")
        if removed:
            print(f"    - No longer reachable: {list(removed)}")
    else:
        print("  ℹ️  Reachability unchanged")

def demo_multi_object_scenario():
    """
    Demonstrate reachability with multiple objects using a different scene.
    """
    print(f"\n🏗️  Multi-Object Scenario Demo")
    print("=" * 50)
    
    try:
        # Try to use benchmark scene which might have more objects
        env = namo_rl.RLEnvironment(
            'data/benchmark_env.xml', 
            'config/benchmark_config.yaml', 
            visualize=False
        )
        
        env.reset()
        
        obs = env.get_observation()
        reachable_objects = env.get_reachable_objects()
        
        print(f"📍 Scene objects: {len([k for k in obs.keys() if k.endswith('_pose')])}")
        print(f"🎯 Reachable objects: {len(reachable_objects)}")
        
        if len(reachable_objects) > 1:
            print("\n🔍 Individual reachability check:")
            for obj in reachable_objects:
                is_reachable = env.is_object_reachable(obj)
                print(f"  {obj}: {'✓' if is_reachable else '✗'}")
        
        print(f"📊 Reachability ratio: {len(reachable_objects)}/{len([k for k in obs.keys() if 'movable' in k and k.endswith('_pose')])}")
        
    except Exception as e:
        print(f"⚠️  Could not load benchmark scene: {e}")
        print("   Using single object demo instead")

if __name__ == "__main__":
    demo_rl_with_reachability()
    demo_multi_object_scenario()
    
    print(f"\n✅ Demo completed!")
    print(f"\n💡 Key takeaways:")
    print(f"   • RL agents can query env.get_reachable_objects() to see valid targets")
    print(f"   • Individual objects can be checked with env.is_object_reachable(name)")  
    print(f"   • Reachability may change after actions (robot moves, objects move)")
    print(f"   • This prevents wasted actions on unreachable objects")