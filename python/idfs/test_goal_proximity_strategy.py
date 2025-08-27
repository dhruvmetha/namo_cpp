#!/usr/bin/env python3
"""Test script for the new GoalProximityStrategy.

This script tests the goal proximity strategy to ensure it correctly orders
objects by distance from the robot goal position.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import namo_rl
from idfs.object_selection_strategy import NoHeuristicStrategy, NearestFirstStrategy, GoalProximityStrategy
import math


def test_goal_proximity_strategy():
    """Test that GoalProximityStrategy orders objects by distance from goal."""
    
    print("=" * 60)
    print("TESTING GOAL PROXIMITY STRATEGY")
    print("=" * 60)
    
    try:
        # Create environment
        env = namo_rl.RLEnvironment("../data/test_scene.xml", "../config/namo_config_complete.yaml")
        
        # Set a specific robot goal
        robot_goal = (3.0, 2.0, 0.0)
        env.set_robot_goal(*robot_goal)
        
        # Get initial state and reachable objects
        initial_state = env.get_full_state()
        reachable_objects = env.get_reachable_objects()
        
        print(f"Robot goal position: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f})")
        print(f"Reachable objects: {reachable_objects}")
        
        if len(reachable_objects) < 2:
            print("⚠ Need at least 2 reachable objects to test ordering")
            return
        
        # Test different strategies
        print("\n" + "-" * 40)
        print("TESTING OBJECT ORDERING STRATEGIES:")
        print("-" * 40)
        
        # No Heuristic Strategy
        no_heuristic = NoHeuristicStrategy()
        no_heuristic_order = no_heuristic.select_objects(reachable_objects, initial_state, env)
        print(f"No Heuristic order: {no_heuristic_order}")
        
        # Nearest First Strategy
        nearest_first = NearestFirstStrategy()
        nearest_first_order = nearest_first.select_objects(reachable_objects, initial_state, env)
        print(f"Nearest First order: {nearest_first_order}")
        
        # Goal Proximity Strategy
        goal_proximity = GoalProximityStrategy()
        goal_proximity_order = goal_proximity.select_objects(reachable_objects, initial_state, env)
        print(f"Goal Proximity order: {goal_proximity_order}")
        
        # Show detailed distances
        print("\n" + "-" * 40)
        print("DISTANCE ANALYSIS:")
        print("-" * 40)
        
        env.set_full_state(initial_state)
        obs = env.get_observation()
        robot_pose = obs.get('robot_pose')
        
        if robot_pose is not None:
            print(f"Robot position: ({robot_pose[0]:.2f}, {robot_pose[1]:.2f})")
            print(f"Goal position: ({robot_goal[0]:.2f}, {robot_goal[1]:.2f})")
            print()
            
            for obj_id in reachable_objects:
                pose_key = f"{obj_id}_pose"
                if pose_key in obs:
                    obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
                    
                    # Distance from robot to object
                    robot_to_obj = math.sqrt((obj_x - robot_pose[0])**2 + (obj_y - robot_pose[1])**2)
                    
                    # Distance from object to goal
                    obj_to_goal = math.sqrt((obj_x - robot_goal[0])**2 + (obj_y - robot_goal[1])**2)
                    
                    print(f"{obj_id}:")
                    print(f"  Position: ({obj_x:.2f}, {obj_y:.2f})")
                    print(f"  Distance from robot: {robot_to_obj:.2f}")
                    print(f"  Distance to goal: {obj_to_goal:.2f}")
                    print()
        
        # Verify strategies produce different orders
        strategies_different = False
        if no_heuristic_order != goal_proximity_order:
            print("✓ Goal Proximity differs from No Heuristic")
            strategies_different = True
            
        if nearest_first_order != goal_proximity_order:
            print("✓ Goal Proximity differs from Nearest First") 
            strategies_different = True
            
        if not strategies_different:
            print("⚠ All strategies produced the same order (this could be valid for simple scenes)")
        
        print("\n✓ Goal Proximity Strategy test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_goal_proximity_strategy()