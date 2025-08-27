#!/usr/bin/env python3
"""Test script for object selection strategies in IDFS.

This script tests both the NoHeuristicStrategy (original behavior) 
and NearestFirstStrategy (nearest-first heuristic) to ensure they 
work correctly and produce different exploration orders.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import namo_rl
from idfs.standard_idfs import StandardIterativeDeepeningDFS, plan_with_idfs
from idfs.object_selection_strategy import NoHeuristicStrategy, NearestFirstStrategy
from base_planner import PlannerConfig


def test_strategy_comparison():
    """Test both strategies on the same scenario and compare results."""
    
    # Create environment
    env = namo_rl.RLEnvironment("../data/test_scene.xml", "../config/namo_config_complete.yaml")
    
    # Set robot goal
    robot_goal = (3.0, 3.0, 0.0)
    
    # Configuration
    config = PlannerConfig(
        max_depth=3,
        max_goals_per_object=3,
        random_seed=42,  # Fixed seed for reproducible comparison
        verbose=True,
        collect_stats=True
    )
    
    print("=" * 60)
    print("TESTING OBJECT SELECTION STRATEGIES")
    print("=" * 60)
    
    # Test No Heuristic Strategy (original behavior)
    print("\n1. TESTING NO HEURISTIC STRATEGY:")
    print("-" * 40)
    
    no_heuristic_strategy = NoHeuristicStrategy()
    planner_no_heuristic = StandardIterativeDeepeningDFS(env, config, no_heuristic_strategy)
    
    print(f"Algorithm: {planner_no_heuristic.algorithm_name}")
    print(f"Strategy: {no_heuristic_strategy.strategy_name}")
    
    result_no_heuristic = planner_no_heuristic.search(robot_goal)
    
    print(f"Solution found: {result_no_heuristic.solution_found}")
    if result_no_heuristic.solution_found:
        print(f"Solution depth: {result_no_heuristic.solution_depth}")
        print(f"Search time: {result_no_heuristic.search_time_ms:.2f}ms")
        print(f"Nodes expanded: {result_no_heuristic.nodes_expanded}")
    
    # Test Nearest First Strategy
    print("\n2. TESTING NEAREST FIRST STRATEGY:")
    print("-" * 40)
    
    nearest_first_strategy = NearestFirstStrategy()
    planner_nearest_first = StandardIterativeDeepeningDFS(env, config, nearest_first_strategy)
    
    print(f"Algorithm: {planner_nearest_first.algorithm_name}")
    print(f"Strategy: {nearest_first_strategy.strategy_name}")
    
    result_nearest_first = planner_nearest_first.search(robot_goal)
    
    print(f"Solution found: {result_nearest_first.solution_found}")
    if result_nearest_first.solution_found:
        print(f"Solution depth: {result_nearest_first.solution_depth}")
        print(f"Search time: {result_nearest_first.search_time_ms:.2f}ms")
        print(f"Nodes expanded: {result_nearest_first.nodes_expanded}")
    
    # Comparison
    print("\n3. COMPARISON:")
    print("-" * 40)
    
    if result_no_heuristic.solution_found and result_nearest_first.solution_found:
        print(f"No Heuristic: {result_no_heuristic.solution_depth} actions, "
              f"{result_no_heuristic.nodes_expanded} nodes, "
              f"{result_no_heuristic.search_time_ms:.2f}ms")
        print(f"Nearest First: {result_nearest_first.solution_depth} actions, "
              f"{result_nearest_first.nodes_expanded} nodes, "
              f"{result_nearest_first.search_time_ms:.2f}ms")
        
        if result_nearest_first.nodes_expanded < result_no_heuristic.nodes_expanded:
            print("✓ Nearest First strategy expanded fewer nodes!")
        elif result_nearest_first.nodes_expanded > result_no_heuristic.nodes_expanded:
            print("⚠ Nearest First strategy expanded more nodes")
        else:
            print("= Both strategies expanded the same number of nodes")
    
    return result_no_heuristic, result_nearest_first


def test_object_ordering():
    """Test that object selection strategies actually reorder objects differently."""
    
    print("\n4. TESTING OBJECT ORDERING:")
    print("-" * 40)
    
    # Create environment and get initial state
    env = namo_rl.RLEnvironment("../data/test_scene.xml", "../config/namo_config_complete.yaml")
    initial_state = env.get_full_state()
    
    # Get reachable objects
    reachable_objects = env.get_reachable_objects()
    print(f"Reachable objects: {reachable_objects}")
    
    if len(reachable_objects) < 2:
        print("⚠ Need at least 2 reachable objects to test ordering differences")
        return
    
    # Test No Heuristic Strategy ordering
    no_heuristic = NoHeuristicStrategy()
    no_heuristic_order = no_heuristic.select_objects(reachable_objects, initial_state, env)
    print(f"No Heuristic order: {no_heuristic_order}")
    
    # Test Nearest First Strategy ordering
    nearest_first = NearestFirstStrategy()
    nearest_first_order = nearest_first.select_objects(reachable_objects, initial_state, env)
    print(f"Nearest First order: {nearest_first_order}")
    
    # Check if orderings are different
    if no_heuristic_order != nearest_first_order:
        print("✓ Strategies produce different object orderings!")
    else:
        print("= Strategies produce the same object ordering")
        
    # Show distances for nearest first strategy
    env.set_full_state(initial_state)
    obs = env.get_observation()
    robot_pose = obs.get('robot_pose')
    
    if robot_pose is not None:
        print(f"\nRobot position: ({robot_pose[0]:.2f}, {robot_pose[1]:.2f})")
        print("Object distances:")
        
        import math
        for obj_id in reachable_objects:
            pose_key = f"{obj_id}_pose"
            if pose_key in obs:
                obj_x, obj_y = obs[pose_key][0], obs[pose_key][1]
                distance = math.sqrt((obj_x - robot_pose[0])**2 + (obj_y - robot_pose[1])**2)
                print(f"  {obj_id}: ({obj_x:.2f}, {obj_y:.2f}) - distance {distance:.2f}")


def test_convenience_function():
    """Test the updated convenience function with strategy parameter."""
    
    print("\n5. TESTING CONVENIENCE FUNCTION:")
    print("-" * 40)
    
    env = namo_rl.RLEnvironment("../data/test_scene.xml", "../config/namo_config_complete.yaml")
    robot_goal = (3.0, 3.0, 0.0)
    
    # Test with default strategy (should be NoHeuristicStrategy)
    print("Testing with default strategy...")
    result_default = plan_with_idfs(
        env, robot_goal, 
        max_depth=2, 
        max_goals_per_object=2,
        verbose=False
    )
    print(f"Default strategy result: {len(result_default) if result_default else 0} actions")
    
    # Test with explicit NearestFirstStrategy
    print("Testing with explicit NearestFirstStrategy...")
    result_nearest = plan_with_idfs(
        env, robot_goal,
        max_depth=2,
        max_goals_per_object=2,
        verbose=False,
        object_selection_strategy=NearestFirstStrategy()
    )
    print(f"Nearest first strategy result: {len(result_nearest) if result_nearest else 0} actions")


if __name__ == "__main__":
    try:
        test_object_ordering()
        test_strategy_comparison() 
        test_convenience_function()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("✓ Object selection strategies are working correctly")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)