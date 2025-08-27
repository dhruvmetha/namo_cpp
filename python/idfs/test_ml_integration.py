"""Integration test for ML-enhanced IDFS planner.

This script tests the integration of ML-based object selection and goal generation
strategies with the IDFS planner, including fallback behavior and error handling.
"""

import sys
import os
import argparse

# Ensure namo_rl is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import namo_rl

# Import IDFS components
from idfs.standard_idfs import plan_with_idfs, plan_with_ml_idfs
from idfs.ml_strategies import MLObjectSelectionStrategy, MLGoalSelectionStrategy
from idfs.object_selection_strategy import NearestFirstStrategy, GoalProximityStrategy
from idfs.goal_selection_strategy import RandomGoalStrategy, GridGoalStrategy


def test_basic_strategies():
    """Test basic strategy functionality without ML models."""
    print("=== Testing Basic Strategy Integration ===")
    
    # Create mock environment (this would need to be replaced with real environment)
    try:
        env = namo_rl.RLEnvironment("../data/test_scene.xml", "../config/namo_config_complete.yaml")
        print("✅ Environment loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        print("Skipping basic strategy tests")
        return False
    
    # Test different goal strategies with heuristic object selection
    strategies_to_test = [
        ("Random Goals + Nearest Objects", RandomGoalStrategy(), NearestFirstStrategy()),
        ("Grid Goals + Goal Proximity", GridGoalStrategy(num_angles=4, num_distances=2), GoalProximityStrategy()),
    ]
    
    robot_goal = (1.5, 2.0, 0.0)
    
    for strategy_name, goal_strategy, object_strategy in strategies_to_test:
        print(f"\n--- Testing {strategy_name} ---")
        try:
            result = plan_with_idfs(
                env=env,
                robot_goal=robot_goal,
                max_depth=2,  # Shallow depth for quick testing
                max_goals_per_object=3,
                verbose=True,
                object_selection_strategy=object_strategy,
                goal_selection_strategy=goal_strategy
            )
            
            if result:
                print(f"✅ Found solution with {len(result)} actions")
                for i, action in enumerate(result):
                    print(f"  Action {i}: {action.object_id} -> ({action.x:.2f}, {action.y:.2f})")
            else:
                print("ℹ️  No solution found (expected for shallow depth)")
                
        except Exception as e:
            print(f"❌ Strategy failed: {e}")
    
    return True


def test_ml_strategies_fallback():
    """Test ML strategies with fallback behavior (models don't need to exist)."""
    print("\n=== Testing ML Strategy Fallback Behavior ===")
    
    try:
        env = namo_rl.RLEnvironment("../data/test_scene.xml", "../config/namo_config_complete.yaml")
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        return False
    
    # Test ML strategies with non-existent model paths (should fall back to heuristics)
    print("Testing ML object selection with fallback...")
    
    ml_object_strategy = MLObjectSelectionStrategy(
        object_model_path="fake/nonexistent/path",
        fallback_strategy=NearestFirstStrategy(),
        verbose=True
    )
    
    ml_goal_strategy = MLGoalSelectionStrategy(
        goal_model_path="fake/nonexistent/goal/path", 
        object_model_path="fake/nonexistent/object/path",
        fallback_strategy=RandomGoalStrategy(),
        verbose=True
    )
    
    robot_goal = (1.5, 2.0, 0.0)
    
    try:
        result = plan_with_idfs(
            env=env,
            robot_goal=robot_goal,
            max_depth=2,
            max_goals_per_object=3,
            verbose=True,
            object_selection_strategy=ml_object_strategy,
            goal_selection_strategy=ml_goal_strategy
        )
        
        if result:
            print(f"✅ Fallback strategies worked - found solution with {len(result)} actions")
        else:
            print("ℹ️  Fallback strategies worked - no solution found (expected for shallow depth)")
            
    except Exception as e:
        print(f"❌ ML strategy fallback failed: {e}")
        return False
    
    return True


def test_ml_strategies_real_models(object_model_path: str, goal_model_path: str):
    """Test ML strategies with real trained models."""
    print("\n=== Testing ML Strategies with Real Models ===")
    
    if not os.path.exists(object_model_path):
        print(f"❌ Object model not found at {object_model_path}")
        return False
        
    if not os.path.exists(goal_model_path):
        print(f"❌ Goal model not found at {goal_model_path}")
        return False
    
    try:
        env = namo_rl.RLEnvironment("../data/test_scene.xml", "../config/namo_config_complete.yaml")
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        return False
    
    robot_goal = (1.5, 2.0, 0.0)
    
    print("Testing ML-enhanced IDFS with real models...")
    
    try:
        result = plan_with_ml_idfs(
            env=env,
            robot_goal=robot_goal,
            object_model_path=object_model_path,
            goal_model_path=goal_model_path,
            max_depth=3,
            max_goals_per_object=5,
            verbose=True,
            ml_samples=16,  # Fewer samples for faster testing
            device="cuda",
            fallback_to_heuristics=True
        )
        
        if result:
            print(f"🎯 ML-enhanced IDFS found solution with {len(result)} actions:")
            for i, action in enumerate(result):
                print(f"  Action {i}: {action.object_id} -> ({action.x:.2f}, {action.y:.2f})")
        else:
            print("ℹ️  ML-enhanced IDFS completed but no solution found")
            
    except Exception as e:
        print(f"❌ ML-enhanced IDFS failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test ML-enhanced IDFS integration")
    parser.add_argument("--object-model", 
                       help="Path to object inference model")
    parser.add_argument("--goal-model", 
                       help="Path to goal inference model")
    parser.add_argument("--skip-basic", action="store_true", 
                       help="Skip basic strategy tests")
    parser.add_argument("--skip-fallback", action="store_true", 
                       help="Skip fallback behavior tests")
    parser.add_argument("--only-ml", action="store_true", 
                       help="Only test ML models (requires both model paths)")
    
    args = parser.parse_args()
    
    success_count = 0
    total_tests = 0
    
    if not args.only_ml:
        if not args.skip_basic:
            total_tests += 1
            if test_basic_strategies():
                success_count += 1
        
        if not args.skip_fallback:
            total_tests += 1
            if test_ml_strategies_fallback():
                success_count += 1
    
    if args.object_model and args.goal_model:
        total_tests += 1
        if test_ml_strategies_real_models(args.object_model, args.goal_model):
            success_count += 1
    elif args.only_ml:
        print("❌ --only-ml requires both --object-model and --goal-model")
        return 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())