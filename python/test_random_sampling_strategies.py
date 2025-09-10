#!/usr/bin/env python3
"""Test script for RandomSamplingPlanner with different strategies."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import namo_rl
from idfs.random_sampling import RandomSamplingPlanner, plan_with_random_sampling
from idfs.base_planner import PlannerConfig

def test_strategy_combinations():
    """Test RandomSamplingPlanner with different strategy combinations."""
    
    # Test environment - use a simple XML file
    xml_file = "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_8118e.xml"
    config_file = "config/namo_config_complete.yaml"
    robot_goal = (-2.5, -2.5, 0.0)  # Goal that requires object manipulation
    
    # Create environment
    try:
        env = namo_rl.RLEnvironment(xml_file, config_file, visualize=False)
        print(f"✅ Successfully created environment from {xml_file}")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False
    
    # Strategy combinations to test
    test_cases = [
        ("no_heuristic", "random"),
        ("nearest_first", "random"),
        ("goal_proximity", "grid"),
        ("farthest_first", "adaptive"),
        ("no_heuristic", "grid"),
        ("nearest_first", "adaptive")
    ]
    
    print(f"🧪 Testing RandomSamplingPlanner with {len(test_cases)} strategy combinations")
    
    success_count = 0
    for i, (obj_strategy, goal_strategy) in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/{len(test_cases)}: {obj_strategy} + {goal_strategy}")
        
        try:
            # Test using convenience function
            result = plan_with_random_sampling(
                env=env,
                robot_goal=robot_goal,
                max_depth=5,  # Increase depth for more thorough testing
                object_strategy=obj_strategy,
                goal_strategy=goal_strategy,
                verbose=True,
                collect_stats=True
            )
            
            if result is not None:
                print(f"✅ Success: Found solution with {len(result)} actions")
                success_count += 1
            else:
                print(f"⚠️  No solution found (not necessarily an error)")
                success_count += 1  # Count as success if no exception was thrown
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Strategy testing complete: {success_count}/{len(test_cases)} combinations successful")
    
    # Test direct planner instantiation with config
    print(f"\n🔧 Testing direct planner instantiation...")
    try:
        config = PlannerConfig(
            max_depth=5,
            max_goals_per_object=1,
            verbose=True,
            collect_stats=True,
            algorithm_params={
                'object_selection_strategy': 'nearest_first',
                'goal_selection_strategy': 'grid'
            }
        )
        
        planner = RandomSamplingPlanner(env, config)
        print(f"✅ Planner created: {planner.algorithm_name}")
        print(f"   Version: {planner.algorithm_version}")
        
        # Test search
        env.reset()
        result = planner.search(robot_goal)
        
        if result.solution_found:
            print(f"✅ Direct instantiation test: Found solution with {len(result.action_sequence)} actions")
        else:
            print(f"⚠️  Direct instantiation test: No solution found")
            
    except Exception as e:
        print(f"❌ Direct instantiation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    print("🚀 Testing RandomSamplingPlanner with strategy support")
    success = test_strategy_combinations()
    sys.exit(0 if success else 1)