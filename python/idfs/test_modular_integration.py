#!/usr/bin/env python3
"""Test script to validate modular integration of ML strategies.

This script tests that the modular parallel collection system can correctly
instantiate and use ML-enhanced IDFS planners without actually running
full data collection.
"""

import sys
import os
from unittest.mock import Mock, patch

# Ensure namo_rl is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modular system
from idfs.modular_parallel_collection import (
    ModularCollectionConfig, 
    ModularParallelCollectionManager,
    get_available_object_strategies,
    get_available_goal_strategies,
    validate_object_strategy,
    validate_goal_strategy
)

from idfs.base_planner import PlannerConfig, PlannerFactory
import namo_rl


def test_strategy_discovery():
    """Test that strategy discovery functions work correctly."""
    print("=== Testing Strategy Discovery ===")
    
    # Test object strategies
    obj_strategies = get_available_object_strategies()
    print(f"Available object strategies: {obj_strategies}")
    
    expected_obj = ["no_heuristic", "nearest_first", "goal_proximity", "farthest_first", "ml"]
    for strategy in expected_obj:
        if strategy in obj_strategies:
            print(f"✅ Object strategy '{strategy}' available")
            assert validate_object_strategy(strategy), f"Validation failed for {strategy}"
        else:
            print(f"❌ Object strategy '{strategy}' missing")
    
    # Test goal strategies
    goal_strategies = get_available_goal_strategies()
    print(f"Available goal strategies: {goal_strategies}")
    
    expected_goal = ["random", "grid", "adaptive", "ml"]
    for strategy in expected_goal:
        if strategy in goal_strategies:
            print(f"✅ Goal strategy '{strategy}' available") 
            assert validate_goal_strategy(strategy), f"Validation failed for {strategy}"
        else:
            print(f"❌ Goal strategy '{strategy}' missing")
    
    # Test invalid strategies
    assert not validate_object_strategy("invalid_strategy"), "Should reject invalid object strategy"
    assert not validate_goal_strategy("invalid_strategy"), "Should reject invalid goal strategy"
    print("✅ Invalid strategy rejection works correctly")
    
    return True


def test_config_creation():
    """Test that configurations are created correctly."""
    print("\n=== Testing Configuration Creation ===")
    
    # Test basic config
    config = ModularCollectionConfig(
        output_dir="/tmp/test",
        start_idx=0,
        end_idx=2,
        object_selection_strategy="nearest_first",
        goal_selection_strategy="random"
    )
    
    print(f"✅ Basic config created: obj={config.object_selection_strategy}, goal={config.goal_selection_strategy}")
    
    # Test ML config
    ml_config = ModularCollectionConfig(
        output_dir="/tmp/test_ml",
        start_idx=0,
        end_idx=2,
        object_selection_strategy="ml",
        goal_selection_strategy="ml",
        ml_object_model_path="/fake/object/path",
        ml_goal_model_path="/fake/goal/path",
        ml_samples=16,
        ml_device="cpu",
        ml_fallback=True
    )
    
    print(f"✅ ML config created: samples={ml_config.ml_samples}, device={ml_config.ml_device}")
    
    return True


def test_planner_creation_heuristic():
    """Test that heuristic planners can be created."""
    print("\n=== Testing Heuristic Planner Creation ===")
    
    try:
        # Create mock environment
        env = Mock(spec=namo_rl.RLEnvironment)
        env.get_action_constraints.return_value = Mock(
            min_distance=0.2,
            max_distance=0.8,
            theta_min=0.0,
            theta_max=6.28
        )
        
        # Test different heuristic combinations
        test_configs = [
            ("nearest_first", "random"),
            ("goal_proximity", "grid"),
            ("no_heuristic", "adaptive")
        ]
        
        for obj_strategy, goal_strategy in test_configs:
            config = PlannerConfig(
                max_depth=3,
                max_goals_per_object=3,
                algorithm_params={
                    'object_selection_strategy': obj_strategy,
                    'goal_selection_strategy': goal_strategy
                }
            )
            
            planner = PlannerFactory.create_planner("idfs", env, config)
            
            assert hasattr(planner, 'object_selection_strategy')
            assert hasattr(planner, 'goal_selection_strategy')
            
            print(f"✅ Created IDFS with {obj_strategy} + {goal_strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ Heuristic planner creation failed: {e}")
        return False


def test_planner_creation_ml_fallback():
    """Test that ML planners fall back correctly when models don't exist."""
    print("\n=== Testing ML Planner Fallback ===")
    
    try:
        # Create mock environment
        env = Mock(spec=namo_rl.RLEnvironment)
        env.get_action_constraints.return_value = Mock(
            min_distance=0.2,
            max_distance=0.8,
            theta_min=0.0,
            theta_max=6.28
        )
        
        # Test ML config with non-existent model paths
        config = PlannerConfig(
            max_depth=3,
            max_goals_per_object=3,
            verbose=True,
            algorithm_params={
                'object_selection_strategy': 'ml',
                'goal_selection_strategy': 'ml',
                'ml_object_model_path': '/fake/nonexistent/object/path',
                'ml_goal_model_path': '/fake/nonexistent/goal/path',
                'ml_samples': 16,
                'ml_device': 'cpu',
                'ml_fallback': True
            }
        )
        
        # This should create the planner but ML strategies should fall back to heuristics
        planner = PlannerFactory.create_planner("idfs", env, config)
        
        assert hasattr(planner, 'object_selection_strategy')
        assert hasattr(planner, 'goal_selection_strategy')
        
        print(f"✅ Created ML planner with fallback capability")
        print(f"   Object strategy: {planner.object_selection_strategy.__class__.__name__}")
        print(f"   Goal strategy: {planner.goal_selection_strategy.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML planner creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manager_initialization():
    """Test that the collection manager initializes correctly."""
    print("\n=== Testing Manager Initialization ===")
    
    try:
        # Test with heuristic strategies
        config = ModularCollectionConfig(
            output_dir="/tmp/test_manager",
            start_idx=0,
            end_idx=2,
            object_selection_strategy="nearest_first",
            goal_selection_strategy="random"
        )
        
        manager = ModularParallelCollectionManager(config)
        
        assert manager.config.hostname is not None
        assert manager.config.planner_config is not None
        assert manager.config.planner_config.algorithm_params is not None
        
        params = manager.config.planner_config.algorithm_params
        assert params['object_selection_strategy'] == 'nearest_first'
        assert params['goal_selection_strategy'] == 'random'
        
        print(f"✅ Manager initialized with hostname: {manager.config.hostname}")
        print(f"   Object strategy in config: {params['object_selection_strategy']}")
        print(f"   Goal strategy in config: {params['goal_selection_strategy']}")
        
        # Test with ML strategies
        ml_config = ModularCollectionConfig(
            output_dir="/tmp/test_ml_manager",
            start_idx=0,
            end_idx=2,
            object_selection_strategy="ml",
            goal_selection_strategy="ml",
            ml_object_model_path="/fake/object/path",
            ml_goal_model_path="/fake/goal/path"
        )
        
        ml_manager = ModularParallelCollectionManager(ml_config)
        
        ml_params = ml_manager.config.planner_config.algorithm_params
        assert ml_params['object_selection_strategy'] == 'ml'
        assert ml_params['goal_selection_strategy'] == 'ml'
        assert ml_params['ml_object_model_path'] == '/fake/object/path'
        assert ml_params['ml_goal_model_path'] == '/fake/goal/path'
        
        print(f"✅ ML manager initialized correctly")
        print(f"   ML object model: {ml_params['ml_object_model_path']}")
        print(f"   ML goal model: {ml_params['ml_goal_model_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manager initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("ML-Enhanced IDFS Modular Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Strategy Discovery", test_strategy_discovery),
        ("Config Creation", test_config_creation),
        ("Heuristic Planner Creation", test_planner_creation_heuristic),
        ("ML Planner Fallback", test_planner_creation_ml_fallback),
        ("Manager Initialization", test_manager_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n🎉 {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n💥 {test_name}: ERROR - {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\nYou can now run ML-enhanced IDFS using:")
        print("python python/idfs/modular_parallel_collection.py --object-strategy ml --goal-strategy ml ...")
        return 0
    else:
        print("❌ Some tests failed - check the output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())