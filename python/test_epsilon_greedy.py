#!/usr/bin/env python3
"""Test script for epsilon-greedy goal strategy implementation."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from idfs.ml_strategies import EpsilonGreedyGoalStrategy, MLGoalSelectionStrategy
from idfs.goal_selection_strategy import RandomGoalStrategy
import namo_rl

def test_epsilon_greedy_basic():
    """Test basic epsilon-greedy functionality."""
    print("🧪 Testing EpsilonGreedyGoalStrategy...")
    
    # Create mock ML and Random strategies for testing
    class MockMLStrategy:
        def generate_goals(self, object_id, state, env, max_goals):
            from idfs.goal_selection_strategy import Goal
            return [Goal(x=1.0, y=1.0, theta=0.0) for _ in range(max_goals)]
        
        @property 
        def strategy_name(self):
            return "Mock ML"
    
    class MockRandomStrategy:
        def generate_goals(self, object_id, state, env, max_goals):
            from idfs.goal_selection_strategy import Goal
            return [Goal(x=2.0, y=2.0, theta=1.57) for _ in range(max_goals)]
        
        @property
        def strategy_name(self):
            return "Mock Random"
    
    # Test different epsilon values
    for epsilon in [0.0, 0.5, 1.0]:
        print(f"  Testing epsilon={epsilon}")
        
        strategy = EpsilonGreedyGoalStrategy(
            ml_strategy=MockMLStrategy(),
            random_strategy=MockRandomStrategy(), 
            epsilon=epsilon,
            verbose=True
        )
        
        # Generate goals (mock environment)
        goals = strategy.generate_goals("test_object", None, None, 5)
        
        print(f"    Generated {len(goals)} goals")
        print(f"    Strategy name: {strategy.strategy_name}")
        
        # Count ML vs Random goals (based on x coordinate)
        ml_count = sum(1 for goal in goals if goal.x == 1.0)
        random_count = sum(1 for goal in goals if goal.x == 2.0)
        
        print(f"    ML goals: {ml_count}, Random goals: {random_count}")
        
        # Verify expectations
        if epsilon == 0.0:
            assert ml_count == 5 and random_count == 0, f"Expected pure ML for epsilon=0.0"
        elif epsilon == 1.0:
            assert ml_count == 0 and random_count == 5, f"Expected pure random for epsilon=1.0"
        
        print(f"    ✅ epsilon={epsilon} test passed\n")

def test_epsilon_validation():
    """Test epsilon parameter validation."""
    print("🧪 Testing epsilon validation...")
    
    class DummyStrategy:
        def generate_goals(self, *args):
            return []
        @property
        def strategy_name(self):
            return "Dummy"
    
    # Test invalid epsilon values
    for invalid_epsilon in [-0.1, 1.1, 2.0]:
        try:
            EpsilonGreedyGoalStrategy(
                ml_strategy=DummyStrategy(),
                random_strategy=DummyStrategy(),
                epsilon=invalid_epsilon
            )
            assert False, f"Should have raised ValueError for epsilon={invalid_epsilon}"
        except ValueError:
            print(f"    ✅ Correctly rejected epsilon={invalid_epsilon}")
    
    # Test valid epsilon values
    for valid_epsilon in [0.0, 0.1, 0.5, 0.9, 1.0]:
        try:
            strategy = EpsilonGreedyGoalStrategy(
                ml_strategy=DummyStrategy(),
                random_strategy=DummyStrategy(),
                epsilon=valid_epsilon
            )
            print(f"    ✅ Correctly accepted epsilon={valid_epsilon}")
        except ValueError:
            assert False, f"Should not have raised ValueError for epsilon={valid_epsilon}"

if __name__ == "__main__":
    print("🚀 Starting epsilon-greedy goal strategy tests...\n")
    
    try:
        test_epsilon_validation()
        print()
        test_epsilon_greedy_basic()
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)