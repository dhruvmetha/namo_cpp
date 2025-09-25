#!/usr/bin/env python3
"""
Test script for action refinement functionality.

This script validates that:
1. SolutionRefiner can be imported and instantiated
2. Basic refinement operations work with mock data
3. The integration with data collection pipeline is functional
"""

import sys
import os
sys.path.append('python')

def test_solution_refiner_import():
    """Test that SolutionRefiner can be imported."""
    try:
        from namo.planners.idfs.solution_refiner import SolutionRefiner
        refiner = SolutionRefiner()
        print("✅ SolutionRefiner imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import SolutionRefiner: {e}")
        return False

def test_modular_episode_result_fields():
    """Test that ModularEpisodeResult has the new refinement fields."""
    try:
        from namo.data_collection.modular_parallel_collection import ModularEpisodeResult
        from dataclasses import fields

        field_names = [f.name for f in fields(ModularEpisodeResult)]

        required_fields = ['refined_action_sequence', 'refinement_accepted', 'refinement_stats']
        missing_fields = [f for f in required_fields if f not in field_names]

        if missing_fields:
            print(f"❌ Missing refinement fields in ModularEpisodeResult: {missing_fields}")
            return False
        else:
            print("✅ ModularEpisodeResult has all required refinement fields")
            return True
    except Exception as e:
        print(f"❌ Failed to check ModularEpisodeResult fields: {e}")
        return False

def test_modular_worker_task_fields():
    """Test that ModularWorkerTask has the new refinement configuration."""
    try:
        from namo.data_collection.modular_parallel_collection import ModularWorkerTask
        from dataclasses import fields

        field_names = [f.name for f in fields(ModularWorkerTask)]

        required_fields = ['refine_actions', 'validate_refinement']
        missing_fields = [f for f in required_fields if f not in field_names]

        if missing_fields:
            print(f"❌ Missing refinement configuration in ModularWorkerTask: {missing_fields}")
            return False
        else:
            print("✅ ModularWorkerTask has all required refinement configuration fields")
            return True
    except Exception as e:
        print(f"❌ Failed to check ModularWorkerTask fields: {e}")
        return False

def test_refinement_functions_exist():
    """Test that refinement functions are available in the module."""
    try:
        from namo.data_collection.modular_parallel_collection import apply_action_refinement
        print("✅ apply_action_refinement function is available")
        return True
    except ImportError as e:
        print(f"❌ apply_action_refinement function not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error importing refinement functions: {e}")
        return False

def test_position_deviation_calculation():
    """Test position deviation calculation logic."""
    try:
        from namo.planners.idfs.solution_refiner import SolutionRefiner

        refiner = SolutionRefiner()

        # Test exact match
        target = (1.0, 2.0, 0.0)
        actual = (1.0, 2.0, 0.0)
        deviation = refiner.calculate_position_deviation(target, actual)

        if abs(deviation) < 1e-6:
            print("✅ Position deviation calculation works for exact match")
        else:
            print(f"❌ Expected deviation ~0.0, got {deviation}")
            return False

        # Test known distance
        target = (0.0, 0.0, 0.0)
        actual = (3.0, 4.0, 0.0)  # Should be 5.0 units away
        deviation = refiner.calculate_position_deviation(target, actual)

        if abs(deviation - 5.0) < 1e-6:
            print("✅ Position deviation calculation works for 3-4-5 triangle")
        else:
            print(f"❌ Expected deviation ~5.0, got {deviation}")
            return False

        return True
    except Exception as e:
        print(f"❌ Position deviation calculation test failed: {e}")
        return False

def test_mock_refinement_data_structures():
    """Test refinement with mock data structures."""
    try:
        from namo.planners.idfs.solution_refiner import SolutionRefiner

        refiner = SolutionRefiner()

        # Mock action sequence
        action_sequence = [
            {"object_id": "box_movable", "target": (2.0, 1.0, 0.0)}
        ]

        # Mock post-action states (simulating slight physics discrepancy)
        post_action_states = [
            {"box_movable_pose": [1.95, 0.98, 0.02]}  # Slightly different from target
        ]

        # Test extraction without validation
        refined_actions, deviations = refiner.extract_actual_positions(action_sequence, post_action_states)

        if len(refined_actions) == 1 and len(deviations) == 1:
            print("✅ Action refinement extraction works with mock data")

            # Check that actual position was used
            refined_target = refined_actions[0]["target"]
            expected_target = (1.95, 0.98, 0.02)

            if refined_target == expected_target:
                print("✅ Refined action uses actual achieved position")
            else:
                print(f"❌ Expected refined target {expected_target}, got {refined_target}")
                return False

            # Check deviation calculation
            if 0.05 < deviations[0] < 0.1:  # Approximate expected deviation
                print("✅ Position deviation calculated correctly")
            else:
                print(f"❌ Unexpected deviation value: {deviations[0]}")
                return False

        else:
            print(f"❌ Expected 1 action and 1 deviation, got {len(refined_actions)} and {len(deviations)}")
            return False

        return True
    except Exception as e:
        print(f"❌ Mock refinement test failed: {e}")
        return False

def main():
    """Run all refinement tests."""
    print("🧪 Testing Action Refinement Implementation")
    print("=" * 50)

    tests = [
        test_solution_refiner_import,
        test_modular_episode_result_fields,
        test_modular_worker_task_fields,
        test_refinement_functions_exist,
        test_position_deviation_calculation,
        test_mock_refinement_data_structures
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Action refinement is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())