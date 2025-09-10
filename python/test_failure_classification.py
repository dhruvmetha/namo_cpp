#!/usr/bin/env python3
"""Test the failure classification system with realistic error scenarios."""

import os
import sys

# Add idfs directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'idfs'))

from idfs.failure_codes import FailureCode, FailureClassifier, create_failure_info, get_failure_statistics

def test_basic_classification():
    """Test basic error message classification."""
    print("🧪 Testing basic failure classification...")
    
    test_cases = [
        ("Search timeout exceeded after 300 seconds", FailureCode.TIMEOUT),
        ("No reachable objects available for manipulation", FailureCode.NO_REACHABLE_OBJECTS),
        ("Robot collision detected during push action", FailureCode.ROBOT_COLLISION),
        ("Failed to load ML model from path", FailureCode.ML_MODEL_LOAD_FAILED),
        ("Maximum depth of 5 reached without solution", FailureCode.MAX_DEPTH_REACHED),
        ("Terminal check limit exceeded", FailureCode.MAX_TERMINAL_CHECKS),
        ("Environment reset failed", FailureCode.ENVIRONMENT_RESET_FAILED),
        ("Out of memory error", FailureCode.MEMORY_ERROR),
        ("Some random unknown error occurred", FailureCode.UNKNOWN_ERROR)
    ]
    
    correct = 0
    for error_msg, expected_code in test_cases:
        classified_code = FailureClassifier.classify_failure(error_msg)
        if classified_code == expected_code:
            print(f"  ✅ '{error_msg}' -> {classified_code.name}")
            correct += 1
        else:
            print(f"  ❌ '{error_msg}' -> Expected: {expected_code.name}, Got: {classified_code.name}")
    
    print(f"  📊 Classification accuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.1f}%)")
    return correct == len(test_cases)

def test_exception_classification():
    """Test classification with actual exception objects."""
    print("\n🧪 Testing exception-based classification...")
    
    correct = 0
    total = 0
    
    try:
        raise TimeoutError("Planning exceeded time limit")
    except Exception as e:
        failure_info = create_failure_info(str(e), e)
        expected_code = FailureCode.TIMEOUT
        total += 1
        if failure_info['failure_code'] == expected_code:
            print(f"  ✅ TimeoutError -> {FailureCode(failure_info['failure_code']).name}")
            correct += 1
        else:
            print(f"  ❌ TimeoutError -> Expected: {expected_code.name}, Got: {FailureCode(failure_info['failure_code']).name}")
    
    try:
        raise MemoryError("Insufficient memory")
    except Exception as e:
        failure_info = create_failure_info(str(e), e)
        expected_code = FailureCode.MEMORY_ERROR
        total += 1
        if failure_info['failure_code'] == expected_code:
            print(f"  ✅ MemoryError -> {FailureCode(failure_info['failure_code']).name}")
            correct += 1
        else:
            print(f"  ❌ MemoryError -> Expected: {expected_code.name}, Got: {FailureCode(failure_info['failure_code']).name}")
    
    return correct == total

def test_failure_statistics():
    """Test failure statistics calculation."""
    print("\n🧪 Testing failure statistics...")
    
    # Create mock episode results with different failure codes
    mock_episodes = [
        {'success': True, 'solution_found': True},  # Success
        {'success': True, 'solution_found': True},  # Success
        {'success': False, 'solution_found': False, 'failure_code': int(FailureCode.TIMEOUT)},
        {'success': False, 'solution_found': False, 'failure_code': int(FailureCode.TIMEOUT)},
        {'success': False, 'solution_found': False, 'failure_code': int(FailureCode.NO_REACHABLE_OBJECTS)},
        {'success': False, 'solution_found': False, 'failure_code': int(FailureCode.ROBOT_COLLISION)},
        {'success': False, 'solution_found': False, 'failure_code': int(FailureCode.UNKNOWN_ERROR)},
    ]
    
    stats = get_failure_statistics(mock_episodes)
    
    print(f"  📊 Total episodes: {stats['total_episodes']}")
    print(f"  ✅ Successful episodes: {stats['successful_episodes']}")
    print(f"  ❌ Failed episodes: {stats['failed_episodes']}")
    print(f"  📈 Success rate: {stats['success_rate']:.1f}%")
    
    print("  Top failures:")
    if stats['failure_breakdown']:
        for desc, info in sorted(stats['failure_breakdown'].items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"    • {desc}: {info['count']} episodes ({info['percentage']:.1f}%)")
    
    # Basic validation
    expected_total = 7
    expected_success = 2
    expected_failed = 5
    
    if (stats['total_episodes'] == expected_total and 
        stats['successful_episodes'] == expected_success and 
        stats['failed_episodes'] == expected_failed):
        print("  ✅ Statistics calculation correct")
        return True
    else:
        print(f"  ❌ Statistics incorrect - Expected: {expected_total}/{expected_success}/{expected_failed}")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing edge cases...")
    
    # Empty error message
    code = FailureClassifier.classify_failure("")
    print(f"  📝 Empty message -> {code.name} (expected: UNKNOWN_ERROR)")
    
    # None error message
    code = FailureClassifier.classify_failure(None)
    print(f"  📝 None message -> {code.name} (expected: UNKNOWN_ERROR)")
    
    # Very long error message with multiple patterns
    long_msg = "The system encountered a timeout while checking for robot collision during object manipulation which failed due to no reachable objects"
    code = FailureClassifier.classify_failure(long_msg)
    print(f"  📝 Complex message -> {code.name} (should match first pattern found)")
    
    # Empty episode list for statistics
    empty_stats = get_failure_statistics([])
    print(f"  📝 Empty episodes -> Success rate: {empty_stats['success_rate']:.1f}% (expected: 0.0%)")
    
    return True

def main():
    """Run all failure classification tests."""
    print("🔬 Testing NAMO Failure Classification System")
    print("=" * 60)
    
    tests = [
        ("Basic Classification", test_basic_classification),
        ("Exception Classification", test_exception_classification), 
        ("Failure Statistics", test_failure_statistics),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"  🎯 {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"  💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Final summary
    print(f"\n{'='*60}")
    print("📈 Test Results:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Failure classification system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the failure classification system.")
        return 1

if __name__ == "__main__":
    sys.exit(main())