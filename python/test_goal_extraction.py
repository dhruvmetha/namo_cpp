#!/usr/bin/env python3
"""
Comprehensive test suite for XML goal extraction functionality.

This script validates goal extraction across multiple environment files,
tests edge cases, and measures performance.
"""

import time
import statistics
from pathlib import Path
from xml_goal_parser import extract_goal_from_xml, extract_goal_with_fallback, validate_goal_coordinates, batch_extract_goals, GoalExtractionError


def test_goal_extraction_comprehensive():
    """Run comprehensive goal extraction tests."""
    
    print("🧪 Comprehensive XML Goal Extraction Tests")
    print("=" * 60)
    
    # Test 1: Basic extraction from sample files
    print("\n1️⃣ Basic Goal Extraction Test")
    sample_files = [
        "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100a.xml",
        "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100b.xml",
        "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100c.xml",
        "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100d.xml",
        "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100e.xml"
    ]
    
    goals_extracted = []
    for xml_file in sample_files:
        try:
            goal = extract_goal_from_xml(xml_file)
            valid = validate_goal_coordinates(goal)
            goals_extracted.append(goal)
            print(f"✅ {Path(xml_file).name}: {goal} {'✓' if valid else '✗'}")
        except GoalExtractionError as e:
            print(f"❌ {Path(xml_file).name}: {e}")
    
    # Test 2: Goal uniqueness
    print(f"\n2️⃣ Goal Uniqueness Test")
    unique_goals = set(goals_extracted)
    print(f"   Goals extracted: {len(goals_extracted)}")
    print(f"   Unique goals: {len(unique_goals)}")
    print(f"   Uniqueness: {'✅ PASS' if len(unique_goals) == len(goals_extracted) else '❌ FAIL'}")
    
    # Test 3: Performance test  
    print(f"\n3️⃣ Performance Test")
    times = []
    for xml_file in sample_files:
        start_time = time.time()
        try:
            goal = extract_goal_from_xml(xml_file)
            times.append(time.time() - start_time)
        except Exception:
            pass
    
    if times:
        avg_time = statistics.mean(times) * 1000  # Convert to ms
        max_time = max(times) * 1000
        print(f"   Average extraction time: {avg_time:.2f} ms")
        print(f"   Max extraction time: {max_time:.2f} ms") 
        print(f"   Performance: {'✅ PASS' if avg_time < 10 else '❌ SLOW'}")
    
    # Test 4: Fallback functionality
    print(f"\n4️⃣ Fallback Test")
    nonexistent_file = "nonexistent_file.xml"
    fallback_goal = (-1.0, -1.0, 0.0)
    
    result_goal = extract_goal_with_fallback(nonexistent_file, fallback_goal)
    fallback_works = result_goal == fallback_goal
    print(f"   Fallback goal: {fallback_goal}")
    print(f"   Result goal: {result_goal}")
    print(f"   Fallback: {'✅ PASS' if fallback_works else '❌ FAIL'}")
    
    # Test 5: Batch extraction
    print(f"\n5️⃣ Batch Extraction Test")
    batch_results = batch_extract_goals(sample_files)
    successful_batch = sum(1 for r in batch_results.values() if r['success'])
    print(f"   Files processed: {len(sample_files)}")
    print(f"   Successful extractions: {successful_batch}")
    print(f"   Batch extraction: {'✅ PASS' if successful_batch == len(sample_files) else '❌ FAIL'}")
    
    # Test 6: Goal coordinate validation
    print(f"\n6️⃣ Coordinate Validation Test")
    test_goals = [
        (0.0, 0.0, 0.0),      # Valid center
        (4.9, 4.9, 0.0),      # Valid near boundary
        (-4.9, -4.9, 0.0),    # Valid near boundary
        (10.0, 0.0, 0.0),     # Invalid - out of bounds
        (0.0, 10.0, 0.0),     # Invalid - out of bounds
        (float('nan'), 0.0, 0.0),  # Invalid - NaN
    ]
    
    validation_results = []
    for goal in test_goals:
        valid = validate_goal_coordinates(goal)
        validation_results.append(valid)
        status = "✅ Valid" if valid else "❌ Invalid"
        print(f"   {goal}: {status}")
    
    # Expected: first 3 valid, last 3 invalid
    expected_validations = [True, True, True, False, False, False]
    validation_correct = validation_results == expected_validations
    print(f"   Validation logic: {'✅ PASS' if validation_correct else '❌ FAIL'}")
    
    # Overall test summary
    print(f"\n🏁 Test Summary")
    print("=" * 30)
    
    test_results = [
        len(goals_extracted) == len(sample_files),  # Basic extraction
        len(unique_goals) == len(goals_extracted),  # Uniqueness
        avg_time < 10 if times else False,          # Performance  
        fallback_works,                             # Fallback
        successful_batch == len(sample_files),      # Batch
        validation_correct                          # Validation
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


def test_goal_diversity():
    """Test goal diversity across different environment sets."""
    
    print(f"\n🌟 Goal Diversity Analysis")
    print("=" * 40)
    
    # Find more XML files for diversity test
    import glob
    xml_pattern = "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/**/*.xml"
    all_xml_files = glob.glob(xml_pattern, recursive=True)[:20]  # Test first 20 files
    
    print(f"Analyzing {len(all_xml_files)} environment files...")
    
    goals = []
    failed_extractions = 0
    
    for xml_file in all_xml_files:
        try:
            goal = extract_goal_from_xml(xml_file)
            goals.append(goal)
        except Exception:
            failed_extractions += 1
    
    if goals:
        # Calculate statistics
        x_coords = [g[0] for g in goals]
        y_coords = [g[1] for g in goals]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        unique_goals = len(set(goals))
        
        print(f"   Goals extracted: {len(goals)}")
        print(f"   Failed extractions: {failed_extractions}")
        print(f"   Unique goals: {unique_goals}")
        print(f"   Goal diversity: {unique_goals/len(goals)*100:.1f}%")
        print(f"   X coordinate range: {x_range:.2f}")
        print(f"   Y coordinate range: {y_range:.2f}")
        print(f"   Average goal: ({statistics.mean(x_coords):.2f}, {statistics.mean(y_coords):.2f}, 0.0)")
        
        diversity_good = unique_goals / len(goals) > 0.8  # Expect > 80% unique
        range_good = x_range > 5.0 and y_range > 5.0     # Expect good spatial spread
        
        print(f"   Diversity assessment: {'✅ GOOD' if diversity_good and range_good else '⚠️ LIMITED'}")
        
        return diversity_good and range_good
    else:
        print("❌ No goals could be extracted!")
        return False


if __name__ == "__main__":
    """Run all goal extraction tests."""
    import sys
    
    # Run comprehensive tests
    test1_passed = test_goal_extraction_comprehensive()
    test2_passed = test_goal_diversity()
    
    overall_success = test1_passed and test2_passed
    
    print(f"\n🎯 Overall Test Result: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
    
    sys.exit(0 if overall_success else 1)