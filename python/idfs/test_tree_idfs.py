#!/usr/bin/env python3
"""Test script for Tree-IDFS algorithm and performance comparison.

This script provides comprehensive testing of the Tree-IDFS implementation
and comparison with the original IDFS algorithm.
"""

import sys
import os
import time
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import namo_rl
from base_planner import PlannerFactory, PlannerConfig, compare_planners
from xml_goal_parser import extract_goal_with_fallback

# Import planners to register them
from standard_idfs import StandardIterativeDeepeningDFS  
from tree_idfs import TreeIterativeDeepeningDFS


def test_single_environment():
    """Test both algorithms on a single environment."""
    
    print("🧪 Testing Tree-IDFS vs IDFS on single environment")
    print("=" * 50)
    
    # Setup environment
    xml_file = "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100a.xml"
    config_file = "config/namo_config_complete.yaml"
    
    try:
        env = namo_rl.RLEnvironment(xml_file, config_file, visualize=False)
        robot_goal = extract_goal_with_fallback(xml_file, (-0.5, 1.3, 0.0))
        
        print(f"Environment: {os.path.basename(xml_file)}")
        print(f"Robot goal: {robot_goal}")
        print()
        
        # Test configurations
        configs = {
            "idfs": PlannerConfig(
                max_depth=3,
                max_goals_per_object=3,
                random_seed=42,
                verbose=True,
                collect_stats=True
            ),
            "tree_idfs": PlannerConfig(
                max_depth=3,
                max_goals_per_object=3,
                random_seed=42,
                verbose=True,
                collect_stats=True
            )
        }
        
        # Run comparison
        results = compare_planners(env, robot_goal, configs, num_trials=3)
        
        # Analyze results
        print("\n📊 Results Summary:")
        print("-" * 30)
        
        for algo_name, trials in results.items():
            print(f"\n{algo_name.upper()}:")
            
            success_count = sum(1 for r in trials if r.solution_found)
            avg_time = sum(r.search_time_ms for r in trials if r.search_time_ms) / len(trials)
            avg_nodes = sum(r.nodes_expanded for r in trials if r.nodes_expanded) / len([r for r in trials if r.nodes_expanded])
            avg_checks = sum(r.terminal_checks for r in trials if r.terminal_checks) / len([r for r in trials if r.terminal_checks])
            
            print(f"  Success rate: {success_count}/{len(trials)} ({success_count/len(trials)*100:.1f}%)")
            print(f"  Avg search time: {avg_time:.1f}ms")
            print(f"  Avg nodes expanded: {avg_nodes:.1f}")
            print(f"  Avg terminal checks: {avg_checks:.1f}")
            
            # Show individual trial details
            for i, result in enumerate(trials):
                status = "✅" if result.solution_found else "❌"
                print(f"  Trial {i+1}: {status} {result.search_time_ms:.1f}ms, "
                      f"{result.nodes_expanded} nodes, {result.terminal_checks} checks")
        
        print("\n🎯 Key Insights:")
        idfs_results = results["idfs"]
        tree_results = results["tree_idfs"]
        
        idfs_avg_nodes = sum(r.nodes_expanded for r in idfs_results if r.nodes_expanded) / len([r for r in idfs_results if r.nodes_expanded])
        tree_avg_nodes = sum(r.nodes_expanded for r in tree_results if r.nodes_expanded) / len([r for r in tree_results if r.nodes_expanded])
        
        if tree_avg_nodes < idfs_avg_nodes:
            savings = (idfs_avg_nodes - tree_avg_nodes) / idfs_avg_nodes * 100
            print(f"✨ Tree-IDFS saved {savings:.1f}% node expansions on average")
        else:
            print("⚠️  Tree-IDFS didn't show expected node savings (possible implementation issue)")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_tree_persistence():
    """Test that Tree-IDFS properly maintains tree structure."""
    
    print("\n🌳 Testing Tree Persistence")
    print("=" * 30)
    
    xml_file = "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100a.xml"
    config_file = "config/namo_config_complete.yaml"
    
    try:
        env = namo_rl.RLEnvironment(xml_file, config_file, visualize=False)
        robot_goal = extract_goal_with_fallback(xml_file, (-0.5, 1.3, 0.0))
        
        config = PlannerConfig(
            max_depth=3,
            max_goals_per_object=2,
            random_seed=42,
            verbose=False,
            collect_stats=True
        )
        
        # Create Tree-IDFS planner
        planner = PlannerFactory.create_planner("tree_idfs", env, config)
        
        print("Running multiple searches to test tree persistence...")
        
        # Run multiple searches on the same planner
        for i in range(3):
            env.reset()
            result = planner.search(robot_goal)
            
            print(f"Search {i+1}: {result.nodes_expanded} nodes expanded, "
                  f"{result.terminal_checks} terminal checks")
            
            if i == 0:
                first_nodes = result.nodes_expanded
            else:
                if result.nodes_expanded < first_nodes:
                    print(f"  ✅ Tree reuse detected! Saved {first_nodes - result.nodes_expanded} expansions")
                else:
                    print(f"  ⚠️  No tree reuse benefit observed")
        
    except Exception as e:
        print(f"❌ Tree persistence test failed: {e}")


def performance_benchmark():
    """Benchmark Tree-IDFS vs IDFS performance."""
    
    print("\n⚡ Performance Benchmark")
    print("=" * 25)
    
    xml_file = "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9/easy/set1/benchmark_1/env_config_100a.xml"
    config_file = "config/namo_config_complete.yaml"
    
    try:
        env = namo_rl.RLEnvironment(xml_file, config_file, visualize=False)
        robot_goal = extract_goal_with_fallback(xml_file, (-0.5, 1.3, 0.0))
        
        # Benchmark configurations
        benchmark_configs = {
            "IDFS (depth=2)": ("idfs", PlannerConfig(max_depth=2, max_goals_per_object=3, random_seed=42, collect_stats=True)),
            "Tree-IDFS (depth=2)": ("tree_idfs", PlannerConfig(max_depth=2, max_goals_per_object=3, random_seed=42, collect_stats=True)),
            "IDFS (depth=3)": ("idfs", PlannerConfig(max_depth=3, max_goals_per_object=3, random_seed=42, collect_stats=True)),
            "Tree-IDFS (depth=3)": ("tree_idfs", PlannerConfig(max_depth=3, max_goals_per_object=3, random_seed=42, collect_stats=True)),
        }
        
        results = {}
        
        for name, (algo, config) in benchmark_configs.items():
            print(f"\nBenchmarking {name}...")
            
            trial_results = []
            for trial in range(5):
                env.reset()
                planner = PlannerFactory.create_planner(algo, env, config)
                planner.reset()
                
                start = time.time()
                result = planner.search(robot_goal)
                end = time.time()
                
                trial_results.append({
                    'time_ms': (end - start) * 1000,
                    'nodes_expanded': result.nodes_expanded,
                    'terminal_checks': result.terminal_checks,
                    'solution_found': result.solution_found
                })
            
            # Calculate averages
            avg_time = sum(r['time_ms'] for r in trial_results) / len(trial_results)
            avg_nodes = sum(r['nodes_expanded'] for r in trial_results if r['nodes_expanded']) / len([r for r in trial_results if r['nodes_expanded']])
            avg_checks = sum(r['terminal_checks'] for r in trial_results if r['terminal_checks']) / len([r for r in trial_results if r['terminal_checks']])
            success_rate = sum(1 for r in trial_results if r['solution_found']) / len(trial_results)
            
            results[name] = {
                'avg_time_ms': avg_time,
                'avg_nodes': avg_nodes,
                'avg_checks': avg_checks,
                'success_rate': success_rate
            }
            
            print(f"  Avg time: {avg_time:.1f}ms")
            print(f"  Avg nodes: {avg_nodes:.1f}")
            print(f"  Avg checks: {avg_checks:.1f}")
            print(f"  Success rate: {success_rate*100:.1f}%")
        
        # Compare algorithms
        print("\n📈 Performance Comparison:")
        print("-" * 40)
        
        for depth in [2, 3]:
            idfs_key = f"IDFS (depth={depth})"
            tree_key = f"Tree-IDFS (depth={depth})"
            
            if idfs_key in results and tree_key in results:
                idfs_nodes = results[idfs_key]['avg_nodes']
                tree_nodes = results[tree_key]['avg_nodes']
                
                if tree_nodes < idfs_nodes:
                    savings = (idfs_nodes - tree_nodes) / idfs_nodes * 100
                    print(f"Depth {depth}: Tree-IDFS saved {savings:.1f}% nodes")
                else:
                    overhead = (tree_nodes - idfs_nodes) / idfs_nodes * 100
                    print(f"Depth {depth}: Tree-IDFS used {overhead:.1f}% more nodes")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


def main():
    """Run all Tree-IDFS tests."""
    
    print("🚀 Tree-IDFS Testing Suite")
    print("=" * 50)
    
    # Check available planners
    available = PlannerFactory.list_available_planners()
    print(f"Available planners: {available}")
    
    if "tree_idfs" not in available:
        print("❌ Tree-IDFS not available! Check implementation.")
        return 1
    
    try:
        # Run tests
        test_single_environment()
        test_tree_persistence()
        performance_benchmark()
        
        print("\n✅ All tests completed!")
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())