#!/usr/bin/env python3
"""Example script for running ML-enhanced IDFS with modular parallel collection.

This script demonstrates how to use the modular parallel collection system
with ML-based object and goal selection strategies.
"""

import os
import sys
import argparse
from pathlib import Path

def run_basic_ml_collection():
    """Run basic ML-enhanced data collection with minimal workers."""
    
    # Example model paths - these would need to be adjusted to actual trained models
    object_model_path = "/common/home/dm1487/robotics_research/ktamp/learning/outputs/rel_reach_coord_object_dit/mse/2025-08-10_05-33-43"
    goal_model_path = "/common/home/dm1487/robotics_research/ktamp/learning/outputs/rel_reach_coord_goal_dit/mse/2025-08-10_06-59-27"
    
    # Check if model paths exist
    if not os.path.exists(object_model_path):
        print(f"❌ Object model path does not exist: {object_model_path}")
        print("Please update the path in this script to point to your trained object model")
        return 1
    
    if not os.path.exists(goal_model_path):
        print(f"❌ Goal model path does not exist: {goal_model_path}")
        print("Please update the path in this script to point to your trained goal model")
        return 1
    
    # Command to run modular parallel collection with ML strategies
    cmd_args = [
        "python", "python/idfs/modular_parallel_collection.py",
        "--algorithm", "idfs",
        "--object-strategy", "ml", 
        "--goal-strategy", "ml",
        "--ml-object-model", object_model_path,
        "--ml-goal-model", goal_model_path,
        "--ml-samples", "32",
        "--ml-device", "cuda",
        "--output-dir", "/tmp/ml_idfs_test",
        "--start-idx", "0",
        "--end-idx", "2",  # Just 2 environments for testing
        "--workers", "1",  # Minimal workers as requested
        "--episodes-per-env", "1",
        "--max-depth", "5",
        "--max-goals-per-object", "5",
        "--search-timeout", "120.0",  # 2 minutes per search
        "--verbose"
    ]
    
    print("Running ML-enhanced IDFS data collection...")
    print(f"Command: {' '.join(cmd_args)}")
    
    # Execute the command
    import subprocess
    try:
        result = subprocess.run(cmd_args, check=True)
        print(f"✅ Collection completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Collection failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Error running collection: {e}")
        return 1


def run_comparison_collection():
    """Run comparison between heuristic and ML strategies."""
    
    # Model paths
    object_model_path = "/common/home/dm1487/robotics_research/ktamp/learning/outputs/rel_reach_coord_object_dit/mse/2025-08-10_05-33-43"
    goal_model_path = "/common/home/dm1487/robotics_research/ktamp/learning/outputs/rel_reach_coord_goal_dit/mse/2025-08-10_06-59-27"
    
    configurations = [
        {
            "name": "Heuristic IDFS",
            "args": [
                "--algorithm", "idfs",
                "--object-strategy", "nearest_first",
                "--goal-strategy", "random"
            ]
        },
        {
            "name": "ML-Enhanced IDFS",
            "args": [
                "--algorithm", "idfs", 
                "--object-strategy", "ml",
                "--goal-strategy", "ml",
                "--ml-object-model", object_model_path,
                "--ml-goal-model", goal_model_path,
                "--ml-samples", "32",
                "--ml-device", "cuda"
            ]
        }
    ]
    
    # Check model paths for ML configuration
    if not (os.path.exists(object_model_path) and os.path.exists(goal_model_path)):
        print("❌ ML model paths don't exist, running only heuristic configuration")
        configurations = [configurations[0]]  # Only heuristic
    
    for config in configurations:
        print(f"\n=== Running {config['name']} ===")
        
        cmd_args = [
            "python", "python/idfs/modular_parallel_collection.py"
        ] + config["args"] + [
            "--output-dir", f"/tmp/comparison_{config['name'].lower().replace(' ', '_')}",
            "--start-idx", "0",
            "--end-idx", "3", 
            "--workers", "1",
            "--episodes-per-env", "2",
            "--max-depth", "4",
            "--max-goals-per-object", "5",
            "--search-timeout", "60.0"
        ]
        
        print(f"Command: {' '.join(cmd_args)}")
        
        import subprocess
        try:
            result = subprocess.run(cmd_args, check=True)
            print(f"✅ {config['name']} completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ {config['name']} failed with exit code {e.returncode}")
        except Exception as e:
            print(f"❌ Error running {config['name']}: {e}")
    
    return 0


def show_usage_examples():
    """Show usage examples for different scenarios."""
    
    print("ML-Enhanced IDFS Usage Examples")
    print("=" * 50)
    print()
    
    print("1. Basic ML-enhanced IDFS (both object and goal ML):")
    print("   python python/idfs/modular_parallel_collection.py \\")
    print("     --algorithm idfs \\")
    print("     --object-strategy ml \\")
    print("     --goal-strategy ml \\")
    print("     --ml-object-model /path/to/object/model \\")
    print("     --ml-goal-model /path/to/goal/model \\")
    print("     --output-dir /tmp/ml_test \\")
    print("     --start-idx 0 --end-idx 5 \\")
    print("     --workers 1")
    print()
    
    print("2. Mixed strategies (ML object selection, random goals):")
    print("   python python/idfs/modular_parallel_collection.py \\")
    print("     --algorithm idfs \\")
    print("     --object-strategy ml \\")
    print("     --goal-strategy random \\")
    print("     --ml-object-model /path/to/object/model \\")
    print("     --output-dir /tmp/mixed_test \\")
    print("     --start-idx 0 --end-idx 5 \\")
    print("     --workers 1")
    print()
    
    print("3. Heuristic-only strategies (no ML models needed):")
    print("   python python/idfs/modular_parallel_collection.py \\")
    print("     --algorithm idfs \\")
    print("     --object-strategy nearest_first \\")
    print("     --goal-strategy grid \\")
    print("     --output-dir /tmp/heuristic_test \\")
    print("     --start-idx 0 --end-idx 5 \\")
    print("     --workers 1")
    print()
    
    print("4. ML with custom parameters:")
    print("   python python/idfs/modular_parallel_collection.py \\")
    print("     --algorithm idfs \\")
    print("     --object-strategy ml \\")
    print("     --goal-strategy ml \\")
    print("     --ml-object-model /path/to/object/model \\")
    print("     --ml-goal-model /path/to/goal/model \\")
    print("     --ml-samples 64 \\")
    print("     --ml-device cpu \\")
    print("     --output-dir /tmp/ml_custom \\")
    print("     --start-idx 0 --end-idx 3 \\")
    print("     --workers 1")
    print()
    
    print("Available strategies:")
    print("  Object: no_heuristic, nearest_first, goal_proximity, farthest_first, ml")
    print("  Goal: random, grid, adaptive, ml")
    print()
    
    print("Notes:")
    print("- Use --workers 1 for minimal resource usage")
    print("- ML strategies require corresponding model paths")
    print("- ML strategies are independent - no preprocessing coupling")
    print("- If ML fails, strategies will skip objects/goals gracefully")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="ML-Enhanced IDFS Collection Examples")
    parser.add_argument("--mode", choices=["basic", "comparison", "examples"],
                       default="examples", help="Mode to run")
    
    args = parser.parse_args()
    
    if args.mode == "examples":
        show_usage_examples()
        return 0
    elif args.mode == "basic":
        return run_basic_ml_collection()
    elif args.mode == "comparison":
        return run_comparison_collection()
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())