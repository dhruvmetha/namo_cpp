#!/usr/bin/env python3
"""Parallel Sequential Runner

This script runs multiple instances of sequential_ml_collection.py in parallel,
each processing a single environment (start_idx, end_idx) pair.

This combines the benefits of:
- Sequential collection: Efficient ML model reuse within each process
- Parallel execution: Multiple processes running simultaneously

Usage:
    python parallel_sequential_runner.py --start-idx 0 --end-idx 100 --workers 4 [other args]
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def create_index_pairs(start_idx: int, end_idx: int) -> List[Tuple[int, int]]:
    """Create (start_idx, end_idx) pairs with increment of 1."""
    return [(i, i + 1) for i in range(start_idx, end_idx)]

def run_sequential_collection(args: argparse.Namespace, start_idx: int, end_idx: int) -> Tuple[bool, str, float]:
    """Run a single instance of sequential_ml_collection.py."""
    start_time = time.time()
    
    # Build command
    cmd = [
        args.python_path,
        "python/idfs/sequential_ml_collection.py",
        "--start-idx", str(start_idx),
        "--end-idx", str(end_idx),
        "--output-dir", args.output_dir,
        "--algorithm", args.algorithm,
    ]
    
    # Add strategy arguments
    if args.object_strategy:
        cmd.extend(["--object-strategy", args.object_strategy])
    if args.goal_strategy:
        cmd.extend(["--goal-strategy", args.goal_strategy])
    
    # Add ML model paths
    if args.ml_object_model:
        cmd.extend(["--ml-object-model", args.ml_object_model])
    if args.ml_goal_model:
        cmd.extend(["--ml-goal-model", args.ml_goal_model])
    
    # Add ML parameters
    if args.ml_samples:
        cmd.extend(["--ml-samples", str(args.ml_samples)])
    if args.ml_device:
        cmd.extend(["--ml-device", args.ml_device])
    
    # Add other parameters
    if args.episodes_per_env:
        cmd.extend(["--episodes-per-env", str(args.episodes_per_env)])
    if args.max_depth:
        cmd.extend(["--max-depth", str(args.max_depth)])
    if args.max_goals_per_object:
        cmd.extend(["--max-goals-per-object", str(args.max_goals_per_object)])
    if args.max_terminal_checks:
        cmd.extend(["--max-terminal-checks", str(args.max_terminal_checks)])
    if args.search_timeout:
        cmd.extend(["--search-timeout", str(args.search_timeout)])
    if args.xml_dir:
        cmd.extend(["--xml-dir", args.xml_dir])
    if args.config_file:
        cmd.extend(["--config-file", args.config_file])
    
    # Add flags
    if args.verbose:
        cmd.append("--verbose")
    if args.filter_minimum_length:
        cmd.append("--filter-minimum-length")
    
    # Set environment variables
    env = os.environ.copy()
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath
    if args.cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    
    try:
        # Run the subprocess
        result = subprocess.run(
            cmd, 
            env=env,
            capture_output=True, 
            text=True,
            cwd=os.getcwd()
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return True, f"Environment {start_idx} completed successfully", duration
        else:
            error_msg = f"Environment {start_idx} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            return False, error_msg, duration
            
    except Exception as e:
        duration = time.time() - start_time
        return False, f"Environment {start_idx} crashed: {str(e)}", duration

def main():
    """Main entry point for parallel sequential runner."""
    parser = argparse.ArgumentParser(description="Parallel Sequential ML Data Collection Runner")
    
    # Required arguments
    parser.add_argument("--start-idx", type=int, required=True,
                        help="Starting index for environment file subset")
    parser.add_argument("--end-idx", type=int, required=True,
                        help="Ending index for environment file subset (exclusive)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for collected data")
    parser.add_argument("--workers", type=int, required=True,
                        help="Number of parallel worker processes")
    
    # Python execution
    parser.add_argument("--python-path", type=str, 
                        default="/common/users/dm1487/envs/mjxrl/bin/python",
                        help="Path to Python executable")
    parser.add_argument("--pythonpath", type=str,
                        default="/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_westeros",
                        help="PYTHONPATH for the subprocess")
    
    # CUDA device management
    parser.add_argument("--cuda-devices", type=str, nargs='+',
                        help="List of CUDA devices to cycle through (e.g., 0 1 2)")
    parser.add_argument("--cuda-device", type=int,
                        help="Single CUDA device to use for all processes")
    
    # Algorithm selection (forwarded to sequential script)
    parser.add_argument("--algorithm", type=str, default="idfs",
                        help="Planning algorithm to use")
    parser.add_argument("--object-strategy", type=str,
                        help="Object selection strategy")
    parser.add_argument("--goal-strategy", type=str,
                        help="Goal selection strategy")
    
    # ML-specific arguments
    parser.add_argument("--ml-object-model", type=str,
                        help="Path to ML object inference model")
    parser.add_argument("--ml-goal-model", type=str,
                        help="Path to ML goal inference model")
    parser.add_argument("--ml-samples", type=int, default=32,
                        help="Number of ML inference samples")
    parser.add_argument("--ml-device", type=str, default="cuda",
                        help="ML inference device")
    
    # Other parameters (forwarded to sequential script)
    parser.add_argument("--episodes-per-env", type=int, default=1,
                        help="Number of episodes to collect per environment")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum search depth")
    parser.add_argument("--max-goals-per-object", type=int, default=5,
                        help="Maximum goals to sample per object")
    parser.add_argument("--max-terminal-checks", type=int,
                        help="Maximum terminal checks before stopping search")
    parser.add_argument("--search-timeout", type=float,
                        help="Search timeout in seconds")
    parser.add_argument("--xml-dir", type=str,
                        help="Base directory for XML environment files")
    parser.add_argument("--config-file", type=str,
                        help="NAMO configuration file")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose algorithm output")
    parser.add_argument("--filter-minimum-length", action="store_true",
                        help="Only keep episodes with minimum action sequence length")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_idx < 0:
        print("❌ Error: start-idx must be non-negative")
        return 1
    
    if args.end_idx <= args.start_idx:
        print("❌ Error: end-idx must be greater than start-idx")
        return 1
    
    if args.workers <= 0:
        print("❌ Error: workers must be positive")
        return 1
    
    # Create index pairs
    index_pairs = create_index_pairs(args.start_idx, args.end_idx)
    total_environments = len(index_pairs)
    
    print(f"🚀 Starting parallel sequential collection")
    print(f"📊 Algorithm: {args.algorithm}")
    print(f"🔢 Processing {total_environments} environments with {args.workers} parallel processes")
    print(f"🧠 Each process will load ML models once and reuse them")
    if args.cuda_devices:
        print(f"🎮 CUDA devices: {args.cuda_devices} (cycling)")
    elif args.cuda_device is not None:
        print(f"🎮 CUDA device: {args.cuda_device}")
    
    # Assign CUDA devices cyclically if multiple devices specified
    if args.cuda_devices:
        cuda_assignments = {}
        for i, (start, end) in enumerate(index_pairs):
            cuda_assignments[(start, end)] = args.cuda_devices[i % len(args.cuda_devices)]
    
    start_time = time.time()
    completed_tasks = 0
    failed_tasks = []
    successful_tasks = []
    
    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_pair = {}
        for start_idx, end_idx in index_pairs:
            # Assign CUDA device for this task
            task_args = args
            if args.cuda_devices:
                task_args.cuda_device = cuda_assignments[(start_idx, end_idx)]
            elif args.cuda_device is not None:
                task_args.cuda_device = args.cuda_device
            else:
                task_args.cuda_device = None
            
            future = executor.submit(run_sequential_collection, task_args, start_idx, end_idx)
            future_to_pair[future] = (start_idx, end_idx)
        
        # Process completed tasks with progress bar
        with tqdm(total=len(index_pairs), desc="Processing environments", unit="env") as pbar:
            for future in as_completed(future_to_pair):
                start_idx, end_idx = future_to_pair[future]
                completed_tasks += 1
                
                try:
                    success, message, duration = future.result()
                    if success:
                        successful_tasks.append((start_idx, duration))
                        pbar.set_postfix({
                            "success": len(successful_tasks),
                            "failed": len(failed_tasks),
                            "avg_time": f"{sum(d for _, d in successful_tasks)/len(successful_tasks):.1f}s" if successful_tasks else "N/A"
                        })
                    else:
                        failed_tasks.append((start_idx, message))
                        if not args.verbose:
                            print(f"\n❌ Environment {start_idx} failed")
                        else:
                            print(f"\n❌ Environment {start_idx} failed: {message}")
                        pbar.set_postfix({
                            "success": len(successful_tasks),
                            "failed": len(failed_tasks),
                            "avg_time": f"{sum(d for _, d in successful_tasks)/len(successful_tasks):.1f}s" if successful_tasks else "N/A"
                        })
                
                except Exception as e:
                    failed_tasks.append((start_idx, str(e)))
                    print(f"\n💥 Environment {start_idx} crashed: {e}")
                    pbar.set_postfix({
                        "success": len(successful_tasks),
                        "failed": len(failed_tasks),
                        "avg_time": f"{sum(d for _, d in successful_tasks)/len(successful_tasks):.1f}s" if successful_tasks else "N/A"
                    })
                
                pbar.update(1)
    
    # Final summary
    total_time = time.time() - start_time
    success_rate = len(successful_tasks) / total_environments * 100
    avg_time_per_env = sum(d for _, d in successful_tasks) / len(successful_tasks) if successful_tasks else 0
    
    print(f"\n🎉 Parallel sequential collection complete!")
    print(f"📊 Environments: {len(successful_tasks)}/{total_environments} successful")
    print(f"🎯 Success rate: {success_rate:.1f}%")
    print(f"⏱️  Total time: {total_time/60:.1f}m (avg {avg_time_per_env:.1f}s/env)")
    print(f"💾 Output directory: {args.output_dir}")
    
    # Print failed environments if any
    if failed_tasks:
        print(f"\n❌ Failed environments: {len(failed_tasks)}")
        for env_idx, error in failed_tasks[:5]:  # Show first 5 failures
            print(f"   Environment {env_idx}: {error[:100]}...")
        if len(failed_tasks) > 5:
            print(f"   ... and {len(failed_tasks) - 5} more")
    
    return 0 if success_rate == 100.0 else 1


if __name__ == "__main__":
    sys.exit(main())