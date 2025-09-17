#!/usr/bin/env python3
"""
Parallel MCTS Data Collection with Multi-Host Support

This module enables massively parallel data collection across multiple hosts by:
1. Distributing environment subsets via start/end indices
2. Using hostname-based naming for cross-host coordination
3. Multiprocessing with process pools for CPU utilization
4. Robust error handling and progress tracking
5. Memory-efficient worker processes with independent MuJoCo environments

Architecture:
- Manager Process: XML file discovery, work distribution, progress tracking
- Worker Processes: Independent MCTS data collection per environment
- File System: Hostname-based output organization for multi-host coordination

Usage:
    python parallel_data_collection.py --output-dir ./data --start-idx 0 --end-idx 100 --workers 8 \
        --episodes-per-env 5 --mcts-budget 100 --random-seed 42
"""

import os
import sys
import argparse
import socket
import pickle
import time
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from multiprocessing import Pool, Queue, Manager, Value, Lock
from queue import Empty
import glob
from tqdm import tqdm

# NAMO imports
import namo_rl
from namo.planners.mcts.hierarchical_mcts import CleanHierarchicalMCTS
from namo.config.mcts_config import MCTSConfig
from namo.data_collection.alphazero_data_collection import MCTSDataExtractor, SingleEnvironmentDataCollector, EpisodeData
from namo.core.xml_goal_parser import extract_goal_with_fallback

import random

# Configuration-driven design with no hardcoded values

@dataclass
class MCTSHyperparameters:
    """MCTS algorithm hyperparameters."""
    simulation_budget: int = 100
    max_rollout_steps: int = 5
    k: float = 2.0  # UCB exploration parameter
    alpha: float = 0.5  # Progressive widening parameter
    c_exploration: float = 1.414  # UCB exploration constant
    top_k_goals: int = 3  # Number of goal proposals to extract per object


@dataclass
class ParallelConfig:
    """Configuration for parallel data collection."""
    # Environment and I/O
    xml_base_dir: str = "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9"
    config_file: str = "config/namo_config_complete.yaml"
    output_dir: str = "./parallel_data"
    hostname: str = None  # Auto-detected if None
    
    # Data collection parameters
    start_idx: int = 0
    end_idx: int = 100
    episodes_per_env: int = 3
    max_steps_per_episode: int = 10
    
    # Parallel processing
    num_workers: int = 8
    
    # MCTS parameters
    mcts: MCTSHyperparameters = None
    
    # Control parameters
    random_seed: int = 42
    early_termination_threshold: int = 2  # Stop after N consecutive zero-sample episodes
    fallback_goal: tuple = (-0.5, 1.3, 0.0)  # Default robot goal if XML parsing fails
    
    def __post_init__(self):
        if self.mcts is None:
            self.mcts = MCTSHyperparameters()
    

@dataclass 
class WorkerTask:
    """Task specification for worker process."""
    task_id: str
    xml_file: str
    config_file: str
    output_dir: str
    episodes_per_env: int
    max_steps: int
    mcts_config: MCTSConfig
    data_extractor_config: dict


@dataclass
class WorkerResult:
    """Result from worker process."""
    task_id: str
    success: bool
    error_message: str = ""
    episodes_collected: int = 0
    total_training_samples: int = 0
    processing_time: float = 0.0
    zero_sample_episodes: int = 0
    skipped_environment: bool = False


def discover_environment_files(base_dir: str, start_idx: int, end_idx: int, 
                              shuffle_files: bool = True) -> List[str]:
    """Discover and filter XML environment files by index range.
    
    Args:
        base_dir: Directory to search for XML files
        start_idx: Starting index for subset selection  
        end_idx: Ending index for subset selection (-1 for all remaining)
        shuffle_files: Whether to shuffle files before subset selection
    
    Returns:
        List of XML file paths in the specified range
    """
    xml_pattern = os.path.join(base_dir, "**", "*.xml")
    all_xml_files = sorted(glob.glob(xml_pattern, recursive=True))
    
    if shuffle_files:
        random.shuffle(all_xml_files)
    
    # Apply subset selection
    if end_idx == -1:
        end_idx = len(all_xml_files)
    
    subset_files = all_xml_files[start_idx:end_idx]
    
    return subset_files


def generate_hostname_prefix() -> str:
    """Generate hostname-based prefix for output files."""
    hostname = socket.gethostname()
    # Extract short hostname (e.g., "westeros" from "westeros.cs.rutgers.edu")
    short_hostname = hostname.split('.')[0]
    return short_hostname


def generate_goal_for_environment(xml_file: str, fallback_goal: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Extract actual goal position from XML environment file."""
    return extract_goal_with_fallback(xml_file, fallback_goal)



def worker_process(task: WorkerTask) -> WorkerResult:
    """Worker process function for parallel MCTS data collection."""
    start_time = time.time()
    result = WorkerResult(task_id=task.task_id, success=False)
    
    try:
        
        # Initialize data extractor
        data_extractor = MCTSDataExtractor(**task.data_extractor_config)
        
        # Initialize single environment collector
        collector = SingleEnvironmentDataCollector(
            xml_file=task.xml_file,
            config_file=task.config_file,
            data_extractor=data_extractor,
            output_dir=task.output_dir
        )
        
        total_samples = 0
        episodes_collected = 0
        zero_sample_episodes = 0
        
        # Collect multiple episodes for this environment
        for episode_idx in range(task.episodes_per_env):
            try:
                # Generate goal for this episode
                fallback_goal = task.config.fallback_goal if hasattr(task, 'config') else (-0.5, 1.3, 0.0)
                robot_goal = generate_goal_for_environment(task.xml_file, fallback_goal)
                episode_id = f"{task.task_id}_episode_{episode_idx}"
                
                # Collect episode data with pickle-only saving
                episode_data = collector.collect_episode_data(
                    robot_goal=robot_goal,
                    mcts_config=task.mcts_config,
                    max_steps=task.max_steps,
                    episode_id=episode_id,
                    save_json_summary=False  # Skip JSON to save I/O
                )
                
                sample_count = len(episode_data.step_data)
                
                if sample_count > 0:
                    episodes_collected += 1
                    total_samples += sample_count
                else:
                    zero_sample_episodes += 1
                
                # Early termination if all episodes produce 0 samples
                early_threshold = task.config.early_termination_threshold if hasattr(task, 'config') else 2
                if zero_sample_episodes >= early_threshold and total_samples == 0:
                    break
                    
            except Exception as e:
                # Continue with other episodes
                continue
        
        result.success = True
        result.episodes_collected = episodes_collected
        result.total_training_samples = total_samples
        result.processing_time = time.time() - start_time
        result.zero_sample_episodes = zero_sample_episodes
        result.skipped_environment = (total_samples == 0 and zero_sample_episodes >= 2)
        
        
    except Exception as e:
        result.error_message = f"Worker failed: {str(e)}\n{traceback.format_exc()}"
        result.processing_time = time.time() - start_time
    
    return result


class ParallelDataCollectionManager:
    """Manager for parallel MCTS data collection across multiple processes.
    
    Coordinates distributed data collection by:
    - Discovering and partitioning XML environment files
    - Creating worker tasks with proper configuration
    - Managing multiprocessing execution with progress tracking
    - Collecting results and generating comprehensive summaries
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        
        # Auto-detect hostname if not provided
        if self.config.hostname is None:
            self.config.hostname = generate_hostname_prefix()
        
        # Setup output directory with hostname
        self.output_base = Path(self.config.output_dir)
        self.output_dir = self.output_base / f"data_{self.config.hostname}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup progress tracking
        self.progress_file = self.output_dir / "collection_progress.txt"
        
    
    def create_tasks(self) -> List[WorkerTask]:
        """Create worker tasks from environment file subset.
        
        Returns:
            List of WorkerTask objects ready for parallel execution
        """
        # Discover environment files
        xml_files = discover_environment_files(
            self.config.xml_base_dir, 
            self.config.start_idx, 
            self.config.end_idx
        )
        
        # Create MCTS configuration from hyperparameters
        mcts_config = MCTSConfig(
            simulation_budget=self.config.mcts.simulation_budget,
            max_rollout_steps=self.config.mcts.max_rollout_steps,
            k=self.config.mcts.k,
            alpha=self.config.mcts.alpha,
            c_exploration=self.config.mcts.c_exploration
        )
        
        # Create data extractor configuration
        data_extractor_config = {"top_k_goals": self.config.mcts.top_k_goals}
        
        # Create tasks
        tasks = []
        for i, xml_file in enumerate(xml_files):
            task_id = f"{self.config.hostname}_env_{self.config.start_idx + i:06d}"
            
            task = WorkerTask(
                task_id=task_id,
                xml_file=xml_file,
                config_file=self.config.config_file,
                output_dir=str(self.output_dir),
                episodes_per_env=self.config.episodes_per_env,
                max_steps=self.config.max_steps_per_episode,
                mcts_config=mcts_config,
                data_extractor_config=data_extractor_config
            )
            # Add config reference for worker process
            task.config = self.config
            tasks.append(task)
        
        return tasks
    
    def run_parallel_collection(self):
        """Execute parallel data collection with progress tracking."""
        
        # Create tasks
        tasks = self.create_tasks()
        if not tasks:
            return
        
        # Initialize progress tracking
        start_time = time.time()
        completed_tasks = 0
        total_episodes = 0
        total_samples = 0
        failed_tasks = []
        
        # Execute tasks in parallel with progress bar
        with Pool(processes=self.config.num_workers) as pool:
            # Submit all tasks
            results = []
            with tqdm(total=len(tasks), desc="Collecting data", unit="env") as pbar:
                for result in pool.imap_unordered(worker_process, tasks):
                    completed_tasks += 1
                    results.append(result)
                    
                    if result.success:
                        total_episodes += result.episodes_collected
                        total_samples += result.total_training_samples
                        pbar.set_postfix({
                            "episodes": total_episodes,
                            "samples": total_samples,
                            "failed": len(failed_tasks)
                        })
                    else:
                        failed_tasks.append(result)
                        pbar.set_postfix({
                            "episodes": total_episodes,
                            "samples": total_samples,
                            "failed": len(failed_tasks)
                        })
                    
                    pbar.update(1)
                    
                    # Update progress file
                    self._update_progress(completed_tasks, len(tasks), total_episodes, total_samples, failed_tasks)
        
        # Calculate additional statistics
        skipped_environments = sum(1 for r in results if r.skipped_environment)
        zero_sample_episodes = sum(r.zero_sample_episodes for r in results)
        
        # Final summary
        total_time = time.time() - start_time
        success_rate = (len(tasks) - len(failed_tasks)) / len(tasks) * 100
        
        # Final summary
        print(f"\nCollection complete: {total_episodes} episodes, {total_samples} samples")
        print(f"Skipped: {skipped_environments} environments, {zero_sample_episodes} zero-sample episodes")
        print(f"Success rate: {success_rate:.1f}% ({total_time/60:.1f}m)")
        
        self._save_final_summary(tasks, results, total_time)
    
    def _update_progress(self, completed: int, total: int, episodes: int, samples: int, failed: List):
        """Update progress tracking file."""
        progress_info = {
            'timestamp': time.time(),
            'hostname': self.config.hostname,
            'completed_tasks': completed,
            'total_tasks': total,
            'completion_percentage': completed / total * 100,
            'total_episodes': episodes,
            'total_samples': samples,
            'failed_tasks': len(failed),
            'config': asdict(self.config)
        }
        
        with open(self.progress_file, 'w') as f:
            f.write(f"Progress Update - {time.ctime()}\n")
            f.write(f"Host: {progress_info['hostname']}\n")
            f.write(f"Completed: {completed}/{total} ({progress_info['completion_percentage']:.1f}%)\n")
            f.write(f"Episodes: {episodes}, Samples: {samples}, Failed: {len(failed)}\n")
    
    def _save_final_summary(self, tasks: List[WorkerTask], results: List[WorkerResult], total_time: float):
        """Save comprehensive summary of data collection run."""
        summary = {
            'collection_metadata': {
                'hostname': self.config.hostname,
                'start_time': time.time() - total_time,
                'end_time': time.time(),
                'total_duration_seconds': total_time,
                'config': asdict(self.config)
            },
            'task_summary': {
                'total_tasks': len(tasks),
                'successful_tasks': sum(1 for r in results if r.success),
                'failed_tasks': sum(1 for r in results if not r.success),
                'success_rate': sum(1 for r in results if r.success) / len(results) * 100 if results else 0
            },
            'data_summary': {
                'total_episodes': sum(r.episodes_collected for r in results),
                'total_training_samples': sum(r.total_training_samples for r in results),
                'zero_sample_episodes': sum(r.zero_sample_episodes for r in results),
                'skipped_environments': sum(1 for r in results if r.skipped_environment),
                'average_samples_per_episode': None,
                'average_processing_time': sum(r.processing_time for r in results) / len(results) if results else 0
            },
            'failed_tasks': [
                {
                    'task_id': r.task_id,
                    'error_message': r.error_message,
                    'processing_time': r.processing_time
                }
                for r in results if not r.success
            ]
        }
        
        total_episodes = summary['data_summary']['total_episodes']
        total_samples = summary['data_summary']['total_training_samples']
        if total_episodes > 0:
            summary['data_summary']['average_samples_per_episode'] = total_samples / total_episodes
        
        # Save summary
        summary_file = self.output_dir / f"collection_summary_{self.config.hostname}.pkl"
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        


def main():
    """Main entry point for parallel data collection."""
    parser = argparse.ArgumentParser(description="Parallel MCTS Data Collection")
    
    # Required arguments
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for collected data")
    parser.add_argument("--start-idx", type=int, required=True,
                        help="Starting index for environment file subset")
    parser.add_argument("--end-idx", type=int, required=True,
                        help="Ending index for environment file subset (exclusive)")
    
    # Optional arguments
    parser.add_argument("--workers", type=int, default=12,
                        help="Number of parallel worker processes")
    parser.add_argument("--episodes-per-env", type=int, default=1,
                        help="Number of episodes to collect per environment")
    parser.add_argument("--mcts-budget", type=int, default=100,
                        help="MCTS simulation budget per search")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum steps per episode")
    parser.add_argument("--mcts-rollout-steps", type=int, default=5,
                        help="Maximum steps per MCTS rollout simulation")
    parser.add_argument("--mcts-k", type=float, default=2.0,
                        help="UCB exploration parameter for MCTS")
    parser.add_argument("--mcts-alpha", type=float, default=0.5,
                        help="Progressive widening parameter for MCTS")
    parser.add_argument("--mcts-c-exploration", type=float, default=1.414,
                        help="UCB exploration constant for MCTS")
    parser.add_argument("--top-k-goals", type=int, default=3,
                        help="Number of goal proposals to extract per object")
    parser.add_argument("--early-termination-threshold", type=int, default=2,
                        help="Stop after N consecutive zero-sample episodes")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--xml-dir", type=str, 
                        default="../ml4kp_ktamp/resources/models/custom_walled_envs/aug9",
                        help="Base directory for XML environment files")
    parser.add_argument("--config-file", type=str, 
                        default="config/namo_config_complete.yaml",
                        help="NAMO configuration file")
    parser.add_argument("--hostname", type=str, default=None,
                        help="Override hostname for output naming (auto-detected if not provided)")
    
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
    
    # Create MCTS hyperparameters
    mcts_params = MCTSHyperparameters(
        simulation_budget=args.mcts_budget,
        max_rollout_steps=args.mcts_rollout_steps,
        k=args.mcts_k,
        alpha=args.mcts_alpha,
        c_exploration=args.mcts_c_exploration,
        top_k_goals=args.top_k_goals
    )
    
    # Create configuration
    config = ParallelConfig(
        xml_base_dir=args.xml_dir,
        config_file=args.config_file,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        episodes_per_env=args.episodes_per_env,
        num_workers=args.workers,
        max_steps_per_episode=args.max_steps,
        mcts=mcts_params,
        random_seed=args.random_seed,
        early_termination_threshold=args.early_termination_threshold,
        hostname=args.hostname
    )
    
    # Set global random seed for reproducibility
    random.seed(config.random_seed)
    
    # Execute parallel data collection
    try:
        manager = ParallelDataCollectionManager(config)
        manager.run_parallel_collection()
        return 0
    
    except KeyboardInterrupt:
        print("\n🛑 Data collection interrupted by user")
        return 1
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())