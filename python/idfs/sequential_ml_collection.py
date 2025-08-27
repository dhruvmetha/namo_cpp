#!/usr/bin/env python3
"""Sequential ML Data Collection Pipeline

This module provides a sequential (single-process) data collection system optimized 
for ML inference. Unlike the parallel version, it loads ML models once and reuses them 
across all episodes, eliminating the overhead of repeated model loading.

Key features:
1. Single-process execution for efficient ML model reuse
2. ML models loaded once at initialization 
3. Persistent planner instance across episodes
4. Compatible interface with parallel collection script
5. Real-time progress tracking

Usage:
    python sequential_ml_collection.py --algorithm idfs --goal-strategy ml --ml-goal-model ./model --output-dir ./data --start-idx 0 --end-idx 10
"""

import os
import sys
import argparse
import socket
import pickle
import time
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict
import glob
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NAMO imports
import namo_rl
from idfs.base_planner import BasePlanner, PlannerConfig, PlannerResult, PlannerFactory
from xml_goal_parser import extract_goal_with_fallback

# Import and register all available planners
from idfs.standard_idfs import StandardIterativeDeepeningDFS
from idfs.tree_idfs import TreeIterativeDeepeningDFS
from idfs.random_sampling import RandomSamplingPlanner

# Import strategies for validation
from idfs.object_selection_strategy import ObjectSelectionStrategy

import random
random.seed(42)


def get_available_object_strategies() -> List[str]:
    """Get list of available object selection strategies."""
    return ["no_heuristic", "nearest_first", "goal_proximity", "farthest_first", "ml"]


def get_available_goal_strategies() -> List[str]:
    """Get list of available goal selection strategies."""
    return ["random", "grid", "adaptive", "ml"]


def validate_object_strategy(strategy_name: str) -> bool:
    """Validate if object selection strategy name is supported."""
    return strategy_name in get_available_object_strategies()


def validate_goal_strategy(strategy_name: str) -> bool:
    """Validate if goal selection strategy name is supported."""
    return strategy_name in get_available_goal_strategies()


@dataclass
class SequentialCollectionConfig:
    """Configuration for sequential ML data collection."""
    
    # Data collection
    xml_base_dir: str = "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9"
    config_file: str = "config/namo_config_complete.yaml"
    output_dir: str = "./sequential_data"
    start_idx: int = 0
    end_idx: int = 100
    episodes_per_env: int = 3
    
    # Algorithm selection
    algorithm: str = "idfs"  # Default algorithm
    planner_config: PlannerConfig = None  # Will use default if None
    
    # Strategy selection (for algorithms that support it)
    object_selection_strategy: str = "no_heuristic"  # Default object strategy
    goal_selection_strategy: str = "random"  # Default goal strategy
    
    # ML-specific parameters (only used when strategies are "ml")
    ml_object_model_path: str = None
    ml_goal_model_path: str = None
    ml_samples: int = 32
    ml_device: str = "cuda"
    
    # Episode filtering options
    filter_minimum_length: bool = False  # Only keep episodes with minimum action sequence length per environment
    
    # Debugging options
    verbose: bool = False
    
    hostname: str = None  # Auto-detected if None


@dataclass
class SequentialEpisodeResult:
    """Result from a single episode using sequential collection."""
    episode_id: str
    algorithm: str
    algorithm_version: str
    success: bool
    solution_found: bool
    solution_depth: Optional[int] = None
    search_time_ms: Optional[float] = None
    nodes_expanded: Optional[int] = None
    terminal_checks: Optional[int] = None
    max_depth_reached: Optional[int] = None
    action_sequence: Optional[List[Dict]] = None
    algorithm_stats: Optional[Dict[str, Any]] = None
    error_message: str = ""
    
    # State information - SE(2) poses before each action
    state_observations: Optional[List[Dict[str, List[float]]]] = None
    
    # State information - SE(2) poses after each action is executed
    post_action_state_observations: Optional[List[Dict[str, List[float]]]] = None
    
    # Static object information (sizes, types) - stored once per environment
    static_object_info: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Meta information
    xml_file: str = ""
    robot_goal: Optional[Tuple[float, float, float]] = None


@dataclass
class SequentialEnvironmentResult:
    """Result from processing a single environment with multiple episodes."""
    env_id: str
    success: bool
    error_message: str = ""
    episodes_collected: int = 0
    processing_time: float = 0.0
    episode_results: List[SequentialEpisodeResult] = None
    # Episode filtering statistics
    episodes_before_filtering: int = 0
    episodes_filtered_out: int = 0
    
    def __post_init__(self):
        if self.episode_results is None:
            self.episode_results = []


def discover_environment_files(base_dir: str, start_idx: int, end_idx: int) -> List[str]:
    """Discover and filter XML environment files by index range."""
    sets = [1, 2]
    benchmarks = [1, 2, 3, 4, 5]
    all_xml_files = []
    for set in sets:
        for benchmark in benchmarks:
            xml_pattern = os.path.join(base_dir, "medium", f"set{set}", f"benchmark_{benchmark}", "*.xml")
            sorted_xml_files = sorted(glob.glob(xml_pattern, recursive=True))
            all_xml_files.extend(sorted_xml_files[1000:1100])
    # Apply subset selection
    if end_idx == -1:
        end_idx = len(all_xml_files)
        
    subset_files = all_xml_files[start_idx:end_idx]
    return subset_files


def generate_hostname_prefix() -> str:
    """Generate hostname-based prefix for output files."""
    hostname = socket.gethostname()
    short_hostname = hostname.split('.')[0]
    return short_hostname


def generate_goal_for_environment(xml_file: str) -> Tuple[float, float, float]:
    """Extract goal position from XML environment file."""
    fallback_goal = (-0.5, 1.3, 0.0)
    return extract_goal_with_fallback(xml_file, fallback_goal)


class SequentialMLCollectionManager:
    """Manager for sequential ML data collection optimized for model reuse."""
    
    def __init__(self, config: SequentialCollectionConfig):
        self.config = config
        
        # Auto-detect hostname if not provided
        if self.config.hostname is None:
            self.config.hostname = generate_hostname_prefix()
        
        # Setup default planner config if not provided
        if self.config.planner_config is None:
            self.config.planner_config = PlannerConfig(
                max_depth=5,
                max_goals_per_object=5,
                collect_stats=True,
                verbose=self.config.verbose,
                algorithm_params={
                    'object_selection_strategy': self.config.object_selection_strategy,
                    'goal_selection_strategy': self.config.goal_selection_strategy,
                    'ml_object_model_path': self.config.ml_object_model_path,
                    'ml_goal_model_path': self.config.ml_goal_model_path,
                    'ml_samples': self.config.ml_samples,
                    'ml_device': self.config.ml_device
                }
            )
        else:
            # Add strategies to existing config if not already present
            if self.config.planner_config.algorithm_params is None:
                self.config.planner_config.algorithm_params = {}
            
            params = self.config.planner_config.algorithm_params
            if 'object_selection_strategy' not in params:
                params['object_selection_strategy'] = self.config.object_selection_strategy
            if 'goal_selection_strategy' not in params:
                params['goal_selection_strategy'] = self.config.goal_selection_strategy
            if 'ml_object_model_path' not in params:
                params['ml_object_model_path'] = self.config.ml_object_model_path
            if 'ml_goal_model_path' not in params:
                params['ml_goal_model_path'] = self.config.ml_goal_model_path
            if 'ml_samples' not in params:
                params['ml_samples'] = self.config.ml_samples
            if 'ml_device' not in params:
                params['ml_device'] = self.config.ml_device
        
        # Setup output directory
        self.output_base = Path(self.config.output_dir)
        self.output_dir = self.output_base / f"sequential_data_{self.config.hostname}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Preload ML models once
        self.preloaded_object_model = None
        self.preloaded_goal_model = None
        self._preload_ml_models()
        
        # Persistent planner instance (will be created on first environment)
        self.planner = None
    
    def _preload_ml_models(self):
        """Preload ML models once for reuse across all episodes."""
        # Preload object model if using ML object strategy
        if (self.config.object_selection_strategy == "ml" and 
            self.config.ml_object_model_path):
            print(f"🔄 Loading object model from {self.config.ml_object_model_path}")
            self.preloaded_object_model = self._load_object_model(
                self.config.ml_object_model_path
            )
            if self.preloaded_object_model:
                print("✅ Object model loaded successfully")
            else:
                print("❌ Failed to load object model")
        
        # Preload goal model if using ML goal strategy  
        if (self.config.goal_selection_strategy == "ml" and 
            self.config.ml_goal_model_path):
            print(f"🔄 Loading goal model from {self.config.ml_goal_model_path}")
            self.preloaded_goal_model = self._load_goal_model(
                self.config.ml_goal_model_path
            )
            if self.preloaded_goal_model:
                print("✅ Goal model loaded successfully")
            else:
                print("❌ Failed to load goal model")
    
    def _load_object_model(self, model_path: str) -> Optional[Any]:
        """Load object inference model."""
        try:
            # Add learning package to path
            learning_path = "/common/home/dm1487/robotics_research/ktamp/learning"
            if learning_path not in sys.path:
                sys.path.append(learning_path)
            
            from ktamp_learning.object_inference_model import ObjectInferenceModel
            
            object_model = ObjectInferenceModel(
                model_path=model_path,
                device=self.config.ml_device
            )
            return object_model
            
        except Exception as e:
            print(f"Failed to load object model: {e}")
            if self.config.verbose:
                traceback.print_exc()
            return None
    
    def _load_goal_model(self, model_path: str) -> Optional[Any]:
        """Load goal inference model."""
        try:
            # Add learning package to path  
            learning_path = "/common/home/dm1487/robotics_research/ktamp/learning"
            if learning_path not in sys.path:
                sys.path.append(learning_path)
            
            from ktamp_learning.goal_inference_model import GoalInferenceModel
            
            goal_model = GoalInferenceModel(
                model_path=model_path,
                device=self.config.ml_device
            )
            return goal_model
            
        except Exception as e:
            print(f"Failed to load goal model: {e}")
            if self.config.verbose:
                traceback.print_exc()
            return None
    
    def _create_planner_config_with_models(self, xml_file: str) -> PlannerConfig:
        """Create planner config with preloaded models injected."""
        import copy
        config_copy = copy.deepcopy(self.config.planner_config)
        
        if config_copy.algorithm_params is None:
            config_copy.algorithm_params = {}
        
        # Inject preloaded models
        if self.preloaded_object_model is not None:
            config_copy.algorithm_params['preloaded_object_model'] = self.preloaded_object_model
        
        if self.preloaded_goal_model is not None:
            config_copy.algorithm_params['preloaded_goal_model'] = self.preloaded_goal_model
        
        # Add XML file path for ML strategies
        xml_path = xml_file
        if xml_path.startswith('../ml4kp_ktamp/resources/models/'):
            # Remove the base part to get relative path from the ML models' base directory
            xml_relative_path = xml_path.replace('../ml4kp_ktamp/resources/models/', '')
            config_copy.algorithm_params['xml_file'] = xml_relative_path
        else:
            # Fallback to filename
            config_copy.algorithm_params['xml_file'] = os.path.basename(xml_path)
        
        return config_copy
    
    def _process_environment(self, xml_file: str, env_index: int) -> SequentialEnvironmentResult:
        """Process a single environment with multiple episodes."""
        start_time = time.time()
        env_id = f"{self.config.hostname}_env_{env_index:06d}"
        result = SequentialEnvironmentResult(env_id=env_id, success=False)
        
        try:
            # Initialize environment
            env = namo_rl.RLEnvironment(xml_file, self.config.config_file, visualize=True)
            episode_results = []
            
            # Collect static object information once per environment (for efficiency)
            try:
                static_object_info = env.get_object_info()
            except AttributeError:
                # Fallback if get_object_info is not available in this build
                static_object_info = {}
            
            # Create or update planner with preloaded models
            if self.planner is None:
                # Create planner with models already injected
                config_with_models = self._create_planner_config_with_models(xml_file)
                self.planner = PlannerFactory.create_planner(
                    self.config.algorithm, env, config_with_models
                )
                if self.config.verbose:
                    print(f"Created {self.config.algorithm} planner with preloaded models")
            else:
                # Update planner's environment for this XML file
                # Most planners store env reference, so we can update it
                self.planner.env = env
            
            # Collect episodes for this environment
            for episode_idx in range(self.config.episodes_per_env):
                # Generate goal for this episode
                robot_goal = generate_goal_for_environment(xml_file)
                episode_id = f"{env_id}_episode_{episode_idx}"
                
                try:
                    # Reset environment and planner for this episode
                    env.reset()
                    self.planner.reset()
                    
                    # Check initial reachability before search
                    env.set_robot_goal(*robot_goal)
                    
                    # Run planning
                    planner_result = self.planner.search(robot_goal)
                    
                    # Create episode result
                    episode_result = SequentialEpisodeResult(
                        episode_id=episode_id,
                        algorithm=self.planner.algorithm_name,
                        algorithm_version=self.planner.algorithm_version,
                        success=planner_result.success,
                        solution_found=planner_result.solution_found,
                        solution_depth=planner_result.solution_depth,
                        search_time_ms=planner_result.search_time_ms,
                        nodes_expanded=planner_result.nodes_expanded,
                        terminal_checks=planner_result.terminal_checks,
                        max_depth_reached=planner_result.max_depth_reached,
                        algorithm_stats=planner_result.algorithm_stats,
                        state_observations=planner_result.state_observations,  # SE(2) poses before each action
                        post_action_state_observations=planner_result.post_action_state_observations,  # SE(2) poses after each action
                        static_object_info=static_object_info if planner_result.solution_found else None,  # Only store when solution found
                        xml_file=xml_file,
                        robot_goal=robot_goal
                    )
                    
                    if planner_result.solution_found and planner_result.action_sequence:
                        episode_result.action_sequence = [
                            {
                                "object_id": action.object_id,
                                "target": (action.x, action.y, action.theta)
                            }
                            for action in planner_result.action_sequence
                        ]
                    
                    if not planner_result.success:
                        episode_result.error_message = planner_result.error_message
                    
                    episode_results.append(episode_result)
                    
                except Exception as e:
                    # Create failed episode result
                    episode_result = SequentialEpisodeResult(
                        episode_id=episode_id,
                        algorithm=self.config.algorithm,
                        algorithm_version="unknown",
                        success=False,
                        solution_found=False,
                        state_observations=None,  # No state observations for failed episodes
                        post_action_state_observations=None,  # No post-action state observations for failed episodes
                        static_object_info=None,  # No static info for failed episodes
                        error_message=str(e),
                        xml_file=xml_file,
                        robot_goal=robot_goal
                    )
                    episode_results.append(episode_result)
                    
                    if self.config.verbose:
                        print(f"Episode {episode_id} failed: {e}")
            
            # Filter episodes by minimum action sequence length if requested
            episodes_before_filtering = len(episode_results)
            episodes_filtered_out = 0
            
            if self.config.filter_minimum_length and episode_results:
                # Find successful episodes with action sequences
                successful_episodes = [ep for ep in episode_results if ep.solution_found and ep.action_sequence]
                
                if successful_episodes:
                    # Find minimum action sequence length among successful episodes
                    min_length = min(len(ep.action_sequence) for ep in successful_episodes)
                    
                    # Keep only episodes with minimum length (including failed episodes for context)
                    filtered_episodes = []
                    for ep in episode_results:
                        if not ep.solution_found:
                            # Keep failed episodes for completeness
                            filtered_episodes.append(ep)
                        elif ep.action_sequence and len(ep.action_sequence) == min_length:
                            # Keep successful episodes with minimum length
                            filtered_episodes.append(ep)
                        # else: filter out successful episodes with longer sequences
                    
                    episodes_filtered_out = len(episode_results) - len(filtered_episodes)
                    episode_results = filtered_episodes
            
            # Save results for this environment
            env_result_data = {
                "env_id": env_id,
                "success": True,
                "episodes_collected": len(episode_results),
                "episodes_before_filtering": episodes_before_filtering,
                "episodes_filtered_out": episodes_filtered_out,
                "processing_time": time.time() - start_time,
                "episode_results": [asdict(ep) for ep in episode_results]
            }
            
            output_file = self.output_dir / f"{env_id}_results.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(env_result_data, f)
            
            # Set result for return
            result.success = True
            result.episodes_collected = len(episode_results)
            result.episodes_before_filtering = episodes_before_filtering
            result.episodes_filtered_out = episodes_filtered_out
            result.processing_time = time.time() - start_time
            result.episode_results = episode_results
            
        except Exception as e:
            result.error_message = f"Environment processing failed: {str(e)}\n{traceback.format_exc()}"
            result.processing_time = time.time() - start_time
            result.episodes_collected = len(episode_results) if 'episode_results' in locals() else 0
        
        return result
    
    def run_sequential_collection(self):
        """Execute sequential data collection with progress tracking."""
        
        # Discover environment files
        xml_files = discover_environment_files(
            self.config.xml_base_dir, 
            self.config.start_idx, 
            self.config.end_idx
        )
        
        if not xml_files:
            print("❌ No environment files found in the specified range")
            return
        
        # Initialize progress tracking
        start_time = time.time()
        completed_envs = 0
        total_episodes = 0
        failed_envs = []
        
        print(f"🚀 Starting sequential ML data collection")
        print(f"📊 Algorithm: {self.config.algorithm}")
        print(f"🔢 Processing {len(xml_files)} environments sequentially")
        if self.preloaded_object_model or self.preloaded_goal_model:
            print("🧠 Using preloaded ML models for efficient inference")
        
        # Process each environment sequentially with progress bar
        all_results = []
        with tqdm(total=len(xml_files), desc="Collecting data", unit="env") as pbar:
            for i, xml_file in enumerate(xml_files):
                env_index = self.config.start_idx + i
                result = self._process_environment(xml_file, env_index)
                all_results.append(result)
                
                completed_envs += 1
                total_episodes += result.episodes_collected
                
                if result.success:
                    pbar.set_postfix({
                        "episodes": total_episodes,
                        "failed": len(failed_envs)
                    })
                else:
                    failed_envs.append(result)
                    print(f"\n❌ Environment {result.env_id} failed: {result.error_message}")
                    print(f"   → But collected {result.episodes_collected} episodes before failing")
                    pbar.set_postfix({
                        "episodes": total_episodes,
                        "failed": len(failed_envs)
                    })
                
                pbar.update(1)
        
        # Final summary
        total_time = time.time() - start_time
        success_rate = (len(xml_files) - len(failed_envs)) / len(xml_files) * 100
        
        print(f"\n🎉 Collection complete!")
        print(f"📊 Episodes: {total_episodes} total")
        print(f"🎯 Environment success rate: {success_rate:.1f}% ({total_time/60:.1f}m)")
        
        self._save_final_summary(xml_files, all_results, total_time)
    
    def _save_final_summary(self, xml_files: List[str], 
                          results: List[SequentialEnvironmentResult], total_time: float):
        """Save comprehensive summary of data collection run."""
        
        # Collect all episode results
        all_episodes = []
        for result in results:
            if result.episode_results:
                all_episodes.extend([asdict(ep) for ep in result.episode_results])
        
        # Calculate statistics
        successful_episodes = [ep for ep in all_episodes if ep['solution_found']]
        search_times = [ep['search_time_ms'] for ep in all_episodes if ep['search_time_ms']]
        nodes_expanded = [ep['nodes_expanded'] for ep in all_episodes if ep['nodes_expanded']]
        
        # Calculate filtering statistics
        total_before_filtering = sum(result.episodes_before_filtering for result in results)
        total_filtered_out = sum(result.episodes_filtered_out for result in results)
        
        summary = {
            'collection_metadata': {
                'hostname': self.config.hostname,
                'algorithm': self.config.algorithm,
                'total_duration_seconds': total_time,
                'execution_mode': 'sequential',
                'ml_models_preloaded': {
                    'object_model': self.preloaded_object_model is not None,
                    'goal_model': self.preloaded_goal_model is not None
                },
                'config': asdict(self.config)
            },
            'performance_stats': {
                'total_episodes': len(all_episodes),
                'successful_episodes': len(successful_episodes),
                'success_rate': len(successful_episodes) / len(all_episodes) * 100 if all_episodes else 0,
                'avg_search_time_ms': sum(search_times) / len(search_times) if search_times else None,
                'avg_nodes_expanded': sum(nodes_expanded) / len(nodes_expanded) if nodes_expanded else None
            },
            'filtering_stats': {
                'episodes_before_filtering': total_before_filtering,
                'episodes_filtered_out': total_filtered_out,
                'filtering_enabled': self.config.filter_minimum_length,
                'filter_rate': (total_filtered_out / total_before_filtering * 100) if total_before_filtering > 0 else 0
            }
        }
        
        # Save summary
        summary_file = self.output_dir / f"collection_summary_{self.config.hostname}.pkl"
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        # Save human-readable summary
        summary_txt = self.output_dir / f"summary_{self.config.hostname}.txt"
        with open(summary_txt, 'w') as f:
            f.write("Sequential ML Data Collection Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Algorithm: {self.config.algorithm}\n")
            f.write(f"Execution mode: Sequential (single-process)\n")
            f.write(f"Total runtime: {total_time/60:.1f} minutes\n")
            f.write(f"Total episodes: {len(all_episodes)}\n\n")
            
            if self.preloaded_object_model or self.preloaded_goal_model:
                f.write("ML Models Preloaded:\n")
                f.write(f"  Object model: {'Yes' if self.preloaded_object_model else 'No'}\n")
                f.write(f"  Goal model: {'Yes' if self.preloaded_goal_model else 'No'}\n\n")
            
            stats = summary['performance_stats']
            f.write(f"Success rate: {stats['success_rate']:.1f}%\n")
            if stats['avg_search_time_ms']:
                f.write(f"Avg search time: {stats['avg_search_time_ms']:.1f}ms\n")
            if stats['avg_nodes_expanded']:
                f.write(f"Avg nodes expanded: {stats['avg_nodes_expanded']:.1f}\n")
            
            # Add filtering statistics
            filter_stats = summary['filtering_stats']
            f.write(f"\nFiltering Statistics:\n")
            f.write(f"Filtering enabled: {filter_stats['filtering_enabled']}\n")
            if filter_stats['filtering_enabled']:
                f.write(f"Episodes before filtering: {filter_stats['episodes_before_filtering']}\n")
                f.write(f"Episodes filtered out: {filter_stats['episodes_filtered_out']}\n")
                f.write(f"Filter rate: {filter_stats['filter_rate']:.1f}%\n")


def main():
    """Main entry point for sequential ML data collection."""
    parser = argparse.ArgumentParser(description="Sequential ML Data Collection")
    
    # Required arguments
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for collected data")
    parser.add_argument("--start-idx", type=int, required=True,
                        help="Starting index for environment file subset")
    parser.add_argument("--end-idx", type=int, required=True,
                        help="Ending index for environment file subset (exclusive)")
    
    # Algorithm selection
    available_algorithms = PlannerFactory.list_available_planners()
    parser.add_argument("--algorithm", type=str, default="idfs", choices=available_algorithms,
                        help=f"Planning algorithm to use. Options: {available_algorithms}")
    
    # Strategy selection (for algorithms that support it)
    available_obj_strategies = get_available_object_strategies()
    parser.add_argument("--object-strategy", type=str, default="no_heuristic", choices=available_obj_strategies,
                        help=f"Object selection strategy. Options: {available_obj_strategies}")
    
    available_goal_strategies = get_available_goal_strategies()
    parser.add_argument("--goal-strategy", type=str, default="random", choices=available_goal_strategies,
                        help=f"Goal selection strategy. Options: {available_goal_strategies}")
    
    # ML-specific arguments (only needed when using ML strategies)
    parser.add_argument("--ml-object-model", type=str,
                        help="Path to ML object inference model (required for ML object strategy)")
    parser.add_argument("--ml-goal-model", type=str,
                        help="Path to ML goal inference model (required for ML goal strategy)")
    parser.add_argument("--ml-samples", type=int, default=5,
                        help="Number of ML inference samples (default: 32)")
    parser.add_argument("--ml-device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="ML inference device (default: cuda)")
    
    # Optional arguments
    parser.add_argument("--episodes-per-env", type=int, default=1,
                        help="Number of episodes to collect per environment")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum search depth")
    parser.add_argument("--max-goals-per-object", type=int, default=5,
                        help="Maximum goals to sample per object")
    parser.add_argument("--max-terminal-checks", type=int, default=5000,
                        help="Maximum terminal checks before stopping search (default: 5000)")
    parser.add_argument("--search-timeout", type=float, default=300.0,
                        help="Search timeout in seconds (default: 300.0 = 5 minutes)")
    parser.add_argument("--xml-dir", type=str, 
                        default="../ml4kp_ktamp/resources/models/custom_walled_envs/aug9",
                        help="Base directory for XML environment files")
    parser.add_argument("--config-file", type=str, 
                        default="config/namo_config_complete.yaml",
                        help="NAMO configuration file")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose algorithm output")
    parser.add_argument("--filter-minimum-length", action="store_true",
                        help="Only keep episodes with minimum action sequence length per environment")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_idx < 0:
        print("❌ Error: start-idx must be non-negative")
        return 1
    
    if args.end_idx <= args.start_idx:
        print("❌ Error: end-idx must be greater than start-idx")
        return 1
    
    # Validate ML strategy requirements
    if args.object_strategy == "ml" and not args.ml_object_model:
        print("❌ Error: --ml-object-model is required when using ML object strategy")
        return 1
    
    if args.goal_strategy == "ml" and not args.ml_goal_model:
        print("❌ Error: --ml-goal-model is required when using ML goal strategy")
        return 1
    
    # Validate strategy names
    if not validate_object_strategy(args.object_strategy):
        print(f"❌ Error: invalid object strategy '{args.object_strategy}'")
        return 1
    
    if not validate_goal_strategy(args.goal_strategy):
        print(f"❌ Error: invalid goal strategy '{args.goal_strategy}'")
        return 1
    
    # Create planner configuration (strategy will be added by manager)
    planner_config = PlannerConfig(
        max_depth=args.max_depth,
        max_goals_per_object=args.max_goals_per_object,
        max_terminal_checks=args.max_terminal_checks,
        max_search_time_seconds=args.search_timeout,
        verbose=args.verbose,
        collect_stats=True
    )
    
    # Create configuration
    config = SequentialCollectionConfig(
        xml_base_dir=args.xml_dir,
        config_file=args.config_file,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        episodes_per_env=args.episodes_per_env,
        algorithm=args.algorithm,
        object_selection_strategy=args.object_strategy,
        goal_selection_strategy=args.goal_strategy,
        ml_object_model_path=args.ml_object_model,
        ml_goal_model_path=args.ml_goal_model,
        ml_samples=args.ml_samples,
        ml_device=args.ml_device,
        filter_minimum_length=args.filter_minimum_length,
        verbose=args.verbose,
        planner_config=planner_config
    )
    
    # Execute sequential data collection
    try:
        manager = SequentialMLCollectionManager(config)
        manager.run_sequential_collection()
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