#!/usr/bin/env python3
"""Modular Parallel Data Collection Pipeline

This module provides a refactored parallel data collection system that supports
pluggable planning algorithms. Users can easily swap between IDFS, Tree-IDFS, 
MCTS, or other algorithms without changing the collection infrastructure.

Key features:
1. Algorithm-agnostic data collection
2. Pluggable planner interface
3. Consistent performance metrics across algorithms
4. Algorithm comparison capabilities
5. Backward compatibility with existing configs

Usage:
    python modular_parallel_collection.py --algorithm tree_idfs --output-dir ./data --start-idx 0 --end-idx 10
"""

import os
import sys
import argparse
import socket
import pickle
import time
import traceback
import signal
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict
from multiprocessing import Pool
import glob
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NAMO imports
import namo_rl
from namo.core import BasePlanner, PlannerConfig, PlannerResult, PlannerFactory
from namo.core.xml_goal_parser import extract_goal_with_fallback

# Import and register all available planners
from namo.planners.idfs.standard_idfs import StandardIterativeDeepeningDFS
from namo.planners.idfs.tree_idfs import TreeIterativeDeepeningDFS
from namo.planners.sampling.random_sampling import RandomSamplingPlanner
from namo.planners.idfs.optimal_idfs import OptimalIterativeDeepeningDFS

# Import strategies for validation
from namo.strategies.object_selection_strategy import ObjectSelectionStrategy

# Import failure classification system
from namo.planners.idfs.failure_codes import FailureCode, FailureClassifier, create_failure_info, get_failure_statistics

# Import solution smoothing system
from namo.planners.idfs.solution_smoother import SolutionSmoother

import random
random.seed(42)

def create_goal_checker(robot_goal):
    """Create a goal checker function for the smoother."""
    def check_goal(env):
        current_state = env.get_observation()
        robot_pos = current_state.get("robot", [0.0, 0.0, 0.0])
        
        # Simple distance-based goal check
        dx = robot_pos[0] - robot_goal[0]
        dy = robot_pos[1] - robot_goal[1]
        distance = (dx*dx + dy*dy)**0.5
        
        return distance < 0.1  # 10cm tolerance
    return check_goal


def get_available_object_strategies() -> List[str]:
    """Get list of available object selection strategies."""
    return ["no_heuristic", "nearest_first", "goal_proximity", "farthest_first", "ml"]


def get_available_goal_strategies() -> List[str]:
    """Get list of available goal selection strategies."""
    return ["random", "grid", "adaptive", "discretized", "ml"]


def validate_object_strategy(strategy_name: str) -> bool:
    """Validate if object selection strategy name is supported."""
    return strategy_name in get_available_object_strategies()


def validate_goal_strategy(strategy_name: str) -> bool:
    """Validate if goal selection strategy name is supported."""
    return strategy_name in get_available_goal_strategies()


@dataclass
class ModularCollectionConfig:
    """Configuration for modular parallel data collection."""
    
    # Data collection
    xml_base_dir: str = "../ml4kp_ktamp/resources/models/custom_walled_envs/aug9"
    config_file: str = "config/namo_config_complete.yaml"
    output_dir: str = "./modular_data"
    start_idx: int = 0
    end_idx: int = 100
    episodes_per_env: int = 3
    num_workers: int = 8
    
    # Algorithm selection
    algorithm: str = "idfs"  # Default algorithm
    planner_config: PlannerConfig = None  # Will use default if None
    
    # Strategy selection (for algorithms that support it)
    object_selection_strategy: str = "no_heuristic"  # Default object strategy
    goal_selection_strategy: str = "random"  # Default goal strategy
    
    # ML-specific parameters (only used when strategies are "ml")
    ml_object_model_path: str = None
    
    # Solution smoothing
    smooth_solutions: bool = False
    max_smooth_actions: int = 20
    ml_goal_model_path: str = None
    ml_samples: int = 32
    ml_device: str = "cuda"
    epsilon: float = None  # For epsilon-greedy goal strategy
    
    # Episode filtering options
    filter_minimum_length: bool = False  # Only keep episodes with minimum action sequence length per environment
    
    
    hostname: str = None  # Auto-detected if None


@dataclass
class ModularWorkerTask:
    """Task specification for modular worker process."""
    task_id: str
    xml_file: str
    config_file: str
    output_dir: str
    episodes_per_env: int
    algorithm: str
    planner_config: PlannerConfig
    # Model paths for worker-side preloading (models can't be pickled)
    ml_object_model_path: Optional[str] = None
    ml_goal_model_path: Optional[str] = None
    # Filtering options
    filter_minimum_length: bool = False
    # Solution smoothing options
    smooth_solutions: bool = False
    max_smooth_actions: int = 20


@dataclass
class ModularEpisodeResult:
    """Result from a single episode using modular planner interface."""
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
    
    # Failure classification
    failure_code: Optional[int] = None
    failure_description: str = ""
    
    # State information - SE(2) poses before each action
    
    # Solution smoothing results
    original_action_sequence: Optional[List[Dict]] = None  # Original solution before smoothing
    smoothing_stats: Optional[Dict[str, Any]] = None  # Smoothing statistics
    state_observations: Optional[List[Dict[str, List[float]]]] = None
    
    # State information - SE(2) poses after each action is executed
    post_action_state_observations: Optional[List[Dict[str, List[float]]]] = None
    
    # Static object information (sizes, types) - stored once per environment
    static_object_info: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Meta information
    xml_file: str = ""
    robot_goal: Optional[Tuple[float, float, float]] = None


@dataclass
class ModularWorkerResult:
    """Result from modular worker process."""
    task_id: str
    success: bool
    error_message: str = ""
    episodes_collected: int = 0
    processing_time: float = 0.0
    episode_results: List[ModularEpisodeResult] = None
    # Episode filtering statistics
    episodes_before_filtering: int = 0
    episodes_filtered_out: int = 0
    
    def __post_init__(self):
        if self.episode_results is None:
            self.episode_results = []


# Planners are registered automatically when imported


def discover_environment_files(base_dir: str, start_idx: int, end_idx: int) -> List[str]:
    """Discover and filter XML environment files by index range."""
   
   
    # all_xml_files = []
    # with open('./notebooks/unsolved_envs.pkl', 'rb') as f:
    #     unsolved_envs = pickle.load(f)
    # for env_name in unsolved_envs:
    #     xml_file = os.path.join(base_dir, env_name)
    #     all_xml_files.append(xml_file)
    # all_xml_files = sorted(all_xml_files)
    

    folder = "train_envs"
    all_xml_files = []
    for d in ['very_hard']:
        with open(f'{folder}/envs_names_{d}.pkl', 'rb') as f:
            envs_names = pickle.load(f)
        for env_name in envs_names:
            xml_file = os.path.join(base_dir, env_name)
            all_xml_files.append(xml_file)
    all_xml_files = sorted(all_xml_files)
    
    # sets = [1, 2]
    # benchmarks = [1, 2, 3, 4, 5]
    # all_xml_files = []
    # for set in sets:
    #     for benchmark in benchmarks:
    #         xml_pattern = os.path.join(base_dir, "medium", f"set{set}", f"benchmark_{benchmark}", "*.xml")
    #         sorted_xml_files = sorted(glob.glob(xml_pattern, recursive=True))
    #         all_xml_files.extend(sorted_xml_files[:1000]) # train
    #         # all_xml_files.extend(sorted_xml_files[1000:1100]) # test
    # # Apply subset selection
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


def _worker_preload_object_model(model_path: str, config: PlannerConfig) -> Optional[Any]:
    """Preload object model within worker process."""
    try:
        # Add learning package to path
        learning_path = "/common/home/dm1487/robotics_research/ktamp/learning"
        if learning_path not in sys.path:
            sys.path.append(learning_path)
        
        from ktamp_learning.object_inference_model import ObjectInferenceModel
        
        # Get device from config
        device = "cuda"
        if config.algorithm_params and 'ml_device' in config.algorithm_params:
            device = config.algorithm_params['ml_device']
        
        print(f"[Worker] Loading ObjectInferenceModel from {model_path}")
        object_model = ObjectInferenceModel(
            model_path=model_path,
            device=device
        )
        print(f"[Worker] ✅ Object model loaded successfully")
        return object_model
        
    except Exception as e:
        print(f"[Worker] ❌ Failed to load object model: {e}")
        return None


def _worker_preload_goal_model(model_path: str, config: PlannerConfig) -> Optional[Any]:
    """Preload goal model within worker process."""
    try:
        # Add learning package to path  
        learning_path = "/common/home/dm1487/robotics_research/ktamp/learning"
        if learning_path not in sys.path:
            sys.path.append(learning_path)
        
        from ktamp_learning.goal_inference_model import GoalInferenceModel
        
        # Get device from config
        device = "cuda"
        if config.algorithm_params and 'ml_device' in config.algorithm_params:
            device = config.algorithm_params['ml_device']
        
        print(f"[Worker] Loading GoalInferenceModel from {model_path}")
        goal_model = GoalInferenceModel(
            model_path=model_path,
            device=device
        )
        print(f"[Worker] ✅ Goal model loaded successfully")
        return goal_model
        
    except Exception as e:
        print(f"[Worker] ❌ Failed to load goal model: {e}")
        return None


def _inject_preloaded_models(planner_config: PlannerConfig, 
                           preloaded_object_model: Optional[Any],
                           preloaded_goal_model: Optional[Any]) -> PlannerConfig:
    """Create a copy of planner config with preloaded ML models injected."""
    import copy
    config_copy = copy.deepcopy(planner_config)
    
    if config_copy.algorithm_params is None:
        config_copy.algorithm_params = {}
    
    # Inject preloaded models into parameters
    if preloaded_object_model is not None:
        config_copy.algorithm_params['preloaded_object_model'] = preloaded_object_model
    
    if preloaded_goal_model is not None:
        config_copy.algorithm_params['preloaded_goal_model'] = preloaded_goal_model
    
    return config_copy


def modular_worker_process(task: ModularWorkerTask) -> ModularWorkerResult:
    """Worker process function for modular parallel data collection."""
    start_time = time.time()
    result = ModularWorkerResult(task_id=task.task_id, success=False)
    
    try:
        # Initialize environment
        env = namo_rl.RLEnvironment(task.xml_file, task.config_file, visualize=False)
        episode_results = []
        
        # Collect static object information once per environment (for efficiency)
        try:
            static_object_info = env.get_object_info()
        except AttributeError:
            # Fallback for environments without get_object_info method
            static_object_info = {}
        
        # Preload ML models within worker process to avoid repeated loading
        planner = None
        preloaded_object_model = None
        preloaded_goal_model = None
        
        # Preload models if using ML strategies
        if task.ml_object_model_path:
            preloaded_object_model = _worker_preload_object_model(task.ml_object_model_path, task.planner_config)
        if task.ml_goal_model_path:
            preloaded_goal_model = _worker_preload_goal_model(task.ml_goal_model_path, task.planner_config)
        
        # Inject preloaded models into config
        if preloaded_object_model is not None or preloaded_goal_model is not None:
            config_with_models = _inject_preloaded_models(task.planner_config, 
                                                         preloaded_object_model, 
                                                         preloaded_goal_model)
        else:
            config_with_models = task.planner_config
        
        # Collect episodes for this environment
        for episode_idx in range(task.episodes_per_env):
            # Generate goal for this episode
            robot_goal = generate_goal_for_environment(task.xml_file)
            episode_id = f"{task.task_id}_episode_{episode_idx}"
            
            try:
                # Reset environment
                env.reset()
                
                # Create planner only once per worker (not per episode)
                if planner is None:
                    # Add XML file path to planner config for ML strategies
                    if config_with_models.algorithm_params is None:
                        config_with_models.algorithm_params = {}
                    # ML models seem to have base path "/common/home/dm1487/robotics_research/ktamp/ml4kp_ktamp/resources/models/"
                    # Extract the relative path from that base directory
                    import os
                    xml_path = task.xml_file
                    if xml_path.startswith('../ml4kp_ktamp/resources/models/'):
                        # Remove the base part to get relative path from the ML models' base directory
                        xml_relative_path = xml_path.replace('../ml4kp_ktamp/resources/models/', '')
                        config_with_models.algorithm_params['xml_file'] = xml_relative_path
                    else:
                        # Fallback to filename
                        config_with_models.algorithm_params['xml_file'] = os.path.basename(xml_path)
                    
                    # Create planner with models already injected
                    planner = PlannerFactory.create_planner(task.algorithm, env, config_with_models)
                
                # Reset planner for this episode (but don't recreate it)
                planner.reset()
                
                
                # Check initial reachability before search
                env.set_robot_goal(*robot_goal)
                
                # Run planning
                planner_result = planner.search(robot_goal)
                
                # Special handling for optimal planner: save all minimum solutions as separate episodes
                # This provides more training data while maintaining backward compatibility
                is_optimal_planner = hasattr(planner, 'get_all_minimum_solutions')
                
                if (is_optimal_planner and planner_result.solution_found and 
                    planner_result.algorithm_stats and 
                    planner_result.algorithm_stats.get('num_minimum_solutions', 0) > 1):
                    
                    # Get all minimum solutions for optimal planner
                    all_solutions = planner.get_all_minimum_solutions()
                    
                    # Create a separate episode for each minimum solution
                    for solution_idx, (actions, states, post_states) in enumerate(all_solutions):
                        solution_episode_id = f"{episode_id}_solution_{solution_idx}"
                        
                        # Create episode result for this solution
                        episode_result = ModularEpisodeResult(
                            episode_id=solution_episode_id,
                            algorithm=planner.algorithm_name,
                            algorithm_version=planner.algorithm_version,
                            success=planner_result.success,
                            solution_found=True,  # This solution exists
                            solution_depth=len(actions),  # Depth of this specific solution
                            search_time_ms=planner_result.search_time_ms,
                            nodes_expanded=planner_result.nodes_expanded,
                            terminal_checks=planner_result.terminal_checks,
                            max_depth_reached=planner_result.max_depth_reached,
                            algorithm_stats={
                                **planner_result.algorithm_stats,
                                'solution_index': solution_idx,  # Track which solution this is
                                'total_minimum_solutions': len(all_solutions)
                            },
                            state_observations=states,  # This solution's states
                            post_action_state_observations=post_states,  # This solution's post-action states
                            static_object_info=static_object_info,
                            xml_file=task.xml_file,
                            robot_goal=robot_goal,
                            failure_code=None,
                            failure_description=""
                        )
                        
                        # Add action sequence for this solution
                        original_action_sequence = [
                            {
                                "object_id": action.object_id,
                                "target": (action.x, action.y, action.theta)
                            }
                            for action in actions
                        ]
                        
                        # Apply solution smoothing if enabled
                        if task.smooth_solutions and original_action_sequence:
                            smoother = SolutionSmoother(max_search_actions=task.max_smooth_actions)
                            goal_checker = create_goal_checker(robot_goal)
                            
                            # Convert to format expected by smoother
                            smoother_actions = [
                                {
                                    "object_name": act["object_id"],
                                    "target_pose": {"x": act["target"][0], "y": act["target"][1], "theta": act["target"][2]}
                                }
                                for act in original_action_sequence
                            ]
                            
                            smooth_result = smoother.smooth_solution(env, smoother_actions, goal_checker)
                            
                            # Convert back to standard format
                            if smooth_result["smoothed_solution"] != smooth_result["original_solution"]:
                                episode_result.action_sequence = [
                                    {
                                        "object_id": act["object_name"],
                                        "target": (act["target_pose"]["x"], act["target_pose"]["y"], act["target_pose"]["theta"])
                                    }
                                    for act in smooth_result["smoothed_solution"]
                                ]
                                episode_result.original_action_sequence = original_action_sequence
                                episode_result.smoothing_stats = smooth_result["smoothing_stats"]
                            else:
                                # No improvement found - still record original sequence for metadata
                                episode_result.action_sequence = original_action_sequence
                                episode_result.original_action_sequence = original_action_sequence
                                episode_result.smoothing_stats = smooth_result["smoothing_stats"]
                        else:
                            episode_result.action_sequence = original_action_sequence
                        
                        episode_results.append(episode_result)
                else:
                    # Standard behavior for non-optimal planners or single solutions
                    # Create episode result with failure classification
                    failure_info = None
                    if not planner_result.success:
                        failure_info = create_failure_info(planner_result.error_message)
                    
                    episode_result = ModularEpisodeResult(
                        episode_id=episode_id,
                        algorithm=planner.algorithm_name,
                        algorithm_version=planner.algorithm_version,
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
                        xml_file=task.xml_file,
                        robot_goal=robot_goal,
                        failure_code=failure_info['failure_code'] if failure_info else None,
                        failure_description=failure_info['failure_description'] if failure_info else ""
                    )
                    
                    if planner_result.solution_found and planner_result.action_sequence:
                        original_action_sequence = [
                            {
                                "object_id": action.object_id,
                                "target": (action.x, action.y, action.theta)
                            }
                            for action in planner_result.action_sequence
                        ]
                        
                        # Apply solution smoothing if enabled
                        if task.smooth_solutions and original_action_sequence:
                            smoother = SolutionSmoother(max_search_actions=task.max_smooth_actions)
                            goal_checker = create_goal_checker(robot_goal)
                            
                            # Convert to format expected by smoother
                            smoother_actions = [
                                {
                                    "object_name": act["object_id"],
                                    "target_pose": {"x": act["target"][0], "y": act["target"][1], "theta": act["target"][2]}
                                }
                                for act in original_action_sequence
                            ]
                            
                            smooth_result = smoother.smooth_solution(env, smoother_actions, goal_checker)
                            
                            # Convert back to standard format
                            if smooth_result["smoothed_solution"] != smooth_result["original_solution"]:
                                episode_result.action_sequence = [
                                    {
                                        "object_id": act["object_name"],
                                        "target": (act["target_pose"]["x"], act["target_pose"]["y"], act["target_pose"]["theta"])
                                    }
                                    for act in smooth_result["smoothed_solution"]
                                ]
                                episode_result.original_action_sequence = original_action_sequence
                                episode_result.smoothing_stats = smooth_result["smoothing_stats"]
                            else:
                                # No improvement found - still record original sequence for metadata
                                episode_result.action_sequence = original_action_sequence
                                episode_result.original_action_sequence = original_action_sequence
                                episode_result.smoothing_stats = smooth_result["smoothing_stats"]
                        else:
                            episode_result.action_sequence = original_action_sequence
                    
                    if not planner_result.success:
                        episode_result.error_message = planner_result.error_message
                    
                    episode_results.append(episode_result)
                
            except Exception as e:
                # Create failed episode result with failure classification
                failure_info = create_failure_info(str(e), e)
                
                episode_result = ModularEpisodeResult(
                    episode_id=episode_id,
                    algorithm=task.algorithm,
                    algorithm_version="unknown",
                    success=False,
                    solution_found=False,
                    state_observations=None,  # No state observations for failed episodes
                    post_action_state_observations=None,  # No post-action state observations for failed episodes
                    static_object_info=None,  # No static info for failed episodes
                    error_message=str(e),
                    xml_file=task.xml_file,
                    robot_goal=robot_goal,
                    failure_code=failure_info['failure_code'],
                    failure_description=failure_info['failure_description']
                )
                episode_results.append(episode_result)
        
        # Filter episodes by minimum action sequence length if requested
        episodes_before_filtering = len(episode_results)
        episodes_filtered_out = 0
        
        if task.filter_minimum_length and episode_results:
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
        
        # Save results
        worker_result_data = {
            "task_id": task.task_id,
            "success": True,
            "episodes_collected": len(episode_results),
            "episodes_before_filtering": episodes_before_filtering,
            "episodes_filtered_out": episodes_filtered_out,
            "processing_time": time.time() - start_time,
            "episode_results": [asdict(ep) for ep in episode_results]
        }
        
        output_file = Path(task.output_dir) / f"{task.task_id}_results.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(worker_result_data, f)
        
        # Save additional smoothed file if smoothing was enabled and successful
        if task.smooth_solutions:
            smoothed_episodes = []
            print(f"[DEBUG] Checking {len(episode_results)} episodes for smoothing...")
            for ep in episode_results:
                print(f"[DEBUG] Episode {ep.episode_id}: solution_found={ep.solution_found}, has_action_seq={ep.action_sequence is not None}")
                if ep.action_sequence:
                    print(f"[DEBUG]   Action sequence length: {len(ep.action_sequence)}")
                
                has_original = hasattr(ep, 'original_action_sequence') and ep.original_action_sequence is not None
                print(f"[DEBUG]   Has original_action_sequence: {has_original}")
                if has_original:
                    print(f"[DEBUG]   Original sequence length: {len(ep.original_action_sequence)}")
                    print(f"[DEBUG]   Sequences equal? {ep.action_sequence == ep.original_action_sequence}")
                
                if hasattr(ep, 'smoothing_stats') and ep.smoothing_stats:
                    print(f"[DEBUG]   Smoothing stats: {ep.smoothing_stats}")
                    
                # Include ALL episodes where smoothing was attempted (has smoothing_stats)
                if (ep.solution_found and ep.action_sequence and 
                    hasattr(ep, 'smoothing_stats') and ep.smoothing_stats is not None):
                    # This episode had smoothing attempted - create smoothed version
                    was_improved = (hasattr(ep, 'original_action_sequence') and ep.original_action_sequence is not None and
                                  len(ep.action_sequence) < len(ep.original_action_sequence))
                    print(f"[DEBUG] ✅ Episode {ep.episode_id} had smoothing attempted (improved: {was_improved})")
                    smoothed_ep = type(ep)(
                        episode_id=ep.episode_id,
                        algorithm=ep.algorithm,
                        algorithm_version=ep.algorithm_version,
                        success=ep.success,
                        solution_found=ep.solution_found,
                        solution_depth=len(ep.action_sequence),  # Use smoothed length
                        search_time_ms=ep.search_time_ms,
                        nodes_expanded=ep.nodes_expanded,
                        terminal_checks=ep.terminal_checks,
                        max_depth_reached=ep.max_depth_reached,
                        action_sequence=ep.action_sequence,  # Smoothed sequence
                        algorithm_stats=ep.algorithm_stats,
                        error_message=ep.error_message,
                        failure_code=ep.failure_code,
                        failure_description=ep.failure_description,
                        # Keep only states/observations for the smoothed sequence
                        state_observations=ep.state_observations[:len(ep.action_sequence)] if ep.state_observations else None,
                        post_action_state_observations=ep.post_action_state_observations[:len(ep.action_sequence)] if ep.post_action_state_observations else None,
                        static_object_info=ep.static_object_info,
                        xml_file=ep.xml_file,
                        robot_goal=ep.robot_goal,
                        # Include smoothing metadata
                        original_action_sequence=ep.original_action_sequence,
                        smoothing_stats=ep.smoothing_stats
                    )
                    smoothed_episodes.append(smoothed_ep)
            
            print(f"[DEBUG] Found {len(smoothed_episodes)} episodes that were successfully smoothed")
            if smoothed_episodes:
                print(f"[DEBUG] Saving smoothed results file...")
                # Create smoothed result data
                smoothed_result_data = {
                    "task_id": task.task_id,
                    "success": True,
                    "episodes_collected": len(smoothed_episodes),
                    "episodes_before_filtering": len(smoothed_episodes),
                    "episodes_filtered_out": 0,
                    "processing_time": worker_result_data["processing_time"],
                    "episode_results": [asdict(ep) for ep in smoothed_episodes],
                    "smoothing_metadata": {
                        "original_episodes_count": len(episode_results),
                        "smoothed_episodes_count": len(smoothed_episodes),
                        "smoothing_enabled": True
                    }
                }
                
                # Save smoothed file
                smoothed_file = Path(task.output_dir) / f"{task.task_id}_smoothed_results.pkl"
                with open(smoothed_file, 'wb') as f:
                    pickle.dump(smoothed_result_data, f)
        
        # Set result for return
        result.success = True
        result.episodes_collected = len(episode_results)
        result.episodes_before_filtering = episodes_before_filtering
        result.episodes_filtered_out = episodes_filtered_out
        result.processing_time = time.time() - start_time
        result.episode_results = episode_results
        
    except Exception as e:
        result.error_message = f"Worker failed: {str(e)}\n{traceback.format_exc()}"
        result.processing_time = time.time() - start_time
        result.episodes_collected = len(episode_results) if 'episode_results' in locals() else 0
        
        # Log failure classification for worker-level failures
        failure_info = create_failure_info(str(e), e)
        print(f"[Worker] Failure classified as: {failure_info['failure_description']} (code: {failure_info['failure_code']})")
    
    return result


class ModularParallelCollectionManager:
    """Manager for modular parallel data collection."""
    
    def __init__(self, config: ModularCollectionConfig):
        self.config = config
        self._pool = None  # Track pool for signal handling
        
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
            if 'epsilon' not in params:
                params['epsilon'] = self.config.epsilon
        
        # Setup output directory
        self.output_base = Path(self.config.output_dir)
        self.output_dir = self.output_base / f"modular_data_{self.config.hostname}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup progress tracking
        self.progress_file = self.output_dir / "collection_progress.txt"
    
    def create_tasks(self) -> List[ModularWorkerTask]:
        """Create worker tasks from environment file subset."""
        # Discover environment files
        xml_files = discover_environment_files(
            self.config.xml_base_dir, 
            self.config.start_idx, 
            self.config.end_idx
        )
        
        # Create tasks
        tasks = []
        for i, xml_file in enumerate(xml_files):
            task_id = f"{self.config.hostname}_env_{self.config.start_idx + i:06d}"
            
            task = ModularWorkerTask(
                task_id=task_id,
                xml_file=xml_file,
                config_file=self.config.config_file,
                output_dir=str(self.output_dir),
                episodes_per_env=self.config.episodes_per_env,
                algorithm=self.config.algorithm,
                planner_config=self.config.planner_config,
                ml_object_model_path=self.config.ml_object_model_path,
                ml_goal_model_path=self.config.ml_goal_model_path,
                filter_minimum_length=self.config.filter_minimum_length,
                smooth_solutions=self.config.smooth_solutions,
                max_smooth_actions=self.config.max_smooth_actions
            )
            tasks.append(task)
        
        # Pass ML model paths to workers (models will be preloaded within each worker)
        if (self.config.object_selection_strategy == "ml" or 
            self.config.goal_selection_strategy == "ml"):
            print("📋 Preparing ML model paths for worker-side preloading...")
            
            # Inject model paths into tasks for worker-side loading
            for task in tasks:
                if self.config.object_selection_strategy == "ml":
                    task.ml_object_model_path = self.config.ml_object_model_path
                if self.config.goal_selection_strategy == "ml":
                    task.ml_goal_model_path = self.config.ml_goal_model_path
        
        return tasks
    
    def run_parallel_collection(self):
        """Execute modular parallel data collection with progress tracking."""
        
        # Create tasks
        tasks = self.create_tasks()
        if not tasks:
            return
        
        # Initialize progress tracking
        start_time = time.time()
        completed_tasks = 0
        total_episodes = 0
        failed_tasks = []
        
        print(f"🚀 Starting modular parallel data collection")
        print(f"📊 Algorithm: {self.config.algorithm}")
        print(f"🔢 Processing {len(tasks)} environments with {self.config.num_workers} workers")
        
        # Execute tasks in parallel with progress bar
        pool = None
        try:
            pool = Pool(processes=self.config.num_workers)
            self._pool = pool  # Store for signal handling
            results = []
            with tqdm(total=len(tasks), desc="Collecting data", unit="env") as pbar:
                for result in pool.imap_unordered(modular_worker_process, tasks):
                    completed_tasks += 1
                    results.append(result)
                    
                    # Count episodes regardless of worker success/failure
                    total_episodes += result.episodes_collected
                    
                    if result.success:
                        pbar.set_postfix({
                            "episodes": total_episodes,
                            "failed": len(failed_tasks)
                        })
                    else:
                        failed_tasks.append(result)
                        print(f"\n❌ Task {result.task_id} failed: {result.error_message}")
                        print(f"   → But collected {result.episodes_collected} episodes before failing")
                        pbar.set_postfix({
                            "episodes": total_episodes,
                            "failed": len(failed_tasks)
                        })
                    
                    pbar.update(1)
        finally:
            self._cleanup_pool(pool)
            self._pool = None  # Clear reference
        
        # Final summary
        total_time = time.time() - start_time
        success_rate = (len(tasks) - len(failed_tasks)) / len(tasks) * 100
        
        print(f"\n🎉 Collection complete!")
        print(f"📊 Episodes: {total_episodes} total")
        print(f"🎯 Task success rate: {success_rate:.1f}% ({total_time/60:.1f}m)")
        
        self._save_final_summary(tasks, results, total_time)
    
    def _cleanup_pool(self, pool: Optional[Pool], timeout: float = 30.0):
        """Robustly cleanup multiprocessing pool with timeout."""
        if pool is None:
            return
        
        try:
            # Stop accepting new tasks
            pool.close()
            
            # Wait for workers to finish (pool.join() doesn't support timeout)
            # We implement timeout manually
            import threading
            join_thread = threading.Thread(target=pool.join)
            join_thread.start()
            join_thread.join(timeout=timeout)
            
            if join_thread.is_alive():
                # Timeout reached, force terminate
                print(f"⚠️  Workers didn't finish within {timeout}s, force terminating...")
                pool.terminate()
                pool.join()  # This should be fast after terminate
            
        except Exception as e:
            print(f"⚠️  Warning: Pool cleanup had issues: {e}")
            try:
                # Force terminate as fallback
                pool.terminate()
                pool.join()
            except:
                print("❌ Warning: Could not force terminate worker processes")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\n🛑 Received signal {signum}. Cleaning up workers...")
        if self._pool is not None:
            self._cleanup_pool(self._pool, timeout=10.0)
        print("🧹 Cleanup complete. Exiting...")
        sys.exit(1)
    
    def _save_final_summary(self, tasks: List[ModularWorkerTask], 
                          results: List[ModularWorkerResult], total_time: float):
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
        total_before_filtering = sum(result.episodes_before_filtering for result in results if hasattr(result, 'episodes_before_filtering'))
        total_filtered_out = sum(result.episodes_filtered_out for result in results if hasattr(result, 'episodes_filtered_out'))
        
        # Calculate failure statistics
        failure_stats = get_failure_statistics(all_episodes)
        
        summary = {
            'collection_metadata': {
                'hostname': self.config.hostname,
                'algorithm': self.config.algorithm,
                'total_duration_seconds': total_time,
                'execution_mode': 'parallel',
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
            },
            'failure_analysis': failure_stats
        }
        
        # Save summary
        summary_file = self.output_dir / f"collection_summary_{self.config.hostname}.pkl"
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        # Save human-readable summary
        summary_txt = self.output_dir / f"summary_{self.config.hostname}.txt"
        with open(summary_txt, 'w') as f:
            f.write("Modular Parallel Data Collection Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Algorithm: {self.config.algorithm}\n")
            f.write(f"Execution mode: Parallel ({self.config.num_workers} workers)\n")
            f.write(f"Total runtime: {total_time/60:.1f} minutes\n")
            f.write(f"Total episodes: {len(all_episodes)}\n\n")
            
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
            
            # Add failure analysis
            failure_analysis = summary['failure_analysis']
            f.write(f"\nFailure Analysis:\n")
            f.write(f"Failed episodes: {failure_analysis['failed_episodes']}\n")
            if failure_analysis['failure_breakdown']:
                f.write(f"Top failure reasons:\n")
                # Sort failures by count (descending)
                sorted_failures = sorted(
                    failure_analysis['failure_breakdown'].items(), 
                    key=lambda x: x[1]['count'], 
                    reverse=True
                )
                for failure_desc, info in sorted_failures[:5]:  # Top 5 failures
                    f.write(f"  • {failure_desc}: {info['count']} episodes ({info['percentage']:.1f}%)\n")


def main():
    """Main entry point for modular parallel data collection."""
    parser = argparse.ArgumentParser(description="Modular Parallel Data Collection")
    
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
    parser.add_argument("--ml-samples", type=int, default=32,
                        help="Number of ML inference samples (default: 32)")
    parser.add_argument("--ml-device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="ML inference device (default: cuda)")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Epsilon for epsilon-greedy goal strategy (0.0=pure ML, 1.0=pure random). If specified with --goal-strategy=ml, uses epsilon-greedy mixing.")
    
    # Optional arguments
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes")
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
    parser.add_argument("--smooth-solutions", action="store_true",
                        help="Apply exhaustive smoothing to find minimal subsequences")
    parser.add_argument("--max-smooth-actions", type=int, default=20,
                        help="Maximum solution length to attempt smoothing on (default: 20)")
    
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
    
    # Validate ML strategy requirements
    if args.object_strategy == "ml" and not args.ml_object_model:
        print("❌ Error: --ml-object-model is required when using ML object strategy")
        return 1
    
    if args.goal_strategy == "ml" and not args.ml_goal_model:
        print("❌ Error: --ml-goal-model is required when using ML goal strategy")
        return 1
    
    # Validate epsilon parameter
    if args.epsilon is not None:
        if args.goal_strategy != "ml":
            print("❌ Error: --epsilon can only be used with --goal-strategy=ml")
            return 1
        if not (0.0 <= args.epsilon <= 1.0):
            print("❌ Error: --epsilon must be between 0.0 and 1.0")
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
    config = ModularCollectionConfig(
        xml_base_dir=args.xml_dir,
        config_file=args.config_file,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        episodes_per_env=args.episodes_per_env,
        num_workers=args.workers,
        algorithm=args.algorithm,
        object_selection_strategy=args.object_strategy,
        goal_selection_strategy=args.goal_strategy,
        ml_object_model_path=args.ml_object_model,
        ml_goal_model_path=args.ml_goal_model,
        ml_samples=args.ml_samples,
        ml_device=args.ml_device,
        epsilon=args.epsilon,
        smooth_solutions=args.smooth_solutions,
        max_smooth_actions=args.max_smooth_actions,
        filter_minimum_length=args.filter_minimum_length,
        planner_config=planner_config
    )
    
    # Execute parallel data collection
    try:
        manager = ModularParallelCollectionManager(config)
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, manager._signal_handler)
        signal.signal(signal.SIGTERM, manager._signal_handler)
        
        manager.run_parallel_collection()
        return 0
    
    except KeyboardInterrupt:
        print("\n🛑 Data collection interrupted by user")
        print("🧹 Cleaning up worker processes...")
        return 1
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())