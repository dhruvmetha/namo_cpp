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
from dataclasses import dataclass, asdict, replace
from multiprocessing import Pool
import glob
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NAMO imports
import namo_rl
from namo.core import BasePlanner, PlannerConfig, PlannerResult, PlannerFactory
from namo.core.xml_goal_parser import extract_goal_with_fallback

# Import all available planners (self-register on import)
from namo.planners.idfs.standard_idfs import StandardIterativeDeepeningDFS
from namo.planners.idfs.tree_idfs import TreeIterativeDeepeningDFS
from namo.planners.sampling.random_sampling import RandomSamplingPlanner
from namo.planners.idfs.optimal_idfs import OptimalIterativeDeepeningDFS
from namo.planners.opening.region_opening import RegionOpeningPlanner

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
        # Use the environment's built-in reachability checking
        # which uses wavefront planning to determine if robot can reach goal
        return env.is_robot_goal_reachable()
    return check_goal


def apply_solution_smoothing(episode_result, env, original_action_sequence, original_states, original_post_states,
                           robot_goal, task):
    """
    Apply solution smoothing and update episode result with smoothed data.

    Args:
        episode_result: Episode result object to update
        env: Environment instance
        original_action_sequence: Original action sequence from planner
        original_states: Original state observations
        original_post_states: Original post-action state observations
        robot_goal: Robot goal position
        task: Worker task configuration
    """
    if not task.smooth_solutions or not original_action_sequence:
        # No smoothing - use original data
        episode_result.action_sequence = original_action_sequence
        episode_result.state_observations = original_states
        episode_result.post_action_state_observations = original_post_states
        return

    smoother = SolutionSmoother(max_search_actions=task.max_smooth_actions)
    goal_checker = create_goal_checker(robot_goal)

    # Store original trajectory data
    episode_result.original_action_sequence = original_action_sequence
    episode_result.original_state_observations = original_states
    episode_result.original_post_action_state_observations = original_post_states

    # Convert to format expected by smoother
    smoother_actions = [
        {
            "object_name": act["object_id"],
            "target_pose": {"x": act["target"][0], "y": act["target"][1], "theta": act["target"][2]}
        }
        for act in original_action_sequence
    ]

    smooth_result = smoother.smooth_solution(env, smoother_actions, goal_checker)

    # Convert smoothed solution back to standard format
    smoothed_action_sequence = [
        {
            "object_id": act["object_name"],
            "target": (act["target_pose"]["x"], act["target_pose"]["y"], act["target_pose"]["theta"])
        }
        for act in smooth_result["smoothed_solution"]
    ]

    # Use state observations collected by the smoother
    episode_result.state_observations = smooth_result.get("smoothed_state_observations", [])
    episode_result.post_action_state_observations = smooth_result.get("smoothed_post_action_state_observations", [])
    episode_result.action_sequence = smoothed_action_sequence
    episode_result.smoothing_stats = smooth_result["smoothing_stats"]


def apply_action_refinement(episode_result, env, robot_goal, task):
    """
    Apply action refinement to replace action targets with actual achieved positions.
    Only accepts refinements that pass validation (still solve the navigation task).

    Args:
        episode_result: Episode result with smoothed data
        env: Environment instance
        robot_goal: Robot goal position
        task: Worker task configuration
    """
    if not task.refine_actions or not episode_result.action_sequence:
        # No refinement requested or no actions to refine
        episode_result.refinement_accepted = False
        episode_result.refinement_stats = {"attempted": False, "reason": "not_requested"}
        return

    if not episode_result.post_action_state_observations:
        # Need post-action states for refinement
        episode_result.refinement_accepted = False
        episode_result.refinement_stats = {"attempted": False, "reason": "missing_post_action_states"}
        return

    from namo.planners.idfs.solution_refiner import SolutionRefiner

    refiner = SolutionRefiner()
    goal_checker = create_goal_checker(robot_goal)

    # Attempt refinement with validation
    refinement_result = refiner.refine_with_validation(env, episode_result, goal_checker)

    # Update episode result with refinement data
    episode_result.refinement_accepted = refinement_result['refinement_accepted']
    episode_result.refinement_stats = refinement_result['refinement_stats']

    if refinement_result['refinement_accepted']:
        # Accept refined actions
        episode_result.refined_action_sequence = refinement_result['refined_action_sequence']
    else:
        # Keep refined_action_sequence as None to indicate rejection
        episode_result.refined_action_sequence = None




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
    algorithm: str = "region_opening"  # Default algorithm
    planner_config: PlannerConfig = None  # Will use default if None

    # Solution smoothing
    smooth_solutions: bool = False
    max_smooth_actions: int = 20

    # Action refinement (post-smoothing step)
    refine_actions: bool = False
    validate_refinement: bool = True

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
    # Filtering options
    filter_minimum_length: bool = False
    # Solution smoothing options
    smooth_solutions: bool = False
    max_smooth_actions: int = 20
    # Action refinement options (post-smoothing step)
    refine_actions: bool = False
    validate_refinement: bool = True


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

    # Original trajectory (full, untruncated)
    original_state_observations: Optional[List[Dict[str, List[float]]]] = None  # Original states before each action
    original_post_action_state_observations: Optional[List[Dict[str, List[float]]]] = None  # Original states after each action

    # Smoothed trajectory (newly computed for smoothed sequence)
    state_observations: Optional[List[Dict[str, List[float]]]] = None  # Smoothed states before each action
    post_action_state_observations: Optional[List[Dict[str, List[float]]]] = None  # Smoothed states after each action

    # Reachable objects information (for mask generation)
    reachable_objects_before_action: Optional[List[List[str]]] = None  # Reachable objects before each action
    reachable_objects_after_action: Optional[List[List[str]]] = None  # Reachable objects after each action

    # Action refinement results (post-smoothing step)
    refined_action_sequence: Optional[List[Dict]] = None  # Refined action sequence with actual achieved positions
    refinement_accepted: Optional[bool] = None  # Whether refinement passed validation
    refinement_stats: Optional[Dict[str, Any]] = None  # Refinement statistics and metrics

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
    

    # all_xml_files = []
    # folder = 'medium_train_envs'
    # for d in ['very_hard']:
    #     with open(f'{folder}/envs_names_{d}.pkl', 'rb') as f:
    #         envs_names = pickle.load(f)
    #     for env_name in envs_names:
    #         xml_file = os.path.join(base_dir, env_name)
    #         all_xml_files.append(xml_file)
    # all_xml_files = sorted(all_xml_files)
    
    sets = [1, 2]
    benchmarks = [1, 2, 3, 4, 5]
    all_xml_files = []
    for set in sets:
        for benchmark in benchmarks:
            xml_pattern = os.path.join(base_dir, "medium", f"set{set}", f"benchmark_{benchmark}", "*.xml")
            sorted_xml_files = sorted(glob.glob(xml_pattern, recursive=True))
            all_xml_files.extend(sorted_xml_files[:1000]) # train
            # all_xml_files.extend(sorted_xml_files[1000:1100]) # test
    # Apply subset selection
    if end_idx == -1:
        end_idx = len(all_xml_files)
        
    random.seed(42)
    random.shuffle(all_xml_files)
    subset_files = all_xml_files[start_idx:end_idx]
    
    return subset_files

    # solved_envs = []
    # with open('1_push_solved.pkl', 'rb') as f:
    #     solved_envs = pickle.load(f)
    # solved_envs = set(solved_envs)
    # print("solved_envs", len(solved_envs))
    # sets = [1, 2]
    # benchmarks = [1, 2, 3, 4, 5]
    # all_xml_files = []
    
    # for set_idx in sets:
    #     for benchmark_idx in benchmarks:
    #         xml_pattern = os.path.join(base_dir, "medium", f"set{set_idx}", f"benchmark_{benchmark_idx}", "*.xml")
    #         sorted_xml_files = sorted(glob.glob(xml_pattern, recursive=True))
    #         for xml_file in sorted_xml_files[:1000]:
    #                 all_xml_files.append(xml_file.split(f'{base_dir}/')[-1])
    #         # all_xml_files.extend(sorted_xml_files[1000:1100]) # test
    # # Apply subset selection
    # subset_files = all_xml_files[start_idx:end_idx]
    # subset_files = set(subset_files) - solved_envs
    # subset_files = list(subset_files)
    # if end_idx == -1:
    #     end_idx = len(all_xml_files)
    # subset_files = sorted(subset_files)
    # print("unsolved envs", len(subset_files))
    # final_subset_files = []
    # for subset_file in subset_files:
    #     xml_file = os.path.join(base_dir, subset_file)
    #     final_subset_files.append(xml_file)
    # final_subset_files = sorted(final_subset_files)
    # return final_subset_files

def generate_hostname_prefix() -> str:
    """Generate hostname-based prefix for output files."""
    hostname = socket.gethostname()
    short_hostname = hostname.split('.')[0]
    return short_hostname


def generate_goal_for_environment(xml_file: str) -> Tuple[float, float, float]:
    """Extract goal position from XML environment file."""
    fallback_goal = (-0.5, 1.3, 0.0)
    return extract_goal_with_fallback(xml_file, fallback_goal)


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

        # Create planner once per worker
        planner = None
        
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
                    planner = PlannerFactory.create_planner(task.algorithm, env, task.planner_config)
                
                # Reset planner for this episode (but don't recreate it)
                planner.reset()
                
                
                # Check initial reachability before search
                env.set_robot_goal(*robot_goal)
                
                # Run planning
                planner_result = planner.search(robot_goal)

                # Special handling for region_opening planner: convert AttemptResults to episodes
                is_region_opening = task.algorithm == "region_opening"

                if is_region_opening and planner_result.algorithm_stats and 'attempt_results' in planner_result.algorithm_stats:
                    # Process each AttemptResult as a separate episode
                    for attempt_idx, attempt in enumerate(planner_result.algorithm_stats['attempt_results']):
                        attempt_episode_id = f"{episode_id}_neighbour_{attempt_idx}_{attempt.neighbour_region_label}"

                        # Build action sequence from attempt (handle both single push and multi-push chains)
                        action_sequence = []
                        solution_depth = 0
                        if attempt.success:
                            if attempt.goal_chain:
                                # Multi-push chain
                                for goal in attempt.goal_chain:
                                    action_sequence.append({
                                        "object_id": attempt.chosen_object_id,
                                        "target": (goal.x, goal.y, goal.theta)
                                    })
                                solution_depth = len(attempt.goal_chain)
                            elif attempt.chosen_goal:
                                # Single push
                                action_sequence = [{
                                    "object_id": attempt.chosen_object_id,
                                    "target": attempt.chosen_goal
                                }]
                                solution_depth = 1

                        # Create episode result for this attempt
                        # For region opening, use the actual region_goal_used instead of XML goal
                        # This ensures the goal mask matches what the planner validated
                        actual_goal = attempt.region_goal_used if attempt.region_goal_used else robot_goal

                        episode_result = ModularEpisodeResult(
                            episode_id=attempt_episode_id,
                            algorithm=planner.algorithm_name,
                            algorithm_version=planner.algorithm_version,
                            success=attempt.success,
                            solution_found=attempt.success,
                            solution_depth=solution_depth,
                            search_time_ms=attempt.timing_ms,
                            nodes_expanded=None,
                            terminal_checks=None,
                            max_depth_reached=solution_depth,
                            algorithm_stats={
                                'neighbour_region_label': attempt.neighbour_region_label,
                                'validation_method': attempt.validation_method,
                                'connectivity_before': attempt.connectivity_before,
                                'connectivity_after': attempt.connectivity_after,
                                'region_goal_used': attempt.region_goal_used,
                                'chosen_object_id': attempt.chosen_object_id,
                                'chain_depth': attempt.chain_depth,
                                'total_cost': getattr(attempt, 'total_cost', None),
                                'skill_calls_before_success': getattr(attempt, 'skill_calls_before_success', None),
                                'solutions_found_for_neighbour': getattr(attempt, 'solutions_found_for_neighbour', None),
                                'solutions_cap_for_neighbour': getattr(attempt, 'solutions_cap_for_neighbour', None),
                                'solutions_total_for_neighbour': getattr(attempt, 'solutions_total_for_neighbour', None),
                                'pushes_total_for_neighbour': getattr(attempt, 'pushes_total_for_neighbour', None),
                            },
                            action_sequence=action_sequence,
                            state_observations=attempt.state_observations,
                            post_action_state_observations=attempt.post_action_state_observations,
                            reachable_objects_before_action=attempt.reachable_objects_before_action,
                            reachable_objects_after_action=attempt.reachable_objects_after_action,
                            static_object_info=static_object_info,
                            xml_file=task.xml_file,
                            robot_goal=actual_goal,
                            error_message=attempt.error_message or "",
                            failure_code=None,
                            failure_description=attempt.error_message or ""
                        )

                        episode_results.append(episode_result)

                    # Continue to next episode
                    continue

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
                        apply_solution_smoothing(
                            episode_result, env, original_action_sequence, states, post_states,
                            robot_goal, task
                        )

                        # Apply action refinement if enabled (post-smoothing step)
                        apply_action_refinement(episode_result, env, robot_goal, task)

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
                        apply_solution_smoothing(
                            episode_result, env, original_action_sequence,
                            planner_result.state_observations, planner_result.post_action_state_observations,
                            robot_goal, task
                        )

                        # Apply action refinement if enabled (post-smoothing step)
                        apply_action_refinement(episode_result, env, robot_goal, task)

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

        # Filter out episodes with empty action sequences (robot already at goal)
        # These episodes are successful but provide no useful training data
        initial_count = len(episode_results)
        episode_results = [ep for ep in episode_results if not (ep.solution_found and (not ep.action_sequence or len(ep.action_sequence) == 0))]
        empty_action_filtered = initial_count - len(episode_results)
        episodes_filtered_out += empty_action_filtered

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
    
    return result


class ModularParallelCollectionManager:
    """Manager for modular parallel data collection."""
    
    def __init__(self, config: ModularCollectionConfig):
        self.config = config
        self._pool = None  # Track pool for signal handling
        
        # Auto-detect hostname if not provided
        if self.config.hostname is None:
            self.config.hostname = generate_hostname_prefix()
        
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

            task_planner_config = self.config.planner_config
            if task_planner_config is not None:
                base_algorithm_params = task_planner_config.algorithm_params or {}
                task_algorithm_params = dict(base_algorithm_params)
                task_algorithm_params['xml_file'] = xml_file
                task_planner_config = replace(
                    task_planner_config,
                    algorithm_params=task_algorithm_params
                )
            
            task = ModularWorkerTask(
                task_id=task_id,
                xml_file=xml_file,
                config_file=self.config.config_file,
                output_dir=str(self.output_dir),
                episodes_per_env=self.config.episodes_per_env,
                algorithm=self.config.algorithm,
                planner_config=task_planner_config,
                filter_minimum_length=self.config.filter_minimum_length,
                smooth_solutions=self.config.smooth_solutions,
                max_smooth_actions=self.config.max_smooth_actions,
                refine_actions=self.config.refine_actions,
                validate_refinement=self.config.validate_refinement
            )
            tasks.append(task)
        
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
    # Pre-parse only --config-yaml to allow YAML defaults with CLI overrides
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-yaml", type=str, help="Path to YAML config file for defaults")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Modular Parallel Data Collection", parents=[pre_parser])
    
    # Core arguments (YAML may provide defaults; CLI can override)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for collected data (required if not provided via YAML)")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Starting index for environment file subset (required if not provided via YAML)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Ending index for environment file subset (exclusive) (required if not provided via YAML)")
    
    # Algorithm selection
    available_algorithms = PlannerFactory.list_available_planners()
    parser.add_argument("--algorithm", type=str, default="region_opening", choices=available_algorithms,
                        help=f"Planning algorithm to use. Options: {available_algorithms}")

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

    # Region opening planner arguments (only those used by RegionOpeningPlanner)
    parser.add_argument("--region-allow-collisions", action="store_true",
                        help="Allow object collisions during region opening pushes (default: False, terminate on collision)")
    parser.add_argument("--region-max-chain-depth", type=int, default=1,
                        help="Maximum chain depth for region opening: 1=single push, 2=2-push chains, 3=3-push chains (default: 1)")
    parser.add_argument("--region-max-solutions-per-neighbor", type=int, default=10,
                        help="Maximum solutions to keep per neighbor region (default: 10)")
    parser.add_argument("--region-max-recorded-solutions-per-neighbor", type=int, default=2,
                        help="Maximum solutions to record/save per neighbor (subset of found, default: 2)")
    parser.add_argument("--region-frontier-beam-width", type=int, default=None,
                        help="Optional beam width (K) to cap frontier per chain depth; None/<=0 disables")
    parser.add_argument("--goal-sampler", type=str, default=None,
                        choices=["primitive", "ml", "ml_primitive"],
                        help="Goal sampler override for region opening (primitive default)")
    parser.add_argument("--ml-goal-model", type=str,
                        help="Hydra output directory containing diffusion goal model")
    parser.add_argument("--ml-device", type=str, default="cuda",
                        help="Device to load diffusion goal model on")
    parser.add_argument("--ml-samples", type=int, default=32,
                        help="Number of diffusion samples per inference")
    parser.add_argument("--ml-min-goals", type=int, default=1,
                        help="Minimum ML goals required before accepting inference")
    parser.add_argument("--ml-match-position-tolerance", type=float, default=0.2,
                        help="Max positional error (m) between ML pose and primitive slot")
    parser.add_argument("--ml-match-angle-tolerance", type=float, default=0.35,
                        help="Max angular error (rad) between ML pose and primitive slot")
    parser.add_argument("--ml-match-angle-weight", type=float, default=0.5,
                        help="Weight applied to angular error in matching score")
    parser.add_argument("--ml-match-max-per-call", type=int, default=8,
                        help="Maximum ML goals to align per sampler call")
    parser.add_argument("--primitive-data-dir", type=str, default="data",
                        help="Directory containing primitive motion databases")
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
    parser.add_argument("--refine-actions", action="store_true",
                        help="Apply action refinement using actual achieved positions (post-smoothing step)")
    parser.add_argument("--validate-refinement", action="store_true", default=True,
                        help="Validate that refined actions still solve the task (default: True)")
    
    # If YAML provided, load and set parser defaults before final parse
    if pre_args.config_yaml:
        try:
            import yaml  # Requires PyYAML
            with open(pre_args.config_yaml, 'r') as f:
                yaml_cfg = yaml.safe_load(f) or {}
            if not isinstance(yaml_cfg, dict):
                yaml_cfg = {}
            # Only pass known keys; argparse will ignore unknown via set_defaults
            parser.set_defaults(**yaml_cfg)
        except Exception as e:
            print(f"⚠️  Warning: could not load YAML config '{pre_args.config_yaml}': {e}")

    args = parser.parse_args(remaining_argv)
    
    # Validate required arguments presence (after YAML + CLI merge)
    if args.output_dir is None or args.start_idx is None or args.end_idx is None:
        print("❌ Error: --output-dir, --start-idx, and --end-idx are required (via CLI or YAML)")
        return 1

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
    
    # Create planner configuration
    # Build algorithm_params with only the region opening parameters that are actually used
    algorithm_params = {}
    if args.algorithm == "region_opening":
        algorithm_params["primitive_data_dir"] = args.primitive_data_dir
        algorithm_params.update({
            "region_allow_collisions": args.region_allow_collisions,
            "region_max_chain_depth": args.region_max_chain_depth,
            "region_max_solutions_per_neighbor": args.region_max_solutions_per_neighbor,
        })
        # Optionally cap how many of the found solutions are recorded/saved per neighbor
        algorithm_params["region_max_recorded_solutions_per_neighbor"] = args.region_max_recorded_solutions_per_neighbor
        if args.region_frontier_beam_width is not None:
            algorithm_params["region_frontier_beam_width"] = args.region_frontier_beam_width

        if args.goal_sampler:
            algorithm_params["goal_sampler"] = args.goal_sampler
        if args.goal_sampler and args.goal_sampler.lower() in {"ml", "ml_primitive"}:
            if not args.ml_goal_model:
                parser.error("--ml-goal-model is required when goal sampler is set to 'ml'")
            algorithm_params.update({
                "ml_goal_model_path": args.ml_goal_model,
                "ml_device": args.ml_device,
                "ml_samples": args.ml_samples,
                "ml_min_goals": args.ml_min_goals,
                "ml_match_position_tolerance": args.ml_match_position_tolerance,
                "ml_match_angle_tolerance": args.ml_match_angle_tolerance,
                "ml_match_angle_weight": args.ml_match_angle_weight,
                "ml_match_max_per_call": args.ml_match_max_per_call,
                "primitive_data_dir": args.primitive_data_dir,
            })
        elif args.ml_goal_model:
            # Allow users to specify ML params even without explicit sampler flag
            algorithm_params.update({
                "goal_sampler": "ml",
                "ml_goal_model_path": args.ml_goal_model,
                "ml_device": args.ml_device,
                "ml_samples": args.ml_samples,
                "ml_min_goals": args.ml_min_goals,
                "ml_match_position_tolerance": args.ml_match_position_tolerance,
                "ml_match_angle_tolerance": args.ml_match_angle_tolerance,
                "ml_match_angle_weight": args.ml_match_angle_weight,
                "ml_match_max_per_call": args.ml_match_max_per_call,
                "primitive_data_dir": args.primitive_data_dir,
            })

    planner_config = PlannerConfig(
        max_depth=args.max_depth,
        max_goals_per_object=args.max_goals_per_object,
        max_terminal_checks=args.max_terminal_checks,
        max_search_time_seconds=args.search_timeout,
        verbose=args.verbose,
        collect_stats=True,
        algorithm_params=algorithm_params if algorithm_params else None
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
        smooth_solutions=args.smooth_solutions,
        max_smooth_actions=args.max_smooth_actions,
        refine_actions=args.refine_actions,
        validate_refinement=args.validate_refinement,
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