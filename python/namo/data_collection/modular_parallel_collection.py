#!/usr/bin/env python3
"""Modular Parallel Data Collection Pipeline

Algorithm-agnostic parallel data collection. Planners are pluggable through
the BasePlanner / PlannerFactory interface; the collection infrastructure
does not depend on which one is in use.

Usage:
    python modular_parallel_collection.py --algorithm region_opening --output-dir ./data --start-idx 0 --end-idx 10
"""

import os
import sys
import argparse
import socket
import pickle
import time
import traceback
import signal
import re
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
# Add namo visualization directory to path for ML adapters
namo_viz_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualization")
if namo_viz_path not in sys.path:
    sys.path.append(namo_viz_path)

from namo.core import BasePlanner, PlannerConfig, PlannerResult, PlannerFactory
from namo.core.xml_goal_parser import extract_goal_with_fallback

# Import all available planners (self-register on import)
from namo.planners.sampling.random_sampling import RandomSamplingPlanner
from namo.planners.opening.region_opening import RegionOpeningPlanner
from namo.planners.sampling.uniform_rollout_sampler import UniformRolloutSampler  # noqa: F401 — registers on import

# Import strategies for validation
from namo.strategies.object_selection_strategy import ObjectSelectionStrategy

# Import failure classification system
from namo.planners.utils.failure_codes import FailureCode, FailureClassifier, create_failure_info, get_failure_statistics

# Import solution smoothing system
from namo.planners.utils.solution_smoother import SolutionSmoother

import random
DEFAULT_GLOBAL_SEED = int(os.environ.get("NAMO_GLOBAL_SEED", "42"))
random.seed(DEFAULT_GLOBAL_SEED)


def set_global_seed(seed: int) -> None:
    """Set global RNG seed for deterministic planning runs."""
    random.seed(int(seed))


def _sanitize_run_name(name: str) -> str:
    """Collapse run labels to filesystem-safe tokens."""
    sanitized = re.sub(r"[^0-9A-Za-z._-]", "_", name.strip())
    return sanitized or "run"

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

    # Episode filtering options
    filter_minimum_length: bool = False  # Only keep episodes with minimum action sequence length per environment

    # Manifest file for fast loading (pre-generated list of XML files)
    manifest_file: Optional[str] = None

    hostname: str = None  # Auto-detected if None
    run_name: Optional[str] = None  # Optional suffix for per-run directories
    unique_run_dir: bool = False  # Auto-generate per-run subdirectory suffix when True


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
    # Region/object skip dict (blacklist): skip specific (region, object) pairs during neighbor exploration
    # Parsed from manifest file (tab-separated format: xml_path\tregion1:obj1,region2:obj2,...)
    # Dict maps region_label -> list of object_ids to skip (empty list = skip entire region)
    region_object_skip: Optional[Dict[str, List[str]]] = None


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

    # Static object information (sizes, types) - stored once per environment
    static_object_info: Optional[Dict[str, Dict[str, Any]]] = None

    # Meta information
    xml_file: str = ""
    robot_goal: Optional[Tuple[float, float, float]] = None

    # Collision tracking for hardness metrics (aggregated across all pushes in chain)
    any_wall_collision: bool = False  # Did any push hit a wall?
    unique_movable_collision_count: int = 0  # Number of unique movable objects hit across all pushes


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


def discover_environment_files(base_dir: str, start_idx: int, end_idx: int, manifest_file: str = None) -> List[Tuple[str, Optional[Dict[str, List[str]]]]]:
    """Discover and filter XML environment files by index range.

    Args:
        base_dir: Base directory containing XML files (used if no manifest)
        start_idx: Starting index for subset selection
        end_idx: Ending index for subset selection (exclusive)
        manifest_file: Optional path to pre-generated manifest file for fast loading.
            Supports extended tab-separated format with two styles:
            1. Region-only: xml_path[\\tregion1,region2,...] - skip entire regions
            2. Triplets: xml_path[\\tregion1:obj1,region1:obj2,region2:obj3,...] - skip specific (region,object) pairs

    Returns:
        List of tuples: (xml_path, region_object_skip) where region_object_skip is None or
        a dict mapping region_label -> list of object_ids to skip (empty list = skip all objects for that region)
    """
    if manifest_file and os.path.exists(manifest_file):
        # Fast path: read from pre-generated manifest
        print(f"Loading from manifest: {manifest_file}")
        with open(manifest_file, 'r') as f:
            # Read only the lines we need for memory efficiency
            all_entries = []
            for i, line in enumerate(f):
                if i >= end_idx:
                    break
                if i >= start_idx:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # Parse tab-separated format: xml_path[\tregion1:obj1,region2:obj2,...]
                    parts = line.split('\t')
                    xml_path = parts[0].strip()
                    region_object_skip = None
                    if len(parts) > 1 and parts[1].strip():
                        # Parse comma-separated entries
                        region_object_skip = {}
                        for entry in parts[1].split(','):
                            entry = entry.strip()
                            if not entry:
                                continue
                            if ':' in entry:
                                # Triplet format: region:object
                                region, obj = entry.split(':', 1)
                                region = region.strip()
                                obj = obj.strip()
                                if region not in region_object_skip:
                                    region_object_skip[region] = []
                                region_object_skip[region].append(obj)
                            else:
                                # Region-only format (backward compatible): skip all objects for this region
                                region_object_skip[entry] = []  # Empty list = skip entire region
                    all_entries.append((xml_path, region_object_skip))

        # Count entries with filters
        filtered_count = sum(1 for _, rs in all_entries if rs)
        total_skips = sum(
            sum(len(objs) if objs else 1 for objs in rs.values())
            for _, rs in all_entries if rs
        )
        print(f"Loaded {len(all_entries)} environments from manifest (indices {start_idx}:{end_idx})")
        if filtered_count > 0:
            print(f"  {filtered_count} environments have skip filters ({total_skips} total region:object pairs)")
        return all_entries

    # Fallback: discover files directly (slow for millions of files)
    print(f"No manifest provided, scanning directory: {base_dir}")
    print("  (For faster loading with millions of files, generate a manifest first)")
    print("  (Run: python scripts/generate_xml_manifest.py --input-dir <dir> --output manifest.txt)")

    xml_pattern = os.path.join(base_dir, "**", "*.xml")
    all_xml_files = glob.glob(xml_pattern, recursive=True)
    print(f"Found {len(all_xml_files)} environments before subset selection")
    all_xml_files = [file for file in tqdm(all_xml_files) if not file.endswith('_temp.xml')]
    all_xml_files = sorted(all_xml_files)
    random.seed(42)
    random.shuffle(all_xml_files)
    subset_files = all_xml_files[start_idx:end_idx]

    # Return as tuples with no region_object_skip
    return [(f, None) for f in subset_files]

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
    # Pin per-worker math-library thread pools to 1 so N workers don't oversubscribe the box.
    # torch, cv2 (renderer), and BLAS each default to <ncores> intra-op threads; with N workers that
    # is N*ncores threads on ncores cores -> catastrophic context-switch thrash -> no scene finishes.
    # cv2 does NOT honour OMP_NUM_THREADS, so it MUST be pinned here in code. Gated by env for A/B.
    if os.environ.get("NAMO_PIN_THREADS", "0") == "1":
        try:
            import cv2  # noqa: F401
            cv2.setNumThreads(1)
        except Exception:
            pass
        try:
            import torch  # noqa: F401
            torch.set_num_threads(1)
        except Exception:
            pass
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
                emits_attempt_results = task.algorithm in ("region_opening", "uniform_rollout_sampler")

                if emits_attempt_results and planner_result.algorithm_stats and 'attempt_results' in planner_result.algorithm_stats:
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
                                        "target": (goal.x, goal.y, goal.theta),
                                        "edge_idx": int(getattr(goal, "edge_idx", -1)),
                                        "depth": int(getattr(goal, "depth", -1)),
                                    })
                                solution_depth = len(attempt.goal_chain)
                            elif attempt.chosen_goal:
                                # Single push (sampler success path: goal_chain is None).
                                # Backfill primitive identity from the first winning entry
                                # in primitive_trial_log so action_sequence matches what
                                # the trial log records — otherwise replay tools see -1/-1.
                                winning_edge_idx = -1
                                winning_depth = -1
                                if attempt.primitive_trial_log:
                                    for entry in attempt.primitive_trial_log:
                                        if entry.get("success"):
                                            winning_edge_idx = int(entry.get("edge_idx", -1))
                                            winning_depth = int(entry.get("depth", -1))
                                            break
                                action_sequence = [{
                                    "object_id": attempt.chosen_object_id,
                                    "target": attempt.chosen_goal,
                                    "edge_idx": winning_edge_idx,
                                    "depth": winning_depth,
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
                                'region_goals_sampled': attempt.region_goals_sampled,
                                'chosen_object_id': attempt.chosen_object_id,
                                'goal_strategy_profile': getattr(attempt, 'goal_strategy_profile', None),
                                'chain_depth': attempt.chain_depth,
                                'total_cost': getattr(attempt, 'total_cost', None),
                                'skill_calls_before_success': getattr(attempt, 'skill_calls_before_success', None),
                                'solutions_found_for_neighbour': getattr(attempt, 'solutions_found_for_neighbour', None),
                                'solutions_cap_for_neighbour': getattr(attempt, 'solutions_cap_for_neighbour', None),
                                'solutions_total_for_neighbour': getattr(attempt, 'solutions_total_for_neighbour', None),
                                'pushes_total_for_neighbour': getattr(attempt, 'pushes_total_for_neighbour', None),
                                'failure_reason': getattr(attempt, 'failure_reason', None),
                                'candidate_objects_count': getattr(attempt, 'candidate_objects_count', None),
                                'ml_goals_generated': getattr(attempt, 'ml_goals_generated', None),
                                'ml_goals_aligned': getattr(attempt, 'ml_goals_aligned', None),
                                'ml_diffusion_calls': getattr(attempt, 'ml_diffusion_calls', None),
                                'ml_mask_vote_attach_calls': getattr(attempt, 'ml_mask_vote_attach_calls', None),
                                'ml_mask_vote_attach_ms_total': getattr(attempt, 'ml_mask_vote_attach_ms_total', None),
                                'ml_mask_vote_attach_ms_avg': getattr(attempt, 'ml_mask_vote_attach_ms_avg', None),
                                'reachable_edges_count': getattr(attempt, 'reachable_edges_count', None),
                                'primitive_ranking_calls': getattr(attempt, 'primitive_ranking_calls', None),
                                'primitive_ranking_ms_total': getattr(attempt, 'primitive_ranking_ms_total', None),
                                'primitive_ranking_ms_avg': getattr(attempt, 'primitive_ranking_ms_avg', None),
                                'primitive_ranking_candidates_total': getattr(attempt, 'primitive_ranking_candidates_total', None),
                                'primitive_ranking_candidates_avg': getattr(attempt, 'primitive_ranking_candidates_avg', None),
                                'push_exec_count': getattr(attempt, 'push_exec_count', None),
                                'push_exec_ms_total': getattr(attempt, 'push_exec_ms_total', None),
                                'push_exec_ms_avg': getattr(attempt, 'push_exec_ms_avg', None),
                                'push_exec_ms_by_depth': getattr(attempt, 'push_exec_ms_by_depth', None),
                                'goal_generation_calls': getattr(attempt, 'goal_generation_calls', None),
                                'goal_generation_ms_total': getattr(attempt, 'goal_generation_ms_total', None),
                                'goal_generation_ms_avg': getattr(attempt, 'goal_generation_ms_avg', None),
                                'opening_validation_calls': getattr(attempt, 'opening_validation_calls', None),
                                'opening_validation_ms_total': getattr(attempt, 'opening_validation_ms_total', None),
                                'opening_validation_ms_avg': getattr(attempt, 'opening_validation_ms_avg', None),
                                'opening_validation_goal_checks_total': getattr(attempt, 'opening_validation_goal_checks_total', None),
                                'opening_validation_goal_checks_avg_per_call': getattr(attempt, 'opening_validation_goal_checks_avg_per_call', None),
                                'opening_validation_reachability_calls': getattr(attempt, 'opening_validation_reachability_calls', None),
                                'opening_validation_reachability_ms_total': getattr(attempt, 'opening_validation_reachability_ms_total', None),
                                'opening_validation_reachability_ms_avg': getattr(attempt, 'opening_validation_reachability_ms_avg', None),
                                'chain_observation_replay_calls': getattr(attempt, 'chain_observation_replay_calls', None),
                                'chain_observation_replay_ms_total': getattr(attempt, 'chain_observation_replay_ms_total', None),
                                'chain_observation_replay_ms_avg': getattr(attempt, 'chain_observation_replay_ms_avg', None),
                                'aligned_primitives': getattr(attempt, 'aligned_primitives', None),
                                'ml_goals_raw': getattr(attempt, 'ml_goals_raw', None),
                                'reachable_edges': getattr(attempt, 'reachable_edges', None),
                                # Hybrid decomposition tracking
                                'phase_push_counts': getattr(attempt, 'phase_push_counts', None),
                                'solved_in_phase': getattr(attempt, 'solved_in_phase', ''),
                                # F characterization: per-primitive trial log (exhaustive mode only)
                                'primitive_trial_log': getattr(attempt, 'primitive_trial_log', None),
                                'reachability_log': getattr(attempt, 'reachability_log', None),
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
                            failure_description=attempt.error_message or "",
                            any_wall_collision=getattr(attempt, 'any_wall_collision', False),
                            unique_movable_collision_count=getattr(attempt, 'unique_movable_collision_count', 0),
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
                                "target": (action.x, action.y, action.theta),
                                "edge_idx": int(getattr(action, "edge_idx", -1)),
                                "depth": int(getattr(action, "depth", -1)),
                            }
                            for action in actions
                        ]
                        
                        # Apply solution smoothing if enabled
                        apply_solution_smoothing(
                            episode_result, env, original_action_sequence, states, post_states,
                            robot_goal, task
                        )

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
                                "target": (action.x, action.y, action.theta),
                                "edge_idx": int(getattr(action, "edge_idx", -1)),
                                "depth": int(getattr(action, "depth", -1)),
                            }
                            for action in planner_result.action_sequence
                        ]
                        
                        # Apply solution smoothing if enabled
                        apply_solution_smoothing(
                            episode_result, env, original_action_sequence,
                            planner_result.state_observations, planner_result.post_action_state_observations,
                            robot_goal, task
                        )

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
        base_dir_name = f"modular_data_{self.config.hostname}"
        run_suffix = None

        if self.config.run_name:
            run_suffix = _sanitize_run_name(self.config.run_name)
        elif self.config.unique_run_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_suffix = f"start{self.config.start_idx:06d}_end{self.config.end_idx:06d}_{timestamp}"

        if run_suffix:
            final_dir_name = f"{base_dir_name}_{run_suffix}"
        else:
            final_dir_name = base_dir_name

        self.output_dir = self.output_base / final_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"🗂️  Run directory: {self.output_dir}")
        
        # Setup progress tracking
        self.progress_file = self.output_dir / "collection_progress.txt"
    
    def create_tasks(self) -> List[ModularWorkerTask]:
        """Create worker tasks from environment file subset."""
        # Discover environment files (returns list of (xml_path, region_skip) tuples)
        env_entries = discover_environment_files(
            self.config.xml_base_dir,
            self.config.start_idx,
            self.config.end_idx,
            manifest_file=self.config.manifest_file
        )

        # Create tasks
        tasks = []
        for i, (xml_file, region_object_skip) in enumerate(env_entries):
            task_id = f"{self.config.hostname}_env_{self.config.start_idx + i:06d}"

            task_planner_config = self.config.planner_config
            if task_planner_config is not None:
                base_algorithm_params = task_planner_config.algorithm_params or {}
                task_algorithm_params = dict(base_algorithm_params)
                task_algorithm_params['xml_file'] = xml_file
                # Pass region_object_skip to planner (from manifest or None)
                if region_object_skip:
                    task_algorithm_params['region_object_skip'] = region_object_skip
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
                region_object_skip=region_object_skip
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
    parser.add_argument("--search-timeout", type=float, default=600.0,
                        help="Search timeout in seconds (default: 600.0 = 10 minutes)")
    parser.add_argument("--goals-per-region", type=int, default=100,
                        help="Number of robot goal samples per region for validation (default: 100)")

    # Region opening planner arguments (only those used by RegionOpeningPlanner)
    parser.add_argument("--region-allow-collisions", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow object collisions during region opening pushes (default: True). Use --no-region-allow-collisions for strict mode where any object collision aborts the push (intended for evaluation, not data collection). Robot-collisions always abort regardless.")
    parser.add_argument("--region-max-chain-depth", type=int, default=1,
                        help="Maximum chain depth for region opening: 1=single push, 2=2-push chains, 3=3-push chains (default: 1)")
    parser.add_argument("--region-max-solutions-per-neighbor", type=int, default=10,
                        help="Maximum solutions to keep per neighbor region (default: 10)")
    parser.add_argument("--region-max-recorded-solutions-per-neighbor", type=int, default=2,
                        help="Maximum solutions to record/save per neighbor (subset of found, default: 2)")
    parser.add_argument("--region-frontier-beam-width", type=int, default=None,
                        help="Optional beam width (K) to cap frontier per chain depth; None/<=0 disables")
    parser.add_argument("--region-chain-link-cost", type=int, default=0,
                        help="Additional cost per chain link beyond first push (default: 0)")
    parser.add_argument("--region-min-reachable-fraction", type=float, default=0.2,
                        help="Fraction of goal-region samples that must be reachable for an opening to count as success (default: 0.2). With dense disc samples on the goal site, 1.0 = full goal site reachable.")
    parser.add_argument("--region-ml-ignore-blacklist", action="store_true",
                        help="Allow ML-scored primitives to bypass edge blacklist")
    parser.add_argument("--region-selection-strategy", type=str, default="ml_first",
                        choices=["ml_first", "cost_first"],
                        help="Frontier priority: ml_first (ML-derived first) or cost_first (shallow first)")
    parser.add_argument("--profile-geometric", action="store_true",
                        help="Collect goal-strategy timing breakdown for geometric_transport (adds goal_strategy_profile to PKLs)")
    parser.add_argument("--goal-strategy", type=str, default=None,
                        choices=["primitive", "ml", "ml_primitive", "ml_fallback", "ml_primitive_fallback",
                                 "ml_async", "ml_primitive_async", "ml_driven_async",
                                 "geometric", "geometric_transport",
                                 "scorer", "f_scorer",
                                 "random_rollout", "random"],
                        help="Goal strategy for region opening (primitive default)")
    parser.add_argument("--scorer-ckpt", type=str, default=None,
                        help="Checkpoint for the 'scorer' goal strategy (defaults to champion sharp)")
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
    parser.add_argument("--primitive-prefix", type=str, default="",
                        help="Filename prefix for per-robot primitive calibration. "
                             "'' = motion_primitives_15_*.dat (30cm point-robot, legacy). "
                             "'car_' = car_motion_primitives_15_*.dat (7cm diff-drive car). "
                             "MUST match the robot in --config-file/--namo-config.")
    parser.add_argument("--shuffle-edges", action="store_true",
                        help="Randomize edge ordering in primitive strategy (useful for difficulty analysis)")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Random seed for reproducible edge shuffling (None = random each call)")
    parser.add_argument("--target-goal-region", action="store_true",
                        help="Region opening only attempts to open the XML's goal region "
                             "(label='goal' in the snapshot), not all unreachable neighbors. "
                             "Robot already in goal region → no episodes recorded. "
                             "Goal region not an immediate neighbor → one fail attempt with "
                             "failure_reason='target_not_immediate_neighbor' (phase-2 candidate).")
    parser.add_argument("--rollout-samples-per-state", type=int, default=None,
                        help="When --goal-strategy random_rollout is set, cap the number of "
                             "primitive candidates per state to K (random subset). Combined "
                             "with --region-max-chain-depth, gives thin random walks. "
                             "Default None = no cap (all ~600 candidates, just random order).")
    # ----------------- Uniform rollout sampler arguments -----------------
    parser.add_argument("--sampler-max-chain-depth", type=int, default=1, choices=[1],
                        help="v0 supports depth 0 only (max_chain_depth=1). "
                             "Deeper depths are a follow-up spec.")
    parser.add_argument("--sampler-region-goal-samples", type=int, default=5,
                        help="K points to sample per neighbor region for goal_sample_region mask "
                             "(stored in env_metadata.per_neighbor_region_goals).")
    parser.add_argument("--sampler-num-depths", type=int, default=10,
                        help="Number of push depths per edge (matches motion-primitive resolution).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global RNG seed for deterministic planning (default: NAMO_GLOBAL_SEED env var or 42)")
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
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional suffix appended to the per-host output directory to separate runs")
    parser.add_argument("--unique-run-dir", action="store_true",
                        help="Automatically append start/end indices and timestamp to output directory for each run")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to pre-generated manifest file for fast loading (use generate_xml_manifest.py to create)")

    # If YAML provided, load and set parser defaults before final parse
    if pre_args.config_yaml:
        try:
            import yaml  # Requires PyYAML
            with open(pre_args.config_yaml, 'r') as f:
                yaml_cfg = yaml.safe_load(f) or {}
            if not isinstance(yaml_cfg, dict):
                yaml_cfg = {}
            # Expand ${ENV} in string values so configs stay machine-portable
            # (e.g. xml_dir: ${NAMO_DATASETS}/car_envs). See python/namo/paths.py.
            yaml_cfg = {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in yaml_cfg.items()}
            # Only pass known keys; argparse will ignore unknown via set_defaults
            parser.set_defaults(**yaml_cfg)
        except Exception as e:
            print(f"⚠️  Warning: could not load YAML config '{pre_args.config_yaml}': {e}")

    args = parser.parse_args(remaining_argv)

    # Global RNG seed (affects region_opening planner and any random sampling).
    set_global_seed(args.seed if args.seed is not None else DEFAULT_GLOBAL_SEED)
    
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
            "region_chain_link_cost": args.region_chain_link_cost,
            "region_min_reachable_fraction": args.region_min_reachable_fraction,
            "primitive_prefix": args.primitive_prefix,
            "region_ml_ignore_blacklist": args.region_ml_ignore_blacklist,
            "region_selection_strategy": args.region_selection_strategy,
            "profile_geometric": args.profile_geometric,
            "region_exhaustive_mode": getattr(args, 'region_exhaustive_mode', False),
            # Beast LABEL mode: exhaustive setups + early-stop finish sweep + score/rank log + cost-prune off.
            # Set via --config-yaml (region_label_mode), same set_defaults path as region_exhaustive_mode.
            "region_label_mode": getattr(args, 'region_label_mode', False),
            # Reject 1-push-solvable roots before any depth-2 expansion while retaining a minimal
            # audit record. YAML-only collection knob; default False preserves existing runs.
            "region_stop_after_root_opener": getattr(args, 'region_stop_after_root_opener', False),
            # Enforced per-scene (per-neighbour) time budget so rich depth-2 scenes don't hog a worker.
            # Same set_defaults(yaml) path; must be in this dict to reach the planner.
            "region_timeout_per_neighbour_sec": getattr(args, 'region_timeout_per_neighbour_sec', None),
            "region_label_topk": getattr(args, 'region_label_topk', 0),
            # Round-2 EXHAUST-ON-MISS finish policy (set via --config-yaml, set_defaults path). MUST be
            # in this dict to reach the planner (set_defaults alone is inert — this bit us on
            # region_label_mode/region_timeout). 0 = off (use label_topk / plain early-stop).
            "region_exhaust_on_miss_topk": getattr(args, 'region_exhaust_on_miss_topk', 0),
            # horizon-Q sampled collection: uniform k-subset of (edge,depth) candidates per chain level
            # (0 = off). Set via --config-yaml (region_sample_k), like the exhaustive-mode keys.
            "region_sample_k": getattr(args, 'region_sample_k', 0),
            "region_sample_restarts": getattr(args, 'region_sample_restarts', 1),
            "shuffle_edges": args.shuffle_edges,
            "shuffle_seed": args.shuffle_seed,
            "target_goal_region": args.target_goal_region,
            "rollout_samples_per_state": args.rollout_samples_per_state,
        })
        # Optionally cap how many of the found solutions are recorded/saved per neighbor
        algorithm_params["region_max_recorded_solutions_per_neighbor"] = args.region_max_recorded_solutions_per_neighbor
        if args.region_frontier_beam_width is not None:
            algorithm_params["region_frontier_beam_width"] = args.region_frontier_beam_width

        if args.goal_strategy:
            algorithm_params["goal_strategy"] = args.goal_strategy
        if args.goal_strategy and args.goal_strategy.lower() in {"scorer", "f_scorer"}:
            # F-scorer goal ranking: renderer must use the SAME namo config as the env so the
            # scorer's crop matches its training distribution. scorer_ckpt optional (defaults to sharp).
            algorithm_params["namo_config_path"] = args.config_file
            # Honor --ml-device (default cuda) for the scorer too, so CPU-only SLURM nodes can run
            # scorer-guided collection (ScorerGoalStrategy else defaults the model to cuda -> crash).
            algorithm_params["ml_device"] = args.ml_device
            if getattr(args, "scorer_ckpt", None):
                algorithm_params["scorer_ckpt"] = args.scorer_ckpt
        if args.goal_strategy and args.goal_strategy.lower() in {"ml", "ml_primitive"}:
            if not args.ml_goal_model:
                parser.error("--ml-goal-model is required when goal strategy is 'ml'")
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
            # Allow users to specify ML params even without explicit strategy flag
            algorithm_params.update({
                "goal_strategy": "ml",
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

    if args.algorithm == "uniform_rollout_sampler":
        algorithm_params["max_chain_depth"] = args.sampler_max_chain_depth
        algorithm_params["region_goal_samples_per_neighbor"] = args.sampler_region_goal_samples
        algorithm_params["num_depths"] = args.sampler_num_depths
        algorithm_params["primitive_prefix"] = args.primitive_prefix
        algorithm_params["primitive_data_dir"] = args.primitive_data_dir
        algorithm_params["config_file_path"] = args.config_file
        algorithm_params["seed"] = args.seed if args.seed is not None else DEFAULT_GLOBAL_SEED

    planner_config = PlannerConfig(
        max_depth=args.max_depth,
        max_goals_per_object=args.max_goals_per_object,
        max_terminal_checks=args.max_terminal_checks,
        max_search_time_seconds=args.search_timeout,
        goals_per_region=args.goals_per_region,
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
        filter_minimum_length=args.filter_minimum_length,
        planner_config=planner_config,
        manifest_file=args.manifest,
        run_name=args.run_name,
        unique_run_dir=args.unique_run_dir
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
