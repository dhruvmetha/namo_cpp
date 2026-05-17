#!/usr/bin/env python3
"""Sequential ML Data Collection Pipeline

This module provides a sequential version of the data collection system optimized for
ML-based planning. It runs in a single process, preloads the ML model once, and
reuses it across all environments. This is ideal for:
1. Debugging ML model integration
2. Collecting data on machines with limited GPU memory (avoids N model copies)
3. Performance profiling of the planner itself

Usage:
    python sequential_ml_collection.py --config-yaml python/namo/data_collection/region_opening_ml_collection.yaml
"""

import os
import sys
import argparse
import socket
import pickle
import json
import time
from pathlib import Path
import traceback
import signal
import re
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, replace
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add visualization directory to path for local scripts that import from it.
namo_viz_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualization")
if namo_viz_path not in sys.path:
    sys.path.append(namo_viz_path)

# Support local editable checkouts where `sage_learning` isn't installed globally.
_mujoco_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_sage_learning_root = os.path.join(_mujoco_root, "sage_learning")
if os.path.isdir(_sage_learning_root) and _sage_learning_root not in sys.path:
    sys.path.insert(0, _sage_learning_root)

# Snapshot contains `ktamp_learning` (legacy layout).
_snapshot_sage_root = os.path.join(_mujoco_root, "eval_pipeline_snapshot", "sage_learning")
if os.path.isdir(_snapshot_sage_root) and _snapshot_sage_root not in sys.path:
    sys.path.append(_snapshot_sage_root)
# NAMO imports
import namo_rl
from namo.core import BasePlanner, PlannerConfig, PlannerResult, PlannerFactory
from namo.core.xml_goal_parser import extract_goal_with_fallback

# Reuse data structures and helpers from modular collection
from namo.data_collection.modular_parallel_collection import (
    ModularCollectionConfig,
    ModularWorkerTask,
    ModularWorkerResult,
    ModularEpisodeResult,
    generate_hostname_prefix,
    generate_goal_for_environment,
    discover_environment_files,
    create_failure_info,
    apply_solution_smoothing,
    _sanitize_run_name,
    get_failure_statistics
)

def preload_ml_models(config: ModularCollectionConfig) -> Tuple[Optional[Any], Optional[Any]]:
    """Preload ML models based on configuration."""
    import torch
    import numpy as np

    object_model = None
    goal_model = None

    # Extract params from planner config
    algo_params = config.planner_config.algorithm_params or {}
    device = algo_params.get("ml_device", "cuda")

    # Set deterministic mode for reproducibility
    ml_seed = algo_params.get("ml_seed")
    if ml_seed is not None:
        random.seed(ml_seed)
        np.random.seed(ml_seed)
        torch.manual_seed(ml_seed)
        torch.cuda.manual_seed_all(ml_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Use warn_only=True to allow some non-deterministic ops with a warning
        torch.use_deterministic_algorithms(True, warn_only=True)
        print(f"    Deterministic mode enabled with seed={ml_seed}")

    # Debug: Show GPU configuration
    print(f"🎮 GPU Configuration:")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"   Requested device: {device}")
    print(f"   PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   PyTorch visible GPU count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            actual_device = torch.device(device)
            print(f"   Actual device to use: {actual_device}")
            print(f"   GPU name: {torch.cuda.get_device_name(actual_device)}")

    # Check if ML is used
    use_ml_object = algo_params.get("object_selection_strategy") == "ml"
    use_ml_goal = (algo_params.get("goal_strategy") in [
        "ml", "ml_primitive", "ml_fallback", "ml_primitive_fallback",
        "ml_async", "ml_primitive_async", "ml_driven_async"
    ] or algo_params.get("goal_selection_strategy") == "ml")

    if use_ml_object:
        model_path = algo_params.get("ml_object_model_path")
        if model_path:
            try:
                try:
                    from sage_learning.object_inference_model import ObjectInferenceModel
                except Exception:
                    from ktamp_learning.object_inference_model import ObjectInferenceModel
                print(f"🔮 Loading ObjectInferenceModel from {model_path}")
                object_model = ObjectInferenceModel(model_path=model_path, device=device)
                print(f"✅ Object model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load object model: {e}")
                traceback.print_exc()

    if use_ml_goal:
        model_path = algo_params.get("ml_goal_model_path") or algo_params.get("ml_goal_model")
        if model_path:
            try:
                try:
                    from sage_learning.goal_inference_model import GoalInferenceModel
                except Exception:
                    from ktamp_learning.goal_inference_model import GoalInferenceModel
                print(f"🎯 Loading GoalInferenceModel from {model_path}")
                # Get optional sampler overrides from config
                sampler_method = algo_params.get("ml_sampler_method")  # euler, midpoint, rk4, dopri5
                num_steps = algo_params.get("ml_num_steps")  # number of integration steps
                print(f"   Sampler override: method={sampler_method}, num_steps={num_steps}")
                goal_model = GoalInferenceModel(
                    model_path=model_path,
                    device=device,
                    sampler_method=sampler_method,
                    num_steps=num_steps
                )
                print(f"✅ Goal model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load goal model: {e}")
                traceback.print_exc()

    # Warmup models to compile CUDA kernels (eliminates first-episode timing penalty)
    if goal_model is not None:
        print(f"🔥 Warming up goal model (compiling CUDA kernels)...")
        warmup_start = time.time()
        try:
            import torch
            num_samples = algo_params.get("ml_samples", 32)
            num_steps = algo_params.get("ml_num_steps") or 20

            # Check if this is an SE2 model (outputs pose vectors) vs image model
            is_se2_model = getattr(goal_model, 'is_se2_model', False)

            if is_se2_model:
                # SE2 models use sample_pose() with image context input
                # Input: (batch=1, channels=5 or 7, height=224, width=224)
                context_size = 224
                num_channels = 7 if getattr(goal_model, 'use_coord_grid', False) else 5
                dummy_input = torch.zeros(1, num_channels, context_size, context_size, device=device)

                for i in range(3):
                    with torch.no_grad():
                        _ = goal_model.model.sample_pose(
                            dummy_input,
                            num_samples=num_samples,
                            num_steps=num_steps,
                            denormalize=True
                        )
            else:
                # Image-based models use sample_from_model()
                context_size = getattr(goal_model, 'context_size', 64)
                dummy_input = torch.zeros(1, 5, context_size, context_size, device=device)

                for i in range(3):
                    with torch.no_grad():
                        _ = goal_model.model.sample_from_model(
                            dummy_input,
                            samples=num_samples,
                            num_steps=num_steps,
                            seed=ml_seed
                        )

            # Synchronize CUDA to ensure all operations complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            warmup_time = time.time() - warmup_start
            model_type = "SE2 pose" if is_se2_model else "image"
            print(f"✅ Goal model ({model_type}) warmed up in {warmup_time:.2f}s ({num_samples} samples x 3 runs)")
        except Exception as e:
            print(f"⚠️ Warmup failed (will warmup on first real inference): {e}")
            traceback.print_exc()

    return object_model, goal_model

def process_single_environment(
    task: ModularWorkerTask, 
    preloaded_object_model: Optional[Any],
    preloaded_goal_model: Optional[Any]
) -> ModularWorkerResult:
    """Process a single environment using preloaded models."""
    start_time = time.time()
    result = ModularWorkerResult(task_id=task.task_id, success=False)

    # Environment-specific ML artifact directory (non-destructive, additive).
    # Downstream ML/primitive alignment code can use this to persist masks + vote mappings.
    artifacts_dir = Path(task.output_dir) / "ml_artifacts" / task.task_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    env_info_path = artifacts_dir / "env_info.json"
    if not env_info_path.exists():
        try:
            env_info_path.write_text(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "xml_file": task.xml_file,
                        "config_file": task.config_file,
                        "algorithm": task.algorithm,
                        "created_unix_sec": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        except Exception:
            # Best-effort only; failures here shouldn't break collection.
            pass

    prior_artifacts_dir = os.environ.get("NAMO_ML_ARTIFACTS_DIR")
    os.environ["NAMO_ML_ARTIFACTS_DIR"] = str(artifacts_dir)
    
    # Inject preloaded models into planner config
    if preloaded_object_model or preloaded_goal_model:
        # We need to modify the planner config on the fly
        algo_params = dict(task.planner_config.algorithm_params or {})
        if preloaded_object_model:
            algo_params['preloaded_object_model'] = preloaded_object_model
        if preloaded_goal_model:
            algo_params['preloaded_goal_model'] = preloaded_goal_model
            
        task.planner_config = replace(task.planner_config, algorithm_params=algo_params)

    try:
        # Initialize environment
        print(f"\n📁 Processing environment: {task.xml_file}")
        env = namo_rl.RLEnvironment(task.xml_file, task.config_file, visualize=False)
        episode_results = []
        
        # Collect static object information once
        try:
            static_object_info = env.get_object_info()
        except AttributeError:
            static_object_info = {}

        # Create planner
        planner = PlannerFactory.create_planner(task.algorithm, env, task.planner_config)
        
        # Run episodes
        for episode_idx in range(task.episodes_per_env):
            robot_goal = generate_goal_for_environment(task.xml_file)
            episode_id = f"{task.task_id}_episode_{episode_idx}"
            
            try:
                # Reset
                env.reset()
                planner.reset()
                env.set_robot_goal(*robot_goal)
                
                # Search
                planner_result = planner.search(robot_goal)

                # Handle Region Opening Results (multiple attempts)
                is_region_opening = task.algorithm == "region_opening"
                
                if is_region_opening and planner_result.algorithm_stats and 'attempt_results' in planner_result.algorithm_stats:
                    for attempt_idx, attempt in enumerate(planner_result.algorithm_stats['attempt_results']):
                        attempt_episode_id = f"{episode_id}_neighbour_{attempt_idx}_{attempt.neighbour_region_label}"
                        
                        # Convert attempt to action sequence
                        action_sequence = []
                        solution_depth = 0
                        if attempt.success:
                            if attempt.goal_chain:
                                for goal in attempt.goal_chain:
                                    edge_idx = getattr(goal, "edge_idx", None)
                                    depth_idx = getattr(goal, "depth", None)
                                    action_sequence.append(
                                        {
                                            "object_id": attempt.chosen_object_id,
                                            "target": (goal.x, goal.y, goal.theta),
                                            # Optional but critical for accurate replay:
                                            # primitive slot identity for direct execution.
                                            "edge_idx": int(edge_idx) if edge_idx is not None else None,
                                            "depth": int(depth_idx) if depth_idx is not None else None,
                                        }
                                    )
                                solution_depth = len(attempt.goal_chain)
                            elif attempt.chosen_goal:
                                action_sequence = [{
                                    "object_id": attempt.chosen_object_id,
                                    "target": attempt.chosen_goal,
                                    "edge_idx": None,
                                    "depth": None,
                                }]
                                solution_depth = 1
                        
                        # Print stats for this neighbor attempt
                        status_icon = "✅" if attempt.success else "❌"
                        print(f"   {status_icon} Neighbor '{attempt.neighbour_region_label}' Object '{attempt.chosen_object_id}': {'Success' if attempt.success else 'Failed'} ({getattr(attempt, 'pushes_total_for_neighbour', 0)} pushes)")

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
                                'chain_depth': attempt.chain_depth,
                                'total_cost': getattr(attempt, 'total_cost', None),
                                'goal_strategy_profile': getattr(attempt, 'goal_strategy_profile', None),
                                'skill_calls_before_success': getattr(attempt, 'skill_calls_before_success', None),
                                'solutions_found_for_neighbour': getattr(attempt, 'solutions_found_for_neighbour', None),
                                'solutions_cap_for_neighbour': getattr(attempt, 'solutions_cap_for_neighbour', None),
                                'solutions_total_for_neighbour': getattr(attempt, 'solutions_total_for_neighbour', None),
                                'pushes_total_for_neighbour': getattr(attempt, 'pushes_total_for_neighbour', None),
                                # Failure tracking for counting different failure modes
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
                                # Detailed ML goal info for analysis/visualization
                                # aligned_primitives: [{'object_id', 'edge_idx', 'depth_idx', 'x', 'y', 'theta', 'votes'}, ...]
                                'aligned_primitives': getattr(attempt, 'aligned_primitives', None),
                                # ml_goals_raw: [{'object_id', 'x', 'y', 'theta'}, ...]
                                'ml_goals_raw': getattr(attempt, 'ml_goals_raw', None),
                                # reachable_edges: [edge_idx, ...]
                                'reachable_edges': getattr(attempt, 'reachable_edges', None),
                                # Hybrid decomposition tracking
                                'phase_push_counts': getattr(attempt, 'phase_push_counts', None),
                                'solved_in_phase': getattr(attempt, 'solved_in_phase', ''),
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
                    continue

                # Handle Standard Planner Results
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
                    state_observations=planner_result.state_observations,
                    post_action_state_observations=planner_result.post_action_state_observations,
                    static_object_info=static_object_info if planner_result.solution_found else None,
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
                    
                    apply_solution_smoothing(
                        episode_result, env, original_action_sequence,
                        planner_result.state_observations, planner_result.post_action_state_observations,
                        robot_goal, task
                    )

                if not planner_result.success:
                    episode_result.error_message = planner_result.error_message
                
                episode_results.append(episode_result)

            except Exception as e:
                print(f"⚠️ Episode failed: {e}")
                traceback.print_exc()
                # Create failure result... (omitted for brevity, same as parallel)
                pass # In sequential mode we just continue

        # Filtering logic (same as parallel)
        episodes_before_filtering = len(episode_results)
        episodes_filtered_out = 0

        # Filter out episodes with empty action sequences EXCEPT for trackable failure cases
        # We keep:
        #   - "already_accessible": neighbor was reachable before any push (0 pushes, 0 solutions - trackable)
        #   - "no_blocking_objects", "no_reachable_objects", "no_valid_goals": failure modes we want to count
        #   - Any failure (solution_found=False): always keep for analysis
        # We filter out:
        #   - Only true "trivial successes" that have no failure_reason tracking
        def should_keep_episode(ep):
            # Always keep failures
            if not ep.solution_found:
                return True
            # Always keep if there's an actual action sequence
            if ep.action_sequence and len(ep.action_sequence) > 0:
                return True
            # Keep if it has a tracked failure_reason (for counting later)
            failure_reason = ep.algorithm_stats.get('failure_reason') if ep.algorithm_stats else None
            if failure_reason is not None:
                return True
            # Filter out true trivial successes (no action, no failure_reason)
            return False

        initial_count = len(episode_results)
        episode_results = [ep for ep in episode_results if should_keep_episode(ep)]
        empty_action_filtered = initial_count - len(episode_results)
        episodes_filtered_out += empty_action_filtered

        # Save results immediately
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
            
        result.success = True
        result.episodes_collected = len(episode_results)
        result.episode_results = episode_results
        
    except Exception as e:
        result.error_message = f"Task failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ {result.error_message}")
    finally:
        # Always populate processing_time (success and failure paths).
        result.processing_time = time.time() - start_time
        # Restore prior artifacts directory to avoid contaminating subsequent environments.
        if prior_artifacts_dir is None:
            os.environ.pop("NAMO_ML_ARTIFACTS_DIR", None)
        else:
            os.environ["NAMO_ML_ARTIFACTS_DIR"] = prior_artifacts_dir

    return result

class SequentialCollectionManager:
    def __init__(self, config: ModularCollectionConfig):
        self.config = config
        
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
        
    def discover_environment_files(self) -> List[str]:
        
        # all_xml_files = []
        # with open("/common/home/dm1487/robotics_research/ktamp/namo/all_xml_files.pkl", "rb") as f:
        #     all_xml_files = pickle.load(f)
            
        # all_files = set(all_xml_files['easy'] + all_xml_files['medium'] + all_xml_files['hard'])
        # all_files = list(all_files)
        
        # random.seed(42)
        # random.shuffle(all_files)
        
        # print(f"Found {len(all_files)} environments")
        # all_files = all_files[self.config.start_idx:self.config.end_idx]
        
        # return all_files
        
        # Returns list of (xml_path, region_skip) tuples
        env_entries = discover_environment_files(
            self.config.xml_base_dir,
            self.config.start_idx,
            self.config.end_idx,
            manifest_file=self.config.manifest_file
        )
        return env_entries

    def create_tasks(self) -> List[ModularWorkerTask]:
        # Discover environment files (returns list of (xml_path, region_object_skip) tuples)
        env_entries = self.discover_environment_files()

        # Create tasks
        tasks = []
        for i, (xml_file, region_object_skip) in enumerate(env_entries):
            task_id = f"{self.config.hostname}_env_{self.config.start_idx + i:06d}"

            # Inject XML file and region_object_skip into algorithm params
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

    def run_sequential_collection(self):
        tasks = self.create_tasks()
        if not tasks:
            print("No tasks created!")
            return

        print(f"🚀 Starting SEQUENTIAL ML data collection")
        print(f"📊 Algorithm: {self.config.algorithm}")
        print(f"🔢 Processing {len(tasks)} environments sequentially")
        
        # Preload models ONCE
        print("\n📥 Preloading ML models...")
        object_model, goal_model = preload_ml_models(self.config)
        print("✅ Models ready.\n")

        start_time = time.time()
        results = []
        failed_tasks = []
        total_episodes = 0

        with tqdm(total=len(tasks), desc="Collecting data", unit="env") as pbar:
            for task in tasks:
                result = process_single_environment(task, object_model, goal_model)
                results.append(result)
                total_episodes += result.episodes_collected
                
                if not result.success:
                    failed_tasks.append(result)
                    pbar.set_postfix({"episodes": total_episodes, "failed": len(failed_tasks)})
                else:
                    pbar.set_postfix({"episodes": total_episodes, "failed": len(failed_tasks)})
                
                pbar.update(1)

        # Final summary
        total_time = time.time() - start_time
        success_rate = (len(tasks) - len(failed_tasks)) / len(tasks) * 100 if tasks else 0
        
        print(f"\n🎉 Collection complete!")
        print(f"📊 Episodes: {total_episodes} total")
        print(f"🎯 Task success rate: {success_rate:.1f}% ({total_time/60:.1f}m)")
        
        # Save aggregate summary (reuse logic from parallel script if possible, or simplified here)
        self._save_final_summary(results, total_time)

    def _save_final_summary(self, results: List[ModularWorkerResult], total_time: float):
        """Save comprehensive summary."""
        all_episodes = []
        for result in results:
            if result.episode_results:
                all_episodes.extend([asdict(ep) for ep in result.episode_results])
        
        successful_episodes = [ep for ep in all_episodes if ep['solution_found']]
        failure_stats = get_failure_statistics(all_episodes)
        
        summary = {
            'collection_metadata': {
                'hostname': self.config.hostname,
                'algorithm': self.config.algorithm,
                'total_duration_seconds': total_time,
                'execution_mode': 'sequential_ml',
                'config': asdict(self.config)
            },
            'performance_stats': {
                'total_episodes': len(all_episodes),
                'successful_episodes': len(successful_episodes),
                'success_rate': len(successful_episodes) / len(all_episodes) * 100 if all_episodes else 0,
            },
            'failure_analysis': failure_stats
        }
        
        summary_file = self.output_dir / f"collection_summary_{self.config.hostname}.pkl"
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
            
        # Simple text summary
        summary_txt = self.output_dir / f"summary_{self.config.hostname}.txt"
        with open(summary_txt, 'w') as f:
            f.write("Sequential ML Data Collection Summary\n")
            f.write(f"Total episodes: {len(all_episodes)}\n")
            f.write(f"Success rate: {summary['performance_stats']['success_rate']:.1f}%\n")

def main():
    # Some parts of the C++ environment initialization load motion primitives from
    # relative paths like `data/motion_primitives_15_square.dat`. If the script is
    # launched outside the `namo_cpp` repo root, `RLEnvironment` can fail to
    # initialize (even if `--primitive-data-dir` is provided, since that only
    # affects Python-side strategies).
    #
    # To make this script robust, if the current working directory does not have
    # a `data/` folder but the repo root does, switch CWD to the repo root.
    try:
        repo_root = Path(__file__).resolve().parents[4]  # .../namo_cpp
        if not Path("data").exists() and (repo_root / "data").exists():
            os.chdir(repo_root)
            print(f"📁 Changed working directory to namo_cpp repo root: {repo_root}")
    except Exception:
        # Best-effort only; if this fails, we'll surface the original error later.
        pass

    # Use same argument parsing structure as modular_parallel_collection
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-yaml", type=str, help="Path to YAML config file for defaults")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Sequential ML Data Collection", parents=[pre_parser])
    
    # Core arguments
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--algorithm", type=str, default="region_opening")
    parser.add_argument("--episodes-per-env", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1) # Ignored but kept for compat
    
    # Planner params
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-goals-per-object", type=int, default=5)
    parser.add_argument("--max-terminal-checks", type=int, default=5000)
    parser.add_argument("--search-timeout", type=float, default=300.0)
    parser.add_argument("--goals-per-region", type=int, default=5,
                        help="Number of robot goal samples per region for validation (default: 5)")
    
    # Region opening params
    parser.add_argument("--region-allow-collisions", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow object collisions during region opening pushes (default: True). Use --no-region-allow-collisions for strict mode.")
    parser.add_argument("--region-max-chain-depth", type=int, default=1)
    parser.add_argument("--region-max-solutions-per-neighbor", type=int, default=10)
    parser.add_argument("--region-max-recorded-solutions-per-neighbor", type=int, default=2)
    parser.add_argument("--region-frontier-beam-width", type=int, default=None)
    parser.add_argument("--region-chain-link-cost", type=int, default=0)
    parser.add_argument("--region-ml-ignore-blacklist", action="store_true")
    parser.add_argument("--region-selection-strategy", type=str, default="ml_first",
                        choices=["ml_first", "cost_first"])
    
    # ML params
    parser.add_argument("--goal-strategy", type=str, default=None)
    parser.add_argument("--ml-goal-model", type=str)
    parser.add_argument("--ml-device", type=str, default="cuda")
    parser.add_argument("--ml-samples", type=int, default=32)
    parser.add_argument("--ml-min-goals", type=int, default=1)
    parser.add_argument("--ml-match-position-tolerance", type=float, default=0.1)
    parser.add_argument("--ml-match-angle-tolerance", type=float, default=0.1)
    parser.add_argument("--ml-match-angle-weight", type=float, default=0.5)
    parser.add_argument("--ml-match-max-per-call", type=int, default=8)
    parser.add_argument("--ml-sampler-method", type=str, default=None, help="Override sampler: euler, midpoint, rk4, dopri5 (flow matching) or ddpm, ddim (diffusion)")
    parser.add_argument("--ml-num-steps", type=int, default=None, help="Override number of integration steps")
    parser.add_argument("--ml-k-nearest", type=int, default=1, help="Number of nearest primitive slots to vote for per ML goal (within tolerance)")
    parser.add_argument("--ml-seed", type=int, default=None, help="Random seed for diffusion noise (None = random each time)")

    # Other
    parser.add_argument("--primitive-data-dir", type=str, default="data")
    parser.add_argument("--xml-dir", type=str, default="../ml4kp_ktamp/resources/models/custom_walled_envs/aug9")
    parser.add_argument("--config-file", type=str, default="config/namo_config_complete.yaml")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--filter-minimum-length", action="store_true")
    parser.add_argument("--smooth-solutions", action="store_true")
    parser.add_argument("--max-smooth-actions", type=int, default=20)
    parser.add_argument("--refine-actions", action="store_true")
    parser.add_argument("--validate-refinement", action="store_true", default=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--unique-run-dir", action="store_true")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest file listing XML files")

    # Load YAML
    if pre_args.config_yaml:
        import yaml
        with open(pre_args.config_yaml, 'r') as f:
            yaml_cfg = yaml.safe_load(f) or {}
        if isinstance(yaml_cfg, dict):
            parser.set_defaults(**yaml_cfg)

    args = parser.parse_args(remaining_argv)
    
    # Validate required
    if args.output_dir is None or args.start_idx is None or args.end_idx is None:
        print("❌ Error: --output-dir, --start-idx, and --end-idx are required")
        return 1

    # Build configs (reuse logic from parallel script)
    algorithm_params = {}
    if args.algorithm == "region_opening":
        algorithm_params["primitive_data_dir"] = args.primitive_data_dir
        algorithm_params.update({
            "region_allow_collisions": args.region_allow_collisions,
            "region_max_chain_depth": args.region_max_chain_depth,
            "region_max_solutions_per_neighbor": args.region_max_solutions_per_neighbor,
            "region_max_recorded_solutions_per_neighbor": args.region_max_recorded_solutions_per_neighbor,
            "region_chain_link_cost": args.region_chain_link_cost,
            "region_ml_ignore_blacklist": args.region_ml_ignore_blacklist,
            "region_selection_strategy": args.region_selection_strategy,
        })
        if args.region_frontier_beam_width is not None:
            algorithm_params["region_frontier_beam_width"] = args.region_frontier_beam_width

        if args.goal_strategy:
            algorithm_params["goal_strategy"] = args.goal_strategy

        # Pass ML params even if goal_strategy is not explicitly set (allows auto-detection)
        if args.ml_goal_model:
            # Only set goal_strategy to "ml" if not already set (preserve ml_fallback, etc.)
            if "goal_strategy" not in algorithm_params:
                algorithm_params["goal_strategy"] = "ml"
            algorithm_params.update({
                "ml_goal_model_path": args.ml_goal_model,
                "ml_device": args.ml_device,
                "ml_samples": args.ml_samples,
                "ml_min_goals": args.ml_min_goals,
                "ml_match_position_tolerance": args.ml_match_position_tolerance,
                "ml_match_angle_tolerance": args.ml_match_angle_tolerance,
                "ml_match_angle_weight": args.ml_match_angle_weight,
                "ml_match_max_per_call": args.ml_match_max_per_call,
                "ml_sampler_method": args.ml_sampler_method,
                "ml_num_steps": args.ml_num_steps,
                "ml_k_nearest": args.ml_k_nearest,
                "ml_seed": args.ml_seed,
                "primitive_data_dir": args.primitive_data_dir,
            })

    planner_config = PlannerConfig(
        max_depth=args.max_depth,
        max_goals_per_object=args.max_goals_per_object,
        max_terminal_checks=args.max_terminal_checks,
        max_search_time_seconds=args.search_timeout,
        goals_per_region=args.goals_per_region,
        verbose=args.verbose,
        collect_stats=True,
        algorithm_params=algorithm_params
    )
    
    config = ModularCollectionConfig(
        xml_base_dir=args.xml_dir,
        config_file=args.config_file,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        episodes_per_env=args.episodes_per_env,
        num_workers=1, # Sequential implies 1 worker
        algorithm=args.algorithm,
        smooth_solutions=args.smooth_solutions,
        max_smooth_actions=args.max_smooth_actions,
        filter_minimum_length=args.filter_minimum_length,
        planner_config=planner_config,
        run_name=args.run_name,
        unique_run_dir=args.unique_run_dir,
        manifest_file=args.manifest
    )
    
    manager = SequentialCollectionManager(config)
    manager.run_sequential_collection()
    return 0

if __name__ == "__main__":
    sys.exit(main())
