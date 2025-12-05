#!/usr/bin/env python3
"""
Batch Mask Collection Pipeline for NAMO Data

This script processes directories of NAMO planning data (.pkl files) and generates
compressed mask datasets for machine learning. It filters for non-trivial successful
episodes and creates 224x224 masks for each valid episode.

Usage:
    python -m mask_generation.batch_collection --input-dir /path/to/pkl/files --output-dir /path/to/output --workers 8
    # or
    python batch_collection.py --input-dir /path/to/pkl/files --output-dir /path/to/output --workers 8
    
    # For debugging (single-threaded)
    python -m mask_generation.batch_collection --input-dir /path/to/pkl/files --output-dir /path/to/output --serial

Generated Masks (9 total, 224x224 each):
- Binary masks: robot, goal, movable, static, reachable, target_object, target_goal
- Distance fields: robot_distance, goal_distance (with cost model: 1 for free, 4 for movable)

Output Format:
- Compressed .npz files: output_dir/task_id/episode_id.npz
- Each .npz contains all 9 masks plus metadata
"""

import argparse
import os
import sys
import pickle
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from .visualizer import NAMODataVisualizer


def is_valid_episode(episode: Dict[str, Any]) -> bool:
    """Check if episode is valid for mask generation.
    
    Args:
        episode: Episode data dictionary
        
    Returns:
        True if episode should be processed (non-trivial successful episode)
    """
    # Must have found a solution
    if not episode.get('solution_found', False):
        return False
    
    # Must have at least one action (non-trivial)
    action_sequence = episode.get('action_sequence', [])
    if not action_sequence or len(action_sequence) == 0:
        return False
    
    # Must have state observations
    state_observations = episode.get('state_observations', [])
    if not state_observations:
        return False
    
    return True


def filter_episodes_by_minimum_length(episodes: List[Dict[str, Any]], 
                                    filter_minimum_length: bool = False) -> Tuple[List[Dict[str, Any]], int, int]:
    """Filter episodes to keep only those with minimum action sequence length per environment.
    
    Args:
        episodes: List of episode dictionaries
        filter_minimum_length: Whether to apply minimum length filtering
        
    Returns:
        Tuple of (filtered_episodes, episodes_before_filtering, episodes_filtered_out)
    """
    episodes_before_filtering = len(episodes)
    
    if not filter_minimum_length:
        return episodes, episodes_before_filtering, 0
    
    # Group episodes by task_id (environment)
    task_groups = {}
    for episode in episodes:
        episode_id = episode.get('episode_id', '')
        if '_episode_' in episode_id:
            task_id = episode_id.split('_episode_')[0]
        else:
            # Fallback: use filename prefix
            task_id = 'unknown_task'
        
        if task_id not in task_groups:
            task_groups[task_id] = []
        task_groups[task_id].append(episode)
    
    # Filter each group to keep only minimum length episodes
    filtered_episodes = []
    for task_id, task_episodes in task_groups.items():
        # Find valid successful episodes with action sequences
        valid_episodes = [ep for ep in task_episodes if is_valid_episode(ep)]
        
        if not valid_episodes:
            # No valid episodes in this task, skip
            continue
        
        # Find minimum action sequence length
        min_length = min(len(ep.get('action_sequence', [])) for ep in valid_episodes)
        
        # Keep only episodes with minimum length
        for episode in valid_episodes:
            action_sequence = episode.get('action_sequence', [])
            if len(action_sequence) == min_length:
                filtered_episodes.append(episode)
    
    episodes_filtered_out = episodes_before_filtering - len(filtered_episodes)
    return filtered_episodes, episodes_before_filtering, episodes_filtered_out


def split_episode_into_trajectory_suffixes(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split a multi-step episode into trajectory suffix training examples.

    For an n-push episode with states [S0, S1, ..., Sn-1] and actions [A0, A1, ..., An-1],
    creates n training examples:
      - (S0, [A0, A1, ..., An-1])
      - (S1, [A1, A2, ..., An-1])
      - ...
      - (Sn-1, [An-1])

    Args:
        episode: Original episode data

    Returns:
        List of episode dictionaries, one per step
    """
    action_sequence = episode.get('action_sequence', [])
    state_observations = episode.get('state_observations', [])
    post_action_state_observations = episode.get('post_action_state_observations', [])
    reachable_before = episode.get('reachable_objects_before_action', [])
    reachable_after = episode.get('reachable_objects_after_action', [])

    n_steps = len(action_sequence)

    # Single-step episode - return as-is
    if n_steps <= 1:
        return [episode]

    # Multi-step episode - create trajectory suffixes
    suffix_episodes = []
    base_episode_id = episode.get('episode_id', '')

    for step_i in range(n_steps):
        # Create new episode for this step
        suffix_episode = episode.copy()

        # Update episode_id to indicate step
        suffix_episode['episode_id'] = f"{base_episode_id}_step_{step_i}"

        # Use state at step i
        suffix_episode['state_observations'] = [state_observations[step_i]] if step_i < len(state_observations) else state_observations[-1:]
        suffix_episode['post_action_state_observations'] = [post_action_state_observations[step_i]] if step_i < len(post_action_state_observations) else post_action_state_observations[-1:]

        # Use reachable objects at step i
        if reachable_before and step_i < len(reachable_before):
            suffix_episode['reachable_objects_before_action'] = [reachable_before[step_i]]
        if reachable_after and step_i < len(reachable_after):
            suffix_episode['reachable_objects_after_action'] = [reachable_after[step_i]]

        # Use remaining actions from step i onwards
        suffix_episode['action_sequence'] = action_sequence[step_i:]

        # Update solution depth to reflect remaining actions
        suffix_episode['solution_depth'] = len(action_sequence[step_i:])

        # NEW: Store ALL remaining states for multi-horizon goal mask generation
        # all_future_states[0] = current state Si (before action i)
        # all_future_states[1] = state Si+1 (after action i)
        # all_future_states[2] = state Si+2 (after action i+1)
        # ... etc
        all_future_states = [state_observations[step_i]] if step_i < len(state_observations) else []
        # Add all post-action states from step_i onwards
        if post_action_state_observations:
            all_future_states.extend(post_action_state_observations[step_i:])

        suffix_episode['all_future_states'] = all_future_states

        suffix_episodes.append(suffix_episode)

    return suffix_episodes


def assign_difficulty_annotation(episode: Dict[str, Any]) -> None:
    """Annotate an episode with difficulty score and label."""
    stats = episode.get('algorithm_stats') or {}
    pushes = stats.get('pushes_total_for_neighbour')
    solutions = stats.get('solutions_total_for_neighbour')

    score = None
    if pushes is not None and pushes > 0 and solutions is not None:
        score = float(solutions) / float(pushes)

    if score is None:
        label = 'unknown'
    elif score > 0.9:
        label = 'easy'
    elif score > 0.1:
        label = 'medium'
    else:
        label = 'hard'

    episode['difficulty_score'] = score
    episode['difficulty_label'] = label


def process_episode(episode: Dict[str, Any], visualizer: NAMODataVisualizer) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Process a single episode to generate masks and metadata.

    Args:
        episode: Episode data dictionary
        visualizer: NAMODataVisualizer instance

    Returns:
        Tuple of (masks_dict, metadata_dict)
    """
    # Generate masks with multi-horizon goal predictions
    # If episode has 'all_future_states', use multihorizon generation
    # Otherwise fall back to standard batch generation
    if 'all_future_states' in episode and episode['all_future_states']:
        masks = visualizer.generate_episode_masks_multihorizon(episode)
    else:
        masks = visualizer.generate_episode_masks_batch(episode)

    # Extract metadata
    metadata = {
        'episode_id': episode.get('episode_id', ''),
        'task_id': episode.get('episode_id', '').split('_episode_')[0] if '_episode_' in episode.get('episode_id', '') else '',
        'algorithm': episode.get('algorithm', ''),
        'solution_depth': episode.get('solution_depth'),
        'search_time_ms': episode.get('search_time_ms'),
        'nodes_expanded': episode.get('nodes_expanded'),
        'action_sequence': episode.get('action_sequence', []),
        'robot_goal': episode.get('robot_goal', [0, 0, 0]),
        'xml_file': episode.get('xml_file', '')
    }

    # --- Add Robot State (Pose + Radius) ---
    state_observations = episode.get('state_observations', [])
    robot_pose = [0.0, 0.0, 0.0]
    if state_observations and len(state_observations) > 0:
        # Use final state (same as used for masks)
        final_state = state_observations[-1]
        if 'robot_pose' in final_state:
            robot_pose = final_state['robot_pose']
    
    # Combine pose [x, y, theta] and radius [r] into single array [x, y, theta, r]
    robot_radius = 0.15  # Constant used in visualizer
    metadata['robot_state'] = np.array(list(robot_pose) + [robot_radius], dtype=np.float32)

    # --- Add Target Object Pose and Corners ---
    # Extract environment info to get bounds and object sizes
    env_info = visualizer._extract_env_info_from_episode(episode)
    static_object_info = episode.get('static_object_info') or {}
    action_sequence = episode.get('action_sequence', [])
    
    target_object_id = None
    if action_sequence and len(action_sequence) > 0:
        target_object_id = action_sequence[0].get('object_id')
    
    if target_object_id and state_observations:
        final_state = state_observations[-1]
        # Find object pose in final state
        # Note: object_id might be "obstacle_1_movable" but state key is "obstacle_1_movable_pose"
        # Or sometimes they match. Visualizer logic handles suffix stripping.
        
        # Try to find the object pose
        obj_pose = None
        obj_base_name = target_object_id
        
        # Check direct match or with _pose suffix
        for key, pose in final_state.items():
            if key == target_object_id or key == f"{target_object_id}_pose":
                obj_pose = pose
                obj_base_name = key.replace('_pose', '')
                break
        
        if obj_pose is not None:
            metadata['target_object_pose'] = obj_pose
            
            # Get object size
            obj_info = static_object_info.get(obj_base_name, {})
            if 'size_x' in obj_info and 'size_y' in obj_info:
                size_x = obj_info['size_x']
                size_y = obj_info['size_y']
                theta = obj_pose[2]
                
                # Compute corners
                corners_world = visualizer.compute_box_corners_world(
                    obj_pose[0], obj_pose[1], size_x, size_y, theta)
                corners_px = visualizer.compute_box_corners_pixel(
                    obj_pose[0], obj_pose[1], size_x, size_y, theta, env_info.world_bounds)
                
                metadata['target_object_corners_world'] = corners_world
                metadata['target_object_corners_px'] = corners_px

    # --- Add Target Goal Poses, Corners, and Deltas (Multi-horizon) ---
    if action_sequence and target_object_id:
        # Get object size (needed for goal corners)
        obj_base_name = target_object_id
        obj_info = static_object_info.get(obj_base_name, {})
        
        if 'size_x' in obj_info and 'size_y' in obj_info:
            size_x = obj_info['size_x']
            size_y = obj_info['size_y']
            
            current_obj_pose = metadata.get('target_object_pose')
            current_obj_corners_w = metadata.get('target_object_corners_world')
            current_obj_corners_px = metadata.get('target_object_corners_px')
            
            # Initialize lists for per-action data
            target_goal_poses = []
            target_goal_corners_world = []
            target_goal_corners_px = []
            target_goal_pose_deltas_world = []
            target_goal_pose_deltas_obj = []
            target_goal_corner_deltas_world = []
            target_goal_corner_deltas_px = []
            target_goal_corner_deltas_obj = []
            
            for action in action_sequence:
                target_pose = action.get('target')
                if target_pose and len(target_pose) >= 3:
                    target_goal_poses.append(target_pose)
                    
                    # Compute corners for goal pose
                    g_corners_w = visualizer.compute_box_corners_world(
                        target_pose[0], target_pose[1], size_x, size_y, target_pose[2])
                    g_corners_px = visualizer.compute_box_corners_pixel(
                        target_pose[0], target_pose[1], size_x, size_y, target_pose[2], env_info.world_bounds)
                    
                    target_goal_corners_world.append(g_corners_w)
                    target_goal_corners_px.append(g_corners_px)
                    
                    # Compute deltas if we have current object pose
                    if current_obj_pose is not None:
                        # Pose Deltas
                        d_pose_world, d_pose_obj = visualizer.compute_pose_deltas(target_pose, current_obj_pose)
                        target_goal_pose_deltas_world.append(d_pose_world)
                        target_goal_pose_deltas_obj.append(d_pose_obj)
                        
                        # Corner Deltas
                        if current_obj_corners_w is not None:
                            # World frame corner delta
                            d_corners_world = g_corners_w - current_obj_corners_w
                            target_goal_corner_deltas_world.append(d_corners_world.astype(np.float32))
                            
                            # Object frame corner delta: Rotate world delta by -obj_theta
                            # R(-theta) = [[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]]
                            theta = current_obj_pose[2]
                            c, s = np.cos(-theta), np.sin(-theta)
                            R_inv = np.array([[c, -s], [s, c]])
                            d_corners_obj = d_corners_world @ R_inv.T
                            target_goal_corner_deltas_obj.append(d_corners_obj.astype(np.float32))
                        else:
                            target_goal_corner_deltas_world.append(np.zeros((4,2), dtype=np.float32))
                            target_goal_corner_deltas_obj.append(np.zeros((4,2), dtype=np.float32))
                            
                        if current_obj_corners_px is not None:
                            # Pixel frame corner delta
                            d_corners_px = g_corners_px - current_obj_corners_px
                            target_goal_corner_deltas_px.append(d_corners_px.astype(np.int32))
                        else:
                            target_goal_corner_deltas_px.append(np.zeros((4,2), dtype=np.int32))
                            
                    else:
                        # Fallback if object pose missing
                        target_goal_pose_deltas_world.append(np.zeros(3, dtype=np.float32))
                        target_goal_pose_deltas_obj.append(np.zeros(3, dtype=np.float32))
                        target_goal_corner_deltas_world.append(np.zeros((4,2), dtype=np.float32))
                        target_goal_corner_deltas_px.append(np.zeros((4,2), dtype=np.int32))
                        target_goal_corner_deltas_obj.append(np.zeros((4,2), dtype=np.float32))
            
            # Store in metadata
            if target_goal_poses:
                metadata['target_goal_poses'] = np.array(target_goal_poses, dtype=np.float32)
                metadata['target_goal_corners_world'] = np.array(target_goal_corners_world, dtype=np.float32)
                metadata['target_goal_corners_px'] = np.array(target_goal_corners_px, dtype=np.int32)
                metadata['target_goal_pose_deltas_world'] = np.array(target_goal_pose_deltas_world, dtype=np.float32)
                metadata['target_goal_pose_deltas_obj'] = np.array(target_goal_pose_deltas_obj, dtype=np.float32)
                metadata['target_goal_corner_deltas_world'] = np.array(target_goal_corner_deltas_world, dtype=np.float32)
                metadata['target_goal_corner_deltas_px'] = np.array(target_goal_corner_deltas_px, dtype=np.int32)
                metadata['target_goal_corner_deltas_obj'] = np.array(target_goal_corner_deltas_obj, dtype=np.float32)

    if 'difficulty_label' in episode:
        metadata['difficulty_label'] = episode.get('difficulty_label', 'unknown')
        metadata['difficulty_score'] = episode.get('difficulty_score')

    return masks, metadata


def save_episode_data(masks: Dict[str, np.ndarray], metadata: Dict[str, Any], 
                     output_path: str) -> None:
    """Save episode masks and metadata to compressed npz file.
    
    Args:
        masks: Dictionary of mask arrays
        metadata: Episode metadata dictionary
        output_path: Output file path (.npz)
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Combine masks and metadata for saving
    save_dict = dict(masks)  # Copy masks
    
    # Add metadata as separate fields (avoiding object arrays)
    save_dict['episode_id'] = np.array([metadata['episode_id']], dtype='U')
    save_dict['task_id'] = np.array([metadata['task_id']], dtype='U')
    save_dict['algorithm'] = np.array([metadata['algorithm']], dtype='U')

    # Handle None values explicitly (region opening planner sets these to None)
    solution_depth = metadata.get('solution_depth')
    save_dict['solution_depth'] = np.array([solution_depth if solution_depth is not None else -1], dtype=np.int32)

    search_time = metadata.get('search_time_ms')
    save_dict['search_time_ms'] = np.array([search_time if search_time is not None else -1.0], dtype=np.float32)

    nodes_expanded = metadata.get('nodes_expanded')
    save_dict['nodes_expanded'] = np.array([nodes_expanded if nodes_expanded is not None else -1], dtype=np.int32)

    save_dict['robot_goal'] = np.array(metadata.get('robot_goal', [0, 0, 0]), dtype=np.float32)
    save_dict['xml_file'] = np.array([metadata.get('xml_file', '')], dtype='U')

    if 'difficulty_label' in metadata:
        save_dict['difficulty_label'] = np.array([metadata.get('difficulty_label', 'unknown')], dtype='U')
        score = metadata.get('difficulty_score')
        save_dict['difficulty_score'] = np.array([score if score is not None else -1.0], dtype=np.float32)

    # Count number of goal mask horizons (goal_mask_a1, goal_mask_a2, etc.)
    num_goal_horizons = sum(1 for key in masks.keys() if key.startswith('goal_mask_a'))
    save_dict['num_goal_horizons'] = np.array([num_goal_horizons], dtype=np.int32)

    # Save action sequence as separate arrays for object_ids and targets
    action_seq = metadata.get('action_sequence', [])
    if action_seq:
        object_ids = [action.get('object_id', '') for action in action_seq]
        targets = [action.get('target', [0, 0, 0]) for action in action_seq]
        save_dict['action_object_ids'] = np.array(object_ids, dtype='U')
        save_dict['action_targets'] = np.array(targets, dtype=np.float32)
    else:
        save_dict['action_object_ids'] = np.array([], dtype='U')
        save_dict['action_targets'] = np.array([[]], dtype=np.float32)
    
    # --- Save New Metadata Fields ---
    # Robot State (Pose + Radius)
    if 'robot_state' in metadata:
        save_dict['robot_state'] = np.array(metadata['robot_state'], dtype=np.float32)
    
    # Target Object Pose and Corners
    if 'target_object_pose' in metadata:
        save_dict['target_object_pose'] = np.array(metadata['target_object_pose'], dtype=np.float32)
    
    if 'target_object_corners_world' in metadata:
        save_dict['target_object_corners_world'] = np.array(metadata['target_object_corners_world'], dtype=np.float32)
        
    if 'target_object_corners_px' in metadata:
        save_dict['target_object_corners_px'] = np.array(metadata['target_object_corners_px'], dtype=np.int32)
        
    # Target Goal Poses, Corners, and Deltas (Multi-horizon)
    if 'target_goal_poses' in metadata:
        save_dict['target_goal_poses'] = np.array(metadata['target_goal_poses'], dtype=np.float32)
        
    if 'target_goal_corners_world' in metadata:
        save_dict['target_goal_corners_world'] = np.array(metadata['target_goal_corners_world'], dtype=np.float32)
        
    if 'target_goal_corners_px' in metadata:
        save_dict['target_goal_corners_px'] = np.array(metadata['target_goal_corners_px'], dtype=np.int32)
        
    if 'target_goal_pose_deltas_world' in metadata:
        save_dict['target_goal_pose_deltas_world'] = np.array(metadata['target_goal_pose_deltas_world'], dtype=np.float32)
        
    if 'target_goal_pose_deltas_obj' in metadata:
        save_dict['target_goal_pose_deltas_obj'] = np.array(metadata['target_goal_pose_deltas_obj'], dtype=np.float32)

    if 'target_goal_corner_deltas_world' in metadata:
        save_dict['target_goal_corner_deltas_world'] = np.array(metadata['target_goal_corner_deltas_world'], dtype=np.float32)

    if 'target_goal_corner_deltas_px' in metadata:
        save_dict['target_goal_corner_deltas_px'] = np.array(metadata['target_goal_corner_deltas_px'], dtype=np.int32)

    if 'target_goal_corner_deltas_obj' in metadata:
        save_dict['target_goal_corner_deltas_obj'] = np.array(metadata['target_goal_corner_deltas_obj'], dtype=np.float32)

    # Save as compressed npz
    np.savez_compressed(output_path, **save_dict)


def process_pkl_file_worker(pkl_file: str, output_dir: str, filter_minimum_length: bool = False,
                            split_difficulty: bool = False) -> Tuple[int, int, str]:
    """Worker function to process a single pickle file.
    
    This function is designed to be called by multiprocessing workers.
    Each worker gets its own NAMODataVisualizer instance to avoid sharing issues.
    
    Args:
        pkl_file: Path to pickle file
        output_dir: Base output directory
        filter_minimum_length: Whether to filter episodes by minimum action sequence length
        split_difficulty: Whether to compute difficulty labels and split outputs
        
    Returns:
        Tuple of (total_episodes, processed_episodes, pkl_file)
    """
    # Create visualizer instance for this worker
    visualizer = NAMODataVisualizer(figsize=(10, 8))
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return 0, 0, pkl_file
    
    episodes = data.get('episode_results', [])
    total_episodes = len(episodes)
    
    # Apply minimum length filtering if requested
    filtered_episodes, _, _ = filter_episodes_by_minimum_length(episodes, filter_minimum_length)
    
    processed_episodes = 0
    
    for episode in filtered_episodes:
        if is_valid_episode(episode):
            try:
                if split_difficulty:
                    assign_difficulty_annotation(episode)

                # Split multi-step episodes into trajectory suffix examples
                suffix_episodes = split_episode_into_trajectory_suffixes(episode)

                # Process each suffix as a separate training example
                for suffix_episode in suffix_episodes:
                    # Generate masks and metadata
                    masks, metadata = process_episode(suffix_episode, visualizer)

                    # Create output path: output_dir/task_id/episode_id.npz
                    task_id = metadata['task_id']
                    episode_id = metadata['episode_id']
                    base_dir = output_dir
                    if split_difficulty:
                        label = metadata.get('difficulty_label', 'unknown') or 'unknown'
                        base_dir = os.path.join(base_dir, label)
                    output_path = os.path.join(base_dir, task_id, f"{episode_id}.npz")

                    # Save data
                    save_episode_data(masks, metadata, output_path)
                    processed_episodes += 1

            except Exception as e:
                # Suppress individual episode errors for cleaner parallel output
                continue
    
    return total_episodes, processed_episodes, pkl_file


def process_pkl_file(pkl_file: str, visualizer: NAMODataVisualizer, 
                    output_dir: str, filter_minimum_length: bool = False,
                    split_difficulty: bool = False) -> Tuple[int, int]:
    """Legacy single-threaded processing function for compatibility.
    
    Args:
        pkl_file: Path to pickle file
        visualizer: NAMODataVisualizer instance
        output_dir: Base output directory
        filter_minimum_length: Whether to filter episodes by minimum action sequence length
        split_difficulty: Whether to compute difficulty labels and split outputs
        
    Returns:
        Tuple of (total_episodes, processed_episodes)
    """
    total_episodes, processed_episodes, _ = process_pkl_file_worker(
        pkl_file, output_dir, filter_minimum_length, split_difficulty)
    return total_episodes, processed_episodes


def main():
    parser = argparse.ArgumentParser(description='Batch NAMO mask collection pipeline')
    parser.add_argument('--input-dir', required=True, help='Directory containing .pkl files')
    parser.add_argument('--output-dir', required=True, help='Output directory for .npz files')
    parser.add_argument('--pattern', default='*_results.pkl', help='File pattern to match (default: *_results.pkl)')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of parallel workers (default: auto-detect CPU count)')
    parser.add_argument('--serial', action='store_true', 
                       help='Use serial processing instead of parallel (for debugging)')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization (slower)')
    parser.add_argument('--filter-minimum-length', action='store_true',
                       help='Only process episodes with minimum action sequence length per environment')
    parser.add_argument('--split-difficulty', action='store_true',
                       help='Split outputs into easy/medium/hard folders and store difficulty metadata')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Find all pickle files - support recursive pattern from run_mask_generation.py
    if '**' in args.pattern:
        pkl_files = glob.glob(os.path.join(args.input_dir, args.pattern), recursive=True)
    else:
        pkl_files = glob.glob(os.path.join(args.input_dir, args.pattern))
    
    if not pkl_files:
        print(f"Error: No files found matching pattern: {os.path.join(args.input_dir, args.pattern)}")
        sys.exit(1)
    
    print(f"Found {len(pkl_files)} pickle files to process")
    if args.filter_minimum_length:
        print("🔍 Minimum length filtering ENABLED - only episodes with shortest action sequences per environment will be processed")
    else:
        print("🔍 Minimum length filtering DISABLED - all valid episodes will be processed")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine number of workers
    if args.serial:
        num_workers = 1
    else:
        num_workers = args.workers if args.workers is not None else mp.cpu_count()
        # Limit workers to avoid overwhelming the system
        num_workers = min(num_workers, len(pkl_files), mp.cpu_count())
    
    print(f"Using {num_workers} workers for processing")
    
    # Process all files
    total_episodes = 0
    total_processed = 0
    
    if num_workers == 1:
        # Serial processing (original behavior)
        visualizer = NAMODataVisualizer(figsize=(10, 8))
        for pkl_file in tqdm(pkl_files, desc="Processing files"):
            file_episodes, file_processed = process_pkl_file(
                pkl_file, visualizer, args.output_dir,
                args.filter_minimum_length, args.split_difficulty)
            total_episodes += file_episodes
            total_processed += file_processed
    else:
        # Parallel processing
        print("Starting parallel processing...")
        
        with mp.Pool(num_workers) as pool:
            # Create partial function with fixed output_dir and filter setting
            worker_func = partial(
                process_pkl_file_worker,
                output_dir=args.output_dir,
                filter_minimum_length=args.filter_minimum_length,
                split_difficulty=args.split_difficulty)
            
            # Process files with progress bar
            results = []
            with tqdm(total=len(pkl_files), desc="Processing files") as pbar:
                # Submit all jobs
                for pkl_file in pkl_files:
                    result = pool.apply_async(worker_func, (pkl_file,))
                    results.append(result)
                
                # Collect results as they complete
                for result in results:
                    try:
                        file_episodes, file_processed, pkl_file = result.get()
                        total_episodes += file_episodes
                        total_processed += file_processed
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing file: {e}")
                        pbar.update(1)
    
    # Print summary statistics
    print(f"\n=== Processing Complete ===")
    print(f"Files processed: {len(pkl_files)}")
    print(f"Total episodes found: {total_episodes}")
    print(f"Valid episodes processed: {total_processed}")
    if total_episodes > 0:
        print(f"Success rate: {total_processed/total_episodes*100:.1f}%")
    else:
        print("Success rate: 0.0%")
    print(f"Output directory: {args.output_dir}")
    print(f"Generated {total_processed} compressed .npz files")


if __name__ == "__main__":
    main()