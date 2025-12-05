#!/usr/bin/env python3
"""
Batch Mask Collection Pipeline for NAMO Data

This script processes directories of NAMO planning data (.pkl files) and generates
compressed mask datasets for machine learning. It filters for non-trivial successful
episodes and creates 224x224 masks for each valid episode.

================================================================================
USAGE EXAMPLES
================================================================================

FASTEST: NPZ then convert to HDF5 (recommended for large datasets):
    # Step 1: Generate NPZ files (fast parallel disk writes)
    python -m namo.visualization.mask_generation.batch_collection \\
        --input-dir /path/to/pkl/files \\
        --output-dir /path/to/npz \\
        --local-only \\
        --workers 48

    # Step 2: Convert to HDF5 (in sage_learning repo)
    python scripts/convert_to_hdf5.py /path/to/npz /path/to/data.h5

Direct HDF5 Output (slower due to IPC overhead):
    # Local masks only (for training with use_local: true)
    python -m namo.visualization.mask_generation.batch_collection \\
        --input-dir /path/to/pkl/files \\
        --output-dir /unused \\
        --hdf5 /path/to/output.h5 \\
        --local-only \\
        --workers 16

    # Global masks only
    python -m namo.visualization.mask_generation.batch_collection \\
        --input-dir /path/to/pkl/files \\
        --output-dir /unused \\
        --hdf5 /path/to/output.h5 \\
        --global-only \\
        --workers 16

    # Both global and local masks (largest output)
    python -m namo.visualization.mask_generation.batch_collection \\
        --input-dir /path/to/pkl/files \\
        --output-dir /unused \\
        --hdf5 /path/to/output.h5 \\
        --workers 16

NPZ Output (legacy, slower for training):
    python -m namo.visualization.mask_generation.batch_collection \\
        --input-dir /path/to/pkl/files \\
        --output-dir /path/to/output \\
        --workers 8

Serial mode (for debugging):
    python -m namo.visualization.mask_generation.batch_collection \\
        --input-dir /path/to/pkl/files \\
        --output-dir /unused \\
        --hdf5 /path/to/output.h5 \\
        --serial

================================================================================
COMMAND LINE OPTIONS
================================================================================

Required:
    --input-dir         Directory containing .pkl files from data collection
    --output-dir        Output directory for .npz files (ignored if --hdf5 is set)

Output format:
    --hdf5 PATH         Output to single HDF5 file (RECOMMENDED for 100k+ samples)
                        Much faster training startup vs many .npz files

Mask selection:
    --local-only        Generate only local (object-centered) masks
    --global-only       Generate only global masks
    (default)           Generate both global and local masks

Performance:
    --workers N         Number of parallel workers (default: auto-detect CPU count)
    --serial            Use single-threaded processing (for debugging)

Filtering:
    --filter-minimum-length   Only keep episodes with shortest action sequence per env
    --split-difficulty        Split outputs by difficulty (easy/medium/hard folders)

Other:
    --pattern GLOB      File pattern to match (default: *_results.pkl)
    --visualize         Enable visualization (slower)

================================================================================
GENERATED MASKS (224x224 each)
================================================================================

Global masks (--global-only or default):
    robot           Robot position mask
    goal            Robot goal position mask
    static          Static obstacles mask
    movable         Movable objects mask
    reachable       Reachable area mask
    target_object   The object being pushed
    target_goal     Where the object should go
    goal_region     Goal region mask

Local masks (--local-only or default):
    local_static             Static obstacles (object-centered crop)
    local_movable            Movable objects (object-centered crop)
    local_target_object      Target object (object-centered crop)
    local_target_goal        Target goal position (object-centered crop)
    local_robot_region       Robot reachability (BFS from robot position on inflated obstacles)
    local_goal_sample_region Goal sample reachability (BFS from first goal sample on inflated obstacles)

================================================================================
OUTPUT FORMAT
================================================================================

NPZ mode: output_dir/task_id/episode_id.npz
    - One file per training sample
    - Slow to load during training (100k file opens)

HDF5 mode: single .h5 file
    - All samples in one file
    - Fast training startup (single file handle)
    - Auto-detected by sage_learning data loader
    - Place as: data_dir.h5 next to data_dir/ folder

================================================================================
PROCESSING PIPELINE (HDF5 mode)
================================================================================

Step 1/2: Collecting valid episodes        (serial, fast - loads .pkl files)
Step 2/2: Generating masks & writing HDF5  (parallel generation, streaming writes)
          - Workers generate masks in parallel
          - Results streamed to HDF5 as they complete (low memory usage)
"""

import argparse
import os
import sys
import pickle
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from .visualizer import NAMODataVisualizer


class HDF5Writer:
    """Incremental HDF5 writer for streaming mask data."""

    def __init__(self, output_path: str, chunk_size: int = 1000, resize_increment: int = 10000):
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 output. Install with: pip install h5py")
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.resize_increment = resize_increment  # Pre-allocate this many slots at a time
        self.h5_file: Optional[h5py.File] = None
        self.datasets: Dict[str, h5py.Dataset] = {}
        self.current_idx = 0
        self.current_capacity = 0  # Track allocated capacity
        self.initialized = False

    def _init_datasets(self, masks: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        """Initialize HDF5 datasets based on first sample."""
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        self.h5_file = h5py.File(self.output_path, 'w')

        # Create resizable datasets for each mask
        for key, arr in masks.items():
            shape = (0,) + arr.shape
            maxshape = (None,) + arr.shape
            chunks = (self.chunk_size,) + arr.shape
            self.datasets[key] = self.h5_file.create_dataset(
                key, shape=shape, maxshape=maxshape, dtype=arr.dtype,
                chunks=chunks, compression='gzip', compression_opts=4
            )

        # Create string datasets for metadata (variable length)
        dt_str = h5py.special_dtype(vlen=str)
        for str_key in ['episode_id', 'task_id', 'algorithm', 'xml_file', 'difficulty_label']:
            self.datasets[str_key] = self.h5_file.create_dataset(
                str_key, shape=(0,), maxshape=(None,), dtype=dt_str
            )

        # Create numeric metadata datasets
        self.datasets['solution_depth'] = self.h5_file.create_dataset(
            'solution_depth', shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self.datasets['search_time_ms'] = self.h5_file.create_dataset(
            'search_time_ms', shape=(0,), maxshape=(None,), dtype=np.float32
        )
        self.datasets['nodes_expanded'] = self.h5_file.create_dataset(
            'nodes_expanded', shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self.datasets['robot_goal'] = self.h5_file.create_dataset(
            'robot_goal', shape=(0, 3), maxshape=(None, 3), dtype=np.float32
        )
        self.datasets['difficulty_score'] = self.h5_file.create_dataset(
            'difficulty_score', shape=(0,), maxshape=(None,), dtype=np.float32
        )

        self.initialized = True

    def add_sample(self, masks: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        """Add a single sample to the HDF5 file."""
        if not self.initialized:
            self._init_datasets(masks, metadata)

        # Resize and add mask data
        for key, arr in masks.items():
            if key in self.datasets:
                ds = self.datasets[key]
                ds.resize(self.current_idx + 1, axis=0)
                ds[self.current_idx] = arr

        # Add string metadata
        for str_key in ['episode_id', 'task_id', 'algorithm', 'xml_file']:
            if str_key in self.datasets:
                ds = self.datasets[str_key]
                ds.resize(self.current_idx + 1, axis=0)
                ds[self.current_idx] = metadata.get(str_key, '')

        # Add difficulty label
        if 'difficulty_label' in self.datasets:
            ds = self.datasets['difficulty_label']
            ds.resize(self.current_idx + 1, axis=0)
            ds[self.current_idx] = metadata.get('difficulty_label', 'unknown')

        # Add numeric metadata
        if 'solution_depth' in self.datasets:
            ds = self.datasets['solution_depth']
            ds.resize(self.current_idx + 1, axis=0)
            val = metadata.get('solution_depth')
            ds[self.current_idx] = val if val is not None else -1

        if 'search_time_ms' in self.datasets:
            ds = self.datasets['search_time_ms']
            ds.resize(self.current_idx + 1, axis=0)
            val = metadata.get('search_time_ms')
            ds[self.current_idx] = val if val is not None else -1.0

        if 'nodes_expanded' in self.datasets:
            ds = self.datasets['nodes_expanded']
            ds.resize(self.current_idx + 1, axis=0)
            val = metadata.get('nodes_expanded')
            ds[self.current_idx] = val if val is not None else -1

        if 'robot_goal' in self.datasets:
            ds = self.datasets['robot_goal']
            ds.resize(self.current_idx + 1, axis=0)
            ds[self.current_idx] = metadata.get('robot_goal', [0, 0, 0])

        if 'difficulty_score' in self.datasets:
            ds = self.datasets['difficulty_score']
            ds.resize(self.current_idx + 1, axis=0)
            val = metadata.get('difficulty_score')
            ds[self.current_idx] = val if val is not None else -1.0

        self.current_idx += 1

    def close(self):
        """Close the HDF5 file and store final sample count."""
        if self.h5_file is not None:
            self.h5_file.attrs['n_samples'] = self.current_idx
            self.h5_file.close()
            self.h5_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


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


def process_episode(episode: Dict[str, Any], visualizer: NAMODataVisualizer,
                    generate_local: bool = True,
                    local_only: bool = False,
                    local_crop_size: float = 5.0,
                    use_highres: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Process a single episode to generate masks and metadata.

    Args:
        episode: Episode data dictionary
        visualizer: NAMODataVisualizer instance
        generate_local: Whether to generate local object-centered masks (default: True)
        local_only: If True, only generate local masks (skip global) (default: False)
        local_crop_size: Size of local crop region in meters (default: 5.0)
        use_highres: Use unified highres rendering for both global and local (default: True)

    Returns:
        Tuple of (masks_dict, metadata_dict)
    """
    local_metadata = None

    if use_highres:
        # Use unified high-res rendering (renders once, creates both global and local)
        result = visualizer.generate_all_masks_highres(
            episode, local_crop_size_meters=local_crop_size
        )

        if local_only:
            # Only local masks
            if result['local'] is not None:
                masks = result['local']
                local_metadata = result['local_metadata']
            else:
                masks = {}  # No local masks available
        else:
            # Global masks, optionally with local
            masks = result['global']
            if generate_local and result['local'] is not None:
                masks.update(result['local'])
                local_metadata = result['local_metadata']
    else:
        # Legacy path: separate generation
        if local_only:
            masks = {}
            local_masks = visualizer.generate_local_episode_masks(
                episode, crop_size_meters=local_crop_size
            )
            if local_masks is not None:
                local_metadata = local_masks.pop('local_metadata', None)
                masks = local_masks
        else:
            if 'all_future_states' in episode and episode['all_future_states']:
                masks = visualizer.generate_episode_masks_multihorizon(episode)
            else:
                masks = visualizer.generate_episode_masks_batch(episode)

            if generate_local:
                local_masks = visualizer.generate_local_episode_masks(
                    episode, crop_size_meters=local_crop_size
                )
                if local_masks is not None:
                    local_metadata = local_masks.pop('local_metadata', None)
                    masks.update(local_masks)

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

    if 'difficulty_label' in episode:
        metadata['difficulty_label'] = episode.get('difficulty_label', 'unknown')
        metadata['difficulty_score'] = episode.get('difficulty_score')

    # Add local mask metadata if available
    if local_metadata is not None:
        metadata['local_metadata'] = local_metadata

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

    # Save local mask metadata if present
    local_meta = metadata.get('local_metadata')
    if local_meta is not None:
        save_dict['local_object_center'] = np.array(local_meta['object_center'], dtype=np.float32)
        save_dict['local_object_theta'] = np.array([local_meta['object_theta']], dtype=np.float32)
        save_dict['local_bounds'] = np.array(local_meta['local_bounds'], dtype=np.float32)
        save_dict['local_crop_size_meters'] = np.array([local_meta['crop_size_meters']], dtype=np.float32)
        save_dict['local_resolution'] = np.array([local_meta['resolution']], dtype=np.float32)
        save_dict['has_local_masks'] = np.array([True], dtype=bool)
    else:
        save_dict['has_local_masks'] = np.array([False], dtype=bool)

    # Save as compressed npz
    np.savez_compressed(output_path, **save_dict)


def process_pkl_file_worker(pkl_file: str, output_dir: str, filter_minimum_length: bool = False,
                            split_difficulty: bool = False,
                            generate_local: bool = True,
                            local_only: bool = False) -> Tuple[int, int, str]:
    """Worker function to process a single pickle file.

    This function is designed to be called by multiprocessing workers.
    Each worker gets its own NAMODataVisualizer instance to avoid sharing issues.

    Args:
        pkl_file: Path to pickle file
        output_dir: Base output directory
        filter_minimum_length: Whether to filter episodes by minimum action sequence length
        split_difficulty: Whether to compute difficulty labels and split outputs
        generate_local: Whether to generate local (object-centered) masks
        local_only: If True, only generate local masks (skip global)

    Returns:
        Tuple of (total_episodes, processed_episodes, pkl_file)
    """
    # Create visualizer instance for this worker
    visualizer = NAMODataVisualizer(figsize=(10, 8))

    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception:
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
                    masks, metadata = process_episode(
                        suffix_episode, visualizer,
                        generate_local=generate_local, local_only=local_only
                    )

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

            except Exception:
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


def _process_pkl_file_for_hdf5(args: Tuple[str, bool, bool, bool, bool]) -> List[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    """Worker function to process an entire pkl file for HDF5 output.

    Args:
        args: Tuple of (pkl_file, filter_minimum_length, split_difficulty, generate_local, local_only)

    Returns:
        List of (masks, metadata) tuples for all episodes in the file
    """
    pkl_file, filter_minimum_length, split_difficulty, generate_local, local_only = args
    results = []

    # One visualizer per pkl file (reused for all episodes in file)
    visualizer = NAMODataVisualizer(figsize=(10, 8))

    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return results

    episodes = data.get('episode_results', [])

    # Apply filtering
    filtered_episodes, _, _ = filter_episodes_by_minimum_length(episodes, filter_minimum_length)

    for episode in filtered_episodes:
        if is_valid_episode(episode):
            try:
                if split_difficulty:
                    assign_difficulty_annotation(episode)

                suffix_episodes = split_episode_into_trajectory_suffixes(episode)

                for suffix_episode in suffix_episodes:
                    masks, metadata = process_episode(
                        suffix_episode, visualizer,
                        generate_local=generate_local, local_only=local_only
                    )
                    if masks:
                        results.append((masks, metadata))
            except Exception:
                continue

    return results


def _collect_valid_episodes(pkl_files: List[str],
                            filter_minimum_length: bool) -> Tuple[List[Dict[str, Any]], int]:
    """Collect all valid episodes from pkl files.

    Returns:
        Tuple of (list of valid episodes, total episode count)
    """
    all_episodes = []
    total_episodes = 0

    for pkl_file in tqdm(pkl_files, desc="Loading pkl files"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            continue

        episodes = data.get('episode_results', [])
        total_episodes += len(episodes)

        filtered_episodes, _, _ = filter_episodes_by_minimum_length(
            episodes, filter_minimum_length)

        for episode in filtered_episodes:
            if is_valid_episode(episode):
                all_episodes.append(episode)

    return all_episodes, total_episodes


def process_to_hdf5(pkl_files: List[str], output_path: str,
                    filter_minimum_length: bool = False,
                    split_difficulty: bool = False,
                    num_workers: int = None,
                    generate_local: bool = True,
                    local_only: bool = False) -> Tuple[int, int]:
    """Process all pkl files and write directly to a single HDF5 file.

    Args:
        pkl_files: List of pickle file paths
        output_path: Output HDF5 file path
        filter_minimum_length: Whether to filter by minimum action sequence length
        split_difficulty: Whether to compute difficulty annotations
        num_workers: Number of parallel workers (None = auto-detect)
        generate_local: Whether to generate local (object-centered) masks
        local_only: If True, only generate local masks (skip global)

    Returns:
        Tuple of (total_episodes, processed_episodes)
    """
    if local_only:
        mask_desc = "local only"
    elif generate_local:
        mask_desc = "global + local"
    else:
        mask_desc = "global only"

    print(f"Processing {len(pkl_files)} pkl files -> HDF5")
    print(f"  Mask type: {mask_desc}")

    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = min(num_workers, len(pkl_files))
    print(f"  Using {num_workers} workers")

    # Prepare args for workers - process whole pkl files (like NPZ mode)
    worker_args = [
        (pkl_file, filter_minimum_length, split_difficulty, generate_local, local_only)
        for pkl_file in pkl_files
    ]

    total_processed = 0

    with HDF5Writer(output_path) as h5_writer:
        if num_workers == 1:
            # Serial processing
            for args in tqdm(pkl_files, desc="Processing pkl files"):
                results = _process_pkl_file_for_hdf5(
                    (args, filter_minimum_length, split_difficulty, generate_local, local_only)
                )
                for masks, metadata in results:
                    h5_writer.add_sample(masks, metadata)
                    total_processed += 1
        else:
            # Parallel processing - process pkl files in parallel (like NPZ mode)
            with mp.Pool(num_workers) as pool:
                with tqdm(total=len(pkl_files), desc="Processing pkl files") as pbar:
                    for results in pool.imap_unordered(_process_pkl_file_for_hdf5, worker_args, chunksize=1):
                        for masks, metadata in results:
                            h5_writer.add_sample(masks, metadata)
                            total_processed += 1
                        pbar.update(1)

    print(f"  Total samples written: {total_processed}")
    return len(pkl_files), total_processed


def main():
    parser = argparse.ArgumentParser(description='Batch NAMO mask collection pipeline')
    parser.add_argument('--input-dir', required=True, help='Directory containing .pkl files')
    parser.add_argument('--output-dir', required=True, help='Output directory for .npz files')
    parser.add_argument('--pattern', default='*_results.pkl', help='File pattern to match (default: *_results.pkl)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect CPU count)')
    parser.add_argument('--serial', action='store_true',
                       help='Use serial processing instead of parallel (for debugging)')
    parser.add_argument('--hdf5', type=str, default=None,
                       help='Output to single HDF5 file instead of many .npz files (much faster for training)')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization (slower)')
    parser.add_argument('--filter-minimum-length', action='store_true',
                       help='Only process episodes with minimum action sequence length per environment')
    parser.add_argument('--split-difficulty', action='store_true',
                       help='Split outputs into easy/medium/hard folders and store difficulty metadata')
    parser.add_argument('--local-only', action='store_true',
                       help='Generate only local (object-centered) masks, skip global masks')
    parser.add_argument('--global-only', action='store_true',
                       help='Generate only global masks, skip local masks')

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.local_only and args.global_only:
        print("Error: --local-only and --global-only are mutually exclusive")
        sys.exit(1)

    # Determine mask generation mode
    generate_local = not args.global_only  # True unless --global-only is set

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
        print("Minimum length filtering ENABLED - only episodes with shortest action sequences per environment will be processed")
    else:
        print("Minimum length filtering DISABLED - all valid episodes will be processed")

    # HDF5 output mode
    if args.hdf5:
        if not HAS_H5PY:
            print("Error: h5py required for HDF5 output. Install with: pip install h5py")
            sys.exit(1)

        # Determine number of workers
        if args.serial:
            num_workers = 1
        else:
            num_workers = args.workers if args.workers is not None else mp.cpu_count()

        mask_mode = "local only" if args.local_only else ("global only" if args.global_only else "global + local")
        print(f"Output mode: Single HDF5 file -> {args.hdf5}")
        print(f"Mask mode: {mask_mode}")
        print(f"Using {num_workers} workers for parallel mask generation")

        total_episodes, total_processed = process_to_hdf5(
            pkl_files, args.hdf5,
            args.filter_minimum_length, args.split_difficulty,
            num_workers=num_workers,
            generate_local=generate_local,
            local_only=args.local_only
        )

        # Print summary
        print(f"\n=== Processing Complete ===")
        print(f"Files processed: {len(pkl_files)}")
        print(f"Total episodes found: {total_episodes}")
        print(f"Valid episodes processed: {total_processed}")
        if total_episodes > 0:
            print(f"Success rate: {total_processed/total_episodes*100:.1f}%")
        print(f"Output HDF5 file: {args.hdf5}")

        # Report file size
        if os.path.exists(args.hdf5):
            size_gb = os.path.getsize(args.hdf5) / (1024**3)
            print(f"File size: {size_gb:.2f} GB")
        return

    # NPZ output mode (original behavior)
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine number of workers
    if args.serial:
        num_workers = 1
    else:
        num_workers = args.workers if args.workers is not None else mp.cpu_count()
        # Limit workers to avoid overwhelming the system
        num_workers = min(num_workers, len(pkl_files), mp.cpu_count())

    mask_mode = "local only" if args.local_only else ("global only" if args.global_only else "global + local")
    print(f"Using {num_workers} workers for processing")
    print(f"Mask mode: {mask_mode}")

    # Process all files
    total_episodes = 0
    total_processed = 0

    if num_workers == 1:
        # Serial processing (original behavior)
        visualizer = NAMODataVisualizer(figsize=(10, 8))
        for pkl_file in tqdm(pkl_files, desc="Processing files"):
            file_episodes, file_processed, _ = process_pkl_file_worker(
                pkl_file, args.output_dir,
                args.filter_minimum_length, args.split_difficulty,
                generate_local, args.local_only)
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
                split_difficulty=args.split_difficulty,
                generate_local=generate_local,
                local_only=args.local_only)

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
                        file_episodes, file_processed, _ = result.get()
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