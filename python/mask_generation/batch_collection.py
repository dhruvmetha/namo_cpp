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


def process_episode(episode: Dict[str, Any], visualizer: NAMODataVisualizer) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Process a single episode to generate masks and metadata.
    
    Args:
        episode: Episode data dictionary
        visualizer: NAMODataVisualizer instance
        
    Returns:
        Tuple of (masks_dict, metadata_dict)
    """
    # Generate 9 masks (excluding combined distance field)
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
    save_dict['solution_depth'] = np.array([metadata.get('solution_depth', -1)], dtype=np.int32)
    save_dict['search_time_ms'] = np.array([metadata.get('search_time_ms', -1.0)], dtype=np.float32)
    save_dict['nodes_expanded'] = np.array([metadata.get('nodes_expanded', -1)], dtype=np.int32)
    save_dict['robot_goal'] = np.array(metadata.get('robot_goal', [0, 0, 0]), dtype=np.float32)
    save_dict['xml_file'] = np.array([metadata.get('xml_file', '')], dtype='U')
    
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
    
    # Save as compressed npz
    np.savez_compressed(output_path, **save_dict)


def process_pkl_file_worker(pkl_file: str, output_dir: str, filter_minimum_length: bool = False) -> Tuple[int, int, str]:
    """Worker function to process a single pickle file.
    
    This function is designed to be called by multiprocessing workers.
    Each worker gets its own NAMODataVisualizer instance to avoid sharing issues.
    
    Args:
        pkl_file: Path to pickle file
        output_dir: Base output directory
        filter_minimum_length: Whether to filter episodes by minimum action sequence length
        
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
                # Generate masks and metadata
                masks, metadata = process_episode(episode, visualizer)
                
                # Create output path: output_dir/task_id/episode_id.npz
                task_id = metadata['task_id']
                episode_id = metadata['episode_id']
                output_path = os.path.join(output_dir, task_id, f"{episode_id}.npz")
                
                # Save data
                save_episode_data(masks, metadata, output_path)
                processed_episodes += 1
                
            except Exception as e:
                # Suppress individual episode errors for cleaner parallel output
                continue
    
    return total_episodes, processed_episodes, pkl_file


def process_pkl_file(pkl_file: str, visualizer: NAMODataVisualizer, 
                    output_dir: str, filter_minimum_length: bool = False) -> Tuple[int, int]:
    """Legacy single-threaded processing function for compatibility.
    
    Args:
        pkl_file: Path to pickle file
        visualizer: NAMODataVisualizer instance
        output_dir: Base output directory
        filter_minimum_length: Whether to filter episodes by minimum action sequence length
        
    Returns:
        Tuple of (total_episodes, processed_episodes)
    """
    total_episodes, processed_episodes, _ = process_pkl_file_worker(pkl_file, output_dir, filter_minimum_length)
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
            file_episodes, file_processed = process_pkl_file(pkl_file, visualizer, args.output_dir, args.filter_minimum_length)
            total_episodes += file_episodes
            total_processed += file_processed
    else:
        # Parallel processing
        print("Starting parallel processing...")
        
        with mp.Pool(num_workers) as pool:
            # Create partial function with fixed output_dir and filter setting
            worker_func = partial(process_pkl_file_worker, output_dir=args.output_dir, filter_minimum_length=args.filter_minimum_length)
            
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