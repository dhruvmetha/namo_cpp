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
import json
import math
import os
import re
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

from sage_learning.visualizer import NAMODataVisualizer

# Global list to collect skipped episodes (for multiprocessing)
_skipped_episodes_lock = None
_skipped_episodes_file = None


def has_region_overlap(masks: Dict[str, np.ndarray]) -> bool:
    """Check if robot_region and goal_sample_region have any overlapping pixels.

    Args:
        masks: Dictionary of mask arrays

    Returns:
        True if there is overlap (regions are connected), False otherwise
    """
    # Check local masks first (preferred for local-only mode)
    robot_key = 'local_robot_region' if 'local_robot_region' in masks else 'robot_region'
    goal_key = 'local_goal_sample_region' if 'local_goal_sample_region' in masks else 'goal_sample_region'

    if robot_key not in masks or goal_key not in masks:
        return False  # Can't check, assume no overlap

    robot_region = masks[robot_key]
    goal_region = masks[goal_key]

    # Check for overlap (both masks have value > 0.5 at same pixel)
    overlap = np.logical_and(robot_region > 0.5, goal_region > 0.5)
    return np.any(overlap)


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

        # Solution counts for sample weighting
        self.datasets['solutions_found'] = self.h5_file.create_dataset(
            'solutions_found', shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self.datasets['solutions_total'] = self.h5_file.create_dataset(
            'solutions_total', shape=(0,), maxshape=(None,), dtype=np.int32
        )
        self.datasets['pushes_total'] = self.h5_file.create_dataset(
            'pushes_total', shape=(0,), maxshape=(None,), dtype=np.int32
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

        # Add solution counts for sample weighting
        if 'solutions_found' in self.datasets:
            ds = self.datasets['solutions_found']
            ds.resize(self.current_idx + 1, axis=0)
            val = metadata.get('solutions_found')
            ds[self.current_idx] = val if val is not None else -1

        if 'solutions_total' in self.datasets:
            ds = self.datasets['solutions_total']
            ds.resize(self.current_idx + 1, axis=0)
            val = metadata.get('solutions_total')
            ds[self.current_idx] = val if val is not None else -1

        if 'pushes_total' in self.datasets:
            ds = self.datasets['pushes_total']
            ds.resize(self.current_idx + 1, axis=0)
            val = metadata.get('pushes_total')
            ds[self.current_idx] = val if val is not None else -1

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


# --- DEAD-END rendering (horizon-Q H0b, opt-in via --include-dead-ends) -------------------------
# A dead-end = tried-and-all-failed episode (primitive_trial_log non-empty, no success). These carry
# NO state_observations and NO action_sequence, so the default path drops them — but the budget-Q
# VALUE head must see them to learn "low"/unsolvable (horizon_q_build_journal.md §9, task #23).

def _is_dead_end_episode(episode: Dict[str, Any]) -> bool:
    """Tried-and-all-failed (H0b dead-end): has primitive push trials, none succeeded."""
    if episode.get('solution_found', False):
        return False
    log = (episode.get('algorithm_stats') or {}).get('primitive_trial_log') or []
    return bool(log) and not any(t.get('success') for t in log)


def _initial_state_from_xml(xml_path: str, static_object_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rebuild state_observations[0] for a dead-end episode by parsing the scene XML.

    Dead-ends record no observations, but collection RESETS from the XML, so the initial state IS the
    XML: movable geom pos/euler + the robot body pos. The parse matches build_episode_validsets.py's
    `_pose_from_xml` (same regex), so the npz `object_center` anchors identically to the validset key.
    """
    try:
        txt = open(xml_path).read()
    except Exception:
        return None
    state: Dict[str, Any] = {}
    for name in static_object_info:
        if not name.endswith('_movable'):
            continue
        m = re.search(rf'<geom name="{re.escape(name)}"[^>]*?pos="([^"]+)"', txt)
        if not m:
            return None
        p = m.group(1).split()
        em = re.search(rf'<geom name="{re.escape(name)}"[^>]*?euler="([^"]+)"', txt)
        theta = math.radians(float(em.group(1).split()[2])) if em else 0.0  # scene euler in degrees
        state[f'{name}_pose'] = (float(p[0]), float(p[1]), theta)
    rb = re.search(r'<body name="(?:car|robot)"[^>]*?pos="([^"]+)"', txt)
    if not rb:
        return None
    rp = rb.group(1).split()
    state['robot_pose'] = (float(rp[0]), float(rp[1]), 0.0)
    return state


def _synthesize_dead_end_episode(episode: Dict[str, Any],
                                 reachable_by_xml: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
    """Return a render-ready copy of a dead-end episode, or None if it can't be done truthfully.

    - state_observations[0]: rebuilt from the XML (initial state == XML by construction).
    - action_sequence: ONE pseudo-action carrying ONLY object_id — the visualizer then centers the
      crops on the chosen object and emits the documented sentinels (edge/depth_idx_a1 = -1,
      se2_target_a1 = 0, goal_mask_a1 all-zero). No visualizer change needed.
    - reachable-objects list: BEST-EFFORT from a same-xml sibling episode or the sidecar map. It only
      fills the GLOBAL 'reachable' npz mask — the scorer's 5 consumed channels (static, movable,
      target_object, robot_region, goal_sample_region; see build_scorer_dataset.py CHANS) never read
      it, and robot_region/goal_sample_region are BFS-computed at render time from the synthesized
      state. So a missing list is left absent (blank global mask, key unused), never fabricated.
    """
    st = episode.get('algorithm_stats') or {}
    obj = st.get('chosen_object_id')
    xml = episode.get('xml_file')
    soi = episode.get('static_object_info') or {}
    if not obj or not xml:
        return None
    state = _initial_state_from_xml(xml, soi)
    if state is None or f'{obj}_pose' not in state:
        return None
    synth = episode.copy()
    synth['state_observations'] = [state]
    synth['post_action_state_observations'] = []
    synth['action_sequence'] = [{'object_id': obj}]
    reach = reachable_by_xml.get(xml)
    if reach:
        synth['reachable_objects_before_action'] = [list(reach)]
    synth['solution_depth'] = 0
    return synth


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
                    wide_crop_size: Optional[float] = None,
                    tight_crop_size: Optional[float] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Process a single episode to generate dual-crop masks + SE(2) targets.

    The visualizer renders the env once and emits both crops:
      - wide (1.2 m default): mask-prediction supervision, includes goal_mask_a*
      - tight (0.5 m default): SE(2)/index supervision, object-centered context

    SE(2) targets and primitive indices (edge_idx, depth_idx) are
    crop-independent and emitted alongside.

    Args:
        episode: Episode data dictionary
        visualizer: NAMODataVisualizer instance
        generate_local: Whether to generate local (dual-crop) masks
        local_only: If True, skip global masks
        wide_crop_size: Side length (m) of the wide crop. None → visualizer default (1.2).
        tight_crop_size: Side length (m) of the tight crop. None → visualizer default (0.5).

    Returns:
        Tuple of (masks_dict, metadata_dict). The masks_dict contains keys with
        `local_wide_*`, `local_tight_*` prefixes (plus optional `<name>` global
        keys) and `se2_target_a{i}` / `edge_idx_a{i}` / `depth_idx_a{i}` scalars.
    """
    local_wide_metadata = None
    local_tight_metadata = None

    result = visualizer.generate_all_masks_highres(
        episode,
        wide_crop_size_meters=wide_crop_size,
        tight_crop_size_meters=tight_crop_size,
    )

    if result is None:
        return {}, None

    masks: Dict[str, np.ndarray] = {}
    if not local_only:
        masks.update(result['global'])

    if generate_local:
        if result['local_wide'] is not None:
            masks.update(result['local_wide'])
            local_wide_metadata = result['local_wide_metadata']
        if result['local_tight'] is not None:
            masks.update(result['local_tight'])
            local_tight_metadata = result['local_tight_metadata']

    # SE(2)/edge/depth scalars are always carried — they're per-action targets,
    # not per-crop image data, so cheap to keep alongside any combination.
    se2_targets = result.get('se2_targets') or {}
    for k, v in se2_targets.items():
        masks[k] = v

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

    # Extract solution counts from algorithm_stats (for sample weighting)
    alg_stats = episode.get('algorithm_stats') or {}
    metadata['solutions_found'] = alg_stats.get('solutions_found_for_neighbour')
    metadata['solutions_total'] = alg_stats.get('solutions_total_for_neighbour')
    metadata['pushes_total'] = alg_stats.get('pushes_total_for_neighbour')

    if 'difficulty_label' in episode:
        metadata['difficulty_label'] = episode.get('difficulty_label', 'unknown')
        metadata['difficulty_score'] = episode.get('difficulty_score')

    if local_wide_metadata is not None:
        metadata['local_wide_metadata'] = local_wide_metadata
    if local_tight_metadata is not None:
        metadata['local_tight_metadata'] = local_tight_metadata

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

    # Save solution counts for sample weighting
    solutions_found = metadata.get('solutions_found')
    save_dict['solutions_found'] = np.array([solutions_found if solutions_found is not None else -1], dtype=np.int32)

    solutions_total = metadata.get('solutions_total')
    save_dict['solutions_total'] = np.array([solutions_total if solutions_total is not None else -1], dtype=np.int32)

    pushes_total = metadata.get('pushes_total')
    save_dict['pushes_total'] = np.array([pushes_total if pushes_total is not None else -1], dtype=np.int32)

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

    # Per-crop metadata (wide + tight). Both share the same object_center /
    # object_theta — saved once under each prefix so downstream loaders can
    # read either independently.
    for prefix_key, meta_key in (('local_wide', 'local_wide_metadata'),
                                 ('local_tight', 'local_tight_metadata')):
        meta = metadata.get(meta_key)
        if meta is None:
            save_dict[f'has_{prefix_key}_masks'] = np.array([False], dtype=bool)
            continue
        save_dict[f'{prefix_key}_object_center'] = np.array(meta['object_center'], dtype=np.float32)
        save_dict[f'{prefix_key}_object_theta'] = np.array([meta['object_theta']], dtype=np.float32)
        save_dict[f'{prefix_key}_bounds'] = np.array(meta['local_bounds'], dtype=np.float32)
        save_dict[f'{prefix_key}_crop_size_meters'] = np.array([meta['crop_size_meters']], dtype=np.float32)
        save_dict[f'{prefix_key}_resolution'] = np.array([meta['resolution']], dtype=np.float32)
        save_dict[f'has_{prefix_key}_masks'] = np.array([True], dtype=bool)

    # Save as compressed npz
    np.savez_compressed(output_path, **save_dict)


def _count_solutions_per_region(episodes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count the number of successful episodes per neighbour region.

    Args:
        episodes: List of episode dictionaries from a pickle file

    Returns:
        Dictionary mapping neighbour_region_label to count of successful episodes
    """
    from collections import defaultdict
    counts = defaultdict(int)

    for ep in episodes:
        if not ep.get('solution_found', False):
            continue
        alg_stats = ep.get('algorithm_stats') or {}
        region_label = alg_stats.get('neighbour_region_label')
        if region_label:
            counts[region_label] += 1

    return dict(counts)


def process_pkl_file_worker(pkl_file: str, output_dir: str, filter_minimum_length: bool = False,
                            split_difficulty: bool = False,
                            generate_local: bool = True,
                            local_only: bool = False,
                            filter_overlaps: bool = False,
                            namo_config_path: Optional[str] = None,
                            wide_crop_size: Optional[float] = None,
                            tight_crop_size: Optional[float] = None,
                            include_dead_ends: bool = False,
                            dead_ends_only: bool = False,
                            reachable_sidecar: Optional[Dict[str, List[str]]] = None) -> Tuple[int, int, str, List[str]]:
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
        filter_overlaps: If True, skip episodes where robot_region and goal_region overlap
        namo_config_path: Path to the namo YAML config. Enables unified wavefront
            (region masks use WavefrontSnapshotExporter, matching the C++ wavefront
            the planner used during collection). Strongly recommended for car-robot
            runs — without it, masks fall back to under-inflated regions.

    Returns:
        Tuple of (total_episodes, processed_episodes, pkl_file, skipped_episodes)
        skipped_episodes is a list of "pkl_file:episode_id" strings for overlapping episodes
    """
    # Create visualizer instance for this worker
    visualizer = NAMODataVisualizer(figsize=(10, 8), namo_config_path=namo_config_path)
    skipped_episodes = []

    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return 0, 0, pkl_file, []

    episodes = data.get('episode_results', [])

    # Count solutions per region (for sample weighting)
    solutions_per_region = _count_solutions_per_region(episodes)
    total_episodes = len(episodes)

    # Apply minimum length filtering if requested
    filtered_episodes, _, _ = filter_episodes_by_minimum_length(episodes, filter_minimum_length)

    processed_episodes = 0

    # DEAD-END support (opt-in): initial reachable set is PER-SCENE (initial state == XML), so a
    # solvable sibling episode on the same xml supplies it; the sidecar map covers pure-dead-end scenes.
    reachable_by_xml: Dict[str, List[str]] = {}
    if include_dead_ends:
        for ep in episodes:
            xml = ep.get('xml_file')
            rl = ep.get('reachable_objects_before_action')
            if xml and rl and rl[0] and xml not in reachable_by_xml:
                reachable_by_xml[xml] = list(rl[0])
        for xml, lst in (reachable_sidecar or {}).items():
            reachable_by_xml.setdefault(xml, lst)

    for episode in filtered_episodes:
        if include_dead_ends and _is_dead_end_episode(episode):
            try:
                synth = _synthesize_dead_end_episode(episode, reachable_by_xml)
                if synth is None:
                    skipped_episodes.append(f"{pkl_file}:{episode.get('episode_id', 'unknown')}:deadend_no_reachable_or_pose")
                    continue
                masks, metadata = process_episode(
                    synth, visualizer,
                    generate_local=generate_local, local_only=local_only,
                    wide_crop_size=wide_crop_size,
                    tight_crop_size=tight_crop_size,
                )
                if not masks:
                    continue
                output_path = os.path.join(output_dir, metadata['task_id'], f"{metadata['episode_id']}.npz")
                save_episode_data(masks, metadata, output_path)
                processed_episodes += 1
            except Exception:
                continue
        elif is_valid_episode(episode) and not dead_ends_only:
            try:
                # Inject correct solutions_found count from episode counting
                # (solutions_found_for_neighbour is broken in existing data)
                alg_stats = episode.get('algorithm_stats') or {}
                region_label = alg_stats.get('neighbour_region_label')
                if region_label and region_label in solutions_per_region:
                    if 'algorithm_stats' not in episode:
                        episode['algorithm_stats'] = {}
                    episode['algorithm_stats']['solutions_found_for_neighbour'] = solutions_per_region[region_label]

                if split_difficulty:
                    assign_difficulty_annotation(episode)

                # Split multi-step episodes into trajectory suffix examples
                suffix_episodes = split_episode_into_trajectory_suffixes(episode)

                # Process each suffix as a separate training example
                for suffix_episode in suffix_episodes:
                    # Generate masks and metadata
                    masks, metadata = process_episode(
                        suffix_episode, visualizer,
                        generate_local=generate_local, local_only=local_only,
                        wide_crop_size=wide_crop_size,
                        tight_crop_size=tight_crop_size,
                    )

                    # Skip if no masks (e.g., missing region_goals_sampled)
                    if not masks:
                        continue

                    # Check for region overlap if filtering is enabled
                    if filter_overlaps and has_region_overlap(masks):
                        episode_id = metadata.get('episode_id', 'unknown')
                        skipped_episodes.append(f"{pkl_file}:{episode_id}")
                        continue  # Skip this episode

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

    return total_episodes, processed_episodes, pkl_file, skipped_episodes


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


def _process_pkl_file_for_hdf5(args: Tuple) -> List[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    """Worker function to process an entire pkl file for HDF5 output.

    Args:
        args: Tuple of (pkl_file, filter_minimum_length, split_difficulty,
                        generate_local, local_only, namo_config_path,
                        wide_crop_size, tight_crop_size)

    Returns:
        List of (masks, metadata) tuples for all episodes in the file
    """
    if len(args) == 8:
        (pkl_file, filter_minimum_length, split_difficulty,
         generate_local, local_only, namo_config_path,
         wide_crop_size, tight_crop_size) = args
    elif len(args) == 6:
        # Older 6-arg signature (no per-crop sizes — fall back to visualizer defaults)
        pkl_file, filter_minimum_length, split_difficulty, generate_local, local_only, namo_config_path = args
        wide_crop_size = None
        tight_crop_size = None
    else:
        # Older 5-arg signature
        pkl_file, filter_minimum_length, split_difficulty, generate_local, local_only = args
        namo_config_path = None
        wide_crop_size = None
        tight_crop_size = None
    results = []

    # One visualizer per pkl file (reused for all episodes in file)
    visualizer = NAMODataVisualizer(figsize=(10, 8), namo_config_path=namo_config_path)

    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return results

    episodes = data.get('episode_results', [])

    # Count solutions per region (for sample weighting)
    solutions_per_region = _count_solutions_per_region(episodes)

    # Apply filtering
    filtered_episodes, _, _ = filter_episodes_by_minimum_length(episodes, filter_minimum_length)

    for episode in filtered_episodes:
        if is_valid_episode(episode):
            try:
                # Inject correct solutions_found count from episode counting
                # (solutions_found_for_neighbour is broken in existing data)
                alg_stats = episode.get('algorithm_stats') or {}
                region_label = alg_stats.get('neighbour_region_label')
                if region_label and region_label in solutions_per_region:
                    if 'algorithm_stats' not in episode:
                        episode['algorithm_stats'] = {}
                    episode['algorithm_stats']['solutions_found_for_neighbour'] = solutions_per_region[region_label]

                if split_difficulty:
                    assign_difficulty_annotation(episode)

                suffix_episodes = split_episode_into_trajectory_suffixes(episode)

                for suffix_episode in suffix_episodes:
                    masks, metadata = process_episode(
                        suffix_episode, visualizer,
                        generate_local=generate_local, local_only=local_only,
                        wide_crop_size=wide_crop_size,
                        tight_crop_size=tight_crop_size,
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
                    local_only: bool = False,
                    namo_config_path: Optional[str] = None,
                    wide_crop_size: Optional[float] = None,
                    tight_crop_size: Optional[float] = None) -> Tuple[int, int]:
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
        (pkl_file, filter_minimum_length, split_difficulty,
         generate_local, local_only, namo_config_path,
         wide_crop_size, tight_crop_size)
        for pkl_file in pkl_files
    ]

    total_processed = 0

    with HDF5Writer(output_path) as h5_writer:
        if num_workers == 1:
            # Serial processing
            for args in tqdm(pkl_files, desc="Processing pkl files"):
                results = _process_pkl_file_for_hdf5(
                    (args, filter_minimum_length, split_difficulty,
                     generate_local, local_only, namo_config_path,
                     wide_crop_size, tight_crop_size)
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
    parser.add_argument('--input-dir', required=False, default=None, help='Directory containing .pkl files (mutually exclusive with --pkl-list)')
    parser.add_argument('--pkl-list', required=False, default=None,
                        help='Path to a text file listing absolute PKL paths, one per line. Use this for sharded sbatch runs where each shard processes a different subset.')
    parser.add_argument('--output-dir', required=True, help='Output directory for .npz files')
    parser.add_argument('--pattern', default='*_results.pkl', help='File pattern to match (default: *_results.pkl, only used with --input-dir)')
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
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of pkl files to process (for testing)')
    parser.add_argument('--include-dead-ends', action='store_true',
                        help='ALSO render tried-and-all-failed (dead-end) episodes: state rebuilt from the '
                             'scene XML, pseudo-a1 (edge/depth_idx_a1=-1 sentinel). The 5 scorer channels '
                             '(static/movable/target_object/robot_region/goal_sample_region) are all derivable '
                             'at render time. Default off = legacy behavior. (horizon-Q H0b, task #23)')
    parser.add_argument('--dead-ends-only', action='store_true',
                        help='Render ONLY dead-end episodes (implies --include-dead-ends). Use to add dead-end '
                             'npz to an output tree whose solvable npz already exist — no re-render, no dupes.')
    parser.add_argument('--reachable-sidecar', type=str, default=None,
                        help='OPTIONAL JSON {xml_path: [reachable object names]} to fill the global '
                             '"reachable" npz mask for dead-ends (unused by the scorer pipeline; best-effort '
                             'borrow from a same-xml sibling episode happens regardless).')
    parser.add_argument('--filter-overlaps', action='store_true',
                       help='Skip episodes where robot_region and goal_region overlap (connected regions)')
    parser.add_argument('--namo-config', type=str,
                       default='config/namo_config_complete_skill15_car_1x.yaml',
                       help=('Path to the namo YAML config the planner used during '
                             'data collection. Region masks are computed via '
                             'WavefrontSnapshotExporter and the robot footprint MUST '
                             'match the C++ runtime — otherwise the visible robot '
                             'region disagrees with the action targets in the data. '
                             'Fixed to the canonical car 1x d5 runtime profile.'))
    parser.add_argument('--wide-crop-size', type=float, default=1.2,
                        help=('Side length (m) of the WIDE object-centered crop. '
                              'Used for mask-prediction supervision (must contain a '
                              'full push: max primitive delta + obj half-extent ≈ '
                              '1.14 m). Default 1.2 m.'))
    parser.add_argument('--tight-crop-size', type=float, default=0.5,
                        help=('Side length (m) of the TIGHT object-centered crop. '
                              'Used for SE(2)/primitive-index supervision (output is '
                              'scalars). Default 0.5 m.'))

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.local_only and args.global_only:
        print("Error: --local-only and --global-only are mutually exclusive")
        sys.exit(1)

    # Determine mask generation mode
    generate_local = not args.global_only  # True unless --global-only is set

    # Resolve PKL input (either --input-dir scan or --pkl-list file)
    if not args.input_dir and not args.pkl_list:
        print("Error: must provide either --input-dir or --pkl-list")
        sys.exit(1)
    if args.input_dir and args.pkl_list:
        print("Error: --input-dir and --pkl-list are mutually exclusive")
        sys.exit(1)

    if args.pkl_list:
        if not os.path.isfile(args.pkl_list):
            print(f"Error: --pkl-list file not found: {args.pkl_list}")
            sys.exit(1)
        with open(args.pkl_list, "r") as f:
            pkl_files = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        # Filter out missing files (warn but continue)
        existing = [p for p in pkl_files if os.path.isfile(p)]
        if len(existing) < len(pkl_files):
            print(f"WARNING: {len(pkl_files) - len(existing)} PKL paths in list not found, skipping them")
        pkl_files = existing
    else:
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory does not exist: {args.input_dir}")
            sys.exit(1)
        # Find all pickle files - support recursive pattern from run_mask_generation.py
        if '**' in args.pattern:
            pkl_files = glob.glob(os.path.join(args.input_dir, args.pattern), recursive=True)
        else:
            pkl_files = glob.glob(os.path.join(args.input_dir, args.pattern))

    if not pkl_files:
        print(f"Error: No PKL files to process (from {'pkl-list' if args.pkl_list else 'input-dir glob'})")
        sys.exit(1)

    # Limit number of files if --max-files specified
    if args.max_files is not None and args.max_files < len(pkl_files):
        print(f"Found {len(pkl_files)} pickle files, limiting to {args.max_files}")
        pkl_files = pkl_files[:args.max_files]
    else:
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
            local_only=args.local_only,
            namo_config_path=args.namo_config,
            wide_crop_size=args.wide_crop_size,
            tight_crop_size=args.tight_crop_size,
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
    if args.filter_overlaps:
        print("Overlap filtering ENABLED - skipping episodes with connected robot/goal regions")

    # Process all files
    total_episodes = 0
    total_processed = 0
    all_skipped_episodes = []

    reachable_sidecar = None
    if args.reachable_sidecar:
        with open(args.reachable_sidecar) as f:
            reachable_sidecar = json.load(f)
        print(f"Loaded reachable sidecar: {len(reachable_sidecar)} scenes")
    if args.dead_ends_only:
        args.include_dead_ends = True

    if num_workers == 1:
        # Serial processing (original behavior)
        for pkl_file in tqdm(pkl_files, desc="Processing files"):
            file_episodes, file_processed, _, skipped = process_pkl_file_worker(
                pkl_file, args.output_dir,
                args.filter_minimum_length, args.split_difficulty,
                generate_local, args.local_only, args.filter_overlaps,
                namo_config_path=args.namo_config,
                wide_crop_size=args.wide_crop_size,
                tight_crop_size=args.tight_crop_size,
                include_dead_ends=args.include_dead_ends,
                dead_ends_only=args.dead_ends_only,
                reachable_sidecar=reachable_sidecar)
            total_episodes += file_episodes
            total_processed += file_processed
            all_skipped_episodes.extend(skipped)
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
                local_only=args.local_only,
                filter_overlaps=args.filter_overlaps,
                namo_config_path=args.namo_config,
                wide_crop_size=args.wide_crop_size,
                tight_crop_size=args.tight_crop_size,
                include_dead_ends=args.include_dead_ends,
                dead_ends_only=args.dead_ends_only,
                reachable_sidecar=reachable_sidecar)

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
                        file_episodes, file_processed, _, skipped = result.get()
                        total_episodes += file_episodes
                        total_processed += file_processed
                        all_skipped_episodes.extend(skipped)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing file: {e}")
                        pbar.update(1)

    # Write skipped episodes to file if overlap filtering was enabled
    if args.filter_overlaps and all_skipped_episodes:
        skipped_log_path = os.path.join(args.output_dir, "skipped_overlapping_episodes.txt")
        with open(skipped_log_path, 'w') as f:
            f.write(f"# Episodes skipped due to robot_region/goal_region overlap\n")
            f.write(f"# Total skipped: {len(all_skipped_episodes)}\n")
            f.write(f"# Format: pkl_file:episode_id\n\n")
            for entry in all_skipped_episodes:
                f.write(f"{entry}\n")
        print(f"Skipped {len(all_skipped_episodes)} overlapping episodes (logged to {skipped_log_path})")

    # Print summary statistics
    print(f"\n=== Processing Complete ===")
    print(f"Files processed: {len(pkl_files)}")
    print(f"Total episodes found: {total_episodes}")
    print(f"Valid episodes processed: {total_processed}")
    if args.filter_overlaps:
        num_skipped = len(all_skipped_episodes)
        total_before_filter = total_processed + num_skipped
        if total_before_filter > 0:
            skip_pct = num_skipped / total_before_filter * 100
            print(f"Episodes skipped (overlap): {num_skipped} ({skip_pct:.1f}%)")
        else:
            print(f"Episodes skipped (overlap): {num_skipped}")
    if total_episodes > 0:
        print(f"Success rate: {total_processed/total_episodes*100:.1f}%")
    else:
        print("Success rate: 0.0%")
    print(f"Output directory: {args.output_dir}")
    print(f"Generated {total_processed} compressed .npz files")


if __name__ == "__main__":
    main()
