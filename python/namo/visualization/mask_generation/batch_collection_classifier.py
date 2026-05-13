#!/usr/bin/env python3
"""
Batch Mask Collection for F-Characterization Classifier Training

Reads exhaustive F-characterization pkl data and generates NPZ files
containing scene masks + F grid (60x10 primitive success/fail labels).

Unlike batch_collection.py (which processes solution episodes),
this script processes exhaustive trial logs — deduplicating by
(xml, object, region) and extracting the full F grid per instance.

Usage:
    python -m namo.visualization.mask_generation.batch_collection_classifier \
        --input-dir /common/users/dm1487/namo_data/f_characterization/1_push_exhaustive_train/modular_data_rlab5 \
        --output-dir /common/users/dm1487/namo_data/f_characterization/classifier_train_npz \
        --workers 16

    # Multiple input dirs (from different machines):
    python -m namo.visualization.mask_generation.batch_collection_classifier \
        --input-dir /path/to/modular_data_rlab5 /path/to/modular_data_rlab6 \
        --output-dir /path/to/output \
        --workers 16
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import Pool
from functools import partial

# Add sage_learning to path for NAMODataVisualizer
NAMO_ROOT = Path(__file__).resolve().parents[4]
SAGE_ROOT = NAMO_ROOT.parent / "sage_learning"
sys.path.insert(0, str(SAGE_ROOT))

from sage_learning.visualizer import NAMODataVisualizer


# ── Instance extraction ──────────────────────────────────────────────────

def extract_instances_from_pkl(pkl_path: str) -> List[Dict[str, Any]]:
    """Extract unique (xml, object, region) instances from an exhaustive pkl file.

    Returns list of instance dicts, each containing the episode data needed
    for mask generation plus the primitive_trial_log for the F grid.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    seen = set()
    instances = []

    for ep in data.get('episode_results', []):
        stats = ep.get('algorithm_stats') or {}
        tlog = stats.get('primitive_trial_log')
        if not tlog:
            continue

        xml_file = ep.get('xml_file', '')
        obj_id = stats.get('chosen_object_id', '?')
        region = stats.get('neighbour_region_label', '?')
        key = (xml_file, obj_id, region)

        if key in seen:
            continue
        seen.add(key)

        # Build the F grid (60x10)
        f_grid = np.full((60, 10), np.nan, dtype=np.float32)
        wall_grid = np.full((60, 10), np.nan, dtype=np.float32)
        for trial in tlog:
            ei, d = trial['edge_idx'], trial['depth']
            f_grid[ei, d] = 1.0 if trial['success'] else 0.0
            wall_grid[ei, d] = 1.0 if trial['wall_collision'] else 0.0

        F = int(np.nansum(f_grid))
        R = int((~np.isnan(f_grid)).sum())

        # Skip F=0 instances (multi-push problems)
        if F == 0:
            continue

        instances.append({
            'episode': ep,
            'xml_file': xml_file,
            'object_id': obj_id,
            'region': region,
            'f_grid': f_grid,
            'wall_grid': wall_grid,
            'F': F,
            'R': R,
            'ratio': F / R if R > 0 else 0.0,
            'pkl_path': pkl_path,
        })

    return instances


# ── Mask generation ──────────────────────────────────────────────────────

def generate_masks_for_instance(instance: Dict[str, Any],
                                visualizer: NAMODataVisualizer,
                                local_crop_size: float = 5.0
                                ) -> Optional[Dict[str, np.ndarray]]:
    """Generate scene masks for a single instance using the existing visualizer.

    The episode data from exhaustive collection has the same structure as
    regular collection data, so we can reuse the visualizer directly.
    """
    episode = instance['episode']

    # Try high-res rendering (produces both global and local masks)
    try:
        result = visualizer.generate_all_masks_highres(
            episode, local_crop_size_meters=local_crop_size
        )
    except Exception as e:
        # Fallback: try batch masks (global only, no local)
        try:
            result = None
            masks = visualizer.generate_episode_masks_batch(episode)
            if masks:
                return masks
        except Exception:
            pass
        return None

    if result is None:
        # generate_all_masks_highres returns None if region_goals_sampled missing
        # Fall back to basic mask generation
        try:
            masks = visualizer.generate_episode_masks_batch(episode)
            return masks if masks else None
        except Exception:
            return None

    # Combine global + local masks
    masks = result.get('global', {})
    local_masks = result.get('local')
    if local_masks:
        masks.update(local_masks)

    return masks if masks else None


# ── Save ─────────────────────────────────────────────────────────────────

def save_classifier_data(masks: Dict[str, np.ndarray],
                         instance: Dict[str, Any],
                         output_path: str) -> None:
    """Save scene masks + F grid + metadata to NPZ."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_dict = dict(masks)

    # Add F characterization data
    save_dict['f_grid'] = instance['f_grid']           # (60, 10) success/fail/nan
    save_dict['wall_grid'] = instance['wall_grid']     # (60, 10) wall collision flags
    save_dict['F'] = np.array([instance['F']], dtype=np.int32)
    save_dict['R'] = np.array([instance['R']], dtype=np.int32)
    save_dict['f_ratio'] = np.array([instance['ratio']], dtype=np.float32)

    # Reachability mask (which primitives were evaluated = reachable)
    r_mask = (~np.isnan(instance['f_grid'])).astype(np.float32)
    save_dict['r_mask'] = r_mask                       # (60, 10) binary reachability

    # Metadata
    save_dict['xml_file'] = np.array([instance['xml_file']], dtype='U')
    save_dict['object_id'] = np.array([instance['object_id']], dtype='U')
    save_dict['region'] = np.array([instance['region']], dtype='U')

    ep = instance['episode']
    save_dict['robot_goal'] = np.array(ep.get('robot_goal', [0, 0, 0]), dtype=np.float32)

    # Episode metadata
    save_dict['episode_id'] = np.array([ep.get('episode_id', '')], dtype='U')

    alg_stats = ep.get('algorithm_stats') or {}
    save_dict['solutions_found'] = np.array(
        [alg_stats.get('solutions_found_for_neighbour', -1)], dtype=np.int32)
    save_dict['pushes_total'] = np.array(
        [alg_stats.get('pushes_total_for_neighbour', -1)], dtype=np.int32)

    # Local mask metadata if present
    # (already included in masks dict from generate_all_masks_highres)

    np.savez_compressed(output_path, **save_dict)


# ── Worker ───────────────────────────────────────────────────────────────

def process_pkl_file_worker(pkl_file: str, output_dir: str,
                            local_crop_size: float = 5.0
                            ) -> Tuple[int, int, str]:
    """Worker: process one pkl file, generate NPZ for each unique instance."""
    visualizer = NAMODataVisualizer(figsize=(10, 8))

    try:
        instances = extract_instances_from_pkl(pkl_file)
    except Exception as e:
        print(f"  Error loading {pkl_file}: {e}")
        return 0, 0, pkl_file

    processed = 0
    for inst in instances:
        try:
            masks = generate_masks_for_instance(inst, visualizer,
                                                local_crop_size=local_crop_size)
            if not masks:
                continue

            # Output path: output_dir/env_name/obj_region.npz
            env_name = Path(inst['xml_file']).stem
            safe_region = inst['region'].replace('/', '_')
            fname = f"{env_name}_{inst['object_id']}_{safe_region}.npz"
            output_path = os.path.join(output_dir, env_name, fname)

            save_classifier_data(masks, inst, output_path)
            processed += 1

        except Exception as e:
            continue

    return len(instances), processed, pkl_file


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate classifier training data from exhaustive F-characterization")
    parser.add_argument('--input-dir', nargs='+', required=True,
                        help='Directory(s) with *_results.pkl files')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for NPZ files')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--serial', action='store_true',
                        help='Run in serial mode (for debugging)')
    parser.add_argument('--local-crop-size', type=float, default=5.0,
                        help='Local crop size in meters')
    args = parser.parse_args()

    # Collect all pkl files
    pkl_files = []
    for input_dir in args.input_dir:
        pkl_files.extend(sorted(Path(input_dir).glob('*_results.pkl')))
    pkl_files = [str(p) for p in pkl_files]

    print(f"Found {len(pkl_files)} pkl files")
    print(f"Output: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    worker_fn = partial(process_pkl_file_worker,
                        output_dir=args.output_dir,
                        local_crop_size=args.local_crop_size)

    total_instances = 0
    total_processed = 0

    if args.serial:
        for pkl_file in pkl_files:
            n_inst, n_proc, _ = worker_fn(pkl_file)
            total_instances += n_inst
            total_processed += n_proc
            print(f"  {Path(pkl_file).stem}: {n_proc}/{n_inst} instances")
    else:
        with Pool(args.workers) as pool:
            results = pool.map(worker_fn, pkl_files)
        for n_inst, n_proc, pkl_file in results:
            total_instances += n_inst
            total_processed += n_proc

    print(f"\nDone: {total_processed}/{total_instances} instances saved to {args.output_dir}")


if __name__ == '__main__':
    main()
