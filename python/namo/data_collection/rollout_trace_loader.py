"""Utility for reading UniformRolloutSampler pkls into analysis-friendly records.

Each pkl from the modular collection pipeline contains episode_results — one entry
per (object, neighbor) AttemptResult. This module flattens those into a list of
dicts with explicit F/R/f_ratio fields and a reconstructed (60, 10) f_grid.

Compatible with batch_collection_classifier.py's extract_instances_from_pkl, but
exposes the richer fields (per_neighbor_region_goals, etc.) that the new sampler
stores in env_metadata.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List

import numpy as np


def load_attempts_from_pkl(pkl_path: str) -> List[Dict[str, Any]]:
    """Load a worker pkl and return one record per (xml, object, neighbor).

    Each record contains the f_grid, F, R, f_ratio, region_goals_sampled, and
    references back to the original episode for downstream mask rendering.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    records: List[Dict[str, Any]] = []
    for ep in data.get("episode_results", []):
        stats = ep.get("algorithm_stats") or {}
        trial_log = stats.get("primitive_trial_log")
        if not trial_log:
            continue

        f_grid = np.full((60, 10), np.nan, dtype=np.float32)
        for trial in trial_log:
            ei = int(trial["edge_idx"])
            d = int(trial["depth"])
            if 0 <= ei < 60 and 0 <= d < 10:
                f_grid[ei, d] = 1.0 if trial["success"] else 0.0

        f_count = int(np.nansum(f_grid))
        r_count = int((~np.isnan(f_grid)).sum())

        records.append({
            "pkl_path": pkl_path,
            "xml_file": ep.get("xml_file", ""),
            "object_id": stats.get("chosen_object_id"),
            "neighbor": stats.get("neighbour_region_label"),
            "robot_goal": ep.get("robot_goal"),
            "f_grid": f_grid,
            "F": f_count,
            "R": r_count,
            "f_ratio": (f_count / r_count) if r_count > 0 else 0.0,
            "region_goals_sampled": stats.get("region_goals_sampled"),
            "episode": ep,                          # full episode for mask rendering
        })

    return records
