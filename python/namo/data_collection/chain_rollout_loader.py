"""Reader for v1 chain sampler pkls.

The v1 UniformRolloutSampler (chain mode) emits per-step transition records
in ``algorithm_stats["rollout_trace"]``. This loader walks the trace,
reconstructs per-rollout chains via ``parent_id``, and exposes
analysis-friendly records.

Compared to ``rollout_trace_loader.load_attempts_from_pkl`` (which is
strictly 1-push / depth-0), this module understands chain context:

* ``chain_depth`` per record (0 = first push in a rollout, 1 = second, ...).
* ``parent_id`` linking each push to the previous one in its rollout.
* ``rollout_id`` grouping records into independent rollouts.
* ``per_neighbor_opening`` per push (which neighbor became reachable, if any).
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def load_chain_pkl(pkl_path: str) -> List[Dict[str, Any]]:
    """Load a worker pkl and return one record per (xml, object, neighbor) attempt.

    Each record has chain-aware fields:

    * ``f_grid_depth0``: (60, 10) numpy array of depth-0 success labels.
      Filled from chain_depth==0 entries only (same semantics as v0 f_grid).
      Use this for backward-compat 1-push F-characterization.
    * ``rollouts``: list of per-rollout chains (each a list of step dicts).
    * ``solved_rollouts``: subset of ``rollouts`` whose final step opened the goal.
    * ``F_depth0``, ``R_depth0``, ``f_ratio_depth0``: classic 1-push counts.
    * ``F_chain``, ``R_chain``: chain-aware counts (any rollout that opened goal).
    """
    import numpy as np  # local import — kept light

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    records: List[Dict[str, Any]] = []
    for ep in data.get("episode_results", []):
        stats = ep.get("algorithm_stats") or {}
        rollout_trace = stats.get("rollout_trace") or []
        trial_log = stats.get("primitive_trial_log") or []
        if not (rollout_trace or trial_log):
            continue

        # Depth-0 f_grid: same shape as v0 readers, only chain_depth=0 entries.
        f_grid = np.full((60, 10), np.nan, dtype=np.float32)
        for trial in trial_log:
            cd = trial.get("chain_depth", 0)
            if cd != 0:
                # Shouldn't happen — Phase 1 shim filters this. Skip defensively.
                continue
            ei = int(trial.get("edge_idx", -1))
            d = int(trial.get("depth", -1))
            if 0 <= ei < 60 and 0 <= d < 10:
                f_grid[ei, d] = 1.0 if trial.get("success") else 0.0
        F0 = int(np.nansum(f_grid))
        R0 = int((~np.isnan(f_grid)).sum())

        # Chain reconstruction: group rollout_trace by rollout_id.
        rollouts: List[List[Dict[str, Any]]] = []
        if rollout_trace:
            by_rollout: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for rec in rollout_trace:
                by_rollout[int(rec.get("rollout_id", 0))].append(rec)
            for rid in sorted(by_rollout):
                chain = sorted(by_rollout[rid], key=lambda r: int(r["chain_depth"]))
                rollouts.append(chain)

        # Solved rollouts: any rollout whose final step opened the goal.
        solved_rollouts: List[List[Dict[str, Any]]] = []
        for chain in rollouts:
            if not chain:
                continue
            last = chain[-1]
            opens = last.get("per_neighbor_opening") or {}
            if opens.get("goal", False):
                solved_rollouts.append(chain)

        F_chain = len(solved_rollouts)
        R_chain = len(rollouts)

        records.append({
            "pkl_path": str(pkl_path),
            "xml_file": ep.get("xml_file", ""),
            "object_id": stats.get("chosen_object_id"),
            "neighbor": stats.get("neighbour_region_label"),
            "robot_goal": ep.get("robot_goal"),
            "success": bool(ep.get("solution_found")),
            "chain_depth": stats.get("chain_depth"),
            "failure_reason": stats.get("failure_reason"),
            "pushes_total": stats.get("pushes_total_for_neighbour"),
            "rollouts_total": stats.get("rollouts_total_for_neighbour"),
            "solutions_total": stats.get("solutions_total_for_neighbour"),
            # 1-push view (backward-compat)
            "f_grid_depth0": f_grid,
            "F_depth0": F0,
            "R_depth0": R0,
            "f_ratio_depth0": (F0 / max(R0, 1)) if R0 > 0 else 0.0,
            # Chain view
            "rollouts": rollouts,
            "solved_rollouts": solved_rollouts,
            "F_chain": F_chain,
            "R_chain": R_chain,
            "f_ratio_chain": (F_chain / max(R_chain, 1)) if R_chain > 0 else 0.0,
            # Pass-through to episode for downstream renderers
            "episode": ep,
        })

    return records


def iter_chains(pkl_path: str) -> Iterator[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Yield (attempt_record, chain) for every chain in a pkl.

    `chain` is a list of step dicts (transition records) ordered by chain_depth.
    Useful when you want to walk every chain across every attempt without
    materializing the full nested structure.
    """
    for rec in load_chain_pkl(pkl_path):
        for chain in rec["rollouts"]:
            yield rec, chain


def summarize_pkl(pkl_path: str) -> Dict[str, Any]:
    """One-line summary stats for a pkl. Cheap to call across many files."""
    recs = load_chain_pkl(pkl_path)
    n_attempts = len(recs)
    n_solved = sum(1 for r in recs if r["success"])
    total_rollouts = sum(int(r["R_chain"] or 0) for r in recs)
    total_solving = sum(int(r["F_chain"] or 0) for r in recs)
    total_pushes = sum(int(r["pushes_total"] or 0) for r in recs)
    return {
        "pkl_path": str(pkl_path),
        "n_attempts": n_attempts,
        "n_solved_attempts": n_solved,
        "total_rollouts": total_rollouts,
        "total_solving_chains": total_solving,
        "total_pushes": total_pushes,
        "f_ratio_chain": (total_solving / max(total_rollouts, 1)) if total_rollouts > 0 else 0.0,
    }
