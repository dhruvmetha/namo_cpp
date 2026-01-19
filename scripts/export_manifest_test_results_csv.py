#!/usr/bin/env python3
"""Export manifest_test evaluation pickles to a single CSV.

This walks `eval_results/manifest_test/**/**/shard_*/modular_data_*/*_results.pkl`,
loads per-environment pickles, and writes one CSV row per neighbour-attempt episode.

The CSV is intentionally "analysis friendly": mostly scalar fields + small JSON blobs.
For deep debugging (e.g., full state observations or raw ML goals), use the pickles directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_kwargs):
        return iterable


_SHARD_RE = re.compile(
    r"^shard_(?P<shard>\d+)_start(?P<start>\d+)_end(?P<end>\d+)_job(?P<job>\d+)$"
)

# Ensure `namo` is importable when unpickling summaries that include FailureCode enums.
_NAMO_CPP_ROOT = Path(__file__).resolve().parents[1]
_PY_DIR = _NAMO_CPP_ROOT / "python"
_BUILD_PY_DIR = _NAMO_CPP_ROOT / "build_python"
if _PY_DIR.is_dir():
    sys.path.insert(0, str(_PY_DIR))
if _BUILD_PY_DIR.is_dir():
    sys.path.insert(0, str(_BUILD_PY_DIR))


def _to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def _json_compact(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(value)


def _find_collection_summary(modular_dir: Path) -> Optional[Path]:
    for p in modular_dir.glob("collection_summary_*.pkl"):
        return p
    return None


def _extract_checkpoint_from_summary(summary: Dict[str, Any]) -> str:
    cfg = (summary.get("collection_metadata") or {}).get("config") or {}
    planner_cfg = cfg.get("planner_config") or {}
    algo_params = planner_cfg.get("algorithm_params") or {}
    return (
        algo_params.get("ml_goal_model_path")
        or algo_params.get("ml_goal_model")
        or ""
    )


def _parse_path_metadata(pkl_path: Path) -> Dict[str, Any]:
    # Expected structure:
    #   eval_results/manifest_test/<model_tag>/<ckpt_stem>/shard_*_job*/modular_data_<host>/*_results.pkl
    parts = pkl_path.parts
    try:
        manifest_idx = parts.index("manifest_test")
        model_tag = parts[manifest_idx + 1]
        ckpt_stem = parts[manifest_idx + 2]
    except Exception:
        model_tag = ""
        ckpt_stem = ""

    shard_dir = ""
    shard_idx = ""
    shard_start = ""
    shard_end = ""
    job_id = ""
    for part in parts:
        m = _SHARD_RE.match(part)
        if m:
            shard_dir = part
            shard_idx = int(m.group("shard"))
            shard_start = int(m.group("start"))
            shard_end = int(m.group("end"))
            job_id = int(m.group("job"))
            break

    host = ""
    for part in parts:
        if part.startswith("modular_data_"):
            host = part[len("modular_data_") :]
            break

    return {
        "model_tag": model_tag,
        "ckpt_stem": ckpt_stem,
        "shard_dir": shard_dir,
        "shard_idx": shard_idx,
        "shard_start": shard_start,
        "shard_end": shard_end,
        "job_id": job_id,
        "host": host,
    }


def _iter_result_pickles(root: Path) -> Iterable[Path]:
    yield from root.rglob("*_results.pkl")

def _normalize_ckpt_stem(ckpt_path: str) -> str:
    stem = Path(ckpt_path).name
    if stem.endswith(".ckpt"):
        stem = stem[:-5]
    # Mirrors `tr '=.' '__'` used in the SLURM worker.
    return stem.replace("=", "_").replace(".", "_")


def _load_ckpt_list_mapping(ckpt_list_path: Path) -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}
    if not ckpt_list_path.exists():
        return mapping
    for line in ckpt_list_path.read_text().splitlines():
        ckpt = line.strip()
        if not ckpt or ckpt.startswith("#"):
            continue
        ckpt_path = Path(ckpt)
        # Expected: .../<model_tag>/checkpoints/<file>.ckpt
        # model_tag is the parent of "checkpoints" (or two-level up from file).
        model_tag = ckpt_path.parent.parent.name if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent.name
        ckpt_stem = _normalize_ckpt_stem(ckpt)
        mapping[(model_tag, ckpt_stem)] = ckpt
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="eval_results/manifest_test",
        help="Root directory containing manifest_test results.",
    )
    parser.add_argument(
        "--ckpt-list",
        type=str,
        default=None,
        help="Optional checkpoint list (1 path per line) to recover checkpoint_path without unpickling summaries. "
             "Defaults to scripts/manifest_test_checkpoints.txt when present.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. If omitted, writes under <root>/all_attempts_<timestamp>.csv",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        import datetime as _dt

        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = root / f"all_attempts_{ts}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_list_path = None
    if args.ckpt_list:
        ckpt_list_path = Path(args.ckpt_list).resolve()
    else:
        candidate = _NAMO_CPP_ROOT / "scripts" / "manifest_test_checkpoints.txt"
        if candidate.exists():
            ckpt_list_path = candidate
    ckpt_map = _load_ckpt_list_mapping(ckpt_list_path) if ckpt_list_path else {}

    # Cache per modular_data_<host> directory → checkpoint path (and other run metadata)
    modular_meta_cache: Dict[Path, Dict[str, Any]] = {}

    # Fixed schema (stable column ordering)
    fieldnames = [
        # Run metadata
        "checkpoint_path",
        "model_tag",
        "ckpt_stem",
        "job_id",
        "shard_idx",
        "shard_start",
        "shard_end",
        "host",
        "results_pkl",
        # Episode identity
        "xml_file",
        "episode_id",
        "neighbour_region_label",
        "chosen_object_id",
        # Outcome + costs
        "success",
        "solution_depth",
        "chain_depth",
        "search_time_ms",
        "pushes_total_for_neighbour",
        "total_cost",
        # Solution accounting
        "solutions_total_for_neighbour",
        "solutions_found_for_neighbour",
        "solutions_cap_for_neighbour",
        # Validation / failure
        "validation_method",
        "failure_reason",
        # ML diagnostics (scalars)
        "ml_goals_generated",
        "ml_goals_aligned",
        "reachable_edges_count",
        # Phase tracking (small JSON)
        "solved_in_phase",
        "phase_push_counts_json",
        # Hardness metrics
        "any_wall_collision",
        "unique_movable_collision_count",
    ]

    result_paths = list(_iter_result_pickles(root))
    if not result_paths:
        print(f"ERROR: no *_results.pkl found under {root}", file=sys.stderr)
        return 2

    episodes_written = 0
    pkls_read = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pkl_path in tqdm(result_paths, desc="Exporting", unit="pkl"):
            pkls_read += 1
            meta = _parse_path_metadata(pkl_path)

            # Find / cache modular run metadata (checkpoint path)
            modular_dir = pkl_path.parent
            if modular_dir not in modular_meta_cache:
                summary_path = _find_collection_summary(modular_dir)
                ckpt = ""
                if summary_path is not None:
                    try:
                        with open(summary_path, "rb") as sf:
                            summary = pickle.load(sf)
                        ckpt = _extract_checkpoint_from_summary(_to_dict(summary))
                    except Exception:
                        ckpt = ""
                modular_meta_cache[modular_dir] = {"checkpoint_path": ckpt}

            ckpt_path = modular_meta_cache[modular_dir].get("checkpoint_path", "")
            if not ckpt_path and ckpt_map:
                key = (meta.get("model_tag", ""), meta.get("ckpt_stem", ""))
                ckpt_path = ckpt_map.get(key, "")

            try:
                with open(pkl_path, "rb") as pf:
                    payload = pickle.load(pf)
            except Exception as e:
                print(f"WARNING: failed to load {pkl_path}: {e}", file=sys.stderr)
                continue

            payload = _to_dict(payload)
            episodes = payload.get("episode_results") or []

            for ep in episodes:
                ep = _to_dict(ep)
                stats = _to_dict(ep.get("algorithm_stats") or {})

                row = {
                    "checkpoint_path": ckpt_path,
                    "model_tag": meta.get("model_tag", ""),
                    "ckpt_stem": meta.get("ckpt_stem", ""),
                    "job_id": meta.get("job_id", ""),
                    "shard_idx": meta.get("shard_idx", ""),
                    "shard_start": meta.get("shard_start", ""),
                    "shard_end": meta.get("shard_end", ""),
                    "host": meta.get("host", ""),
                    "results_pkl": str(pkl_path),
                    "xml_file": ep.get("xml_file", ""),
                    "episode_id": ep.get("episode_id", ""),
                    "neighbour_region_label": stats.get("neighbour_region_label", ""),
                    "chosen_object_id": stats.get("chosen_object_id", ""),
                    "success": ep.get("solution_found", ep.get("success", "")),
                    "solution_depth": ep.get("solution_depth", ""),
                    "chain_depth": stats.get("chain_depth", ""),
                    "search_time_ms": ep.get("search_time_ms", ""),
                    "pushes_total_for_neighbour": stats.get("pushes_total_for_neighbour", ""),
                    "total_cost": stats.get("total_cost", ""),
                    "solutions_total_for_neighbour": stats.get("solutions_total_for_neighbour", ""),
                    "solutions_found_for_neighbour": stats.get("solutions_found_for_neighbour", ""),
                    "solutions_cap_for_neighbour": stats.get("solutions_cap_for_neighbour", ""),
                    "validation_method": stats.get("validation_method", ""),
                    "failure_reason": stats.get("failure_reason", ""),
                    "ml_goals_generated": stats.get("ml_goals_generated", ""),
                    "ml_goals_aligned": stats.get("ml_goals_aligned", ""),
                    "reachable_edges_count": stats.get("reachable_edges_count", ""),
                    "solved_in_phase": stats.get("solved_in_phase", ""),
                    "phase_push_counts_json": _json_compact(stats.get("phase_push_counts")),
                    "any_wall_collision": ep.get("any_wall_collision", ""),
                    "unique_movable_collision_count": ep.get("unique_movable_collision_count", ""),
                }

                writer.writerow(row)
                episodes_written += 1

    print(f"Wrote CSV: {out_path}")
    print(f"Pickles read: {pkls_read}")
    print(f"Episodes written: {episodes_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
