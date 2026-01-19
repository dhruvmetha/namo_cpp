#!/usr/bin/env python3
"""Aggregate per-(env, region, object) statistics from `all_attempts_with_ckpt.csv`.

Outputs:
  - aggregates_by_checkpoint_and_difficulty.csv
  - failure_reasons_by_checkpoint_and_difficulty.csv

Difficulty classification matches `scripts/eval_1push.py`:
  ratio = solutions_total_for_neighbour / pushes_total_for_neighbour  (reference only)
  easy   if ratio > easy_threshold
  medium if ratio > hard_threshold
  hard   otherwise

Important: the raw exported CSV may contain multiple rows for the same
`(xml_file, neighbour_region_label, chosen_object_id)` because the reference
planner can record multiple successful openings per neighbour/object.

To match the evaluation granularity used by `scripts/eval_1push.py`, this script
deduplicates to the *first* row per `(checkpoint_path, results_pkl, neighbour_region_label, chosen_object_id)`
(i.e., the first AttemptResult per object within a given environment pickle).

Reference checkpoint is auto-detected as:
  1) explicit sentinel "REFERENCE_*" (preferred)
  2) legacy `/namo/models/1_push_model/` path
You can override with --reference-checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_bool(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    # Fallback: non-empty string → True
    return True


def _parse_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_int(val: Any) -> Optional[int]:
    f = _parse_float(val)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.fmean(values)


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.median(values)


@dataclass
class MetricBucket:
    values_all: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    values_success: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    n_total: int = 0
    n_solution_found: int = 0
    n_opening_success: int = 0
    failure_reasons_all: Counter = field(default_factory=Counter)
    failure_reasons_failures: Counter = field(default_factory=Counter)

    def add_row(self, row: Dict[str, str], opening_success: bool, solution_found: bool) -> None:
        self.n_total += 1
        if solution_found:
            self.n_solution_found += 1
        if opening_success:
            self.n_opening_success += 1

        failure_reason = (row.get("failure_reason") or "").strip() or "unknown"
        self.failure_reasons_all[failure_reason] += 1
        if not opening_success:
            self.failure_reasons_failures[failure_reason] += 1

        # Numeric metrics (per-episode)
        metrics: Dict[str, Optional[float]] = {
            "search_time_ms": _parse_float(row.get("search_time_ms")),
            "pushes_total_for_neighbour": _parse_float(row.get("pushes_total_for_neighbour")),
            "solution_depth": _parse_float(row.get("solution_depth")),
            "chain_depth": _parse_float(row.get("chain_depth")),
            "total_cost": _parse_float(row.get("total_cost")),
            "solutions_total_for_neighbour": _parse_float(row.get("solutions_total_for_neighbour")),
            "solutions_found_for_neighbour": _parse_float(row.get("solutions_found_for_neighbour")),
            "ml_goals_generated": _parse_float(row.get("ml_goals_generated")),
            "ml_goals_aligned": _parse_float(row.get("ml_goals_aligned")),
            "reachable_edges_count": _parse_float(row.get("reachable_edges_count")),
            # Collisions
            "any_wall_collision": 1.0 if _parse_bool(row.get("any_wall_collision")) else 0.0,
            "unique_movable_collision_count": _parse_float(row.get("unique_movable_collision_count")),
        }

        for name, val in metrics.items():
            if val is None:
                continue
            self.values_all[name].append(val)
            if opening_success:
                self.values_success[name].append(val)


def _detect_reference_checkpoint(rows: List[Dict[str, str]]) -> str:
    candidates = sorted({r.get("checkpoint_path", "") for r in rows if r.get("checkpoint_path")})
    # Prefer an explicit, non-ML "oracle" reference run if present.
    for c in candidates:
        if c in {"REFERENCE_PRIMITIVE", "REFERENCE_ORACLE"}:
            return c
    for c in candidates:
        if c.startswith("REFERENCE_"):
            return c
    for c in candidates:
        if "/namo/models/1_push_model/" in c:
            return c
    return candidates[0] if candidates else ""


def _difficulty_from_ratio(ratio: float, easy_th: float, hard_th: float) -> str:
    if ratio > easy_th:
        return "easy"
    if ratio > hard_th:
        return "medium"
    return "hard"

def _dedup_first_per_env_object(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Mimic eval_1push.py's per-file key de-dup (first AttemptResult wins)."""
    seen: set[Tuple[str, str, str, str]] = set()
    out: List[Dict[str, str]] = []
    for row in rows:
        ckpt = row.get("checkpoint_path", "")
        pkl = row.get("results_pkl", "")
        region = row.get("neighbour_region_label", "")
        obj = row.get("chosen_object_id", "")
        key = (ckpt, pkl, region, obj)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="eval_results/manifest_test/all_attempts_with_ckpt.csv",
        help="CSV produced by export_manifest_test_results_csv.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/manifest_test",
        help="Directory to write aggregated CSVs into.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path used as the reference for difficulty categorization.",
    )
    parser.add_argument("--easy-threshold", type=float, default=0.75)
    parser.add_argument("--hard-threshold", type=float, default=0.25)
    parser.add_argument(
        "--subset",
        type=str,
        choices=["all", "reference_success", "reference_success_intersection"],
        default="all",
        help=(
            "Which subset to aggregate. "
            "`all` aggregates everything. "
            "`reference_success` keeps only (env,region,object) triplets that the reference opened. "
            "`reference_success_intersection` further restricts to triplets present in ALL checkpoints (eval_1push-style)."
        ),
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable per-(results_pkl, region, object) de-duplication (not recommended).",
    )
    parser.add_argument(
        "--exclude-checkpoint-substr",
        action="append",
        default=["/namo/models/1_push_model/"],
        help="Exclude any rows whose checkpoint_path contains this substring. "
             "Repeatable. Default excludes the legacy 1_push_model checkpoint.",
    )
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise SystemExit(f"ERROR: input not found: {in_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f))

    excludes = [s for s in (args.exclude_checkpoint_substr or []) if s]
    if excludes:
        rows = [
            r for r in rows
            if not any(ex in (r.get("checkpoint_path") or "") for ex in excludes)
        ]

    if not rows:
        raise SystemExit(f"ERROR: no rows in {in_path}")

    if not args.no_dedup:
        rows = _dedup_first_per_env_object(rows)

    reference_ckpt = args.reference_checkpoint or _detect_reference_checkpoint(rows)
    if not reference_ckpt:
        raise SystemExit("ERROR: could not detect reference checkpoint (no checkpoint_path values?)")

    # Optional: restrict to reference-success subset
    if args.subset != "all":
        # First build the reference-success key set from *deduped* rows.
        ref_success_keys: set[Tuple[str, str, str]] = set()
        for row in rows:
            if row.get("checkpoint_path") != reference_ckpt:
                continue
            solution_found = _parse_bool(row.get("success"))
            pushes = _parse_int(row.get("pushes_total_for_neighbour")) or 0
            opening_success = bool(solution_found and pushes > 0)
            if not opening_success:
                continue
            key = (
                row.get("xml_file", ""),
                row.get("neighbour_region_label", ""),
                row.get("chosen_object_id", ""),
            )
            ref_success_keys.add(key)

        subset_keys = set(ref_success_keys)
        if args.subset == "reference_success_intersection":
            keys_by_ckpt: Dict[str, set[Tuple[str, str, str]]] = defaultdict(set)
            for row in rows:
                ckpt = row.get("checkpoint_path") or ""
                if not ckpt:
                    continue
                key = (
                    row.get("xml_file", ""),
                    row.get("neighbour_region_label", ""),
                    row.get("chosen_object_id", ""),
                )
                keys_by_ckpt[ckpt].add(key)
            all_key_sets = [s for s in keys_by_ckpt.values() if s]
            intersection_keys = set.intersection(*all_key_sets) if all_key_sets else set()
            subset_keys = subset_keys & intersection_keys

        rows = [
            r
            for r in rows
            if (
                r.get("xml_file", ""),
                r.get("neighbour_region_label", ""),
                r.get("chosen_object_id", ""),
            )
            in subset_keys
        ]

        if not rows:
            raise SystemExit(f"ERROR: subset={args.subset} produced 0 rows")

    # Build reference mapping: (xml, region, object) -> (ratio, opening_success, difficulty)
    ref_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        if row.get("checkpoint_path") != reference_ckpt:
            continue
        key = (
            row.get("xml_file", ""),
            row.get("neighbour_region_label", ""),
            row.get("chosen_object_id", ""),
        )
        pushes = _parse_int(row.get("pushes_total_for_neighbour")) or 0
        solutions_total = _parse_int(row.get("solutions_total_for_neighbour")) or 0
        solution_found = _parse_bool(row.get("success"))
        opening_success = bool(solution_found and pushes > 0)
        ratio = (solutions_total / pushes) if pushes > 0 else 0.0

        # With de-dup on, each key should appear exactly once; keep first.
        if key not in ref_map:
            ref_map[key] = {"opening_success": opening_success, "ratio": ratio}

    for key, info in ref_map.items():
        info["difficulty"] = _difficulty_from_ratio(
            float(info.get("ratio", 0.0)),
            easy_th=args.easy_threshold,
            hard_th=args.hard_threshold,
        )

    # Aggregate
    buckets: Dict[Tuple[str, str], MetricBucket] = defaultdict(MetricBucket)

    def bucket_for(ckpt: str, difficulty: str) -> MetricBucket:
        return buckets[(ckpt, difficulty)]

    for row in rows:
        ckpt = row.get("checkpoint_path", "")
        if not ckpt:
            continue

        solution_found = _parse_bool(row.get("success"))
        pushes = _parse_int(row.get("pushes_total_for_neighbour")) or 0
        opening_success = bool(solution_found and pushes > 0)

        key = (
            row.get("xml_file", ""),
            row.get("neighbour_region_label", ""),
            row.get("chosen_object_id", ""),
        )
        ref_info = ref_map.get(key)
        if ref_info is None:
            difficulty = "unclassified"
        elif ref_info.get("opening_success"):
            difficulty = str(ref_info.get("difficulty", "unclassified"))
        else:
            difficulty = "unsolved_by_reference"

        # overall + per difficulty
        bucket_for(ckpt, "overall").add_row(row, opening_success=opening_success, solution_found=solution_found)
        bucket_for(ckpt, difficulty).add_row(row, opening_success=opening_success, solution_found=solution_found)

    # Write aggregate metrics
    suffix = "" if args.subset == "all" else f"__subset_{args.subset}"
    agg_path = out_dir / f"aggregates_by_checkpoint_and_difficulty{suffix}.csv"
    metric_names = [
        "search_time_ms",
        "pushes_total_for_neighbour",
        "solution_depth",
        "chain_depth",
        "total_cost",
        "solutions_total_for_neighbour",
        "solutions_found_for_neighbour",
        "ml_goals_generated",
        "ml_goals_aligned",
        "reachable_edges_count",
        "any_wall_collision",
        "unique_movable_collision_count",
    ]

    fieldnames = [
        "checkpoint_path",
        "difficulty",
        "n_total",
        "n_solution_found",
        "n_opening_success",
        "solution_found_rate",
        "opening_success_rate",
    ]
    for name in metric_names:
        fieldnames.extend(
            [
                f"{name}_mean_all",
                f"{name}_median_all",
                f"{name}_mean_success",
                f"{name}_median_success",
            ]
        )

    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (ckpt, diff), b in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
            row_out: Dict[str, Any] = {
                "checkpoint_path": ckpt,
                "difficulty": diff,
                "n_total": b.n_total,
                "n_solution_found": b.n_solution_found,
                "n_opening_success": b.n_opening_success,
                "solution_found_rate": (b.n_solution_found / b.n_total) if b.n_total else 0.0,
                "opening_success_rate": (b.n_opening_success / b.n_total) if b.n_total else 0.0,
            }
            for name in metric_names:
                vals_all = b.values_all.get(name, [])
                vals_succ = b.values_success.get(name, [])
                row_out[f"{name}_mean_all"] = _mean(vals_all)
                row_out[f"{name}_median_all"] = _median(vals_all)
                row_out[f"{name}_mean_success"] = _mean(vals_succ)
                row_out[f"{name}_median_success"] = _median(vals_succ)
            w.writerow(row_out)

    # Write failure reason counts
    failures_path = out_dir / f"failure_reasons_by_checkpoint_and_difficulty{suffix}.csv"
    all_reasons: List[str] = sorted(
        {reason for b in buckets.values() for reason in b.failure_reasons_all.keys()}
    )

    fail_fields = [
        "checkpoint_path",
        "difficulty",
        "n_total",
        "n_opening_success",
        "n_failures",
    ] + [f"count_all__{r}" for r in all_reasons] + [f"count_failures__{r}" for r in all_reasons]

    with open(failures_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fail_fields)
        w.writeheader()
        for (ckpt, diff), b in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
            out: Dict[str, Any] = {
                "checkpoint_path": ckpt,
                "difficulty": diff,
                "n_total": b.n_total,
                "n_opening_success": b.n_opening_success,
                "n_failures": b.n_total - b.n_opening_success,
            }
            for r in all_reasons:
                out[f"count_all__{r}"] = b.failure_reasons_all.get(r, 0)
                out[f"count_failures__{r}"] = b.failure_reasons_failures.get(r, 0)
            w.writerow(out)

    print(f"Reference checkpoint: {reference_ckpt}")
    print(f"Subset: {args.subset}")
    print(f"Wrote: {agg_path}")
    print(f"Wrote: {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
