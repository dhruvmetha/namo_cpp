#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _checkpoint_label(checkpoint_path: str) -> str:
    p = Path(checkpoint_path)
    # Expected layout: <run_dir>/checkpoints/<file>.ckpt
    # Use <run_dir>/<file>.ckpt for readability; fall back to basename.
    try:
        run_dir = p.parents[1].name
        return f"{run_dir}/{p.name}"
    except Exception:
        return p.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input aggregates CSV (default: namo_cpp/eval_results/manifest_test/aggregates_by_checkpoint_and_difficulty.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output summary CSV (default: namo_cpp/eval_results/manifest_test/summary_by_checkpoint_easy_medium_hard.csv)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    in_path = Path(args.input).resolve() if args.input else (repo_root / "namo_cpp/eval_results/manifest_test/aggregates_by_checkpoint_and_difficulty.csv")
    out_path = Path(args.output).resolve() if args.output else (repo_root / "namo_cpp/eval_results/manifest_test/summary_by_checkpoint_easy_medium_hard.csv")

    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")

    # Keep this small & focused: success rates + success-only stats per difficulty.
    # Include `overall` so the reference row doesn't look "perfect" when only
    # viewing easy/medium/hard (those categories are defined by reference success).
    difficulties = ["overall", "easy", "medium", "hard"]
    extra_base_only = ["unsolved_by_reference", "unclassified"]
    base_cols = [
        "n_total",
        "n_solution_found",
        "n_opening_success",
        "solution_found_rate",
        "opening_success_rate",
    ]
    metrics = [
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
    metric_cols = []
    for m in metrics:
        metric_cols.append(f"{m}_mean_success")
        metric_cols.append(f"{m}_median_success")

    selected_cols = base_cols + metric_cols

    rows_by_ckpt_and_diff: dict[tuple[str, str], dict[str, str]] = {}
    checkpoint_paths: set[str] = set()
    with in_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Empty CSV: {in_path}")
        for row in reader:
            ckpt = row.get("checkpoint_path", "")
            diff = row.get("difficulty", "")
            if not ckpt or (diff not in difficulties and diff not in extra_base_only):
                continue
            checkpoint_paths.add(ckpt)
            rows_by_ckpt_and_diff[(ckpt, diff)] = row

    out_header = ["checkpoint_path", "checkpoint_label"]
    for diff in difficulties:
        for col in selected_cols:
            out_header.append(f"{diff}__{col}")
    for diff in extra_base_only:
        for col in base_cols:
            out_header.append(f"{diff}__{col}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()
        for ckpt in sorted(checkpoint_paths):
            out_row: dict[str, str] = {"checkpoint_path": ckpt, "checkpoint_label": _checkpoint_label(ckpt)}
            for diff in difficulties:
                src = rows_by_ckpt_and_diff.get((ckpt, diff))
                for col in selected_cols:
                    out_row[f"{diff}__{col}"] = "" if src is None else src.get(col, "")
            for diff in extra_base_only:
                src = rows_by_ckpt_and_diff.get((ckpt, diff))
                for col in base_cols:
                    out_row[f"{diff}__{col}"] = "" if src is None else src.get(col, "")
            writer.writerow(out_row)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
