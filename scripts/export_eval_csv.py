#!/usr/bin/env python3
"""Export per-triplet results from RSS eval pkls into tidy CSVs for replotting.

Usage:
    python scripts/export_eval_csv.py --config scripts/eval_1push_config.yaml --out 1_push_eval/raw.csv
    python scripts/export_eval_csv.py --config scripts/eval_2push_config.yaml --out 2_push_eval/raw.csv

Output columns:
    model, env, region, object, success, pushes, time_ms,
    solved_in_phase, wall_collision, movable_collisions, failure_reason, source_dir

`model` follows the yaml `name:` field. Reference (oracle) entries are
prefixed `ref::` so you can filter them out for planner comparisons.
"""
import argparse
import csv
import pickle
from glob import glob
from pathlib import Path

import yaml


def iter_triplets(data_dir: str):
    pattern = str(data_dir).rstrip("/") + "/**/*.pkl"
    for f in glob(pattern, recursive=True):
        if "collection_summary" in f:
            continue
        try:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
        except Exception as e:
            print(f"skip {f}: {e}")
            continue
        for ep in data.get("episode_results", []):
            xml = ep.get("xml_file", "")
            if "easy" in xml:  # mirrors eval_*.py default
                continue
            alg = ep.get("algorithm_stats", {}) or {}
            region = alg.get("neighbour_region_label")
            obj = alg.get("chosen_object_id", "")
            if region is None or not obj:
                continue
            pushes = alg.get("pushes_total_for_neighbour", 0) or 0
            yield {
                "env": xml,
                "region": region,
                "object": obj,
                "success": bool(ep.get("solution_found", False)) and pushes > 0,
                "pushes": pushes,
                "time_ms": ep.get("search_time_ms", 0.0) or 0.0,
                "solved_in_phase": alg.get("solved_in_phase", "") or "",
                "wall_collision": bool(ep.get("any_wall_collision", False)),
                "movable_collisions": ep.get("unique_movable_collision_count", 0) or 0,
                "failure_reason": alg.get("failure_reason", "") or "",
            }


def export(config_path: str, out_path: str):
    cfg = yaml.safe_load(open(config_path))
    models = []
    ref = cfg.get("reference")
    if isinstance(ref, list):
        for r in ref:
            models.append(("ref::" + r["name"], r["dir"]))
    elif isinstance(ref, dict):
        models.append(("ref::" + ref["name"], ref["dir"]))
    for b in cfg.get("baselines", []) or []:
        models.append((b["name"], b["dir"]))
    for m in cfg.get("learned", []) or []:
        models.append((m["name"], m["dir"]))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "env", "region", "object", "success", "pushes", "time_ms",
            "solved_in_phase", "wall_collision", "movable_collisions",
            "failure_reason", "source_dir"]
    n_total = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name, d in models:
            seen = set()  # dedup per (env, region, object) within a model
            n = 0
            for row in iter_triplets(d):
                k = (row["env"], row["region"], row["object"])
                if k in seen:
                    continue
                seen.add(k)
                row["model"] = name
                row["source_dir"] = d
                w.writerow(row)
                n += 1
            print(f"{name:50s} {n:6d} rows  ({d})")
            n_total += n
    print(f"\nwrote {n_total} rows to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export(args.config, args.out)
