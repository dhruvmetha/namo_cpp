#!/usr/bin/env python3
"""Build a phase-4 manifest by unioning every (env, failed-object) pair from the
pipeline that hasn't yet been retried at K=100. Concretely:

  A. Phase-1 PARTIAL-FAILURE envs — solved by one object, but another object failed.
     That failed object wasn't carried forward to phase-2 (because the env was
     considered solved), so K=100 at depth-2 might still find a 2-push solution.

  B. Phase-2 PARTIAL-FAILURE envs — solved at depth-2 K=20 by one object, but
     another object failed even at depth-2.

  C. Phase-3 PARTIAL-FAILURE envs — solved at depth-2 K=50 by one object, but
     another still failed.

  D. Phase-3 STILL-FAILING envs — never solved in any phase. Excludes
     `goal_region_not_in_snapshot` (goal wavefront-walled-off — can't be fixed
     by deeper search, fundamentally unsolvable).

These four sets are disjoint by construction (an env that succeeds in phase-N
won't enter phase-(N+1)). Together they cover every (env, object) attempt the
pipeline has tried so far that ended in failure with a retryable reason.

Usage:
    python scripts/build_phase4_manifest.py \\
        --phase1-dir /scratch/dm1487/outputs/car_v1_aug9_phase1 \\
        --phase2-dir /scratch/dm1487/outputs/car_v1_aug9_phase2 \\
        --phase3-dir /scratch/dm1487/outputs/car_v1_aug9_phase3 \\
        --output     /scratch/dm1487/manifests/car_envs_v1_aug9_phase4.txt
"""

import argparse
import glob
import os
import pickle
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path


RETRYABLE_REASONS = {
    "all_pushes_failed",
    "no_reachable_objects",
    "no_reachable_edges",
    # Intentionally EXCLUDE "goal_region_not_in_snapshot": goal wavefront-walled-off,
    # deeper search or bigger K can't help.
}


def _walk_pkl(pkl_path: str):
    """Per-PKL worker. Returns list of (xml_path, had_success, fail_reason_or_None)."""
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
    except Exception:
        return []
    out = []
    for ep in d.get("episode_results") or []:
        xml = ep.get("xml_file") or ""
        if not xml:
            continue
        had_success = bool(ep.get("success"))
        r = None
        if not had_success:
            r = (ep.get("algorithm_stats") or {}).get("failure_reason", "unknown")
        out.append((xml, had_success, r))
    return out


def collect_env_state(pkl_dir: Path, workers: int):
    """Aggregate per-env: did any episode succeed, what failure reasons appeared."""
    # Explicit patterns (flat + sharded). Avoid recursive ** which would walk
    # each shard's envs/ symlink forest and stall on networked /scratch.
    pkls = sorted(
        glob.glob(str(pkl_dir / "modular_data_*" / "*_results.pkl"))
        + glob.glob(str(pkl_dir / "shard_*" / "pkls" / "modular_data_*" / "*_results.pkl"))
    )
    print(f"  Scanning {len(pkls)} PKLs in {pkl_dir} with {workers} workers", file=sys.stderr)
    by_env = {}
    with Pool(processes=workers) as pool:
        for chunk in pool.imap_unordered(_walk_pkl, pkls, chunksize=64):
            for xml, had_success, reason in chunk:
                slot = by_env.setdefault(xml, {"success": False, "fail_reasons": Counter()})
                if had_success:
                    slot["success"] = True
                elif reason:
                    slot["fail_reasons"][reason] += 1
    return by_env


def filter_partial_fail(by_env: dict) -> list:
    """Envs that had ≥1 success AND ≥1 retryable failure (partial-failure envs)."""
    return [
        x for x, info in by_env.items()
        if info["success"] and any(r in RETRYABLE_REASONS for r in info["fail_reasons"])
    ]


def filter_all_fail(by_env: dict) -> list:
    """Envs that had NO success and ≥1 retryable failure (still-failing, retryable)."""
    return [
        x for x, info in by_env.items()
        if (not info["success"]) and any(r in RETRYABLE_REASONS for r in info["fail_reasons"])
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-dir", required=True)
    ap.add_argument("--phase2-dir", required=True)
    ap.add_argument("--phase3-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    print("Phase-1 (looking for partial-failure envs):", file=sys.stderr)
    p1 = collect_env_state(Path(args.phase1_dir), args.workers)
    p1_partial = filter_partial_fail(p1)
    print(f"  → {len(p1_partial)} partial-failure envs", file=sys.stderr)

    print("Phase-2 (looking for partial-failure envs):", file=sys.stderr)
    p2 = collect_env_state(Path(args.phase2_dir), args.workers)
    p2_partial = filter_partial_fail(p2)
    print(f"  → {len(p2_partial)} partial-failure envs", file=sys.stderr)

    print("Phase-3 (partial-failure AND still-failing envs):", file=sys.stderr)
    p3 = collect_env_state(Path(args.phase3_dir), args.workers)
    p3_partial = filter_partial_fail(p3)
    p3_allfail = filter_all_fail(p3)
    print(f"  → {len(p3_partial)} partial-failure envs", file=sys.stderr)
    print(f"  → {len(p3_allfail)} still-failing envs (retryable)", file=sys.stderr)

    combined = list(set(p1_partial) | set(p2_partial) | set(p3_partial) | set(p3_allfail))
    combined.sort()
    random.seed(args.seed)
    random.shuffle(combined)

    # Overlap diagnostics (should all be empty by construction)
    sets = {"p1_partial": set(p1_partial), "p2_partial": set(p2_partial),
            "p3_partial": set(p3_partial), "p3_allfail": set(p3_allfail)}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            ov = sets[names[i]] & sets[names[j]]
            if ov:
                print(f"  WARN: {len(ov)} env(s) overlap between {names[i]} and {names[j]} (unexpected)", file=sys.stderr)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for xml in combined:
            f.write(xml + "\n")

    print(f"\nPhase-4 manifest: {len(combined)} envs", file=sys.stderr)
    print(f"  A) phase-1 partial-fail:  {len(p1_partial)}", file=sys.stderr)
    print(f"  B) phase-2 partial-fail:  {len(p2_partial)}", file=sys.stderr)
    print(f"  C) phase-3 partial-fail:  {len(p3_partial)}", file=sys.stderr)
    print(f"  D) phase-3 still-failing: {len(p3_allfail)}", file=sys.stderr)
    print(f"  Written to: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
