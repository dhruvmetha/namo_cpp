#!/usr/bin/env python3
"""Build a phase-2 manifest from phase-1 collection PKLs.

Phase 1 = depth-1 random_rollout with target_goal_region. Most envs succeed.
The ones that don't (failure_reason in {all_pushes_failed, goal_region_not_in_snapshot})
need deeper search (phase 2 = depth-2, K=20).

This script walks a phase-1 OUTPUT_DIR, finds every env whose ALL recorded
episodes failed, and emits one XML path per line into a new manifest.

Usage:
    python scripts/build_phase2_manifest.py \\
        --phase1-dir /scratch/dm1487/outputs/car_v1_aug9_phase1 \\
        --output     /scratch/dm1487/manifests/car_envs_v1_aug9_phase2.txt

Notes:
    - Reads `*_results.pkl` recursively from --phase1-dir
    - Envs with zero episodes (robot already in goal region) are excluded — they
      need no further work
    - Envs with at least one successful episode are excluded — phase 1 covered them
    - Remaining envs get one line in the manifest, sorted, then shuffled with the
      same seed=42 the original manifest uses (so phase 2 is reproducible too)
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


def _walk_pkl(pkl_path: str):
    """Per-PKL worker. Returns (xml_path, had_success_bool, fail_reasons_list, n_episodes)."""
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
    except Exception as e:
        return [("__error__", False, [f"load_error:{e}"], 0)]
    episodes = d.get("episode_results") or []
    if not episodes:
        return [("__no_eps__", False, [], 0)]
    out = []
    for ep in episodes:
        xml = ep.get("xml_file") or ""
        if not xml:
            continue
        had_success = bool(ep.get("success"))
        r = None
        if not had_success:
            stats = ep.get("algorithm_stats") or {}
            r = stats.get("failure_reason", "unknown")
        out.append((xml, had_success, [r] if r else [], 1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase1-dir", required=True,
                        help="Directory containing phase-1 modular_data_*/*.pkl files")
    parser.add_argument("--output", required=True,
                        help="Destination manifest file (one XML path per line)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Shuffle seed for reproducibility (default 42, matches main manifest)")
    parser.add_argument("--include-reasons", nargs="+",
                        default=["all_pushes_failed", "goal_region_not_in_snapshot",
                                 "no_reachable_objects", "no_reachable_edges"],
                        help="Which failure_reason values count as phase-2 candidates")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                        help="Parallel worker processes (default: all cores)")
    args = parser.parse_args()

    phase1_dir = Path(args.phase1_dir)
    if not phase1_dir.is_dir():
        print(f"ERROR: phase1-dir not a directory: {phase1_dir}", file=sys.stderr)
        return 1

    pkls = sorted(glob.glob(str(phase1_dir / "modular_data_*" / "*_results.pkl")))
    print(f"Scanning {len(pkls)} phase-1 PKL files in {phase1_dir} with {args.workers} workers", file=sys.stderr)

    include = set(args.include_reasons)
    by_env = {}
    reason_counter = Counter()
    envs_with_no_episodes = 0
    total_episodes = 0

    with Pool(processes=args.workers) as pool:
        for chunk in pool.imap_unordered(_walk_pkl, pkls, chunksize=64):
            for xml, had_success, reasons, n_eps in chunk:
                if xml == "__no_eps__":
                    envs_with_no_episodes += 1
                    continue
                if xml == "__error__":
                    print(f"  WARN: {reasons[0]}", file=sys.stderr)
                    continue
                total_episodes += n_eps
                slot = by_env.setdefault(xml, {"success": False, "fail_reasons": Counter()})
                if had_success:
                    slot["success"] = True
                else:
                    for r in reasons:
                        slot["fail_reasons"][r] += 1
                        reason_counter[r] += 1

    n_envs_total = len(by_env)
    phase2_candidates = []
    for xml, info in by_env.items():
        if info["success"]:
            continue  # phase 1 covered it
        # All recorded episodes failed; check if any of the failure reasons match
        if any(r in include for r in info["fail_reasons"]):
            phase2_candidates.append(xml)

    phase2_candidates.sort()
    random.seed(args.seed)
    random.shuffle(phase2_candidates)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for xml in phase2_candidates:
            f.write(xml + "\n")

    print(f"  Total episodes:          {total_episodes}", file=sys.stderr)
    print(f"  Envs with ≥1 episode:    {n_envs_total}", file=sys.stderr)
    print(f"  Envs skipped (no eps):   {envs_with_no_episodes}", file=sys.stderr)
    print(f"  Phase-2 candidates:      {len(phase2_candidates)} ({100*len(phase2_candidates)/max(n_envs_total,1):.1f}% of envs with episodes)", file=sys.stderr)
    print(f"  Failure-reason breakdown across all failed episodes:", file=sys.stderr)
    for r, c in reason_counter.most_common():
        marker = "✓" if r in include else "✗"
        print(f"    {marker} {c:6d}  {r}", file=sys.stderr)
    print(f"  Manifest written: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
