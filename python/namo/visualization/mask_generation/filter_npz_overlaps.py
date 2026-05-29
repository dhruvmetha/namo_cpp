#!/usr/bin/env python3
"""Move NPZs with robot_region/goal_sample_region overlap out of phase mask dirs.

Mirrors what `batch_collection --filter-overlaps` does, but as a post-process:
for each NPZ we already wrote, check if the planner-view robot region overlaps
the goal_sample_region (= degenerate "already there" cases). If yes, move the
NPZ to a sibling dir `<phase_mask_dir>_overlap_filtered/`, preserving its
relative path inside the env subdir.

Run via SLURM. Picker scripts that glob phase mask dirs only see clean NPZs.

Usage:
    python filter_npz_overlaps.py <phase_mask_dir> [<phase_mask_dir> ...]
        --workers 32 [--dry-run]
"""
import argparse
import glob
import os
import shutil
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np


def _check_npz(npz_path):
    """Return (has_overlap, missing_keys) for one NPZ. Errors -> (None, True).

    Gate matches batch_collection.py has_region_overlap():
    robot_region ∩ goal_sample_region (the broader gate — the region where
    the planner sampled goals). Catches cases where the wavefront-reachable
    set from the robot's pose already includes the goal region.
    """
    try:
        with np.load(npz_path) as d:
            if "robot_region" not in d.files or "goal_sample_region" not in d.files:
                return (None, True)
            rr = d["robot_region"]
            gr = d["goal_sample_region"]
            return (bool(np.any((rr > 0.5) & (gr > 0.5))), False)
    except Exception:
        return (None, True)


def _process_one(args):
    """Worker: classify + optionally move one NPZ.

    Returns one of "overlap", "clean", "missing", "moved", "dry_overlap".
    """
    npz_path, src_root, dst_root, dry_run = args
    has_overlap, missing = _check_npz(npz_path)
    if missing:
        return "missing"
    if not has_overlap:
        return "clean"
    # Move to mirror path under dst_root
    rel = os.path.relpath(npz_path, src_root)
    dst = os.path.join(dst_root, rel)
    if dry_run:
        return "dry_overlap"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.move(npz_path, dst)
        return "moved"
    except Exception:
        return "move_error"


def filter_dir(src_root, workers, dry_run):
    """Walk one phase mask dir, classify NPZs, move overlapping ones to sibling
    `<src_root>_overlap_filtered/` preserving relative structure."""
    if not os.path.isdir(src_root):
        print(f"  SKIP (not a dir): {src_root}", file=sys.stderr)
        return Counter()
    npzs = glob.glob(os.path.join(src_root, "**", "*.npz"), recursive=True)
    if not npzs:
        print(f"  {src_root}: 0 NPZs", file=sys.stderr)
        return Counter()

    dst_root = src_root.rstrip("/") + "_overlap_filtered"
    if dry_run:
        print(f"  [DRY] {src_root}: {len(npzs)} NPZs", file=sys.stderr)
    else:
        print(f"  {src_root}: {len(npzs)} NPZs -> overlap moved to {dst_root}",
              file=sys.stderr)
        os.makedirs(dst_root, exist_ok=True)

    args = [(p, src_root, dst_root, dry_run) for p in npzs]
    counts = Counter()
    with Pool(processes=workers) as pool:
        for r in pool.imap_unordered(_process_one, args, chunksize=128):
            counts[r] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+",
                    help="Phase mask dirs (e.g. /scratch/dm1487/outputs/v3_phase1_masks)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true",
                    help="Count overlap but don't move files")
    args = ap.parse_args()

    print(f"workers={args.workers} dry_run={args.dry_run} dirs={len(args.dirs)}",
          file=sys.stderr)
    grand = Counter()
    per_dir = {}
    for d in args.dirs:
        c = filter_dir(d, args.workers, args.dry_run)
        per_dir[d] = c
        grand.update(c)

    print("\n=== per-dir summary ===")
    print(f"{'dir':70s} {'total':>8s} {'overlap':>8s} {'clean':>8s} {'missing':>8s}")
    for d, c in per_dir.items():
        tot = c["clean"] + c["moved"] + c["dry_overlap"] + c["missing"] + c["move_error"]
        ov = c["moved"] + c["dry_overlap"]
        cl = c["clean"]
        ms = c["missing"]
        pct = 100 * ov / max(tot, 1)
        print(f"{os.path.basename(d):70s} {tot:8d} {ov:8d} ({pct:4.1f}%) {cl:8d} {ms:8d}")
    print("\n=== grand totals ===")
    print(dict(grand))


if __name__ == "__main__":
    sys.exit(main())
