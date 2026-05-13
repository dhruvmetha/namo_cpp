"""Diverse car env generation: randomly sample num_objects ∈ {3..10} per seed,
round-robin across templates, until a global XML target is reached.

No per-num_objects bucket targets — the dataset's distribution over num_objects
emerges from uniform random sampling weighted by per-N yield (higher N tends to
yield more region pairs per seed, so the final dataset skews toward high N).

Usage (smoke):
    python gen_car_envs_diverse.py --total-target 1000 --output-root /tmp/smoke

Usage (full 100k):
    python gen_car_envs_diverse.py --total-target 100000
"""
import argparse
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

REPO = Path("/common/home/dm1487/robotics_research/ktamp/namo")
GENERATOR = "/common/home/dm1487/robotics_research/ktamp/mujoco_env_creator/generate_envs.py"
NAMO_CFG = REPO / "config/namo_config_car.yaml"
TEMPLATES_ROOT = REPO / "templates/aug9_car"
PY = "/common/users/dm1487/envs/mjxrl/bin/python"

# Object size range (min, max in meters, box side length).
OBJECT_SIZE_RANGE = (0.09, 0.5)


def count_xmls(root: Path) -> int:
    return sum(1 for _ in root.rglob("env_*.xml"))


def run_generator(template_xml: Path, num_objects: int, seed_start: int,
                  batch_size: int, output_dir: Path, num_workers: int,
                  size_range: Tuple[float, float],
                  clearance_radius: float = 0.02,
                  timeout_s: int = 1200) -> int:
    cmd = [
        PY, GENERATOR, str(template_xml),
        "--namo-config", str(NAMO_CFG),
        "--robot-scale", "0.233",
        "--num-objects", str(num_objects),
        "--num-envs", str(batch_size),
        "--output-dir", str(output_dir),
        "--num-workers", str(num_workers),
        "--start-seed", str(seed_start),
        "--object-size-range", str(size_range[0]), str(size_range[1]),
        "--clearance-radius", str(clearance_radius),
    ]
    env = os.environ.copy()
    # generate_envs.py now uses WavefrontSnapshotExporter.from_geometry — no
    # namo_rl / MuJoCo import on the hot path, so PYTHONPATH/LD_PRELOAD aren't
    # required for the pure-Python generation flow.
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(REPO),
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout_s}s for seed {seed_start}")
        return 0
    if proc.returncode != 0:
        print(f"    WARN: rc={proc.returncode}  stderr-tail: {proc.stderr[-200:]}")
    m = re.search(r"Generated (\d+) environment file", proc.stdout)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path,
                    default=Path("/common/home/dm1487/scratch_namo/car_envs_100k"))
    ap.add_argument("--total-target", type=int, default=100000,
                    help="Total XML target (global, no per-bucket sub-targets)")
    ap.add_argument("--num-workers", type=int, default=64,
                    help="Workers passed to each generator invocation")
    ap.add_argument("--batch-seeds", type=int, default=200,
                    help="Seeds per generator invocation (each batch uses one random N). "
                         "Bigger amortizes subprocess startup over more work.")
    ap.add_argument("--start-seed", type=int, default=100000)
    ap.add_argument("--max-seconds", type=int, default=86400,
                    help="Bail out of the whole run after this many wallclock seconds")
    ap.add_argument("--min-num-objects", type=int, default=3)
    ap.add_argument("--max-num-objects", type=int, default=10)
    ap.add_argument("--clearance-radius", type=float, default=0.02,
                    help="Required obstacle-free disc around robot/goal cells (m). "
                         "Default 0.02m (2cm buffer beyond car's 5cm half-extent).")
    ap.add_argument("--rng-seed", type=int, default=42,
                    help="Seed for the (num_objects, template) sampler")
    args = ap.parse_args()

    templates = []
    for s in ("set1", "set2"):
        for b in range(1, 6):
            t = TEMPLATES_ROOT / s / f"benchmark_{b}.xml"
            if t.exists():
                templates.append((s, b, t))
    if not templates:
        sys.exit("no templates found under " + str(TEMPLATES_ROOT))

    rng = random.Random(args.rng_seed)
    n_choices = list(range(args.min_num_objects, args.max_num_objects + 1))

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Templates: {len(templates)}")
    print(f"num_objects sampled uniformly from {n_choices}")
    print(f"Total target: {args.total_target}")
    print(f"Batch seeds per call: {args.batch_seeds}, --num-workers {args.num_workers}")
    print(f"Output root: {args.output_root}")
    print()

    seed = args.start_seed
    overall_start = time.time()
    current = count_xmls(args.output_root)
    n_stats = {n: 0 for n in n_choices}
    batch_idx = 0

    # Stratified sampling: enumerate every (template, N) bucket once, shuffle
    # the order, then iterate round-robin. Pass 0 guarantees each bucket sees
    # one batch before any bucket sees two. Repeat passes until target hit.
    buckets = [(t, n) for t in templates for n in n_choices]
    rng.shuffle(buckets)
    print(f"Stratified buckets: {len(buckets)} ({len(templates)} templates × {len(n_choices)} N values)")
    print()

    pass_idx = 0
    while current < args.total_target:
        for (template_tuple, N) in buckets:
            if current >= args.total_target:
                break
            elapsed = time.time() - overall_start
            if elapsed > args.max_seconds:
                print(f"\nTIMEOUT: hit {args.max_seconds}s with {current}/{args.total_target}")
                break

            s, b, template = template_tuple
            out_dir = args.output_root / s / f"benchmark_{b}"
            out_dir.mkdir(parents=True, exist_ok=True)

            before = count_xmls(args.output_root)
            seed_used = seed
            seed += args.batch_seeds
            run_generator(template, N, seed_used, args.batch_seeds, out_dir, args.num_workers,
                          OBJECT_SIZE_RANGE, clearance_radius=args.clearance_radius)
            after = count_xmls(args.output_root)
            delta = after - before
            current = after
            n_stats[N] += delta
            batch_idx += 1

            ratio = delta / max(1, args.batch_seeds)
            print(f"  [t={elapsed:.0f}s pass={pass_idx} batch={batch_idx}] N={N} {s}/benchmark_{b} "
                  f"seeds {seed_used}..{seed_used + args.batch_seeds - 1}: "
                  f"+{delta} ({ratio:.2f}/seed) total={current}/{args.total_target}")
        else:
            pass_idx += 1
            continue
        break  # we broke out of the for-loop (target hit or timeout); break outer too

    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"DONE. Total: {current} XMLs in {overall_elapsed:.0f}s ({current/max(1,overall_elapsed):.2f} XMLs/sec)")
    print()
    print(f"{'num_objects':>12}  {'count':>10}  {'fraction':>10}")
    for N in n_choices:
        frac = n_stats[N] / max(1, current)
        print(f"{N:>12}  {n_stats[N]:>10}  {frac*100:>9.1f}%")


if __name__ == "__main__":
    main()
