"""Generate set2/benchmark_3 car envs bucketed by exact region-graph hop count.

Drives mujoco_env_creator/generate_envs.py with `--exact-hop K` for K=1,2,3.
For each hop bucket we sample num_objects uniformly from {K, K+1, ..., max_objects}
(N < K can never realize K hops since region count <= N+1).

Output layout:
    <output-root>/hop_1/run_XXXX/env_XXXX_pair_XXX.xml
    <output-root>/hop_2/run_XXXX/env_XXXX_pair_XXX.xml
    <output-root>/hop_3/run_XXXX/env_XXXX_pair_XXX.xml

Usage (smoke, ~100/bucket):
    python scripts/gen_car_envs_set2bench3_hop.py \
        --output-root /common/home/dm1487/scratch_namo/set2_bench3_hop_smoke
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
TEMPLATE = REPO / "templates/aug9_car/set2/benchmark_3.xml"
PY = "/common/users/dm1487/envs/mjxrl/bin/python"

OBJECT_SIZE_RANGE = (0.09, 0.5)


def count_xmls(root: Path) -> int:
    return sum(1 for _ in root.rglob("env_*.xml"))


def run_generator(num_objects: int, seed_start: int, batch_size: int,
                  output_dir: Path, num_workers: int, exact_hop: int,
                  size_range: Tuple[float, float],
                  clearance_radius: float = 0.02,
                  timeout_s: int = 1200) -> int:
    cmd = [
        PY, GENERATOR, str(TEMPLATE),
        "--namo-config", str(NAMO_CFG),
        "--robot-scale", "0.233",
        "--num-objects", str(num_objects),
        "--num-envs", str(batch_size),
        "--output-dir", str(output_dir),
        "--num-workers", str(num_workers),
        "--start-seed", str(seed_start),
        "--object-size-range", str(size_range[0]), str(size_range[1]),
        "--clearance-radius", str(clearance_radius),
        "--exact-hop", str(exact_hop),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(REPO),
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout_s}s for seed {seed_start}")
        return 0
    if proc.returncode != 0:
        print(f"    WARN: rc={proc.returncode}  stderr-tail: {proc.stderr[-300:]}")
    m = re.search(r"Generated (\d+) environment file", proc.stdout)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True,
                    help="Root dir; per-hop subdirs hop_1/hop_2/hop_3 are created.")
    ap.add_argument("--target-per-hop", type=int, default=100,
                    help="XML target per hop bucket (default 100 for smoke).")
    ap.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-objects", type=int, default=3,
                    help="Upper bound on sampled num_objects.")
    ap.add_argument("--batch-seeds", type=int, default=100,
                    help="Seeds per generator invocation.")
    ap.add_argument("--num-workers", type=int, default=32)
    ap.add_argument("--start-seed", type=int, default=200000)
    ap.add_argument("--max-seconds", type=int, default=14400,
                    help="Wallclock bailout for the whole run.")
    ap.add_argument("--clearance-radius", type=float, default=0.02)
    ap.add_argument("--rng-seed", type=int, default=42,
                    help="Seed for the num_objects sampler.")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        sys.exit(f"template not found: {TEMPLATE}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.rng_seed)

    print(f"Template: {TEMPLATE}")
    print(f"Hops: {args.hops}, target per hop: {args.target_per_hop}")
    print(f"max_objects={args.max_objects}, batch_seeds={args.batch_seeds}, workers={args.num_workers}")
    print(f"Output root: {args.output_root}\n")

    overall_start = time.time()
    seed = args.start_seed

    for K in args.hops:
        hop_dir = args.output_root / f"hop_{K}"
        hop_dir.mkdir(parents=True, exist_ok=True)
        n_choices = list(range(K, args.max_objects + 1))
        if not n_choices:
            print(f"[hop_{K}] no valid num_objects (need N >= K={K} <= max={args.max_objects}); skipping")
            continue

        current = count_xmls(hop_dir)
        print(f"[hop_{K}] starting at {current}/{args.target_per_hop}, "
              f"sampling N from {n_choices}")
        batch_idx = 0
        stale = 0  # consecutive zero-yield batches; trigger early bail

        while current < args.target_per_hop:
            elapsed = time.time() - overall_start
            if elapsed > args.max_seconds:
                print(f"[hop_{K}] TIMEOUT at {current}/{args.target_per_hop}")
                break

            N = rng.choice(n_choices)
            before = count_xmls(hop_dir)
            seed_used = seed
            seed += args.batch_seeds
            run_generator(
                num_objects=N,
                seed_start=seed_used,
                batch_size=args.batch_seeds,
                output_dir=hop_dir,
                num_workers=args.num_workers,
                exact_hop=K,
                size_range=OBJECT_SIZE_RANGE,
                clearance_radius=args.clearance_radius,
            )
            after = count_xmls(hop_dir)
            delta = after - before
            current = after
            batch_idx += 1
            stale = 0 if delta > 0 else stale + 1

            print(f"  [t={elapsed:.0f}s hop={K} batch={batch_idx} N={N}] "
                  f"seeds {seed_used}..{seed_used + args.batch_seeds - 1}: "
                  f"+{delta} total={current}/{args.target_per_hop}")

            if stale >= 20:
                print(f"[hop_{K}] BAIL: 20 consecutive zero-yield batches; "
                      f"hop={K} likely infeasible on this template at N<={args.max_objects}")
                break

        print(f"[hop_{K}] done at {current} XMLs in {time.time() - overall_start:.0f}s\n")

    total = sum(count_xmls(args.output_root / f"hop_{K}") for K in args.hops)
    print("=" * 60)
    print(f"DONE. Total: {total} XMLs across hops {args.hops} "
          f"in {time.time() - overall_start:.0f}s")
    for K in args.hops:
        print(f"  hop_{K}: {count_xmls(args.output_root / f'hop_{K}')}")


if __name__ == "__main__":
    main()
