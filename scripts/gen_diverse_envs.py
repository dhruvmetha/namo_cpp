"""Driver: generate a diverse car-env dataset across templates × density modes.

For each (template, density_mode) pair invokes generate_envs.py with
mode-specific (num_objects, object_size_range) and a fresh start_seed.
Tallies the result.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/common/home/dm1487/robotics_research/ktamp/namo")
GENERATOR = "/common/home/dm1487/robotics_research/ktamp/mujoco_env_creator/generate_envs.py"
NAMO_CFG = REPO / "config/namo_config_car.yaml"
TEMPLATES_ROOT = REPO / "templates/aug9_car"
PYTHONPATH = f"{REPO}/build_python_mjxrl_{os.uname().nodename.split('.')[0]}:{REPO}/python"
PY = "/common/users/dm1487/envs/mjxrl/bin/python"

# Density modes — varies obstacle count AND obstacle size simultaneously.
MODES = {
    "sparse": dict(num_objects=6,  size_range=(0.09, 0.18)),
    "medium": dict(num_objects=10, size_range=(0.12, 0.25)),
    "dense":  dict(num_objects=15, size_range=(0.18, 0.33)),
}


def run_generator(template_xml: Path, mode_name: str, mode_cfg: dict,
                  num_envs: int, seed: int, output_dir: Path,
                  num_workers: int = 4) -> int:
    """Invoke generate_envs.py once. Returns number of env XML files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, GENERATOR, str(template_xml),
        "--namo-config", str(NAMO_CFG),
        "--robot-scale", "0.233",
        "--num-envs", str(num_envs),
        "--num-objects", str(mode_cfg["num_objects"]),
        "--object-size-range", str(mode_cfg["size_range"][0]), str(mode_cfg["size_range"][1]),
        "--output-dir", str(output_dir),
        "--num-workers", str(num_workers),
        "--start-seed", str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    proc = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    # parse last "Generated N environment file(s)" line
    last = ""
    for line in proc.stdout.splitlines():
        if "Generated" in line and "environment file" in line:
            last = line
    if not last:
        # Stderr may have errors
        return 0
    import re
    m = re.search(r"Generated (\d+) environment file", last)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-total", type=int, default=100,
                    help="Approximate total envs to produce across all (template, mode) cells.")
    ap.add_argument("--output-root", type=Path,
                    default=Path("/common/home/dm1487/scratch_namo/diverse_car_envs"))
    ap.add_argument("--start-seed", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    templates = []
    for s in ("set1", "set2"):
        for b in range(1, 6):
            t = TEMPLATES_ROOT / s / f"benchmark_{b}.xml"
            if t.exists():
                templates.append((s, b, t))

    n_cells = len(templates) * len(MODES)
    target_per_cell = max(1, args.target_total // n_cells)
    # Empirical avg pairs/job across cells from the trial run was ~12. Use that to
    # compute --num-envs per cell so the produced count tracks the target.
    EMPIRICAL_PAIRS_PER_JOB = 12
    base_num_envs = max(1, (target_per_cell + EMPIRICAL_PAIRS_PER_JOB - 1) // EMPIRICAL_PAIRS_PER_JOB)

    print(f"Diverse generation: {len(templates)} templates × {len(MODES)} modes = {n_cells} cells")
    print(f"Target ≈ {args.target_total} envs total (~{target_per_cell} per cell)")
    print(f"--num-envs per cell: {base_num_envs} (jobs)\n")

    if args.output_root.exists():
        print(f"WARNING: {args.output_root} exists; new files will accumulate alongside old ones")
    args.output_root.mkdir(parents=True, exist_ok=True)

    seed = args.start_seed
    grand_total = 0
    rows = []
    for (s, b, template_xml) in templates:
        for mode_name, mode_cfg in MODES.items():
            out_dir = args.output_root / s / mode_name
            n = run_generator(template_xml, mode_name, mode_cfg,
                              base_num_envs, seed, out_dir, args.num_workers)
            seed += base_num_envs + 1
            grand_total += n
            rows.append((s, b, mode_name, n))
            print(f"  {s}/benchmark_{b}/{mode_name:6} → {n} envs")

    print()
    print(f"Total: {grand_total} envs at {args.output_root}")
    print()
    print(f"{'template':<30} {'sparse':>8} {'medium':>8} {'dense':>8}")
    by_t = {}
    for s, b, m, n in rows:
        key = f"{s}/benchmark_{b}"
        by_t.setdefault(key, {})[m] = n
    for key in sorted(by_t):
        c = by_t[key]
        print(f"  {key:<30} {c.get('sparse',0):>8} {c.get('medium',0):>8} {c.get('dense',0):>8}")


if __name__ == "__main__":
    main()
