"""Render every successful 1-push episode in pass1_collection.

Folder structure:
  /common/users/dm1487/corl2026/namo/videos_v11/
    set1_benchmark_3__run_0153_pair_009/
      obstacle_1_movable.mp4
      obstacle_5_movable.mp4
      obstacle_7_movable.mp4
    set1_benchmark_5__run_0010_pair_008/
      obstacle_2_movable.mp4
    ...
"""
import os, pickle, subprocess, sys
from collections import defaultdict
from pathlib import Path

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--pass-dir", default="/common/users/dm1487/corl2026/namo/pass1_collection")
ap.add_argument("--out-root", default="/common/home/dm1487/robotics_research/ktamp/namo/videos/pass1_v12")
ap.add_argument("--chain-length", type=int, default=1,
                help="Only render episodes whose action_sequence has this length")
args = ap.parse_args()

PASS_DIR = Path(args.pass_dir)
OUT_ROOT = Path(args.out_root)
CHAIN_LEN = args.chain_length
REPO = "/common/home/dm1487/robotics_research/ktamp/namo"
PY = "/common/users/dm1487/envs/mjxrl/bin/python"

# Collect (pkl_path, episode_index_in_pkl, xml_file, action_seq)
tasks = []
for pkl in sorted(PASS_DIR.rglob("*_results.pkl")):
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    eps = data.get("episode_results", [])
    # Index successful episodes of the requested chain length within this pkl
    matched_so_far = -1
    for ep in eps:
        actions = ep.get("action_sequence") or []
        if not actions or len(actions) != CHAIN_LEN:
            continue
        if not ep.get("success"):
            continue
        matched_so_far += 1
        xml = ep.get("xml_file") or ""
        # Video name: combine all obj_ids in chain for multi-push
        obj_id = "+".join(a.get("object_id", "?") for a in actions)
        tasks.append((pkl, matched_so_far, xml, obj_id))

print(f"Total tasks: {len(tasks)}")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

ok = 0; fail = 0
for i, (pkl, ep_idx, xml, obj_id) in enumerate(tasks):
    # Build env-folder name from xml path
    xml_p = Path(xml)
    # e.g. /.../envs_10k/set1/benchmark_5/benchmark_5/run_0153/env_0153_pair_009.xml
    # → set1_benchmark_5__run_0153_pair_009
    parts = xml_p.parts
    # find the set name and run/pair
    try:
        set_name = next(p for p in parts if p.startswith("set"))
        bench_name = next(p for p in parts if p.startswith("benchmark_"))
        run_name = next(p for p in parts if p.startswith("run_"))
        pair_name = xml_p.stem  # env_NNNN_pair_NNN
    except StopIteration:
        env_dir = OUT_ROOT / xml_p.stem
    else:
        env_dir = OUT_ROOT / f"{set_name}_{bench_name}__{run_name}_{pair_name}"
    env_dir.mkdir(parents=True, exist_ok=True)
    qpos = env_dir / f"{obj_id}_qpos.txt"
    mp4 = env_dir / f"{obj_id}.mp4"

    if mp4.exists():
        ok += 1; continue  # idempotent: skip already-rendered

    # Replay
    env_proc = os.environ.copy()
    r = subprocess.run([PY, f"{REPO}/scripts/replay_solution.py",
        "--results-pkl", str(pkl), "--xml", xml,
        "--qpos-out", str(qpos),
        "--chain-length", str(CHAIN_LEN), "--success-only",
        "--episode-idx", str(ep_idx)],
        env=env_proc, capture_output=True, text=True)
    if r.returncode != 0 or not qpos.exists():
        fail += 1
        print(f"[{i+1}/{len(tasks)}] REPLAY FAIL {xml_p.stem} {obj_id}: {r.stderr[:120]}")
        continue

    # Render
    env_proc["MUJOCO_GL"] = "egl"
    r2 = subprocess.run([PY, f"{REPO}/scripts/render_qpos_simple.py",
        xml, str(qpos), str(mp4),
        "--frame-skip", "25", "--width", "480", "--height", "480"],
        env=env_proc, capture_output=True, text=True)
    if r2.returncode != 0 or not mp4.exists():
        fail += 1
        print(f"[{i+1}/{len(tasks)}] RENDER FAIL {xml_p.stem} {obj_id}: {r2.stderr[:120]}")
        continue

    # Clean up qpos dump (keep only mp4)
    qpos.unlink(missing_ok=True)
    ok += 1
    if (i+1) % 20 == 0:
        print(f"[{i+1}/{len(tasks)}] OK so far: {ok}, FAIL: {fail}")

print(f"\nDONE. ok={ok} fail={fail}. Out: {OUT_ROOT}")
