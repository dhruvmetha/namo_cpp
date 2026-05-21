#!/usr/bin/env python3
"""Run a few teleport+push sequences on the car and dump qpos for video render.

Each step is: instant SE(2) teleport to the chosen edge point, then
diff-drive wheel push.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/common/home/dm1487/robotics_research/ktamp/namo")
BUILD_DIR = REPO_ROOT / f"build_python_mjxrl_{os.uname().nodename.split('.')[0]}"
sys.path.insert(0, str(BUILD_DIR))
sys.path.insert(0, str(REPO_ROOT / "python"))

import namo_rl  # noqa: E402


def run(xml: str, config: str, qpos_out: str, path_out: str | None,
        episodes: list[tuple[str, int, int]]) -> None:
    os.environ["NAMO_QPOS_DUMP"] = qpos_out
    if path_out:
        os.environ["NAMO_NAV_LOG"] = "1"
    if Path(qpos_out).exists():
        Path(qpos_out).unlink()

    # NAV_LOG goes to stderr; pipe it to a file via tee in caller if needed.

    env = namo_rl.RLEnvironment(xml, config, False)
    env.reset()
    env.set_collision_checking(False)  # let pushes complete even if mild contact

    print(f"Loaded env: {xml}")
    print(f"Reachable objects (initial): {env.get_reachable_objects()}")

    for i, (obj, edge_idx, depth) in enumerate(episodes):
        reachable = env.get_reachable_objects()
        if obj not in reachable:
            print(f"[{i}] SKIP {obj}: not reachable. reachable={reachable}")
            continue
        edges = env.get_reachable_edges(obj)
        if edge_idx not in edges:
            # fall back to a reachable edge
            if not edges:
                print(f"[{i}] SKIP {obj}: no reachable edges")
                continue
            chosen = edges[len(edges) // 2]
            print(f"[{i}] edge {edge_idx} not reachable for {obj} "
                  f"({len(edges)} reachable). Using {chosen} instead.")
            edge_idx = chosen

        action = namo_rl.Action()
        action.object_id = obj
        action.edge_idx = edge_idx
        action.depth = depth
        action.x = action.y = action.theta = 0.0
        result = env.step(action)
        info_str = ", ".join(f"{k}={v}" for k, v in result.info.items())
        print(f"[{i}] push {obj} edge={edge_idx} depth={depth} → "
              f"done={result.done} reward={result.reward:.3f} {info_str}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(
        REPO_ROOT / "test_xml/little-car-modeling-package/artifacts/nav_env_3000e.xml"))
    ap.add_argument("--config", default=str(REPO_ROOT / "config/namo_config_car.yaml"))
    ap.add_argument("--qpos", default="/tmp/teleport_push_demo.qpos")
    ap.add_argument("--path-log", default="/tmp/teleport_push_demo.navlog")
    args = ap.parse_args()

    # A few hand-picked episodes: different objects, different edges/depths.
    # edge_idx is auto-clamped to a reachable one if needed.
    episodes = [
        ("obstacle_3_movable", 8,  3),   # nearest big box, side push
        ("obstacle_3_movable", 24, 3),   # same box, opposite side
        ("obstacle_2_movable", 8,  2),   # different box, short push
        ("obstacle_4_movable", 16, 3),   # different box, side push
    ]

    run(args.xml, args.config, args.qpos, args.path_log, episodes)
    print(f"\nqpos dump: {args.qpos}")
    print("Render with:")
    print(f"  python {REPO_ROOT}/test_xml/little-car-modeling-package/scripts/"
          f"render_nav_video.py {args.xml} {args.qpos} /tmp/teleport_push_demo.mp4")


if __name__ == "__main__":
    main()
