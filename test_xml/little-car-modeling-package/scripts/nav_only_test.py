"""Pure free-space navigation tests in nav_env_3000e.xml.

Uses RLEnvironment.navigate_to (C++ trapezoidal nav, no push). Strategy:

    1. Reset env (car spawns at the default location from XML).
    2. Sample random (x, y) targets in world bounds.
    3. Call navigate_to(x, y, theta). The wavefront returns "no_path" if
       the target is unreachable; we discard those.
    4. For navigable targets, report (success, steps, pos_err, yaw_err).

This skips the question of where to start — we always start where the env
spawned the car — and avoids needing to know obstacle layout in advance.
"""
from __future__ import annotations
import argparse
import math
import os
import random
import sys
from pathlib import Path

REPO = Path("/common/home/dm1487/robotics_research/ktamp/namo")
HOST = os.uname().nodename.split('.')[0]
sys.path.insert(0, str(REPO / f"build_python_mjxrl_{HOST}"))
sys.path.insert(0, str(REPO / "python"))

import namo_rl


def deg(rad: float) -> float:
    return math.degrees(rad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default=str(
        REPO / "test_xml/little-car-modeling-package/artifacts/nav_env_3000e.xml"))
    ap.add_argument("--config", default=str(REPO / "config/namo_config_car.yaml"))
    ap.add_argument("--n", type=int, default=20, help="Number of attempts to sample.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    env = namo_rl.RLEnvironment(args.xml, args.config, False)
    env.reset()
    # 3000e env outer walls are at ±0.7. Use ±0.6 to leave margin from walls.
    xmin, xmax = -0.60, 0.60
    ymin, ymax = -0.60, 0.60
    print(f"Goal sample bounds: x∈[{xmin:.3f},{xmax:.3f}]  y∈[{ymin:.3f},{ymax:.3f}]")

    obs = env.get_observation()
    rp = obs.get("robot_pose", [0, 0, 0])
    print(f"Robot start: ({rp[0]:+.3f}, {rp[1]:+.3f}, {deg(rp[2]):+.2f}°)\n")

    successes = []
    failures_no_path = 0
    failures_nav = []

    attempt = 0
    while attempt < args.n:
        attempt += 1
        # Sample a random goal
        gx = random.uniform(xmin, xmax)
        gy = random.uniform(ymin, ymax)
        gtheta = random.uniform(-math.pi, math.pi)

        # Reset before each test so we always start from the same pose.
        env.reset()

        r = env.navigate_to(gx, gy, gtheta)

        # Discard unreachable goals — they're not testing nav.
        if r.failure_reason == "no_path":
            failures_no_path += 1
            print(f"[{attempt:02d}] goal=({gx:+.3f},{gy:+.3f},{deg(gtheta):+.0f}°)  no_path (skip)")
            continue

        ok = r.success
        line = (f"[{attempt:02d}] goal=({gx:+.3f},{gy:+.3f},{deg(gtheta):+.0f}°)  "
                f"{'OK ' if ok else 'FAIL'}  steps={r.steps_used:5d}  "
                f"pos_err={r.pos_error_m*1000:6.1f}mm  "
                f"yaw_err={deg(r.yaw_error_rad):6.2f}°")
        if not ok:
            line += f"  reason={r.failure_reason!r} col={r.collision_object!r}"
        print(line)

        if ok:
            successes.append(r)
        else:
            failures_nav.append(r)

    # Summary
    n_navigable = len(successes) + len(failures_nav)
    print("\n=== summary ===")
    print(f"  attempts:          {attempt}")
    print(f"  no_path (skipped): {failures_no_path}")
    print(f"  navigable cases:   {n_navigable}")
    print(f"    successes:       {len(successes)}")
    print(f"    nav failures:    {len(failures_nav)}")
    if successes:
        avg_pos = sum(r.pos_error_m for r in successes) / len(successes) * 1000
        avg_yaw = sum(deg(r.yaw_error_rad) for r in successes) / len(successes)
        avg_t   = sum(r.steps_used for r in successes) / len(successes) * 0.01
        print(f"  avg success pos_err: {avg_pos:.1f} mm")
        print(f"  avg success yaw_err: {avg_yaw:.2f}°")
        print(f"  avg success time:    {avg_t:.2f} s")


if __name__ == "__main__":
    main()
