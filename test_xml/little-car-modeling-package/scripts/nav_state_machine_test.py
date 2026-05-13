"""Pure free-space nav test — point A to point B in the 3000e env, no pushes.

Mirrors the C++ DiffDriveNavigation::execute state machine:
  segment_path(path, sharp_turn_threshold)
  for each segment:
      rotate_trap(env, segment.heading)
      drive_trap(env, segment.endpoint, segment.heading)
  rotate_trap(env, target_theta)

Path is hand-specified (or a simple straight line) — the wavefront planner
isn't invoked here; we just want to test the state machine itself with the
trapezoidal controllers.

Dumps qpos for video render.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

# Reuse rotate_trap / drive_trap from the prototype
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pd_prototype import (rotate_trap, drive_trap, make_scene, settle,
                           get_yaw, get_xy, get_yaw_rate, dump_qpos_line,
                           wrap, n_substeps, step_tick)


def segment_path(waypoints, heading_threshold=0.35):
    """Group consecutive waypoints with similar heading into straight segments.
    Mirrors C++ segment_path.
    Returns list of dicts: {end_xy, heading}.
    """
    segments = []
    if len(waypoints) < 2:
        return segments
    cur_heading = math.atan2(waypoints[1][1] - waypoints[0][1],
                              waypoints[1][0] - waypoints[0][0])
    seg_end = 1
    for i in range(1, len(waypoints) - 1):
        h = math.atan2(waypoints[i + 1][1] - waypoints[i][1],
                        waypoints[i + 1][0] - waypoints[i][0])
        if abs(wrap(h - cur_heading)) > heading_threshold:
            segments.append(dict(end_xy=waypoints[i], heading=cur_heading))
            cur_heading = h
            seg_end = i + 1
        else:
            seg_end = i + 1
    segments.append(dict(end_xy=waypoints[seg_end], heading=cur_heading))
    return segments


def run_nav(scene, waypoints, target_theta, qpos_path: Path):
    """Run the state machine: rotate-drive per segment, then final rotate.
    Dumps qpos for every tick (rotation phases use phase_id=0, drive=1,
    final-rotate=2).
    """
    # Open a single qpos file that all phases append to.
    f = qpos_path.open("w")
    m, d = scene["model"], scene["data"]

    # Track total ticks across all phases for the trace
    state = dict(t=0, file=f)

    def make_qpos_appender(phase_id):
        """Wrap the prototype's rotate/drive so they dump to our shared file
        instead of opening their own."""
        return None  # we let them write their own files, we'll concatenate

    segs = segment_path(waypoints)
    print(f"Path has {len(waypoints)} waypoints, {len(segs)} segments.")
    for i, s in enumerate(segs):
        print(f"  seg {i}: end=({s['end_xy'][0]:+.3f}, {s['end_xy'][1]:+.3f})  "
              f"heading={math.degrees(s['heading']):+.2f}°")
    print(f"  final target_theta = {math.degrees(target_theta):+.2f}°")

    # We can't trivially reuse rotate_trap/drive_trap (they open their own
    # files). Inline the loops here, writing to the shared file with the
    # right phase_id per phase.

    b = scene["wheelbase"]; r = scene["wheel_r"]
    LA, RA = scene["left_act"], scene["right_act"]
    n_sub = n_substeps(m)
    dt = 0.01

    # Saturation / convergence
    omega_max = 1.0; alpha_max = 5.0; theta_converged = 0.01
    v_max = 0.10; accel_max = 0.5; xy_converged = 0.005
    K_heading = 2.0
    max_phase_steps = 6000

    def append_qpos(phase_id):
        f.write(dump_qpos_line(m, d, phase_id))

    def rotate_to(target_theta, phase_id):
        nonlocal state
        steps = 0
        while steps < max_phase_steps:
            yaw = get_yaw(scene)
            err = wrap(target_theta - yaw)
            append_qpos(phase_id)
            if abs(err) < theta_converged:
                break
            sgn = 1.0 if err >= 0 else -1.0
            w_des = math.sqrt(2.0 * alpha_max * abs(err))
            w_des = min(w_des, omega_max)
            omega_cmd = sgn * w_des
            d.ctrl[LA] = (-omega_cmd * b / 2.0) / r
            d.ctrl[RA] = (+omega_cmd * b / 2.0) / r
            step_tick(m, d, n_sub)
            steps += 1
        return steps

    K_heading_p = 0.0
    K_heading_d = 0.0
    def drive_to(end_xy, heading):
        steps = 0
        cos_h = math.cos(heading); sin_h = math.sin(heading)
        while steps < max_phase_steps:
            x, y = get_xy(scene)
            yaw = get_yaw(scene)
            yaw_rate = get_yaw_rate(scene)
            dx = end_xy[0] - x; dy = end_xy[1] - y
            along = dx * cos_h + dy * sin_h
            append_qpos(1)
            if along < xy_converged:
                break
            v_des = min(math.sqrt(2.0 * accel_max * max(along, 0.0)), v_max)
            heading_err = wrap(heading - yaw)
            # PD heading correction: P pulls toward heading, D damps yaw_rate.
            omega_corr_raw = K_heading_p * heading_err - K_heading_d * yaw_rate
            omega_corr = max(-0.5 * omega_max,
                              min(0.5 * omega_max, omega_corr_raw))
            wL = (v_des - omega_corr * b / 2.0) / r
            wR = (v_des + omega_corr * b / 2.0) / r
            d.ctrl[LA] = wL; d.ctrl[RA] = wR
            step_tick(m, d, n_sub)
            steps += 1
        return steps

    total_ticks = 0
    for i, seg in enumerate(segs):
        print(f"  -> rotate to seg {i} heading {math.degrees(seg['heading']):+.2f}°")
        n = rotate_to(seg["heading"], 0)
        print(f"     rotate ticks: {n}, yaw now {math.degrees(get_yaw(scene)):+.3f}°")
        total_ticks += n

        print(f"  -> drive to seg {i} end {seg['end_xy']}")
        n = drive_to(seg["end_xy"], seg["heading"])
        x, y = get_xy(scene)
        print(f"     drive ticks: {n}, pos now ({x:+.4f}, {y:+.4f})")
        total_ticks += n

    print(f"  -> final rotate to {math.degrees(target_theta):+.2f}°")
    n = rotate_to(target_theta, 2)
    print(f"     final rotate ticks: {n}, yaw now {math.degrees(get_yaw(scene)):+.3f}°")
    total_ticks += n

    # End pose
    x, y = get_xy(scene)
    yaw = get_yaw(scene)
    target_xy = waypoints[-1]
    pos_err = math.hypot(target_xy[0] - x, target_xy[1] - y)
    yaw_err = math.degrees(wrap(target_theta - yaw))

    print(f"\n=== Final ===")
    print(f"  Total ticks: {total_ticks}  ({total_ticks * dt:.2f} s)")
    print(f"  End pos: ({x:+.4f}, {y:+.4f})  target: ({target_xy[0]:+.4f}, {target_xy[1]:+.4f})  err: {pos_err*1000:.2f} mm")
    print(f"  End yaw: {math.degrees(yaw):+.3f}°  target: {math.degrees(target_theta):+.3f}°  err: {yaw_err:+.3f}°")

    f.close()
    return dict(total_ticks=total_ticks, pos_err_mm=pos_err * 1000, yaw_err_deg=yaw_err)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default=str(
        PROJECT_ROOT / "artifacts/nav_env_3000e.xml"))
    ap.add_argument("--start_x", type=float, default=-0.40)
    ap.add_argument("--start_y", type=float, default=0.40)
    ap.add_argument("--start_theta_deg", type=float, default=0.0)
    ap.add_argument("--waypoints", type=str, default=None,
                    help="Override path: 'x1,y1 x2,y2 ...'. If unset, "
                         "uses scenario A→B based on --scenario.")
    ap.add_argument("--scenario", default="straight",
                    choices=["straight", "Lturn", "Sturn"],
                    help="Pre-baked test scenarios when --waypoints unset")
    ap.add_argument("--target_theta_deg", type=float, default=None,
                    help="Final heading. Default: same as last segment's heading.")
    ap.add_argument("--qpos", type=str, default="/tmp/nav_test.qpos")
    args = ap.parse_args()

    sc = make_scene(xml_override=args.xml,
                    car_xy=(args.start_x, args.start_y))

    # Set initial heading
    fj = sc["fj_qpos"]
    yaw = math.radians(args.start_theta_deg)
    sc["data"].qpos[fj + 3] = math.cos(yaw / 2)
    sc["data"].qpos[fj + 6] = math.sin(yaw / 2)
    mujoco.mj_forward(sc["model"], sc["data"])
    settle(sc, n_steps=500)

    # Build waypoints
    if args.waypoints:
        waypoints = []
        for p in args.waypoints.split():
            x, y = p.split(",")
            waypoints.append((float(x), float(y)))
    else:
        # Pre-baked scenarios
        if args.scenario == "straight":
            # Straight line in open region (no obstacles between)
            waypoints = [(args.start_x, args.start_y), (0.40, args.start_y)]
        elif args.scenario == "Lturn":
            # L-shape: drive +x, turn, drive +y
            waypoints = [(args.start_x, args.start_y),
                         (0.20, args.start_y),
                         (0.20, args.start_y - 0.30)]
        elif args.scenario == "Sturn":
            # S-shape: zigzag
            waypoints = [(args.start_x, args.start_y),
                         (-0.10, args.start_y),
                         (-0.10, args.start_y - 0.20),
                         (0.20, args.start_y - 0.20)]

    # Default target_theta = last segment's heading
    if args.target_theta_deg is None:
        target_theta = math.atan2(waypoints[-1][1] - waypoints[-2][1],
                                   waypoints[-1][0] - waypoints[-2][0])
    else:
        target_theta = math.radians(args.target_theta_deg)

    r = run_nav(sc, waypoints, target_theta, Path(args.qpos))


if __name__ == "__main__":
    main()
