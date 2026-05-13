#!/usr/bin/env python3
"""Run a real NAMO push on the car and observe rotation rebound from qpos dump.

Captures both:
  - per-tick qpos via NAMO_QPOS_DUMP (yaw history + active/wait split)
  - segment plan via NAMO_NAV_LOG=1 stderr ([NAV_DEBUG] lines)

so each rotation phase is reported with: start heading, target heading
(from the C++ planner), target endpoint, exit yaw, post-wait yaw, and rebound.

Usage:
    python observe_nav_rebound.py [--xml PATH] [--object NAME] [--edge IDX]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/common/home/dm1487/robotics_research/ktamp/namo")
HOST = os.uname().nodename.split('.')[0]
BUILD_DIR = REPO_ROOT / f"build_python_mjxrl_{HOST}"
sys.path.insert(0, str(BUILD_DIR))
sys.path.insert(0, str(REPO_ROOT / "python"))

import namo_rl  # noqa: E402


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def find_car_freejoint_qpos_adr(xml_path: str) -> int:
    """Use mujoco to find the qpos address of car_freejoint.
    Other movable obstacles also have freejoints, so we cannot assume 0.
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml_path)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")
    if jid < 0:
        raise RuntimeError("car_freejoint not found in model")
    return int(m.jnt_qposadr[jid])


def parse_qpos_dump(path: Path, car_qpos_adr: int):
    """Return list of (phase_id, x, y, yaw) per dumped frame, indexing
    into qpos at the car freejoint position (x,y,z,qw,qx,qy,qz)."""
    out = []
    base = 2 + car_qpos_adr  # tokens: [phase, nq, qpos_0, qpos_1, ...]
    with path.open() as f:
        for line in f:
            toks = line.split()
            if len(toks) < base + 7:
                continue
            phase = int(toks[0])
            x = float(toks[base + 0])
            y = float(toks[base + 1])
            qw = float(toks[base + 3])
            qx = float(toks[base + 4])
            qy = float(toks[base + 5])
            qz = float(toks[base + 6])
            yaw = quat_to_yaw(qw, qx, qy, qz)
            out.append((phase, x, y, yaw))
    return out


def parse_navlog_segments(navlog_path: Path):
    """Parse [NAV_DEBUG] lines emitted by DiffDriveNavigation.

    Returns:
      segments: list of dicts with keys {idx, end_x, end_y, heading_rad}
      target_object: str (from "[NAV_DEBUG] target_object=... target_body=...")
    """
    segments = []
    target_object = ""
    if not navlog_path.exists():
        return segments, target_object
    with navlog_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("[NAV_DEBUG] target_object="):
                # [NAV_DEBUG] target_object=X target_body=Y
                parts = line.split()
                for p in parts:
                    if p.startswith("target_object="):
                        target_object = p.split("=", 1)[1]
            # Lines look like:  "    seg 0: end=(0.123,0.456) heading=1.5708 rad"
            if "seg " in line and "end=(" in line and "heading=" in line:
                # Extract idx
                try:
                    idx = int(line.split("seg")[1].split(":")[0].strip())
                    end_str = line.split("end=(")[1].split(")")[0]
                    ex, ey = (float(v) for v in end_str.split(","))
                    h = float(line.split("heading=")[1].split()[0])
                    segments.append(dict(idx=idx, end_x=ex, end_y=ey, heading_rad=h))
                except Exception:
                    continue
    return segments, target_object


def split_into_runs(samples):
    """Split flat sample list into contiguous same-phase runs.
    Returns list of (phase_id, start_idx, end_idx_exclusive).
    """
    runs = []
    if not samples:
        return runs
    cur_phase = samples[0][0]
    start = 0
    for i, s in enumerate(samples[1:], start=1):
        if s[0] != cur_phase:
            runs.append((cur_phase, start, i))
            cur_phase = s[0]
            start = i
    runs.append((cur_phase, start, len(samples)))
    return runs


def find_active_wait_split(yaws):
    """Within a rotation run, the active phase ends and wait phase begins
    when commanded ω drops to 0. Mechanically this shows up as a sharp
    yaw-rate sign flip (rebound kick) followed by near-zero rates. We
    detect by finding the tick of largest negative-direction rate change
    OR the first sign flip of dyaw.
    Returns split_idx so [0, split_idx) is active and [split_idx, N) is wait.
    """
    if len(yaws) < 3:
        return len(yaws)
    dyaws = [wrap(yaws[i + 1] - yaws[i]) for i in range(len(yaws) - 1)]
    # Determine main rotation direction
    sgn = 1.0 if sum(dyaws[: max(1, len(dyaws) // 4)]) >= 0 else -1.0
    # Find first index where dyaw reverses sign relative to main direction
    for i in range(1, len(dyaws)):
        if sgn * dyaws[i] < -1e-5:
            return i + 1
    return len(yaws)


def deg(r: float) -> float:
    return math.degrees(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default=str(
        REPO_ROOT / "test_xml/little-car-modeling-package/artifacts/nav_env.xml"))
    ap.add_argument("--config", default=str(REPO_ROOT / "config/namo_config_car.yaml"))
    ap.add_argument("--object", default=None,
                    help="If unset, picks first reachable object.")
    ap.add_argument("--edge", type=int, default=None,
                    help="Edge index (auto-picks middle reachable if unset).")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--qpos", default="/tmp/observe_rebound.qpos")
    ap.add_argument("--csv", default="/tmp/observe_rebound.csv")
    args = ap.parse_args()

    qpos_path = Path(args.qpos)
    if qpos_path.exists():
        qpos_path.unlink()
    navlog_path = Path(args.qpos).with_suffix(".navlog")
    if navlog_path.exists():
        navlog_path.unlink()

    # IMPORTANT: do NOT set NAMO_FORCE_TELEPORT_NAV — we want DiffDriveNavigation.
    os.environ["NAMO_QPOS_DUMP"] = args.qpos
    os.environ["NAMO_NAV_LOG"] = "1"

    env = namo_rl.RLEnvironment(args.xml, args.config, False)
    env.reset()
    env.set_collision_checking(False)

    # Redirect C++ stderr (where [NAV_DEBUG] / [NAV_PATH] / [NAV_POSE] go) to a log file.
    saved_stderr_fd = os.dup(2)
    log_fd = os.open(str(navlog_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log_fd, 2)
    os.close(log_fd)

    reachable = env.get_reachable_objects()
    print(f"Loaded: {args.xml}")
    print(f"Reachable: {reachable}")

    obj = args.object or (reachable[0] if reachable else None)
    if obj is None:
        print("No reachable objects — aborting.", file=sys.stderr)
        sys.exit(1)
    edges = env.get_reachable_edges(obj)
    if not edges:
        print(f"No reachable edges for {obj}", file=sys.stderr)
        sys.exit(1)
    edge = args.edge if args.edge in edges else edges[len(edges) // 2]

    action = namo_rl.Action()
    action.object_id = obj
    action.edge_idx = edge
    action.depth = args.depth
    action.x = action.y = action.theta = 0.0

    print(f"Pushing object={obj}  edge={edge}  depth={args.depth}")
    sys.stdout.flush()
    result = env.step(action)
    # Restore stderr
    os.dup2(saved_stderr_fd, 2)
    os.close(saved_stderr_fd)
    print(f"  done={result.done} reward={result.reward:.3f} "
          f"info={dict(result.info)}")
    print(f"  navlog: {navlog_path}")

    car_adr = find_car_freejoint_qpos_adr(args.xml)
    print(f"car_freejoint qpos adr = {car_adr}")
    samples = parse_qpos_dump(qpos_path, car_adr)
    print(f"\nqpos dump frames: {len(samples)}")

    segments, target_obj_log = parse_navlog_segments(navlog_path)
    print(f"\nPlanner segments ({len(segments)} total) "
          f"[from NAV_DEBUG, target_object='{target_obj_log}']:")
    for s in segments:
        print(f"  seg {s['idx']}: end=({s['end_x']:+.4f}, {s['end_y']:+.4f})  "
              f"heading={deg(s['heading_rad']):+.4f} deg")

    # Compute the final-rotate target: for the push at edge `edge` of object
    # `obj`, the C++ controller computes push_theta = angle from edge_point to
    # mid_point (line 299–302 of namo_push_controller.cpp). We can derive it
    # via the obj pose + size + edge_idx, but a simpler proxy is: the final
    # rotate is into the push direction, which is roughly toward the object
    # center from the contact face. Read the object's pose and the car's pre-
    # push position from the trace: the angle from car (just before phase 3)
    # to the object center is the push target.
    # Find last sample of phase 2 and first sample of phase 3 to extract
    # the actual achieved heading at start of push.
    push_target = None
    for ph, x, y, yaw in samples:
        if ph == 3:
            # First phase-3 frame's yaw is the post-final-rotate heading
            push_target = ("first phase-3 yaw (achieved)", deg(yaw))
            break

    runs = split_into_runs(samples)
    print(f"Phase runs (phase_id, frames):")
    for ph, s, e in runs:
        print(f"  phase={ph}  frames=[{s}..{e})  count={e - s}")

    # Analyze each rotation phase (phase 0 = segment rotate, phase 2 = final rotate)
    # Phase 1 = drive_straight_to, skip.
    print("\n=== Rotation phases ===")
    rotation_runs = [r for r in runs if r[0] in (0, 2) and (r[2] - r[1]) > 5]
    # Map phase-0 rotation runs to planner segments in order. The first phase-0
    # run wraps the initial sample + first segment-rotate; subsequent phase-0
    # runs each correspond to the next segment rotate.
    seg_idx_for_run: list[int | None] = []
    next_seg = 0
    for ph, s, e in rotation_runs:
        if ph == 0:
            seg_idx_for_run.append(next_seg if next_seg < len(segments) else None)
            next_seg += 1
        else:
            seg_idx_for_run.append(None)  # phase-2 final rotate: target is push_theta
    if not rotation_runs:
        print("(no rotation phases logged)")
    csv_lines = ["run_idx,frame_in_run,phase,kind,t_s,x,y,yaw_deg\n"]
    for ridx, (ph, s, e) in enumerate(rotation_runs):
        yaws = [samples[i][3] for i in range(s, e)]
        split = find_active_wait_split(yaws)
        active_yaws = yaws[:split]
        wait_yaws = yaws[split:]

        if not active_yaws:
            continue

        target_anchor = active_yaws[0]
        exit_yaw = active_yaws[-1]
        final_yaw = wait_yaws[-1] if wait_yaws else exit_yaw
        rotated = wrap(exit_yaw - target_anchor)
        rebound = wrap(exit_yaw - final_yaw)

        print(f"\n  [run {ridx}] phase_id={ph}  frames={e - s}  "
              f"(active={len(active_yaws)}, wait={len(wait_yaws)})")
        # Where the car was at start of this rotation
        x0, y0 = samples[s][1], samples[s][2]
        x_exit, y_exit = samples[s + len(active_yaws) - 1][1], samples[s + len(active_yaws) - 1][2]
        x_final, y_final = samples[e - 1][1], samples[e - 1][2]
        print(f"    pos at run start : ({x0:+.4f}, {y0:+.4f})")
        print(f"    pos at exit      : ({x_exit:+.4f}, {y_exit:+.4f})")
        print(f"    pos after wait   : ({x_final:+.4f}, {y_final:+.4f})")
        # Target heading from planner
        seg_i = seg_idx_for_run[ridx]
        if ph == 0 and seg_i is not None and seg_i < len(segments):
            seg = segments[seg_i]
            target_h = seg["heading_rad"]
            print(f"    planner segment  : seg {seg_i}  "
                  f"target_end=({seg['end_x']:+.4f}, {seg['end_y']:+.4f})  "
                  f"target_heading={deg(target_h):+.4f} deg")
            print(f"    headed-to delta  : exit {deg(wrap(target_h - exit_yaw)):+.4f} deg, "
                  f"final {deg(wrap(target_h - final_yaw)):+.4f} deg")
        elif ph == 2 and push_target is not None:
            print(f"    final rotate     : push_theta target inferred from "
                  f"{push_target[0]} = {push_target[1]:+.4f} deg")
        print(f"    yaw at run start : {deg(target_anchor):+.4f} deg")
        print(f"    yaw at exit      : {deg(exit_yaw):+.4f} deg  "
              f"(rotated {deg(rotated):+.4f})")
        print(f"    yaw after wait   : {deg(final_yaw):+.4f} deg  "
              f"(rebound {deg(rebound):+.4f})")
        # Sample first 3 wait ticks for the kick
        if len(wait_yaws) >= 1:
            kicks = []
            prev = exit_yaw
            for k, y in enumerate(wait_yaws[:5]):
                dy = wrap(y - prev)
                kicks.append(f"     wait[{k}]: yaw={deg(y):+.4f}  dyaw={deg(dy):+.4f}")
                prev = y
            print("    first wait ticks:")
            for line in kicks:
                print(line)

        for i, y in enumerate(active_yaws):
            f = s + i
            csv_lines.append(f"{ridx},{i},{ph},active,"
                             f"{f * 0.01:.4f},{samples[f][1]:.6f},{samples[f][2]:.6f},"
                             f"{deg(y):.6f}\n")
        for i, y in enumerate(wait_yaws):
            f = s + len(active_yaws) + i
            csv_lines.append(f"{ridx},{len(active_yaws) + i},{ph},wait,"
                             f"{f * 0.01:.4f},{samples[f][1]:.6f},{samples[f][2]:.6f},"
                             f"{deg(y):.6f}\n")

    Path(args.csv).write_text("".join(csv_lines))
    print(f"\nCSV trace: {args.csv}")
    print(f"qpos dump: {args.qpos}")


if __name__ == "__main__":
    main()
