#!/usr/bin/env python3
"""Render a top-down 2D video of a captured nav trajectory, annotating
each rotation phase with planner target heading and observed rebound.

CPU-only (matplotlib + ffmpeg), no MuJoCo render, no GPU required.

Inputs:
    --xml SCENE_XML            (for walls + initial obstacle poses)
    --qpos QPOS_DUMP           (per-tick "phase nq qpos..." from NAMO_QPOS_DUMP)
    --navlog NAV_DEBUG_LOG     (optional, for planner segment headings)
    --output OUT.mp4

Output is a single panel: top-down env, planned segment endpoints (green dots),
car body rectangle colored by current phase, heading arrow, and a HUD
overlay showing phase, current yaw, and planner-target yaw for rotation phases.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.transforms import Affine2D


PHASE_COLOR = {
    0: "#3b82f6",  # blue: segment rotate
    1: "#22c55e",  # green: drive straight
    2: "#ef4444",  # red: final rotate
    3: "#f97316",  # orange: push
}
PHASE_LABEL = {
    0: "rotate (segment)",
    1: "drive straight",
    2: "rotate (final/push aim)",
    3: "push",
}


def parse_qpos(path: Path):
    out = []
    with path.open() as f:
        for line in f:
            toks = line.split()
            if len(toks) < 3:
                continue
            phase = int(toks[0])
            nq = int(toks[1])
            q = [float(v) for v in toks[2:2 + nq]]
            out.append((phase, q))
    return out


def find_car_qpos_adr(xml_path: Path) -> int:
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")
    return int(m.jnt_qposadr[jid])


def quat_to_yaw(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def parse_walls(xml_path: Path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    walls = []
    wb = root.find(".//body[@name='walls']")
    if wb is not None:
        for g in wb.findall("geom"):
            pos = [float(v) for v in g.get("pos", "0 0 0").split()]
            size = [float(v) for v in g.get("size", "0 0 0").split()]
            walls.append((pos[0], pos[1], size[0], size[1]))
    return walls


def parse_obstacles(xml_path: Path):
    """Return list of (name, x, y, theta, sx, sy) from XML initial poses."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    out = []
    wb = root.find("worldbody")
    if wb is None:
        return out
    for body in wb.findall("body"):
        name = body.get("name", "")
        if not name.endswith("_movable"):
            continue
        g = body.find("geom")
        if g is None:
            continue
        pos = [float(v) for v in g.get("pos", "0 0 0").split()]
        size = [float(v) for v in g.get("size", "0 0 0").split()]
        eul = g.get("euler", "0 0 0").split()
        theta = float(eul[2]) * math.pi / 180.0 if len(eul) >= 3 else 0.0
        out.append((name, pos[0], pos[1], theta, size[0], size[1]))
    return out


def parse_navlog(navlog_path: Path):
    segs = []
    if not navlog_path or not navlog_path.exists():
        return segs
    with navlog_path.open() as f:
        for line in f:
            line = line.strip()
            if "seg " in line and "end=(" in line and "heading=" in line:
                try:
                    idx = int(line.split("seg")[1].split(":")[0].strip())
                    end_str = line.split("end=(")[1].split(")")[0]
                    ex, ey = (float(v) for v in end_str.split(","))
                    h = float(line.split("heading=")[1].split()[0])
                    segs.append((idx, ex, ey, h))
                except Exception:
                    continue
    return segs


def split_into_runs(samples):
    runs = []
    if not samples:
        return runs
    cur = samples[0][0]
    s = 0
    for i, ph in enumerate(samples[1:], start=1):
        if ph != cur:
            runs.append((cur, s, i))
            cur = ph
            s = i
    runs.append((cur, s, len(samples)))
    return runs


def find_active_wait_split(yaws):
    if len(yaws) < 3:
        return len(yaws)
    dyaws = [yaws[i + 1] - yaws[i] for i in range(len(yaws) - 1)]
    # unwrap each
    for i, d in enumerate(dyaws):
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        dyaws[i] = d
    sgn = 1.0 if sum(dyaws[: max(1, len(dyaws) // 4)]) >= 0 else -1.0
    for i in range(1, len(dyaws)):
        if sgn * dyaws[i] < -1e-5:
            return i + 1
    return len(yaws)


def build_target_per_frame(samples, segs):
    """For each frame, compute (target_yaw_deg, label) for HUD.
    Phase 0: planner segment heading (in order of phase-0 runs).
    Phase 1: same as the prior phase-0 (drive heading equals segment heading).
    Phase 2: the achieved yaw in the first phase-3 frame (the push aim).
    Phase 3: the same push aim.
    """
    # Pre-compute push-aim from first phase-3 frame yaw
    # samples is (phase, x, y, yaw)
    push_aim = None
    for s in samples:
        if s[0] == 3:
            push_aim = s[3]
            break

    # Walk runs and assign target per frame
    runs = split_into_runs([(s[0], s[3]) for s in samples])
    # Map each phase-0 / phase-1 run to a planner segment in order. The first
    # phase-0 run is just the initial sample + immediate wait; assign seg 0.
    seg_iter = iter(segs)
    cur_seg_heading = None
    target_per_frame = [None] * len(samples)
    for ph, s, e in runs:
        if ph == 0 or ph == 1:
            try:
                _, _, _, h = next(seg_iter) if ph == 0 else (None, None, None, cur_seg_heading)
                if ph == 0:
                    cur_seg_heading = h
            except StopIteration:
                h = cur_seg_heading
            for i in range(s, e):
                target_per_frame[i] = (math.degrees(h) if h is not None else None,
                                       "segment heading")
        elif ph == 2:
            for i in range(s, e):
                target_per_frame[i] = (math.degrees(push_aim) if push_aim is not None else None,
                                       "push aim")
        elif ph == 3:
            for i in range(s, e):
                target_per_frame[i] = (math.degrees(push_aim) if push_aim is not None else None,
                                       "push aim")
    return target_per_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--qpos", required=True)
    ap.add_argument("--navlog", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--decimate", type=int, default=2,
                    help="Render every Nth frame to keep video short")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--rotation-run", type=int, default=None,
                    help="If set, render only the Nth rotation run "
                         "(0-indexed, counting only phase-0/phase-2 runs of >5 frames). "
                         "Pre/post buffer adds 5 frames either side.")
    ap.add_argument("--phases", type=str, default=None,
                    help="Comma-separated list of phase IDs to keep "
                         "(e.g. '0,2' for rotation only). Default: all.")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    qpos_path = Path(args.qpos)
    navlog_path = Path(args.navlog) if args.navlog else None

    car_adr = find_car_qpos_adr(xml_path)
    raw = parse_qpos(qpos_path)
    print(f"Loaded {len(raw)} frames; car_freejoint adr={car_adr}")

    samples = []  # (phase, x, y, yaw)
    for phase, q in raw:
        x = q[car_adr + 0]
        y = q[car_adr + 1]
        qw = q[car_adr + 3]
        qx = q[car_adr + 4]
        qy = q[car_adr + 5]
        qz = q[car_adr + 6]
        samples.append((phase, x, y, quat_to_yaw(qw, qx, qy, qz)))

    walls = parse_walls(xml_path)
    obstacles = parse_obstacles(xml_path)
    segments = parse_navlog(navlog_path) if navlog_path else []
    print(f"Walls: {len(walls)}, obstacles: {len(obstacles)}, "
          f"planner segments: {len(segments)}")

    # Bounds
    xs = [s[1] for s in samples] + [w[0] - w[2] for w in walls] + [w[0] + w[2] for w in walls]
    ys = [s[2] for s in samples] + [w[1] - w[3] for w in walls] + [w[1] + w[3] for w in walls]
    xmin, xmax = min(xs) - 0.05, max(xs) + 0.05
    ymin, ymax = min(ys) - 0.05, max(ys) + 0.05

    # Find rotation runs and rebound annotations
    just_phase = [(s[0], s[3]) for s in samples]
    runs = split_into_runs(just_phase)
    # For each rotation run (phase 0 / 2), compute exit yaw and post-wait yaw
    rotation_info = {}  # frame_idx -> dict
    for ph, s, e in runs:
        if ph not in (0, 2):
            continue
        if e - s < 5:
            continue
        yaws = [samples[i][3] for i in range(s, e)]
        split = find_active_wait_split(yaws)
        if split == 0 or split >= len(yaws):
            continue
        exit_idx = s + split - 1
        final_idx = e - 1
        rotation_info[exit_idx] = dict(kind="exit", phase=ph)
        rotation_info[final_idx] = dict(kind="final", phase=ph,
                                        rebound_deg=math.degrees(
                                            yaws[split - 1] - yaws[-1]))

    # Per-frame target heading
    targets = build_target_per_frame(samples, segments)

    # Decimate frames for output
    indices = list(range(0, len(samples), args.decimate))

    fig_w_in = args.width / 100
    fig_h_in = args.height / 100
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=100)

    tmpdir = tempfile.mkdtemp(prefix="rebound_render_")
    print(f"Writing PNGs to {tmpdir}")

    # Pre-build trajectory of (x, y, phase) so we can draw cumulative path
    traj_x = [s[1] for s in samples]
    traj_y = [s[2] for s in samples]
    traj_p = [s[0] for s in samples]

    # Track persistent annotation list (rebound markers stick after they happen)
    annotations = []  # list of (idx_first_visible, x, y, text)

    for frame_i, idx in enumerate(indices):
        ax.clear()
        # Walls
        for wx, wy, hx, hy in walls:
            ax.add_patch(patches.Rectangle(
                (wx - hx, wy - hy), 2 * hx, 2 * hy, color="dimgray", alpha=0.85))
        # Obstacles (initial poses)
        for name, ox, oy, oth, sx, sy in obstacles:
            r = patches.Rectangle((-sx, -sy), 2 * sx, 2 * sy,
                                  color="goldenrod", alpha=0.5,
                                  ec="black", lw=0.4)
            r.set_transform(Affine2D().rotate(oth).translate(ox, oy) + ax.transData)
            ax.add_patch(r)
        # Planner segment endpoints (green dots)
        for sidx, ex, ey, h in segments:
            ax.plot(ex, ey, "o", color="lime", markersize=7,
                    markeredgecolor="darkgreen", zorder=4)
            ax.text(ex + 0.005, ey + 0.005, f"S{sidx}", color="darkgreen",
                    fontsize=7, zorder=5)

        # Trajectory up to current frame, colored by phase per segment
        if idx > 0:
            cur_phase = traj_p[0]
            seg_start = 0
            for j in range(1, idx + 1):
                if traj_p[j] != cur_phase or j == idx:
                    pj = j + 1 if j == idx else j
                    ax.plot(traj_x[seg_start:pj], traj_y[seg_start:pj],
                            color=PHASE_COLOR.get(cur_phase, "black"),
                            linewidth=2.0, alpha=0.9)
                    cur_phase = traj_p[j]
                    seg_start = j

        # Stick rebound markers up to this frame
        for marker_idx, info in rotation_info.items():
            if marker_idx > idx:
                continue
            mx, my = samples[marker_idx][1], samples[marker_idx][2]
            myaw = samples[marker_idx][3]
            if info["kind"] == "exit":
                ax.plot(mx, my, "x", color="red", markersize=8, mew=2, zorder=6)
            elif info["kind"] == "final":
                ax.plot(mx, my, "o", color="purple", markersize=6,
                        markerfacecolor="white", markeredgewidth=1.5, zorder=6)
                ax.text(mx + 0.008, my - 0.008,
                        f"rebound {info['rebound_deg']:+.1f}°",
                        color="purple", fontsize=7, zorder=6)

        # Current car pose
        ph, cx, cy, cyaw = samples[idx]
        L, W = 0.07, 0.076
        car = patches.Rectangle((-L / 2, -W / 2), L, W,
                                color=PHASE_COLOR.get(ph, "navy"),
                                alpha=0.95, ec="black", lw=0.8)
        car.set_transform(Affine2D().rotate(cyaw).translate(cx, cy) + ax.transData)
        ax.add_patch(car)
        # Heading arrow
        ax.arrow(cx, cy, 0.04 * math.cos(cyaw), 0.04 * math.sin(cyaw),
                 head_width=0.012, color="red", zorder=7)

        # Target heading arrow (dashed, gray) if rotation phase
        if ph in (0, 2) and targets[idx] is not None and targets[idx][0] is not None:
            tdeg = targets[idx][0]
            trad = math.radians(tdeg)
            ax.arrow(cx, cy, 0.05 * math.cos(trad), 0.05 * math.sin(trad),
                     head_width=0.010, color="black", linestyle="--",
                     alpha=0.6, zorder=6)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_title(f"NAMO push trace — phase {ph} ({PHASE_LABEL.get(ph, '')})",
                     fontsize=10)

        # HUD
        cur_yaw_deg = math.degrees(cyaw)
        target_str = "—"
        if targets[idx] is not None and targets[idx][0] is not None:
            target_str = f"{targets[idx][0]:+.2f}° ({targets[idx][1]})"
        hud = (f"frame {idx}/{len(samples) - 1}   t={idx*0.01:.2f}s\n"
               f"yaw     = {cur_yaw_deg:+.2f}°\n"
               f"target  = {target_str}\n"
               f"pose    = ({cx:+.3f}, {cy:+.3f})")
        ax.text(0.02, 0.98, hud, transform=ax.transAxes,
                fontsize=9, va="top", ha="left", family="monospace",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

        # Phase legend
        leg_y = 0.02
        for p in [0, 1, 2, 3]:
            ax.text(0.02, leg_y, f"{p}: {PHASE_LABEL[p]}",
                    transform=ax.transAxes, fontsize=8,
                    color=PHASE_COLOR[p], family="monospace")
            leg_y += 0.025

        png = Path(tmpdir) / f"f{frame_i:06d}.png"
        fig.savefig(str(png), dpi=100, bbox_inches="tight")
        if frame_i % 50 == 0:
            print(f"  rendered {frame_i+1}/{len(indices)}")

    plt.close(fig)

    # Encode with ffmpeg
    out = Path(args.output)
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", str(Path(tmpdir) / "f%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out)
    ]
    print("Encoding:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
