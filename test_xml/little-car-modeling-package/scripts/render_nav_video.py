#!/usr/bin/env python3
"""Render a MuJoCo video + matplotlib trajectory from a qpos dump.

Inputs:
  - XML path (scene)
  - qpos dump file (one frame per line: "phase nq q0 q1 ...")
  - output MP4 path
  - optional: --path-file with wavefront waypoints ("x,y x,y ..." on one line)

Layout: left half = MuJoCo render (top-down), right half = matplotlib
trajectory with planned path overlay, advancing in sync with the video.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import mujoco
import numpy as np
from PIL import Image


def parse_qpos(path):
    frames = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            phase = int(parts[0])
            nq = int(parts[1])
            q = [float(v) for v in parts[2:2+nq]]
            frames.append((phase, q))
    return frames


def parse_walls(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    walls = []
    wb = root.find(".//body[@name='walls']")
    if wb is not None:
        for g in wb.findall("geom"):
            pos = [float(v) for v in g.get("pos", "0 0 0").split()]
            size = [float(v) for v in g.get("size", "0 0 0").split()]
            walls.append((pos[0], pos[1], size[0], size[1]))
    return walls


def render_mj_frame(renderer, model, data, q, camera):
    data.qpos[:len(q)] = q
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera)
    return renderer.render()


def load_path(path_file):
    if not path_file or not os.path.exists(path_file):
        return []
    with open(path_file) as f:
        line = f.read().strip()
    if line.startswith("[NAV_PATH]"):
        line = line[len("[NAV_PATH]"):].strip()
    waypoints = []
    for tok in line.split():
        try:
            x, y = tok.split(",")
            waypoints.append((float(x), float(y)))
        except ValueError:
            pass
    return waypoints


def render_mpl_frame(ax, xml_path, trajectory_so_far, phase_colors, bounds,
                     planned_path=None):
    ax.clear()
    # Walls
    for wx, wy, hx, hy in parse_walls(xml_path):
        ax.add_patch(patches.Rectangle(
            (wx - hx, wy - hy), 2*hx, 2*hy, color="gray", alpha=0.9))

    # All movable objects (initial pose from XML)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for body in root.find("worldbody").findall("body"):
        name = body.get("name", "")
        if not name.endswith("_movable"):
            continue
        g = body.find("geom")
        pos = [float(v) for v in g.get("pos", "0 0 0").split()]
        size = [float(v) for v in g.get("size", "0 0 0").split()]
        euler = float(g.get("euler", "0 0 0").split()[2]) * math.pi/180
        rect = patches.Rectangle((-size[0], -size[1]), 2*size[0], 2*size[1],
                                 color="gold", alpha=0.6, ec="black", lw=0.3)
        rect.set_transform(Affine2D().rotate(euler).translate(pos[0], pos[1]) + ax.transData)
        ax.add_patch(rect)

    # Planned wavefront path (green line)
    if planned_path:
        xs = [p[0] for p in planned_path]
        ys = [p[1] for p in planned_path]
        ax.plot(xs, ys, color="lime", linewidth=2.0, alpha=0.7, zorder=2,
                label="wavefront plan")

    # Trajectory so far, phase-colored
    by_phase = {}
    for ph, (x, y, t) in trajectory_so_far:
        by_phase.setdefault(ph, []).append((x, y))
    for ph in sorted(by_phase.keys()):
        pts = by_phase[ph]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=phase_colors.get(ph, "black"),
                linewidth=2.2, alpha=0.95)
        ax.plot(xs, ys, ".", color=phase_colors.get(ph, "black"), markersize=2)

    # Current car pose (last sample)
    if trajectory_so_far:
        ph, (rx, ry, rt) = trajectory_so_far[-1]
        L, W = 0.07, 0.076
        r = patches.Rectangle((-L/2, -W/2), L, W,
                              color=phase_colors.get(ph, "navy"), alpha=0.9,
                              ec="black", lw=0.8)
        r.set_transform(Affine2D().rotate(rt).translate(rx, ry) + ax.transData)
        ax.add_patch(r)
        ax.arrow(rx, ry, 0.025*math.cos(rt), 0.025*math.sin(rt),
                 head_width=0.008, color="red", zorder=6)

    ax.set_xlim(bounds[0]-0.05, bounds[1]+0.05)
    ax.set_ylim(bounds[2]-0.05, bounds[3]+0.05)
    ax.set_aspect("equal")
    ax.set_title("Navigation trajectory", fontsize=9)
    ax.grid(True, alpha=0.2)


def extract_robot_pose_from_qpos(q, freejoint_adr):
    x = q[freejoint_adr + 0]
    y = q[freejoint_adr + 1]
    # quat: w, x, y, z at adr+3 .. adr+6
    w, qx, qy, qz = q[freejoint_adr+3], q[freejoint_adr+4], q[freejoint_adr+5], q[freejoint_adr+6]
    theta = math.atan2(2*(w*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return x, y, theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("qpos_dump")
    ap.add_argument("output_mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="Render every Nth qpos frame (1 = every frame). 25 = ~20fps from a 500Hz qpos dump.")
    ap.add_argument("--cam-dist", type=float, default=1.2)
    ap.add_argument("--cam-elevation", type=float, default=-89.0)  # top-down
    ap.add_argument("--cam-azimuth", type=float, default=90.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--path-file", default=None, help="File with wavefront waypoints")
    args = ap.parse_args()

    frames = parse_qpos(args.qpos_dump)
    print(f"Loaded {len(frames)} qpos frames")
    if not frames:
        sys.exit(1)

    planned_path = load_path(args.path_file)
    print(f"Loaded {len(planned_path)} planned waypoints")

    # Load MuJoCo model with a visual global element
    with open(args.xml) as f:
        xml_str = f.read()
    # Inject visual global for offscreen framebuffer
    if "<visual>" not in xml_str:
        visual = f'  <visual><global offwidth="{args.width}" offheight="{args.height}"/></visual>\n'
        xml_str = xml_str.replace("<worldbody>", visual + "  <worldbody>")

    # Inject planned path as green sphere sites
    if planned_path:
        # Subsample so rendering stays clean (one marker every ~1cm)
        step = max(1, len(planned_path) // 60)
        path_sites = []
        for i, (x, y) in enumerate(planned_path[::step]):
            path_sites.append(
                f'<site name="path_pt_{i}" pos="{x:.4f} {y:.4f} 0.005" '
                f'size="0.006" rgba="0 1 0 0.7" type="sphere"/>'
            )
        path_xml = "\n    ".join(path_sites)
        xml_str = xml_str.replace("</worldbody>", f"    {path_xml}\n  </worldbody>")

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    # Freejoint qpos address
    fj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")
    if fj_id < 0:
        print("car_freejoint not found", file=sys.stderr); sys.exit(1)
    freejoint_adr = model.jnt_qposadr[fj_id]

    # Environment bounds
    bounds = [-0.6, 0.6, -0.6, 0.6]
    # try to read from XML walls
    walls = parse_walls(args.xml)
    if walls:
        xs = [wx-hx for wx, wy, hx, hy in walls] + [wx+hx for wx, wy, hx, hy in walls]
        ys = [wy-hy for wx, wy, hx, hy in walls] + [wy+hy for wx, wy, hx, hy in walls]
        bounds = [min(xs), max(xs), min(ys), max(ys)]

    # Renderer
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, 0.05]
    camera.distance = args.cam_dist
    camera.azimuth = args.cam_azimuth
    camera.elevation = args.cam_elevation

    phase_colors = {0: "orange", 1: "dodgerblue", 2: "crimson"}

    # Matplotlib figure — half-width + half-height for side-by-side
    fig = plt.figure(figsize=(args.width*2/100, args.height/100), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    trajectory_samples = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, (phase, q) in enumerate(frames):
            # Render MuJoCo frame
            data.qpos[:len(q)] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            mj_img = renderer.render()

            # Add current pose to trajectory
            rx, ry, rt = extract_robot_pose_from_qpos(q, freejoint_adr)
            trajectory_samples.append((phase, (rx, ry, rt)))

            # Render matplotlib panel
            render_mpl_frame(ax, args.xml, trajectory_samples, phase_colors, bounds,
                             planned_path=planned_path)
            fig.canvas.draw()
            mpl_img = np.asarray(fig.canvas.buffer_rgba())[..., :3]

            # Stitch side-by-side
            h_mj, w_mj, _ = mj_img.shape
            h_mp, w_mp, _ = mpl_img.shape
            # match heights
            if h_mp != h_mj:
                mpl_resized = np.asarray(Image.fromarray(mpl_img).resize(
                    (int(w_mp * h_mj / h_mp), h_mj)))
                mpl_img = mpl_resized
            combined = np.concatenate([mj_img, mpl_img], axis=1)
            Image.fromarray(combined).save(tmp / f"f{i:06d}.png")

            if i % 20 == 0:
                print(f"  rendered {i}/{len(frames)}")

        # Encode with ffmpeg
        out = Path(args.output_mp4)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp / "f%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)
        ], check=True, capture_output=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
