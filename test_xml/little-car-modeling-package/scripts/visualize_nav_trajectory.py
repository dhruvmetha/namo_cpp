#!/usr/bin/env python3
"""Visualize the car's navigation trajectory.

Runs a push action with NAMO_NAV_LOG=1 set, captures the path and pose
samples printed to stderr, and plots everything in matplotlib.
"""

import argparse
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np


RUNNER = r"""
import os, sys, namo_rl
env = namo_rl.RLEnvironment(sys.argv[1], sys.argv[2], False)
action = namo_rl.Action()
action.object_id = sys.argv[3]
action.edge_idx = int(sys.argv[4])
action.depth = int(sys.argv[5])

obs_before = env.get_observation()
print('INIT_ROBOT', obs_before['robot_pose'])
print('INIT_OBJ', obs_before[sys.argv[3] + '_pose'])
result = env.step(action)
obs_after = env.get_observation()
print('FINAL_ROBOT', obs_after['robot_pose'])
print('FINAL_OBJ', obs_after[sys.argv[3] + '_pose'])
print('RESULT', result.done, dict(result.info))
"""


def run_and_capture(xml, config, obj, edge, depth):
    env = os.environ.copy()
    env["NAMO_NAV_LOG"] = "1"
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = repo_root / "build_python"
    if not build_dir.is_dir() or not any(build_dir.glob("namo_rl*.so")):
        raise RuntimeError(
            "Canonical namo_rl build missing at build_python. Build with:\n"
            "  cmake -S . -B build_python -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON\n"
            "  cmake --build build_python --target namo_rl -j$(nproc)"
        )
    env["PYTHONPATH"] = f"{build_dir}:{env.get('PYTHONPATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    )
    p = subprocess.run(
        [sys.executable, "-c", RUNNER,
         xml, config, obj, str(edge), str(depth)],
        env=env, capture_output=True, text=True,
    )
    return p.stdout, p.stderr


def parse_output(stdout, stderr):
    path = []
    poses = []
    init_robot = None
    init_obj = None
    final_robot = None
    final_obj = None

    for line in stderr.splitlines():
        if line.startswith("[NAV_PATH]"):
            coords = line[len("[NAV_PATH]"):].strip().split()
            path = [tuple(map(float, c.split(","))) for c in coords]
        elif line.startswith("[NAV_POSE]"):
            parts = line.split()
            poses.append((float(parts[1]), float(parts[2]),
                          float(parts[3]), int(parts[4])))

    for line in stdout.splitlines():
        if line.startswith("INIT_ROBOT"):
            init_robot = eval(line.split(None, 1)[1])
        elif line.startswith("INIT_OBJ"):
            init_obj = eval(line.split(None, 1)[1])
        elif line.startswith("FINAL_ROBOT"):
            final_robot = eval(line.split(None, 1)[1])
        elif line.startswith("FINAL_OBJ"):
            final_obj = eval(line.split(None, 1)[1])

    return dict(path=path, poses=poses,
                init_robot=init_robot, init_obj=init_obj,
                final_robot=final_robot, final_obj=final_obj)


def draw(ax, xml_path, data, title):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    walls = root.find(".//body[@name='walls']")
    if walls is not None:
        for g in walls.findall("geom"):
            pos = [float(v) for v in g.get("pos", "0 0 0").split()]
            size = [float(v) for v in g.get("size", "0 0 0").split()]
            ax.add_patch(patches.Rectangle(
                (pos[0] - size[0], pos[1] - size[1]),
                2*size[0], 2*size[1], color="gray", alpha=0.8))

    # All movable objects: initial pose (faint) and draw final pose (solid)
    for body in root.find("worldbody").findall("body"):
        name = body.get("name", "")
        if not name.endswith("_movable"):
            continue
        g = body.find("geom")
        pos = [float(v) for v in g.get("pos", "0 0 0").split()]
        size = [float(v) for v in g.get("size", "0 0 0").split()]
        euler = float(g.get("euler", "0 0 0").split()[2]) * math.pi/180
        rect = patches.Rectangle((-size[0], -size[1]), 2*size[0], 2*size[1],
                                 color="gold", alpha=0.25, ec="black", lw=0.3)
        t = Affine2D().rotate(euler).translate(pos[0], pos[1]) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    # Wavefront path
    if data["path"]:
        px = [p[0] for p in data["path"]]
        py = [p[1] for p in data["path"]]
        ax.plot(px, py, color="lime", linewidth=1.2, alpha=0.6, label="wavefront path")

    # Nav trajectory colored by phase
    phase_colors = {0: "orange", 1: "dodgerblue", 2: "crimson"}
    phase_names = {0: "rotate to path", 1: "pure pursuit", 2: "rotate to push"}
    if data["poses"]:
        by_phase = {}
        for x, y, t, ph in data["poses"]:
            by_phase.setdefault(ph, []).append((x, y))
        for ph, pts in by_phase.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=phase_colors.get(ph, "black"),
                    linewidth=2.0, alpha=0.9, label=phase_names.get(ph, f"phase {ph}"))

    # Initial car (faint)
    if data["init_robot"]:
        rx, ry, rt = data["init_robot"]
        r = patches.Rectangle((-0.035, -0.038), 0.07, 0.076,
                              color="blue", alpha=0.25)
        r.set_transform(Affine2D().rotate(rt).translate(rx, ry) + ax.transData)
        ax.add_patch(r)
    # Final car (solid)
    if data["final_robot"]:
        rx, ry, rt = data["final_robot"]
        r = patches.Rectangle((-0.035, -0.038), 0.07, 0.076,
                              color="navy", alpha=0.9, ec="black", lw=0.8)
        r.set_transform(Affine2D().rotate(rt).translate(rx, ry) + ax.transData)
        ax.add_patch(r)
        ax.arrow(rx, ry,
                 0.03*math.cos(rt), 0.03*math.sin(rt),
                 head_width=0.012, color="red", zorder=5)

    # Final object (solid, at its moved pose)
    if data["final_obj"]:
        # reuse size from XML obstacle — simple for this use case
        pass

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xml")
    parser.add_argument("config")
    parser.add_argument("output_png")
    parser.add_argument("--object", default=None)
    parser.add_argument("--edges", default="0,15,30,45")
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    # Default target object
    obj = args.object if args.object else "obstacle_1_movable"

    edges = [int(e) for e in args.edges.split(",")]
    fig, axes = plt.subplots(1, len(edges), figsize=(5.0*len(edges), 5.0))
    if len(edges) == 1:
        axes = [axes]

    for ax, edge in zip(axes, edges):
        print(f"Running edge={edge}...")
        stdout, stderr = run_and_capture(args.xml, args.config, obj, edge, args.depth)
        data = parse_output(stdout, stderr)
        moved = 0.0
        if data["init_obj"] and data["final_obj"]:
            dx = data["final_obj"][0] - data["init_obj"][0]
            dy = data["final_obj"][1] - data["init_obj"][1]
            moved = math.hypot(dx, dy)
        draw(ax, args.xml, data, f"edge={edge}  moved={moved*1000:.0f}mm")

    plt.tight_layout()
    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
