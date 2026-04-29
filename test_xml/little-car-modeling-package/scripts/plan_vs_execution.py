#!/usr/bin/env python3
"""Side-by-side visualization: planned wavefront path vs. executed trajectory.

For each selected edge:
  Left panel  — environment with wavefront path (green), robot start pose,
                and target edge point.
  Right panel — same environment with the robot's actual executed trajectory,
                phase-colored (orange=rotate1, blue=pure pursuit, red=rotate2),
                plus final robot pose.
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


def draw_walls_and_objects(ax, xml_path, movable_alpha=0.5):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    walls = root.find(".//body[@name='walls']")
    if walls is not None:
        for g in walls.findall("geom"):
            pos = [float(v) for v in g.get("pos", "0 0 0").split()]
            size = [float(v) for v in g.get("size", "0 0 0").split()]
            ax.add_patch(patches.Rectangle(
                (pos[0] - size[0], pos[1] - size[1]),
                2*size[0], 2*size[1], color="gray", alpha=0.9))

    for body in root.find("worldbody").findall("body"):
        name = body.get("name", "")
        if not name.endswith("_movable"):
            continue
        g = body.find("geom")
        pos = [float(v) for v in g.get("pos", "0 0 0").split()]
        size = [float(v) for v in g.get("size", "0 0 0").split()]
        euler = float(g.get("euler", "0 0 0").split()[2]) * math.pi/180
        rect = patches.Rectangle((-size[0], -size[1]), 2*size[0], 2*size[1],
                                 color="gold", alpha=movable_alpha, ec="black", lw=0.3)
        t = Affine2D().rotate(euler).translate(pos[0], pos[1]) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)


def draw_car(ax, x, y, theta, color="navy", alpha=1.0):
    L, W = 0.07, 0.076
    r = patches.Rectangle((-L/2, -W/2), L, W,
                          color=color, alpha=alpha, ec="black", lw=0.8)
    r.set_transform(Affine2D().rotate(theta).translate(x, y) + ax.transData)
    ax.add_patch(r)
    ax.arrow(x, y, (L/2)*math.cos(theta)*0.85, (L/2)*math.sin(theta)*0.85,
             head_width=0.012, color="red", zorder=6, alpha=alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xml")
    parser.add_argument("config")
    parser.add_argument("output_png")
    parser.add_argument("--object", default="obstacle_1_movable")
    parser.add_argument("--edge", type=int, default=45,
                        help="Single edge index to visualize (pick a successful one)")
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    print(f"Running {args.object} edge={args.edge}...")
    stdout, stderr = run_and_capture(args.xml, args.config,
                                      args.object, args.edge, args.depth)
    d = parse_output(stdout, stderr)

    moved = 0.0
    if d["init_obj"] and d["final_obj"]:
        moved = math.hypot(d["final_obj"][0] - d["init_obj"][0],
                           d["final_obj"][1] - d["init_obj"][1])

    fig, (ax_plan, ax_exec) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left: the plan ---
    draw_walls_and_objects(ax_plan, args.xml)
    if d["path"]:
        px = [p[0] for p in d["path"]]
        py = [p[1] for p in d["path"]]
        ax_plan.plot(px, py, color="lime", linewidth=2.0, alpha=0.85, label="wavefront path")
        # Start and end markers
        ax_plan.plot(px[0], py[0], "o", color="blue", markersize=9, label="start")
        ax_plan.plot(px[-1], py[-1], "^", color="red", markersize=10, label="target edge")
    if d["init_robot"]:
        draw_car(ax_plan, d["init_robot"][0], d["init_robot"][1], d["init_robot"][2],
                 color="blue", alpha=0.5)
    ax_plan.set_aspect("equal")
    ax_plan.set_title(f"PLAN (wavefront path, {len(d['path'])} waypoints)", fontsize=10)
    ax_plan.legend(loc="best", fontsize=8)
    ax_plan.autoscale_view()

    # --- Right: the execution ---
    draw_walls_and_objects(ax_exec, args.xml)
    if d["path"]:
        # faint plan overlay
        px = [p[0] for p in d["path"]]
        py = [p[1] for p in d["path"]]
        ax_exec.plot(px, py, color="lime", linewidth=1.0, alpha=0.3, label="plan (ref)")

    phase_colors = {0: "orange", 1: "dodgerblue", 2: "crimson"}
    phase_names = {0: "phase 1: rotate to path", 1: "phase 2: pure pursuit", 2: "phase 3: rotate to push"}
    if d["poses"]:
        by_phase = {}
        for x, y, t, ph in d["poses"]:
            by_phase.setdefault(ph, []).append((x, y))
        for ph in [0, 1, 2]:
            if ph in by_phase:
                pts = by_phase[ph]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax_exec.plot(xs, ys, color=phase_colors[ph],
                             linewidth=2.5, alpha=0.9, label=phase_names[ph])
                # small markers for pose samples
                ax_exec.plot(xs, ys, ".", color=phase_colors[ph], markersize=3, alpha=0.7)
    if d["init_robot"]:
        draw_car(ax_exec, d["init_robot"][0], d["init_robot"][1], d["init_robot"][2],
                 color="blue", alpha=0.3)
    if d["final_robot"]:
        draw_car(ax_exec, d["final_robot"][0], d["final_robot"][1], d["final_robot"][2],
                 color="navy", alpha=0.95)
    ax_exec.set_aspect("equal")
    ax_exec.set_title(f"EXECUTION (moved obj {moved*1000:.0f}mm, {len(d['poses'])} poses)", fontsize=10)
    ax_exec.legend(loc="best", fontsize=8)
    ax_exec.autoscale_view()

    fig.suptitle(f"{Path(args.xml).name}  target={args.object}  edge={args.edge}",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
