#!/usr/bin/env python3
"""Matplotlib visualization of a navigation + push action.

For each attempt, render:
  - Environment (walls, objects)
  - Planned wavefront path
  - Car before nav (filled) and after nav (outlined)
  - Object before and after
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np

import namo_rl


def parse_walls(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    walls = []
    wall_body = root.find(".//body[@name='walls']")
    if wall_body is not None:
        for g in wall_body.findall("geom"):
            pos = [float(v) for v in g.get("pos", "0 0 0").split()]
            size = [float(v) for v in g.get("size", "0 0 0").split()]
            walls.append((pos[0], pos[1], size[0], size[1]))
    return walls


def draw_scene(ax, xml_path, env, obj_poses_by_name, robot_pose, path=None,
               title="", robot_alpha=1.0, robot_color="blue", label=None):
    walls = parse_walls(xml_path)
    for wx, wy, hx, hy in walls:
        ax.add_patch(patches.Rectangle(
            (wx - hx, wy - hy), 2*hx, 2*hy, color="gray", alpha=0.9
        ))

    info = env.get_object_info()
    for name, pose in obj_poses_by_name.items():
        if name not in info:
            continue
        meta = info[name]
        sx, sy = meta["size_x"], meta["size_y"]
        rect = patches.Rectangle((-sx, -sy), 2*sx, 2*sy,
                                 color="gold", alpha=0.9, ec="black", lw=0.5)
        t = Affine2D().rotate(pose[2]).translate(pose[0], pose[1]) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    if path is not None and len(path) > 1:
        pxs = [p[0] for p in path]
        pys = [p[1] for p in path]
        ax.plot(pxs, pys, color="lime", linewidth=1.5, alpha=0.7, zorder=3, label="wavefront path")

    rx, ry, rtheta = robot_pose
    car_len, car_wid = 0.07, 0.076
    car_rect = patches.Rectangle((-car_len/2, -car_wid/2), car_len, car_wid,
                                 color=robot_color, alpha=robot_alpha, zorder=4,
                                 ec="black", lw=0.8,
                                 label=label)
    t = Affine2D().rotate(rtheta).translate(rx, ry) + ax.transData
    car_rect.set_transform(t)
    ax.add_patch(car_rect)

    front_dx = (car_len/2) * math.cos(rtheta)
    front_dy = (car_len/2) * math.sin(rtheta)
    ax.arrow(rx, ry, front_dx*0.8, front_dy*0.8,
             head_width=0.01, color="red", zorder=5, alpha=robot_alpha)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xml")
    parser.add_argument("config")
    parser.add_argument("output_png")
    parser.add_argument("--object", default=None)
    parser.add_argument("--edges", type=str, default="0,15,30,45",
                        help="Comma-separated edge indices to attempt")
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    edges = [int(e) for e in args.edges.split(",")]

    env = namo_rl.RLEnvironment(args.xml, args.config, False)
    baseline = env.get_full_state()
    obs_init = env.get_observation()
    obj = args.object
    if obj is None:
        reachable = env.get_reachable_objects()
        if not reachable:
            print("No reachable objects", file=sys.stderr)
            sys.exit(1)
        obj = reachable[0]

    init_robot = obs_init["robot_pose"]
    init_obj = obs_init[f"{obj}_pose"]

    # Extract wavefront path once
    import ctypes  # Unused, but ok

    fig, axes = plt.subplots(1, len(edges), figsize=(5*len(edges), 5))
    if len(edges) == 1:
        axes = [axes]

    for ax, edge in zip(axes, edges):
        env.set_full_state(baseline)

        action = namo_rl.Action()
        action.object_id = obj
        action.edge_idx = edge
        action.depth = args.depth

        ob_before = env.get_observation()[f"{obj}_pose"]
        result = env.step(action)
        ob_after_obs = env.get_observation()
        final_robot = ob_after_obs["robot_pose"]
        final_obj = ob_after_obs[f"{obj}_pose"]
        moved = math.hypot(final_obj[0]-ob_before[0], final_obj[1]-ob_before[1])

        # Initial state
        draw_scene(ax, args.xml, env,
                   {obj: init_obj}, init_robot,
                   path=None,
                   title=f"edge={edge}  moved={moved*1000:.0f}mm",
                   robot_alpha=0.25, robot_color="blue", label="start")

        # Final state
        draw_scene(ax, args.xml, env,
                   {obj: final_obj}, final_robot,
                   path=None,
                   title=f"edge={edge}  moved={moved*1000:.0f}mm",
                   robot_alpha=1.0, robot_color="navy", label="end")

        ax.autoscale_view()

    plt.tight_layout()
    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
