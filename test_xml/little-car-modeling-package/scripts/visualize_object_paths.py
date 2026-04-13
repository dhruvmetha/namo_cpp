"""Visualize object trajectories during car pushes.

Traces the object's (x, y, theta) path for each edge/depth combination
and plots them as trajectories from the origin.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAMO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
import mujoco

from car_model.generate_model import generate_all
from car_model.parameters import default_parameters
from generate_car_primitives import (
    OBJECT_CONFIGS,
    generate_edge_points,
    generate_scene_xml,
    quat_to_yaw,
    yaw_to_quat,
)


def trace_object_path(model, data, car_params, obj_config, edges, edge_idx, push_steps,
                      push_speed=10.0, push_step_duration=0.5, settle_steps=500,
                      sample_every=25):
    """Run a push and return the object's trajectory as [(x, y, theta), ...]."""
    dt = model.opt.timestep
    steps_per_push = int(push_step_duration / dt)

    car_fj = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")]
    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_1_movable")
    obj_jnt = model.jnt_qposadr[model.body_jntadr[obj_body]]

    ex, ey, heading = edges[edge_idx]

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[obj_jnt:obj_jnt + 3] = [0, 0, obj_config.half_size_z]
    data.qpos[obj_jnt + 3:obj_jnt + 7] = [1, 0, 0, 0]
    car_z = car_params.wheel_radius_m + car_params.scene_spawn_height_m
    data.qpos[car_fj:car_fj + 3] = [ex, ey, car_z]
    data.qpos[car_fj + 3:car_fj + 7] = yaw_to_quat(heading)
    mujoco.mj_forward(model, data)

    # Settle
    data.ctrl[left_act] = 0
    data.ctrl[right_act] = 0
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    # Record trajectory during push
    trajectory = []
    step = 0

    def sample():
        nonlocal step
        if step % sample_every == 0:
            pos = data.qpos[obj_jnt:obj_jnt + 3].copy()
            quat = data.qpos[obj_jnt + 3:obj_jnt + 7].copy()
            trajectory.append((float(pos[0]), float(pos[1]), quat_to_yaw(quat)))
        step += 1

    sample()  # initial position

    # Push
    data.ctrl[left_act] = push_speed
    data.ctrl[right_act] = push_speed
    for _ in range(push_steps * steps_per_push):
        mujoco.mj_step(model, data)
        sample()

    # Brief settle to see where it ends
    data.ctrl[left_act] = 0
    data.ctrl[right_act] = 0
    for _ in range(200):
        mujoco.mj_step(model, data)
        sample()

    return trajectory


def draw_rotated_rect(ax, x, y, theta, half_sx, half_sy, **kwargs):
    """Draw a rotated rectangle."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    corners = [
        (-half_sx, -half_sy), (half_sx, -half_sy),
        (half_sx, half_sy), (-half_sx, half_sy), (-half_sx, -half_sy)
    ]
    xs = [x + cx * cos_t - cy * sin_t for cx, cy in corners]
    ys = [y + cx * sin_t + cy * cos_t for cx, cy in corners]
    ax.plot(xs, ys, **kwargs)


def main():
    car_params = default_parameters()
    generate_all(PROJECT_ROOT / "assets", params=car_params)

    output_dir = PROJECT_ROOT / "artifacts" / "object_paths"
    output_dir.mkdir(parents=True, exist_ok=True)

    for obj in OBJECT_CONFIGS:
        scene_xml = generate_scene_xml(obj, car_params)
        scene_path = PROJECT_ROOT / "assets" / "mjcf" / f"paths_{obj.name}.xml"
        scene_path.write_text(scene_xml, encoding="utf-8")

        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)

        edges = generate_edge_points(
            obj.half_size_x, obj.half_size_y,
            points_per_face=15,
            robot_half_length=car_params.body_half_length_m,
            clearance=0.005,
        )

        # ── Plot 1: All edges at depth 5, colored by face ────────────
        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
        fig.suptitle(f"Object Paths: {obj.name} ({obj.half_size_x*200:.1f}x{obj.half_size_y*200:.1f}cm)", fontsize=14)

        ax = axes[0]
        ax.set_title("All edges, depth=5 (colored by face)")
        face_colors = ['red', 'blue', 'green', 'orange']
        face_names = ['+x face', '+y face', '-x face', '-y face']

        for edge_idx in range(60):
            face = edge_idx // 15
            traj = trace_object_path(model, data, car_params, obj, edges, edge_idx, push_steps=5)
            xs = [t[0] * 1000 for t in traj]
            ys = [t[1] * 1000 for t in traj]
            ax.plot(xs, ys, color=face_colors[face], alpha=0.3, linewidth=0.8)
            # End point
            ax.plot(xs[-1], ys[-1], 'o', color=face_colors[face], markersize=2, alpha=0.5)

        # Draw initial object outline
        draw_rotated_rect(ax, 0, 0, 0, obj.half_size_x * 1000, obj.half_size_y * 1000,
                         color='black', linewidth=2)

        for c, n in zip(face_colors, face_names):
            ax.plot([], [], color=c, linewidth=2, label=n)
        ax.legend(fontsize=8)
        ax.set_xlim(-350, 350)
        ax.set_ylim(-350, 350)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

        # ── Plot 2: Center edge, depths 1-10, showing trajectory shape ────
        ax = axes[1]
        ax.set_title("Center edge per face, depths 1-10")
        center_edges = [7, 22, 37, 52]  # center of each face

        cmap = plt.cm.viridis
        for face_idx, center_edge in enumerate(center_edges):
            for depth in range(1, 11):
                traj = trace_object_path(model, data, car_params, obj, edges, center_edge, push_steps=depth)
                xs = [t[0] * 1000 for t in traj]
                ys = [t[1] * 1000 for t in traj]
                color = cmap(depth / 10)
                ax.plot(xs, ys, color=color, alpha=0.5, linewidth=0.8)
                # Draw final object outline
                final = traj[-1]
                draw_rotated_rect(ax, final[0]*1000, final[1]*1000, final[2],
                                 obj.half_size_x*1000, obj.half_size_y*1000,
                                 color=face_colors[face_idx], linewidth=0.5, alpha=0.3)

        draw_rotated_rect(ax, 0, 0, 0, obj.half_size_x * 1000, obj.half_size_y * 1000,
                         color='black', linewidth=2)
        ax.set_xlim(-550, 550)
        ax.set_ylim(-550, 550)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

        # ── Plot 3: Extreme vs center comparison with object outlines ────
        ax = axes[2]
        ax.set_title("Face 2 (-x): extreme vs center, depth=5")

        for edge_idx, label, color, ls in [
            (30, 'left extreme', 'red', '-'),
            (37, 'center', 'black', '-'),
            (44, 'right extreme', 'blue', '-'),
        ]:
            traj = trace_object_path(model, data, car_params, obj, edges, edge_idx, push_steps=5)
            xs = [t[0] * 1000 for t in traj]
            ys = [t[1] * 1000 for t in traj]
            ax.plot(xs, ys, color=color, linewidth=1.5, label=label, linestyle=ls)
            # Draw final object outline
            final = traj[-1]
            draw_rotated_rect(ax, final[0]*1000, final[1]*1000, final[2],
                             obj.half_size_x*1000, obj.half_size_y*1000,
                             color=color, linewidth=1.5, alpha=0.5, linestyle='--')

        draw_rotated_rect(ax, 0, 0, 0, obj.half_size_x * 1000, obj.half_size_y * 1000,
                         color='black', linewidth=2)
        ax.legend(fontsize=9)
        ax.set_xlim(-100, 350)
        ax.set_ylim(-200, 200)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

        plt.tight_layout()
        path = output_dir / f"object_paths_{obj.name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
