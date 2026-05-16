"""Render primitive generation runs at corner edges for visualization.

Sets obstacle friction to 0 (already the default in primitive_gen scenes) and
runs push at 0.2 m/s (push_speed = 13.333 rad/s). Dumps qpos and renders mp4.

Edge layout (from generate_edge_points):
  Face 0 (+x): edges 0..14   — corners at 0, 14
  Face 1 (+y): edges 15..29  — corners at 15, 29
  Face 2 (-x): edges 30..44  — corners at 30, 44
  Face 3 (-y): edges 45..59  — corners at 45, 59
"""
import argparse
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

REPO = Path("/common/home/dm1487/robotics_research/ktamp/namo")
SCENES_DIR = REPO / "test_xml/little-car-modeling-package/assets/mjcf"

# Reuse the edge generator + car params from the existing script
sys.path.insert(0, str(REPO / "test_xml/little-car-modeling-package/scripts"))
from generate_car_primitives import generate_edge_points, yaw_to_quat  # type: ignore
from car_model.parameters import default_parameters  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["square", "wide", "tall"], default="square")
    ap.add_argument("--edges", type=int, nargs="+", default=[0, 14, 15, 29],
                    help="Edge indices (corners by default)")
    ap.add_argument("--depths", type=int, nargs="+", default=[5, 10],
                    help="push_steps (1-10)")
    ap.add_argument("--push-speed", type=float, default=13.333,
                    help="Wheel velocity rad/s (default 13.333 ≈ 0.2 m/s with 0.015m wheel)")
    ap.add_argument("--frame-skip", type=int, default=10)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--cam-dist", type=float, default=2.0)
    ap.add_argument("--out-dir", default=str(REPO / "videos/primitive_gen_corners"))
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_path = SCENES_DIR / f"primitive_gen_{args.shape}.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    push_step_duration = 0.5
    steps_per_push = int(push_step_duration / dt)

    # IDs
    car_qpos = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")]
    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_1_movable")
    obj_qpos = model.jnt_qposadr[model.body_jntadr[obj_body_id]]

    # Object half-sizes (from XML: obstacle_1_movable's geom size)
    obj_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_1_movable")
    half_sx = float(model.geom_size[obj_geom_id, 0])
    half_sy = float(model.geom_size[obj_geom_id, 1])
    half_sz = float(model.geom_size[obj_geom_id, 2])

    car_params = default_parameters()
    edges = generate_edge_points(
        half_sx, half_sy, points_per_face=15,
        robot_half_length=car_params.body_half_length_m,
        clearance=0.005,
    )

    # Renderer
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, half_sz]
    cam.distance = args.cam_dist
    cam.azimuth = 90.0
    cam.elevation = -90.0   # straight-down top-down

    for edge_idx in args.edges:
        if edge_idx < 0 or edge_idx >= len(edges):
            print(f"skip edge {edge_idx} (out of range)")
            continue
        edge_x, edge_y, heading = edges[edge_idx]

        for push_steps in args.depths:
            mujoco.mj_resetData(model, data)

            # Place object at origin
            data.qpos[obj_qpos:obj_qpos+3] = [0, 0, half_sz]
            data.qpos[obj_qpos+3:obj_qpos+7] = [1, 0, 0, 0]

            # Place car
            car_z = car_params.wheel_radius_m + car_params.scene_spawn_height_m
            data.qpos[car_qpos:car_qpos+3] = [edge_x, edge_y, car_z]
            data.qpos[car_qpos+3:car_qpos+7] = yaw_to_quat(heading)

            mujoco.mj_forward(model, data)

            # Settle
            data.ctrl[left_act] = 0
            data.ctrl[right_act] = 0
            for _ in range(500):
                mujoco.mj_step(model, data)

            # Push for push_steps × steps_per_push
            data.ctrl[left_act] = args.push_speed
            data.ctrl[right_act] = args.push_speed

            frames = []
            total_steps = push_steps * steps_per_push
            for t in range(total_steps):
                mujoco.mj_step(model, data)
                if t % args.frame_skip == 0:
                    renderer.update_scene(data, camera=cam)
                    frames.append(renderer.render())

            # Stop + a few coast frames
            data.ctrl[left_act] = 0
            data.ctrl[right_act] = 0
            for _ in range(50):
                mujoco.mj_step(model, data)
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())

            mp4 = out_dir / f"{args.shape}_edge{edge_idx:02d}_depth{push_steps:02d}_speed{args.push_speed:.1f}.mp4"
            writer = imageio.get_writer(str(mp4), fps=20, codec="libx264", quality=8, format="FFMPEG")
            for f in frames:
                writer.append_data(f)
            writer.close()

            obj_end_x = float(data.qpos[obj_qpos])
            obj_end_y = float(data.qpos[obj_qpos+1])
            obj_quat = data.qpos[obj_qpos+3:obj_qpos+7]
            w, x_, y_, z_ = obj_quat
            yaw = math.atan2(2.0*(w*z_ + x_*y_), 1.0 - 2.0*(y_*y_ + z_*z_))
            print(f"edge={edge_idx:2d} depth={push_steps:2d}: "
                  f"final obj=({obj_end_x:+.4f},{obj_end_y:+.4f}, yaw={math.degrees(yaw):+.1f}°)  "
                  f"|d|={math.hypot(obj_end_x, obj_end_y):.4f}m  → {mp4.name}")

    renderer.close()
    print(f"\nVideos in: {out_dir}")


if __name__ == "__main__":
    main()
