"""Render videos of the car pushing objects from each face."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAMO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio
import mujoco
import numpy as np

from car_model.generate_model import generate_all
from car_model.parameters import default_parameters
from generate_car_primitives import (
    OBJECT_CONFIGS,
    SCALE,
    generate_edge_points,
    generate_scene_xml,
    quat_to_yaw,
    yaw_to_quat,
)


def render_push(
    obj_config,
    edge_idx: int,
    push_steps: int,
    output_path: Path,
    push_speed: float = 10.0,
    push_step_duration: float = 0.5,
    settle_steps: int = 500,
    fps: int = 30,
    frame_every: int = 5,
    exit_ramp_ticks: int = 0,
    entry_ramp_ticks: int = 0,
):
    car_params = default_parameters()
    generate_all(PROJECT_ROOT / "assets", params=car_params)

    scene_xml = generate_scene_xml(obj_config, car_params)
    scene_path = PROJECT_ROOT / "assets" / "mjcf" / f"push_video_{obj_config.name}.xml"
    scene_path.write_text(scene_xml, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    steps_per_push = int(push_step_duration / dt)

    renderer = mujoco.Renderer(model, height=480, width=640)

    # IDs
    car_fj_qpos = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")]
    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_1_movable")
    obj_jnt_qpos = model.jnt_qposadr[model.body_jntadr[obj_body_id]]

    # Edge points
    edges = generate_edge_points(
        obj_config.half_size_x, obj_config.half_size_y,
        points_per_face=15,
        robot_half_length=car_params.body_half_length_m,
        clearance=0.005,
    )
    edge_x, edge_y, heading = edges[edge_idx]
    face = edge_idx // 15
    face_names = ["+x", "+y", "-x", "-y"]

    # Camera setup - top-down view centered on the action
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.03]
    cam.distance = 0.5
    cam.azimuth = 90
    cam.elevation = -90  # top-down

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[obj_jnt_qpos : obj_jnt_qpos + 3] = [0, 0, obj_config.half_size_z]
    data.qpos[obj_jnt_qpos + 3 : obj_jnt_qpos + 7] = [1, 0, 0, 0]
    car_z = car_params.wheel_radius_m + car_params.scene_spawn_height_m
    data.qpos[car_fj_qpos : car_fj_qpos + 3] = [edge_x, edge_y, car_z]
    data.qpos[car_fj_qpos + 3 : car_fj_qpos + 7] = yaw_to_quat(heading)
    mujoco.mj_forward(model, data)

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8, format="FFMPEG")
    step = 0

    def capture():
        nonlocal step
        if step % frame_every == 0:
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
        step += 1

    # Phase 1: Settle (show initial state)
    data.ctrl[left_act] = 0
    data.ctrl[right_act] = 0
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        capture()

    # Record object before
    obj_before = data.qpos[obj_jnt_qpos : obj_jnt_qpos + 3].copy()

    # Phase 1b (optional): Entry ramp — linearly increase ctrl from 0 -> push_speed
    if entry_ramp_ticks > 0:
        N = entry_ramp_ticks
        for k in range(1, N + 1):
            cmd = push_speed * (k / N)
            data.ctrl[left_act] = cmd
            data.ctrl[right_act] = cmd
            mujoco.mj_step(model, data)
            capture()

    # Phase 2: Push
    data.ctrl[left_act] = push_speed
    data.ctrl[right_act] = push_speed
    for _ in range(push_steps * steps_per_push):
        mujoco.mj_step(model, data)
        capture()

    # Phase 2b (optional): Exit ramp — linearly decrease ctrl from push_speed to 0
    if exit_ramp_ticks > 0:
        N = exit_ramp_ticks
        for k in range(1, N + 1):
            cmd = push_speed * (1.0 - k / N)
            data.ctrl[left_act] = cmd
            data.ctrl[right_act] = cmd
            mujoco.mj_step(model, data)
            capture()

    # Phase 3: Stop and settle
    data.ctrl[left_act] = 0
    data.ctrl[right_act] = 0
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        capture()

    obj_after = data.qpos[obj_jnt_qpos : obj_jnt_qpos + 3].copy()
    displacement = math.sqrt((obj_after[0] - obj_before[0])**2 + (obj_after[1] - obj_before[1])**2)

    writer.close()
    renderer.close()

    print(f"  {face_names[face]} face, edge {edge_idx}, depth {push_steps}: "
          f"displacement={displacement*1000:.1f}mm → {output_path.name}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-speed", type=float, default=10.0, help="Wheel velocity during push (rad/s)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "push_videos",
                        help="Directory to write videos to")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix appended to each video filename (e.g. '_20rad') so multiple speeds can coexist")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render one push per face at medium depth, plus short/max for one face,
    # plus glancing corner pushes (edge idx 0/15/30/45 = corner-end of each face)
    # and the opposite corner (idx 14/29/44/59) for one face.
    demos = [
        # (edge_idx, push_steps, label)
        # Center pushes (face midpoint)
        (7, 5, "face0_center_depth5"),    # +x face, center edge, medium push
        (22, 5, "face1_center_depth5"),   # +y face, center edge, medium push
        (37, 5, "face2_center_depth5"),   # -x face, center edge, medium push
        (52, 5, "face3_center_depth5"),   # -y face, center edge, medium push
        (37, 1, "face2_center_depth1"),   # -x face, short push
        (37, 10, "face2_center_depth10"), # -x face, max push
        # Corner pushes (edge offset to one end of the face)
        (0,  5, "face0_corner0_depth5"),  # +x face, low-t corner
        (14, 5, "face0_corner1_depth5"),  # +x face, high-t corner (opposite)
        (15, 5, "face1_corner0_depth5"),  # +y face, low-t corner
        (30, 5, "face2_corner0_depth5"),  # -x face, low-t corner
        (44, 5, "face2_corner1_depth5"),  # -x face, high-t corner (opposite)
        (45, 5, "face3_corner0_depth5"),  # -y face, low-t corner
    ]

    for obj in OBJECT_CONFIGS:
        print(f"Rendering push videos for {obj.description} @ {args.push_speed} rad/s")
        print(f"  Object half-sizes: {obj.half_size_x*100:.1f}x{obj.half_size_y*100:.1f}cm")
        for edge_idx, push_steps, label in demos:
            render_push(
                obj, edge_idx, push_steps,
                output_path=output_dir / f"push_{obj.name}_{label}{args.output_suffix}.mp4",
                push_speed=args.push_speed,
            )
        print()

    print(f"Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
