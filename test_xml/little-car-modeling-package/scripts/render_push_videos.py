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
    cam.elevation = -60

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

    # Phase 2: Push
    data.ctrl[left_act] = push_speed
    data.ctrl[right_act] = push_speed
    for _ in range(push_steps * steps_per_push):
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
    output_dir = PROJECT_ROOT / "artifacts" / "push_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use square object for demo
    obj = OBJECT_CONFIGS[0]  # square
    print(f"Rendering push videos for {obj.description}")
    print(f"  Object half-sizes: {obj.half_size_x*100:.1f}x{obj.half_size_y*100:.1f}cm")
    print()

    # Render one push per face at medium depth, plus one at center edge max depth
    demos = [
        # (edge_idx, push_steps, label)
        (7, 5, "face0_center_depth5"),    # +x face, center edge, medium push
        (22, 5, "face1_center_depth5"),   # +y face, center edge, medium push
        (37, 5, "face2_center_depth5"),   # -x face, center edge, medium push
        (52, 5, "face3_center_depth5"),   # -y face, center edge, medium push
        (37, 1, "face2_center_depth1"),   # -x face, short push
        (37, 10, "face2_center_depth10"), # -x face, max push
    ]

    for edge_idx, push_steps, label in demos:
        render_push(
            obj, edge_idx, push_steps,
            output_path=output_dir / f"push_{obj.name}_{label}.mp4",
        )

    print(f"\nVideos saved to: {output_dir}")


if __name__ == "__main__":
    main()
