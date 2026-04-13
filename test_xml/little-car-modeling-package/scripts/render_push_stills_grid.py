"""Render a grid of push stills: center + extreme edges, multiple depths."""
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
    generate_edge_points,
    generate_scene_xml,
    yaw_to_quat,
)


def render_single_push(model, data, renderer, cam, obj_config, car_params,
                        edge_idx, edges, push_steps, push_speed=10.0,
                        push_step_duration=0.5, settle_steps=500):
    """Run a push and return (before_frame, after_frame, displacement_mm)."""
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

    obj_before = data.qpos[obj_jnt:obj_jnt + 3].copy()

    # Capture before frame
    renderer.update_scene(data, camera=cam)
    before_frame = renderer.render().copy()

    # Push
    data.ctrl[left_act] = push_speed
    data.ctrl[right_act] = push_speed
    for _ in range(push_steps * steps_per_push):
        mujoco.mj_step(model, data)

    # Stop and settle
    data.ctrl[left_act] = 0
    data.ctrl[right_act] = 0
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    obj_after = data.qpos[obj_jnt:obj_jnt + 3].copy()
    displacement = math.sqrt((obj_after[0] - obj_before[0]) ** 2 + (obj_after[1] - obj_before[1]) ** 2)

    # Capture after frame
    renderer.update_scene(data, camera=cam)
    after_frame = renderer.render().copy()

    return before_frame, after_frame, displacement * 1000


def main():
    car_params = default_parameters()
    generate_all(PROJECT_ROOT / "assets", params=car_params)

    obj = OBJECT_CONFIGS[0]  # square
    scene_xml = generate_scene_xml(obj, car_params)
    scene_path = PROJECT_ROOT / "assets" / "mjcf" / "stills_grid_gen.xml"
    scene_path.write_text(scene_xml, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)

    edges = generate_edge_points(
        obj.half_size_x, obj.half_size_y,
        points_per_face=15,
        robot_half_length=car_params.body_half_length_m,
        clearance=0.005,
    )

    # Top-down camera
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.03]
    cam.distance = 0.55
    cam.azimuth = 90
    cam.elevation = -90  # directly overhead

    out_dir = PROJECT_ROOT / "artifacts" / "push_stills"
    out_dir.mkdir(parents=True, exist_ok=True)

    # For face 2 (-x face, pushing in +x direction):
    # edge 30 = first point (extreme left)
    # edge 37 = center point
    # edge 44 = last point (extreme right)
    #
    # For face 0 (+x face, pushing in -x direction):
    # edge 0 = first, edge 7 = center, edge 14 = last

    configs = [
        # (face_label, edge_idx, push_steps, description)
        # -x face (pushing +x): extremes + center
        ("-x face, left extreme",   30, 5, "face2_edge0"),
        ("-x face, center",         37, 5, "face2_center"),
        ("-x face, right extreme",  44, 5, "face2_edge14"),
        # +x face (pushing -x): extremes + center
        ("+x face, left extreme",   0,  5, "face0_edge0"),
        ("+x face, center",         7,  5, "face0_center"),
        ("+x face, right extreme",  14, 5, "face0_edge14"),
        # +y face (pushing -y): extremes + center
        ("+y face, left extreme",   15, 5, "face1_edge0"),
        ("+y face, center",         22, 5, "face1_center"),
        ("+y face, right extreme",  29, 5, "face1_edge14"),
        # Depth comparison on center edge
        ("-x face, center, depth 1",  37, 1, "face2_center_d1"),
        ("-x face, center, depth 5",  37, 5, "face2_center_d5"),
        ("-x face, center, depth 10", 37, 10, "face2_center_d10"),
    ]

    print(f"Rendering {len(configs)} push stills for {obj.description}")
    print(f"  Object: {obj.half_size_x*200:.1f}x{obj.half_size_y*200:.1f}cm")
    print(f"  Camera: top-down\n")

    for label, edge_idx, push_steps, filename in configs:
        before, after, disp = render_single_push(
            model, data, renderer, cam, obj, car_params,
            edge_idx, edges, push_steps,
        )

        # Save before and after side by side
        combined = np.concatenate([before, after], axis=1)
        path = out_dir / f"grid_{filename}_d{push_steps}.png"
        imageio.imwrite(str(path), combined)
        print(f"  {label} (depth={push_steps}): {disp:.1f}mm → {path.name}")

    renderer.close()
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
