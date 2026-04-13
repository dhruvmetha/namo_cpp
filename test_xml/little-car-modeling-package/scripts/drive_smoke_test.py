from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from car_model.generate_model import generate_all
from car_model.parameters import default_parameters

PYTHON = "/home/shanoriel/miniforge3/envs/leworldmodel/bin/python"

# Derive tipped threshold from car geometry (80% of chassis geom center height)
_params = default_parameters()
_TIP_CHASSIS_Z_THRESHOLD = 0.8 * _params.body_center_z_m


def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def run_behavior(name: str, left_ctrl: float, right_ctrl: float, steps: int = 1500) -> dict[str, float | str]:
    params = default_parameters()
    output = generate_all(PROJECT_ROOT / "assets")
    model = mujoco.MjModel.from_xml_path(str(output["mjcf_scene"]))
    data = mujoco.MjData(model)

    free_joint_qpos = model.jnt_qposadr[0]
    data.qpos[free_joint_qpos : free_joint_qpos + 3] = np.array([0.0, 0.0, params.scene_spawn_height_m])
    data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")
    car_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
    chassis_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_chassis_collision")
    support_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rear_support")

    start_pos = data.qpos[free_joint_qpos : free_joint_qpos + 3].copy()
    start_yaw = quat_to_yaw(data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7].copy())

    data.ctrl[left_act] = left_ctrl
    data.ctrl[right_act] = right_ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)

    end_pos = data.qpos[free_joint_qpos : free_joint_qpos + 3].copy()
    end_yaw = quat_to_yaw(data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7].copy())
    up_dot = float(data.xmat[car_body_id][8])
    tipped = up_dot < 0.95 or float(data.geom_xpos[chassis_geom_id][2]) < _TIP_CHASSIS_Z_THRESHOLD

    return {
        "behavior": name,
        "left_ctrl": left_ctrl,
        "right_ctrl": right_ctrl,
        "dx_m": float(end_pos[0] - start_pos[0]),
        "dy_m": float(end_pos[1] - start_pos[1]),
        "yaw_delta_rad": float(end_yaw - start_yaw),
        "up_dot": up_dot,
        "chassis_geom_z_m": float(data.geom_xpos[chassis_geom_id][2]),
        "support_geom_z_m": float(data.geom_xpos[support_geom_id][2]),
        "tipped": str(tipped),
    }


if __name__ == "__main__":
    print(f"Using Python: {PYTHON}")
    print("Coordinate convention: +x forward, +y left, +z up. Rear support pad is at negative x.")
    for spec in [
        ("forward", 18.0, 18.0),
        ("backward", -18.0, -18.0),
        ("left_turn", 8.0, 18.0),
        ("right_turn", 18.0, 8.0),
    ]:
        print(run_behavior(*spec))
