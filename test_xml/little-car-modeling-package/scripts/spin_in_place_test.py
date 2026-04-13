"""Test pure in-place rotation: spin 360 degrees and measure XY drift."""
from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from car_model.generate_model import generate_all
from car_model.parameters import default_parameters


def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def run_spin_test(
    turn_speed: float = 10.0,
    target_revolutions: float = 1.0,
    settle_steps: int = 1000,
) -> dict:
    params = default_parameters()
    output = generate_all(PROJECT_ROOT / "assets", params=params)
    model = mujoco.MjModel.from_xml_path(str(output["mjcf_scene"]))
    data = mujoco.MjData(model)

    free_joint_qpos = model.jnt_qposadr[0]
    data.qpos[free_joint_qpos : free_joint_qpos + 3] = np.array([0.0, 0.0, params.scene_spawn_height_m])
    data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")

    # Settle
    data.ctrl[left_act] = 0.0
    data.ctrl[right_act] = 0.0
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    # Record start pose
    start_pos = data.qpos[free_joint_qpos : free_joint_qpos + 3].copy()
    start_yaw = quat_to_yaw(data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7])

    # Spin: left wheel backward, right wheel forward
    target_yaw_change = target_revolutions * 2.0 * math.pi
    total_yaw_accumulated = 0.0
    prev_yaw = start_yaw
    step_count = 0
    max_steps = 500000

    # Track XY trajectory for drift analysis
    xy_samples = [(float(start_pos[0]), float(start_pos[1]))]
    max_drift = 0.0

    data.ctrl[left_act] = -turn_speed
    data.ctrl[right_act] = turn_speed

    while step_count < max_steps:
        mujoco.mj_step(model, data)
        step_count += 1

        current_pos = data.qpos[free_joint_qpos : free_joint_qpos + 3].copy()
        current_yaw = quat_to_yaw(data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7])

        # Accumulate yaw change (handling wrapping)
        dyaw = current_yaw - prev_yaw
        if dyaw > math.pi:
            dyaw -= 2.0 * math.pi
        elif dyaw < -math.pi:
            dyaw += 2.0 * math.pi
        total_yaw_accumulated += dyaw
        prev_yaw = current_yaw

        # Track drift
        drift = math.sqrt(
            (current_pos[0] - start_pos[0]) ** 2
            + (current_pos[1] - start_pos[1]) ** 2
        )
        max_drift = max(max_drift, drift)

        if step_count % 100 == 0:
            xy_samples.append((float(current_pos[0]), float(current_pos[1])))

        if abs(total_yaw_accumulated) >= target_yaw_change:
            break

    # Stop and settle
    data.ctrl[left_act] = 0.0
    data.ctrl[right_act] = 0.0
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    end_pos = data.qpos[free_joint_qpos : free_joint_qpos + 3].copy()
    end_yaw = quat_to_yaw(data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7])

    final_drift = math.sqrt(
        (end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2
    )

    return {
        "turn_speed_rad_s": turn_speed,
        "target_revolutions": target_revolutions,
        "actual_yaw_change_deg": math.degrees(total_yaw_accumulated),
        "steps": step_count,
        "sim_time_s": step_count * model.opt.timestep,
        "start_xy_mm": (round(start_pos[0] * 1000, 3), round(start_pos[1] * 1000, 3)),
        "end_xy_mm": (round(end_pos[0] * 1000, 3), round(end_pos[1] * 1000, 3)),
        "final_drift_mm": round(final_drift * 1000, 4),
        "max_drift_during_spin_mm": round(max_drift * 1000, 4),
        "final_drift_as_pct_of_body": round(final_drift / params.body_size_m * 100, 2),
    }


if __name__ == "__main__":
    print("=== Spin-in-place drift test ===\n")

    for revs in [0.25, 0.5, 1.0, 2.0]:
        result = run_spin_test(turn_speed=10.0, target_revolutions=revs)
        print(f"  {revs} rev ({result['actual_yaw_change_deg']:.1f} deg):")
        print(f"    final drift:     {result['final_drift_mm']:.4f} mm")
        print(f"    max drift:       {result['max_drift_during_spin_mm']:.4f} mm")
        print(f"    drift/body size: {result['final_drift_as_pct_of_body']:.2f}%")
        print(f"    sim time:        {result['sim_time_s']:.2f} s")
        print()
