"""Detailed spin-in-place quality test: check for jerky motion, velocity smoothness."""
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


def run_spin_quality(turn_speed: float = 10.0, target_deg: float = 360.0):
    params = default_parameters()
    output = generate_all(PROJECT_ROOT / "assets", params=params)
    model = mujoco.MjModel.from_xml_path(str(output["mjcf_scene"]))
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    free_joint_qpos = model.jnt_qposadr[0]
    data.qpos[free_joint_qpos : free_joint_qpos + 3] = np.array([0.0, 0.0, params.scene_spawn_height_m])
    data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")

    # Settle
    for _ in range(1000):
        mujoco.mj_step(model, data)

    start_pos = data.qpos[free_joint_qpos : free_joint_qpos + 3].copy()
    start_yaw = quat_to_yaw(data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7])

    # Spin
    target_rad = math.radians(target_deg)
    total_yaw = 0.0
    prev_yaw = start_yaw

    # Sample every N steps for analysis
    sample_every = 50  # every 50 steps = every 0.1s at dt=0.002
    times = []
    yaw_rates = []  # deg/s
    xy_drifts = []  # mm
    x_positions = []
    y_positions = []
    yaw_accum = []

    prev_sample_yaw = start_yaw
    prev_sample_time = 0.0
    step = 0

    data.ctrl[left_act] = -turn_speed
    data.ctrl[right_act] = turn_speed

    while abs(total_yaw) < target_rad and step < 500000:
        mujoco.mj_step(model, data)
        step += 1

        pos = data.qpos[free_joint_qpos : free_joint_qpos + 3]
        yaw = quat_to_yaw(data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7])

        dyaw = yaw - prev_yaw
        if dyaw > math.pi: dyaw -= 2 * math.pi
        elif dyaw < -math.pi: dyaw += 2 * math.pi
        total_yaw += dyaw
        prev_yaw = yaw

        if step % sample_every == 0:
            t = step * dt
            # Yaw rate since last sample
            sample_dyaw = yaw - prev_sample_yaw
            if sample_dyaw > math.pi: sample_dyaw -= 2 * math.pi
            elif sample_dyaw < -math.pi: sample_dyaw += 2 * math.pi
            sample_dt = t - prev_sample_time

            rate_deg_s = math.degrees(sample_dyaw / sample_dt) if sample_dt > 0 else 0
            drift_mm = math.sqrt((pos[0] - start_pos[0])**2 + (pos[1] - start_pos[1])**2) * 1000

            times.append(t)
            yaw_rates.append(rate_deg_s)
            xy_drifts.append(drift_mm)
            x_positions.append(float(pos[0]) * 1000)
            y_positions.append(float(pos[1]) * 1000)
            yaw_accum.append(math.degrees(total_yaw))

            prev_sample_yaw = yaw
            prev_sample_time = t

    # Analysis
    rates = np.array(yaw_rates)
    # Skip first few samples (acceleration phase)
    steady_start = max(3, len(rates) // 5)
    steady_rates = rates[steady_start:]

    print(f"=== Spin Quality: {target_deg}° at ctrl={turn_speed} ===\n")
    print(f"Duration: {step * dt:.2f}s ({step} steps)")
    print(f"Samples: {len(times)} (every {sample_every * dt * 1000:.0f}ms)")
    print()

    print("Yaw rate (steady-state):")
    print(f"  Mean:   {np.mean(steady_rates):.2f} deg/s")
    print(f"  Std:    {np.std(steady_rates):.3f} deg/s")
    print(f"  Min:    {np.min(steady_rates):.2f} deg/s")
    print(f"  Max:    {np.max(steady_rates):.2f} deg/s")
    print(f"  Jitter: {np.max(steady_rates) - np.min(steady_rates):.3f} deg/s ({(np.max(steady_rates) - np.min(steady_rates)) / np.mean(steady_rates) * 100:.2f}% of mean)")
    print()

    # Check for sudden rate changes (jerk)
    rate_diffs = np.diff(steady_rates)
    print("Rate changes between samples (jerk indicator):")
    print(f"  Mean |delta|: {np.mean(np.abs(rate_diffs)):.4f} deg/s")
    print(f"  Max |delta|:  {np.max(np.abs(rate_diffs)):.4f} deg/s")
    print(f"  Std delta:    {np.std(rate_diffs):.4f} deg/s")
    print()

    drifts = np.array(xy_drifts)
    print("XY drift:")
    print(f"  Max:   {np.max(drifts):.4f} mm")
    print(f"  Final: {drifts[-1]:.4f} mm")
    print()

    # Check for oscillation in x/y
    x = np.array(x_positions)
    y = np.array(y_positions)
    print("Position oscillation:")
    print(f"  X range: {np.min(x):.4f} to {np.max(x):.4f} mm (span: {np.max(x) - np.min(x):.4f} mm)")
    print(f"  Y range: {np.min(y):.4f} to {np.max(y):.4f} mm (span: {np.max(y) - np.min(y):.4f} mm)")


if __name__ == "__main__":
    run_spin_quality(turn_speed=10.0, target_deg=360.0)
    print()
    print("=" * 50)
    print()
    run_spin_quality(turn_speed=10.0, target_deg=90.0)
