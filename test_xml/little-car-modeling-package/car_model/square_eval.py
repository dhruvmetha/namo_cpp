from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from car_model.generate_model import generate_all
from car_model.parameters import CarParameters, default_parameters


@dataclass(frozen=True)
class SegmentResult:
    segment_index: int
    start_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    heading_start_rad: float
    heading_end_rad: float
    distance_along_heading_m: float
    yaw_change_rad: float
    tipped_during_segment: bool
    min_up_dot: float
    min_chassis_geom_z_m: float


@dataclass(frozen=True)
class SquareEvalResult:
    final_position_m: tuple[float, float, float]
    final_yaw_rad: float
    final_yaw_deg: float
    closure_error_m: float
    heading_error_rad: float
    heading_error_deg: float
    tipped_any_time: bool
    min_up_dot: float
    min_chassis_geom_z_m: float
    segments: list[SegmentResult]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["segments"] = [asdict(segment) for segment in self.segments]
        return data


def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def wrap_angle(angle_rad: float) -> float:
    return float(math.atan2(math.sin(angle_rad), math.cos(angle_rad)))


def _pose_state(data: mujoco.MjData, free_joint_qpos: int) -> tuple[np.ndarray, float]:
    pos = data.qpos[free_joint_qpos : free_joint_qpos + 3].copy()
    yaw = quat_to_yaw(data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7].copy())
    return pos, yaw


def evaluate_square_path(
    asset_root: Path,
    params: CarParameters | None = None,
    side_length_m: float = 0.10,
    turn_angle_rad: float = math.pi / 2.0,
    settle_steps: int = 1000,
    settle_between_phases_steps: int = 100,
    forward_speed_ctrl: float = 10.0,
    heading_kp: float = 4.0,
    turn_speed_ctrl: float = 10.0,
    max_forward_steps: int = 50000,
    max_turn_steps: int = 50000,
    tip_up_dot_threshold: float = 0.95,
    tip_chassis_geom_z_threshold_m: float | None = None,
    step_callback: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
) -> SquareEvalResult:
    params = params or default_parameters()
    if tip_chassis_geom_z_threshold_m is None:
        tip_chassis_geom_z_threshold_m = 0.8 * params.body_center_z_m
    output = generate_all(asset_root, params=params)
    model = mujoco.MjModel.from_xml_path(str(output["mjcf_scene"]))
    data = mujoco.MjData(model)

    free_joint_qpos = model.jnt_qposadr[0]
    data.qpos[free_joint_qpos : free_joint_qpos + 3] = np.array([0.0, 0.0, params.scene_spawn_height_m])
    data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    if step_callback is not None:
        step_callback(model, data)

    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")
    car_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
    front_chassis_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_chassis_collision")
    rear_chassis_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rear_chassis_collision")

    min_up_dot = float("inf")
    min_chassis_geom_z = float("inf")
    tipped_any_time = False

    def update_stability_metrics() -> tuple[bool, float, float]:
        nonlocal min_up_dot, min_chassis_geom_z, tipped_any_time
        up_dot = float(data.xmat[car_body_id][8])
        chassis_geom_z = min(
            float(data.geom_xpos[front_chassis_geom_id][2]),
            float(data.geom_xpos[rear_chassis_geom_id][2]),
        )
        min_up_dot = min(min_up_dot, up_dot)
        min_chassis_geom_z = min(min_chassis_geom_z, chassis_geom_z)
        tipped = up_dot < tip_up_dot_threshold or chassis_geom_z < tip_chassis_geom_z_threshold_m
        tipped_any_time = tipped_any_time or tipped
        return tipped, up_dot, chassis_geom_z

    def step_and_monitor(steps: int) -> tuple[bool, float, float]:
        phase_tipped = False
        phase_min_up_dot = float("inf")
        phase_min_chassis_geom_z = float("inf")
        for _ in range(steps):
            mujoco.mj_step(model, data)
            if step_callback is not None:
                step_callback(model, data)
            tipped, up_dot, chassis_geom_z = update_stability_metrics()
            phase_tipped = phase_tipped or tipped
            phase_min_up_dot = min(phase_min_up_dot, up_dot)
            phase_min_chassis_geom_z = min(phase_min_chassis_geom_z, chassis_geom_z)
        return phase_tipped, phase_min_up_dot, phase_min_chassis_geom_z

    data.ctrl[left_act] = 0.0
    data.ctrl[right_act] = 0.0
    step_and_monitor(settle_steps)

    initial_pos, initial_yaw = _pose_state(data, free_joint_qpos)
    segments: list[SegmentResult] = []

    for segment_index in range(4):
        segment_start_pos, segment_heading_start = _pose_state(data, free_joint_qpos)
        heading_vector = np.array([math.cos(segment_heading_start), math.sin(segment_heading_start)])
        segment_tipped = False
        segment_min_up_dot = float("inf")
        segment_min_chassis_geom_z = float("inf")

        for _ in range(max_forward_steps):
            current_pos, current_yaw = _pose_state(data, free_joint_qpos)
            heading_error = wrap_angle(current_yaw - segment_heading_start)
            data.ctrl[left_act] = forward_speed_ctrl + heading_kp * heading_error
            data.ctrl[right_act] = forward_speed_ctrl - heading_kp * heading_error
            tipped, up_dot, chassis_geom_z = step_and_monitor(1)
            segment_tipped = segment_tipped or tipped
            segment_min_up_dot = min(segment_min_up_dot, up_dot)
            segment_min_chassis_geom_z = min(segment_min_chassis_geom_z, chassis_geom_z)
            distance_along_heading = float(np.dot(current_pos[:2] - segment_start_pos[:2], heading_vector))
            if distance_along_heading >= side_length_m:
                break
        else:
            raise RuntimeError(f"Forward segment {segment_index + 1} did not reach {side_length_m:.3f} m within {max_forward_steps} steps")

        data.ctrl[left_act] = 0.0
        data.ctrl[right_act] = 0.0
        tipped, up_dot, chassis_geom_z = step_and_monitor(settle_between_phases_steps)
        segment_tipped = segment_tipped or tipped
        segment_min_up_dot = min(segment_min_up_dot, up_dot)
        segment_min_chassis_geom_z = min(segment_min_chassis_geom_z, chassis_geom_z)

        turn_start_pos, turn_heading_start = _pose_state(data, free_joint_qpos)
        for _ in range(max_turn_steps):
            current_pos, current_yaw = _pose_state(data, free_joint_qpos)
            yaw_change = wrap_angle(current_yaw - turn_heading_start)
            data.ctrl[left_act] = -turn_speed_ctrl
            data.ctrl[right_act] = turn_speed_ctrl
            tipped, up_dot, chassis_geom_z = step_and_monitor(1)
            segment_tipped = segment_tipped or tipped
            segment_min_up_dot = min(segment_min_up_dot, up_dot)
            segment_min_chassis_geom_z = min(segment_min_chassis_geom_z, chassis_geom_z)
            if yaw_change >= turn_angle_rad:
                break
        else:
            raise RuntimeError(f"Turn segment {segment_index + 1} did not reach {turn_angle_rad:.3f} rad within {max_turn_steps} steps")

        data.ctrl[left_act] = 0.0
        data.ctrl[right_act] = 0.0
        tipped, up_dot, chassis_geom_z = step_and_monitor(settle_between_phases_steps)
        segment_tipped = segment_tipped or tipped
        segment_min_up_dot = min(segment_min_up_dot, up_dot)
        segment_min_chassis_geom_z = min(segment_min_chassis_geom_z, chassis_geom_z)

        segment_end_pos, segment_heading_end = _pose_state(data, free_joint_qpos)
        distance_along_heading = float(np.dot(segment_end_pos[:2] - segment_start_pos[:2], heading_vector))
        yaw_change = wrap_angle(segment_heading_end - segment_heading_start)
        segments.append(
            SegmentResult(
                segment_index=segment_index + 1,
                start_xy_m=(float(segment_start_pos[0]), float(segment_start_pos[1])),
                end_xy_m=(float(segment_end_pos[0]), float(segment_end_pos[1])),
                heading_start_rad=float(segment_heading_start),
                heading_end_rad=float(segment_heading_end),
                distance_along_heading_m=distance_along_heading,
                yaw_change_rad=yaw_change,
                tipped_during_segment=segment_tipped,
                min_up_dot=segment_min_up_dot,
                min_chassis_geom_z_m=segment_min_chassis_geom_z,
            )
        )

    final_pos, final_yaw = _pose_state(data, free_joint_qpos)
    closure_error = float(np.linalg.norm(final_pos[:2] - initial_pos[:2]))
    heading_error = wrap_angle(final_yaw - initial_yaw)

    return SquareEvalResult(
        final_position_m=(float(final_pos[0]), float(final_pos[1]), float(final_pos[2])),
        final_yaw_rad=float(final_yaw),
        final_yaw_deg=float(math.degrees(final_yaw)),
        closure_error_m=closure_error,
        heading_error_rad=heading_error,
        heading_error_deg=float(math.degrees(heading_error)),
        tipped_any_time=tipped_any_time,
        min_up_dot=min_up_dot,
        min_chassis_geom_z_m=min_chassis_geom_z,
        segments=segments,
    )
