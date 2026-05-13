"""Log every-tick telemetry for one generator-side primitive (pre-settle, push, post-settle).

Reproduces the generator's exact control loop and writes a CSV with one row per sim tick:
  tick, phase, obj_x, obj_y, obj_theta, car_x, car_y, car_theta,
  wheel_left_vel, wheel_right_vel, wheel_left_ctrl, wheel_right_ctrl

Phases: 0=pre_settle, 1=push, 2=post_settle.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from car_model.parameters import default_parameters
from generate_car_primitives import (
    OBJECT_CONFIGS,
    generate_edge_points,
    generate_scene_xml,
    quat_to_yaw,
    yaw_to_quat,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="wide", choices=["square", "wide", "tall"])
    ap.add_argument("--edge", type=int, default=22)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--push-speed", type=float, default=10.0)
    ap.add_argument("--push-step-duration", type=float, default=0.5)
    ap.add_argument("--settle-steps", type=int, default=500)
    ap.add_argument("--entry-ramp-ticks", type=int, default=0,
                    help="If >0, BEFORE the main push spin wheels with ctrl ramped 0->push_speed "
                         "over this many ticks (phase=4). 0 = instant start (default).")
    ap.add_argument("--exit-ramp-ticks", type=int, default=0,
                    help="If >0, after the main push run this many extra ticks linearly ramping "
                         "ctrl from push_speed down to 0 (phase=3). 0 = rectangular pulse (default).")
    ap.add_argument("--clearance", type=float, default=0.005,
                    help="Gap (m) between car bumper and object face at push start. "
                         "0.005 matches generator default; 0.037 matches runtime offset.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    obj = next(c for c in OBJECT_CONFIGS if c.name == args.shape)
    car_params = default_parameters()

    # Build scene XML (same as generator)
    scene_xml = generate_scene_xml(obj, car_params)
    scene_path = PROJECT_ROOT / "assets" / "mjcf" / f"primitive_gen_{obj.name}.xml"
    scene_path.write_text(scene_xml, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    steps_per_push = int(args.push_step_duration / dt)

    # IDs
    car_fj = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")]
    car_fj_v = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")]
    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")
    left_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
    right_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")
    left_v = model.jnt_dofadr[left_jnt]
    right_v = model.jnt_dofadr[right_jnt]
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_1_movable")
    obj_jnt_q = model.jnt_qposadr[model.body_jntadr[obj_body]]

    # Place object at origin, car at edge
    edges = generate_edge_points(
        obj.half_size_x, obj.half_size_y,
        points_per_face=15,
        robot_half_length=car_params.body_half_length_m,
        clearance=args.clearance,
    )
    edge_x, edge_y, heading = edges[args.edge]

    mujoco.mj_resetData(model, data)
    data.qpos[obj_jnt_q : obj_jnt_q + 3] = [0, 0, obj.half_size_z]
    data.qpos[obj_jnt_q + 3 : obj_jnt_q + 7] = [1, 0, 0, 0]
    car_z = car_params.wheel_radius_m + car_params.scene_spawn_height_m
    data.qpos[car_fj : car_fj + 3] = [edge_x, edge_y, car_z]
    data.qpos[car_fj + 3 : car_fj + 7] = yaw_to_quat(heading)
    mujoco.mj_forward(model, data)

    def quat_to_pitch(q):
        # MuJoCo qpos quat is [w,x,y,z]
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        return math.asin(sinp)

    # Logging
    rows = []
    def snap(tick: int, phase: int):
        obj_pos = data.qpos[obj_jnt_q : obj_jnt_q + 3]
        obj_quat = data.qpos[obj_jnt_q + 3 : obj_jnt_q + 7]
        car_pos = data.qpos[car_fj : car_fj + 3]
        car_quat = data.qpos[car_fj : car_fj + 0]  # placeholder
        car_quat = data.qpos[car_fj + 3 : car_fj + 7]
        rows.append((
            tick, phase,
            float(obj_pos[0]), float(obj_pos[1]), quat_to_yaw(obj_quat),
            float(car_pos[0]), float(car_pos[1]), float(car_pos[2]),
            quat_to_yaw(car_quat), quat_to_pitch(car_quat),
            float(data.qvel[left_v]), float(data.qvel[right_v]),
            float(data.ctrl[left_act]), float(data.ctrl[right_act]),
        ))

    tick = 0

    # Phase 0: pre-settle
    data.ctrl[left_act] = 0
    data.ctrl[right_act] = 0
    for _ in range(args.settle_steps):
        mujoco.mj_step(model, data); tick += 1
        snap(tick, 0)

    obj_before = data.qpos[obj_jnt_q : obj_jnt_q + 3].copy()
    yaw_before = quat_to_yaw(data.qpos[obj_jnt_q + 3 : obj_jnt_q + 7])

    # Phase 4 (optional): entry ramp — linearly increase ctrl from 0 -> push_speed
    if args.entry_ramp_ticks > 0:
        N = args.entry_ramp_ticks
        for k in range(1, N + 1):
            cmd = args.push_speed * (k / N)
            data.ctrl[left_act] = cmd
            data.ctrl[right_act] = cmd
            mujoco.mj_step(model, data); tick += 1
            snap(tick, 4)

    # Phase 1: push
    data.ctrl[left_act] = args.push_speed
    data.ctrl[right_act] = args.push_speed
    for _ in range(args.depth * steps_per_push):
        mujoco.mj_step(model, data); tick += 1
        snap(tick, 1)

    # Phase 3 (optional): exit ramp — linearly decrease ctrl from push_speed to 0
    if args.exit_ramp_ticks > 0:
        N = args.exit_ramp_ticks
        for k in range(1, N + 1):
            cmd = args.push_speed * (1.0 - k / N)
            data.ctrl[left_act] = cmd
            data.ctrl[right_act] = cmd
            mujoco.mj_step(model, data); tick += 1
            snap(tick, 3)

    # Phase 2: post-settle
    data.ctrl[left_act] = 0
    data.ctrl[right_act] = 0
    for _ in range(args.settle_steps):
        mujoco.mj_step(model, data); tick += 1
        snap(tick, 2)

    obj_after = data.qpos[obj_jnt_q : obj_jnt_q + 3].copy()
    yaw_after = quat_to_yaw(data.qpos[obj_jnt_q + 3 : obj_jnt_q + 7])
    dx, dy = obj_after[0] - obj_before[0], obj_after[1] - obj_before[1]
    dtheta = yaw_after - yaw_before
    while dtheta > math.pi: dtheta -= 2 * math.pi
    while dtheta < -math.pi: dtheta += 2 * math.pi

    out_dir = PROJECT_ROOT / "artifacts" / "push_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (
        out_dir / f"log_{obj.name}_edge{args.edge}_depth{args.depth}.csv"
    )
    arr = np.array(rows, dtype=[
        ("tick", "i4"), ("phase", "i4"),
        ("obj_x", "f8"), ("obj_y", "f8"), ("obj_theta", "f8"),
        ("car_x", "f8"), ("car_y", "f8"), ("car_z", "f8"),
        ("car_theta", "f8"), ("car_pitch", "f8"),
        ("wheel_left_vel", "f8"), ("wheel_right_vel", "f8"),
        ("wheel_left_ctrl", "f8"), ("wheel_right_ctrl", "f8"),
    ])
    header = ",".join(arr.dtype.names)
    with open(out_path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

    n_pre = sum(1 for r in rows if r[1] == 0)
    n_entry = sum(1 for r in rows if r[1] == 4)
    n_push = sum(1 for r in rows if r[1] == 1)
    n_exit = sum(1 for r in rows if r[1] == 3)
    n_post = sum(1 for r in rows if r[1] == 2)
    print(f"Wrote {out_path} ({len(rows)} rows)")
    print(f"  pre-settle ticks:  {n_pre}")
    if n_entry:
        print(f"  entry-ramp ticks:  {n_entry}")
    print(f"  push ticks:        {n_push}")
    if n_exit:
        print(f"  exit-ramp ticks:   {n_exit}")
    print(f"  post-settle ticks: {n_post}")
    print(f"Recorded delta (matches what generator would store):")
    print(f"  dx={dx*1000:.2f}mm  dy={dy*1000:.2f}mm  dtheta={math.degrees(dtheta):.2f}deg")


if __name__ == "__main__":
    main()
