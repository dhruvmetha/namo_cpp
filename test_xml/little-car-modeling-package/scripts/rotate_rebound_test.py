"""Reproduce the in-place-rotation rebound seen during navigation.

Mirrors C++ DiffDriveNavigation::rotate_in_place exactly:
  - constant ω = angular_speed (sign chosen by sign of error)
  - exit when |target - yaw| < theta_threshold
  - zero control + step for wait_steps (passive coast)
  - measure final yaw

Expected symptom: target 90deg, controller exits near 90deg, then yaw
decays back toward ~87deg during the wait. We print yaw every control
tick across the entire phase including wait, and a CSV for plotting.

Run:
    python rotate_rebound_test.py                       # default 90deg sweep
    python rotate_rebound_test.py --target_deg 180 --wait_steps 60
    python rotate_rebound_test.py --csv out.csv
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

from car_model.generate_model import generate_all
from car_model.parameters import default_parameters


CONTROL_DT = 0.01  # matches NAMOEnvironment::apply_control


def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return float(math.atan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def step_control_tick(model, data, n_substeps: int) -> None:
    for _ in range(n_substeps):
        mujoco.mj_step(model, data)


def rotate_in_place(
    model,
    data,
    qpos_adr: int,
    left_act: int,
    right_act: int,
    target_theta: float,
    *,
    angular_speed: float = 1.0,
    theta_threshold: float = 0.05,
    wait_steps: int = 30,
    max_ctrl_steps: int = 6000,
    wheelbase: float,
    wheel_radius: float,
    log,
):
    """Mirror of C++ rotate_in_place. Returns (steps, samples)."""
    n_sub = max(1, int(round(CONTROL_DT / model.opt.timestep)))

    # Active rotation phase
    active_steps = 0
    while active_steps < max_ctrl_steps:
        yaw = quat_to_yaw(data.qpos[qpos_adr + 3 : qpos_adr + 7])
        err = wrap_angle(target_theta - yaw)
        log("active", active_steps, yaw, err,
            data.qvel[model.jnt_dofadr[0] + 5])  # qvel z-rot of freejoint
        if abs(err) < theta_threshold:
            break

        omega = angular_speed * (1.0 if err > 0 else -1.0)
        # Diff-drive wheel velocities for pure rotation
        wheel_omega_left = (-omega * wheelbase / 2.0) / wheel_radius
        wheel_omega_right = (+omega * wheelbase / 2.0) / wheel_radius
        data.ctrl[left_act] = wheel_omega_left
        data.ctrl[right_act] = wheel_omega_right
        step_control_tick(model, data, n_sub)
        active_steps += 1

    # Passive wait: zero control, keep stepping
    data.ctrl[left_act] = 0.0
    data.ctrl[right_act] = 0.0
    for w in range(wait_steps):
        step_control_tick(model, data, n_sub)
        yaw = quat_to_yaw(data.qpos[qpos_adr + 3 : qpos_adr + 7])
        err = wrap_angle(target_theta - yaw)
        log("wait", active_steps + w + 1, yaw, err,
            data.qvel[model.jnt_dofadr[0] + 5])

    return active_steps, wait_steps


def compute_wheel_geometry(model) -> tuple[float, float]:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_collision")
    wheel_radius = float(model.geom_size[geom_id, 0])
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel")
    wheelbase = abs(float(model.body_pos[left_body, 1]) -
                    float(model.body_pos[right_body, 1]))
    return wheel_radius, wheelbase


def run_one(target_deg: float, *, angular_speed: float, theta_threshold: float,
            wait_steps: int, csv_path: Path | None) -> dict:
    params = default_parameters()
    output = generate_all(PROJECT_ROOT / "assets", params=params)
    model = mujoco.MjModel.from_xml_path(str(output["mjcf_scene"]))
    data = mujoco.MjData(model)

    qpos_adr = model.jnt_qposadr[0]
    data.qpos[qpos_adr : qpos_adr + 3] = np.array([0.0, 0.0, params.scene_spawn_height_m])
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")
    wheel_radius, wheelbase = compute_wheel_geometry(model)

    # Initial settle
    data.ctrl[left_act] = 0.0
    data.ctrl[right_act] = 0.0
    for _ in range(1000):
        mujoco.mj_step(model, data)

    samples: list[tuple[str, int, float, float, float]] = []
    def log(phase, step, yaw, err, yaw_rate):
        samples.append((phase, step, yaw, err, yaw_rate))

    # We also track the post-step pose at the iteration that triggered exit.
    # The active loop logs *pre-step*, so the actual yaw at exit is captured by
    # the FIRST wait sample (which is right after exit, before any wait tick).
    # We capture it explicitly below.

    target_theta = math.radians(target_deg)
    active, _ = rotate_in_place(
        model, data, qpos_adr, left_act, right_act, target_theta,
        angular_speed=angular_speed,
        theta_threshold=theta_threshold,
        wait_steps=wait_steps,
        wheelbase=wheelbase, wheel_radius=wheel_radius,
        log=log,
    )

    # The active loop logs PRE-step then increments. The break happens at the
    # iteration whose pre-step yaw satisfies |err|<threshold, so the LAST
    # active sample (samples[active]) is the post-step yaw of the prior tick
    # AND the value that triggered exit.
    last_active_idx = active  # samples[active] is the breaking sample
    yaw_at_exit = samples[last_active_idx][2]
    err_at_exit = samples[last_active_idx][3]
    yaw_final = samples[-1][2]
    err_final = samples[-1][3]

    rebound_rad = wrap_angle(yaw_at_exit - yaw_final)
    result = {
        "target_deg": target_deg,
        "active_ticks": active,
        "wait_ticks": wait_steps,
        "yaw_at_exit_deg": math.degrees(yaw_at_exit),
        "yaw_final_deg": math.degrees(yaw_final),
        "err_at_exit_deg": math.degrees(err_at_exit),
        "err_final_deg": math.degrees(err_final),
        "rebound_deg": math.degrees(rebound_rad),
        "wheelbase_m": wheelbase,
        "wheel_radius_m": wheel_radius,
    }

    if csv_path is not None:
        with csv_path.open("w") as f:
            f.write("phase,step,t_s,yaw_deg,err_deg,yaw_rate_dps\n")
            for phase, step, yaw, err, rate in samples:
                t = step * CONTROL_DT
                f.write(f"{phase},{step},{t:.4f},{math.degrees(yaw):.6f},"
                        f"{math.degrees(err):.6f},{math.degrees(rate):.4f}\n")
        result["csv"] = str(csv_path)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target_deg", type=float, default=None,
                    help="If set, run only this target. Otherwise sweep [45,90,180].")
    ap.add_argument("--angular_speed", type=float, default=1.0)
    ap.add_argument("--theta_threshold", type=float, default=0.05)
    ap.add_argument("--wait_steps", type=int, default=30)
    ap.add_argument("--csv", type=str, default=None,
                    help="If set, write per-tick yaw/err CSV (only valid with --target_deg).")
    args = ap.parse_args()

    targets = [args.target_deg] if args.target_deg is not None else [45.0, 90.0, 180.0]
    for t in targets:
        csv_path = Path(args.csv) if (args.csv and args.target_deg is not None) else None
        r = run_one(t,
                    angular_speed=args.angular_speed,
                    theta_threshold=args.theta_threshold,
                    wait_steps=args.wait_steps,
                    csv_path=csv_path)
        print(f"--- target={r['target_deg']:.1f}deg "
              f"(omega={args.angular_speed} rad/s, "
              f"thresh={args.theta_threshold:.3f} rad, "
              f"wait={args.wait_steps} ticks) ---")
        print(f"  active ticks    : {r['active_ticks']}  ({r['active_ticks']*CONTROL_DT:.2f}s)")
        print(f"  yaw at exit     : {r['yaw_at_exit_deg']:.4f}deg  "
              f"(err {r['err_at_exit_deg']:+.4f}deg)")
        print(f"  yaw after wait  : {r['yaw_final_deg']:.4f}deg  "
              f"(err {r['err_final_deg']:+.4f}deg)")
        print(f"  rebound (exit→final): {r['rebound_deg']:+.4f}deg")
        if "csv" in r:
            print(f"  csv: {r['csv']}")
        print()


if __name__ == "__main__":
    main()
