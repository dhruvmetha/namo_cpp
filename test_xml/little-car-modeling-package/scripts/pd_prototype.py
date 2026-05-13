"""Compare bang-bang vs PD controllers for rotate_in_place and drive_straight.

Runs all four modes on the standalone car scene, writes qpos dumps and CSV
traces, prints summary metrics for each. Use render_nav_video.py to make MP4s.

Modes:
  rotate_bb : current bang-bang rotate-in-place (constant ω until threshold,
              then ctrl=0 + 30-tick wait). Reproduces the C++ rotate_in_place.
  rotate_pd : new PD outer loop on chassis yaw, outputs commanded chassis ω,
              fed through diff-drive to the velocity-mode wheel actuators.
              No wait phase.
  drive_bb  : current drive-straight (equal wheel ω at v/r until within
              xy_threshold). Reproduces the C++ drive_straight_to.
  drive_pd  : new P controller on distance-to-endpoint along segment heading.
              Wheel speed shrinks smoothly to zero at endpoint. No wait.
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


# ---------- helpers ----------
def quat_to_yaw(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def make_scene(xml_override: str | None = None, car_xy=(0.0, 0.0)):
    """Load car-bearing scene. If xml_override given, use that XML (e.g.
    a NAMO env with obstacles). The car is placed at car_xy facing +x;
    we deliberately pick a location away from obstacles so the test is
    contact-free except for the wheels↔floor."""
    params = default_parameters()
    if xml_override:
        model = mujoco.MjModel.from_xml_path(xml_override)
    else:
        out = generate_all(PROJECT_ROOT / "assets", params=params)
        model = mujoco.MjModel.from_xml_path(str(out["mjcf_scene"]))
    data = mujoco.MjData(model)
    car_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")
    fj_qpos = int(model.jnt_qposadr[car_jid])
    fj_qvel = int(model.jnt_dofadr[car_jid])
    # Place car at requested xy, facing +x
    data.qpos[fj_qpos + 0] = car_xy[0]
    data.qpos[fj_qpos + 1] = car_xy[1]
    data.qpos[fj_qpos + 2] = params.scene_spawn_height_m
    data.qpos[fj_qpos + 3:fj_qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    # Geometry
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_collision")
    r = float(model.geom_size[geom, 0])
    lb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    rb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel")
    b = abs(float(model.body_pos[lb, 1]) - float(model.body_pos[rb, 1]))
    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")
    return dict(model=model, data=data, fj_qpos=fj_qpos, fj_qvel=fj_qvel,
                wheel_r=r, wheelbase=b, left_act=left_act, right_act=right_act,
                params=params)


def n_substeps(model):
    return max(1, int(round(CONTROL_DT / model.opt.timestep)))


def step_tick(model, data, n):
    for _ in range(n):
        mujoco.mj_step(model, data)


def settle(scene, n_steps=1000):
    m, d = scene["model"], scene["data"]
    d.ctrl[scene["left_act"]] = 0.0
    d.ctrl[scene["right_act"]] = 0.0
    for _ in range(n_steps):
        mujoco.mj_step(m, d)


def get_yaw(scene):
    d = scene["data"]
    a = scene["fj_qpos"]
    return quat_to_yaw(d.qpos[a + 3], d.qpos[a + 4], d.qpos[a + 5], d.qpos[a + 6])


def get_xy(scene):
    d = scene["data"]
    a = scene["fj_qpos"]
    return float(d.qpos[a + 0]), float(d.qpos[a + 1])


def get_yaw_rate(scene):
    return float(scene["data"].qvel[scene["fj_qvel"] + 5])


def get_velocity(scene):
    vx = float(scene["data"].qvel[scene["fj_qvel"] + 0])
    vy = float(scene["data"].qvel[scene["fj_qvel"] + 1])
    return vx, vy


def dump_qpos_line(model, data, phase):
    line = [str(phase), str(model.nq)]
    for i in range(model.nq):
        line.append(f"{data.qpos[i]:.6f}")
    return " ".join(line) + "\n"


# ---------- rotate-in-place: bang-bang (current C++ behavior) ----------
def rotate_bb(scene, target_theta, *, angular_speed=1.0, theta_threshold=0.05,
              wait_steps=30, max_steps=6000, qpos_path: Path):
    m, d = scene["model"], scene["data"]
    n_sub = n_substeps(m)
    b = scene["wheelbase"]
    r = scene["wheel_r"]
    LA, RA = scene["left_act"], scene["right_act"]
    log = qpos_path.open("w")
    samples = []  # (phase, t, yaw, yaw_rate)
    t_steps = 0

    # Active phase
    while t_steps < max_steps:
        yaw = get_yaw(scene)
        yr = get_yaw_rate(scene)
        err = wrap(target_theta - yaw)
        samples.append(("active", t_steps, math.degrees(yaw), math.degrees(yr)))
        log.write(dump_qpos_line(m, d, 0))
        if abs(err) < theta_threshold:
            break
        omega = angular_speed * (1 if err > 0 else -1)
        d.ctrl[LA] = (-omega * b / 2.0) / r
        d.ctrl[RA] = (+omega * b / 2.0) / r
        step_tick(m, d, n_sub)
        t_steps += 1

    # Wait phase
    d.ctrl[LA] = 0.0
    d.ctrl[RA] = 0.0
    for _ in range(wait_steps):
        step_tick(m, d, n_sub)
        t_steps += 1
        yaw = get_yaw(scene)
        yr = get_yaw_rate(scene)
        samples.append(("wait", t_steps, math.degrees(yaw), math.degrees(yr)))
        log.write(dump_qpos_line(m, d, 0))

    log.close()
    return samples


# ---------- rotate-in-place: TRAPEZOIDAL (closed-loop sqrt profile) ----------
def rotate_trap(scene, target_theta, *, alpha_max=5.0, omega_max=1.0,
                theta_converged=0.01, max_steps=6000, qpos_path: Path):
    """Closed-loop trapezoidal: ω_des = sqrt(2·α_max·|err|), saturated at ω_max.
    Re-evaluates each tick on actual remaining error, so slip is absorbed.
    Exit condition: |err| < theta_converged (position only, no speed gate)."""
    m, d = scene["model"], scene["data"]
    n_sub = n_substeps(m)
    b = scene["wheelbase"]; r = scene["wheel_r"]
    LA, RA = scene["left_act"], scene["right_act"]
    log = qpos_path.open("w")
    samples = []
    t_steps = 0

    while t_steps < max_steps:
        yaw = get_yaw(scene)
        yr = get_yaw_rate(scene)
        err = wrap(target_theta - yaw)
        samples.append(("active", t_steps, math.degrees(yaw), math.degrees(yr)))
        log.write(dump_qpos_line(m, d, 0))

        if abs(err) < theta_converged:
            break

        sgn = 1.0 if err >= 0 else -1.0
        w_des = math.sqrt(2.0 * alpha_max * abs(err))
        w_des = min(w_des, omega_max)
        omega_cmd = sgn * w_des

        d.ctrl[LA] = (-omega_cmd * b / 2.0) / r
        d.ctrl[RA] = (+omega_cmd * b / 2.0) / r
        step_tick(m, d, n_sub)
        t_steps += 1

    log.close()
    return samples


# ---------- drive-straight: TRAPEZOIDAL (closed-loop sqrt profile) ----------
def drive_trap(scene, end_xy, heading, *, accel_max=0.5, v_max=0.10,
               xy_converged=0.005, max_steps=6000, qpos_path: Path):
    m, d = scene["model"], scene["data"]
    n_sub = n_substeps(m)
    r = scene["wheel_r"]
    LA, RA = scene["left_act"], scene["right_act"]
    log = qpos_path.open("w")
    samples = []
    t_steps = 0
    cos_h = math.cos(heading); sin_h = math.sin(heading)

    while t_steps < max_steps:
        x, y = get_xy(scene)
        vx, vy = get_velocity(scene)
        dx = end_xy[0] - x; dy = end_xy[1] - y
        along = dx * cos_h + dy * sin_h
        samples.append(("active", t_steps, x, y, vx, vy))
        log.write(dump_qpos_line(m, d, 1))

        if along < xy_converged:
            break

        v_des = math.sqrt(2.0 * accel_max * along)
        v_des = min(v_des, v_max)
        wheel_omega = v_des / r
        d.ctrl[LA] = wheel_omega
        d.ctrl[RA] = wheel_omega
        step_tick(m, d, n_sub)
        t_steps += 1

    log.close()
    return samples


# ---------- rotate-in-place: PD (new) ----------
def rotate_pd(scene, target_theta, *, Kp=5.0, Kd=1.0, omega_max=1.0,
              theta_converged=0.005, rate_converged=0.05,
              max_steps=6000, qpos_path: Path):
    """Outer-loop PD on chassis yaw → commanded chassis angular velocity ω*.
    Fed through diff-drive kinematics to wheel velocity actuators.
    Exits when both error and yaw rate are small (system is at rest at target).
    No wait phase.
    """
    m, d = scene["model"], scene["data"]
    n_sub = n_substeps(m)
    b = scene["wheelbase"]
    r = scene["wheel_r"]
    LA, RA = scene["left_act"], scene["right_act"]
    log = qpos_path.open("w")
    samples = []
    t_steps = 0

    while t_steps < max_steps:
        yaw = get_yaw(scene)
        yr = get_yaw_rate(scene)
        err = wrap(target_theta - yaw)
        samples.append(("active", t_steps, math.degrees(yaw), math.degrees(yr)))
        log.write(dump_qpos_line(m, d, 0))

        # Convergence: at target AND at rest
        if abs(err) < theta_converged and abs(yr) < rate_converged:
            break

        # PD law: command desired chassis angular velocity
        omega_cmd = clamp(Kp * err - Kd * yr, -omega_max, +omega_max)
        d.ctrl[LA] = (-omega_cmd * b / 2.0) / r
        d.ctrl[RA] = (+omega_cmd * b / 2.0) / r
        step_tick(m, d, n_sub)
        t_steps += 1

    log.close()
    return samples


# ---------- drive straight: bang-bang (current C++ behavior) ----------
def drive_bb(scene, end_xy, heading, *, linear_speed=0.10, xy_threshold=0.015,
             wait_steps=30, max_steps=6000, qpos_path: Path):
    m, d = scene["model"], scene["data"]
    n_sub = n_substeps(m)
    r = scene["wheel_r"]
    LA, RA = scene["left_act"], scene["right_act"]
    log = qpos_path.open("w")
    samples = []  # (phase, t, x, y, vx_world, vy_world)
    t_steps = 0
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    wheel_omega = linear_speed / r

    while t_steps < max_steps:
        x, y = get_xy(scene)
        vx, vy = get_velocity(scene)
        dx = end_xy[0] - x
        dy = end_xy[1] - y
        dist = math.hypot(dx, dy)
        along = dx * cos_h + dy * sin_h
        samples.append(("active", t_steps, x, y, vx, vy))
        log.write(dump_qpos_line(m, d, 1))
        if dist < xy_threshold or along < 0.0:
            break
        d.ctrl[LA] = wheel_omega
        d.ctrl[RA] = wheel_omega
        step_tick(m, d, n_sub)
        t_steps += 1

    # Wait
    d.ctrl[LA] = 0.0
    d.ctrl[RA] = 0.0
    for _ in range(wait_steps):
        step_tick(m, d, n_sub)
        t_steps += 1
        x, y = get_xy(scene)
        vx, vy = get_velocity(scene)
        samples.append(("wait", t_steps, x, y, vx, vy))
        log.write(dump_qpos_line(m, d, 1))

    log.close()
    return samples


# ---------- drive straight: P controller (new) ----------
def drive_pd(scene, end_xy, heading, *, Kp=5.0, v_max=0.10,
             xy_converged=0.005, rate_converged=0.01,
             max_steps=6000, qpos_path: Path):
    """Outer-loop P on signed distance to endpoint along segment heading.
    Wheel velocity = clamp(Kp · along, 0, v_max) / r.
    Both wheels equal (no curvature commands; pure straight-line).
    Exits when within xy_converged AND chassis nearly at rest.
    """
    m, d = scene["model"], scene["data"]
    n_sub = n_substeps(m)
    r = scene["wheel_r"]
    LA, RA = scene["left_act"], scene["right_act"]
    log = qpos_path.open("w")
    samples = []
    t_steps = 0
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    while t_steps < max_steps:
        x, y = get_xy(scene)
        vx, vy = get_velocity(scene)
        speed = math.hypot(vx, vy)
        dx = end_xy[0] - x
        dy = end_xy[1] - y
        along = dx * cos_h + dy * sin_h
        samples.append(("active", t_steps, x, y, vx, vy))
        log.write(dump_qpos_line(m, d, 1))

        if abs(along) < xy_converged and speed < rate_converged:
            break

        # P law on distance along heading; clamp to [0, v_max] (no reverse)
        v_cmd = clamp(Kp * along, 0.0, v_max)
        wheel_omega = v_cmd / r
        d.ctrl[LA] = wheel_omega
        d.ctrl[RA] = wheel_omega
        step_tick(m, d, n_sub)
        t_steps += 1

    log.close()
    return samples


# ---------- analysis ----------
def deg(r):
    return math.degrees(r)


def analyze_rotation(samples, target_deg):
    """Find the active→wait split (or active end) and report exit & final yaw."""
    # Find first wait sample
    first_wait = None
    for i, s in enumerate(samples):
        if s[0] == "wait":
            first_wait = i
            break
    if first_wait is None:
        # PD: no wait. Exit = last active.
        exit_idx = len(samples) - 1
    else:
        exit_idx = first_wait - 1
    final_idx = len(samples) - 1
    exit_yaw = samples[exit_idx][2]
    final_yaw = samples[final_idx][2]
    rebound = exit_yaw - final_yaw
    return dict(
        ticks=len(samples),
        exit_yaw_deg=exit_yaw,
        final_yaw_deg=final_yaw,
        target_deg=target_deg,
        err_at_exit_deg=target_deg - exit_yaw,
        err_at_final_deg=target_deg - final_yaw,
        rebound_deg=rebound,
    )


def analyze_drive(samples, end_xy):
    final = samples[-1]
    fx, fy = final[2], final[3]
    err = math.hypot(end_xy[0] - fx, end_xy[1] - fy)
    # Find first wait sample (bb only)
    first_wait = None
    for i, s in enumerate(samples):
        if s[0] == "wait":
            first_wait = i
            break
    if first_wait is None:
        exit_idx = len(samples) - 1
    else:
        exit_idx = first_wait - 1
    ex, ey = samples[exit_idx][2], samples[exit_idx][3]
    return dict(
        ticks=len(samples),
        end_xy=end_xy,
        exit_xy=(ex, ey),
        final_xy=(fx, fy),
        final_err_m=err,
        target_dist=math.hypot(end_xy[0], end_xy[1]),
    )


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target_deg", type=float, default=90.0)
    ap.add_argument("--drive_dist", type=float, default=0.30, help="meters")
    ap.add_argument("--Kp_rot", type=float, default=5.0)
    ap.add_argument("--Kd_rot", type=float, default=1.0)
    ap.add_argument("--Kp_drv", type=float, default=5.0)
    ap.add_argument("--out_dir", type=str, default="/tmp/pd_prototype")
    ap.add_argument("--xml", default=None,
                    help="Optional scene XML to load instead of empty car scene")
    ap.add_argument("--car_x", type=float, default=0.0)
    ap.add_argument("--car_y", type=float, default=0.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- ROTATE: bang-bang ----
    print("\n==== rotate bang-bang (current C++ behavior) ====")
    sc = make_scene(xml_override=args.xml, car_xy=(args.car_x, args.car_y))
    settle(sc)
    s_bb = rotate_bb(sc, math.radians(args.target_deg),
                    qpos_path=out / "rotate_bb.qpos")
    r_bb = analyze_rotation(s_bb, args.target_deg)
    for k, v in r_bb.items():
        print(f"  {k} = {v}")

    # ---- ROTATE: PD ----
    print(f"\n==== rotate PD (Kp={args.Kp_rot}, Kd={args.Kd_rot}) ====")
    sc = make_scene(xml_override=args.xml, car_xy=(args.car_x, args.car_y))
    settle(sc)
    s_pd = rotate_pd(sc, math.radians(args.target_deg),
                    Kp=args.Kp_rot, Kd=args.Kd_rot,
                    qpos_path=out / "rotate_pd.qpos")
    r_pd = analyze_rotation(s_pd, args.target_deg)
    for k, v in r_pd.items():
        print(f"  {k} = {v}")

    # ---- DRIVE: bang-bang ----
    print("\n==== drive bang-bang (current C++ behavior) ====")
    sc = make_scene(xml_override=args.xml, car_xy=(args.car_x, args.car_y))
    settle(sc)
    end = (args.drive_dist, 0.0)
    heading = 0.0
    s_dbb = drive_bb(sc, end, heading,
                    qpos_path=out / "drive_bb.qpos")
    r_dbb = analyze_drive(s_dbb, end)
    for k, v in r_dbb.items():
        print(f"  {k} = {v}")

    # ---- DRIVE: P ----
    print(f"\n==== drive P (Kp={args.Kp_drv}) ====")
    sc = make_scene(xml_override=args.xml, car_xy=(args.car_x, args.car_y))
    settle(sc)
    s_dpd = drive_pd(sc, end, heading,
                    Kp=args.Kp_drv,
                    qpos_path=out / "drive_pd.qpos")
    r_dpd = analyze_drive(s_dpd, end)
    for k, v in r_dpd.items():
        print(f"  {k} = {v}")

    # ---- ROTATE: TRAPEZOIDAL ----
    print(f"\n==== rotate trapezoidal (alpha_max=5.0) ====")
    sc = make_scene(xml_override=args.xml, car_xy=(args.car_x, args.car_y))
    settle(sc)
    s_tr = rotate_trap(sc, math.radians(args.target_deg),
                       qpos_path=out / "rotate_trap.qpos")
    r_tr = analyze_rotation(s_tr, args.target_deg)
    for k, v in r_tr.items():
        print(f"  {k} = {v}")

    # ---- DRIVE: TRAPEZOIDAL ----
    print(f"\n==== drive trapezoidal (accel_max=0.5) ====")
    sc = make_scene(xml_override=args.xml, car_xy=(args.car_x, args.car_y))
    settle(sc)
    s_dtr = drive_trap(sc, end, heading,
                       qpos_path=out / "drive_trap.qpos")
    r_dtr = analyze_drive(s_dtr, end)
    for k, v in r_dtr.items():
        print(f"  {k} = {v}")

    # Also save the empty car scene XML path for rendering
    params = default_parameters()
    g = generate_all(PROJECT_ROOT / "assets", params=params)
    print(f"\nScene XML: {g['mjcf_scene']}")
    print(f"qpos dumps in: {out}")


if __name__ == "__main__":
    main()
