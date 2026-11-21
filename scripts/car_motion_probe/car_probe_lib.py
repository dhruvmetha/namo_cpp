"""Per-tick measurement helpers for the v1 diff-drive car.

Read-only probe: nothing here edits a scene, a config or the C++ code. The probes load
a pool scene with the pip `mujoco` package, which steps the same model the C++ side
steps. The C++ wheel command is `d->ctrl[actuator] = omega` and its step is a bare
`mj_step` (diff_drive_adapter.cpp:138, mujoco_wrapper.cpp:66), so driving `ctrl` here
reproduces what the push controller does tick for tick. pip mujoco is 3.2.7 while the
C++ engine on CS is 3.2.8; probe_push.py checks that gap by replaying a C++ push.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

DT = 0.002
CRUISE_MS = 0.05357               # skill.push_tracker_max_speed x kCarWheelMaxSpeedMs (1.0)

WHEELS = {"L": ("left_wheel", "left_wheel_joint", "left_wheel_collision", "left_wheel_drive"),
          "R": ("right_wheel", "right_wheel_joint", "right_wheel_collision", "right_wheel_drive")}
CASTERS = {"F": ("front_support_body", "front_support"),
           "B": ("rear_support_body", "rear_support")}
CHASSIS_GEOMS = ("front_chassis_collision", "rear_chassis_collision")


def open_floor_xml(pool_xml: str, out_path: Path) -> Path:
    """The pool scene with its walls and movable blocks removed. Car, floor, options and
    actuators stay byte-identical, so the car drives on the same physics with room to move."""
    root = ET.parse(pool_xml).getroot()
    world = root.find("worldbody")
    for body in list(world.findall("body")):
        name = body.get("name", "")
        if name == "walls":
            for child in list(body):
                body.remove(child)
        elif name.endswith("_movable"):
            world.remove(body)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="unicode")
    return out_path


class Car:
    """Name lookups for one loaded model."""

    def __init__(self, m: mujoco.MjModel, block_geoms=()):
        n = lambda t, s: mujoco.mj_name2id(m, t, s)
        B, J, G, A = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT,
                      mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_ACTUATOR)
        self.m = m
        self.car = n(B, "car")
        fj = n(J, "car_freejoint")
        self.qadr, self.vadr = m.jnt_qposadr[fj], m.jnt_dofadr[fj]
        self.floor = n(G, "floor")
        self.wheel = {k: dict(body=n(B, b), dof=m.jnt_dofadr[n(J, j)], qpos=m.jnt_qposadr[n(J, j)],
                              geom=n(G, g), act=n(A, a)) for k, (b, j, g, a) in WHEELS.items()}
        self.caster = {k: dict(body=n(B, b), geom=n(G, g)) for k, (b, g) in CASTERS.items() if n(B, b) >= 0}
        # geometry read from the model, so v1 and v2 cars are measured by the same code
        self.wheel_r = float(m.geom_size[self.wheel["L"]["geom"]][0])
        self.track = float(abs(m.body_pos[self.wheel["L"]["body"]][1] - m.body_pos[self.wheel["R"]["body"]][1]))
        self.cruise_omega = CRUISE_MS / self.wheel_r     # what the v1 controller sends at cruise
        self.supports = [f"w{k}" for k in self.wheel] + [f"c{k}" for k in self.caster]
        self.chassis = {n(G, g) for g in CHASSIS_GEOMS}
        self.block = {n(G, g): g for g in block_geoms}
        self.robot_geoms = self.chassis | {w["geom"] for w in self.wheel.values()} | \
            {c["geom"] for c in self.caster.values()}


def settle(m, d, car: Car, ticks=250, zero_first=True):
    """What the push controller does before a push: zero every velocity, wheels at 0, step."""
    if zero_first:
        d.qvel[:] = 0
    d.ctrl[:] = 0
    for _ in range(ticks):
        mujoco.mj_step(m, d)


def _body_vel(m, d, body):
    """(angular, linear) world-frame velocity of a body at its centre of mass."""
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, res, 0)
    return res[:3], res[3:]


def record(m, d, car: Car):
    """One tick's signals, read after mj_step.

    mj_step leaves contacts, cvel and qacc describing the state at the START of the tick
    while qpos/qvel are the state at its end, so reading both mixes two ticks. mj_forward
    recomputes the derived quantities for the current state. The extra call is not
    bit-neutral: probe_drive.py --check-identity measured at most 2e-8 m of qpos
    difference against a run without it, far below anything these probes report.
    """
    mujoco.mj_forward(m, d)
    R = d.xmat[car.car].reshape(3, 3)
    head, left, up = R[:, 0], R[:, 1], R[:, 2]
    v = d.qvel[car.vadr:car.vadr + 3]                 # body-origin velocity, world frame
    w = R @ d.qvel[car.vadr + 3:car.vadr + 6]          # local -> world
    a = d.qacc[car.vadr:car.vadr + 3]
    hxy = np.array([head[0], head[1], 0.0]); hxy /= np.linalg.norm(hxy)
    lxy = np.array([-hxy[1], hxy[0], 0.0])
    out = dict(
        t=d.time, x=d.xpos[car.car][0], y=d.xpos[car.car][1], z=d.xpos[car.car][2],
        yaw=np.arctan2(head[1], head[0]),
        pitch=np.degrees(np.arcsin(np.clip(-head[2], -1, 1))),   # nose down positive
        roll=np.degrees(np.arcsin(np.clip(left[2], -1, 1))),
        v_fwd=v @ hxy, v_lat=v @ lxy, yaw_rate=w[2], a_fwd=a @ hxy,
    )
    # wheels: command, joint speed, torque, slip
    for k, wh in car.wheel.items():
        omega = d.qvel[wh["dof"]]
        wang, wlin = _body_vel(m, d, wh["body"])
        v_c = wlin @ hxy
        rw = car.wheel_r * omega
        out[f"ctrl_{k}"] = d.ctrl[wh["act"]]
        out[f"omega_{k}"] = omega
        out[f"tau_{k}"] = d.actuator_force[wh["act"]]
        out[f"slip_{k}"] = (rw - v_c) / max(abs(rw), abs(v_c), 1e-4)
        out[f"vctr_{k}"] = v_c
    # contacts with the floor and with blocks
    groups = {f"w{k}": {wh["geom"]} for k, wh in car.wheel.items()}
    groups.update({f"c{k}": {c["geom"]} for k, c in car.caster.items()})   # only casters the model has
    groups["chassis"] = car.chassis
    for g in groups:
        out[f"n_{g}"] = 0; out[f"fn_{g}"] = 0.0; out[f"ft_{g}"] = 0.0; out[f"mu_{g}"] = np.nan
    out["n_block"] = 0; out["fn_block"] = 0.0; out["mu_block"] = np.nan
    out["n_block_floor"] = 0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        pair = {g1, g2}
        mujoco.mj_contactForce(m, d, i, f6)
        fn, ft = f6[0], float(np.hypot(f6[1], f6[2]))
        if car.floor in pair:
            other = (pair - {car.floor}).pop()
            for g, geoms in groups.items():
                if other in geoms:
                    out[f"n_{g}"] += 1; out[f"fn_{g}"] += fn; out[f"ft_{g}"] += ft
                    out[f"mu_{g}"] = c.friction[0]
                    if g.startswith("c"):
                        key = g[1]
                        body = car.caster[key]["body"]
                        cang, clin = _body_vel(m, d, body)
                        slip = clin + np.cross(cang, c.pos - d.xipos[body])
                        out[f"cslip_{key}"] = float(np.hypot(slip[0], slip[1]))
                        out[f"cspeed_{key}"] = float(np.hypot(clin[0], clin[1]))
            if other in car.block:
                out["n_block_floor"] += 1
        elif pair & car.robot_geoms and pair & set(car.block):
            out["n_block"] += 1; out["fn_block"] += fn; out["mu_block"] = c.friction[0]
    return out


def run(m, d, car: Car, ctrl_fn, ticks, qpos_log=None):
    """Step `ticks` times with ctrl from ctrl_fn(tick) -> (omega_L, omega_R)."""
    rows = []
    for i in range(ticks):
        wl, wr = ctrl_fn(i)
        d.ctrl[car.wheel["L"]["act"]] = wl
        d.ctrl[car.wheel["R"]["act"]] = wr
        mujoco.mj_step(m, d)
        rows.append(record(m, d, car))
        if qpos_log is not None:
            qpos_log.append(d.qpos.copy())
    keys = set(rows[0]) | {f"{a}_{k}" for a in ("cslip", "cspeed") for k in car.caster}
    return {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}


def jerk(a):
    return np.diff(a) / DT
