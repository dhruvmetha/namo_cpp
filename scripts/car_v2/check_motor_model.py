#!/usr/bin/env python3
"""Check that the v2 `general` actuator behaves as a PWM DC motor in pip mujoco.

The car from a converted scene is fixed 10 cm above the floor (freejoint removed), so the
wheels spin free and only the motor, the wheel inertia and the gearbox frictionloss act.

1. Torque law: actuator_force == clip(STALL*clip(u,-1,1) - STALL/NO_LOAD*omega, +-STALL) over a grid.
2. ctrl_units="omega" gives the same torque for ctrl = u / (CRUISE_PWM / cruise_omega), where
   cruise_omega = CRUISE_MS / wheel_radius is what the v1 controller sends at cruise.
3. Free wheel with gearbox friction: steady speed NO_LOAD*(u - FRICTION) above the friction
   threshold, still below it.

  python check_motor_model.py --xml <v1 pool scene>
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import car_v2_block as V        # noqa: E402
import convert_v1_to_v2 as C    # noqa: E402


def lifted(scene, **opts):
    root = C.convert_root(ET.parse(scene).getroot(), **(V.V2 | opts))
    car = root.find("worldbody/body[@name='car']")
    car.remove(car.find("freejoint"))
    # Without its freejoint the car is welded to the world, and MuJoCo does not filter
    # contacts with world-welded geoms, so v2's wheels (inside the 7 x 7 chassis box at a
    # 4 cm track) would collide with the chassis here. In a scene the freejoint stays and
    # the parent-child filter removes those pairs. Switch the chassis off for this rig only.
    for g in car.findall("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")
    x, y, _ = car.get("pos").split()
    car.set("pos", f"{x} {y} 0.1")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return m, mujoco.MjData(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    args = ap.parse_args()
    res = {"mujoco": mujoco.__version__}
    S, W0, D = V.STALL_TORQUE, V.NO_LOAD_SPEED, V.GEARBOX_FRICTION_PWM

    m, d = lifted(args.xml)
    mo, do = lifted(args.xml, ctrl_units="omega")
    jl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
    dof = m.jnt_dofadr[jl]
    cruise_omega = V.CRUISE_MS / V.BODY[V.V2["body"]]["wheel_radius"]
    err, err_units = 0.0, 0.0
    for u in (-1.5, -1.0, -0.4, 0.0, 0.29, 0.4, 1.0, 1.5):
        for w in (-20.0, 0.0, 3.5, 13.84, 34.6, 45.0):
            for mm, dd, c in ((m, d, u), (mo, do, u / (V.CRUISE_PWM / cruise_omega))):
                dd.qvel[:] = 0
                dd.qvel[dof] = w
                dd.ctrl[:] = c
                mujoco.mj_forward(mm, dd)
                want = np.clip(S * np.clip(u, -1, 1) - S / W0 * w, -S, S)
                e = abs(dd.actuator_force[0] - want)
                if mm is m:
                    err = max(err, e)
                else:
                    err_units = max(err_units, e)
    res["torque_law_max_abs_err_Nm"] = err
    res["omega_units_max_abs_err_Nm"] = err_units

    spin = {}
    for u in (0.03, 0.05, 0.07, 0.15, 0.4, 1.0):
        mujoco.mj_resetData(m, d)
        d.ctrl[:] = u
        speeds = []
        for i in range(1500):                              # 3 s
            mujoco.mj_step(m, d)
            speeds.append(d.qvel[dof])
        speeds = np.array(speeds)
        ss = float(speeds[-250:].mean())
        t63 = next((i for i, s in enumerate(speeds) if s >= 0.632 * ss), None) if ss > 1e-3 else None
        spin[str(u)] = dict(steady_rad_s=ss, expected=max(0.0, W0 * (u - D)),
                            t63_ms=None if t63 is None else 2.0 * (t63 + 1))
    res["free_wheel"] = spin
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
