#!/usr/bin/env python3
"""Does a 30 Hz command hold make the car lunge and hop?

Replays the real robot's own wheel commands from free-driving approaches
(robot_control push_calibration/.../wheel_commands.jsonl, state APPROACHING, before any
block contact) into a car on an open floor, two ways:
  hold30  each command held until the next one, as the hardware sends them (~30 Hz,
          latched on the logged send time)
  tick    the same commands linearly interpolated every 2 ms tick
Open loop: the point is how the car body responds to the command steps, not where it
ends up. PWM maps to the sim wheel command the way the v1 calibration does
(PWM 0.4 <-> 0.05357 m/s of tread speed); a v2 scene built with ctrl_units="omega"
turns that back into the same PWM.

  python probe_hold.py --scene <variant scene.xml> --logs <wheel_commands.jsonl>... --out <dir>
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import car_probe_lib as L

CRUISE_PWM = 0.4


def approach(path):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["state"] == "APPROACHING"]
    t, keep = [], {}
    for r in rows:
        keep[r["t_s"]] = (r["left_cmd"], r["right_cmd"])       # last command sent at that time
    ts = np.array(sorted(keep))
    cmd = np.array([keep[x] for x in ts])
    return ts - ts[0], cmd


def run(scene, ts, cmd, mode):
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    car = L.Car(m)
    L.settle(m, d, car, ticks=500, zero_first=False)
    L.settle(m, d, car, ticks=100)
    to_ctrl = (L.CRUISE_MS / CRUISE_PWM) / car.wheel_r            # PWM -> wheel rad/s command
    n = int(ts[-1] / L.DT)

    def ctrl(i):
        t = i * L.DT
        if mode == "hold30":
            k = np.searchsorted(ts, t, side="right") - 1
            u = cmd[max(k, 0)]
        else:
            u = np.array([np.interp(t, ts, cmd[:, 0]), np.interp(t, ts, cmd[:, 1])])
        return u[0] * to_ctrl, u[1] * to_ctrl

    s = L.run(m, d, car, ctrl, n)
    t = np.arange(n) * L.DT
    j = np.abs(L.jerk(s["a_fwd"]))
    off = sum(s[f"n_{g}"] for g in car.supports) == 0
    v = s["v_fwd"]
    w = int(0.1 / L.DT)                                             # 100 ms moving average
    smooth = np.convolve(v, np.ones(w) / w, mode="same")
    late = t >= 0.3                                                 # after the start from rest
    return dict(
        jerk_max=float(j.max()), jerk_max_after_300ms=float(j[late[1:]].max()),
        jerk_rms_after_300ms=float(np.sqrt(np.mean(j[late[1:]] ** 2))),
        airborne_ticks=int(off.sum()), airborne_ticks_after_300ms=int(off[late].sum()),
        speed_ripple_sd_mms=1000 * float(np.std((v - smooth)[late][w:-w])),
        pitch_max_deg=float(np.abs(s["pitch"]).max()),
        mean_speed_cms=100 * float(v[late].mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, nargs="+", help="name=path pairs")
    ap.add_argument("--logs", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = []
    for spec in args.scene:
        name, path = spec.split("=", 1)
        floor = L.open_floor_xml(path, Path(args.out) / "xml" / f"{name}_open_floor.xml")
        for log in args.logs:
            ts, cmd = approach(log)
            for mode in ("hold30", "tick"):
                r = dict(variant=name, log=str(Path(log).parent), mode=mode, seconds=float(ts[-1]),
                         commands=len(ts), **run(floor, ts, cmd, mode))
                res.append(r)
                print(json.dumps(r), flush=True)
    Path(args.out, "hold_metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
