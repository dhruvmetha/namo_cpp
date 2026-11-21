#!/usr/bin/env python3
"""How close are the v1 and v2 cars to the real microMVP? Replay the logged commands, score against the camera.

Every free-driving approach in real_drive_fit/approach_timeseries.npz (robot_control push_calibration,
33 trials) is replayed open loop from rest on an open floor, commands held as sent, and the sim's
forward speed and yaw rate are compared with the camera on the free frames. Scoring is
fit_drivetrain.lag_score: the same latency grid and the same frames used to pick the v2 form.

Cars:
  v1                 the pool scene as is. Its wheel command is a speed, so PWM goes through the v1
                     calibration (PWM 0.4 <-> 0.05357 m/s of tread speed).
  v1_no_trim         v1 without the firmware's 0.87 right trim, in case v1's calibration already
                     absorbed it.
  v1_logged_cruise   diagnostic, not a car anyone uses: v1's servo with the speed map set to the
                     logged cruise (delivered mean 0.36 PWM <-> 3.66 cm/s), to split v1's error into
                     its speed calibration and the rest.
  v2                 convert_v1_to_v2 with the v2 switches (fitted drivetrain, ctrl_units pwm).
                     ctrl = delivered PWM. Should reproduce drivetrain_fit.json's coulomb score.
Delivered PWM = commanded with the right side x 0.87 (firmware trim), except v1_no_trim.

  python replay_vs_camera.py --xml <v1 pool scene> --fit-dir <real_drive_fit> --out <dir> --procs 8
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "car_motion_probe"))
import car_v2_block as V        # noqa: E402
import convert_v1_to_v2 as C    # noqa: E402
import car_probe_lib as L       # noqa: E402
from fit_drivetrain import TARGETS, TRIM, lag_score   # noqa: E402

V1_PER_PWM = V.CRUISE_MS / V.CRUISE_PWM         # m/s of tread per unit PWM, v1 calibration
LOGGED_PER_PWM = 0.0366 / 0.36                  # forward target: delivered (0.38, 0.34) -> 3.66 cm/s
CARS = {  # name: (scene, m/s per PWM or None for a PWM motor, right trim)
    "v1": ("v1", V1_PER_PWM, TRIM),
    "v1_no_trim": ("v1", V1_PER_PWM, 1.0),
    "v1_logged_cruise": ("v1", LOGGED_PER_PWM, TRIM),
    "v2": ("v2", None, TRIM),
}
XMLS = {}


def _init(xmls):
    XMLS.update(xmls)


def setup(name):
    scene, per_pwm, trim = CARS[name]
    m = mujoco.MjModel.from_xml_string(XMLS[scene])
    car = L.Car(m)
    d = mujoco.MjData(m)
    L.settle(m, d, car, 500, zero_first=False)
    L.settle(m, d, car, 100)
    gain = 1.0 if per_pwm is None else per_pwm / car.wheel_r
    return m, d, car, gain, trim


def replay(args):
    name, key, t_s, lc, rc = args
    m, d, car, gain, trim = setup(name)
    n = int(t_s[-1] / L.DT) + 1

    def ctrl(i):
        k = max(np.searchsorted(t_s, i * L.DT, side="right") - 1, 0)
        return gain * lc[k], gain * trim * rc[k]

    s = L.run(m, d, car, ctrl, n)
    return name, key, 100 * s["v_fwd"], np.degrees(s["yaw_rate"])


def steady(args):
    """One of fit_drivetrain's four steady targets; its (L, R) are already delivered PWM with trim."""
    name, (tname, (ul, ur), meas, kind) = args
    m, d, car, gain, trim = setup(name)
    ur = ur / TRIM * trim                       # v1_no_trim: undo the trim
    s = L.run(m, d, car, lambda i: (gain * ul, gain * ur), 1000)       # 2 s
    late = slice(500, None)
    val = 100 * float(np.mean(s["v_fwd"][late])) if kind == "v" else float(np.degrees(np.mean(np.abs(s["yaw_rate"][late]))))
    return name, tname, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    v2 = out / "v2_scene.xml"
    v2.write_text(ET.tostring(C.convert_root(ET.parse(args.xml).getroot(), **V.V2), encoding="unicode"))
    xmls = {"v1": L.open_floor_xml(args.xml, out / "v1_open_floor.xml").read_text(),
            "v2": L.open_floor_xml(str(v2), out / "v2_open_floor.xml").read_text()}

    Z = np.load(Path(args.fit_dir) / "approach_timeseries.npz")
    keys = sorted({k.split("__")[0] for k in Z.files})
    series = {k: {f.split("__", 1)[1]: Z[f] for f in Z.files if f.startswith(k + "__")} for k in keys}
    jobs = [(c, k, series[k]["t_s"], series[k]["left_cmd"], series[k]["right_cmd"]) for c in CARS for k in keys]
    with Pool(args.procs, initializer=_init, initargs=(xmls,)) as pool:
        res = pool.map(replay, jobs)
        st = pool.map(steady, [(c, t) for c in CARS for t in TARGETS])
    _init(xmls)
    report = {"approaches": len(keys), "cars": {}}
    for c in CARS:
        best, rows = lag_score([(k, v, w) for name, k, v, w in res if name == c], series)
        report["cars"][c] = dict(best=best, by_lag=rows,
                                 steady={t: val for name, t, val in st if name == c})
    report["steady_measured"] = {t: meas for t, _, meas, _ in TARGETS}
    (out / "replay_vs_camera.json").write_text(json.dumps(report, indent=1))
    print(f"{'car':<18} {'lag_s':>5} {'frames':>6} {'v_rmse':>7} {'v_bias':>7} {'yaw_rmse':>8} {'yaw_bias':>8}  "
          + " ".join(f"{t:>8}" for t, *_ in TARGETS))
    print(f"{'camera':<18} {'':>5} {'':>6} {'':>7} {'':>7} {'':>8} {'':>8}  "
          + " ".join(f"{meas:>8.2f}" for _, _, meas, _ in TARGETS))
    for c, r in report["cars"].items():
        b = r["best"]
        print(f"{c:<18} {b['lag_s']:>5.2f} {b['frames']:>6d} {b['v_rmse_cm_s']:>7.3f} {b['v_bias_cm_s']:>7.3f} "
              f"{b['yaw_rmse_deg_s']:>8.2f} {b['yaw_bias_deg_s']:>8.2f}  "
              + " ".join(f"{r['steady'][t]:>8.2f}" for t, *_ in TARGETS))


if __name__ == "__main__":
    main()
