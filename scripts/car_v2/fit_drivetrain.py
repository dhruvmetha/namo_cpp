#!/usr/bin/env python3
"""Fit the v2 drivetrain to the real car's free driving, under three forms of the low-PWM loss.

  (a) coulomb   gearbox friction on the wheel joints: frictionloss = thr * STALL_TORQUE
  (b) deadzone  command remap u -> 0 when |u| < thr, else u unchanged
  (c) affine    command remap u -> sign(u) * (|u| - thr) / (1 - thr) when |u| > thr, else 0

Each form has the same free parameters: NO_LOAD_SPEED (the speed scale), thr, and the
wheel track (only the in-place rotations constrain it). STALL_TORQUE stays at the
datasheet value; the logs cannot see it (no stalls, 15 Hz camera).

Targets are the robust segment means from real_drive_fit/ (robot_control push_calibration,
free driving only, delivered PWM = commanded with the firmware's 0.87 right trim):
  forward   delivered (0.38, 0.34)      3.66 cm/s
  reverse   delivered (-0.14, -0.14)   -0.93 cm/s
  rotate    commanded +-0.15 (delivered 0.15 / -0.1305)   20.8 deg/s
  rotate    commanded +-0.25 (delivered 0.25 / -0.2175)   53.4 deg/s
Stage 1 grids the parameters against those four numbers on the v2 car (published body,
one frictionless caster, no wheel damping, PWM motor). Stage 2 replays every logged
approach (commands held as sent, open loop from rest) with each form's best parameters
and scores forward speed and yaw rate against the camera on the free frames.

  python fit_drivetrain.py --xml <v1 pool scene> --fit-dir <real_drive_fit> --out <dir> --procs 16
"""
import argparse
import itertools
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

TRIM = 0.87
TARGETS = [  # name, delivered (L, R), measured, kind
    ("forward", (0.38, 0.34), 3.66, "v"),
    ("reverse", (-0.14, -0.14), -0.93, "v"),
    ("rot015", (0.15, -0.15 * TRIM), 20.8, "w"),
    ("rot025", (0.25, -0.25 * TRIM), 53.4, "w"),
]
FORMS = ("coulomb", "deadzone", "affine")
XML = None


def remap(u, form, thr):
    a = abs(u)
    if form == "coulomb" or thr <= 0:
        return u
    if a < thr:
        return 0.0
    return u if form == "deadzone" else np.sign(u) * (a - thr) / (1 - thr)


def model(form, w0, thr, track):
    m = mujoco.MjModel.from_xml_string(XML)
    car = L.Car(m)
    for k in "LR":
        dof = car.wheel[k]["dof"]
        m.dof_frictionloss[dof] = thr * V.STALL_TORQUE if form == "coulomb" else 0.0
        m.body_pos[car.wheel[k]["body"]][1] = (track / 2) * (1 if k == "L" else -1)
    for k in "LR":
        a = car.wheel[k]["act"]
        m.actuator_biasprm[a, 2] = -V.STALL_TORQUE / w0
    car = L.Car(m)
    return m, car


def steady(args):
    form, w0, thr, track, target = args
    name, (ul, ur), meas, kind = target
    m, car = model(form, w0, thr, track)
    d = mujoco.MjData(m)
    L.settle(m, d, car, 500, zero_first=False)
    L.settle(m, d, car, 100)
    s = L.run(m, d, car, lambda i: (remap(ul, form, thr), remap(ur, form, thr)), 1000)    # 2 s
    late = slice(500, None)
    val = 100 * float(np.mean(s["v_fwd"][late])) if kind == "v" else float(np.degrees(np.mean(np.abs(s["yaw_rate"][late]))))
    return (form, w0, thr, track, name), val


def replay(args):
    form, w0, thr, track, key, t_s, lc, rc = args
    m, car = model(form, w0, thr, track)
    d = mujoco.MjData(m)
    L.settle(m, d, car, 500, zero_first=False)
    L.settle(m, d, car, 100)
    n = int(t_s[-1] / L.DT) + 1
    dl, dr = lc, rc * TRIM

    def ctrl(i):
        k = max(np.searchsorted(t_s, i * L.DT, side="right") - 1, 0)
        return remap(dl[k], form, thr), remap(dr[k], form, thr)

    s = L.run(m, d, car, ctrl, n)
    return key, 100 * s["v_fwd"], np.degrees(s["yaw_rate"])


def main():
    global XML
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = C.convert_root(ET.parse(args.xml).getroot(), **V.V2)          # ctrl_units pwm
    tmp = out / "v2_scene.xml"
    tmp.write_text(ET.tostring(root, encoding="unicode"))
    XML = L.open_floor_xml(str(tmp), out / "v2_open_floor.xml").read_text()

    # ---- stage 1: steady targets ----
    W0 = np.round(np.arange(6.0, 16.01, 0.5), 2)
    THR = {"coulomb": [0.0, 0.04, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12],
           "deadzone": [0.0, 0.04, 0.08, 0.10, 0.12],
           "affine": [0.0, 0.04, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12]}
    jobs = [(f, w, th, 0.053, t) for f in FORMS for w in W0 for th in THR[f] for t in TARGETS if t[3] == "v"]
    with Pool(args.procs, initializer=_init, initargs=(XML,)) as pool:
        lin = dict(pool.map(steady, jobs))
        best = {}
        for f in FORMS:
            scores = []
            for w, th in itertools.product(W0, THR[f]):
                e = [(lin[(f, w, th, 0.053, n)] - meas) / abs(meas) for n, _, meas, k in TARGETS if k == "v"]
                scores.append((float(np.sqrt(np.mean(np.square(e)))), w, th))
            scores.sort()
            best[f] = scores[0]
        TRK = [0.036, 0.040, 0.044, 0.048, 0.053]
        jobs = [(f, best[f][1], best[f][2], tr, t) for f in FORMS for tr in TRK for t in TARGETS if t[3] == "w"]
        rot = dict(pool.map(steady, jobs))
        fit = {}
        for f in FORMS:
            _, w, th = best[f]
            sc = []
            for tr in TRK:
                e = [(rot[(f, w, th, tr, n)] - meas) / abs(meas) for n, _, meas, k in TARGETS if k == "w"]
                sc.append((float(np.sqrt(np.mean(np.square(e)))), tr))
            sc.sort()
            tr = sc[0][1]
            pred = {n: (lin.get((f, w, th, 0.053, n)) if k == "v" else rot[(f, w, th, tr, n)]) for n, _, _, k in TARGETS}
            fit[f] = dict(no_load_speed=float(w), thr=float(th), track=float(tr),
                          translation_rel_rms=best[f][0], rotation_rel_rms=sc[0][0],
                          predicted=pred, measured={n: meas for n, _, meas, _ in TARGETS})
        print(json.dumps(fit, indent=1), flush=True)

        # ---- stage 2: replay every logged approach ----
        Z = np.load(Path(args.fit_dir) / "approach_timeseries.npz")
        keys = sorted({k.split("__")[0] for k in Z.files})
        series = {k: {f.split("__", 1)[1]: Z[f] for f in Z.files if f.startswith(k + "__")} for k in keys}
        jobs = [(f, fit[f]["no_load_speed"], fit[f]["thr"], fit[f]["track"], k,
                 series[k]["t_s"], series[k]["left_cmd"], series[k]["right_cmd"]) for f in FORMS for k in keys]
        res = pool.map(replay, jobs)
    score = {f: lag_score([(key, vs, ws) for (key, vs, ws), job in zip(res, jobs) if job[0] == f], series)[0]
             for f in FORMS}
    (out / "drivetrain_fit.json").write_text(json.dumps({"steady_fit": fit, "replay": score}, indent=1))
    print(json.dumps(score, indent=1))


LAGS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)


def lag_score(replays, series, lags=LAGS):
    """Score sim replays against the camera on the free frames.

    replays: [(approach key, sim forward speed cm/s per tick, sim yaw rate deg/s per tick)].
    The camera sees motion some fixed time after the command, so each model is scored at the
    latency on `lags` that gives it the lowest speed RMSE. Returns (best, every lag)."""
    rows = []
    for lag in lags:
        ev, ew = [], []
        for key, vs, ws in replays:
            S = series[key]
            t = np.arange(len(vs)) * L.DT
            v = np.interp(S["t_s"] - lag, t, vs, left=0.0)
            w = np.interp(S["t_s"] - lag, t, ws, left=0.0)
            mask = S["free_mask"].astype(bool) & np.isfinite(S["v_fwd_cm_s_5pt"]) & np.isfinite(S["yaw_deg_s_framediff"])
            ev.append(v[mask] - S["v_fwd_cm_s_5pt"][mask])
            ew.append(w[mask] - S["yaw_deg_s_framediff"][mask])
        ev, ew = np.concatenate(ev), np.concatenate(ew)
        rows.append(dict(lag_s=lag, frames=int(len(ev)), v_rmse_cm_s=float(np.sqrt(np.mean(ev ** 2))),
                         v_bias_cm_s=float(ev.mean()), yaw_rmse_deg_s=float(np.sqrt(np.mean(ew ** 2))),
                         yaw_bias_deg_s=float(ew.mean())))
    return min(rows, key=lambda r: r["v_rmse_cm_s"]), rows


def _init(xml):
    global XML
    XML = xml


if __name__ == "__main__":
    main()
