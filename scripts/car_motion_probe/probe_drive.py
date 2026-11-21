#!/usr/bin/env python3
"""v1 car on an open floor: straight drive, spin in place, arc, stop, contact friction.

The car, floor and options come from a pool scene unchanged; only walls and blocks are
removed (car_probe_lib.open_floor_xml) so nothing gets in the way. Wheel commands are
step inputs, which is what the push controller sends: PushPathFollower saturates at
push_tracker_max_speed on a straight push and the wheel actuator receives it at once.

  python probe_drive.py --xml <pool scene> --out <dir>
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import car_probe_lib as L


def steady(x, t, t0):
    sel = t >= t0
    return float(np.nanmean(x[sel])), float(np.nanstd(x[sel]))


def contact_summary(s, t0=0.5):
    sel = s["t"] >= t0
    out = {}
    for g in ("wL", "wR", "cF", "cB", "chassis"):
        n = s[f"n_{g}"][sel]
        out[g] = dict(no_contact_frac=float(np.mean(n == 0)), fn_mean=float(np.mean(s[f"fn_{g}"][sel])),
                      fn_min=float(np.min(s[f"fn_{g}"][sel])), fn_max=float(np.max(s[f"fn_{g}"][sel])),
                      mu=float(np.nanmax(s[f"mu_{g}"])) if np.any(np.isfinite(s[f"mu_{g}"])) else None)
    return out


def airborne(s, z_ref):
    """Ticks with no wheel or caster touching the floor, and the peak hop height."""
    off = (s["n_wL"] + s["n_wR"] + s["n_cF"] + s["n_cB"]) == 0
    return dict(airborne_ticks=int(off.sum()), airborne_ms=1000 * L.DT * int(off.sum()),
                hop_mm=1000 * float(max(0.0, s["z"].max() - z_ref)))


def settle_time_ms(v, target, tol=0.05):
    """First time after which speed stays within tol of target."""
    bad = np.where(np.abs(v - target) > tol * abs(target))[0]
    return 1000 * L.DT * (int(bad[-1]) + 2) if len(bad) else 0.0


def loads(s, t, t0):
    sel = t >= t0
    return {g: dict(fn=float(np.mean(s[f"fn_{g}"][sel])),
                    ft_over_fn=float(np.mean(s[f"ft_{g}"][sel] / np.maximum(s[f"fn_{g}"][sel], 1e-9))))
            for g in ("wL", "wR", "cF", "cB")}


def straight(m, d, car, qlog):
    s = L.run(m, d, car, lambda i: (L.CRUISE_OMEGA, L.CRUISE_OMEGA), 1500, qlog)   # 3 s
    t, v = s["t"] - s["t"][0] + L.DT, s["v_fwd"]
    v_ss, v_sd = steady(v, t, 1.5)
    i10 = int(np.argmax(v >= 0.1 * v_ss)); i90 = int(np.argmax(v >= 0.9 * v_ss))
    j = L.jerk(s["a_fwd"])
    start = t < 0.3
    return s, dict(
        cmd_speed_cms=100 * L.CRUISE_MS, steady_speed_cms=100 * v_ss, speed_ripple_sd_cms=100 * v_sd,
        speed_min_max_after_1s_cms=[100 * float(v[t >= 1].min()), 100 * float(v[t >= 1].max())],
        rise_10_90_ms=1000 * (t[i90] - t[i10]), t_to_90_ms=1000 * t[i90],
        overshoot_pct=100 * (float(v.max()) - v_ss) / v_ss,
        slip_start_peak={k: float(np.max(np.abs(s[f"slip_{k}"][start]))) for k in "LR"},
        slip_steady={k: steady(s[f"slip_{k}"], t, 1.5)[0] for k in "LR"},
        torque_saturated_frac_start={k: float(np.mean(np.abs(s[f"tau_{k}"][start]) >= 0.499)) for k in "LR"},
        pitch_peak_deg=float(np.max(np.abs(s["pitch"]))), pitch_steady_deg=steady(s["pitch"], t, 1.5),
        jerk_rms=float(np.sqrt(np.mean(j ** 2))), jerk_max=float(np.max(np.abs(j))),
        jerk_rms_after_1s=float(np.sqrt(np.mean(j[t[1:] >= 1] ** 2))),
        contacts=contact_summary(s),
        settle_to_5pct_ms=settle_time_ms(v, v_ss),
        **airborne(s, float(np.median(s["z"][t >= 1.5]))),
        caster_slip_over_speed_steady={k: float(np.nanmedian((s[f"cslip_{k}"] / s[f"cspeed_{k}"])[t >= 1.5])) for k in "FB"},
        steady_loads=loads(s, t, 1.5),
        wheel_torque_steady_Nm={k: steady(s[f"tau_{k}"], t, 1.5)[0] for k in "LR"},
        wheel_omega_steady=steady(s["omega_L"], t, 1.5)[0],
    )


def spin(m, d, car, omega, qlog, ticks=2250):
    s = L.run(m, d, car, lambda i: (-omega, omega), ticks, qlog)
    t = s["t"] - s["t"][0] + L.DT
    cmd = 2 * L.WHEEL_R * omega / L.TRACK
    yr, yr_sd = steady(s["yaw_rate"], t, 1.0)
    x0, y0 = s["x"][0], s["y"][0]
    drift = np.hypot(s["x"] - x0, s["y"] - y0)
    turns = float(np.sum(np.abs(np.diff(np.unwrap(s["yaw"])))) / (2 * np.pi))
    sel = t >= 1.0
    cs = {}
    for k in "FB":
        speed, slip = s[f"cspeed_{k}"][sel], s[f"cslip_{k}"][sel]
        ok = np.isfinite(slip) & (speed > 1e-4)
        cs[k] = dict(slip_over_speed_median=float(np.median(slip[ok] / speed[ok])) if ok.any() else None,
                     no_contact_frac=float(np.mean(s[f"n_c{k}"][sel] == 0)))
    return s, dict(cmd_yaw_rate_dps=float(np.degrees(cmd)), actual_yaw_rate_dps=float(np.degrees(yr)),
                   ratio=float(yr / cmd), yaw_rate_sd_dps=float(np.degrees(yr_sd)), turns=turns,
                   centre_drift_final_mm=1000 * float(drift[-1]), centre_drift_max_mm=1000 * float(drift.max()),
                   drift_per_turn_mm=1000 * float(drift[-1]) / max(turns, 1e-6),
                   caster=cs, wheel_slip_steady={k: steady(s[f"slip_{k}"], t, 1.0)[0] for k in "LR"},
                   pitch_peak_deg=float(np.max(np.abs(s["pitch"]))), roll_peak_deg=float(np.max(np.abs(s["roll"]))))


def arc(m, d, car, qlog):
    wl, wr = 0.5 * L.CRUISE_OMEGA, L.CRUISE_OMEGA
    s = L.run(m, d, car, lambda i: (wl, wr), 2500, qlog)              # 5 s
    t = s["t"] - s["t"][0] + L.DT
    vl, vr = L.WHEEL_R * wl, L.WHEEL_R * wr
    cmd_R = (L.TRACK / 2) * (vr + vl) / (vr - vl)
    cmd_w = (vr - vl) / L.TRACK
    sel = t >= 1.0
    x, y = s["x"][sel], s["y"][sel]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    cx, cy, c = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
    R = float(np.sqrt(c + cx ** 2 + cy ** 2))
    resid = np.hypot(x - cx, y - cy) - R
    return s, dict(cmd_radius_cm=100 * cmd_R, fit_radius_cm=100 * R, fit_resid_max_mm=1000 * float(np.abs(resid).max()),
                   cmd_yaw_rate_dps=float(np.degrees(cmd_w)), actual_yaw_rate_dps=float(np.degrees(steady(s["yaw_rate"], t, 1.0)[0])),
                   cmd_speed_cms=100 * (vr + vl) / 2, actual_speed_cms=100 * steady(s["v_fwd"], t, 1.0)[0],
                   lateral_speed_mean_mms=1000 * steady(s["v_lat"], t, 1.0)[0])


def stop(m, d, car, qlog):
    s1 = L.run(m, d, car, lambda i: (L.CRUISE_OMEGA, L.CRUISE_OMEGA), 1000, qlog)   # cruise 2 s
    s = L.run(m, d, car, lambda i: (0.0, 0.0), 500, qlog)                            # wheels to 0
    t = s["t"] - s["t"][0] + L.DT
    v = s["v_fwd"]
    p = np.c_[s["x"], s["y"]]
    p0 = np.array([s1["x"][-1], s1["y"][-1]])
    dist = np.hypot(*(p - p0).T)
    still = np.abs(v) < 1e-3
    i_stop = int(np.argmax(still)) if still.any() else len(v) - 1
    j = L.jerk(np.r_[s1["a_fwd"][-1], s["a_fwd"]])
    return s, dict(v_at_cut_cms=100 * float(s1["v_fwd"][-1]), stop_time_ms=1000 * float(t[i_stop]),
                   stop_dist_mm=1000 * float(dist[i_stop]), final_dist_mm=1000 * float(dist[-1]),
                   peak_decel_ms2=float(-s["a_fwd"].min()), max_reverse_speed_mms=1000 * float(max(0.0, -v.min())),
                   pitch_peak_deg=float(np.max(np.abs(s["pitch"]))),
                   settle_to_1mms_ms=1000 * L.DT * (int(np.where(np.abs(v) > 1e-3)[0][-1]) + 2),
                   **airborne(s, float(np.median(s1["z"][-200:]))),
                   slip_peak={k: float(np.max(np.abs(s[f"slip_{k}"][:50]))) for k in "LR"},
                   jerk_max=float(np.max(np.abs(j))))


def friction_table(m, d, car):
    """Every floor contact the car has right now: the coefficients MuJoCo actually uses."""
    names = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    rows = []
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, f6)
        rows.append(dict(pair=f"{names(c.geom1)}/{names(c.geom2)}", dim=int(c.dim),
                         friction=[round(float(x), 6) for x in c.friction[:3]],
                         geom_friction=[[float(x) for x in m.geom_friction[c.geom1]],
                                        [float(x) for x in m.geom_friction[c.geom2]]],
                         fn=float(f6[0]), ft=float(np.hypot(f6[1], f6[2]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-identity", default=None,
                    help="traces dir from a run without mj_forward in record(); qpos must match exactly")
    args = ap.parse_args()
    out = Path(args.out)
    xml = L.open_floor_xml(args.xml, out / "xml" / "v1_open_floor.xml")
    m = mujoco.MjModel.from_xml_path(str(xml))
    car = L.Car(m)
    res, traces = {"mujoco": mujoco.__version__, "scene": args.xml, "open_floor_xml": str(xml)}, {}

    def fresh():
        d = mujoco.MjData(m)
        L.settle(m, d, car, ticks=500, zero_first=False)   # drop from the 1 cm spawn height
        L.settle(m, d, car, ticks=100)                     # the controller's own pre-push settle
        return d

    d = fresh()
    res["friction_at_rest"] = friction_table(m, d, car)
    res["weight_N"] = float(sum(m.body_mass[b] for b in range(m.nbody) if b and
                                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "car") in
                                (b, m.body_rootid[b])) * 9.81)
    jobs = [("a_straight", lambda d, q: straight(m, d, car, q)),
            ("b_spin_fast", lambda d, q: spin(m, d, car, L.CRUISE_OMEGA, q)),
            ("b_spin_slow", lambda d, q: spin(m, d, car, 0.25 * L.CRUISE_OMEGA, q, ticks=4500)),
            ("c_arc", lambda d, q: arc(m, d, car, q)),
            ("d_stop", lambda d, q: stop(m, d, car, q))]
    for name, fn in jobs:
        d = fresh()
        qlog = []
        s, met = fn(d, qlog)
        res[name] = met
        np.savez_compressed(out / "traces" / f"{name}.npz", qpos=np.array(qlog), **s)
        if name == "a_straight":
            res["friction_while_driving"] = friction_table(m, d, car)
    if args.check_identity:
        res["identity_max_abs_qpos_diff"] = {
            name: float(np.max(np.abs(np.load(out / "traces" / f"{name}.npz")["qpos"] -
                                      np.load(Path(args.check_identity) / f"{name}.npz")["qpos"])))
            for name, _ in jobs}
    (out / "drive_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
