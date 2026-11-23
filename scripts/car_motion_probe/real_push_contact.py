#!/usr/bin/env python3
"""Does the real car stall when it pushes a block? Real camera logs against the sim e2e pushes, car side only.

Real: every trial under robot_control/push_calibration/obj_1 with wheel commands (the 33 the drivetrain
fit used). The camera tracks the car's marker pose and the block (centre, yaw, 15 x 7 cm) at about
15 Hz. Sim: the push traces probe_e2e.py --kinds push leaves (2 ms ticks), resampled at the camera's
frame period and put through the same pipeline.

Pipeline, the same for both:
  speed    the car's forward speed, slope of a 5-frame linear fit (real_drive_fit/kin.py, the series
           the drivetrain fit scored against)
  push     real: frames in state PUSHING; sim: the push ticks without the settles
  contact  the camera cannot see touch, and the marker-to-body offset is not known well enough for a
           gap test (a 7 x 7 box on the marker sits 3.8 cm short of the block on edges 1 and 15 and
           overlaps it on 45 and 59). So contact comes from the block: it starts at the last frame the
           block centre is within 0.05 cm of where it began the push, before it first passes 0.3 cm
           (prep.py's threshold), and ends at the last push frame that the centre or any corner moves
           faster than 0.5 cm/s. On sim the exact MuJoCo contacts check this window.
  stall    speed below half the free cruise (probe_e2e's stall_frac), over contact frames and over the
           whole push
  class    median contact speed / cruise: below 0.2 stops, 0.2 to 0.8 slows, 0.8 and up keeps cruise
Cruise is the free speed at the push cap: real 3.66 cm/s (the drivetrain fit's forward target,
delivered (0.38, 0.34)); sim the value probe_e2e was given (v2 4.06, v1 5.357 cm/s). Each push also
reports its own speed before contact, at the same command.

  python real_push_contact.py --robot-control <robot_control checkout> --e2e <probe_e2e out dir> --out <dir>
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

REAL_CRUISE = 3.66
ONSET_CM, REST_CM, MOVING_CMS = 0.3, 0.05, 0.5
CLASSES = ((0.2, "stops"), (0.8, "slows"), (np.inf, "keeps cruise"))
W = 5                                   # sim raw speed: 10 ms centred difference, as probe_e2e


def local_fit(t, y, half=2):
    """Slope of a centred (2 half + 1)-frame linear fit; nan at the ends (real_drive_fit/kin.py)."""
    s = np.full(len(t), np.nan)
    for k in range(half, len(t) - half):
        tt = t[k - half:k + half + 1] - t[k]
        yy = y[k - half:k + half + 1]
        s[k] = (tt @ (yy - yy.mean())) / (tt @ tt)
    return s


def block_points(c, yaw, half):
    """Centre and four corners, (n, 5, 2)."""
    pts = np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]) * np.asarray(half)
    cs, sn = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return np.stack([c[:, None, 0] + pts[:, 0] * cs - pts[:, 1] * sn,
                     c[:, None, 1] + pts[:, 0] * sn + pts[:, 1] * cs], -1)


def stats(t, v, c, mask, cruise):
    idx = np.where(mask)[0]
    if not len(idx):
        return None
    vv = v[idx][np.isfinite(v[idx])]
    dur = len(idx) * float(np.median(np.diff(t)))
    travel = float(np.hypot(*(c[idx[-1]] - c[idx[0]])))
    med = float(np.median(vv))
    return dict(contact_s=dur, frames=int(len(idx)), v_median_cms=med, v_over_cruise=med / cruise,
                stall_frac=float(np.mean(vv < 0.5 * cruise)),
                cls=next(name for lim, name in CLASSES if med / cruise < lim),
                block_travel_cm=travel, block_cm_per_s=travel / dur)


def analyse(t, car_xy, car_th, blk_c, blk_yaw, half, cruise, contact=None):
    """One push on camera frames: cm, rad, s. contact = exact per-frame contact flags (sim only)."""
    v = local_fit(t, car_xy[:, 0]) * np.cos(car_th) + local_fit(t, car_xy[:, 1]) * np.sin(car_th)
    P = block_points(blk_c, blk_yaw, half)
    bs = np.max([np.hypot(local_fit(t, P[:, k, 0]), local_fit(t, P[:, k, 1])) for k in range(5)], axis=0)
    bd = np.hypot(*(blk_c - blk_c[0]).T)
    ok = np.isfinite(v)
    out = dict(push_s=float(t[-1] - t[0]), push_frames=int(len(t)),
               block_travel_push_cm=float(np.hypot(*(blk_c[-1] - blk_c[0]))),
               stall_frac_push=float(np.mean(v[ok] < 0.5 * cruise)),
               v_min_after_start_cms=float(np.nanmin(v[4:])) if len(v) > 6 else None)
    moved = np.where(bd > ONSET_CM)[0]
    if not len(moved):
        out["inferred"] = None
    else:
        k = moved[0]
        while k > 0 and bd[k - 1] > REST_CM:
            k -= 1
        live = np.where(bs[k:] > MOVING_CMS)[0]
        e = k + (live[-1] if len(live) else 0)
        win = np.zeros(len(t), bool)
        win[k:e + 1] = True
        out["inferred"] = stats(t, v, blk_c, win, cruise)
        out["inferred"]["onset_s"] = float(t[k] - t[0])
        pre = np.arange(len(t))
        pre = (pre >= 4) & (pre < k - 2) & ok
        out["free_before_contact_cms"] = float(np.median(v[pre])) if pre.sum() >= 3 else None
        if contact is not None and contact.any():
            out["window_vs_exact"] = dict(window_frames_touching=float(contact[win].mean()),
                                          touching_frames_in_window=float(win[contact].mean()))
    if contact is not None:
        out["exact"] = stats(t, v, blk_c, contact, cruise)
    return out


def load_real(d):
    W_ = [json.loads(ln) for ln in open(os.path.join(d, "wheel_commands.jsonl"))]
    M = {}
    for ln in open(os.path.join(d, "mid_obs.jsonl")):
        m = json.loads(ln)
        M[m["observation_timestamp"]] = m
    rows, seen = [], set()
    for w in W_:
        ts = w["obs_timestamp"]
        if ts in seen or ts not in M:
            continue
        seen.add(ts)
        m = M[ts]
        o = m["objects"]["obj_1"]
        rows.append((ts, *m["robot_pose_cm"], o["x_cm"], o["y_cm"], o["theta_deg"], w["state"], o["width_cm"], o["depth_cm"]))
    t, x, y, th, ox, oy, oth = (np.array([r[i] for r in rows], float) for i in range(7))
    st = np.array([r[7] for r in rows])
    P = st == "PUSHING"
    half = (rows[0][8] / 2, rows[0][9] / 2)
    return dict(t=t[P] - t[0], car_xy=np.c_[x, y][P], car_th=np.unwrap(np.radians(th))[P],
                blk_c=np.c_[ox, oy][P], blk_yaw=np.unwrap(np.radians(oth))[P], half=half,
                frame_dt=float(np.median(np.diff(t))))


def load_sim(npz, period):
    Z = np.load(npz)
    dt = float(Z["dt"])
    n = len(Z["car_xy"])
    idx = np.unique(np.round(np.arange(0, (n - 1) * dt + 1e-12, period) / dt).astype(int))
    idx = idx[idx < n]
    xy = Z["car_xy"]
    hd = np.c_[np.cos(Z["car_yaw"]), np.sin(Z["car_yaw"])][W:-W]
    v_raw = 100 * np.sum((xy[2 * W:] - xy[:-2 * W]) / (2 * W * dt) * hd, axis=1)
    touch_raw = Z["block_contacts"][W:-W] > 0
    return dict(t=idx * dt, car_xy=100 * xy[idx], car_th=Z["car_yaw"][idx], blk_c=100 * Z["block_com"][idx],
                blk_yaw=Z["block_yaw"][idx], half=100 * Z["block_half"], contact=Z["block_contacts"][idx] > 0,
                v_raw=v_raw, touch_raw=touch_raw)


def summary(rows):
    def med(key, sub):
        vals = [r[sub][key] for r in rows if r.get(sub)]
        return dict(median=float(np.median(vals)), min=float(np.min(vals)), max=float(np.max(vals)), n=len(vals)) if vals else None
    out = dict(pushes=len(rows),
               stall_frac_push=dict(median=float(np.median([r["stall_frac_push"] for r in rows]))),
               block_travel_push_cm=dict(median=float(np.median([r["block_travel_push_cm"] for r in rows])),
                                         min=float(np.min([r["block_travel_push_cm"] for r in rows])),
                                         max=float(np.max([r["block_travel_push_cm"] for r in rows]))))
    for sub in ("inferred", "exact"):
        if any(r.get(sub) for r in rows):
            out[sub] = {k: med(k, sub) for k in ("stall_frac", "v_over_cruise", "v_median_cms", "contact_s",
                                                 "block_travel_cm", "block_cm_per_s")}
            out[sub]["classes"] = {name: sum(1 for r in rows if r.get(sub) and r[sub]["cls"] == name) for _, name in CLASSES}
    pre = [r["free_before_contact_cms"] for r in rows if r.get("free_before_contact_cms")]
    ratio = [r["inferred"]["v_median_cms"] / r["free_before_contact_cms"] for r in rows
             if r.get("free_before_contact_cms") and r.get("inferred")]
    out["free_before_contact_cms"] = dict(median=float(np.median(pre)), n=len(pre)) if pre else None
    out["contact_over_free_before"] = dict(median=float(np.median(ratio)), min=float(np.min(ratio)),
                                           max=float(np.max(ratio)), n=len(ratio)) if ratio else None
    raw = [r["exact_raw_stall_frac"] for r in rows if r.get("exact_raw_stall_frac") is not None]
    if raw:
        out["exact_raw_stall_frac"] = dict(median=float(np.median(raw)), min=float(np.min(raw)), max=float(np.max(raw)))
    wv = [r["window_vs_exact"] for r in rows if r.get("window_vs_exact")]
    if wv:
        out["window_vs_exact"] = {k: float(np.median([w[k] for w in wv])) for k in wv[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-control", required=True)
    ap.add_argument("--e2e", required=True, help="probe_e2e.py --kinds push output dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    trials = sorted(os.path.dirname(f) for f in glob.glob(
        os.path.join(a.robot_control, "push_calibration/obj_1/edge*/depth*/trial*/wheel_commands.jsonl")))
    arms = {"real": []}
    periods = []
    for d in trials:
        R = load_real(d)
        periods.append(R["frame_dt"])
        r = analyse(R["t"], R["car_xy"], R["car_th"], R["blk_c"], R["blk_yaw"], R["half"], REAL_CRUISE)
        arms["real"].append(dict(push=d.split("obj_1/")[1], **r))
    period = float(np.median(periods))

    E = json.loads((Path(a.e2e) / "e2e_runs.json").read_text())
    cruise = {x["name"]: 100 * x["cruise_ms"] for x in E["arms"]}
    for run in E["runs"]:
        tag = f"{run['arm']}__push__{run['object']}_e{run['edge']}_d{run['depth']}"
        npz = Path(a.e2e) / "traces" / f"{tag}.npz"
        if run["kind"] != "push" or not npz.exists():
            continue
        S = load_sim(npz, period)
        r = analyse(S["t"], S["car_xy"], S["car_th"], S["blk_c"], S["blk_yaw"], S["half"], cruise[run["arm"]], S["contact"])
        c = cruise[run["arm"]]
        r["exact_raw_stall_frac"] = float(np.mean(S["v_raw"][S["touch_raw"]] < 0.5 * c)) if S["touch_raw"].any() else None
        r["push_failure"] = run["result"]["info"].get("failure_reason", "")
        r["block_half_cm"] = [float(x) for x in S["half"]]
        arms.setdefault(run["arm"], []).append(dict(push=f"{run['object']} e{run['edge']} d{run['depth']}", **r))

    report = dict(camera_period_s=period, cruise_cms=dict(real=REAL_CRUISE, **cruise),
                  settings=dict(onset_cm=ONSET_CM, rest_cm=REST_CM, block_moving_cms=MOVING_CMS,
                                classes={name: lim for lim, name in CLASSES}),
                  summary={k: summary(v) for k, v in arms.items()}, pushes=arms)
    (out / "real_push_contact.json").write_text(json.dumps(report, indent=1, default=float))

    def fmt(s, sub, key, f="{:.2f}"):
        x = (s.get(sub) or {}).get(key)
        return f"{f.format(x['median'])} [{f.format(x['min'])}-{f.format(x['max'])}]" if x else "-"
    print(f"camera period {period:.4f} s; cruise cm/s {report['cruise_cms']}")
    print(f"{'arm':<12} {'n':>3} {'stall_contact':>18} {'stall_push':>10} {'v/cruise':>18} {'v/free_before':>18} "
          f"{'block_cm_push':>20} {'block_cm/s_contact':>20} classes")
    for k, s in report["summary"].items():
        cb = s.get("contact_over_free_before")
        cb = f"{cb['median']:.2f} [{cb['min']:.2f}-{cb['max']:.2f}]" if cb else "-"
        print(f"{k:<12} {s['pushes']:>3} {fmt(s, 'inferred', 'stall_frac'):>18} {s['stall_frac_push']['median']:>10.2f} "
              f"{fmt(s, 'inferred', 'v_over_cruise'):>18} {cb:>18} "
              f"{s['block_travel_push_cm']['median']:>8.2f} [{s['block_travel_push_cm']['min']:.2f}-{s['block_travel_push_cm']['max']:.2f}]"
              f" {fmt(s, 'inferred', 'block_cm_per_s'):>20} {s['inferred']['classes'] if s.get('inferred') else '-'}")
        if s.get("exact"):
            print(f"{'  exact':<12} {'':>3} {fmt(s, 'exact', 'stall_frac'):>18} {'':>10} {fmt(s, 'exact', 'v_over_cruise'):>18} "
                  f"{'':>18} {'':>20} {fmt(s, 'exact', 'block_cm_per_s'):>20} {s['exact']['classes']}"
                  f"  raw10ms_stall={s['exact_raw_stall_frac']['median']:.2f} window_vs_exact={s.get('window_vs_exact')}")


if __name__ == "__main__":
    main()
