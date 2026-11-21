#!/usr/bin/env python3
"""One real v1 push through the C++ controller, measured tick by tick.

1. Run the push with the production bindings (`namo_rl`, C++ libmujoco). NAMO_QPOS_DUMP
   writes qpos every tick and NAMO_NAV_LOG=1 prints the wheel command the follower sends
   every tick ([PUSH_CTRL] tick omega_L omega_R mode) and the teleport path ([NAV_PATH]).
2. Replay the same start state, teleport and logged wheel commands in pip mujoco, which
   exposes contacts and forces the bindings do not. The replay is open loop, so how far
   it drifts from the C++ qpos is the check that the two engines and this reconstruction
   agree; the metrics are only trusted if it stays at the millimetre level.

  python probe_push.py --xml <pool scene> --config <car config> --out <dir> [--build-python <checkout>/build_python]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

import car_probe_lib as L

REPO = Path(__file__).resolve().parents[2]


def quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_cpp(xml, config, out, obj, edge, depth):
    dump = out / "traces" / "push_cpp_qpos.txt"
    navlog = out / "traces" / "push_cpp_navlog.txt"
    dump.unlink(missing_ok=True)
    os.environ["NAMO_QPOS_DUMP"] = str(dump)
    os.environ["NAMO_NAV_LOG"] = "1"
    import namo_rl

    env = namo_rl.RLEnvironment(xml, config, visualize=False)
    env.reset()
    reach = list(env.get_reachable_objects())
    if obj is None:
        obj = reach[0]
    edges = list(env.get_reachable_edges(obj))
    if edge is None:
        centres = [e for e in edges if e % 15 == 7]
        edge = centres[0] if centres else edges[len(edges) // 2]
    s0 = env.get_full_state()
    a = namo_rl.Action()
    a.object_id, a.x, a.y, a.theta, a.edge_idx, a.depth = obj, 0.0, 0.0, 0.0, int(edge), int(depth)
    sys.stderr.flush()
    saved = os.dup(2)
    with open(navlog, "w") as f:
        os.dup2(f.fileno(), 2)
        try:
            r = env.step(a)
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
    info = {k: (v if isinstance(v, (bool, int, float, str)) else str(v)) for k, v in dict(r.info).items()}
    frames = np.array([[float(x) for x in ln.split()[2:]] for ln in open(dump) if ln.strip()])
    ctrl, modes, nav, path = [], [], None, None
    for ln in open(navlog):
        if ln.startswith("[PUSH_CTRL]"):
            p = ln.split()
            ctrl.append((float(p[2]), float(p[3]))); modes.append(p[4])
        elif ln.startswith("[PUSH_PATH]") and path is None:
            path = [tuple(map(float, xy.split(","))) for xy in ln.split()[2:]]
        elif ln.startswith("[NAV_PATH]"):
            nav = [tuple(map(float, xy.split(","))) for xy in ln.split()[1:]]
    return dict(obj=obj, edge=int(edge), depth=int(depth), reachable_objects=reach, reachable_edges=edges,
                qpos0=np.array(s0.qpos), qvel0=np.array(s0.qvel), frames=frames,
                ctrl=np.array(ctrl), modes=modes, nav_end=nav[-1] if nav else None, path=path,
                done=bool(r.done), reward=float(r.reward), info=info)


def segments(on):
    """Lengths (ticks) of consecutive True and False runs."""
    runs, cur, n = {True: [], False: []}, on[0], 0
    for v in on:
        if v == cur:
            n += 1
        else:
            runs[cur].append(n); cur, n = v, 1
    runs[cur].append(n)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--object", default=None)
    ap.add_argument("--edge", type=int, default=None)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--settle", type=int, default=100, help="kSettleSteps in namo_push_controller.cpp")
    ap.add_argument("--build-python", default=str(REPO / "build_python"),
                    help="folder holding the compiled namo_rl module; a worktree has none, so point at a built checkout")
    args = ap.parse_args()
    sys.path.insert(0, args.build_python)
    sys.path.insert(0, str(REPO / "python"))
    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)

    c = run_cpp(args.xml, args.config, out, args.object, args.edge, args.depth)
    n_push = len(c["ctrl"])
    S = args.settle
    res = dict(object=c["obj"], edge=c["edge"], depth=c["depth"], push_ticks=n_push,
               cpp_frames=len(c["frames"]), expected_frames=2 * S + n_push,
               done=c["done"], reward=c["reward"], info=c["info"],
               modes={m_: c["modes"].count(m_) for m_ in set(c["modes"])})

    # ---- replay in pip mujoco ----
    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    car = L.Car(m, block_geoms=[c["obj"]])
    d.qpos[:] = c["qpos0"]
    q = d.qpos
    f0 = c["frames"][0]
    q[car.qadr + 0], q[car.qadr + 1] = c["nav_end"]                  # set_robot_se2 x, y
    yaw0 = quat_yaw(f0[car.qadr + 3:car.qadr + 7])                    # push_theta, read back
    q[car.qadr + 3:car.qadr + 7] = [np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)]
    d.qvel[:] = 0
    d.ctrl[:] = 0
    mujoco.mj_forward(m, d)
    qlog = []
    pre = L.run(m, d, car, lambda i: (0.0, 0.0), S, qlog)
    push = L.run(m, d, car, lambda i: tuple(c["ctrl"][i]), n_push, qlog)
    post = L.run(m, d, car, lambda i: (0.0, 0.0), S, qlog) if len(c["frames"]) >= 2 * S + n_push else None
    rq = np.array(qlog)
    cq = c["frames"][:len(rq)]

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c["obj"])
    bq = m.jnt_qposadr[m.body_jntadr[bid]]
    err = lambda a_, b_: np.hypot(a_[:, 0] - b_[:, 0], a_[:, 1] - b_[:, 1])
    pk = slice(S, S + n_push)
    car_err = err(rq[:, car.qadr:car.qadr + 2], cq[:, car.qadr:car.qadr + 2])
    blk_err = err(rq[:, bq:bq + 2], cq[:, bq:bq + 2])
    yaw_err = np.degrees(np.abs(np.angle(np.exp(1j * (np.array([quat_yaw(r[car.qadr + 3:car.qadr + 7]) for r in rq]) -
                                                       np.array([quat_yaw(r[car.qadr + 3:car.qadr + 7]) for r in cq]))))))
    res["replay_vs_cpp"] = dict(car_xy_err_max_mm=1000 * float(car_err.max()), car_xy_err_end_push_mm=1000 * float(car_err[S + n_push - 1]),
                                block_xy_err_max_mm=1000 * float(blk_err.max()), block_xy_err_end_push_mm=1000 * float(blk_err[S + n_push - 1]),
                                car_yaw_err_max_deg=float(yaw_err.max()), pip_mujoco=mujoco.__version__)

    # ---- metrics over the push ticks ----
    s = push
    t = np.arange(n_push) * L.DT
    ctrl = c["ctrl"]
    dctrl = np.abs(np.diff(ctrl, axis=0))
    wcr = L.CRUISE_OMEGA
    v = s["v_fwd"]
    on = s["n_block"] > 0
    seg = segments(on)
    j = L.jerk(s["a_fwd"])
    bx = rq[pk, bq:bq + 2]
    bv = np.r_[0.0, np.hypot(*np.diff(bx, axis=0).T) / L.DT]
    cx = rq[pk, car.qadr:car.qadr + 2]
    res["push"] = dict(
        duration_s=n_push * L.DT,
        ctrl_mean={"L": float(ctrl[:, 0].mean()), "R": float(ctrl[:, 1].mean()), "cruise": wcr},
        ctrl_sd={"L": float(ctrl[:, 0].std()), "R": float(ctrl[:, 1].std())},
        ctrl_step_max=float(dctrl.max()), ctrl_steps_over_0p1=int((dctrl > 0.1).sum()),
        ctrl_total_variation_per_s=float(dctrl.sum() / (n_push * L.DT)),
        steer_frac=float(np.mean(np.abs(ctrl[:, 0] - ctrl[:, 1]) > 0.1 * wcr)),
        wheel_reverse_frac=float(np.mean((ctrl < 0).any(axis=1))),
        omega_track_err_mean=float(np.mean(np.abs(np.c_[s["omega_L"], s["omega_R"]] - ctrl))),
        v_mean_cms=100 * float(v.mean()), v_sd_cms=100 * float(v.std()),
        v_min_max_cms=[100 * float(v.min()), 100 * float(v.max())],
        stall_frac_v_below_half_cruise=float(np.mean(v < 0.5 * L.CRUISE_MS)),
        surge_frac_v_above_1p2_cruise=float(np.mean(v > 1.2 * L.CRUISE_MS)),
        yaw_rate_rms_dps=float(np.degrees(np.sqrt(np.mean(s["yaw_rate"] ** 2)))),
        yaw_rate_max_dps=float(np.degrees(np.abs(s["yaw_rate"]).max())),
        pitch_max_deg=float(np.abs(s["pitch"]).max()),
        airborne_ticks=int(((s["n_wL"] + s["n_wR"] + s["n_cF"] + s["n_cB"]) == 0).sum()),
        block_contact_frac=float(on.mean()), block_contact_onsets=int(np.sum(on[1:] & ~on[:-1]) + on[0]),
        block_contact_on_ms_mean=1000 * L.DT * float(np.mean(seg[True])) if seg[True] else 0.0,
        block_contact_off_ms_mean=1000 * L.DT * float(np.mean(seg[False])) if seg[False] else 0.0,
        first_contact_ms=1000 * L.DT * float(np.argmax(on)) if on.any() else None,
        block_fn_mean_when_on=float(s["fn_block"][on].mean()) if on.any() else 0.0,
        block_fn_max=float(s["fn_block"].max()),
        block_moving_frac=float(np.mean(bv > 1e-3)),
        mu_car_block=float(np.nanmax(s["mu_block"])) if np.isfinite(s["mu_block"]).any() else None,
        car_travel_mm=1000 * float(np.hypot(*(cx[-1] - cx[0]))),
        block_travel_mm=1000 * float(np.hypot(*(bx[-1] - bx[0]))),
        commanded_travel_mm=1000 * L.CRUISE_MS * n_push * L.DT,
        wheel_slip_mean={k: float(np.mean(s[f"slip_{k}"])) for k in "LR"},
        wheel_slip_absmax={k: float(np.abs(s[f"slip_{k}"]).max()) for k in "LR"},
        jerk_rms=float(np.sqrt(np.mean(j ** 2))), jerk_p99=float(np.percentile(np.abs(j), 99)),
        jerk_max=float(np.abs(j).max()),
        jerk_rms_after_100ms=float(np.sqrt(np.mean(j[t[1:] >= 0.1] ** 2))),
    )
    # ---- the same push read straight from the C++ qpos (no replay drift) ----
    W = 5                                                     # centred difference over 10 ms
    cxy = cq[pk, car.qadr:car.qadr + 2]
    yaw = np.unwrap([quat_yaw(r[car.qadr + 3:car.qadr + 7]) for r in cq[pk]])
    vel = (cxy[2 * W:] - cxy[:-2 * W]) / (2 * W * L.DT)
    hd = np.c_[np.cos(yaw), np.sin(yaw)][W:-W]
    v_cpp = np.sum(vel * hd, axis=1)
    yr_cpp = (yaw[2 * W:] - yaw[:-2 * W]) / (2 * W * L.DT)
    bxy = cq[pk, bq:bq + 2]
    bsp = np.hypot(*((bxy[2 * W:] - bxy[:-2 * W]) / (2 * W * L.DT)).T)
    p0, p2 = np.array(c["path"][0]), np.array(c["path"][-1])
    u = (p2 - p0) / np.linalg.norm(p2 - p0)
    cte = (cxy - p0) @ np.array([-u[1], u[0]])
    db = 0.10 * 0.07                                          # cte_deadband = 0.10 * car_size
    cross = np.where(np.diff((np.abs(cte) >= db).astype(int)) != 0)[0] + 1
    big = np.where(dctrl.max(axis=1) > 0.1)[0] + 1
    after = [int(b - cross[cross <= b].max()) for b in big if (cross <= b).any()]
    res["push_cpp"] = dict(
        v_mean_cms=100 * float(v_cpp.mean()), v_sd_cms=100 * float(v_cpp.std()),
        v_min_max_cms=[100 * float(v_cpp.min()), 100 * float(v_cpp.max())],
        stall_frac_v_below_half_cruise=float(np.mean(v_cpp < 0.5 * L.CRUISE_MS)),
        yaw_rate_rms_dps=float(np.degrees(np.sqrt(np.mean(yr_cpp ** 2)))),
        yaw_rate_max_dps=float(np.degrees(np.abs(yr_cpp).max())),
        car_travel_mm=1000 * float(np.hypot(*(cxy[-1] - cxy[0]))),
        block_travel_mm=1000 * float(np.hypot(*(bxy[-1] - bxy[0]))),
        block_moving_frac=float(np.mean(bsp > 1e-3)),
        cte_absmax_mm=1000 * float(np.abs(cte).max()), cte_deadband_mm=1000 * db,
        cte_deadband_crossings=cross.tolist(), big_ctrl_steps=int(len(big)),
        big_ctrl_steps_within_15_ticks_after_crossing=int(sum(a <= 15 for a in after)),
        path=c["path"],
    )
    np.savez_compressed(out / "traces" / "e_push.npz", cte=cte, v_cpp=v_cpp, yr_cpp=yr_cpp,
                        qpos=rq, cpp_qpos=cq, ctrl=ctrl, modes=np.array(c["modes"]), settle=S, **{k: v_ for k, v_ in s.items()})
    (out / "push_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
