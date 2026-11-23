#!/usr/bin/env python3
"""End to end: the car drives to each push start and pushes under the ported 30 Hz controller.

Each arm is a (scene, config) pair run on the C++ bindings (`namo_rl`). For every push start
(picked as in python/scripts/car_v2_controller_checks.py, plus the v1 baseline push
obstacle_1_movable edge 30 depth 4) two fresh processes run:
  nav   env.navigate_to_push_start(obj, edge): the drive alone. ok, failure reason, pose error
        against the pose a teleport would use, sim time.
  push  env.step(action): drive, zero velocities, settle 100 ticks, push, settle 100 ticks.
A fresh process each time because NAMO_QPOS_DUMP opens its file once per process.

The dump holds qpos every tick (phase 1 = drive, 3 = settle and push, rounded to 1e-6).
Contacts come from pip mujoco mj_forward on each dumped qpos; contacts depend on qpos alone,
and the rounding is far below the ~0.1 mm a resting wheel sinks into the floor. Speeds are
10 ms centred differences of the C++ qpos, accelerations and jerks the same again, so jerk
here is a 10 ms-window number: compare arms with each other, not with probe_hold.py's per-tick
jerk.

  python probe_e2e.py --build-python <checkout>/build_python --cwd <checkout> --out <dir> \
      --arm v2=<v2 scene>=<v2 config>=0.0406 --arm v1_teleport=<v1 scene>=<v1 config>=0.05357 ...
The number after the config is the arm's free cruise speed at the push cap in m/s; stall is
speed below half of it. `--kinds push` skips the drive-only runs. Every push also leaves a
per-tick trace in <out>/traces/ (car pose, block centre of mass and yaw, block contacts) for
real_push_contact.py.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

import car_probe_lib as L

BASELINE = ("obstacle_1_movable", 30, 4)
SETTLE = 100                     # kSettleSteps in namo_push_controller.cpp
W = 5                            # half window, ticks: 10 ms centred differences


def pick_pushes(env, n):
    """Same picks as car_v2_controller_checks.py: reachable objects by name, their reachable
    edges spread evenly, depth cycling 0..4."""
    picks = []
    per_object = [(o, list(env.get_reachable_edges(o))) for o in sorted(env.get_reachable_objects())]
    per_object = [(o, e) for o, e in per_object if e]
    i = 0
    while len(picks) < n:
        obj, edges = per_object[i % len(per_object)]
        k = i // len(per_object)
        stride = max(1, len(edges) // max(1, math.ceil(n / len(per_object))))
        picks.append((obj, edges[(k * stride) % len(edges)], len(picks) % 5))
        i += 1
    return picks


def child(args):
    import namo_rl
    env = namo_rl.RLEnvironment(args.xml, args.config, False)
    if args.child == "nav":
        r = dict(env.navigate_to_push_start(args.obj, args.edge))
        res = {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
    else:
        a = namo_rl.Action()
        a.object_id, a.edge_idx, a.depth = args.obj, args.edge, args.depth
        a.x = a.y = a.theta = 0.0
        r = env.step(a)
        res = dict(done=bool(r.done), reward=float(r.reward), info={k: str(v) for k, v in dict(r.info).items()})
    Path(args.result).write_text(json.dumps(res))


def quat_yaw_pitch(q):
    w, x, y, z = q.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    head_z = 2 * (x * z - w * y)
    return yaw, np.degrees(np.arcsin(np.clip(-head_z, -1, 1)))


def diff(x):
    """Centred difference over 2W ticks; output aligned with x[W:-W]."""
    return (x[2 * W:] - x[:-2 * W]) / (2 * W * L.DT)


def motion(Q, car, cruise, contacts):
    """Smoothness over one stretch of dumped qpos."""
    n = len(Q)
    if n < 6 * W + 2:
        return dict(ticks=n)
    xy = Q[:, car.qadr:car.qadr + 2]
    yaw, pitch = quat_yaw_pitch(Q[:, car.qadr + 3:car.qadr + 7])
    yaw = np.unwrap(yaw)
    hd = np.c_[np.cos(yaw), np.sin(yaw)][W:-W]
    v = np.sum(diff(xy) * hd, axis=1)
    acc, yr = diff(v), diff(yaw)
    jerk = diff(acc)
    sup, other, blk = contacts
    return dict(
        ticks=n, seconds=n * L.DT,
        path_mm=1000 * float(np.sum(np.hypot(*np.diff(xy, axis=0).T))),
        v_mean_cms=100 * float(v.mean()), v_peak_cms=100 * float(np.abs(v).max()),
        stall_frac=float(np.mean(v < 0.5 * cruise)),
        accel_peak=float(np.abs(acc).max()), jerk_peak=float(np.abs(jerk).max()),
        jerk_rms=float(np.sqrt(np.mean(jerk ** 2))),
        yaw_rate_peak_dps=float(np.degrees(np.abs(yr).max())),
        pitch_max_deg=float(np.abs(pitch).max()),
        airborne_ticks=int((sup == 0).sum()),
        robot_touch_other_ticks=int((other > 0).sum()),
        block_contact_frac=float(np.mean(blk > 0)),
        block_contact_onsets=int(np.sum((blk[1:] > 0) & (blk[:-1] == 0)) + (blk[0] > 0)),
    )


def contacts_per_tick(m, d, car, Q, block_geoms, block_body):
    """Support, other and block contact counts per tick, and the block's centre of mass. The
    block's qpos is no use for its position: pool blocks have their body frame at the world
    origin and the box offset, so the qpos swings whenever the block turns."""
    sup_geoms = {w["geom"] for w in car.wheel.values()} | {c["geom"] for c in car.caster.values()}
    sup, other, blk = (np.zeros(len(Q), int) for _ in range(3))
    com = np.zeros((len(Q), 2))
    byaw = np.zeros(len(Q))
    box = min(g for g in block_geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
    for i, q in enumerate(Q):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        com[i] = d.xipos[block_body][:2]
        byaw[i] = math.atan2(d.geom_xmat[box][3], d.geom_xmat[box][0])
        for g1, g2 in zip(d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]):
            pair = {int(g1), int(g2)}
            mine = pair & car.robot_geoms
            if not mine:
                continue
            rest = pair - mine
            if car.floor in pair:
                sup[i] += bool(mine & sup_geoms)
            elif rest & block_geoms:
                blk[i] += 1
            elif rest:
                other[i] += 1
    return sup, other, blk, com, byaw, m.geom_size[box][:2].copy()


def job(spec):
    arm, kind, xml, config, cruise, obj, edge, depth, a = spec
    tag = f"{arm}__{kind}__{obj}_e{edge}_d{depth}"
    dump, navlog, result = (Path(a.tmp) / f"{tag}.{s}" for s in ("qpos", "navlog", "json"))
    env = dict(os.environ, NAMO_QPOS_DUMP=str(dump), NAMO_NAV_LOG="1",
               PYTHONPATH=f"{a.build_python}:{Path(a.cwd) / 'python'}",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with open(navlog, "w") as f:
        subprocess.run([sys.executable, __file__, "--child", kind, "--xml", xml, "--config", config,
                        "--obj", obj, "--edge", str(edge), "--depth", str(depth), "--result", str(result)],
                       env=env, cwd=a.cwd, stderr=f, stdout=subprocess.DEVNULL, check=True)
    res = json.loads(result.read_text())
    rows = [ln.split() for ln in open(dump) if ln.strip()] if dump.exists() else []
    phase = np.array([int(r[0]) for r in rows], dtype=int)
    Q = np.array([[float(x) for x in r[2:]] for r in rows]) if rows else np.zeros((0, 0))
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    car = L.Car(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
    block_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid}
    out = dict(arm=arm, kind=kind, object=obj, edge=edge, depth=depth, result=res, frames=len(rows))
    if len(rows):
        *cont, com, byaw, half = contacts_per_tick(m, d, car, Q, block_geoms, bid)
        # A fresh scene spawns the car 1 cm up, and a drive starts at once, so the first ticks are
        # the car dropping onto its wheels. Drive metrics start 100 ms after it first lands.
        i1 = np.where(phase == 1)[0]
        if len(i1):
            landed = i1[cont[0][i1] > 0]
            start = int(landed[0]) + 50 if len(landed) else int(i1[-1]) + 1
            nav = i1[i1 >= start]
            out["spawn_drop_ticks"] = int(landed[0] - i1[0]) if len(landed) else None
            out["drive"] = motion(Q[nav], car, cruise, tuple(c[nav] for c in cont))
        else:
            out["drive"] = None
        if kind == "push":
            i3 = np.where(phase == 3)[0]
            ok = res["info"].get("failure_reason", "") == ""
            n_push = len(i3) - SETTLE - (SETTLE if ok else 0)
            pk = i3[SETTLE:SETTLE + n_push] if n_push > 0 else np.array([], int)
            out["push_ticks"] = int(n_push)
            if len(pk):
                out["push"] = motion(Q[pk], car, cruise, tuple(c[pk] for c in cont))
                out["push"]["block_travel_mm"] = 1000 * float(np.hypot(*(com[pk[-1]] - com[pk[0]])))
                out["push"]["car_travel_mm"] = 1000 * float(np.hypot(*(Q[pk[-1], car.qadr:car.qadr + 2] - Q[pk[0], car.qadr:car.qadr + 2])))
                yaw, _ = quat_yaw_pitch(Q[pk, car.qadr + 3:car.qadr + 7])
                (Path(a.out) / "traces").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(Path(a.out) / "traces" / f"{tag}.npz", dt=L.DT,
                                    car_xy=Q[pk, car.qadr:car.qadr + 2], car_yaw=np.unwrap(yaw),
                                    block_com=com[pk], block_yaw=np.unwrap(byaw[pk]), block_half=half,
                                    block_contacts=cont[2][pk], support_contacts=cont[0][pk])
            ctrl = np.array([[float(x) for x in ln.split()[2:4]] for ln in open(navlog) if ln.startswith("[PUSH_CTRL]")])
            if len(ctrl) > 1:
                dc = np.abs(np.diff(ctrl, axis=0)).max(axis=1)
                ref = np.abs(ctrl).max()
                out["push_cmd"] = dict(updates=len(ctrl), max=float(ref),
                                       steps_over_25pct_of_max=int((dc > 0.25 * ref).sum()),
                                       step_max_frac_of_max=float(dc.max() / ref) if ref > 0 else 0.0)
    for p in (dump, result):
        p.unlink(missing_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=["nav", "push"])
    ap.add_argument("--xml")
    ap.add_argument("--config")
    ap.add_argument("--obj")
    ap.add_argument("--edge", type=int)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--result")
    ap.add_argument("--arm", action="append", help="name=scene=config=cruise_ms")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--kinds", default="nav,push", help="comma list of nav, push")
    ap.add_argument("--build-python")
    ap.add_argument("--cwd", help="checkout the config's relative paths resolve against")
    ap.add_argument("--out")
    ap.add_argument("--tmp", default=f"/dev/shm/{os.environ.get('USER', 'u')}_probe_e2e")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.child:
        return child(a)

    sys.path.insert(0, a.build_python)
    import namo_rl
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(a.tmp).mkdir(parents=True, exist_ok=True)
    os.chdir(a.cwd)
    arms, specs, picks_by_arm = [], [], {}
    for spec in a.arm:
        name, xml, config, cruise = spec.split("=")
        env = namo_rl.RLEnvironment(xml, config, False)
        picks = pick_pushes(env, a.n)
        picks_by_arm[name] = picks
        for obj, edge, depth in picks + [BASELINE]:
            for kind in a.kinds.split(","):
                specs.append((name, kind, xml, config, float(cruise), obj, int(edge), int(depth), a))
        arms.append(dict(name=name, xml=xml, config=config, cruise_ms=float(cruise)))
    with ProcessPoolExecutor(a.procs) as ex:
        res = list(ex.map(job, specs))
    (out / "e2e_runs.json").write_text(json.dumps(dict(arms=arms, picks=picks_by_arm, runs=res), indent=1, default=str))
    print(f"wrote {len(res)} runs to {out / 'e2e_runs.json'}")


if __name__ == "__main__":
    main()
