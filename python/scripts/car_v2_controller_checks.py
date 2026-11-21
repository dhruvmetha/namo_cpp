"""Checks for the opt-in car v2 controller (skill.car_nav_mode / car_controller_hz / car_actuation).

Two subcommands:

  v1-qpos     Run N pushes from one start state and dump the final qpos of each as float.hex, so two
              builds can be compared bit for bit. Use it before and after a C++ change with the v1 config.
  drive-smoke Drive to N push start poses (navigate_to_push_start, no push) and report the pose
              error against the pose a teleport would use, sim time, and any failure reason.
  compare     Diff two v1-qpos dumps; exits 1 on any difference.

Both pick pushes the same way: reachable objects sorted by name, then their reachable edges spread
evenly, depth cycling 0..4. Every push starts from the same saved state.

Example:
  source env.ilab.sh
  python python/scripts/car_v2_controller_checks.py v1-qpos --xml SCENE --config config/namo_config_complete_skill15_car_1x.yaml --out before.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import namo_rl


def pick_pushes(env, n):
    picks = []
    objects = sorted(env.get_reachable_objects())
    per_object = [(obj, env.get_reachable_edges(obj)) for obj in objects]
    per_object = [(obj, edges) for obj, edges in per_object if edges]
    i = 0
    while len(picks) < n:
        obj, edges = per_object[i % len(per_object)]
        k = i // len(per_object)
        stride = max(1, len(edges) // max(1, math.ceil(n / len(per_object))))
        edge = edges[(k * stride) % len(edges)]
        picks.append((obj, edge, len(picks) % 5))
        i += 1
    return picks


def make_action(obj, edge, depth):
    a = namo_rl.Action()
    a.object_id = obj
    a.edge_idx = edge
    a.depth = depth
    a.x = a.y = a.theta = 0.0
    return a


def cmd_v1_qpos(args):
    env = namo_rl.RLEnvironment(args.xml, args.config, False)
    s0 = env.get_full_state()
    rows = []
    for obj, edge, depth in pick_pushes(env, args.n):
        env.set_full_state(s0)
        result = env.step(make_action(obj, edge, depth))
        qpos = list(env.get_full_state().qpos)
        rows.append({
            "object": obj, "edge": edge, "depth": depth,
            "failure_reason": result.info.get("failure_reason", ""),
            "qpos_hex": [float(v).hex() for v in qpos],
        })
    digest = hashlib.sha256(json.dumps([r["qpos_hex"] for r in rows]).encode()).hexdigest()
    with open(args.out, "w") as f:
        json.dump({"xml": args.xml, "config": args.config, "sha256": digest, "pushes": rows}, f, indent=1)
    ran = sum(1 for r in rows if not r["failure_reason"])
    print(f"pushes={len(rows)} ran={ran} qpos_sha256={digest}")


def cmd_compare(args):
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    diffs = 0
    for i, (ra, rb) in enumerate(zip(a["pushes"], b["pushes"])):
        same = ra["qpos_hex"] == rb["qpos_hex"] and ra["failure_reason"] == rb["failure_reason"]
        diffs += not same
        if not same:
            print(f"push {i} {ra['object']} e{ra['edge']} d{ra['depth']} differs")
    print(f"compared={min(len(a['pushes']), len(b['pushes']))} differing={diffs} "
          f"sha_a={a['sha256'][:16]} sha_b={b['sha256'][:16]}")
    raise SystemExit(1 if diffs or len(a["pushes"]) != len(b["pushes"]) else 0)


def cmd_drive_smoke(args):
    import time
    env = namo_rl.RLEnvironment(args.xml, args.config, False)
    s0 = env.get_full_state()
    print(f"{'push start':<24} {'pos_err_mm':>10} {'yaw_err_deg':>11} {'sim_s':>7} {'wall_s':>6}  result")
    for obj, edge, _ in pick_pushes(env, args.n):
        env.set_full_state(s0)
        start = time.perf_counter()
        r = env.navigate_to_push_start(obj, edge)
        wall = time.perf_counter() - start
        (tx, ty, tth), (x, y, th) = r["teleport_pose"], r["final_pose"]
        err_mm = 1000.0 * math.hypot(x - tx, y - ty)
        dyaw = math.degrees(math.atan2(math.sin(th - tth), math.cos(th - tth)))
        label = f"{obj} e{edge}"
        print(f"{label:<24} {err_mm:>10.2f} {dyaw:>11.2f} {r['nav_time_s']:>7.2f} {wall:>6.2f}  "
              f"{r['failure_reason'] or 'ok'}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("v1-qpos")
    q.add_argument("--xml", required=True)
    q.add_argument("--config", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--n", type=int, default=20)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    d = sub.add_parser("drive-smoke")
    d.add_argument("--xml", required=True)
    d.add_argument("--config", required=True)
    d.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    {"v1-qpos": cmd_v1_qpos, "compare": cmd_compare, "drive-smoke": cmd_drive_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
