#!/usr/bin/env python3
"""Aquaman round-1 collector — the deploy search AS the data generator (search-as-collector).

EXP-2026-08-02-bootstrap-value-loop. Per fresh XML (no key, no labels): derive the region-opening
episode via get_region_snapshot (robot region + XML-goal region + the blocking object on their
edge; multi-hop scenes skipped), then run the deploy best-first (`solve_scene`, canonical
defaults: dedupe+jam, --raw scores) and harvest the WHOLE tree:
  - per simmed push (pop): board_id, edge, depth, q, opened, fail, AND the reached state's qpos
    (lightweight capture — no rendering; ctx renders happen at build time only)
  - per board: parent linkage + tries
Collection policy (locked defaults, card §Locked):
  - explore slice: episodes with hash%5==0 run PRIOR=uniform (random order)  [wrong-LOW police]
  - easy quota: solved with sims<=5 -> keep trace only if hash%10==0          [frontier ratchet]
  - audit slice: hash%20==1 episodes run with budget 900 instead of 150      [fresh answer key]
Output: one JSONL per shard; one line per KEPT episode:
  {xml, object_id, mode, audit, solved, sims, plan_len, root_qpos, boards:[...], pops:[...]}
plus a per-shard census line (kept/skipped counts by reason).

  python aquaman_collect.py --manifest M.txt --start 0 --end 100 --ckpt C --out shard.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from scorer_beam import BeamPlanner, make_env, read_manifest, FALLBACK_GOAL, CFG  # noqa: E402
from eval_bestfirst import solve_scene  # noqa: E402
from eval_m3 import sample_goal_points, goal_open_pts  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.planners import get_region_snapshot  # noqa: E402

BUDGET, AUDIT_BUDGET, HMAX = 150, 900, 2


def state_qpos(state):
    """Serialize a full-state object to a plain list (set_full_state round-trips it at build time)."""
    for attr in ("qpos", "positions"):
        v = getattr(state, attr, None)
        if v is not None:
            return [float(x) for x in np.asarray(v).ravel()]
    raise TypeError(f"unknown state type {type(state)}: {dir(state)[:20]}")


def ep_hash(xml, obj):
    return int(hashlib.sha1(f"{xml}|{obj}".encode()).hexdigest()[:8], 16)


def derive_episodes(env):
    """(blocker_object, ...) for the robot->XML-goal 1-hop opening; [] if out of scope."""
    snap = get_region_snapshot(env, goals_per_region=1, goal_radius=0.15, local_info_only=True,
                               seed=0, use_cpp_unified=True, use_xml_goal=True)
    robot = snap.get("robot_label")
    labels = set(snap["region_labels"].values()) if isinstance(snap["region_labels"], dict) else set()
    if not robot or robot in ("robot_goal", "goal"):
        return [], "robot_at_goal"
    if "goal" not in snap["adjacency"].get(robot, set()):
        return [], ("multi_hop" if "goal" in labels else "no_goal_region")
    objs = snap["edge_objects"].get(robot, {}).get("goal", set())
    if objs:
        return sorted(objs), None
    return [], "no_edge_objects"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--mine", action="store_true", help="multi-solution collection (round-3 doctrine)")
    args = ap.parse_args()

    planner = BeamPlanner(args.ckpt, CFG)
    xmls = read_manifest(args.manifest, None)[args.start:args.end]
    census = {"xmls": len(xmls), "kept": 0, "quota_dropped": 0, "already_open": 0,
              "robot_at_goal": 0, "multi_hop": 0, "no_goal_region": 0, "no_edge_objects": 0,
              "env_error": 0, "episodes": 0, "solved": 0}
    t0 = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for xi, xml in enumerate(xmls):
            try:
                env = make_env(xml)
                goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
                env.set_robot_goal(*goal)
                env.get_reachable_objects()
                gp = sample_goal_points(env)
                is_open = (lambda e, p=gp: goal_open_pts(e, p))
                if is_open(env):
                    census["already_open"] += 1
                    continue
                s0 = env.get_full_state()
                eps, reason = derive_episodes(env)
                if reason:
                    census[reason] += 1
                    continue
                for obj in eps:
                    census["episodes"] += 1
                    h = ep_hash(xml, obj)
                    mode = "uniform" if h % 5 == 0 else "model"
                    audit = (h % 20 == 1)
                    budget = AUDIT_BUDGET if audit else args.budget
                    rng = random.Random(h)
                    pops = []
                    capture = lambda st: (state_qpos(st), None)
                    env.set_full_state(s0)
                    solved, sims, plen, boards, end = solve_scene(
                        planner, env, goal, xml, s0, HMAX, budget, mode, "mean5", "q", rng,
                        restrict_obj=obj, is_open=is_open, raw=True,
                        discount="off", dedupe_noop=True, prune_jam_depth=True,
                        trace_out=pops, capture=capture,
                        **({"stop_on_open": False} if args.mine else {}))
                    census["solved"] += int(solved)
                    if solved and sims <= 5 and h % 10 != 0:
                        census["quota_dropped"] += 1
                        continue
                    census["kept"] += 1
                    f.write(json.dumps({
                        "xml": xml, "object_id": obj, "mode": mode, "audit": audit,
                        "solved": solved, "sims": sims, "plan_len": plen,
                        "budget": budget, "root_qpos": state_qpos(s0),
                        "boards": [{k: b[k] for k in ("board_id", "depth", "n_candidates",
                                                      "parent_edge", "parent_depth", "k_failed")
                                    if k in b} for b in boards],
                        "pops": pops}) + "\n")
                    f.flush()
            except Exception as e:  # noqa: BLE001  one bad scene must not kill a shard
                census["env_error"] += 1
                print(f"ERR {xml}: {type(e).__name__} {e}", flush=True)
            if xi % 20 == 0:
                print(f"{xi}/{len(xmls)} kept={census['kept']} t={time.time()-t0:.0f}s", flush=True)
    census["wall_s"] = round(time.time() - t0, 1)
    print("CENSUS " + json.dumps(census), flush=True)


if __name__ == "__main__":
    main()
