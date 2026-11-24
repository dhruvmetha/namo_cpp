#!/usr/bin/env python3
"""MORE's self-supervised collection, run by MORE's own search code.

This is MORE's System 2 (mcts_main.py at arc-l/more 70402c0): plain UCT with no network
generates its own training data. The tree classes are theirs, imported unmodified from the pinned
clone (namo/planners/opening/more_vendor.py); only the simulator, action library and reward
input are ours. Per episode, exactly as mcts_main.py:422-562 runs:

  search    best_action(300) with test=False (:354, :483): no early stop, 300 iterations
            (paper IV-B; constants.py:42 "300 for collection data")
  save      every tree, one row per child of every node with children (:180-212, :551)
  execute   best_child_top's push (:507-519), reset the caches (:561), search again
  stop      when the executed push opens, or after hmax executed pushes (our horizon)

Nothing here loads a model: `--prior uniform` gives `_make_planner` a shell with scorer=None, and
UCT never reads a prior. That is what lets MORE bootstrap from nothing.

Output, one JSONL line per episode (schema more-trees-v2):
  {schema, xml, object_id, seed, solved, calls, first_open_call, plan, seconds, iterations,
   searches: [{search, calls, iterations, tree_nodes, rows, seconds, best}],
   rows:   [{key, search, uid, depth, obj, edge, push_depth, label, num_visits, child_opened}],
   states: {key: qpos}, legal: {key: [[edge, depth], ...]}}
`calls` is every env.step the episode issued, searches included; it is the collection cost.

  python more_collect.py --manifest M.txt --start 0 --end 100 --out shard.jsonl
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

from scorer_beam import make_env, read_manifest, FALLBACK_GOAL  # noqa: E402
from eval_bestfirst import _make_planner  # noqa: E402
from eval_m3 import sample_goal_points, goal_open_pts  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.planners import get_region_snapshot  # noqa: E402
from namo.planners.opening.more_vendor import (  # noqa: E402
    collect_episode, load_more, make_action_source, make_verifier)

SCHEMA = "more-trees-v2"
HMAX = 2
ITERATIONS = 300            # paper IV-B N_max for collection; constants.py:42 comment


def state_qpos(state):
    """Serialize a full-state object to a plain list."""
    for attr in ("qpos", "positions"):
        v = getattr(state, attr, None)
        if v is not None:
            return [float(x) for x in np.asarray(v).ravel()]
    raise TypeError(f"unknown state type {type(state)}: {dir(state)[:20]}")


def ep_hash(xml, obj):
    return int(hashlib.sha1(f"{xml}|{obj}".encode()).hexdigest()[:8], 16)


def derive_episodes(env):
    """(blocker_object, ...) for the robot->XML-goal 1-hop opening; [] if out of scope.

    Copied from aquaman_collect.derive_episodes so MORE trains on the same notion of an
    episode our own collector uses.
    """
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


def run_episode(planner, env, xml, goal, obj, s0, gp, is_open, iterations, seed, hmax=HMAX):
    """One MORE collection episode on (room, blocking object). Returns the JSONL record."""
    rng = random.Random(seed)
    verifier = make_verifier(is_open, gp)
    source = make_action_source(planner, env, goal, xml, hmax, "uniform", "mean5", rng,
                                restrict_obj=obj, raw=True, region_samples=gp, timing=None)
    env.set_full_state(s0)
    res = collect_episode(env, s0, hmax=hmax, iterations=iterations, action_source=source,
                          verifier=verifier, rng_seed=seed % (2 ** 32))
    return {"schema": SCHEMA, "xml": xml, "object_id": obj, "seed": seed, "solved": res["solved"],
            "calls": res["calls"], "first_open_call": res["first_open_call"],
            "plan": [list(k) for k in res["plan"]], "seconds": res["seconds"],
            "iterations": iterations, "searches": res["searches"], "rows": res["rows"],
            "states": {k: state_qpos(st.sim_state) for k, st in res["parents"].items()},
            "legal": {k: [list(x) for x in v] for k, v in res["legal"].items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=ITERATIONS, help="MCTS iterations per search (MORE: 300)")
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()

    load_more()                                     # fail fast on a wrong or edited MORE clone
    planner, _device, _warm = _make_planner("uniform", "", 0)
    xmls = read_manifest(args.manifest, None)[args.start:args.end]
    census = {"xmls": len(xmls), "episodes": 0, "solved": 0, "rows": 0, "calls": 0, "seconds": 0.0,
              "already_open": 0, "robot_at_goal": 0, "multi_hop": 0, "no_goal_region": 0,
              "no_edge_objects": 0, "env_error": 0}
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
                    rec = run_episode(planner, env, xml, goal, obj, s0, gp, is_open, args.iterations,
                                      args.seed_base + ep_hash(xml, obj) % 100000)
                    census["episodes"] += 1
                    census["solved"] += int(rec["solved"])
                    census["rows"] += len(rec["rows"])
                    census["calls"] += rec["calls"]
                    census["seconds"] += rec["seconds"]
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            except Exception as e:  # noqa: BLE001  one bad scene must not kill a shard
                census["env_error"] += 1
                print(f"ERR {xml}: {type(e).__name__} {e}", flush=True)
            if xi % 20 == 0:
                print(f"{xi}/{len(xmls)} eps={census['episodes']} rows={census['rows']} "
                      f"calls={census['calls']} t={time.time()-t0:.0f}s", flush=True)
    census["wall_s"] = round(time.time() - t0, 1)
    print("CENSUS " + json.dumps(census), flush=True)


if __name__ == "__main__":
    main()
