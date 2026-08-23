#!/usr/bin/env python3
"""Exhaustive depth-2 for the real-table scenes: every first push ends in a finish or a proven dead end.

The 2026-08-22 sweep left holes. Of 44278 depth-1 cells across 478 scenes, 1965 were clean pushes
(no open, no collision, no jam) that never had a single second push tried on them, because
`region_opening.py:2295` drops a frontier node with a bare `continue` and writes nothing. Replaying
those 1965 showed all of them still had reachable contacts afterwards, median 15, so none were dead
ends. Roughly 162k second pushes were never simulated.

Rather than patch those holes and stay dependent on the search's bookkeeping, this recomputes depth 2
from scratch with no search, no frontier, no beam, no budget. Pure enumeration:

  for every reachable (edge, depth) at the start state:
      execute it
      if it opens the goal region      -> opener, recorded, no expansion needed
      if it collides or jams           -> recorded, not expandable
      otherwise                        -> try EVERY reachable (edge, depth) from the resulting state
                                          until one opens, or all of them fail

So each first push lands in exactly one of: opener, blocked (collision/jam), setup (some finish
opens), or dead (every finish tried and none opened). `n_finish_tried` is written for every setup and
dead cell, which is the thing the old data could not say.

Success test is the sweep's own, from region_opening.py:3195 -- at least
ceil(CANONICAL_MIN_REACHABLE_FRACTION * n_sampled) of the points sampled inside the goal region are
reachable, and were not before. 0.2 of 100 points, so 20. Deliberate difference from the planner: it
re-samples region points at every node, this samples once at the root and reuses that set, so
before/after are measured against a fixed target instead of a resampled one. `--verify` checks the
depth-1 verdicts against the existing answer key on the same scenes; run it before trusting a sweep.

Result, Amarel array 60760270, 478 tasks, 2.4M sims, zero failures. Over 50330 first pushes:
18606 setup, 17815 dead, 8731 blocked, 5178 opener. Scored on the sweep's own cell set, the hmax=2
tiers move easy 248->350, med 128->71, hard 102->57, and the median solve rate goes 0.311 -> 0.704.
Compare only on matched cells: this enumerates all five depths per contact (median 115) while the
sweep stopped where a push jams (median 102), and on raw denominators 21 scenes look harder purely
from the larger divisor.

Two of 478 scenes still flip the impossible way, solvable to unsolvable, and the cause is not fixable
in code. `set_full_state` restores qpos and qvel but not the MuJoCo warmstart, so the same push from
the same restored state reports `collision_object` None on a fresh env and 'walls' after ~100 prior
pushes, which flips whether it is allowed at all. Exhaustive labelling here carries that noise at
roughly 2 in 478. Note also that the comment at region_opening.py:2931 claims a reported
collision_object is always the robot's own body; it reads 'walls' here, so that comment is wrong and
the planner has the same sensitivity.

  python scripts/pipeline/exhaustive_hmax2.py --shard 0 --nshards 64 --out $NAMO_SCRATCH/exh2
"""
import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "build_python"))
sys.path.insert(0, os.path.join(REPO, "python"))

import namo_rl  # noqa: E402

CFG = os.path.join(REPO, "config", "namo_config_complete_skill15_car_1x.yaml")
GOALS_PER_REGION = 100          # region_opening.py:344, paired with the 0.2 fraction
MIN_FRACTION = 0.2              # CANONICAL_MIN_REACHABLE_FRACTION
SNAPSHOT_SEED = 42
N_DEPTHS = 5
POOL_ROOT = os.path.join(os.environ.get("NAMO_SCRATCH", "/tmp"), "real_buildable")


def goal_region_points(env):
    """The points sampled inside the goal region at this state, or None if there is no goal region.

    `region_goals` is keyed by the region LABEL ("goal", "robot_region"), not by the integer id in
    `region_labels`, and each value is a RegionGoalBundle whose `.goals` are objects with x/y/theta.
    """
    snap = env.get_region_snapshot(GOALS_PER_REGION, -1.0, False, SNAPSHOT_SEED, True)
    bundle = dict(snap.get("region_goals", {})).get("goal")
    if bundle is None:
        return None
    return [[float(g.x), float(g.y)] for g in bundle.goals] or None


def opens(env, pts, bar):
    return env.count_reachable_points(pts)[0] >= bar


def push(env, obj, edge, depth):
    """Execute one primitive. Returns the step result so the caller can read collision/stuck."""
    cur = env.get_observation()[f"{obj}_pose"]
    a = namo_rl.Action()
    a.object_id = obj
    a.x, a.y, a.theta = float(cur[0]), float(cur[1]), float(cur[2])
    a.edge_idx, a.depth = int(edge), int(depth)
    return env.step(a)


def blocked(res):
    """Did this push fail outright, by the planner's rule at region_opening.py:2931?

    ONLY the robot's own body collision (`collision_object`) or a jam counts. Object-to-wall and
    object-to-object contact never fail a push -- the planner says so in a comment right there, and
    a trial-log entry can carry wall_collision=True alongside success=True. Counting those as
    blocked marked 30974 of 50205 cells unexpandable, lost most of the setups, and made 116 scenes
    look like they had regressed to unsolvable.
    """
    info = res.info if hasattr(res, "info") else {}
    return "collision_object" in info or info.get("stuck") == "true"


def sweep_scene(xml, obj):
    """-> dict for one scene, or None if it has no goal region to open."""
    env = namo_rl.RLEnvironment(xml, CFG, False)
    pts = goal_region_points(env)
    if not pts:
        return None
    bar = max(1, math.ceil(MIN_FRACTION * len(pts)))
    root = env.get_full_state()
    before_open = opens(env, pts, bar)

    cells, n_sims = [], 0
    for edge in env.get_reachable_edges(obj):
        for depth in range(N_DEPTHS):
            env.set_full_state(root)
            res = push(env, obj, edge, depth)
            n_sims += 1
            if opens(env, pts, bar) and not before_open:
                cells.append({"edge": edge, "depth": depth, "kind": "opener"})
                continue
            if blocked(res):
                cells.append({"edge": edge, "depth": depth, "kind": "blocked"})
                continue

            # A clean push that did not open. Try every second push until one opens or all fail.
            mid = env.get_full_state()
            finish_edges = env.get_reachable_edges(obj)
            tried, hit = 0, None
            for e2 in finish_edges:
                for d2 in range(N_DEPTHS):
                    env.set_full_state(mid)
                    push(env, obj, e2, d2)
                    tried += 1
                    n_sims += 1
                    if opens(env, pts, bar):
                        hit = [e2, d2]
                        break
                if hit:
                    break
            cells.append({"edge": edge, "depth": depth,
                          "kind": "setup" if hit else "dead",
                          "n_finish_tried": tried, "finish": hit,
                          "n_finish_reachable": len(finish_edges)})
    return {"xml": xml, "object_id": obj, "n_goal_points": len(pts), "bar": bar,
            "goal_open_at_start": bool(before_open), "cells": cells, "n_sims": n_sims}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", required=True, help="json list of {xml, object_id}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pool-root", default=POOL_ROOT,
                    help="scene pool root; output names are paths relative to it, so they stay "
                         "unique and stay stable across boxes where the pool sits elsewhere")
    ap.add_argument("--verify", action="store_true",
                    help="stop after one scene and print its depth-1 verdicts for eyeballing")
    args = ap.parse_args()

    scenes = json.load(open(args.scenes))
    mine = [s for i, s in enumerate(scenes) if i % args.nshards == args.shard]
    os.makedirs(args.out, exist_ok=True)
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} of {len(scenes)} scenes", flush=True)

    t0, done, sims = time.time(), 0, 0
    for s in mine:
        r = sweep_scene(s["xml"], s["object_id"])
        if r is None:
            continue
        # NOT basename(dirname(xml)): scene ids restart per pool, so 478 scenes carry only 221
        # distinct rb_000NN names and some collide six ways. Naming output by that would have
        # let 257 scenes overwrite each other and still look like a clean run.
        name = os.path.relpath(os.path.dirname(s["xml"]), args.pool_root).replace(os.sep, "__")
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(r, f, separators=(",", ":"))
        done += 1
        sims += r["n_sims"]
        if args.verify:
            k = {}
            for c in r["cells"]:
                k[c["kind"]] = k.get(c["kind"], 0) + 1
            print(f"  {name}: {k}, {r['n_sims']} sims, bar {r['bar']}/{r['n_goal_points']}")
            return
        if done % 10 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(mine)} scenes, {sims} sims, {el:.0f}s "
                  f"({1000*el/max(sims,1):.0f} ms/sim)", flush=True)
    el = time.time() - t0
    print(f"shard {args.shard} done: {done} scenes, {sims} sims, {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
