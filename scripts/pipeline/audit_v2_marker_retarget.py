#!/usr/bin/env python3
"""Can the robot reach the goal MARKER after the push, and if not, how far must it retarget?

The hardware executor drives to the goal marker. When the marker's cell is wavefront-blocked it
retargets to the nearest free reachable cell within 12.0 cm and logs success-with-retarget; beyond
12.0 cm it is a failure. That threshold is theirs, derived as half the largest movable's long side
(7.5) plus robot inflation (3.5) plus tier1 margin (0.5). So a scene needs three verdicts, not one:

  strict     some working solution leaves the marker itself reachable
  retarget   no solution does, but one leaves a reachable cell within RETARGET_LIMIT of it
  fail       the nearest reachable cell is further than that on every solution

This matters at scene-pick time. The simulator's own success rule is 20% of the goal REGION's
sampled points becoming reachable, which says nothing about the marker: on the v1 600, 178 scenes
cleared the region bar while leaving the marker unreachable, one of them at 98 of 100 region points.
Those scenes can only ever end in retarget or strict-fail on the table, and finding that out after
building one by hand is the expensive way.

Method. Ring-sample candidate cells outward from the marker, sorted by distance, and hand the whole
list to `count_reachable_points`, whose second return value is the index of the FIRST reachable
point. One call per state gives the retarget distance directly, at the ring resolution, using the
simulator's own wavefront rather than a numpy restatement of it. Rings run past the 12.0 cm limit on
purpose, so a scene that misses by 1 cm is distinguishable from one that misses by 8.

  source env.ilab.sh
  python scripts/pipeline/audit_v2_marker_retarget.py --gallery $NAMO_SCRATCH/viz/real_scenes_all \
      --key $NAMO_SCRATCH/real_buildable/exh_combined_key.json --out marker_retarget_v2.csv
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline"))
sys.path.insert(0, os.path.join(REPO, "build_python"))
sys.path.insert(0, os.path.join(REPO, "python"))

import namo_rl  # noqa: E402
from measure_post_push_clearance import solutions_of, CFG_PATH  # noqa: E402

RETARGET_LIMIT = 0.120           # hardware executor's rule, see module docstring
RING_STEP = 0.005
RING_MAX = 0.160                 # past the limit so a near miss is legible
RING_ANGLES = 24

_KEY = {}


def _init(key_path):
    global _KEY
    _KEY = {os.path.realpath(k): v[0] for k, v in json.load(open(key_path)).items()}


def candidates(marker):
    """The marker, then cells on rings outward from it, sorted by distance."""
    pts = [[marker[0], marker[1]]]
    r = RING_STEP
    while r <= RING_MAX + 1e-9:
        for k in range(RING_ANGLES):
            a = 2 * math.pi * k / RING_ANGLES
            pts.append([marker[0] + r * math.cos(a), marker[1] + r * math.sin(a)])
        r += RING_STEP
    return pts


def _dist_of(idx, marker, pts):
    if idx < 0:
        return None
    p = pts[idx]
    return math.hypot(p[0] - marker[0], p[1] - marker[1])


def one(args):
    fname, card_path = args
    card = json.load(open(card_path))
    ep = _KEY.get(os.path.realpath(card["meta"]["xml"]))
    if ep is None:
        return None
    plans = solutions_of(card, ep)
    if not plans:
        return None

    obj = card["meta"]["object_id"]
    marker = tuple(card["scene"]["goal"][:2])
    pts = candidates(marker)
    env = namo_rl.RLEnvironment(card["meta"]["xml"], CFG_PATH, False)
    root = env.get_full_state()

    best = None                  # smallest retarget distance over solutions; 0.0 means strict
    n_strict = 0
    for plan in plans:
        env.set_full_state(root)
        mid = None
        for i, (edge, depth) in enumerate(plan):
            if mid is not None:
                env.set_full_state(mid)
            cur = env.get_observation()[f"{obj}_pose"]
            a = namo_rl.Action()
            a.object_id = obj
            a.x, a.y, a.theta = float(cur[0]), float(cur[1]), float(cur[2])
            a.edge_idx, a.depth = int(edge), int(depth)
            env.step(a)
            if i == 0:
                mid = env.get_full_state()
        d = _dist_of(env.count_reachable_points(pts)[1], marker, pts)
        if d is None:
            continue
        if d < 1e-9:
            n_strict += 1
        if best is None or d < best:
            best = d

    if best is None:
        verdict, cm = "fail", ""
    elif best < 1e-9:
        verdict, cm = "strict", 0.0
    elif best <= RETARGET_LIMIT + 1e-9:
        verdict, cm = "retarget", round(100 * best, 1)
    else:
        verdict, cm = "fail", round(100 * best, 1)

    m = card["meta"]
    return {"build_id_v2": "", "axis": "hmax2" if m["horizon"] == "2push" else "1push",
            "tier": m["tier"], "verdict": verdict, "retarget_cm": cm,
            "n_solutions": len(plans), "n_solutions_strict": n_strict,
            "solve_rate": m["solve_rate"], "xml": m["xml"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cards = os.path.join(args.gallery, "cards")
    jobs = [(f, os.path.join(cards, f)) for f in sorted(os.listdir(cards)) if f.endswith(".json")]

    t0, rows = time.time(), []
    with Pool(args.workers, initializer=_init, initargs=(args.key,)) as pool:
        for n, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            if r:
                rows.append(r)
            if n % 250 == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    rows.sort(key=lambda r: (r["axis"], r["tier"], r["verdict"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tally = Counter((r["axis"], r["tier"], r["verdict"]) for r in rows)
    for k in sorted(tally):
        print(f"  {k[0]:6s} {k[1]:5s} {k[2]:9s} {tally[k]}", file=sys.stderr)
    print(f"wrote {args.out}: {len(rows)} cards in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
