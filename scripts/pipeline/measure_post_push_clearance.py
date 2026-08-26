#!/usr/bin/env python3
"""How wide is the gap the car has to drive through AFTER the push, on every real-table scene?

The generator's margin test (`gen_real_buildable_scenes.py:256`) asks whether the route is
comfortable with the movable DELETED, at `--margin-cm 10.0`, which works out to a 5.5 cm inflation
radius and so an 11.0 cm corridor. Deleting stands in for "the push gets the block out of the way".

It often does not. Success only requires 20% of the goal region's sampled points to become
reachable, not for the block to leave the doorway, so a scene can clear the margin test at the start
and still finish with the block parked in the gap. On `pool2/med_304/rb_00012` the winning chain
leaves a corridor between 7.0 and 8.0 cm, under the 8.0 cm the normal wavefront rule needs and far
under the 11.0 the margin test blessed. Nothing in the pipeline looks at that state: the gate runs
before any push is known, and the sweep that knows every push never re-checks the geometry.

So this measures it. For each solution, execute the pushes, then bisect the inflation radius at
which the robot stops being connected to the goal region. Twice that radius is the width of the
tightest point on the route.

WHERE THE TWO ENDS SIT IS THE WHOLE MEASUREMENT, and the obvious choices are both wrong.

Anchoring the start at the robot's POST-PUSH pose measures nothing about any corridor. The car
finishes a push in contact with the block it just moved, so its distance to the nearest obstacle is
whatever gap it stopped in: on a random 10 cards the number came back 7.2-8.8 cm and matched the
robot's own local clearance, not the route's, on 6 of them. Use the card's original robot start
instead. It is certified placeable by `start_is_placeable`, and after a successful push it is in the
merged region anyway, so it anchors the same connectivity question in a clean spot.

But the start anchor is not free of bias either, in the opposite direction. A push can leave the
robot on the far side of the block, roomy route in front of it, while the spot it came from is now
the cut-off one. On the tightest cards the two readings differ by 4x: 2.5 cm from the original start
against 10 cm from where the robot actually stands. So measure BOTH and keep the roomier. For the
robot anchor, exempt a 2 cm disc around it, or the contact it stopped in decides the answer instead
of the route.

Anchoring the far end at the XML goal MARKER is wrong for the same reason plus one more: success is
20% of the goal region's sampled points becoming reachable, and the marker itself can sit in a
pocket that never opens (2 of those same 10 cards had the marker inside an inflated obstacle, giving
a clearance of 0 for a scene that plainly opens). Use the region's sampled points and take the
roomiest one, which is the honest reading of "the robot can get into the goal region".

⚠ THE NUMBER IS A MEASUREMENT, NOT A VERDICT, AND IT IS DELIBERATELY NOT A FILTER. The threshold it
would be compared against comes from `compute_rotation_safe_robot_radius_m`, whose name says
diagonal and whose body returns max(hx, hy) = 3.5 cm. Believing that name once cost a whole scene
pool. Cutting scenes on a number that model may have wrong would throw away exactly the scenes where
sim and hardware are most likely to disagree, which are the informative ones. Sort the build order by
it, run the comfortable scenes first to test the planner, then run a few of the tightest on purpose
to find out what the real threshold is.

Resolution. `_blocked_mask` rasterises at GRID_RES = 5 mm, which quantises a clearance to 1 cm and is
too coarse for a question about millimetres of tag noise, so this rasterises at 2 mm locally rather
than changing the shared constant (a finer grid elsewhere can hide a connectivity bug).

Speed. Re-rasterising the scene at every bisection step is what the obvious version does and it does
not finish: 2 mm over the arena is 380k cells, and 9 steps x 8 solutions x 2226 cards of that is
hundreds of billions of operations. Instead rasterise ONCE at zero inflation, take a Euclidean
distance transform, and bisect on thresholds of that one field. "Connected when everything is
inflated by r" is exactly "connected through cells whose distance to the nearest obstacle exceeds
r", so each step is a threshold and a connected-components pass over an array already in memory.

  source env.ilab.sh
  python scripts/pipeline/measure_post_push_clearance.py --gallery $NAMO_SCRATCH/viz/real_scenes_all \
      --key $NAMO_SCRATCH/real_buildable/exh_combined_key.json --out clearance.csv --workers 32
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline"))
sys.path.insert(0, os.path.join(REPO, "build_python"))
sys.path.insert(0, os.path.join(REPO, "python"))

import namo_rl  # noqa: E402
from build_real_scene_replays import statics_from_card  # noqa: E402
from gen_real_buildable_scenes import ARENA_W, ARENA_H, Rect  # noqa: E402

GOALS_PER_REGION = 100          # matches exhaustive_hmax2.py, so the points are the labelled ones
SNAPSHOT_SEED = 42

CFG_PATH = os.path.join(REPO, "config", "namo_config_complete_skill15_car_1x.yaml")
RES = 0.002                      # 2 mm, finer than the shared GRID_RES on purpose
_S8 = np.ones((3, 3), dtype=int)
R_LO, R_HI, R_TOL = 0.005, 0.150, 0.0005
MAX_SOLUTIONS = 8                # per card, spread across the green list

_KEY = {}


def _init(key_path):
    global _KEY
    _KEY = {os.path.realpath(k): v[0] for k, v in json.load(open(key_path)).items()}


NX, NY = int(round(ARENA_W / RES)), int(round(ARENA_H / RES))
_GX, _GY = np.meshgrid((np.arange(NX) + 0.5) * RES, (np.arange(NY) + 0.5) * RES, indexing="ij")


def dist_field(rects):
    """Metres from each free cell to the nearest obstacle surface, arena border included."""
    occ = np.zeros((NX, NY), dtype=bool)
    for r in rects:
        c, s = math.cos(r.yaw), math.sin(r.yaw)
        dx, dy = _GX - r.cx, _GY - r.cy
        occ |= (np.abs(c * dx + s * dy) <= r.hx) & (np.abs(-s * dx + c * dy) <= r.hy)
    # One ring of occupied cells stands in for the arena walls, so the transform measures distance
    # to them too. Without it a route hugging a wall reads as infinitely roomy.
    pad = np.ones((NX + 2, NY + 2), dtype=bool)
    pad[1:-1, 1:-1] = occ
    return ndimage.distance_transform_edt(~pad)[1:-1, 1:-1] * RES


def _cells(pts):
    out = []
    for x, y in pts:
        i, j = int(x / RES), int(y / RES)
        if 0 <= i < NX and 0 <= j < NY:
            out.append((i, j))
    return out


def _exempt(dist, at):
    """Blank out a 2 cm disc so a pose that ends in contact does not decide its own clearance."""
    d = dist.copy()
    gx, gy = _GX, _GY
    d[((gx - at[0]) ** 2 + (gy - at[1]) ** 2) <= 0.02 ** 2] = R_HI * 2
    return d


def _connected_at(dist, r, si, targets):
    """Does any target share the start's free component when everything is inflated by r?"""
    free = dist > r
    if not free[si]:
        return False
    lab, _ = ndimage.label(free, structure=_S8)
    home = lab[si]
    return any(lab[t] == home for t in targets)


def clearance(rects, start, targets, exempt=False, dist=None):
    """Width of the tightest point on the roomiest route into the goal region, in metres.

    0.0 when no route reaches any target. The largest inflation radius that still connects the start
    to some target is half the width of the narrowest gap that route squeezes through.
    """
    si = (int(start[0] / RES), int(start[1] / RES))
    if not (0 <= si[0] < NX and 0 <= si[1] < NY) or not targets:
        return 0.0
    if dist is None:
        dist = dist_field(rects)
    if exempt:
        dist = _exempt(dist, start)
    if not _connected_at(dist, R_LO, si, targets):
        return 0.0
    if _connected_at(dist, R_HI, si, targets):
        return 2 * R_HI
    lo, hi = R_LO, R_HI
    while hi - lo > R_TOL:
        mid = 0.5 * (lo + hi)
        if _connected_at(dist, mid, si, targets):
            lo = mid
        else:
            hi = mid
    return 2 * lo


def solutions_of(card, ep):
    """Up to MAX_SOLUTIONS working plans, spread across the green list rather than taken from
    its head (green is sorted, so the head is edge 0 on nearly every card)."""
    green = [tuple(g) for g in card["green"]]
    openers = {tuple(x) for x in ep["valid_1push"]}
    fin = ep.get("finish_for_setup", {})
    plans = []
    for g in green:
        if g in openers:
            plans.append([g])
        else:
            f = fin.get(f"{g[0]},{g[1]}")
            if f is not None:
                plans.append([g, tuple(f)])
    if len(plans) <= MAX_SOLUTIONS:
        return plans
    step = len(plans) / MAX_SOLUTIONS
    return [plans[int(i * step)] for i in range(MAX_SOLUTIONS)]


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
    statics = statics_from_card(card)
    mov = card["scene"]["movable"][0]
    start = tuple(card["scene"]["robot"][:2])
    env = namo_rl.RLEnvironment(card["meta"]["xml"], CFG_PATH, False)
    bundle = dict(env.get_region_snapshot(GOALS_PER_REGION, -1.0, False, SNAPSHOT_SEED, True)
                  .get("region_goals", {})).get("goal")
    if bundle is None:
        return None
    targets = _cells([(float(g.x), float(g.y)) for g in bundle.goals])
    root = env.get_full_state()

    widths = []
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
        o = env.get_observation()
        p, rp = o[f"{obj}_pose"], o["robot_pose"]
        blk = Rect(p[0], p[1], mov["hw"], mov["hd"], p[2], mov["name"], "mov")
        rects = statics + [blk]
        d = dist_field(rects)
        widths.append(max(clearance(rects, start, targets, dist=d),
                          clearance(rects, (rp[0], rp[1]), targets, exempt=True, dist=d)))

    m = card["meta"]
    return {"file": fname, "horizon": m["horizon"], "tier": m["tier"], "key": m["key"],
            "n_solutions_measured": len(widths),
            "best_cm": round(100 * max(widths), 2),
            "median_cm": round(100 * sorted(widths)[len(widths) // 2], 2),
            "worst_cm": round(100 * min(widths), 2),
            "n_zero": sum(1 for w in widths if w == 0.0),
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

    rows.sort(key=lambda r: (r["horizon"], r["tier"], r["best_cm"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} cards in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
