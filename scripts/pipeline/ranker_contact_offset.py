#!/usr/bin/env python3
"""Does the deployed ranker prefer contacts near the middle of a block face?

Hardware measured that a real block yaws about 0.70 deg/cm per cm of contact offset on corner
pushes and barely at all near the face centre, while our sim self-squares at every offset and so
reproduces hardware ONLY for near-centre contacts. That raises a question with a nice answer either
way: if the learned ranker already prefers centre contacts, it is implicitly staying inside the
regime where our physics is right, and sim-trained ranking transfers better than it has any right
to. If it does not, the offset-coupling model becomes load-bearing for sim-based planning.

⚠ THE OBVIOUS VERSION OF THAT SENTENCE IS PROBABLY WRONG, WHICH IS WHY THIS MEASURES TWO THINGS.
Centre contacts also travel FURTHER in sim: sweeping one real-table scene's face gives 22.2 cm at
the corner rising to 25.5 cm near centre, because a corner push spends energy rotating. A ranker
trained to open regions would pick the longer push on geometry alone, and any offset preference
would fall out as a side effect with nothing to do with physics fidelity. So this records simulated
travel per candidate and reports the offset-score relationship both raw and after partialling travel
out. If the two are collinear the honest answer is that we cannot separate them, and that is a
result to report rather than a gap to paper over.

Offset is the distance of a contact from the centre of the face it sits on, from `contact_points`.
Indices 0-29 lie on the +-hy faces and run along local x, so offset maxes at hx; 30-59 lie on the
+-hx faces and run along local y, maxing at hy. Those two maxima differ (3.5 vs 7.5 cm for obj_1),
so `offset_frac` normalises by the face's own half-width and `offset_cm` keeps the absolute number.
The hardware measurements are absolute and all on the long face, so quote offset_cm against them.

  source env.ilab.sh
  python scripts/pipeline/ranker_contact_offset.py --ckpt <HY5U epoch011.ckpt> \
      --key $NAMO_SCRATCH/real_buildable/exh_combined_key.json --n 200 --out ranker_offset.csv
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("python", "build_python", os.path.join("scripts", "pipeline")):
    sys.path.insert(0, os.path.join(REPO, p))

from namo.rl_loop._bootstrap import ensure_paths  # noqa: E402
ensure_paths()
from namo.rl_loop.policy import Policy  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
import namo_rl  # noqa: E402
from gen_real_buildable_scenes import Rect, contact_points  # noqa: E402

CFG = os.path.join(REPO, "config", "namo_config_complete_skill15_car_1x.yaml")
FALLBACK_GOAL = (0.0, 0.0, 0.0)


def offsets_for(hx, hy):
    """-> {edge: (offset_cm, offset_frac, face)} for all 60 contacts of a block this size."""
    pts = contact_points(Rect(0.0, 0.0, hx, hy, 0.0, "b", "mov"))
    out = {}
    for i, (x, y) in enumerate(pts):
        if i < 30:
            out[i] = (abs(x) * 100, abs(x) / hx, "long" if hx < hy else "short")
        else:
            out[i] = (abs(y) * 100, abs(y) / hy, "short" if hx < hy else "long")
    return out


def block_half_extents(env, obj):
    """Half-extents in metres, read off the observation rather than assumed per object name."""
    o = env.get_observation()
    for k in (f"{obj}_size", f"{obj}_half_extents"):
        if k in o:
            v = list(o[k])
            return float(v[0]), float(v[1])
    return 0.035, 0.075          # obj_1, the only movable in the real-table pool


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = json.load(open(args.key))
    xmls = sorted(key)
    random.Random(args.seed).shuffle(xmls)

    pol = Policy(ckpt=args.ckpt, score_h=1)
    rows, done, t0, skipped = [], 0, time.time(), 0
    for xml in xmls:
        if done >= args.n:
            break
        ep = key[xml][0]
        obj = ep["object_id"]
        try:
            env = namo_rl.RLEnvironment(xml, CFG, False)
            goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal)
            env.get_reachable_objects()
            s0 = env.get_full_state()
            pool = pol.score_pool(env, goal, xml, s0, obj)
        except Exception:
            skipped += 1
            continue
        if len(pool) < 5:
            skipped += 1
            continue

        off = offsets_for(*block_half_extents(env, obj))
        cands = []
        for _o, g, sc in pool:
            e, d = int(g.edge_idx), int(g.depth)
            # travel this candidate produces in sim, the confound the docstring warns about
            env.set_full_state(s0)
            pre = env.get_observation()[f"{obj}_pose"]
            a = namo_rl.Action()
            a.object_id = obj
            a.x, a.y, a.theta = float(pre[0]), float(pre[1]), float(pre[2])
            a.edge_idx, a.depth = e, d
            env.step(a)
            post = env.get_observation()[f"{obj}_pose"]
            cands.append({"xml": xml, "edge": e, "depth": d, "score": float(sc),
                          "offset_cm": round(off[e][0], 2), "offset_frac": round(off[e][1], 4),
                          "face": off[e][2],
                          "travel_cm": round(100 * math.hypot(post[0] - pre[0], post[1] - pre[1]), 2),
                          "dyaw_deg": round((math.degrees(post[2] - pre[2]) + 180) % 360 - 180, 2)})
        cands.sort(key=lambda c: -c["score"])
        for r, c in enumerate(cands):
            c["rank"] = r
            c["n_cand"] = len(cands)
        rows += cands
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{args.n} scenes, {len(rows)} candidates, {time.time()-t0:.0f}s",
                  file=sys.stderr, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {done} scenes, {len(rows)} candidates, {skipped} skipped, "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
