#!/usr/bin/env python3
"""Replay recorded setup+finish pairs on a FRESH env and report how many still open the region.

Run it before trusting a cell of `qualifying_manifest.json`, and before handing a labelled scene to
anything that will execute the push sequence from a clean start, the hardware build path included.

⛔ A miss here is NOT proof of a bad label. `set_full_state` restores qpos and qvel but not the
MuJoCo solver warmstart, so the same push from the same restored state behaves differently
depending on how many pushes ran before it, and the sweep produced its labels with hundreds of
pushes of history in the solver. Measured 2026-08-29 on the 26 hard/2push two-movable scenes: 21
reproduce cold, 5 do not, and replaying one of those 5 with the sweep's prior enumeration in front
of it opened the region at 97 of 100 points against a bar of 20 where the cold replay read 0. So
read the output as a FLOOR on how many labels are real.

Ruled out before settling on that, so nobody repeats them: run-to-run non-determinism (two identical
runs give identical results), a stale wavefront (`get_reachable_objects()` before counting moves
nothing), and near-threshold jitter (0 versus 97 points is not a 0.3 mm effect).

The hard tier misses most because those scenes carry ~3 working setups in ~95 enumerated pushes, so
nearly every recorded solution is the marginal one.

  python scripts/pipeline/replay_check_labels.py --manifest <qualifying_manifest.json> \
      --pool-root $NAMO_SCRATCH/real_buildable_2mov --cell hard_2push
"""
import argparse
import glob
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline"))
from exhaustive_hmax2 import CFG, MIN_FRACTION, goal_region_points, opens, push  # noqa: E402

import namo_rl  # noqa: E402


def sweep_index(pool_root):
    """xml -> its sweep record, over every pull dir under the pool."""
    out = {}
    for d in sorted(glob.glob(os.path.join(pool_root, "*_exh2_pull"))) + \
            sorted(glob.glob(os.path.join(pool_root, "*", "exh2"))):
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                rec = json.load(open(f))
            except Exception:
                continue
            out.setdefault(rec.get("xml", ""), rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pool-root", required=True)
    ap.add_argument("--cell", required=True, help="e.g. hard_2push")
    a = ap.parse_args()

    rows = json.load(open(a.manifest))["cells"][a.cell]
    sweeps = sweep_index(a.pool_root)
    ok = miss = nocontact = skipped = 0
    for r in rows:
        rec = sweeps.get(r["xml"])
        cand = [c for c in (rec or {}).get("cells", [])
                if c["kind"] == "setup" and c.get("finish")
                and (c.get("movable_collisions") or c.get("finish_movable_collisions"))]
        if not cand:
            skipped += 1
            continue
        c = cand[0]
        env = namo_rl.RLEnvironment(os.path.join(a.pool_root, r["relpath"]), CFG, False)
        pts = goal_region_points(env)
        if not pts:
            skipped += 1
            continue
        bar = max(1, math.ceil(MIN_FRACTION * len(pts)))
        root = env.get_full_state()
        env.set_full_state(root)
        r1 = push(env, c["object_id"], c["edge"], c["depth"])
        mid_open = opens(env, pts, bar)
        mid = env.get_full_state()
        o2, e2, d2 = c["finish"]
        env.set_full_state(mid)
        r2 = push(env, o2, e2, d2)
        got = env.count_reachable_points(pts)[0]
        i1 = r1.info if hasattr(r1, "info") else {}
        i2 = r2.info if hasattr(r2, "info") else {}
        touched = bool(i1.get("movable_collisions") or i2.get("movable_collisions"))
        if got >= bar and not mid_open and touched:
            ok += 1
        elif got >= bar and not mid_open:
            nocontact += 1
            print(f"  opens without contact on replay: {r['scene']}")
        else:
            miss += 1
            print(f"  cold replay misses: {r['scene']} mid_open={mid_open} reachable={got}/{bar}")
    print(f"\n{a.cell}: {ok} reproduce, {nocontact} open without contact, "
          f"{miss} missed cold, {skipped} skipped, {len(rows)} total "
          f"-- treat {ok} as a FLOOR, see this file's docstring")


main()
