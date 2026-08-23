#!/usr/bin/env python3
"""Does the labelled push leave the robot able to reach the goal MARKER, not just open the region?

The sweep's `success` (region_opening.py:2982, bar at :3195) fires when at least
`ceil(region_min_reachable_fraction * n_sampled)` of the points sampled INSIDE the neighbour region
become reachable, default fraction 0.2. It says nothing about the one XML goal marker the build
sheet prints, and the two come apart: measured over the 600 shipped scenes, 178 have a labelled push
that clears the sweep's bar by a mile (98 of 100 sampled points, in one case) while leaving the
marker itself unreachable. The region opens; the marker sits in a pocket that does not.

Which one is right depends on what a hardware run counts as a win:
  push, then the region opens              -> the labels already say this, all 600 stand
  push, then drive to the marked goal      -> only the scenes in this file's `marker_reachable=1`

Filtering to `marker_reachable=1` is not the only way to take the strict reading, and it is the
worst one. All 178 failing scenes have reachable space inside the goal region after the push, with
the fraction of the region's sampled points reachable running min 0.20, median 0.89, max 1.00 (the
0.20 floor is just the sweep's own success bar). Moving each marker onto one of those points
recovers every scene. `solve_rate` comes from the region criterion, so a moved marker cannot shift
a scene between tiers either. Re-marking keeps all 600 and costs no re-collection and no re-binning;
filtering throws away 178 scenes for nothing.

So this writes the strict answer per scene beside the sheets rather than baking it into them. The
sheets keep the schema the hardware side already validated 600/600 against; joining is on build_id
plus axis.

  source env.ilab.sh
  python scripts/pipeline/audit_real_scene_marker.py \
      --cards $NAMO_SCRATCH/viz/real_scenes/cards \
      --out handoff/real_scene_build_sheets/marker_reachable.csv
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline"))
sys.path.insert(0, os.path.join(REPO, "build_python"))
sys.path.insert(0, os.path.join(REPO, "python"))

import build_real_scene_replays as B  # noqa: E402  -- reuse its sweep index + push picking
import namo_rl  # noqa: E402


def replay_and_ask(card, entry, env_cache):
    """Run the card's labelled solution, then ask whether the goal marker is reachable.

    Steps whose trial-log entry stored a post-push state get restored; a chain's finish push has no
    stored state and gets executed. Verified equivalent: on 42 scenes, executing the push from the
    start state agreed with restoring the stored state 42 times out of 42, so the sub-mm drift
    set_full_state is known for does not decide this.
    """
    plan = B.pick_plan(card, entry)
    if plan is None:
        return None
    xml = card["meta"]["xml"]
    env = env_cache.get(xml)
    if env is None:
        env = env_cache[xml] = namo_rl.RLEnvironment(xml, B.CFG_PATH, False)
    obj = entry["object_id"]
    for t in plan:
        if t["resulting_state"] is not None:
            rs = namo_rl.RLState()
            rs.qpos = list(t["resulting_state"]["qpos"])
            rs.qvel = list(t["resulting_state"]["qvel"])
            env.set_full_state(rs)
        else:
            cur = env.get_observation()[f"{obj}_pose"]
            a = namo_rl.Action()
            a.object_id = obj
            a.x, a.y, a.theta = float(cur[0]), float(cur[1]), float(cur[2])
            a.edge_idx, a.depth = int(t["edge_idx"]), int(t["depth"])
            env.step(a)
    # local_info_only is what region_opening passes; checked both ways on 165 cells, same verdict
    # every time, so the default here is not load-bearing.
    return bool(env.get_region_snapshot(0, -1.0, False, 42, True)["goal_reachable"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", default=os.path.join(os.environ.get("NAMO_SCRATCH", "/tmp"),
                                                    "viz", "real_scenes", "cards"))
    ap.add_argument("--sweep-root", default=os.path.join(os.environ.get("NAMO_SCRATCH", "/tmp"),
                                                         "real_buildable", "sweep"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cards = {f: json.load(open(os.path.join(args.cards, f)))
             for f in sorted(os.listdir(args.cards)) if f.endswith(".json")}
    idx = B.index_sweep(args.sweep_root, {B.RP(c["meta"]["xml"]) for c in cards.values()},
                        args.workers)

    rows, tally, env_cache = [], Counter(), {}
    for n, (fname, card) in enumerate(cards.items(), 1):
        m = card["meta"]
        entry = idx.get(B.RP(m["xml"]))
        reach = replay_and_ask(card, entry, env_cache) if entry else None
        rows.append({"build_id": m["key"], "axis": fname.split("__")[0], "tier": m["tier"],
                     "push_kind": m["push_kind"],
                     "marker_reachable": "" if reach is None else int(reach),
                     "solve_rate": m["solve_rate"], "xml": m["xml"]})
        tally[(fname.split("__")[0], m["tier"], reach)] += 1
        if n % 100 == 0:
            print(f"  {n}/{len(cards)}", file=sys.stderr, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Axis names come from the card filenames, which are `1push` / `hmax2` -- the same names the
    # sheet directories use, so the CSV joins straight onto them. (Do NOT hardcode `2push` here;
    # that is the card's meta.horizon spelling and matches nothing on disk.)
    for axis in sorted({r["axis"] for r in rows}):
        got = {t: f"{tally[(axis, t, True)]}/{tally[(axis, t, True)] + tally[(axis, t, False)]}"
               for t in ("easy", "med", "hard")}
        print(f"  {axis}: {got}", file=sys.stderr)
    ok = sum(1 for r in rows if r["marker_reachable"] == 1)
    print(f"wrote {args.out}: {ok} of {len(rows)} scenes reach the marker", file=sys.stderr)


if __name__ == "__main__":
    main()
