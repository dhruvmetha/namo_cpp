#!/usr/bin/env python3
"""Gallery replays for exhaustively-labelled real-table scenes, by re-simulating the push.

`build_real_scene_replays.py` reads post-push poses out of the search collection's
`primitive_trial_log`. That only exists for scenes the collection actually ran, which is the 478 of
the first wave. The 1000 scenes of the second wave went straight through `exhaustive_hmax2.py`,
which stores a verdict per (edge, depth) and no geometry at all, so there is nothing to look up and
the push has to be executed here.

Same output schema, same `region_map()` recomputation, so the gallery cannot tell the two apart:
  1push cards, and hmax2 cards whose solution is a single push -- one step, an opener from `green`
  hmax2 cards needing a chain -- the setup, then the finish the sweep paired with it
    (`finish_for_setup` in the key, written by `exh_to_key.py`; without it there is no way to know
    which of the ~100 second pushes was the one that opened)

Choosing the push. `green` is sorted, so taking its head would animate edge 0 on nearly every card
and the gallery would look like one scene repeated. This picks the median entry instead, which is
arbitrary but spreads the choice across the contact ring.

TWO THINGS HAVE TO MATCH THE SWEEP OR A THIRD OF THE CHAINS FAIL TO REPRODUCE.

First, the sequence. The sweep saves the state after the setup and calls `set_full_state` before the
finish; running the two pushes straight through instead leaves a different state, and on 34 chains
that failed to reopen, restoring the way the sweep does recovered 12.

Second, the criterion. `region_map()` is a numpy region decomposition and it reports opened only
when the goal's label stops existing as a region of its own, which is a full merge. The label is the
looser `count_reachable_points >= ceil(0.2 * 100)` the simulator measured. On 40 dropped replays the
simulator said OPEN 29 times where `region_map` said shut, and the gap widens with difficulty (18%
of hmax2/hard against 5% of easy) because a hard scene opens a sliver rather than a doorway. So
`opened` comes from the simulator, and `merge` records what `region_map` thought, for the viewer.

A replay is dropped only when the simulator itself disagrees with its own recorded label.

  source env.ilab.sh
  python scripts/pipeline/build_real_scene_replays_exh.py --gallery $NAMO_SCRATCH/viz/real_scenes_all \
      --key $NAMO_SCRATCH/real_buildable/exh_combined_key.json --workers 32
"""
import argparse
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
from build_real_scene_replays import statics_from_card  # noqa: E402
from build_real_scene_cards import region_map  # noqa: E402
from gen_real_buildable_scenes import Rect  # noqa: E402

CFG_PATH = os.path.join(REPO, "config", "namo_config_complete_skill15_car_1x.yaml")
SOURCE = "re-simulated from exhaustive_hmax2 labels"
GOALS_PER_REGION = 100          # exhaustive_hmax2.py, and region_opening.py:344 behind it
MIN_FRACTION = 0.2
SNAPSHOT_SEED = 42

_KEY = {}


def _init(key_path):
    global _KEY
    _KEY = {os.path.realpath(k): v[0] for k, v in json.load(open(key_path)).items()}


def plan_for(card, ep):
    """-> [(edge, depth), ...], the pushes to animate, or None if this card has no solution."""
    green = [tuple(g) for g in card["green"]]
    if not green:
        return None
    openers = {tuple(x) for x in ep["valid_1push"]}
    single = sorted(g for g in green if g in openers)
    if single:
        return [single[len(single) // 2]]
    setups = sorted(g for g in green if g not in openers)
    if not setups:
        return None
    s = setups[len(setups) // 2]
    fin = ep.get("finish_for_setup", {}).get(f"{s[0]},{s[1]}")
    return None if fin is None else [s, tuple(fin)]


def one(args):
    fname, card_path, out_path = args
    card = json.load(open(card_path))
    ep = _KEY.get(os.path.realpath(card["meta"]["xml"]))
    if ep is None:
        return fname, "no_key_entry"
    plan = plan_for(card, ep)
    if plan is None:
        return fname, "no_solution"

    obj = card["meta"]["object_id"]
    statics = statics_from_card(card)
    mov = card["scene"]["movable"][0]
    start = tuple(card["scene"]["robot"][:2])
    goal = tuple(card["scene"]["goal"][:2])

    env = namo_rl.RLEnvironment(card["meta"]["xml"], CFG_PATH, False)
    bundle = dict(env.get_region_snapshot(GOALS_PER_REGION, -1.0, False, SNAPSHOT_SEED, True)
                  .get("region_goals", {})).get("goal")
    if bundle is None:
        return fname, "no_goal_region"
    pts = [[float(g.x), float(g.y)] for g in bundle.goals]
    bar = max(1, math.ceil(MIN_FRACTION * len(pts)))

    steps, mid = [], None
    for i, (edge, depth) in enumerate(plan):
        if mid is not None:
            env.set_full_state(mid)
        cur = env.get_observation()[f"{obj}_pose"]
        act = namo_rl.Action()
        act.object_id = obj
        act.x, act.y, act.theta = float(cur[0]), float(cur[1]), float(cur[2])
        act.edge_idx, act.depth = int(edge), int(depth)
        env.step(act)
        if i == 0:
            mid = env.get_full_state()
        px, py, pth = (round(float(v), 6) for v in env.get_observation()[f"{obj}_pose"])
        regions = region_map(statics, Rect(px, py, mov["hw"], mov["hd"], pth, mov["name"], "mov"),
                             start, goal)
        steps.append({"i": i + 1, "edge": int(edge), "depth": int(depth),
                      "opened": bool(env.count_reachable_points(pts)[0] >= bar),
                      "merge": "2" not in regions["labels"],
                      "geom": {"movable": {obj: [px, py, pth]}}, "regions": regions})

    if not steps[-1]["opened"]:
        return fname, "sim_disagrees_with_label"
    out = {"schema_version": 1, "key": fname, "source": SOURCE, "steps": steps}
    if not steps[-1]["merge"]:
        out["note"] = ("the simulator's 20%-of-region bar is met but the region decomposition still "
                       "shows two regions: this scene opens a sliver, not a doorway")
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return fname, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", required=True, help="dir holding cards/ and scenes.json")
    ap.add_argument("--key", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cards_dir = os.path.join(args.gallery, "cards")
    out_dir = os.path.join(args.gallery, "replay")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(f, os.path.join(cards_dir, f), os.path.join(out_dir, f))
            for f in sorted(os.listdir(cards_dir)) if f.endswith(".json")]

    t0, done, skip = time.time(), 0, Counter()
    with Pool(args.workers, initializer=_init, initargs=(args.key,)) as pool:
        for n, (fname, reason) in enumerate(pool.imap_unordered(one, jobs, chunksize=8), 1):
            if reason:
                skip[reason] += 1
            else:
                done += 1
            if n % 200 == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    print(f"wrote {done}/{len(jobs)} replays to {out_dir} in {time.time()-t0:.0f}s")
    if skip:
        print("  skipped:", dict(skip))


if __name__ == "__main__":
    main()
