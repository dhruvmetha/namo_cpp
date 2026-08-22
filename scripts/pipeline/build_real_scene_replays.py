#!/usr/bin/env python3
"""Build gallery replay JSONs for the 600 real-table cards, from the exhaustive sweep itself.

`build_real_scene_cards.py` writes cards from `key_final.json` (a summary of `valid_1push` /
`valid_first_push` tuples only -- no post-push geometry). The gallery's step animation needs the
object's actual pose after each push, and that lives one level down, on the exhaustive sweep's
`primitive_trial_log` (`$NAMO_SCRATCH/real_buildable/sweep/**/*.pkl`, `episode_results[i]
["algorithm_stats"]["primitive_trial_log"]`), one entry per push tried, tagged `chain_depth`,
`edge_idx`, `depth`, `success`, and for chain_depth==2 also `parent_edge`/`parent_depth`. Every
entry -- success or not -- carries `resulting_state` (mjModel qpos/qvel after that push), so the
post-push object pose is a lookup (`namo_rl.RLEnvironment.set_full_state` + `get_observation`), not
a re-simulation, and a setup push that FAILED to open (a real, physically-executed push, just not
the opener) is exactly what a 2-push `needs_2_chain` first step is meant to show.

One replay per card:
  1push axis, and hmax2/one_push -- one step: a chain_depth==1, success==True entry drawn from the
    card's own `green` list (so the animated push is one of the dots the card already shows).
  hmax2/needs_2_chain -- two steps: a setup drawn from `green` (== `valid_first_push`, chain_depth==1,
    success==False by construction -- it does not open on its own) followed by whichever
    chain_depth==2 success entry names that setup as its parent.

Region colouring at each step is recomputed from scratch with `build_real_scene_cards.region_map()`
on the moved object's new pose (reusing the card's own static geometry) -- the same function, same
inflation radius, same robot/goal anchor points the card's start-state regions came from. `opened`
is read off that recomputation (whether the goal's label survives as a region distinct from the
robot's), not trusted from the trial log, so the two can be cross-checked against each other.

  source env.ilab.sh
  python scripts/pipeline/build_real_scene_replays.py --out $NAMO_SCRATCH/viz/real_scenes
"""
import argparse
import glob
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline"))
sys.path.insert(0, os.path.join(REPO, "build_python"))
sys.path.insert(0, os.path.join(REPO, "python"))

from build_real_scene_cards import region_map  # noqa: E402  -- the exact decomposition the cards use
from gen_real_buildable_scenes import Rect  # noqa: E402

import namo_rl  # noqa: E402

RP = os.path.realpath
CFG_PATH = os.path.join(REPO, "config", "namo_config_complete_skill15_car_1x.yaml")
SOURCE_NOTE = ("exhaustive real-table sweep primitive_trial_log (post-push resulting_state, "
               "no re-simulation)")


# ---------------------------------------------------------------------------
# Pass 1: scan the sweep once, pull out the goal-region trial log for every xml a card needs.
# ---------------------------------------------------------------------------
_WANT = None  # set per-worker by _init_worker


def _init_worker(want_xmls):
    global _WANT
    _WANT = want_xmls


def _scan_pkl(pkl_path):
    """-> [(realpath(xml), object_id, trimmed_log), ...] for episodes this run needs."""
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
    except Exception:
        return []
    out = []
    for ep in d.get("episode_results") or []:
        xml = ep.get("xml_file")
        if not xml:
            continue
        real = RP(xml)
        if real not in _WANT:
            continue
        st = ep.get("algorithm_stats") or {}
        if st.get("neighbour_region_label") not in (None, "goal"):
            continue  # cards are all meta.region == "goal"; skip the rare non-goal neighbour dump
        log = st.get("primitive_trial_log") or []
        if not log:
            continue
        # Trim to the fields this script reads -- resulting_state is the only bulky one, and it is
        # tiny per entry (qpos/qvel, ~44 floats), so this is mainly dropping unrelated dict keys.
        trimmed = [{"edge_idx": t.get("edge_idx"), "depth": t.get("depth"),
                    "chain_depth": t.get("chain_depth"), "success": bool(t.get("success")),
                    "parent_edge": t.get("parent_edge"), "parent_depth": t.get("parent_depth"),
                    "resulting_state": t.get("resulting_state")} for t in log]
        out.append((real, st.get("chosen_object_id"), trimmed))
    return out


def index_sweep(sweep_root, want_xmls, workers):
    pkls = [p for p in glob.glob(os.path.join(sweep_root, "**", "*.pkl"), recursive=True)
            if "collection_summary" not in os.path.basename(p)]
    print(f"scanning {len(pkls)} pkls under {sweep_root} for {len(want_xmls)} target xmls",
          file=sys.stderr)
    idx = {}
    t0 = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(want_xmls,)) as pool:
        for i, recs in enumerate(pool.imap_unordered(_scan_pkl, pkls, chunksize=4)):
            for real, obj, log in recs:
                if real not in idx:  # first "goal"-region match wins; duplicates are the same episode
                    idx[real] = {"object_id": obj, "log": log}
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(pkls)} pkls, {len(idx)}/{len(want_xmls)} xmls found, "
                      f"{el:.0f}s elapsed", file=sys.stderr)
    print(f"sweep scan done in {time.time()-t0:.0f}s: {len(idx)}/{len(want_xmls)} xmls matched",
          file=sys.stderr)
    return idx


# ---------------------------------------------------------------------------
# Pass 2: per card, pick a solution, decode post-push poses, recompute regions.
# ---------------------------------------------------------------------------
def _by_chain1(log):
    d = {}
    for t in log:
        if t["chain_depth"] == 1:
            d.setdefault((t["edge_idx"], t["depth"]), t)  # trial log is exhaustive; first wins ties
    return d


def _by_chain2_parent(log):
    d = defaultdict(list)
    for t in log:
        if t["chain_depth"] == 2 and t["parent_edge"] is not None:
            d[(t["parent_edge"], t["parent_depth"])].append(t)
    return d


def pick_plan(card, entry):
    """-> (list of trial-log entries to emit as steps) or None if this card has no usable solution."""
    green = [tuple(g) for g in card["green"]]
    d1 = _by_chain1(entry["log"])
    if card["meta"]["horizon"] == "1push" or card["meta"]["push_kind"] == "one_push":
        for e, d in green:
            t = d1.get((e, d))
            if t is not None and t["success"] and t["resulting_state"] is not None:
                return [t]
        # green is a union on the hmax2 axis; fall back to ANY chain_depth==1 success in the raw log
        for t in entry["log"]:
            if t["chain_depth"] == 1 and t["success"] and t["resulting_state"] is not None:
                return [t]
        return None

    # needs_2_chain: green here is valid_first_push -- setup cells that enable a depth-2 open.
    d2p = _by_chain2_parent(entry["log"])
    for pe, pd in green:
        setup = d1.get((pe, pd))
        if setup is None or setup["resulting_state"] is None:
            continue
        for finish in d2p.get((pe, pd), []):
            if finish["success"] and finish["resulting_state"] is not None:
                return [setup, finish]
    # fallback: any depth-2 success at all, walked back to its own parent
    for t in entry["log"]:
        if t["chain_depth"] == 2 and t["success"] and t["resulting_state"] is not None:
            setup = d1.get((t["parent_edge"], t["parent_depth"]))
            if setup is not None and setup["resulting_state"] is not None:
                return [setup, t]
    return None


def statics_from_card(card):
    out = []
    for s in card["scene"]["static"]:
        yaw = 2.0 * math.atan2(s["qz"], s["qw"])
        out.append(Rect(s["x"], s["y"], s["hw"], s["hd"], yaw, s["name"], "brick"))
    return out


def build_replay(card, entry, env_cache):
    plan = pick_plan(card, entry)
    if plan is None:
        return None, "no_usable_trial_log_entry"

    xml = card["meta"]["xml"]
    env = env_cache.get(xml)
    if env is None:
        env = namo_rl.RLEnvironment(xml, CFG_PATH, False)
        env_cache[xml] = env

    obj = entry["object_id"]
    statics = statics_from_card(card)
    mov = card["scene"]["movable"][0]
    hw, hd = mov["hw"], mov["hd"]
    start = tuple(card["scene"]["robot"][:2])
    goal = tuple(card["scene"]["goal"][:2])

    steps = []
    for i, t in enumerate(plan):
        rs = namo_rl.RLState()
        rs.qpos = list(t["resulting_state"]["qpos"])
        rs.qvel = list(t["resulting_state"]["qvel"])
        env.set_full_state(rs)
        obs = env.get_observation()
        px, py, ptheta = (round(float(v), 6) for v in obs[f"{obj}_pose"])

        blocker = Rect(px, py, hw, hd, ptheta, mov["name"], "mov")
        regions = region_map(statics, blocker, start, goal)
        opened = "2" not in regions["labels"]

        steps.append({"i": i + 1, "edge": int(t["edge_idx"]), "depth": int(t["depth"]),
                      "opened": bool(opened), "geom": {"movable": {obj: [px, py, ptheta]}},
                      "regions": regions})
    return steps, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.environ.get("NAMO_SCRATCH", "/tmp"),
                                                   "viz", "real_scenes"),
                    help="gallery data root; reads cards/, writes replay/")
    ap.add_argument("--sweep-root", default=os.path.join(os.environ.get("NAMO_SCRATCH", "/tmp"),
                                                          "real_buildable", "sweep"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cards_dir = os.path.join(args.out, "cards")
    files = sorted(f for f in os.listdir(cards_dir) if f.endswith(".json"))
    cards = {}
    for f in files:
        cards[f] = json.load(open(os.path.join(cards_dir, f)))
    print(f"{len(cards)} cards loaded from {cards_dir}", file=sys.stderr)

    want = {RP(c["meta"]["xml"]) for c in cards.values()}
    sweep_idx = index_sweep(args.sweep_root, want, args.workers)

    out_dir = os.path.join(args.out, "replay")
    os.makedirs(out_dir, exist_ok=True)

    env_cache = {}
    n_ok, skip_reasons = 0, defaultdict(int)
    for fname, card in cards.items():
        xml_real = RP(card["meta"]["xml"])
        entry = sweep_idx.get(xml_real)
        if entry is None:
            skip_reasons["xml_not_found_in_sweep"] += 1
            continue
        steps, reason = build_replay(card, entry, env_cache)
        if steps is None:
            skip_reasons[reason] += 1
            continue
        json.dump({"schema_version": 1, "key": fname, "source": SOURCE_NOTE, "steps": steps},
                  open(os.path.join(out_dir, fname), "w"), separators=(",", ":"))
        n_ok += 1

    print(f"wrote {n_ok}/{len(cards)} replays to {out_dir}", file=sys.stderr)
    if skip_reasons:
        print(f"skipped: {dict(skip_reasons)}", file=sys.stderr)


if __name__ == "__main__":
    main()
