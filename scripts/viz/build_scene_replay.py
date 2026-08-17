#!/usr/bin/env python3
"""Simulate each episode's ground-truth solution and dump the state after every push.

The gallery shows a card's START state. This adds the two states that follow it, so the page can
step start -> after push 1 -> after push 2 and show what "opening the region" actually looks like.

The sequence replayed is the TEST SET's own answer, not a planner's trace:
  1push  -- an opener from the manifest's `valid` list; one push, one following state.
  2push  -- a working setup from `valid_first_push`, then an opener read off the exhaustive GT for
            THAT setup's board (testset_gt_v3.h5, node_kind=depth2, matched on parent_edge/depth).
The campaign's per-episode rows record plan_len only, so the ranker's own push sequence is not
recoverable from them; replaying that needs eval_bestfirst.py --trace-out.

Each step is verified by the simulator: `opened` is env.is_robot_goal_reachable() after the push, so
a step that failed to do what the label says is visible rather than assumed.

    python scripts/viz/build_scene_replay.py --out $NAMO_SCRATCH/viz/scenes [--shard i --nshards n]
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts",
           f"{REPO}/scripts/sandbox", f"{REPO}/scripts/pipeline"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo import eval_sets, paths  # noqa: E402
from viz.trace_schema import rle_encode  # noqa: E402

SCHEMA_VERSION = 1


def suf(p, n=5):
    if isinstance(p, bytes):
        p = p.decode()
    return "/".join(str(p).rstrip("/").split("/")[-n:])


def finish_openers():
    """(xml suffix, object, parent_edge, parent_depth) -> list of (edge, depth) that OPEN there."""
    out = {}
    with h5py.File(paths.SCRATCH / eval_sets._CFG["files"]["twopush_gt_h5"], "r") as f:
        kind = f["node_kind"][:]
        rows = np.where(kind == b"depth2")[0]
        xml, obj = f["xml"][:], f["object_id"][:]
        pe, pd = f["parent_edge"][:], f["parent_depth"][:]
        # Read the whole grid in one go: 104k x 60 x 5 float32 is ~125 MB, while pulling the same
        # rows one at a time is 79k random reads and costs minutes per process.
        vt = f["value_target"][:]
        for i in rows:
            wins = np.argwhere(vt[i] == 1.0)
            if not len(wins):
                continue
            k = (suf(xml[i]), obj[i].decode() if isinstance(obj[i], bytes) else str(obj[i]),
                 int(pe[i]), int(pd[i]))
            out[k] = [(int(e), int(d)) for e, d in wins]
    return out


def snapshot(env, exporter, xml, target, offsets_world, mov_names, cfg):
    """Geometry + region decomposition at the env's CURRENT state, in the card's shapes."""
    obs = env.get_observation()
    info = env.get_object_info()
    opose = obs[f"{target}_pose"]
    off = offsets_world(info[target]["size_x"], info[target]["size_y"], float(opose[2]))
    geom = {"movable": {m: [round(float(c), 6) for c in obs[f"{m}_pose"]] for m in mov_names},
            "robot": [round(float(c), 6) for c in obs["robot_pose"]],
            "contacts": [[round(float(opose[0] + dx), 6), round(float(opose[1] + dy), 6)]
                         for dx, dy in off]}
    snap = exporter.build_snapshot(xml_path=str(paths.resolve(xml)), config_path=cfg,
                                   use_current_state=True)
    rm = snap.region_map
    regions = {"nx": int(rm.shape[0]), "ny": int(rm.shape[1]), "res": float(snap.resolution),
               "origin": [float(snap.bounds[0]), float(snap.bounds[2])],
               "labels": {str(int(k)): v for k, v in snap.region_labels.items()},
               "rle": rle_encode(rm.tolist())}
    return geom, regions


def push(env, prim, make_action, obj, edge, depth):
    """Apply the primitive push (obj, edge, depth). False if that goal does not exist here."""
    state = env.get_full_state()
    for edge_goals in prim.generate_goals(obj, state, env, max_goals=0):
        for g in edge_goals:
            if g is not None and int(g.edge_idx) == edge and int(g.depth) == depth:
                env.step(make_action(obj, g))
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="gallery data root (holds scenes.json + cards/)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tiers", default="", help="comma list, e.g. medium,hard -- easy is the bulk "
                                                "of the population and the least interesting")
    a = ap.parse_args()

    from add_contact_px import contact_offsets_world
    from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter
    from namo.core.xml_goal_parser import extract_goal_with_fallback
    from namo.strategies import PrimitiveGoalStrategy
    from scorer_beam import CFG, DATA_DIR, FALLBACK_GOAL, PRIM_PREFIX, make_env, make_action

    index = json.load(open(os.path.join(a.out, "scenes.json")))
    want = set(t for t in a.tiers.split(",") if t)
    by_xml = defaultdict(list)
    for row in index["cards"]:
        if want and row["tier"] not in want:
            continue
        card = json.load(open(os.path.join(a.out, "cards", row["file"])))
        by_xml[card["meta"]["xml"]].append((row, card))

    fin = finish_openers()
    prim = PrimitiveGoalStrategy(data_dir=DATA_DIR, primitive_prefix=PRIM_PREFIX)
    outdir = os.path.join(a.out, "replay")
    os.makedirs(outdir, exist_ok=True)

    xmls = sorted(by_xml)
    mine = [x for i, x in enumerate(xmls) if i % a.nshards == a.shard]
    t0, n_ok, n_skip = time.time(), 0, 0
    for i, xml in enumerate(mine):
        goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
        for row, card in by_xml[xml]:
            target = card["meta"]["object_id"]
            green = [tuple(p) for p in card["green"]]
            if not green:
                n_skip += 1
                continue
            # Try every recorded setup, not just the first: a given setup may have no finish in the
            # exhaustive GT, or its primitive goal may not exist at this state, and the episode still
            # has a perfectly good solution through one of the others.
            plans = []
            if row["horizon"] == "2push":
                for s0 in green:
                    for f0 in fin.get((suf(xml), target, s0[0], s0[1]), []):
                        plans.append([s0, f0])
            else:
                plans = [[g] for g in green]
            if not plans:
                n_skip += 1
                continue

            if os.path.exists(os.path.join(outdir, row["file"])):
                continue
            steps = []
            for plan in plans[:8]:          # a few attempts, not the whole cross product
                env = make_env(xml)
                env.set_robot_goal(*goal)
                env.get_reachable_objects()
                mov = [k for k, v in env.get_object_info().items()
                       if k != "robot" and "pos_x" not in v]
                steps, failed = [], False
                for step_i, (edge, depth) in enumerate(plan):
                    if not push(env, prim, make_action, target, edge, depth):
                        failed = True
                        break
                    geom, regions = snapshot(env, WavefrontSnapshotExporter(env), xml, target,
                                             contact_offsets_world, mov, CFG)
                    steps.append({"i": step_i + 1, "edge": edge, "depth": depth,
                                  "geom": geom, "regions": regions,
                                  "opened": bool(env.is_robot_goal_reachable())})
                # Keep the first plan that runs AND ends open -- a replay whose last frame is still
                # blocked would show the reader a non-solution.
                if not failed and steps and steps[-1]["opened"]:
                    break
                steps = []
            if not steps:
                n_skip += 1
                continue
            json.dump({"schema_version": SCHEMA_VERSION, "key": row["file"],
                       "source": "ground-truth solution (manifest opener / setup + exhaustive-GT finish)",
                       "steps": steps},
                      open(os.path.join(outdir, row["file"]), "w"))
            n_ok += 1
        if (i + 1) % 20 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"shard {a.shard}: {i+1}/{len(mine)} rooms, {n_ok} replays, {n_skip} skipped, "
                  f"eta {(len(mine)-i-1)/r/60:.1f} min", flush=True)
    print(f"shard {a.shard}: DONE {n_ok} replays, {n_skip} skipped, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
