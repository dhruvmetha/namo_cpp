#!/usr/bin/env python3
"""Export the real-table scenes as gallery cards, in the schema viz/search/gallery.js already reads.

`scripts/viz/build_scene_cards.py` in the namo-scene-gallery worktree does this job for the
canonical car_envs pools, but its inputs are the hardcoded onepush_v3 / pure2push label JSONs, so it
cannot be pointed at a different pool. This writes the same schema from our answer key instead.

Card, one per scene, matching what the gallery fetches from `cards/<file>`:
  scene    bounds, static bricks, the movable, robot pose, goal
  regions  the wavefront region decomposition as run-length pairs, plus a label -> name map
  contacts the 60 push poses
  green    the (edge, depth) pairs that solve it; tried is the denominator
  meta     horizon, object_id, region, density_pct, tier, counts, solve_rate, xml, key

Greens follow the same rule as the original: on the 1push axis a green is an opener (`valid_1push`),
on hmax2 it is anything that opens the region within two pushes (`valid_1push | valid_first_push`).

  python scripts/pipeline/build_real_scene_cards.py --out $NAMO_SCRATCH/viz/real_scenes
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline"))
from eval_common import bin_of  # noqa: E402
from gen_real_buildable_scenes import (ARENA_W, ARENA_H, BORDER_HALF, BRICK_HALF,  # noqa: E402
                                       GRID_RES, INFLATE_R, MOVABLES, Rect,
                                       ROBOT_START_BEARING_DEG, _blocked_mask, _cell,
                                       contact_points)

SHEETS = os.path.join(REPO, "handoff", "real_scene_build_sheets")
_S8 = np.ones((3, 3), dtype=int)


def rects(sheet):
    st = [Rect(b["center_cm"][0] / 100, b["center_cm"][1] / 100,
               b["size_cm"][0] / 200, b["size_cm"][1] / 200,
               math.radians(b["yaw_deg"]), b["marker_hint"], "brick")
          for b in sheet["bricks"]]
    bl = sheet["blocker"]
    hx, hy, _hz = MOVABLES[bl["object"]]
    blk = Rect(bl["center_cm"][0] / 100, bl["center_cm"][1] / 100, hx, hy,
               math.radians(bl["yaw_deg"]), "obstacle_0_movable", "mov")
    return st, blk


def region_map(statics, blocker, start, goal):
    """Label the free space, then name the robot's region and the goal's the way the gallery does.

    The gallery colours by label id and reads names from `labels`, so 1 must be the robot's region
    and 2 the goal's for the picture to mean the same thing it does on the existing pool.
    """
    blocked = _blocked_mask(statics + [blocker], INFLATE_R)
    lab, _n = ndimage.label(~blocked, structure=_S8)
    si, gi = _cell(start), _cell(goal)
    nx, ny = lab.shape
    rid = lambda c: int(lab[c]) if (0 <= c[0] < nx and 0 <= c[1] < ny) else 0
    robot_id, goal_id = rid(si), rid(gi)

    remap = {0: 0}
    names = {}
    nxt = 3
    for old in sorted(set(lab.flatten().tolist())):
        if old == 0:
            continue
        if old == robot_id:
            remap[old] = 1
            names["1"] = "robot"
        elif old == goal_id:
            remap[old] = 2
            names["2"] = "goal"
        else:
            remap[old] = nxt
            names[str(nxt)] = f"region_{nxt}"
            nxt += 1
    out = np.vectorize(remap.get)(lab).astype(int)

    # run-length pairs, column-major (x outer, y inner), matching the existing cards
    rle = []
    flat = out.reshape(-1)
    run_val, run_len = int(flat[0]), 0
    for v in flat:
        v = int(v)
        if v == run_val:
            run_len += 1
        else:
            rle += [run_val, run_len]
            run_val, run_len = v, 1
    rle += [run_val, run_len]
    return {"nx": int(nx), "ny": int(ny), "res": GRID_RES,
            "origin": [0.0, 0.0], "rle": rle, "labels": names}


def build(axis, sheets_by_key, key, out_dir, sheets_dir):
    os.makedirs(os.path.join(out_dir, "cards"), exist_ok=True)
    index, counts = [], Counter()
    for tier in ("easy", "med", "hard"):
        per = defaultdict(lambda: {"bricks": [], "block": None})
        for r in csv.DictReader(open(os.path.join(sheets_dir, axis, f"{tier}.csv"))):
            (per[r["build_id"]]["bricks"].append(r) if r["item"] == "brick"
             else per[r["build_id"]].__setitem__("block", r))
        for bid, d in per.items():
            b = d["block"]
            k = (round(float(b["centre_x_cm"]), 1), round(float(b["centre_y_cm"]), 1),
                 round(float(b["robot_start_x_cm"]), 1), round(float(b["robot_start_y_cm"]), 1))
            sheet, xml = sheets_by_key[k]
            ep = key[os.path.realpath(xml)][0]

            st, blk = rects(sheet)
            start = (sheet["robot_start_cm"][0] / 100, sheet["robot_start_cm"][1] / 100)
            goal = (sheet["goal_cm"][0] / 100, sheet["goal_cm"][1] / 100)

            v1 = [tuple(x) for x in ep["valid_1push"]]
            vf = [tuple(x) for x in ep["valid_first_push"]]
            green = sorted(set(v1) | set(vf)) if axis == "hmax2" else sorted(set(v1))
            tried = [list(x) for x in ep["tried_1push"]]
            density = 100.0 * len(green) / len(tried) if tried else 0.0

            card = {
                "schema_version": 1,
                "scene": {
                    "bounds": [-BORDER_HALF, ARENA_W + BORDER_HALF,
                               -BORDER_HALF, ARENA_H + BORDER_HALF],
                    "static": [{"name": r.name, "x": r.cx, "y": r.cy, "hw": r.hx, "hd": r.hy,
                                "qw": math.cos(r.yaw / 2), "qz": math.sin(r.yaw / 2)} for r in st],
                    "movable": [{"name": blk.name, "x": blk.cx, "y": blk.cy, "theta": blk.yaw,
                                 "hw": blk.hx, "hd": blk.hy}],
                    "robot": [start[0], start[1], math.radians(ROBOT_START_BEARING_DEG)],
                    "goal": [goal[0], goal[1], 0.0],
                },
                "regions": region_map(st, blk, start, goal),
                "contacts": [[round(p[0], 6), round(p[1], 6)] for p in contact_points(blk)],
                "green": [list(g) for g in green],
                "tried": tried,
                "meta": {
                    # gallery.js hardcodes these two values and branches on them, so the axis
                    # names have to be its names: our hmax2 axis IS the within-two-pushes one.
                    "horizon": "1push" if axis == "1push" else "2push",
                    "object_id": blk.name, "region": "goal",
                    "density_pct": round(density, 3), "tier": tier,
                    "n_green": len(green), "n_tried": len(tried),
                    "solve_rate": float(b["solve_rate"]),
                    "solve_rate_1push": (len(set(v1)) / len(tried)) if tried else 0.0,
                    "push_kind": b["push_kind"], "blocker": sheet["blocker"]["object"],
                    "n_bricks": sheet["n_bricks"],
                    "xml": xml, "key": bid,
                },
            }
            h = hashlib.md5(f"{axis}/{bid}".encode()).hexdigest()[:8]
            fname = f"{axis}__{bid}__{h}.json"
            with open(os.path.join(out_dir, "cards", fname), "w") as f:
                json.dump(card, f, separators=(",", ":"))
            # xml rides in the index row so a starred shortlist exports it. Three build-id
            # namespaces now name overlapping scene sets (v1 sheets, v2 sheets, this gallery's
            # own selection), and the xml path is the only join that is safe across all of
            # them. A shortlist without it identifies scenes only within one namespace.
            index.append({"file": fname, "scene": bid, "family": "real_table",
                          "horizon": card["meta"]["horizon"], "object_id": blk.name,
                          "tier": tier, "density_pct": f"{density:.3f}",
                          "n_green": str(len(green)), "n_tried": str(len(tried)),
                          "region": "goal", "xml": xml})
            counts[f"{axis}/{tier}"] += 1
    return index, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="gallery data root; gets cards/ and scenes.json")
    ap.add_argument("--key", default=os.path.join(os.environ.get("NAMO_SCRATCH", "/tmp"),
                                                  "real_buildable", "sweep", "key_final.json"))
    ap.add_argument("--pools", nargs="+",
                    default=[os.path.join(os.environ.get("NAMO_SCRATCH", "/tmp"), "real_buildable")])
    ap.add_argument("--axes", nargs="+", default=["hmax2", "1push"])
    # The shipped 600 live in the repo; a different pool (the full exhaustive set, say) points here
    # at its own export_build_csv.py output so the delivered sheets are never overwritten.
    ap.add_argument("--sheets", default=SHEETS, help="dir holding <axis>/<tier>.csv")
    args = ap.parse_args()

    key = {os.path.realpath(k): v for k, v in json.load(open(args.key)).items()}
    sheets_by_key, collisions = {}, 0
    for pool in args.pools:
        for root, _d, files in os.walk(pool):
            if "build_sheets.json" not in files:
                continue
            for s in json.load(open(os.path.join(root, "build_sheets.json"))):
                xml = os.path.join(root, s["scene_id"], "env.xml")
                if os.path.exists(xml):
                    k = (round(s["blocker"]["center_cm"][0], 1),
                         round(s["blocker"]["center_cm"][1], 1),
                         round(s["robot_start_cm"][0], 1),
                         round(s["robot_start_cm"][1], 1))
                    if k in sheets_by_key:
                        collisions += 1
                    sheets_by_key[k] = (s, xml)
    print(f"{len(sheets_by_key)} sheets indexed, {len(key)} scenes in the key, "
          f"{collisions} rounded-key collisions")

    all_index, all_counts = [], Counter()
    for axis in args.axes:
        idx, cnt = build(axis, sheets_by_key, key, args.out, args.sheets)
        all_index += idx
        all_counts.update(cnt)
        print(f"  {axis}: {len(idx)} cards")

    with open(os.path.join(args.out, "scenes.json"), "w") as f:
        json.dump({"schema_version": 1,
                   "counts": {"cards": len(all_index), "by_tier": dict(all_counts)},
                   "cards": all_index}, f, separators=(",", ":"))
    print(f"wrote {len(all_index)} cards + scenes.json to {args.out}")


if __name__ == "__main__":
    main()
