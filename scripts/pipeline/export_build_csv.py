#!/usr/bin/env python3
"""Flatten selected build sheets into one CSV per tier, for a human working at the table.

One row per PLACEABLE ITEM, with the per-scene fields repeated on every row. That denormalisation
is deliberate: a row has to be buildable on its own, and robot start and goal are per-scene, so a
row without them cannot be acted on. Repeating them costs bytes and removes a join, which is the
right trade for something a person reads next to a ruler.

Column choices that are not obvious:
  long_axis_bearing_deg  the ONLY orientation number to place by. `yaw_deg` is deliberately absent:
                         it is the MuJoCo local-frame rotation and means different things for a
                         brick (long side on local X) than for a block (long side on local Y), and
                         shipping both invites placing by the wrong one.
  n_bricks               so a 3-brick scene is visible before someone starts laying it out and
                         finds it needs wall_12.
  min_pairwise_sep_cm    NOT emitted. Only probe scenes carry it, and a blank cell reads as zero to
                         whatever parses this next. An absent column is honest; an empty one is not.

Rows sort by build_id then item, bricks before blocks.

  python scripts/pipeline/export_build_csv.py --sel-dir <deliver_hmax2> --out-dir <csv dir>
"""
import argparse
import csv
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline"))
from gen_real_buildable_scenes import Rect, contact_breakdown  # noqa: E402

COLUMNS = ["build_id", "tier", "axis", "item", "marker_hint", "centre_x_cm", "centre_y_cm",
           "long_axis_bearing_deg", "long_cm", "short_cm", "height_cm", "n_bricks",
           "robot_start_x_cm", "robot_start_y_cm", "robot_start_bearing_deg",
           "goal_x_cm", "goal_y_cm",
           "solve_rate", "tried", "valid_1push", "valid_first_push",
           "n_contacts_reachable", "n_contacts_cutoff", "n_contacts_collision",
           "angle_convention", "tag_convention"]


def _rect(item, name, kind):
    """Rebuild the world footprint from the sheet. size_cm is [X extent, Y extent] in the item's own
    local frame and yaw_deg rotates that frame, so the two together fix the footprint."""
    sz = item["size_cm"]
    return Rect(item["center_cm"][0] / 100.0, item["center_cm"][1] / 100.0,
                sz[0] / 200.0, sz[1] / 200.0, math.radians(item["yaw_deg"]), name, kind)


def checksum(sheet):
    """(reachable, cutoff, collision) on the blocker at t=0. See contact_breakdown for why."""
    statics = [_rect(b, b["marker_hint"], "brick") for b in sheet["bricks"]]
    blocker = _rect(sheet["blocker"], "obstacle_0_movable", "mov")
    start = (sheet["robot_start_cm"][0] / 100.0, sheet["robot_start_cm"][1] / 100.0)
    return contact_breakdown(statics, blocker, start)


def rows_for(sheet):
    reach, cut, coll = checksum(sheet)
    common = {
        "n_contacts_reachable": reach, "n_contacts_cutoff": cut, "n_contacts_collision": coll,
        "build_id": sheet["build_id"], "tier": sheet["tier"], "axis": sheet["axis"],
        "n_bricks": sheet["n_bricks"],
        "robot_start_x_cm": sheet["robot_start_cm"][0],
        "robot_start_y_cm": sheet["robot_start_cm"][1],
        "robot_start_bearing_deg": sheet["robot_start_bearing_deg"],
        "goal_x_cm": sheet["goal_cm"][0], "goal_y_cm": sheet["goal_cm"][1],
        "solve_rate": sheet["solve_rate"], "tried": sheet["n_tried"],
        "valid_1push": sheet["n_valid_1push"], "valid_first_push": sheet["n_valid_first_push"],
        "angle_convention": sheet["angle_convention"], "tag_convention": sheet["tag_convention"],
    }
    out = []
    for b in sheet["bricks"]:                      # bricks first: they define the passage
        out.append(dict(common, item="brick", marker_hint=b["marker_hint"],
                        centre_x_cm=b["center_cm"][0], centre_y_cm=b["center_cm"][1],
                        long_axis_bearing_deg=b["long_axis_bearing_deg"],
                        long_cm=b["long_cm"], short_cm=b["short_cm"], height_cm=b["height_cm"]))
    bl = sheet["blocker"]
    out.append(dict(common, item="block", marker_hint=bl["object"],
                    centre_x_cm=bl["center_cm"][0], centre_y_cm=bl["center_cm"][1],
                    long_axis_bearing_deg=bl["long_axis_bearing_deg"],
                    long_cm=bl["long_cm"], short_cm=bl["short_cm"], height_cm=bl["height_cm"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel-dir", required=True, help="a select_real_scene_tiers.py --out dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N scenes per tier (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for tier in ("easy", "med", "hard"):
        src = os.path.join(args.sel_dir, f"{tier}.json")
        if not os.path.exists(src):
            continue
        sheets = json.load(open(src))
        if args.limit:
            sheets = sheets[:args.limit]
        rows = []
        for s in sheets:
            rows += rows_for(s)
        rows.sort(key=lambda r: (r["build_id"], 0 if r["item"] == "brick" else 1,
                                 r.get("marker_hint", "")))
        path = os.path.join(args.out_dir, f"{tier}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"{tier:5s}: {len(sheets):3d} scenes, {len(rows):4d} rows -> {path}")


if __name__ == "__main__":
    main()
