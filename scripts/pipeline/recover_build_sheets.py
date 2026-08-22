#!/usr/bin/env python3
"""Rebuild build_sheets.json for scene dirs whose generator died before writing one.

The generator writes its sheet once, at the end of the run, so a generator killed part way leaves
its scenes on disk with no sheet at all. Those scenes then label fine and get dropped at selection
time as "unmatched", which is how 32 labelled scenes went missing from a 100-per-tier request. This
reconstructs the sheet from env.xml, which carries every number the sheet needs.

Recovered sheets are identical in content to generated ones except for `open_frac` and `n_contacts`,
which are search-time steering values rather than scene geometry and are not recoverable from the
XML. They are written as null so a consumer sees an explicit absence rather than a plausible zero.

  python scripts/pipeline/recover_build_sheets.py --pools <dir>...
"""
import argparse
import glob
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline"))
from gen_real_buildable_scenes import (ARENA_W, ARENA_H, MOVABLES,  # noqa: E402
                                       long_axis_bearing_deg)

GEOM = re.compile(r'<geom name="(?P<n>[^"]+)"[^>]*?pos="(?P<p>[^"]+)"[^>]*?'
                  r'euler="0 0 (?P<e>[-\d.]+)"[^>]*?size="(?P<s>[^"]+)"')
CAR = re.compile(r'<body name="car" pos="([^"]+)"')
GOAL = re.compile(r'<site name="goal"[^>]*?pos="([^"]+)"')


def sheet_from_xml(path, scene_id):
    txt = open(path).read()
    bricks, blocker = [], None
    for m in GEOM.finditer(txt):
        name, pos, yaw = m.group("n"), m.group("p").split(), float(m.group("e"))
        size = [float(v) for v in m.group("s").split()]
        cx, cy = float(pos[0]) * 100, float(pos[1]) * 100
        if name.startswith("wall_inner"):
            bricks.append({"marker_hint": f"wall_{10 + len(bricks)}",
                           "center_cm": [round(cx, 1), round(cy, 1)],
                           "long_axis_bearing_deg": long_axis_bearing_deg(size[0], size[1],
                                                                         math.radians(yaw)),
                           "yaw_deg": round(yaw % 180.0, 1),
                           "size_cm": [round(size[0] * 200, 1), round(size[1] * 200, 1)],
                           "long_cm": 19.5, "short_cm": 5.5, "height_cm": 10.0})
        elif "movable" in name:
            xy = [round(size[0] * 200, 1), round(size[1] * 200, 1)]
            obj = next((k for k, v in MOVABLES.items()
                        if abs(v[0] * 200 - xy[0]) < 0.2 and abs(v[1] * 200 - xy[1]) < 0.2), None)
            if obj is None:
                return None
            blocker = {"object": obj, "center_cm": [round(cx, 1), round(cy, 1)],
                       "long_axis_bearing_deg": long_axis_bearing_deg(size[0], size[1],
                                                                     math.radians(yaw)),
                       "yaw_deg": round(yaw % 180.0, 1), "size_cm": xy,
                       "long_cm": max(xy), "short_cm": min(xy),
                       "height_cm": round(size[2] * 200, 1)}
    car, goal = CAR.search(txt), GOAL.search(txt)
    if blocker is None or not bricks or not car or not goal:
        return None
    cs = [float(v) * 100 for v in car.group(1).split()]
    gs = [float(v) * 100 for v in goal.group(1).split()]
    return {"scene_id": scene_id, "arena_cm": [ARENA_W * 100, ARENA_H * 100],
            "bricks": bricks, "blocker": blocker,
            "robot_start_cm": [round(cs[0], 1), round(cs[1], 1)],
            "goal_cm": [round(gs[0], 1), round(gs[1], 1)],
            "run_namo_goal_flag": f"--goal {gs[0]:.0f} {gs[1]:.0f}",
            "n_bricks": len(bricks),
            # steering values, decided at sample time and absent from the XML by nature
            "open_frac": None, "n_contacts": None, "recovered_from_xml": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    args = ap.parse_args()
    made = skipped = failed = 0
    for pool in args.pools:
        for d in sorted(glob.glob(os.path.join(pool, "*"))):
            if not os.path.isdir(d):
                continue
            sheet_path = os.path.join(d, "build_sheets.json")
            have = set()
            if os.path.exists(sheet_path):
                have = {s["scene_id"] for s in json.load(open(sheet_path))}
            rows = json.load(open(sheet_path)) if have else []
            for scene in sorted(glob.glob(os.path.join(d, "rb_*"))):
                sid = os.path.basename(scene)
                if sid in have:
                    skipped += 1
                    continue
                s = sheet_from_xml(os.path.join(scene, "env.xml"), sid)
                if s is None:
                    failed += 1
                else:
                    rows.append(s)
                    made += 1
            if rows:
                with open(sheet_path, "w") as f:
                    json.dump(rows, f, indent=2)
    print(f"recovered={made} already_had={skipped} failed={failed}")


if __name__ == "__main__":
    main()
