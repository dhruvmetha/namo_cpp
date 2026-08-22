#!/usr/bin/env python3
"""Pick N scenes per difficulty tier from a labelled real-buildable pool and write build sheets.

Tiers come from the exhaustive answer key, never from the `open_frac` / contact-count windows the
generator steered on. Those windows only decide what is worth simulating; the simulator decides
what a scene actually is.

TWO AXES, and they disagree enough to matter. On 981 labelled scenes the same episodes split
easy/med/hard/unsolvable as 336/366/78/206 by 1-push solve rate but 748/161/36/41 at hmax=2,
because a setup push rescues most of what one push cannot do. `--axis hmax2` is the default since
that is the search this project deploys (`label_keyhole1_hmax2_tier.py`):

    solve_rate_hmax2 = |valid_1push U valid_first_push| / |tried_1push|
    solve_rate_1push = |valid_1push| / |tried_1push|

with `eval_common.bin_of` cuts, hard < 0.05, med < 0.30, easy >= 0.30.

Selection inside a tier is SPREAD, not first-N. The scenes arrive grouped by generator seed and
band, so taking the head of the list would hand the hardware a hundred near-identical layouts. This
round-robins over (layout shape, blocker object, brick count) buckets so a tier exercises the whole
design space it has.

  python scripts/pipeline/select_real_scene_tiers.py --key <key.json> --pools <dir>... \\
      --out <dir> --per-tier 100 --axis hmax2
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

# this file lives at <repo>/scripts/pipeline/, so THREE dirnames reach the root.
# Two lands in scripts/, which made every remote `cd $REPO && source env.ilab.sh`
# fail at the && and produce a silent no-op launch with an empty collect.log.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline"))
from eval_common import bin_of  # noqa: E402
from gen_real_buildable_scenes import (MOVABLES, BRICK_HALF, Rect,  # noqa: E402
                                       start_is_placeable)


def load_sheets(pools):
    sheets = {}
    for pool in pools:
        for f in glob.glob(os.path.join(pool, "**", "build_sheets.json"), recursive=True):
            root = os.path.dirname(f)
            for s in json.load(open(f)):
                xml = os.path.realpath(os.path.join(root, s["scene_id"], "env.xml"))
                sheets[xml] = s
    return sheets


def tier_of(ep, axis):
    tried = len(ep["tried_1push"])
    if not tried:
        return "unsolvable", 0.0
    v1 = {tuple(x) for x in ep["valid_1push"]}
    if axis == "hmax2":
        good = v1 | {tuple(x) for x in ep["valid_first_push"]}
    else:
        good = v1
    if not good:
        return "unsolvable", 0.0
    rate = len(good) / tried
    return bin_of(rate), rate


def _bearing(size_cm, yaw_deg):
    """World bearing of the LONG side, CCW from +X, in [0,180).

    `size_cm` is [X extent, Y extent] in the item's own local frame, and which of those is the long
    one DIFFERS BY ITEM TYPE: a brick is 19.5 x 5.5 so its long side is local X, while every block
    is taller than wide (obj_1 is 7.0 x 15.0) so its long side is local Y. A raw yaw therefore means
    a different physical orientation on different rows of the same sheet, and a human placing both
    from it would set every block a quarter turn wrong. The hardware repo has the opposite
    convention again, its objects.yaml puts a brick's long side on local Y, so nothing is portable
    except the world bearing.
    """
    return round((yaw_deg + (0.0 if size_cm[0] >= size_cm[1] else 90.0)) % 180.0, 1)


ANGLE_CONVENTION = ("long_axis_bearing_deg = bearing of the item's LONG side in world, "
                    "counter-clockwise from +X, in [0,180). Place by this, not by yaw. yaw_deg is "
                    "the MuJoCo local-frame rotation and its meaning differs between bricks (long "
                    "side on local X) and blocks (long side on local Y).")


def normalize(sheet):
    """Backfill long_axis_bearing_deg onto older sheets, and REFUSE rather than guess.

    Every one of the four 90-degree errors this pipeline hit came from a consumer assuming a
    default for something the producer never stated. So this raises on a missing dimension instead
    of defaulting to a brick's 19.5 x 5.5, and raises on a stored bearing that disagrees with the
    one recomputed from (size_cm, yaw_deg) instead of trusting whichever it found first. A loud
    failure here costs a rerun; a quiet one costs a physical bar placed a quarter turn wrong and a
    hardware result nobody can trace.
    """
    conv = sheet.get("angle_convention")
    if conv is not None and conv != ANGLE_CONVENTION:
        raise ValueError(f"{sheet['scene_id']}: unrecognised angle_convention, refusing to guess "
                         f"what its yaw means:\n  {conv}")
    sheet["angle_convention"] = ANGLE_CONVENTION

    def fix(item, what):
        sz = item.get("size_cm")
        if sz is None:
            lo, sh_ = item.get("long_cm"), item.get("short_cm")
            if lo is None or sh_ is None:
                raise ValueError(f"{sheet['scene_id']}: {what} has neither size_cm nor "
                                 f"long_cm/short_cm; cannot tell which local axis is long")
            sz = [sh_, lo]
        want = _bearing(sz, item["yaw_deg"])
        have = item.get("long_axis_bearing_deg")
        if have is not None and abs((have - want + 90.0) % 180.0 - 90.0) > 0.15:
            raise ValueError(f"{sheet['scene_id']}: {what} stores bearing {have} but (size_cm={sz}, "
                             f"yaw={item['yaw_deg']}) gives {want}. One of them is wrong.")
        # Height must never come out blank. Older sheets predate the field, and a blank cell reads
        # as zero to whatever parses the CSV next, which is the same assume-a-default failure that
        # produced the brick-angle bugs. Bricks are 10.0; a block's height is knowable from its name.
        h = item.get("height_cm")
        if h is None:
            if what.startswith("brick"):
                h = 10.0
            else:
                obj = item.get("object")
                if obj not in MOVABLES:
                    raise ValueError(f"{sheet['scene_id']}: blocker '{obj}' has no height and is "
                                     f"not in MOVABLES; refusing to emit a blank height")
                h = round(MOVABLES[obj][2] * 200, 1)
        return dict(item, long_axis_bearing_deg=want,
                    long_cm=max(sz), short_cm=min(sz), height_cm=h)

    sheet["bricks"] = [fix(b, f"brick {i}") for i, b in enumerate(sheet["bricks"])]
    sheet["blocker"] = fix(sheet["blocker"], "blocker")
    return sheet


def _placeable(sheet):
    st = [Rect(b["center_cm"][0] / 100, b["center_cm"][1] / 100,
               b["size_cm"][0] / 200, b["size_cm"][1] / 200,
               math.radians(b["yaw_deg"]), b["marker_hint"], "brick")
          for b in sheet["bricks"]]
    bl = sheet["blocker"]
    hx, hy, _h = MOVABLES[bl["object"]]
    blk = Rect(bl["center_cm"][0] / 100, bl["center_cm"][1] / 100, hx, hy,
               math.radians(bl["yaw_deg"]), "blocker", "mov")
    return start_is_placeable(st, blk,
                              (sheet["robot_start_cm"][0] / 100, sheet["robot_start_cm"][1] / 100))


def spread(items, n):
    """Round-robin over (layout-ish shape, blocker, brick count) so a tier is not all one design."""
    buckets = collections.defaultdict(list)
    for it in items:
        sh = it["sheet"]
        key = (sh["n_bricks"], sh["blocker"]["object"], len(sh["bricks"]))
        buckets[key].append(it)
    for b in buckets.values():
        b.sort(key=lambda it: it["rate"])
    out, order = [], sorted(buckets)
    while len(out) < n and any(buckets[k] for k in order):
        for k in order:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop(0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True, help="build_2push_validset.py output")
    ap.add_argument("--pools", nargs="+", required=True, help="dirs holding build_sheets.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-tier", type=int, default=100)
    ap.add_argument("--axis", choices=("hmax2", "1push"), default="hmax2")
    args = ap.parse_args()

    key = json.load(open(args.key))
    sheets = load_sheets(args.pools)

    by_tier = collections.defaultdict(list)
    missing = unplaceable = 0
    for xml, eps in key.items():
        sh = sheets.get(os.path.realpath(xml))
        if sh is None:
            missing += 1
            continue
        # one episode per scene by construction (single hop, single blocker); if the collection
        # ever finds more, take the one on the goal boundary, which is the first recorded
        # A scene whose start the car cannot physically occupy wastes a build slot at the table.
        # The wavefront calls the start free because it inflates by max(hx,hy)=3.5, modelling the
        # robot as a disc; the physical 7x7 square's corners reach 4.95. 10 of the first 600
        # delivered scenes were unbuildable for exactly this reason.
        if not _placeable(sh):
            unplaceable += 1
            continue
        ep = eps[0]
        tier, rate = tier_of(ep, args.axis)
        by_tier[tier].append({"xml": xml, "sheet": sh, "rate": rate,
                              "n_tried": len(ep["tried_1push"]),
                              "n_valid_1push": len(ep["valid_1push"]),
                              "n_valid_first": len(ep["valid_first_push"])})

    print(f"key={len(key)} scenes, sheets matched={len(key) - missing}, unmatched={missing}, "
          f"dropped_unplaceable_start={unplaceable}")
    print("available per tier:", {t: len(v) for t, v in sorted(by_tier.items())})

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for tier in ("easy", "med", "hard"):
        picked = spread(by_tier.get(tier, []), args.per_tier)
        summary[tier] = len(picked)
        rows = []
        for i, it in enumerate(picked):
            sheet = normalize(dict(it["sheet"]))
            # Renumber bars from wall_10 and never emit wall_9. Older pools were written with a
            # generator that started at 9, and wall_9 is the legacy bar whose tag is mounted 90 deg
            # off with a compensating width/depth swap in objects.yaml. It is also 19.0 long, not
            # 19.5. Handing a build sheet that names it invites a wrongly-oriented brick.
            sheet["bricks"] = [dict(b, marker_hint=f"wall_{10 + k}")
                               for k, b in enumerate(sheet["bricks"])]
            sheet["tag_convention"] = ("every bar uses the wall_10/wall_11 ArUco mounting; "
                                       "wall_9 is excluded (tag rotated 90 deg, 19.0 cm not 19.5)")
            sheet.update(tier=tier, axis=args.axis, solve_rate=round(it["rate"], 4),
                         n_tried=it["n_tried"], n_valid_1push=it["n_valid_1push"],
                         n_valid_first_push=it["n_valid_first"], xml=it["xml"],
                         build_id=f"{tier}_{i:03d}")
            rows.append(sheet)
        with open(os.path.join(args.out, f"{tier}.json"), "w") as f:
            json.dump(rows, f, indent=2)
        with open(os.path.join(args.out, f"{tier}_manifest.txt"), "w") as f:
            f.write("\n".join(r["xml"] for r in rows) + "\n")
        short = 0 if len(picked) >= args.per_tier else args.per_tier - len(picked)
        note = f"  SHORT BY {short}" if short else ""
        print(f"{tier:5s}: wrote {len(picked)}{note}")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"axis": args.axis, "per_tier_requested": args.per_tier,
                   "selected": summary,
                   "available": {t: len(v) for t, v in by_tier.items()}}, f, indent=2)


if __name__ == "__main__":
    main()
