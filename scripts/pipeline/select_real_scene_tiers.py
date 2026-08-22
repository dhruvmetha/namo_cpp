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
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from eval_common import bin_of  # noqa: E402


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
    missing = 0
    for xml, eps in key.items():
        sh = sheets.get(os.path.realpath(xml))
        if sh is None:
            missing += 1
            continue
        # one episode per scene by construction (single hop, single blocker); if the collection
        # ever finds more, take the one on the goal boundary, which is the first recorded
        ep = eps[0]
        tier, rate = tier_of(ep, args.axis)
        by_tier[tier].append({"xml": xml, "sheet": sh, "rate": rate,
                              "n_tried": len(ep["tried_1push"]),
                              "n_valid_1push": len(ep["valid_1push"]),
                              "n_valid_first": len(ep["valid_first_push"])})

    print(f"key={len(key)} scenes, sheets matched={len(key) - missing}, unmatched={missing}")
    print("available per tier:", {t: len(v) for t, v in sorted(by_tier.items())})

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for tier in ("easy", "med", "hard"):
        picked = spread(by_tier.get(tier, []), args.per_tier)
        summary[tier] = len(picked)
        rows = []
        for i, it in enumerate(picked):
            sheet = dict(it["sheet"])
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
