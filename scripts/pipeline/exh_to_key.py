#!/usr/bin/env python3
"""Turn `exhaustive_hmax2.py` output into the answer-key schema the selector and card builder read.

`build_2push_validset.py` writes `{xml: [episode]}` where an episode carries three lists of
`[edge, depth]` pairs: `tried_1push` (the denominator), `valid_1push` (single pushes that open the
region) and `valid_first_push` (pushes that need a finish after them). The exhaustive sweep already
holds all three, one per cell, under different names:

    kind=opener  -> valid_1push          kind=setup  -> valid_first_push
    every cell   -> tried_1push          kind=dead/blocked -> denominator only

Writing the key instead of a bespoke format means `select_real_scene_tiers.py`,
`export_build_csv.py` and `build_real_scene_cards.py` all run unchanged on exhaustive labels.

PATHS. A sweep run on Amarel stores Amarel paths. `--remap FROM=TO` rewrites the prefix so the key
points at the CS copies the sheets and XMLs actually live at; without it every scene silently misses
its build sheet and the selector reports the whole pool as unmatched.

`--remap` takes the literal prefix the sweep recorded, so read it off the results rather than
copying one from here: `python -c 'import json;print(json.load(open(F))["xml"])'` on any result file
prints the box-local path the run stored, and the right-hand side is this box's `$NAMO_SCRATCH`.

  AMAREL_PREFIX=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["xml"])' \
                  "$(ls $NAMO_SCRATCH/exh2r2/out/*.json | head -1)")
  python scripts/pipeline/exh_to_key.py --out key.json \
      --dirs $NAMO_SCRATCH/exh2/out_good $NAMO_SCRATCH/exh2r2/out \
      --remap "${AMAREL_PREFIX%%/scenes/*}/scenes/=$NAMO_SCRATCH/real_buildable/"
"""
import argparse
import collections
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="exhaustive_hmax2.py --out dirs")
    ap.add_argument("--remap", action="append", default=[], help="FROM=TO path prefix rewrite")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rules = [r.split("=", 1) for r in args.remap]
    key, tally, dup, missing = {}, collections.Counter(), 0, 0
    for d in args.dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            s = json.load(open(f))
            xml = s["xml"]
            for a, b in rules:
                if xml.startswith(a):
                    xml = b + xml[len(a):]
            if not os.path.exists(xml):
                missing += 1
                continue
            if xml in key:
                dup += 1
                continue
            by, finish = collections.defaultdict(list), {}
            for c in s["cells"]:
                by[c["kind"]].append([c["edge"], c["depth"]])
                tally[c["kind"]] += 1
                # The finish the sweep actually found for this setup. Nothing in the key schema
                # carries it, but a replay of a 2-push scene needs it: these scenes were never run
                # through the search collection, so there is no trial log to look a finish up in.
                if c["kind"] == "setup":
                    finish[f'{c["edge"]},{c["depth"]}'] = c["finish"]
            key[xml] = [{
                "object_id": s["object_id"],
                "region": "goal",
                "tried_1push": [p for v in by.values() for p in v],
                "valid_1push": by["opener"],
                "valid_first_push": by["setup"],
                # provenance, ignored downstream but worth carrying: these labels are enumerated,
                # not searched, so `tried_1push` is every reachable (edge, depth) at the root.
                "source": "exhaustive_hmax2",
                "finish_for_setup": finish,
                "n_dead": len(by["dead"]),
                "n_blocked": len(by["blocked"]),
            }]

    with open(args.out, "w") as f:
        json.dump(key, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(key)} scenes, {sum(tally.values())} cells {dict(tally)}")
    if dup or missing:
        print(f"  skipped: {dup} duplicate xml, {missing} xml not on disk after remap")


if __name__ == "__main__":
    main()
