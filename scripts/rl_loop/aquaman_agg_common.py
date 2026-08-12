#!/usr/bin/env python3
"""Cross-campaign gate aggregation on the COMMON episode set [EXP-2026-08-09, 2026-08-12].

`aquaman_agg.py` scores each arm on whatever episodes its own shards contain. That is correct
within one campaign but NOT across campaigns: the family-corpus evals (2026-08-11/12) and the
arjuna/BNG-era evals ran different episode lists (measured: 32 episodes in AJ2 that HY5U never
evaluated, 12 the other way), so their solve@k denominators are different populations and the
numbers are not directly comparable.

This tool intersects the (xml, object_id, region) episode sets of EVERY arm requested, restricts
all arms to that common set, and only then bins by difficulty. Same tier functions and budgets as
`aquaman_agg.py` (imported, never re-defined). Reports the common-set size so any shrinkage is
visible rather than silent.

Usage:
  python aquaman_agg_common.py arms.json out.json
"""
import json
import sys
from pathlib import Path

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "aquaman_agg", str(Path(__file__).resolve().parent / "aquaman_agg.py"))
agg = importlib.util.module_from_spec(_spec)
sys.modules["aquaman_agg"] = agg
_spec.loader.exec_module(agg)


def key(r):
    return (agg.suf(str(r.get("xml"))), r.get("object_id"), r.get("region", "goal"))


def main():
    arms = json.load(open(sys.argv[1]))
    legs = ("1push", "2push")
    rows = {a: {} for a in arms}
    for a, spec in arms.items():
        for leg in legs:
            rows[a][leg] = agg.load(spec[leg]) if spec.get(leg) else []

    common = {}
    for leg in legs:
        sets = [set(map(key, rows[a][leg])) for a in arms if rows[a][leg]]
        common[leg] = set.intersection(*sets) if sets else set()
        sizes = {a: len(set(map(key, rows[a][leg]))) for a in arms if rows[a][leg]}
        print(f"[{leg}] per-arm episodes {sizes} -> common {len(common[leg])}", flush=True)

    out = {}
    for a in arms:
        out[a] = {}
        for leg, tier in (("1push", agg.tier_1p), ("2push", agg.tier_2p)):
            if not rows[a][leg]:
                continue
            sel = [r for r in rows[a][leg] if key(r) in common[leg]]
            out[a][leg] = agg.table(sel, tier)
    out["_common_episodes"] = {leg: len(common[leg]) for leg in legs}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
