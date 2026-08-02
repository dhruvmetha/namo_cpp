#!/usr/bin/env python3
"""Aquaman-0 gate aggregation — canonical-format solve@k tables from best-first shard JSONLs.

Matches the registered postprune aggregate shape: 1push tiers by FIXED solve_rate cuts
(hard<0.05 / medium<0.30 / easy) from onepush_search_eval.json; 2push tiers by `division`
in pure2push_gt_divisions_search_eval.json. Episode match: (xml, object_id [+region]).
"""
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np

LAB = Path("/common/users/dm1487/scratch_namo/datasets/namo_testset_v1/labels")
K1 = json.load(open(LAB / "onepush_search_eval.json"))
K2 = json.load(open(LAB / "pure2push_gt_divisions_search_eval.json"))
BUDGETS = [1, 2, 5, 10, 30, 100, 300, 900]


def suf(p, n=5):
    return "/".join(p.rstrip("/").split("/")[-n:])


K1s = {suf(k): v for k, v in K1.items()}
K2s = {suf(k): v for k, v in K2.items()}


def tier_1p(xml, obj, region):
    for e in K1s.get(suf(xml), []):
        if e["object_id"] == obj and (e.get("region", region) == region or len(K1s.get(suf(xml), [])) == 1):
            sr = len(e["valid"]) / max(len(e["tried"]), 1)
            return "hard" if sr < 0.05 else ("medium" if sr < 0.30 else "easy")
    return None


def tier_2p(xml, obj, region):
    for e in K2s.get(suf(xml), []):
        if e["object_id"] == obj and (e.get("region", region) == region or len(K2s.get(suf(xml), [])) == 1):
            return e["division"]
    return None


def load(dirs):
    rows = []
    for d in dirs:
        for f in glob(f"{d}/shard_*.jsonl"):
            for line in open(f):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def table(rows, tier_fn):
    out = {}
    binned = {}
    unmatched = 0
    for r in rows:
        t = tier_fn(r["xml"], r["object_id"], r.get("region", "goal"))
        if t is None:
            unmatched += 1
            continue
        binned.setdefault(t, []).append(r)
    for t in ["easy", "medium", "hard", "all"]:
        sel = sum(binned.values(), []) if t == "all" else binned.get(t, [])
        n = len(sel)
        if not n:
            continue
        solved_sims = [r["sims"] for r in sel if r["solved"]]
        out[t] = {"n": n,
                  **{f"solve@{b}": round(100 * sum(1 for r in sel if r["solved"] and r["sims"] <= b) / n, 1)
                     for b in BUDGETS},
                  "avg_sims_all": round(float(np.mean([r["sims"] for r in sel])), 1),
                  "avg_sims_to_solve": round(float(np.mean(solved_sims)), 1) if solved_sims else None}
    out["_unmatched"] = unmatched
    return out


def main():
    arms = json.load(open(sys.argv[1]))  # {name: {"1push": [dirs], "2push": [dirs]}}
    res = {}
    for name, legs in arms.items():
        res[name] = {}
        if legs.get("1push"):
            res[name]["1push"] = table(load(legs["1push"]), tier_1p)
        if legs.get("2push"):
            res[name]["2push"] = table(load(legs["2push"]), tier_2p)
        print(name, "done", flush=True)
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
