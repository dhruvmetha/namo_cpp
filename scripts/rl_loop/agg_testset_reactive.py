#!/usr/bin/env python3
"""Aggregate eval_reactive_argmax `--leaf-out` jsonl(s) into open@k stratified by difficulty.

The reactive (policy-mode) eval gives the greedy open-rate with zero search: push i is the ranker's
argmax at the live state, simulated for real. This joins the per-episode leaf records with a tier
file by the (xml, object_id, region) key to stratify by easy / medium / hard + all.

Both canonical horizons are handled by the SAME path, because the two tier files disagree on how a
tier is stored:
  2push -> `division` field, already one of easy/medium/hard (pure2push_gt_divisions_*.json).
  1push -> no `division` field; the tier IS bin_of(solve_rate) over onepush_*.json, the same rule
           scripts/rl_loop/agg_search_eval.py uses, so the policy rows land in the tiers the
           registered search rows were binned into.

Anchor to reproduce (NoHz-v3 reactive, card Phase-0): open@2 = 40.7 all / 59.8 easy / 42.5 medium
/ 26.3 hard. Kill-signal-2 fires if hard open@2 < 35.
"""
import argparse
import glob
import json
from collections import defaultdict

from eval_common import bin_of


def _tier(rec):
    """easy/medium/hard for one record, from whichever field the tier file carries."""
    tier = rec.get("division") or bin_of(rec["solve_rate"])
    return "medium" if tier == "med" else tier          # bin_of says "med", the 2push file says "medium"


def load_divisions(divpath):
    d = json.load(open(divpath))
    return {(x, r["object_id"], r.get("region")): _tier(r) for x in d for r in d[x]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf-glob", required=True, help="glob of eval_reactive_argmax leaf jsonl shard(s)")
    ap.add_argument("--divisions", required=True,
                    help="tier file: a *_gt_divisions_*.json (2push) or onepush_*.json (1push)")
    ap.add_argument("--max-pushes", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    div = load_divisions(a.divisions)
    K = a.max_pushes
    bins = defaultdict(lambda: {"n": 0, "opened": [0] * (K + 1)})
    n_match = n_nomatch = 0
    for fp in sorted(glob.glob(a.leaf_glob)):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            dd = div.get((r["xml"], r["object_id"], r.get("region")))
            if dd is None:
                n_nomatch += 1
                continue
            n_match += 1
            oa = r["opened_at"]
            for tier in (dd, "all"):
                b = bins[tier]
                b["n"] += 1
                if 0 < oa <= K:
                    b["opened"][oa] += 1
    rep = {"n_match": n_match, "n_nomatch": n_nomatch, "max_pushes": K, "by_division": {}}
    for tier, b in bins.items():
        cum = {k: sum(b["opened"][1:k + 1]) for k in range(1, K + 1)}
        rep["by_division"][tier] = {"n": b["n"],
                                    **{f"open@{k}": round(100 * cum[k] / max(1, b["n"]), 1) for k in range(1, K + 1)}}
    # kill-signal-2 is defined on hard open@2 specifically, NOT on whatever K this run used.
    hard = rep["by_division"].get("hard", {})
    rep["kill_signal2_hard_open@2"] = {"value": hard.get("open@2"), "threshold": 35.0,
                                       "status": "PASS" if (hard.get("open@2") or 0) >= 35.0 else "FAIL"}
    json.dump(rep, open(a.out, "w"), indent=2)
    print(json.dumps(rep["by_division"], indent=2))
    print(f"matched {n_match}, unmatched {n_nomatch}; kill-signal-2 (hard open@2): {rep['kill_signal2_hard_open@2']}")


if __name__ == "__main__":
    main()
