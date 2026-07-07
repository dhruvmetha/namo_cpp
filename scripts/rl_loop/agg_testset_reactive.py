#!/usr/bin/env python3
"""Aggregate eval_reactive_argmax `--leaf-out` jsonl(s) into open@k stratified by pure2push division.

The reactive testset eval (eval_reactive_argmax.py on namo_testset_v1/pure2push) gives the 2push
greedy reactive open-rate — the canonical 2push row + kill-signal-2 (hard-2push greedy). It reports
only an aggregate; this joins the per-episode leaf records with pure2push_divisions.json (by the
(xml, object_id, region) key, same as agg_phase0.py) to stratify by easy / medium / hard + all.

Anchor to reproduce (NoHz-v3 reactive, card Phase-0): open@2 = 40.7 all / 59.8 easy / 42.5 medium
/ 26.3 hard. Kill-signal-2 fires if hard open@2 < 35.
"""
import argparse
import glob
import json
from collections import defaultdict


def load_divisions(divpath):
    d = json.load(open(divpath))
    return {(x, r["object_id"], r.get("region")): r.get("division") for x in d for r in d[x]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf-glob", required=True, help="glob of eval_reactive_argmax leaf jsonl shard(s)")
    ap.add_argument("--divisions", required=True, help="pure2push_divisions.json")
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
    # kill-signal-2: hard open@2
    hard = rep["by_division"].get("hard", {})
    rep["kill_signal2_hard_open@2"] = {"value": hard.get(f"open@{K}"), "threshold": 35.0,
                                       "status": "PASS" if (hard.get(f"open@{K}") or 0) >= 35.0 else "FAIL"}
    json.dump(rep, open(a.out, "w"), indent=2)
    print(json.dumps(rep["by_division"], indent=2))
    print(f"matched {n_match}, unmatched {n_nomatch}; kill-signal-2 (hard open@{K}): {rep['kill_signal2_hard_open@2']}")


if __name__ == "__main__":
    main()
