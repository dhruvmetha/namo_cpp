#!/usr/bin/env python3
"""Aggregate the 1push testset reactive eval into open@1/@2 by CANONICAL solve-rate TERTILES.

Reuses the STRICT POSITIONAL episode<->leaf join proven in scripts/sandbox/agg_1push_bottleneck.py:
the harness (eval_reactive_argmax on onepush_episodes.json) writes one leaf per episode in
full_episodes() order = sorted(xml) then per-xml record order, no skips. So gt[i] <-> leaf[i];
we assert len + object_id alignment (the hard gate). Difficulty = equal-count solve_rate tertiles
(hard = lowest third), deterministic from onepush_episodes.json — NOT bin_of fixed thresholds (the
predecessor's documented 1push-tier mistake). Asserts all == weighted tier mean.
"""
import argparse
import glob
import json
import re
from collections import defaultdict

from namo import eval_sets


def build_gt(onepush_key: str):
    k = json.load(open(onepush_key))
    pos = []
    for xml in sorted(k):
        for r in k[xml]:
            pos.append({"xml": xml, "object_id": r["object_id"], "solve_rate": r["solve_rate"]})
    srt = sorted(g["solve_rate"] for g in pos)
    n = len(srt)
    t1, t2 = srt[n // 3], srt[2 * n // 3]
    for g in pos:
        sr = g["solve_rate"]
        g["division"] = "hard" if sr < t1 else ("medium" if sr < t2 else "easy")
    return pos, (t1, t2, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf-glob", required=True, help="glob of eval_reactive_argmax leaf jsonl shard(s)")
    ap.add_argument("--onepush-key",
                    default=str(eval_sets.ONEPUSH))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gt, (t1, t2, n) = build_gt(a.onepush_key)
    files = sorted(glob.glob(a.leaf_glob),
                   key=lambda p: int(re.search(r"shard_(\d+)", p).group(1)))
    rows = []
    for f in files:
        for l in open(f):
            if l.strip():
                rows.append(json.loads(l))
    # KEYED join on (xml, object_id) — identity travels with each row, so shard count, order,
    # and filenames are irrelevant. (The old positional join broke when eval iteration order
    # differed from build_gt's sorted-xml order: 917 mismatches, 2026-07-08.)
    gtk = {(g["xml"], g["object_id"]): g for g in gt}
    assert len(gtk) == len(gt), "GT (xml, object_id) keys not unique — keyed join invalid"
    rk = {(r["xml"], r["object_id"]): r for r in rows}
    dup = len(rows) - len(rk)
    missing = sum(1 for k in gtk if k not in rk)
    extra = sum(1 for k in rk if k not in gtk)
    assert not dup and not missing and not extra, \
        f"KEYED JOIN INCOMPLETE: {dup} duplicate leaves, {missing} GT episodes missing, {extra} unmatched leaves ({len(files)} shards)"

    bins = defaultdict(lambda: {"n": 0, "o1": 0, "o2": 0})
    for k_, g in gtk.items():
        r = rk[k_]
        oa = r.get("opened_at", 0)
        for tier in (g["division"], "all"):
            b = bins[tier]
            b["n"] += 1
            if 0 < oa <= 1:
                b["o1"] += 1
            if 0 < oa <= 2:
                b["o2"] += 1
    rep = {"tertiles": {"hard_below": round(t1, 4), "med_below": round(t2, 4), "n": n},
           "by_division": {}}
    for tier, b in bins.items():
        rep["by_division"][tier] = {"n": b["n"],
                                    "open@1": round(100 * b["o1"] / max(1, b["n"]), 1),
                                    "open@2": round(100 * b["o2"] / max(1, b["n"]), 1)}
    tiers = [t for t in ("easy", "medium", "hard") if t in rep["by_division"]]
    denom = sum(rep["by_division"][t]["n"] for t in tiers)
    wm1 = sum(rep["by_division"][t]["open@1"] * rep["by_division"][t]["n"] for t in tiers) / max(1, denom)
    all1 = rep["by_division"]["all"]["open@1"]
    rep["all_eq_weighted_mean"] = {"all_open@1": all1, "weighted_mean_open@1": round(wm1, 2),
                                   "consistent": abs(all1 - wm1) < 0.2}
    json.dump(rep, open(a.out, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    assert rep["all_eq_weighted_mean"]["consistent"], "all != weighted tier mean — binning inconsistent"


if __name__ == "__main__":
    main()
