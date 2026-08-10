#!/usr/bin/env python3
"""v2 1-push tiers. Rule (agg_search_eval._onepush_divisions): bin_of(episode solve_rate),
cuts hard <0.05 / medium <0.30 / easy >=0.30, keyed (canonical xml, object_id).

v1 stores that number as `solve_rate` in onepush_search_eval.json; the v2 rebuild stores the same
quantity as `solve_rate_1push` in twopush_all.json. Stage 1 proves those agree on v1 before any v2
number is believed.
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from namo import paths                                          # box-portable roots

VER = sys.argv[1] if len(sys.argv) > 1 else "v2"                      # target test-set version
LAB1 = f"{paths.DATASETS}/namo_testset_v1/labels"
LAB2 = f"{paths.DATASETS}/namo_testset_{VER}/labels"


def suf(p, n=5):
    return "/".join(p.rstrip("/").split("/")[-n:])


def bin_of(sr):
    return "hard" if sr < 0.05 else ("medium" if sr < 0.30 else "easy")


man1 = json.load(open(f"{LAB1}/onepush_search_eval.json"))
V1 = {(suf(x), e["object_id"]): float(e["solve_rate"]) for x, eps in man1.items() for e in eps}
tp1 = json.load(open(f"{LAB1}/twopush.json"))
S1 = {(suf(x), e["object_id"]): float(e["solve_rate_1push"]) for x, eps in tp1.items() for e in eps}

both = sorted(set(V1) & set(S1))
agree = sum(1 for k in both if abs(V1[k] - S1[k]) < 1e-9)
tier_agree = sum(1 for k in both if bin_of(V1[k]) == bin_of(S1[k]))
print("=== v1 GATE: solve_rate (manifest) vs solve_rate_1push (twopush.json) ===")
print(f"  manifest eps={len(V1)}  joined={len(both)}  exact value match={agree}  tier match={tier_agree}")
print(f"  registered v1 1push tiers: {dict(Counter(bin_of(v) for v in V1.values()))}")
if tier_agree != len(both):
    print("  GATE FAIL -- v2 numbers below are NOT trustworthy")

tp2 = json.load(open(f"{LAB2}/twopush_all.json"))
V2 = {(suf(x), e["object_id"]): float(e["solve_rate_1push"]) for x, eps in tp2.items() for e in eps}
common = [k for k in V1 if k in V2]
print(f"\n=== v2 1push tiers ===\n  v1 manifest eps={len(V1)}  covered by v2={len(common)}  missing={len(V1)-len(common)}")

flips = Counter()
for k in common:
    a, b = bin_of(V1[k]), bin_of(V2[k])
    if a != b:
        flips[(a, b)] += 1
n_f = sum(flips.values())
print(f"  FLIPS: {n_f} ({100*n_f/max(len(common),1):.1f}%)")
for (a, b), n in flips.most_common():
    print(f"    {a:>6} -> {b:<6} {n}")
print("  old totals (joined):", dict(Counter(bin_of(V1[k]) for k in common)))
print("  new totals (joined):", dict(Counter(bin_of(V2[k]) for k in common)))
print("  v2 totals (all v2 eps):", dict(Counter(bin_of(v) for v in V2.values())))

out = defaultdict(list)
for x, eps in tp2.items():
    for e in eps:
        out[x].append(dict(object_id=e["object_id"], region=e.get("region"),
                           solve_rate=float(e["solve_rate_1push"]),
                           division=bin_of(float(e["solve_rate_1push"])),
                           n_valid=len(e["valid_1push"]), n_tried=len(e["tried_1push"])))
dst = f"{LAB2}/onepush_divisions_{VER}.json"
json.dump(out, open(dst, "w"))
print(f"\n  wrote {dst}: rooms={len(out)} episodes={sum(len(v) for v in out.values())}")
