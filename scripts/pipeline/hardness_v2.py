#!/usr/bin/env python3
"""Recompute exhaustive-GT setup hardness (the ONLY tier source) from node-level GT.

The tier stored in pure2push_gt_divisions_*.json is bin_of(n_setups_gt / |tried_1push|):
  n_setups_gt = distinct (parent_edge, parent_depth) over depth-2 nodes that MOVED the object
                (setup_moved != 0) and have n_win > 0   -- i.e. distinct setup pushes that work
  denominator = |tried_1push| from the twopush label json  (VERIFIED 1016/1016 on v1)
The json's own valid_first_push/tried_first_push ratio does NOT reproduce it (105/1016) --
that was the wrong formula; do not reintroduce it.

Run mode `v1` reproduces the registered v1 tiers and is the correctness gate for mode `v2`.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from namo import paths                                          # box-portable roots

MODE = sys.argv[1]
SCRATCH = str(paths.SCRATCH)
LAB1 = f"{paths.DATASETS}/namo_testset_v1/labels"
LAB2 = f"{paths.DATASETS}/namo_testset_v2/labels"
B = f"{SCRATCH}/curriculum2/beast/round2/testset_finish_gt"
CUTS = ((0.05, "hard"), (0.30, "medium"))


def suf(p, n=5):
    if isinstance(p, bytes):
        p = p.decode()
    return "/".join(p.rstrip("/").split("/")[-n:])


def bin_of(rate):
    for thr, name in CUTS:
        if rate < thr:
            return name
    return "easy"


def n_setups(paths):
    """(xml_suffix, object_id) -> count of distinct WORKING setup pushes."""
    acc = defaultdict(set)
    seen = set()
    for p in paths:
        with h5py.File(p, "r") as f:
            xml, obj = f["xml"][:], f["object_id"][:]
            nk, sm, nw = f["node_kind"][:], f["setup_moved"][:], f["n_win"][:]
            pe, pd = f["parent_edge"][:], f["parent_depth"][:]
            for i in range(len(xml)):
                k = (suf(xml[i]), obj[i].decode() if isinstance(obj[i], bytes) else str(obj[i]))
                seen.add(k)
                kind = nk[i].decode() if isinstance(nk[i], bytes) else str(nk[i])
                if kind != "depth2" or sm[i] == 0 or nw[i] <= 0:
                    continue
                acc[k].add((int(pe[i]), int(pd[i])))
    return {k: len(acc.get(k, ())) for k in seen}


def denominators(label_json):
    """(xml_suffix, object_id) -> |tried_1push|, plus the episode record."""
    d = json.load(open(label_json))
    out = {}
    for x, eps in d.items():
        for e in eps:
            out[(suf(x), e["object_id"])] = len(e["tried_1push"])
    return out


# --------------------------------------------------------------------- v1 gate
if MODE == "v1":
    ns = n_setups([f"{B}/h5/testset_gt.h5"])
    den = denominators(f"{LAB1}/twopush.json")
    div = json.load(open(f"{LAB1}/pure2push_gt_divisions_final35.json"))
    REG = {(suf(x), e["object_id"]): e for x, eps in div.items() for e in eps}

    c = Counter()
    for k, e in REG.items():
        if e.get("n_setups_gt") is None:
            c["null in registry"] += 1
            continue
        if k not in ns or k not in den or not den[k]:
            c["MISSING from h5/labels"] += 1
            continue
        c["checked"] += 1
        c["n_setups MATCH"] += (ns[k] == int(e["n_setups_gt"]))
        c["pct MATCH"] += (abs(100 * ns[k] / den[k] - float(e["setup_hardness_pct"])) < 0.01)
        c["tier MATCH"] += (bin_of(ns[k] / den[k]) == e["division"])
    print("=== v1 GATE: recomputed vs registered pure2push_gt_divisions_final35 ===")
    for k, v in c.most_common():
        print(f"  {k:<22} {v}")
    ok = c["tier MATCH"] == c["checked"] and c["checked"] > 1000
    print(f"\n  GATE {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

# --------------------------------------------------------------------- v2 tiers
ns = n_setups(sorted(glob.glob(f"{B}/h5_v2/shards/*.h5")))
den = denominators(f"{LAB2}/twopush_all.json")
div = json.load(open(f"{LAB1}/pure2push_gt_divisions_final35.json"))
OLD = {(suf(x), e["object_id"]): e for x, eps in div.items() for e in eps}

rows, missing = {}, 0
for k, d in den.items():
    if k not in ns or not d:
        continue
    rows[k] = dict(n_setups_gt=ns[k], setup_hardness_pct=round(100 * ns[k] / d, 3),
                   n_tried_1push=d, division=bin_of(ns[k] / d))
missing = sum(1 for k in OLD if k not in rows)

print(f"=== v2 tiers ===\n  episodes with v2 tier: {len(rows)}   v1 episodes with no v2 root: {missing}")
print("  v2 tier totals (all):", dict(Counter(r["division"] for r in rows.values())))

common = [k for k in OLD if k in rows and OLD[k].get("n_setups_gt") is not None]
flips = Counter()
for k in common:
    a, b = OLD[k]["division"], rows[k]["division"]
    if a != b:
        flips[(a, b)] += 1
n_f = sum(flips.values())
print(f"\n  joined vs v1: {len(common)}   FLIPS: {n_f} ({100*n_f/max(len(common),1):.1f}%)")
for (a, b), n in flips.most_common():
    print(f"    {a:>6} -> {b:<6} {n}")
print("  old totals (joined):", dict(Counter(OLD[k]["division"] for k in common)))
print("  new totals (joined):", dict(Counter(rows[k]["division"] for k in common)))

out = defaultdict(list)
full = json.load(open(f"{LAB2}/twopush_all.json"))
for x, eps in full.items():
    for e in eps:
        k = (suf(x), e["object_id"])
        if k in rows:
            out[x].append(dict(object_id=e["object_id"], region=e.get("region"),
                               division_source="exhaustive_gt_setup_density", **rows[k]))
dst = f"{LAB2}/pure2push_gt_divisions_v2.json"
json.dump(out, open(dst, "w"))
print(f"\n  wrote {dst}: rooms={len(out)} episodes={sum(len(v) for v in out.values())}")
