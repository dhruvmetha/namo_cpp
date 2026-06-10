#!/usr/bin/env python3
"""Step-0 result: grade training-free first-push-value SCALARS against the exhaustive F1'.

Reads the per-first-push leaf dumps (fpv_shard*.jsonl from diag_leaf_s1.py: {xml,obj,edge1,depth1,
mean_top5,maxP,...}) and the gold 2-push key (labels/pure2push.json: per episode `valid_first_push`=F1').
For each episode, rank its candidate first-pushes by each scalar and compute recall@k vs F1' — i.e. does a
TRAINING-FREE lookahead value surface an enabling first-push into the top-k? Compared to the hypergeometric
random floor. If recall@k is high, a learned first-push policy/value may be unnecessary (the GATE for H1+).

Grades against the EXHAUSTIVE F1', NOT the beam's budget-limited good/dead label (that's the whole point of
having the canonical test set).
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from math import comb

SCALARS = ["mean_top5", "maxP", "mean_all", "frac_ge_099", "margin_top1_2"]
KS = [1, 3, 5, 10, 20]
RP = os.path.realpath


def floor_at_k(F, R, k):
    """P(>=1 of F enabling pushes among k distinct draws from R candidates), hypergeometric."""
    if F <= 0 or R <= 0:
        return 0.0
    if k >= R or R - F < k:
        return 1.0
    return 1.0 - comb(R - F, k) / comb(R, k)


def recall_at_k(cands, scalar, f1set, ks):
    """cands = list of (edge,depth,scalarvals). Rank by scalar desc; hit@k = any top-k cell in F1'."""
    order = sorted(cands, key=lambda c: c[2][scalar], reverse=True)
    cells = [(c[0], c[1]) for c in order]
    return {k: int(any(cell in f1set for cell in cells[:k])) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpv-dir", required=True, help="dir of fpv_shard*.jsonl")
    ap.add_argument("--pure2push", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pure = json.load(open(a.pure2push))
    f1 = {}                                   # (realpath, obj) -> set(F1' cells)
    for real, eps in pure.items():
        for e in eps:
            f1[(real, e["object_id"])] = {tuple(c) for c in e["valid_first_push"]}

    # group leaf records by (realpath, obj)
    byep = defaultdict(list)
    nrec = 0
    for jf in glob.glob(os.path.join(a.fpv_dir, "fpv_shard*.jsonl")):
        for ln in open(jf):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            nrec += 1
            key = (RP(r["xml"]), r["obj"])
            byep[key].append((r["edge1"], r["depth1"], r))

    # recall@k per scalar over episodes that (a) have F1' and (b) were swept
    agg = {s: {k: [] for k in KS} for s in SCALARS}
    floor = {k: [] for k in KS}
    n_ep = 0
    for key, cands in byep.items():
        f1set = f1.get(key)
        if not f1set:
            continue
        R = len(cands); F = sum(1 for (e, d, _) in cands if (e, d) in f1set)
        if F == 0:                            # none of the swept first-pushes is enabling (cap/coverage) — skip
            continue
        n_ep += 1
        for s in SCALARS:
            hit = recall_at_k(cands, s, f1set, KS)
            for k in KS:
                agg[s][k].append(hit[k])
        for k in KS:
            floor[k].append(floor_at_k(F, R, k))

    def mean(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else None

    out = {
        "n_episodes_graded": n_ep,
        "n_leaf_records": nrec,
        "recall_at_k_pct": {s: {k: mean(agg[s][k]) for k in KS} for s in SCALARS},
        "random_floor_pct": {k: mean(floor[k]) for k in KS},
        "note": "recall@k = a first-push in F1' (exhaustive enabling set) is in the top-k ranked by the scalar; "
                "graded only on episodes where >=1 enabling first-push was actually swept (first-cap coverage).",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
