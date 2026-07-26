#!/usr/bin/env python3
"""RETIRED: the key-based hit@k this script aggregates is scored against `pure2push.json`'s `valid_first_push` list.
That list is badly incomplete — an exhaustive sweep over `testset_gt.h5` finds a median of 11 good first pushes
per episode where the manifest records only 4, because the manifest's finish search was budget-limited.
These key-based numbers are therefore a lower bound that understates the model and distorts the easy/medium/hard
tier comparison; treat them the same as the "conservative LB" caveat already noted below, just more so.
Canonical replacement: `scripts/eval_auc.py` over `testset_gt.h5` (see docs/experiments/auc_metrics_reconciliation.md).
The sim-grounded oracle-finish hit@k this script also aggregates is unaffected by this issue.

Aggregate the Phase-0 grey-zone k-sweep (phase0_ksweep.py leaves). setup-hit@k = a finishable setup appears in
the model's top-k setup ranking, k=1,2,4,8, split easy/med/hard/all, mean +/- std across 3 NoHz-v3 ckpt-seeds.
Reports sim-grounded (oracle-finish) AND key-based (GT valid_first_push, conservative LB) hit@k."""
import json, glob, os, statistics as st
from collections import defaultdict

from namo import eval_sets

D    = os.environ.get("KD", "/common/users/dm1487/scratch_namo/eval/phase0_ksweep")
OUT  = f"{D}/AGG"
BINS = ["easy", "medium", "hard", "all"]
KS   = [1, 2, 4, 8]
SEEDS = [1, 2, 3]


def load_seed(d):
    rows = []
    for f in glob.glob(f"{d}/shard_*.jsonl"):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def diffmap():
    d = json.load(open(eval_sets.DIVISIONS))
    return {(x, r["object_id"], r["region"]): r["division"] for x in d for r in d[x]}


def seed_hit(rows, dm, field):
    num = {k: defaultdict(int) for k in KS}; den = defaultdict(int)
    for r in rows:
        b = dm.get((r["xml"], r["object_id"], r["region"]))
        if b is None:
            continue
        at = r[field]
        for bb in (b, "all"):
            den[bb] += 1
            for k in KS:
                num[k][bb] += int(0 < at <= k)
    return {k: {b: (100.0 * num[k][b] / den[b] if den[b] else None) for b in BINS} for k in KS}, {b: den[b] for b in BINS}


def band(vals):
    vals = [v for v in vals if v is not None]
    return (st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0), len(vals)) if vals else None


def fmt(t):
    return f"{t[0]:5.1f} ± {t[1]:4.1f}" if t else "  —  "


def main():
    os.makedirs(OUT, exist_ok=True)
    dm = diffmap()
    persim = []; perkey = []; counts = {}
    for s in SEEDS:
        d = f"{D}/s{s}"
        if not os.path.isdir(d):
            continue
        rows = load_seed(d)
        if not rows:
            continue
        hs, counts = seed_hit(rows, dm, "setup_sim_at"); persim.append(hs)
        hk, _ = seed_hit(rows, dm, "setup_key_at"); perkey.append(hk)
    nseed = len(persim)

    def agg(per):
        return {k: {b: band([p[k][b] for p in per]) for b in BINS} for k in KS}
    aggsim = agg(persim); aggkey = agg(perkey)

    L = [f"# Phase-0 grey-zone k-sweep: setup-hit@k (finishable setup in model top-k) — {nseed} NoHz-v3 ckpt-seeds",
         f"episodes/bin: {counts}", "",
         "### setup-hit@k — SIM-GROUNDED (oracle finish; a setup counts if some 2nd push opens)",
         "| k (top setups) | easy | medium | hard | all |", "|---|---|---|---|---|"]
    for k in KS:
        L.append(f"| hit@{k} | " + " | ".join(fmt(aggsim[k][b]) for b in BINS) + " |")
    L += ["", "### setup-hit@k — KEY-BASED (GT valid_first_push; conservative lower bound)",
          "| k (top setups) | easy | medium | hard | all |", "|---|---|---|---|---|"]
    for k in KS:
        L.append(f"| hit@{k} | " + " | ".join(fmt(aggkey[k][b]) for b in BINS) + " |")
    table = "\n".join(L)
    open(f"{OUT}/table.md", "w").write(table)
    res = {"nseed": nseed, "counts": counts,
           "sim": {k: {b: (list(aggsim[k][b]) if aggsim[k][b] else None) for b in BINS} for k in KS},
           "key": {k: {b: (list(aggkey[k][b]) if aggkey[k][b] else None) for b in BINS} for k in KS}}
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    print(table)


if __name__ == "__main__":
    main()
