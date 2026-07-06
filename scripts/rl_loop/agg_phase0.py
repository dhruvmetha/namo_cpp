#!/usr/bin/env python3
"""Aggregate Phase-0 GATE (EXP-2026-07-06-rl-only-self-imitation). CAR, pure2push, greedy protocol, split
easy/med/hard/all, mean +/- std across 3 NoHz-v3 ckpt-seeds. 2push-only (horizon split is trivially 2push).

Reads {D}/s{1,2,3}/shard_*.jsonl (phase0_oracle_decomp.py leaves). Metrics per bin per seed:
  BASELINE open@2   = mean(base_open)                      [ANCHOR: must match reactive-MPC 40.7 all / 59.8/42.5/26.3]
  ARM (i)-any       = mean(armi_any)         oracle setup + learned finish, opens for >=1 GT-valid setup (GATE number)
  ARM (i)-modelpref = mean(armi_modelpref)   " for the top-model-scored GT-valid setup (realistic point estimate)
  ARM (i)-mean      = mean(armi_open/armi_tried)          setup-averaged finish rate
  ARM (ii)          = mean(armii_recoverable)             learned setup lands on a GT-valid setup (key, conservative LB)
  ARM (ii)'sim      = mean(base_open or finish_exists)    sim-grounded: learned setup lands where SOME 2nd push opens
  ARM (iii) taxonomy over FAILED episodes (base_open==0): wrong_setup / failed_finish / aliasing_or_control (%)
Gate: ARM (i)-any >=85 => proceed; <65 => stop/rethink; grey => teacher-forced k-sweep."""
import json, glob, os, statistics as st
from collections import defaultdict

D    = os.environ.get("D", "/common/users/dm1487/scratch_namo/eval/phase0_gate")
LAB  = "/common/users/dm1487/scratch_namo/datasets/namo_testset_v1/labels"
OUT  = f"{D}/AGG"
BINS = ["easy", "medium", "hard", "all"]
SEEDS = [1, 2, 3]
MISS = ["wrong_setup", "failed_finish", "aliasing_or_control"]


def load_seed(d):
    rows = []
    for f in glob.glob(f"{d}/shard_*.jsonl"):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def diffmap():
    d = json.load(open(f"{LAB}/pure2push_divisions.json"))
    return {(x, r["object_id"], r["region"]): r["division"] for x in d for r in d[x]}


def seed_metrics(rows, dm):
    """one seed -> {metric: {bin: rate%}} + counts. Also miss taxonomy over failures."""
    num = defaultdict(lambda: defaultdict(float)); den = defaultdict(lambda: defaultdict(int))
    missnum = defaultdict(lambda: defaultdict(int)); faildens = defaultdict(int); alln = defaultdict(int)
    for r in rows:
        b = dm.get((r["xml"], r["object_id"], r["region"]))
        if b is None:
            continue
        bins = [b, "all"]
        recov_sim = 1 if (r["base_open"] or r.get("finish_exists")) else 0
        armi_mean = (r["armi_open"] / r["armi_tried"]) if r["armi_tried"] else None
        for bb in bins:
            alln[bb] += 1
            den["base_open"][bb] += 1;        num["base_open"][bb] += r["base_open"]
            den["armi_any"][bb] += 1;         num["armi_any"][bb] += r["armi_any"]
            den["armii"][bb] += 1;            num["armii"][bb] += r["armii_recoverable"]
            den["armii_sim"][bb] += 1;        num["armii_sim"][bb] += recov_sim
            if r.get("armi_modelpref") is not None:
                den["armi_modelpref"][bb] += 1; num["armi_modelpref"][bb] += r["armi_modelpref"]
            if armi_mean is not None:
                den["armi_mean"][bb] += 1;    num["armi_mean"][bb] += armi_mean
            if r["base_open"] == 0:
                faildens[bb] += 1
                if r["miss"] in MISS:
                    missnum[bb][r["miss"]] += 1
    metrics = {}
    for m in ["base_open", "armi_any", "armi_modelpref", "armi_mean", "armii", "armii_sim"]:
        metrics[m] = {b: (100.0 * num[m][b] / den[m][b] if den[m][b] else None) for b in BINS}
    # taxonomy: share of FAILED episodes, and share of ALL episodes
    tax_fail = {b: {mm: (100.0 * missnum[b][mm] / faildens[b] if faildens[b] else None) for mm in MISS} for b in BINS}
    tax_all  = {b: {mm: (100.0 * missnum[b][mm] / alln[b] if alln[b] else None) for mm in MISS} for b in BINS}
    return metrics, tax_fail, tax_all, {b: alln[b] for b in BINS}, {b: faildens[b] for b in BINS}


def band(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0), len(vals))


def fmt(t):
    return f"{t[0]:5.1f} ± {t[1]:4.1f}" if t else "  —  "


def main():
    os.makedirs(OUT, exist_ok=True)
    dm = diffmap()
    per = []
    for s in SEEDS:
        d = f"{D}/s{s}"
        if not os.path.isdir(d):
            continue
        rows = load_seed(d)
        if rows:
            per.append((s, seed_metrics(rows, dm)))
    nseed = len(per)
    METR = ["base_open", "armi_any", "armi_modelpref", "armi_mean", "armii", "armii_sim"]
    LBL = {"base_open": "BASELINE open@2 (anchor 40.7)", "armi_any": "ARM(i)-any [GATE]",
           "armi_modelpref": "ARM(i)-modelpref", "armi_mean": "ARM(i)-mean(setup-avg)",
           "armii": "ARM(ii) recoverable (key)", "armii_sim": "ARM(ii)'-sim recoverable"}
    agg = {m: {b: band([p[1][0][m][b] for p in per]) for b in BINS} for m in METR}
    tax_fail = {b: {mm: band([p[1][1][b][mm] for p in per]) for mm in MISS} for b in BINS}
    tax_all  = {b: {mm: band([p[1][2][b][mm] for p in per]) for mm in MISS} for b in BINS}
    counts = per[0][1][3] if per else {}
    faildens = {b: band([p[1][4][b] for p in per]) for b in BINS}

    L = [f"# Phase-0 GATE aggregation (CAR, pure2push, greedy, 2push-only) — {nseed} NoHz-v3 ckpt-seeds",
         f"episodes/bin: {counts}", ""]
    L += ["| metric | easy | medium | hard | all |", "|---|---|---|---|---|"]
    for m in METR:
        L.append(f"| {LBL[m]} | " + " | ".join(fmt(agg[m][b]) for b in BINS) + " |")
    L += ["", "### ARM (iii) miss taxonomy — share of FAILED (base_open=0) episodes (%)",
          "| miss type | easy | medium | hard | all |", "|---|---|---|---|---|"]
    for mm in MISS:
        L.append(f"| {mm} | " + " | ".join(fmt(tax_fail[b][mm]) for b in BINS) + " |")
    L += ["", "### ARM (iii) miss taxonomy — share of ALL episodes (%)",
          "| miss type | easy | medium | hard | all |", "|---|---|---|---|---|"]
    for mm in MISS:
        L.append(f"| {mm} | " + " | ".join(fmt(tax_all[b][mm]) for b in BINS) + " |")
    L += ["", f"failed episodes/bin: " + ", ".join(f"{b}={fmt(faildens[b])}" for b in BINS)]

    g = agg["armi_any"]["all"]
    gate = "PROCEED (>=85)" if g and g[0] >= 85 else ("STOP/RETHINK (<65)" if g and g[0] < 65 else "GREY ZONE (65-85) -> teacher-forced k-sweep")
    L += ["", f"**GATE (ARM(i)-any, all): {fmt(g)} -> {gate}**"]
    table = "\n".join(L)
    open(f"{OUT}/table.md", "w").write(table)
    res = {"nseed": nseed, "counts": counts,
           "metrics": {m: {b: (list(agg[m][b]) if agg[m][b] else None) for b in BINS} for m in METR},
           "tax_fail": {b: {mm: (list(tax_fail[b][mm]) if tax_fail[b][mm] else None) for mm in MISS} for b in BINS},
           "tax_all": {b: {mm: (list(tax_all[b][mm]) if tax_all[b][mm] else None) for mm in MISS} for b in BINS}}
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    print(table)


if __name__ == "__main__":
    main()
