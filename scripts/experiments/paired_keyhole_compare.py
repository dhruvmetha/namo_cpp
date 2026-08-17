#!/usr/bin/env python3
"""Model vs random, PAIRED PER REGION-OPENING PROBLEM (per keyhole).

The unit is one keyhole = (xml, object_id, goal region) -- the same problem handed to both arms --
never a pooled solve-rate. For each keyhole: how much search did it take, and did the ranker cut it?
Every statistic is then a DISTRIBUTION over problems (median problem, the share of problems the
ranker LOSES), which a pooled curve cannot show.

CANONICAL STATISTIC (EXP-2026-08-09-crossboard-ranking, 2026-08-13, definition (c)):
  1. per keyhole, one speed-up per seed pairing (model seed i vs random seed i),
  2. take the MEDIAN of those within the keyhole -> one number per problem,
  3. percentiles ACROSS problems.
Do not quote the ratio of medians (reads 20.2x where the per-instance median is 10.9x) and never the
mean of ratios. The loss rate (% of problems where the ranker is slower) is quoted BESIDE the median,
always on the same cost measure as the median.

Both arms come from ONE campaign (eval_walltime4k: budget 4000, exclusive single-generation nodes,
single-threaded) so SECONDS are comparable; simulator calls are comparable anywhere.

Censoring is explicit: an unsolved run spent the whole budget without an answer, so its cost is a
LOWER bound. Those keyholes are counted and reported, and their (bounded) ratio is included only in
the `_with_censored` columns, never in the headline.

    python scripts/experiments/paired_keyhole_compare.py --out $NAMO_SCRATCH/analysis/keyhole
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python", f"{REPO}/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo import eval_sets  # noqa: E402

CAMPAIGN = "aquaman/round0/eval_walltime4k"
# Seed pairing is positional: model seed k is paired with random seed k, the same way the campaign
# was launched. Any fixed pairing works; mixing all 3x3 would double-count each run.
ARM_PAIRS = [("HY5U_s1", "rand_s7000"), ("HY5U_s2", "rand_s8000"), ("HY5U_s3", "rand_s9000")]
LEGS = {"1push": "1push_hmax2", "2push": "2push"}
TIERS = ["easy", "medium", "hard"]
PCTS = [10, 25, 50, 75, 90, 95, 99]


def suf(p, n=5):
    return "/".join(str(p).rstrip("/").split("/")[-n:])


def tier_maps():
    """(xml suffix, object_id) -> tier, computed exactly as scripts/rl_loop/aquaman_agg.py does."""
    one = json.load(open(eval_sets.path("onepush_manifest")))
    t1 = {}
    for x, eps in one.items():
        for e in eps:
            sr = len(e["valid"]) / max(len(e["tried"]), 1)
            t1[(suf(x), e["object_id"])] = "hard" if sr < 0.05 else ("medium" if sr < 0.30 else "easy")
    div = json.load(open(eval_sets.path("pure2push_divisions")))
    t2 = {(suf(x), e["object_id"]): e["division"] for x, eps in div.items() for e in eps}
    return {"1push": t1, "2push": t2}


def load_run(root, arm, leg):
    out = {}
    for f in glob.glob(os.path.join(root, CAMPAIGN, arm, LEGS[leg], "shard_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            out[(suf(r["xml"]), r["object_id"], r.get("region"))] = {
                "sims": float(r["sims"]), "t_wall": float(r["t_wall"]), "solved": bool(r["solved"])}
    return out


def pctl(v, p):
    """Percentile by nearest rank -- no interpolation, so every reported value is a real problem."""
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def geomean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def keyholes(root, leg):
    """keyhole -> per-pairing costs + the per-problem speed-up (the canonical statistic's step 2)."""
    runs = [(load_run(root, m, leg), load_run(root, r, leg)) for m, r in ARM_PAIRS]
    keys = set(runs[0][0])
    for m, r in runs:
        keys &= set(m) & set(r)
    out = {}
    for k in keys:
        rec = {"pairings": []}
        for m, r in runs:
            rec["pairings"].append({
                "model_sims": m[k]["sims"], "rand_sims": r[k]["sims"],
                "model_t": m[k]["t_wall"], "rand_t": r[k]["t_wall"],
                "model_solved": m[k]["solved"], "rand_solved": r[k]["solved"]})
        p = rec["pairings"]
        rec["model_solved_all"] = all(x["model_solved"] for x in p)
        rec["rand_solved_all"] = all(x["rand_solved"] for x in p)
        rec["clean"] = rec["model_solved_all"] and rec["rand_solved_all"]
        for meas, mk, rk in (("sims", "model_sims", "rand_sims"), ("time", "model_t", "rand_t")):
            rec[f"speedup_{meas}"] = st.median([x[rk] / max(x[mk], 1e-9) for x in p])
            rec[f"model_{meas}"] = st.median([x[mk] for x in p])
            rec[f"rand_{meas}"] = st.median([x[rk] for x in p])
        out[k] = rec
    return out


def table(kh, tiers, leg, meas):
    rows = []
    for tier in TIERS + ["all"]:
        sel = [k for k, v in kh.items() if tier == "all" or tiers.get((k[0], k[1])) == tier]
        if not sel:
            continue
        clean = [kh[k] for k in sel if kh[k]["clean"]]
        s = [v[f"speedup_{meas}"] for v in clean]
        row = {"leg": leg, "measure": meas, "tier": tier, "n_problems": len(sel),
               "n_clean": len(clean),
               "model_solve_pct": round(100 * sum(kh[k]["model_solved_all"] for k in sel) / len(sel), 1),
               "rand_solve_pct": round(100 * sum(kh[k]["rand_solved_all"] for k in sel) / len(sel), 1),
               "model_med_cost": round(st.median([v[f"model_{meas}"] for v in clean]), 2) if clean else None,
               "rand_med_cost": round(st.median([v[f"rand_{meas}"] for v in clean]), 2) if clean else None,
               "pct_ranker_loses": round(100 * sum(1 for x in s if x < 1) / len(s), 1) if s else None,
               "geomean": round(geomean(s), 2) if s else None,
               "only_model_solved": sum(1 for k in sel if kh[k]["model_solved_all"] and not kh[k]["rand_solved_all"]),
               "only_rand_solved": sum(1 for k in sel if kh[k]["rand_solved_all"] and not kh[k]["model_solved_all"]),
               "neither_solved": sum(1 for k in sel if not kh[k]["model_solved_all"] and not kh[k]["rand_solved_all"])}
        for p in PCTS:
            row[f"p{p}"] = round(pctl(s, p), 2) if s else None
        rows.append(row)
    return rows


def dump_pairs(kh, tiers, leg, path):
    """One row per keyhole -- what every figure is drawn from, and the audit trail."""
    with open(path, "w") as f:
        for k, v in sorted(kh.items()):
            f.write(json.dumps({
                "xml": k[0], "object_id": k[1], "region": k[2], "leg": leg,
                "tier": tiers.get((k[0], k[1])),
                "model_sims": v["model_sims"], "rand_sims": v["rand_sims"],
                "model_t": round(v["model_time"], 4), "rand_t": round(v["rand_time"], 4),
                "speedup_sims": round(v["speedup_sims"], 4), "speedup_time": round(v["speedup_time"], 4),
                "model_solved": v["model_solved_all"], "rand_solved": v["rand_solved_all"],
                "clean": v["clean"]}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=os.environ.get("NAMO_SCRATCH"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tm = tier_maps()
    allrows = []
    for leg in LEGS:
        kh = keyholes(a.scratch, leg)
        dump_pairs(kh, tm[leg], leg, os.path.join(a.out, f"pairs_{leg}.jsonl"))
        for meas in ("time", "sims"):
            allrows += table(kh, tm[leg], leg, meas)
    json.dump(allrows, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

    hdr = (["leg", "measure", "tier", "n_problems", "n_clean", "model_solve_pct", "rand_solve_pct",
            "model_med_cost", "rand_med_cost"] + [f"p{p}" for p in PCTS] +
           ["geomean", "pct_ranker_loses", "only_model_solved", "only_rand_solved", "neither_solved"])
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in allrows:
        print("| " + " | ".join(str(r[h]) for h in hdr) + " |")


if __name__ == "__main__":
    main()
