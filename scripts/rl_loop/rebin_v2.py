#!/usr/bin/env python3
"""Re-bin EXISTING eval results onto v2 (fixed-physics) tiers. No new sims.

Population is held FIXED to the v1 eval set: a row is included iff it matches a v1 key, and it keeps
its v1 tier when v2 has no counterpart (121 of 1322 1push episodes, 38 2push). So every difference
below is tier-driven, never population-driven -- switching the manifest wholesale would confound the
two, which is exactly why eval_sets.yaml is untouched.

Tier rules are the registered ones (scripts/rl_loop/aquaman_agg.py):
  1push -> bin_of(solve_rate)                 cuts hard <0.05 / medium <0.30 / easy >=0.30
  2push -> `division` from the GT divisions file
"""
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from namo import paths

L1 = Path(f"{paths.DATASETS}/namo_testset_v1/labels")
L2 = Path(f"{paths.DATASETS}/namo_testset_v2/labels")
EVAL = Path(f"{paths.SCRATCH}/aquaman/round0/eval_bfix")
BUDGETS = [1, 5, 30, 900]
suf = lambda p, n=5: "/".join(p.rstrip("/").split("/")[-n:])
bin_of = lambda s: "hard" if s < 0.05 else ("medium" if s < 0.30 else "easy")


def key_map(path, fn):
    out = {}
    for x, eps in json.load(open(path)).items():
        for e in eps:
            out.setdefault((suf(x), e["object_id"]), fn(e))
    return out


T1_V1 = key_map(L1 / "onepush_search_eval.json", lambda e: bin_of(len(e["valid"]) / max(len(e["tried"]), 1)))
T1_V2 = key_map(L2 / "onepush_divisions_v2.json", lambda e: e["division"])
T2_V1 = key_map(L1 / "pure2push_gt_divisions_search_eval.json", lambda e: e["division"])
T2_V2 = key_map(L2 / "pure2push_gt_divisions_v2.json", lambda e: e["division"])


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


def table(rows, v1map, v2map, use_v2):
    binned, unmatched = {}, 0
    for r in rows:
        k = (suf(r["xml"]), r["object_id"])
        if k not in v1map:                      # population is the v1 set, always
            unmatched += 1
            continue
        t = v2map.get(k, v1map[k]) if use_v2 else v1map[k]
        binned.setdefault(t, []).append(r)
    out = {}
    for t in ("easy", "medium", "hard", "all"):
        sel = sum(binned.values(), []) if t == "all" else binned.get(t, [])
        if not sel:
            continue
        solved = [r["sims"] for r in sel if r["solved"]]
        out[t] = {"n": len(sel),
                  **{f"@{b}": round(100 * sum(1 for r in sel if r["solved"] and r["sims"] <= b) / len(sel), 1)
                     for b in BUDGETS},
                  "s2s": round(float(np.mean(solved)), 1) if solved else None}
    out["_unmatched"] = unmatched
    return out


arms = sys.argv[1:] or ["BNG", "XB", "AJ2", "Bfix", "ARJ", "ANG"]
report = {}
for arm in arms:
    legs = {"1push": sorted(str(p) for p in EVAL.glob(f"{arm}_s*/1push_hmax2")),
            "2push": sorted(str(p) for p in EVAL.glob(f"{arm}_s*/2push"))}
    report[arm] = {}
    for horizon, (a, b) in (("1push", (T1_V1, T1_V2)), ("2push", (T2_V1, T2_V2))):
        if not legs[horizon]:
            continue
        rows = load(legs[horizon])
        report[arm][horizon] = {"v1_tiers": table(rows, a, b, False), "v2_tiers": table(rows, a, b, True)}
    print(f"{arm} done", flush=True)

print(f"\n{'arm':<7} {'horizon':<7} {'tier':<7} {'n v1->v2':<12} " + " ".join(f"{'@'+str(b):>13}" for b in BUDGETS))
worst = 0.0
for arm, hs in report.items():
    for h, d in hs.items():
        for t in ("easy", "medium", "hard", "all"):
            u, v = d["v1_tiers"].get(t), d["v2_tiers"].get(t)
            if not u or not v:
                continue
            cells = []
            for b in BUDGETS:
                delta = v[f"@{b}"] - u[f"@{b}"]
                worst = max(worst, abs(delta))
                cells.append(f"{u[f'@{b}']:>5.1f}->{v[f'@{b}']:<5.1f}"[:13].rjust(13))
            print(f"{arm:<7} {h:<7} {t:<7} {str(u['n'])+'->'+str(v['n']):<12} " + " ".join(cells))
print(f"\nLARGEST single solve@k shift across every arm/horizon/tier: {worst:.1f} pt")
out = Path(f"{paths.SCRATCH}/aquaman/round0/rebin_v2.json")
out.write_text(json.dumps(report, indent=1))
print("wrote", out)
