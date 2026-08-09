#!/usr/bin/env python3
"""Print the AUC metric family for the v2 arms beside every previously measured arm.

The seven metrics answer different questions and have been confused for one another before
(see project_auc_reconciliation) -- so each row is labelled with what it actually compares:
  V1  root pooled          does the model rank the root's good pushes above its bad ones
  V2  root within-board    same, but scored inside each board (shift-invariant)
  F1/F2 finish             the same two for FINISH boards
  V4  setup cell vs dead cells   cell-level, cross-board
  V5  setup cell vs dead board MAX  <- the order statistic; the one stuck at 0.53
  V6  live board max vs dead board max

⛔ ALWAYS READ THE PER-TIER BLOCKS, NOT JUST `all`. The pooled row is dominated by easy+med
(865 of 983 setup boards) and on 2026-08-08 it INVERTED the truth: pooled setup@1 said AJ2NR
(56.7) beat AJ2 (55.1), while per tier AJ2NR wins easy by +8.1 and loses hard by −10.7 — and
hard is the tier the deploy gate is judged on. So this prints every tier by default and puts the
seed bands on `hard`.
"""
import json
from pathlib import Path

R0 = Path("/common/users/dm1487/scratch_namo/aquaman/round0")
PANELS = {"auc_bfix.json": None, "auc_bng.json": None, "auc_arj.json": None, "auc_aj2.json": None, "auc_aj3.json": None}

TIERS = ["all", "easy", "med", "hard"]

rows = {}
for fn in PANELS:
    p = R0 / fn
    if not p.exists():
        continue
    for name, m in json.load(open(p))["models"].items():
        stem = name.rsplit("_s", 1)[0]
        for tier in TIERS:
            if tier not in m:
                continue
            a = m[tier]
            rows.setdefault((stem, tier), []).append({
                "V1": a["separation_root"]["V1_pooled"],
                "V2": a["separation_root"]["V2_within_board"],
                "F1": a["separation_finish"]["F1_pooled"],
                "F2": a["separation_finish"]["F2_within_board"],
                "V4": a["cross_board"]["V4_setupcell_vs_deadcells"],
                "V5": a["cross_board"]["V5_setupcell_vs_deadmax"],
                "V5m": a["cross_board"]["V5m_setupcell_vs_moved_deadmax"],
                "V6": a["cross_board"]["V6_livemax_vs_deadmax"],
                "setup@1": a["rank_setup"]["hit_at_1_pct"],
                "finish@1": a["rank_finish"]["hit_at_1_pct"] if "rank_finish" in a else None,
            })

KEYS = ["V1", "V2", "F1", "F2", "V4", "V5", "V5m", "V6", "setup@1", "finish@1"]
ORDER = [k for k in ("Bfix", "BfixNR", "ANR", "BNG", "ARJ", "AJ2", "AJ2NR")
         if any(stem == k for stem, _ in rows)]
for tier in TIERS:
    print(f"\n=== {tier} ===")
    print(f"{'arm':<8}" + "".join(f"{k:>10}" for k in KEYS))
    for stem in ORDER:
        seeds = rows.get((stem, tier))
        if not seeds:
            continue
        out = f"{stem:<8}"
        for k in KEYS:
            vals = [s[k] for s in seeds if s[k] is not None]
            out += f"{(sum(vals)/len(vals)):>10.3f}" if vals else f"{'-':>10}"
        print(out)
print("\nseed bands (hard tier -- where the aux's remaining value lives):")
for stem in ORDER:
    for k in ("setup@1", "finish@1", "V5"):
        vals = [s[k] for s in rows.get((stem, "hard"), []) if s[k] is not None]
        if vals:
            print(f"  {stem:<7} {k:<9} [{min(vals):.2f}, {max(vals):.2f}]  n_seeds={len(vals)}")
