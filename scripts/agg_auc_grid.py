#!/usr/bin/env python3
"""Render the eval_auc.py grid into the markdown tables the reconciliation doc carries.

    python scripts/agg_auc_grid.py --canonical canonical.json --deadbank deadbank.json \
        [--onepush label=onepush_X.json ...] > tables.md

Seed bands come from label suffixes `_s1/_s2/_s3`: any metric's spread across the seeds of one
condition is the noise floor that a model-to-model delta has to clear. Reported as mean ± half-range
(half-range, not std, because n=3 -- it is the honest "how far apart did they land" number).
"""
import argparse
import collections
import json
import re

import numpy as np

ROOT_ROWS = [("V1", "separation_root", "V1_pooled"), ("V2", "separation_root", "V2_within_board"),
             ("V3", "cross_board", "V3_rootmax_vs_deadmax"), ("V4", "cross_board", "V4_setupcell_vs_deadcells"),
             ("V5", "cross_board", "V5_setupcell_vs_deadmax"), ("V6", "cross_board", "V6_livemax_vs_deadmax"),
             ("F1", "separation_finish", "F1_pooled"), ("F2", "separation_finish", "F2_within_board")]
RANK_ROWS = [("setup hit@1", "rank_setup", "hit_at_1_pct"), ("setup floor@1", "rank_setup", "floor_at_1_pct"),
             ("finish hit@1", "rank_finish", "hit_at_1_pct"), ("finish floor@1", "rank_finish", "floor_at_1_pct")]


def get(block, section, key):
    part = block.get(section)
    return part.get(key) if part else None


def fmt(x):
    return "—" if x is None else (f"{x:.3f}" if abs(x) < 3 else f"{x:.1f}")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["--:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def condition_of(label):
    return re.sub(r"_s\d+$", "", label)


def band(values):
    """mean ± half-range over a condition's seeds."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    return float(np.mean(values)), (max(values) - min(values)) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--deadbank")
    parser.add_argument("--onepush", action="append", default=[], help="label=path.json (eval_scorer output)")
    args = parser.parse_args()

    canonical = json.load(open(args.canonical))
    models = canonical["models"]
    group = "all_tiered" if "all_tiered" in next(iter(models.values())) else "all"

    print(f"### Canonical testset ({canonical['eval_set']}, {canonical['n_episodes_tiered']} tiered episodes)\n")
    labels = list(models)
    rows = [[name] + [fmt(get(models[m][group], sec, key)) for m in labels] for name, sec, key in ROOT_ROWS + RANK_ROWS]
    print(table(["metric"] + labels, rows) + "\n")

    print("### Per tier (all models, V1 / V5 / setup hit@1 vs floor)\n")
    rows = []
    for m in labels:
        for tier in ("easy", "med", "hard"):
            block = models[m].get(tier)
            if not block:
                continue
            rows.append([m, tier, fmt(get(block, "separation_root", "V1_pooled")),
                         fmt(get(block, "cross_board", "V5_setupcell_vs_deadmax")),
                         fmt(get(block, "rank_setup", "hit_at_1_pct")),
                         fmt(get(block, "rank_setup", "floor_at_1_pct"))])
    print(table(["model", "tier", "V1", "V5", "setup hit@1", "floor@1"], rows) + "\n")

    print("### Seed noise floor (mean ± half-range within a condition)\n")
    conditions = collections.defaultdict(list)
    for m in labels:
        conditions[condition_of(m)].append(m)
    rows = []
    for name, sec, key in ROOT_ROWS + RANK_ROWS:
        cells = [name]
        for cond, members in conditions.items():
            if len(members) < 2:
                continue
            result = band([get(models[m][group], sec, key) for m in members])
            cells.append("—" if result is None else f"{fmt(result[0])} ± {result[1]:.3f}")
        rows.append(cells)
    seeded = [c for c, members in conditions.items() if len(members) >= 2]
    print(table(["metric"] + seeded, rows) + "\n")

    if args.deadbank:
        deadbank = json.load(open(args.deadbank))
        print(f"### Same models, dead-bank distribution ({deadbank['eval_set']}) — NOT comparable to the table above\n")
        rows = []
        for name, sec, key in ROOT_ROWS:
            canon = [get(models[m][group], sec, key) for m in labels if m in deadbank["models"]]
            dead = [get(deadbank["models"][m]["all"], sec, key) for m in labels if m in deadbank["models"]]
            pair = [(c, d) for c, d in zip(canon, dead) if c is not None and d is not None]
            if not pair:
                continue
            rows.append([name, fmt(float(np.mean([c for c, _ in pair]))), fmt(float(np.mean([d for _, d in pair]))),
                         fmt(float(np.mean([d - c for c, d in pair])))])
        print(table(["metric", "canonical (mean)", "dead-bank (mean)", "Δ"], rows) + "\n")

    if args.onepush:
        print("### 1-push horizon (canonical onepush manifest, eval_scorer.py live)\n")
        rows = []
        for spec in args.onepush:
            label, path = spec.split("=", 1)
            divisions = json.load(open(path))["divisions"]
            for tier in ("easy", "med", "hard"):
                block = divisions.get(tier, {})
                sep = block.get("score_separation", {})
                rows.append([label, tier, fmt(sep.get("auc_pooled")), fmt(sep.get("auc_within_episode")),
                             fmt(block.get("scorer_realistic", {}).get("@1")), fmt(block.get("floor", {}).get("@1"))])
        print(table(["model", "tier", "opener AUC pooled", "within-episode", "hit@1", "floor@1"], rows) + "\n")


if __name__ == "__main__":
    main()
