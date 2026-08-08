#!/usr/bin/env python3
"""Success vs simulator calls: the label x aux 2x2, against random.

Encoding is deliberate -- COLOUR is the label regime, LINE STYLE is the ranking aux:
    blue   = bootstrap-guess labels (Bfix / BfixNR)
    orange = hard-floor labels      (AJ2 / AJ2NR)
    solid  = ranking aux ON         dashed = aux OFF
so the substitution result is legible without reading the legend twice: the blue pair is far
apart (aux worth +17.7), the orange pair is close (aux worth +6.9).

Three hues only (plus grey) keeps it colour-blind safe; the aux contrast is carried by style,
not by a fourth colour.
"""
import json
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R0 = Path("/common/users/dm1487/scratch_namo/aquaman/round0")
LAB = Path("/common/users/dm1487/scratch_namo/datasets/namo_testset_v1/labels")
K1 = json.load(open(LAB / "onepush_search_eval.json"))
K2 = json.load(open(LAB / "pure2push_gt_divisions_search_eval.json"))


def suf(p, n=5):
    return "/".join(p.rstrip("/").split("/")[-n:])


K1s = {suf(k): v for k, v in K1.items()}
K2s = {suf(k): v for k, v in K2.items()}


def tier_1p(xml, obj, region):
    for e in K1s.get(suf(xml), []):
        if e["object_id"] == obj and (e.get("region", region) == region or len(K1s.get(suf(xml), [])) == 1):
            sr = len(e["valid"]) / max(len(e["tried"]), 1)
            return "hard" if sr < 0.05 else ("medium" if sr < 0.30 else "easy")
    return None


def tier_2p(xml, obj, region):
    for e in K2s.get(suf(xml), []):
        if e["object_id"] == obj and (e.get("region", region) == region or len(K2s.get(suf(xml), [])) == 1):
            return e["division"]
    return None


def load(d):
    rows = []
    for f in glob(f"{d}/shard_*.jsonl"):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def curve_one(rows, tier_fn, tier, budgets):
    sel = [r for r in rows
           if (tier_fn(r["xml"], r["object_id"], r.get("region", "goal")) == tier if tier != "all"
               else tier_fn(r["xml"], r["object_id"], r.get("region", "goal")) is not None)]
    if not sel:
        return None, 0
    sims = np.array([r["sims"] for r in sel], float)
    ok = np.array([bool(r["solved"]) for r in sel])
    return np.array([100.0 * np.mean(ok & (sims <= b)) for b in budgets]), len(sel)


def curves_per_seed(seed_dirs, tier_fn, tier, budgets):
    """Each seed is a separately trained model, so the band is real between-model spread."""
    ys, n = [], 0
    for d in seed_dirs:
        y, n_ep = curve_one(load(d), tier_fn, tier, budgets)
        if y is not None:
            ys.append(y)
            n = n_ep
    if not ys:
        return None, None, None, 0
    Y = np.vstack(ys)
    return Y.mean(axis=0), Y.min(axis=0), Y.max(axis=0), n


BLUE, ORANGE, GREY = "#2563EB", "#EA580C", "#6B7280"
COND = [
    ("random", [f"{R0}/eval_amarel/random_s{s}" for s in (7000, 8000, 9000)],
     {"1push": "1push_hmax2", "2push": "2push_fine"}, GREY, (0, (1, 1.6)), 1.7),
    ("bootstrap labels, aux ON  (Bfix)", [f"{R0}/eval_bfix/Bfix_s{s}" for s in (1, 2, 3)],
     {"1push": "1push_hmax2", "2push": "2push"}, BLUE, "-", 2.0),
    ("bootstrap labels, aux OFF (BfixNR)", [f"{R0}/eval_bfix/BfixNR_s{s}" for s in (1, 2, 3)],
     {"1push": "1push_hmax2", "2push": "2push"}, BLUE, (0, (5, 2)), 2.0),
    ("hard labels, aux ON  (AJ2)", [f"{R0}/eval_bfix/AJ2_s{s}" for s in (1, 2, 3)],
     {"1push": "1push_hmax2", "2push": "2push"}, ORANGE, "-", 2.0),
    ("hard labels, aux OFF (AJ2NR)", [f"{R0}/eval_bfix/AJ2NR_s{s}" for s in (1, 2, 3)],
     {"1push": "1push_hmax2", "2push": "2push"}, ORANGE, (0, (5, 2)), 2.0),
]
BUD = np.unique(np.round(np.logspace(0, np.log10(900), 90)).astype(int))
TIERS = ["easy", "medium", "hard", "all"]

fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.4), sharex=True, sharey=True)
fig.patch.set_facecolor("#fcfcfb")
counts = {}
for ri, (hz, tfn) in enumerate((("1push", tier_1p), ("2push", tier_2p))):
    for ci, tier in enumerate(TIERS):
        ax = axes[ri][ci]
        ax.set_facecolor("#fcfcfb")
        for name, dirs, legs, color, ls, lw in COND:
            y, lo, hi, n = curves_per_seed([f"{d}/{legs[hz]}" for d in dirs], tfn, tier, BUD)
            if y is None:
                continue
            counts[(hz, tier)] = n
            ax.fill_between(BUD, lo, hi, color=color, alpha=0.13, lw=0, zorder=1)
            ax.plot(BUD, y, color=color, ls=ls, lw=lw, label=name, zorder=3)
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.grid(True, which="major", color="#e6e5e2", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c4")
        ax.tick_params(colors="#57534e", labelsize=9)
        ax.set_title(f"{hz}  ·  {tier}   (n≈{counts.get((hz, tier), 0)})",
                     fontsize=11, color="#292524", pad=7)
        if ci == 0:
            ax.set_ylabel("episodes solved (%)", fontsize=10, color="#57534e")
        if ri == 1:
            ax.set_xlabel("simulator calls", fontsize=10, color="#57534e")

# 1push-easy saturates instantly, so its lower-right corner is the only reliably empty space.
axes[0][0].legend(frameon=False, fontsize=8.8, loc="lower right", labelcolor="#292524")
fig.suptitle("Labels × ranking aux — success vs simulator calls   "
             "(line = mean of 3 seeds, band = seed min–max; hmax=2, budget 900, discount off)\n"
             "colour = label regime · solid = aux ON, dashed = aux OFF — the blue pair is far apart, the orange pair is not",
             fontsize=12.5, color="#292524", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.935))
out = R0 / "v2_success_vs_sims.png"
fig.savefig(out, dpi=155, facecolor=fig.get_facecolor())
print("wrote", out)
