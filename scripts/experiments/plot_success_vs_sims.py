#!/usr/bin/env python3
"""Success-vs-simulator-calls curves from a common-episode gate json.

Reads a gate produced by `aquaman_agg_common.py` (so every arm is scored on the SAME episodes)
and draws solve-rate vs budget, one panel per difficulty tier, for 2-push and 1-push.

Usage:
  python plot_success_vs_sims.py --gate <gate_plot.json> --out <dir> \
      --arms random BNG MM HY5 HY5U
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGETS = [1, 2, 5, 10, 30, 100, 300, 900]
# plain-English names: advisors and readers should never need the internal arm codes
LABEL = {
    "random": "Random ordering (baseline)",
    "BNG": "Bootstrap-guess model (previous best)",
    "MM": "Margin-vs-max ranker",
    "HY5": "Hybrid corpus",
    "HY5U": "Hybrid corpus + unreachable-cell rule (best)",
}
STYLE = {
    "random": dict(color="#999999", ls="--", lw=1.8, marker="o", ms=4),
    "BNG": dict(color="#4C72B0", ls="-", lw=1.8, marker="s", ms=4),
    "MM": dict(color="#55A868", ls="-", lw=1.8, marker="^", ms=4),
    "HY5": dict(color="#DD8452", ls="-", lw=1.8, marker="D", ms=4),
    "HY5U": dict(color="#C44E52", ls="-", lw=2.6, marker="*", ms=9),
}


def panel(ax, gate, arms, leg, tier, title):
    for a in arms:
        t = gate.get(a, {}).get(leg, {}).get(tier)
        if not t:
            continue
        ys = [t[f"solve@{b}"] for b in BUDGETS]
        ax.plot(BUDGETS, ys, label=LABEL.get(a, a), **STYLE.get(a, {}))
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=8)
    ax.set_xlabel("simulator calls allowed (log scale)")
    ax.set_ylabel("episodes solved (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
    n = gate[arms[-1]][leg][tier]["n"]
    ax.set_title(f"{title}  (n={n})", fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    gate = json.load(open(a.gate))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for leg, legname in (("2push", "two-push"), ("1push", "one-push")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
        for ax, tier in zip(axes, ("easy", "medium", "hard")):
            panel(ax, gate, a.arms, leg, tier, f"{legname} — {tier}")
        axes[0].legend(fontsize=8, loc="lower right", framealpha=0.95)
        fig.suptitle(f"Success vs simulator calls — {legname} problems "
                     f"(all models scored on the same held-out episodes)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = out / f"success_vs_sims_{leg}.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
