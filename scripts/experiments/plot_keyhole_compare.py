#!/usr/bin/env python3
"""Figures for the paired per-keyhole comparison (input: paired_keyhole_compare.py's pairs_*.jsonl).

Two figures, each answering one question:
  paired_scatter  -- "is the win per problem, or an averaging artifact?" One dot per region-opening
                     problem, model seconds vs random seconds, split by horizon x tier. Colour is
                     POLARITY (cheaper / slower), so the problems the ranker loses are visible
                     instead of hidden in a mean.
  speedup_curve   -- "how big is the win on a typical problem, and where does it live?" Per-problem
                     speed-up sorted into percentiles, one line per tier, with parity marked.

Colours: diverging blue/orange for the win-loss polarity, categorical blue/orange/aqua for tiers;
both validated with the dataviz palette checker. Tier lines are direct-labelled (the aqua step sits
under 3:1 against the surface, so identity never rests on colour alone).
"""
import argparse
import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHEAPER, SLOWER, CENSORED = "#2a78d6", "#eb6834", "#9a9a94"
TIER_COLOR = {"easy": "#2a78d6", "medium": "#eb6834", "hard": "#1baf7a"}
TIERS = ["easy", "medium", "hard"]
LEGS = [("1push", "one-push"), ("2push", "two-push")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e2"


def load(d, leg):
    return [json.loads(l) for l in open(os.path.join(d, f"pairs_{leg}.jsonl"))]


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def paired_scatter(data, out):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.4), sharex="row", sharey="row")
    for row, (leg, legname) in enumerate(LEGS):
        rows = data[leg]
        # One scale per HORIZON: panels within a row are then directly comparable, which is the
        # point of splitting by tier. Sharing across horizons would waste most of the 1push panels.
        costs = [r[k] for r in rows for k in ("rand_t", "model_t")]
        rlo, rhi = max(min(costs) * 0.7, 1e-2), max(costs) * 1.4
        for col, tier in enumerate(TIERS):
            ax = axes[row][col]
            style(ax)
            sel = [r for r in rows if r["tier"] == tier]
            clean = [r for r in sel if r["clean"]]
            cens = [r for r in sel if not r["clean"]]
            win = [r for r in clean if r["speedup_time"] >= 1]
            lose = [r for r in clean if r["speedup_time"] < 1]
            for grp, c in ((win, CHEAPER), (lose, SLOWER)):
                ax.scatter([r["rand_t"] for r in grp], [r["model_t"] for r in grp],
                           s=11, c=c, alpha=0.5, linewidths=0)
            if cens:
                ax.scatter([r["rand_t"] for r in cens], [r["model_t"] for r in cens],
                           s=22, facecolors="none", edgecolors=CENSORED, linewidths=0.9, marker="s")
            ax.plot([rlo, rhi], [rlo, rhi], color=INK, lw=1.0)
            ax.plot([rlo, rhi], [rlo / 10, rhi / 10], color=MUTED, lw=0.9, ls=":")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(rlo, rhi); ax.set_ylim(rlo, rhi)
            s = [r["speedup_time"] for r in clean]
            med = st.median(s)
            loses = 100 * sum(1 for x in s if x < 1) / len(s)
            # Stats live ABOVE the axes: inside the panel they land on the data at some tier.
            ax.set_title(f"{legname} · {tier}\n"
                         f"median {med:.1f}× · ranker loses {loses:.0f}% · n = {len(clean)}"
                         + (f" (+{len(cens)} censored)" if cens else ""),
                         fontsize=9.5, color=INK, loc="left", linespacing=1.5)
            if col == 0:
                ax.set_ylabel("ranker: seconds to solve", fontsize=9, color=MUTED)
            if row == 1:
                ax.set_xlabel("random: seconds to solve", fontsize=9, color=MUTED)

    handles = [plt.Line2D([], [], marker="o", ls="", color=CHEAPER, label="ranker cheaper"),
               plt.Line2D([], [], marker="o", ls="", color=SLOWER, label="ranker slower"),
               plt.Line2D([], [], marker="s", ls="", mfc="none", mec=CENSORED,
                          label="unsolved by an arm (cost is a lower bound)"),
               plt.Line2D([], [], color=INK, lw=1, label="parity"),
               plt.Line2D([], [], color=MUTED, lw=1, ls=":", label="10× cheaper")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("One dot = one region-opening problem, solved by both arms on the same node",
                 y=0.945, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=160)
    print("wrote", out)


def speedup_curve(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, (leg, legname) in zip(axes, LEGS):
        style(ax)
        series = {t: sorted(r["speedup_time"] for r in data[leg] if r["tier"] == t and r["clean"])
                  for t in TIERS}
        floor = min(min(s) for s in series.values()) * 0.8
        # The loss zone, shaded: everything under this band is a problem random did better on.
        ax.axhspan(floor, 1.0, color="#f2f1ec", zorder=0)
        ax.axhline(1.0, color=INK, lw=1.0, ls="--")
        # Label the band, not the axis: an axis-side note lands on the "10^0" tick label.
        ax.text(99, floor * 1.08, "shaded: random was faster on these problems",
                fontsize=8.5, color=MUTED, va="bottom", ha="right")
        for tier in TIERS:
            s = series[tier]
            xs = [100 * (i + 0.5) / len(s) for i in range(len(s))]
            ax.plot(xs, s, color=TIER_COLOR[tier], lw=2)
            mid = s[len(s) // 2]
            ax.plot([50], [mid], marker="o", ms=6, color=TIER_COLOR[tier],
                    mec="#fcfcfb", mew=1.5, zorder=4)
            ax.text(101, s[-1], f"{tier} (n={len(s)})", color=TIER_COLOR[tier], fontsize=9,
                    va="center", ha="left")
        ax.set_yscale("log")
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=floor)
        ax.set_xlabel("problems, sorted by their own speed-up (percentile)", fontsize=9, color=MUTED)
        ax.set_title(legname + "  (dot = the median problem)", fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("speed-up on that problem  (random ÷ ranker, seconds)", fontsize=9, color=MUTED)
    fig.suptitle("Every problem's own speed-up, worst to best — the win is a distribution, not a number",
                 y=0.99, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.93, 0.93))
    fig.savefig(out, dpi=160)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir written by paired_keyhole_compare.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {leg: load(a.data, leg) for leg, _ in LEGS}
    paired_scatter(data, os.path.join(a.out, "paired_scatter_time.png"))
    speedup_curve(data, os.path.join(a.out, "speedup_by_percentile_time.png"))


if __name__ == "__main__":
    main()
