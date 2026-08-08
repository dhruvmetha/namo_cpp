#!/usr/bin/env python3
"""How each arm actually solved: share of episodes closed in ONE push vs TWO, by tier and horizon.

The canonical eval runs hmax=2 on BOTH horizons, so a "1push" episode may be closed by a setup+finish
chain. That is the right task metric (opening the region is the goal) but it is NOT a depth-1 ranking
measurement -- and the two arms differ enormously in which route they take. This figure makes that
visible instead of leaving it inside the aggregate.

Hue = arm (same blue/vermillion as the success-vs-cost figures: colour follows the entity, never rank).
Hatch = the depth-2 share, a secondary encoding so the split survives greyscale and CVD.
"""
import argparse
import json
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from namo import eval_sets
from agg_search_eval import (
    _canonical_xml,
    _load_divisions,
    _normalize_tier,
    _onepush_divisions,
)

HORIZONS = ("1push", "2push")
TIERS = ("easy", "medium", "hard")
COLORS = {"model": "#0072B2", "random": "#D55E00"}
LABEL = {"model": "Learned ranker", "random": "Random"}


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#333333",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    })


def _shares(directory, horizon, divisions):
    """-> {tier: (pct solved in 1 push, pct solved in 2 pushes)} over ALL episodes of the tier."""
    counts = defaultdict(lambda: {1: 0, 2: 0, "n": 0})
    for path in glob.glob(str(Path(directory) / "shard_*.jsonl")):
        for line in open(path):
            row = json.loads(line)
            if horizon == "1push":
                key = (_canonical_xml(row.get("xml_full", row["xml"])), row["object_id"])
            else:
                key = (_canonical_xml(row["xml"]), row["object_id"], row.get("region"))
            division = divisions.get(key)
            if division is None:
                continue
            tier = _normalize_tier(division)
            counts[tier]["n"] += 1
            if row["solved"]:
                counts[tier][int(row["plan_len"])] += 1
    return {
        tier: (100.0 * c[1] / c["n"], 100.0 * c[2] / c["n"])
        for tier, c in counts.items()
        if c["n"]
    }


def _panel(ax, data, tier, horizon):
    xs = np.arange(2)
    for i, arm in enumerate(("model", "random")):
        one, two = data[arm][tier]
        ax.bar(xs[i], one, width=0.62, color=COLORS[arm], linewidth=0)
        ax.bar(xs[i], two, width=0.62, bottom=one, color=COLORS[arm], alpha=0.42,
               hatch="///", edgecolor="white", linewidth=0.0)
        ax.text(xs[i], one + two + 2.0, f"{one:.0f}/{two:.0f}", ha="center", va="bottom",
                fontsize=9.5, color="#333333")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL["model"], LABEL["random"]], fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(f"{tier.capitalize()}", fontsize=12)
    ax.set_ylabel(f"{horizon} episodes solved (%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="campaign root holding {model,random_s*}_{1,2}push/")
    parser.add_argument("--random-seed", default="7000", help="which random seed to display")
    parser.add_argument("--onepush-key", default=str(eval_sets.ONEPUSH))
    parser.add_argument("--divisions", default=str(eval_sets.DIVISIONS))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    div = {"1push": _onepush_divisions(args.onepush_key), "2push": _load_divisions(args.divisions)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.0))
    for r, horizon in enumerate(HORIZONS):
        data = {
            "model": _shares(root / f"model_{horizon}", horizon, div[horizon]),
            "random": _shares(root / f"random_s{args.random_seed}_{horizon}", horizon, div[horizon]),
        }
        for c, tier in enumerate(TIERS):
            _panel(axes[r, c], data, tier, horizon)
        for c in (1, 2):
            axes[r, c].set_ylabel("")
    solid = plt.Rectangle((0, 0), 1, 1, facecolor="#777777", linewidth=0)
    hatched = plt.Rectangle((0, 0), 1, 1, facecolor="#777777", alpha=0.42, hatch="///",
                            edgecolor="white", linewidth=0.0)
    fig.legend([solid, hatched], ["closed in ONE push", "closed in TWO pushes (setup + finish)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("How the region was actually opened · hmax=2 search on both horizons",
                 fontsize=14, fontweight="semibold")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.11, hspace=0.32, wspace=0.18)
    stem = out_dir / "plan_depth_share"
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {stem}.{{png,pdf}}")


if __name__ == "__main__":
    main()
