#!/usr/bin/env python3
"""Plot fixed-tier success-vs-simulator-call curves from agg_search_eval.py outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np


HORIZONS = ("1push", "2push")
TIERS = ("easy", "medium", "hard")
COLORS = {"model": "#0072B2", "random": "#D55E00"}


def _load(path):
    with open(path) as stream:
        return json.load(stream)


def _cuts(report, horizon, tier):
    row = report[horizon][tier]
    cuts = sorted(int(key.split("@", 1)[1]) for key in row if key.startswith("solve@"))
    return np.asarray(cuts), np.asarray([row[f"solve@{cut}"] for cut in cuts], dtype=float)


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


def _panel(ax, model, random_reports, horizon, tier):
    cuts, model_curve = _cuts(model, horizon, tier)
    seed_curves = []
    for report in random_reports:
        seed_cuts, seed_curve = _cuts(report, horizon, tier)
        if not np.array_equal(seed_cuts, cuts):
            raise RuntimeError(f"solve@K cuts differ for {horizon}/{tier}")
        if report[horizon][tier]["n"] != model[horizon][tier]["n"]:
            raise RuntimeError(f"episode count differs for {horizon}/{tier}")
        seed_curves.append(seed_curve)
    seed_curves = np.vstack(seed_curves)
    random_mean = seed_curves.mean(axis=0)
    random_std = seed_curves.std(axis=0, ddof=1)
    ax.fill_between(
        cuts,
        np.clip(random_mean - random_std, 0.0, 100.0),
        np.clip(random_mean + random_std, 0.0, 100.0),
        color=COLORS["random"],
        alpha=0.18,
        step="post",
        linewidth=0,
    )
    ax.plot(cuts, random_mean, color=COLORS["random"], linewidth=2.2, marker="o", markersize=3.5,
            drawstyle="steps-post", label="Random (3 seeds, mean ± SD)")
    ax.plot(cuts, model_curve, color=COLORS["model"], linewidth=2.4, marker="o", markersize=3.5,
            drawstyle="steps-post", label="Learned ranker")
    n = model[horizon][tier]["n"]
    ax.set_title(f"{tier.capitalize()}  ·  n={n}")
    ax.set_xscale("log")
    ax.set_xlim(cuts[0], cuts[-1])
    ax.set_ylim(0, 103)
    ax.xaxis.set_major_locator(FixedLocator(cuts))
    ax.xaxis.set_major_formatter(FixedFormatter([str(cut) for cut in cuts]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
    ax.set_xlabel("Simulator calls")
    ax.set_ylabel("Verified success (%)")


def _save(fig, stem):
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {stem}.{{png,pdf}}")


def _single_horizon(model, random_reports, horizon, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharey=True)
    for ax, tier in zip(axes, TIERS):
        _panel(ax, model, random_reports, horizon, tier)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{horizon}: verified success vs simulator calls · hmax=2", fontsize=14, fontweight="semibold")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.25, wspace=0.12)
    _save(fig, out_dir / f"success_vs_sims_{horizon}")


def _combined(model, random_reports, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.3), sharey=True)
    for row, horizon in enumerate(HORIZONS):
        for col, tier in enumerate(TIERS):
            _panel(axes[row, col], model, random_reports, horizon, tier)
        axes[row, 0].annotate(horizon, xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
                              ha="center", va="center", fontsize=13, fontweight="semibold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Verified region openings vs simulator calls · hmax=2", fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.12, hspace=0.38, wspace=0.12)
    _save(fig, out_dir / "success_vs_sims_both_horizons")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="one agg_search_eval.py JSON for the deterministic ranker")
    parser.add_argument("--random", required=True, nargs=3, help="three seeded random agg_search_eval.py JSONs")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    model = _load(args.model)
    random_reports = [_load(path) for path in args.random]
    normalized = []
    for report in [model, *random_reports]:
        config = dict(report["search"])
        config.pop("prior")
        normalized.append(config)
    if any(config != normalized[0] for config in normalized[1:]):
        raise RuntimeError("aggregate JSONs do not share one search configuration apart from prior")
    if model["search"].get("hmax") != 2:
        raise RuntimeError("expected hmax=2 aggregates")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    for horizon in HORIZONS:
        _single_horizon(model, random_reports, horizon, out_dir)
    _combined(model, random_reports, out_dir)


if __name__ == "__main__":
    main()
