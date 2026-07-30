#!/usr/bin/env python3
"""Plot exact fixed-tier success-vs-simulator-call curves from best-first leaf rows."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np

from namo import eval_sets
from agg_search_eval import load_tiered_rows


HORIZONS = ("1push", "2push")
TIERS = ("easy", "medium", "hard")
SIM_BUDGET = 900
SIM_GRID = np.arange(1, SIM_BUDGET + 1)
SIM_TICKS = (1, 2, 5, 10, 30, 100, 300, 900)
COLORS = {"model": "#0072B2", "random": "#D55E00"}


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


def _success_curve(rows, tier):
    tier_rows = [row for row in rows if row["division"] == tier]
    solved_sims = np.sort([row["sims"] for row in tier_rows if row["solved"]])
    curve = 100.0 * np.searchsorted(solved_sims, SIM_GRID, side="right") / len(tier_rows)
    return curve, len(tier_rows)


def _same_config(left, right):
    left = dict(left)
    right = dict(right)
    left.pop("prior")
    right.pop("prior")
    return left == right


def _load_arm(onepush_dir, twopush_dir, args):
    return load_tiered_rows(
        onepush_dir,
        twopush_dir,
        args.onepush_key,
        args.divisions,
        args.expect_1push,
        args.expect_2push,
    )


def _panel(ax, model, random_seeds, horizon, tier):
    model_curve, n = _success_curve(model[horizon], tier)
    seed_curves = []
    for seed in random_seeds:
        curve, seed_n = _success_curve(seed[horizon], tier)
        if seed_n != n:
            raise RuntimeError(f"episode count differs for {horizon}/{tier}: model={n}, random={seed_n}")
        seed_curves.append(curve)
    seed_curves = np.vstack(seed_curves)
    random_mean = seed_curves.mean(axis=0)
    random_std = seed_curves.std(axis=0, ddof=1)
    ax.fill_between(
        SIM_GRID,
        np.clip(random_mean - random_std, 0.0, 100.0),
        np.clip(random_mean + random_std, 0.0, 100.0),
        color=COLORS["random"],
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(SIM_GRID, random_mean, color=COLORS["random"], linewidth=2.2,
            label="Random (3 seeds, mean ± SD)")
    ax.plot(SIM_GRID, model_curve, color=COLORS["model"], linewidth=2.4, label="Learned ranker")
    ax.set_title(f"{tier.capitalize()}  ·  n={n}")
    ax.set_xscale("log")
    ax.set_xlim(1, SIM_BUDGET)
    ax.set_ylim(0, 103)
    ax.xaxis.set_major_locator(FixedLocator(SIM_TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter([str(cut) for cut in SIM_TICKS]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
    ax.set_xlabel("Simulator calls")
    ax.set_ylabel("Verified success (%)")


def _save(fig, stem):
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {stem}.{{png,pdf}}")


def _single_horizon(model, random_seeds, horizon, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharey=True)
    for ax, tier in zip(axes, TIERS):
        _panel(ax, model, random_seeds, horizon, tier)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"{horizon}: exact verified-success curve · hmax=2",
        fontsize=14,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.25, wspace=0.12)
    _save(fig, out_dir / f"success_vs_sims_{horizon}")


def _combined(model, random_seeds, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.3), sharey=True)
    for row, horizon in enumerate(HORIZONS):
        for col, tier in enumerate(TIERS):
            _panel(axes[row, col], model, random_seeds, horizon, tier)
        axes[row, 0].annotate(
            horizon,
            xy=(-0.30, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="semibold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Exact verified region-opening success vs simulator calls · hmax=2",
        fontsize=15,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.12, hspace=0.38, wspace=0.12)
    _save(fig, out_dir / "success_vs_sims_both_horizons")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-onepush-dir", required=True)
    parser.add_argument("--model-twopush-dir", required=True)
    parser.add_argument("--random-onepush-dirs", required=True, nargs=3)
    parser.add_argument("--random-twopush-dirs", required=True, nargs=3)
    parser.add_argument("--onepush-key", default=str(eval_sets.ONEPUSH))
    parser.add_argument("--divisions", default=str(eval_sets.DIVISIONS))
    parser.add_argument("--expect-1push", type=int, default=eval_sets.EXPECTED["onepush_manifest_episodes"])
    parser.add_argument("--expect-2push", type=int, default=eval_sets.EXPECTED["pure2push_manifest_episodes"])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    model, model_config = _load_arm(args.model_onepush_dir, args.model_twopush_dir, args)
    random_seeds = []
    configs = [model_config]
    for onepush_dir, twopush_dir in zip(args.random_onepush_dirs, args.random_twopush_dirs):
        tiered, config = _load_arm(onepush_dir, twopush_dir, args)
        random_seeds.append(tiered)
        configs.append(config)
    if any(not _same_config(model_config, config) for config in configs[1:]):
        raise RuntimeError("leaf rows do not share one search configuration apart from prior")
    if model_config.get("hmax") != 2:
        raise RuntimeError("expected hmax=2 leaf rows")
    if not model_config.get("dedupe_noop") or not model_config.get("prune_jam_depth"):
        raise RuntimeError("expected no-op dedupe and jam-depth pruning")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    for horizon in HORIZONS:
        _single_horizon(model, random_seeds, horizon, out_dir)
    _combined(model, random_seeds, out_dir)


if __name__ == "__main__":
    main()
