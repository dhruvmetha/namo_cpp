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
# Wall-clock twin of the sims axis. Ticks mirror agg_search_eval.TIME_CUTS so the plot and the table
# report the same budgets. The grid's upper end is taken from the data, not fixed: unlike the 900-sim
# budget there is no a-priori time cap. Only valid within ONE pinned-hardware campaign.
TIME_TICKS = (0.5, 1, 2, 5, 10, 30, 60, 120, 300)
COLORS = {"model": "#0072B2", "random": "#D55E00"}
# Okabe-Ito blue/vermillion. Validated (dataviz six checks, light surface): worst-adjacent CVD dE 21.9
# protan / 30.9 tritan, normal-vision 31.2, contrast >=3:1 -- all PASS. Shared with the sims figures on
# purpose: the same entity keeps the same hue across every figure in the paper.


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


def _cost(row, metric):
    """Cost-to-solution for one episode: simulator calls, or wall-clock seconds on identical hardware."""
    if metric == "sims":
        return row["sims"]
    if row.get("timing") is None:
        raise RuntimeError("time axis requested but rows carry no t_wall (pre-instrumentation artifact)")
    return row["timing"]["t_wall"]


def _success_curve(rows, tier, metric="sims", grid=SIM_GRID):
    tier_rows = [row for row in rows if row["division"] == tier]
    solved = np.sort([_cost(row, metric) for row in tier_rows if row["solved"]])
    curve = 100.0 * np.searchsorted(solved, grid, side="right") / len(tier_rows)
    return curve, len(tier_rows)


def _time_grid(arms):
    """Log grid spanning the observed solve times across every arm (no a-priori wall-clock budget)."""
    times = [
        row["timing"]["t_wall"]
        for arm in arms
        for horizon in HORIZONS
        for row in arm[horizon]
        if row["solved"] and row.get("timing")
    ]
    if not times:
        raise RuntimeError("no solved episodes carry timing")
    # Start just below the fastest solve rather than at a fixed floor: nothing resolves in the first
    # ~0.3 s, and a decade of dead axis compresses the region where the arms actually separate.
    return np.geomspace(float(min(times)) * 0.8, float(max(times)) * 1.05, 1500)


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


def _panel(ax, model, random_seeds, horizon, tier, metric="sims", grid=None):
    grid = SIM_GRID if grid is None else grid
    model_curve, n = _success_curve(model[horizon], tier, metric, grid)
    seed_curves = []
    for seed in random_seeds:
        curve, seed_n = _success_curve(seed[horizon], tier, metric, grid)
        if seed_n != n:
            raise RuntimeError(f"episode count differs for {horizon}/{tier}: model={n}, random={seed_n}")
        seed_curves.append(curve)
    seed_curves = np.vstack(seed_curves)
    random_mean = seed_curves.mean(axis=0)
    random_std = seed_curves.std(axis=0, ddof=1)
    ax.fill_between(
        grid,
        np.clip(random_mean - random_std, 0.0, 100.0),
        np.clip(random_mean + random_std, 0.0, 100.0),
        color=COLORS["random"],
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(grid, random_mean, color=COLORS["random"], linewidth=2.2,
            label="Random (3 seeds, mean ± SD)")
    ax.plot(grid, model_curve, color=COLORS["model"], linewidth=2.4, label="Learned ranker")
    ax.set_title(f"{tier.capitalize()}  ·  n={n}")
    ax.set_xscale("log")
    ticks = SIM_TICKS if metric == "sims" else TIME_TICKS
    ax.set_xlim(grid[0] if metric == "time" else 1, grid[-1] if metric == "time" else SIM_BUDGET)
    ax.set_ylim(0, 103)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
    ax.set_xlabel("Simulator calls" if metric == "sims" else "Wall-clock seconds (1 core, pinned icelake)")
    ax.set_ylabel("Verified success (%)")


def _save(fig, stem):
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {stem}.{{png,pdf}}")


def _single_horizon(model, random_seeds, horizon, out_dir, metric="sims", grid=None):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharey=True)
    for ax, tier in zip(axes, TIERS):
        _panel(ax, model, random_seeds, horizon, tier, metric, grid)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"{horizon}: exact verified-success curve · hmax=2",
        fontsize=14,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.25, wspace=0.12)
    _save(fig, out_dir / f"success_vs_{'sims' if metric == 'sims' else 'time'}_{horizon}")


def _combined(model, random_seeds, out_dir, metric="sims", grid=None):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.3), sharey=True)
    for row, horizon in enumerate(HORIZONS):
        for col, tier in enumerate(TIERS):
            _panel(axes[row, col], model, random_seeds, horizon, tier, metric, grid)
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
        "Exact verified region-opening success vs "
        + ("simulator calls" if metric == "sims" else "wall-clock time (1 core, pinned icelake)")
        + " · hmax=2",
        fontsize=15,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.12, hspace=0.38, wspace=0.12)
    _save(fig, out_dir / f"success_vs_{'sims' if metric == 'sims' else 'time'}_both_horizons")


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
    parser.add_argument("--x-axis", default="sims", choices=["sims", "time", "both"],
                        help="cost axis: simulator calls (default, works on any artifact), wall-clock "
                             "seconds (needs instrumented rows), or both")
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
    metrics = ("sims", "time") if args.x_axis == "both" else (args.x_axis,)
    for metric in metrics:
        # One grid shared by every arm and panel of a metric, so curves are directly comparable.
        grid = SIM_GRID if metric == "sims" else _time_grid([model, *random_seeds])
        for horizon in HORIZONS:
            _single_horizon(model, random_seeds, horizon, out_dir, metric, grid)
        _combined(model, random_seeds, out_dir, metric, grid)


if __name__ == "__main__":
    main()
