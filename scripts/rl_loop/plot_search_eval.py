#!/usr/bin/env python3
"""Plot exact fixed-tier success curves from matched best-first leaf rows."""
import argparse
import json
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
SIM_TICKS = (1, 5, 30, 300, 4000, 10000)
# Wall-clock plots are valid only within one pinned-hardware campaign. Keep the visible log ticks sparse;
# the exact table budgets remain in agg_search_eval.TIME_CUTS.
TIME_TICKS = (0.1, 0.5, 2, 10, 60, 300, 1200)
COLORS = {"model": "#0072B2", "comparison": "#009E73", "random": "#D55E00"}
# Okabe-Ito blue/green/vermillion. The same entity keeps the same hue across both cost axes.


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


def _success_curve(rows, tier, metric, grid):
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


def _curve_stats(arms, horizon, tier, metric, grid):
    curves = []
    n = None
    for arm in arms:
        curve, arm_n = _success_curve(arm[horizon], tier, metric, grid)
        if n is not None and arm_n != n:
            raise RuntimeError(f"episode count differs for {horizon}/{tier}: {n} != {arm_n}")
        n = arm_n
        curves.append(curve)
    curves = np.vstack(curves)
    std = curves.std(axis=0, ddof=1) if len(arms) > 1 else np.zeros_like(curves[0])
    return curves.mean(axis=0), std, n


def _panel(ax, series, horizon, tier, metric, grid, sim_budget, hardware_label):
    expected_n = None
    for item in series:
        mean, std, n = _curve_stats(item["arms"], horizon, tier, metric, grid)
        if expected_n is not None and n != expected_n:
            raise RuntimeError(f"episode count differs for {horizon}/{tier}: {expected_n} != {n}")
        expected_n = n
        if len(item["arms"]) > 1:
            ax.fill_between(
                grid,
                np.clip(mean - std, 0.0, 100.0),
                np.clip(mean + std, 0.0, 100.0),
                color=item["color"],
                alpha=0.16,
                linewidth=0,
            )
        suffix = f" ({len(item['arms'])} seeds, mean ± SD)" if len(item["arms"]) > 1 else ""
        ax.plot(grid, mean, color=item["color"], linewidth=item["linewidth"],
                linestyle=item["linestyle"], label=item["label"] + suffix)
    n = expected_n
    ax.set_title(f"{tier.capitalize()}  ·  n={n}")
    ax.set_xscale("log")
    ticks = [tick for tick in (SIM_TICKS if metric == "sims" else TIME_TICKS)
             if grid[0] <= tick <= grid[-1]]
    if metric == "sims" and sim_budget not in ticks:
        ticks.append(sim_budget)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 103)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
    ax.set_xlabel("Simulator calls" if metric == "sims" else "Wall-clock time (s)")
    ax.set_ylabel("Verified success (%)")


def _save(fig, stem):
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {stem}.{{png,pdf}}")


def _single_horizon(series, horizon, out_dir, metric, grid, sim_budget, hardware_label):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharey=True)
    for ax, tier in zip(axes, TIERS):
        _panel(ax, series, horizon, tier, metric, grid, sim_budget, hardware_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"{horizon}: exact verified-success curve · hmax=2",
        fontsize=14,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.25, wspace=0.12)
    _save(fig, out_dir / f"success_vs_{'sims' if metric == 'sims' else 'time'}_{horizon}")


def _combined(series, out_dir, metric, grid, sim_budget, hardware_label):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.3), sharey=True)
    for row, horizon in enumerate(HORIZONS):
        for col, tier in enumerate(TIERS):
            _panel(axes[row, col], series, horizon, tier, metric, grid, sim_budget, hardware_label)
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
    fig.legend(handles, labels, loc="lower center", ncol=len(series), bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Exact verified region-opening success vs "
        + ("simulator calls" if metric == "sims" else f"wall-clock time ({hardware_label})")
        + " · hmax=2",
        fontsize=15,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.12, hspace=0.38, wspace=0.12)
    _save(fig, out_dir / f"success_vs_{'sims' if metric == 'sims' else 'time'}_both_horizons")


def _write_curve_json(path, series, grids, sim_budget, hardware_label):
    payload = {
        "sim_budget": sim_budget,
        "hardware_label": hardware_label,
        "metrics": {},
    }
    for metric, full_grid in grids.items():
        if metric == "sims":
            grid = np.unique(np.concatenate([
                np.geomspace(1, sim_budget, min(800, sim_budget)).round().astype(int),
                np.asarray([tick for tick in SIM_TICKS if tick <= sim_budget], dtype=int),
            ])).astype(float)
        else:
            indices = np.unique(np.linspace(0, len(full_grid) - 1, min(600, len(full_grid))).round().astype(int))
            grid = full_grid[indices]
        panels = {}
        for horizon in HORIZONS:
            panels[horizon] = {}
            for tier in TIERS:
                values = {}
                n = None
                for item in series:
                    mean, std, n = _curve_stats(item["arms"], horizon, tier, metric, grid)
                    values[item["key"]] = {
                        "label": item["label"],
                        "color": item["color"],
                        "mean": np.round(mean, 4).tolist(),
                        "std": np.round(std, 4).tolist(),
                        "seed_count": len(item["arms"]),
                    }
                panels[horizon][tier] = {"n": n, "series": values}
        payload["metrics"][metric] = {"grid": np.round(grid, 5).tolist(), "panels": panels}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-onepush-dir", required=True, nargs="+")
    parser.add_argument("--model-twopush-dir", required=True, nargs="+")
    parser.add_argument("--model-label", default="Learned ranker")
    parser.add_argument("--comparison-onepush-dir")
    parser.add_argument("--comparison-twopush-dir")
    parser.add_argument("--comparison-label", default="Geometric heuristic")
    parser.add_argument("--random-onepush-dirs", required=True, nargs=3)
    parser.add_argument("--random-twopush-dirs", required=True, nargs=3)
    parser.add_argument("--onepush-key", default=str(eval_sets.ONEPUSH))
    parser.add_argument("--divisions", default=str(eval_sets.DIVISIONS))
    parser.add_argument("--expect-1push", type=int, default=eval_sets.EXPECTED["onepush_manifest_episodes"])
    parser.add_argument("--expect-2push", type=int, default=eval_sets.EXPECTED["pure2push_manifest_episodes"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hardware-label", default="1 core, pinned icelake")
    parser.add_argument("--curve-json", help="optional compact curve data for an interactive plot")
    parser.add_argument("--x-axis", default="sims", choices=["sims", "time", "both"],
                        help="cost axis: simulator calls (default, works on any artifact), wall-clock "
                             "seconds (needs instrumented rows), or both")
    args = parser.parse_args()

    if len(args.model_onepush_dir) != len(args.model_twopush_dir):
        raise RuntimeError("model 1push/2push seed counts differ")
    model_seeds = []
    configs = []
    for onepush_dir, twopush_dir in zip(args.model_onepush_dir, args.model_twopush_dir):
        tiered, config = _load_arm(onepush_dir, twopush_dir, args)
        model_seeds.append(tiered)
        configs.append(config)
    if bool(args.comparison_onepush_dir) != bool(args.comparison_twopush_dir):
        raise RuntimeError("comparison requires both 1push and 2push directories")
    comparison = []
    if args.comparison_onepush_dir:
        tiered, config = _load_arm(args.comparison_onepush_dir, args.comparison_twopush_dir, args)
        comparison.append(tiered)
        configs.append(config)
    random_seeds = []
    for onepush_dir, twopush_dir in zip(args.random_onepush_dirs, args.random_twopush_dirs):
        tiered, config = _load_arm(onepush_dir, twopush_dir, args)
        random_seeds.append(tiered)
        configs.append(config)
    reference_config = configs[0]
    if any(not _same_config(reference_config, config) for config in configs[1:]):
        raise RuntimeError("leaf rows do not share one search configuration apart from prior")
    if reference_config.get("hmax") != 2:
        raise RuntimeError("expected hmax=2 leaf rows")
    if not reference_config.get("dedupe_noop") or not reference_config.get("prune_jam_depth"):
        raise RuntimeError("expected no-op dedupe and jam-depth pruning")
    sim_budget = int(reference_config["sim_budget"])
    series = [
        {"key": "model", "label": args.model_label, "arms": model_seeds,
         "color": COLORS["model"], "linewidth": 2.4, "linestyle": "-"},
    ]
    if comparison:
        series.append(
            {"key": "comparison", "label": args.comparison_label, "arms": comparison,
             "color": COLORS["comparison"], "linewidth": 2.3, "linestyle": "-"}
        )
    series.append(
        {"key": "random", "label": "Random", "arms": random_seeds,
         "color": COLORS["random"], "linewidth": 2.2, "linestyle": "--"}
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    metrics = ("sims", "time") if args.x_axis == "both" else (args.x_axis,)
    all_arms = [arm for item in series for arm in item["arms"]]
    grids = {}
    for metric in metrics:
        # One grid shared by every arm and panel of a metric, so curves are directly comparable.
        grid = np.arange(1, sim_budget + 1) if metric == "sims" else _time_grid(all_arms)
        grids[metric] = grid
        for horizon in HORIZONS:
            _single_horizon(series, horizon, out_dir, metric, grid, sim_budget, args.hardware_label)
        _combined(series, out_dir, metric, grid, sim_budget, args.hardware_label)
    if args.curve_json:
        if set(grids) != {"sims", "time"}:
            raise RuntimeError("--curve-json requires --x-axis both")
        _write_curve_json(args.curve_json, series, grids, sim_budget, args.hardware_label)


if __name__ == "__main__":
    main()
