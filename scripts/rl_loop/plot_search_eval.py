#!/usr/bin/env python3
"""Plot exact fixed-tier success curves from matched best-first leaf rows."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np

from namo import eval_sets
from agg_search_eval import ONEPUSH_CUTS, TWOPUSH_CUTS, _summarize, load_tiered_rows


HORIZONS = ("1push", "2push")
TIERS = ("easy", "medium", "hard")
SIM_TICKS = (1, 2, 5, 10, 30, 100, 300, 900, 4000, 10000)
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
    for key in ("prior", "model_warmup_repeats"):
        left.pop(key, None)
        right.pop(key, None)
    return left == right


def _consistent_warmup_repeats(configs, label):
    values = {
        int(config.get("model_warmup_repeats", 0)) if config.get("prior") == "model" else 0
        for config in configs
    }
    if len(values) != 1:
        raise RuntimeError(f"model warm-up differs within {label}: {sorted(values)}")
    return values.pop()


def _load_arm(onepush_dir, twopush_dir, args):
    expect_onepush = None if args.common_episode_set else args.expect_1push
    expect_twopush = None if args.common_episode_set else args.expect_2push
    return load_tiered_rows(
        onepush_dir,
        twopush_dir,
        args.onepush_key,
        args.divisions,
        expect_onepush,
        expect_twopush,
    )


def _restrict_to_common_episodes(arms):
    """Score every arm on the exact episode intersection, separately by horizon."""
    common = {}
    for horizon in HORIZONS:
        episode_sets = [{row["episode"] for row in arm[horizon]} for arm in arms]
        common[horizon] = set.intersection(*episode_sets)
        for arm in arms:
            arm[horizon] = [row for row in arm[horizon] if row["episode"] in common[horizon]]
    return common


def _aggregate_series(item):
    result = {"seed_count": len(item["arms"])}
    for horizon, cuts in (("1push", ONEPUSH_CUTS), ("2push", TWOPUSH_CUTS)):
        seed_reports = [_summarize(arm[horizon], cuts) for arm in item["arms"]]
        result[horizon] = {}
        for tier in (*TIERS, "all"):
            reports = [report[tier] for report in seed_reports]
            ns = {report["n"] for report in reports}
            if len(ns) != 1:
                raise RuntimeError(f"episode count differs within {item['label']} {horizon}/{tier}: {ns}")
            summary = {"n": ns.pop()}
            for metric in reports[0]:
                if metric == "n":
                    continue
                values = [report[metric] for report in reports]
                if any(value is None for value in values):
                    summary[metric] = None
                    continue
                summary[metric] = {
                    "mean": round(float(np.mean(values)), 3),
                    "sd": round(float(np.std(values, ddof=1)), 3) if len(values) > 1 else 0.0,
                }
            result[horizon][tier] = summary
    return result


def _write_aggregate_json(path, series, common, args, reference_config):
    shared_search = dict(reference_config)
    shared_search.pop("prior")
    shared_search.pop("model_warmup_repeats", None)
    payload = {
        "population": "common episode intersection" if args.common_episode_set else "per-arm population",
        "common_episodes": {horizon: len(common[horizon]) for horizon in HORIZONS} if common else None,
        "excluded_from_canonical": {
            "1push": args.expect_1push - len(common["1push"]),
            "2push": args.expect_2push - len(common["2push"]),
        } if common else None,
        "hardware": args.hardware_label,
        "shared_search": shared_search,
        "series": {
            item["key"]: {
                "label": item["label"],
                "model_warmup_repeats": item["model_warmup_repeats"],
                **_aggregate_series(item),
            }
            for item in series
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {path}")


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
                alpha=item.get("band_alpha", 0.16),
                linewidth=0,
            )
        suffix = (
            f" ({len(item['arms'])} seeds, mean ± SD)"
            if len(item["arms"]) > 1 and item.get("show_seed_count", True)
            else ""
        )
        ax.plot(grid, mean, color=item["color"], linewidth=item["linewidth"],
                linestyle=item["linestyle"], label=item["label"] + suffix,
                zorder=item.get("zorder", 2))
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
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(series)),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"{horizon}: exact verified-success curve · hmax=2",
        fontsize=14,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82,
                        bottom=0.31 if len(series) > 3 else 0.25, wspace=0.12)
    _save(fig, out_dir / f"success_vs_{'sims' if metric == 'sims' else 'time'}_{horizon}")


def _combined(series, out_dir, metric, grid, sim_budget, hardware_label, figure_title=None):
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
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(series)),
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        figure_title or (
            "Exact verified region-opening success vs "
            + ("simulator calls" if metric == "sims" else f"wall-clock time ({hardware_label})")
            + " · hmax=2"
        ),
        fontsize=15,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90,
                        bottom=0.17 if len(series) > 3 else 0.12,
                        hspace=0.38, wspace=0.12)
    _save(fig, out_dir / f"success_vs_{'sims' if metric == 'sims' else 'time'}_both_horizons")


def _solve_at(rows, tier, budget):
    selected = rows if tier == "all" else [row for row in rows if row["division"] == tier]
    if not selected:
        raise RuntimeError(f"no rows for tier {tier}")
    return 100.0 * sum(row["solved"] and row["sims"] <= budget for row in selected) / len(selected)


def _paired_deltas(reference, comparison, horizon, tier, budget):
    if len(reference) != len(comparison):
        raise RuntimeError(
            f"paired delta requires equal seed counts: {len(reference)} != {len(comparison)}"
        )
    return np.asarray([
        _solve_at(arm[horizon], tier, budget) - _solve_at(ref_arm[horizon], tier, budget)
        for ref_arm, arm in zip(reference, comparison)
    ])


def _delta_plot(series, reference_key, out_dir):
    references = [item for item in series if item["key"] == reference_key]
    if len(references) != 1:
        raise RuntimeError(f"delta reference {reference_key!r} matched {len(references)} series")
    reference = references[0]["arms"]
    comparisons = sorted(
        (item for item in series if item.get("delta", False)),
        key=lambda item: item.get("delta_order", 0),
    )
    if not comparisons:
        raise RuntimeError("delta plot requested but no series has delta=true")

    tier_styles = {
        "easy": ("#0072B2", "Easy"),
        "medium": ("#009E73", "Medium"),
        "hard": ("#D55E00", "Hard"),
        "all": ("#222222", "Overall"),
    }
    metrics = (("1push", 1, "One push: solve@1"), ("2push", 5, "Two pushes: solve@5"))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), sharey=True)
    ybase = np.arange(len(comparisons), dtype=float)
    offsets = dict(zip(tier_styles, np.linspace(-0.24, 0.24, len(tier_styles))))

    for ax, (horizon, budget, title) in zip(axes, metrics):
        observed = []
        for tier, (color, _) in tier_styles.items():
            for index, item in enumerate(comparisons):
                deltas = _paired_deltas(reference, item["arms"], horizon, tier, budget)
                mean = float(deltas.mean())
                sd = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
                observed.extend((mean - sd, mean + sd))
                ax.errorbar(
                    mean,
                    ybase[index] + offsets[tier],
                    xerr=sd,
                    fmt="o",
                    color=color,
                    markersize=5.5,
                    capsize=2.5,
                    linewidth=1.25,
                    zorder=3,
                )
        lower = min(min(observed), 0.0)
        upper = max(max(observed), 0.0)
        pad = max(1.2, 0.10 * (upper - lower))
        ax.set_xlim(lower - pad, upper + pad)
        ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--", zorder=1)
        ax.grid(axis="x", color="#E1E1E1", linewidth=0.7)
        ax.set_title(title, fontsize=13, fontweight="semibold")
        ax.set_xlabel("Change from HY5U (percentage points)")
        ax.set_yticks(ybase)
        ax.set_yticklabels([item["label"] for item in comparisons])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].invert_yaxis()

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=label, markersize=6)
        for color, label in tier_styles.values()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005),
               frameon=False)
    fig.suptitle("Effect of removing each HY5U component · paired seeds, mean ± SD",
                 fontsize=15, fontweight="semibold")
    fig.subplots_adjust(left=0.19, right=0.99, top=0.84, bottom=0.19, wspace=0.16)
    _save(fig, out_dir / "ablation_effects")


def _load_series_spec(path, args):
    payload = json.load(open(path))
    series = []
    configs = []
    keys = set()
    for spec in payload["series"]:
        key = spec["key"]
        if key in keys:
            raise RuntimeError(f"duplicate series key {key!r}")
        keys.add(key)
        onepush_dirs = [os.path.expandvars(path) for path in spec["onepush_dirs"]]
        twopush_dirs = [os.path.expandvars(path) for path in spec["twopush_dirs"]]
        if len(onepush_dirs) != len(twopush_dirs):
            raise RuntimeError(f"{key} 1push/2push seed counts differ")
        arms = []
        arm_configs = []
        for onepush_dir, twopush_dir in zip(onepush_dirs, twopush_dirs):
            tiered, config = _load_arm(onepush_dir, twopush_dir, args)
            arms.append(tiered)
            arm_configs.append(config)
        configs.extend(arm_configs)
        series.append({
            "key": key,
            "label": spec["label"],
            "arms": arms,
            "color": spec["color"],
            "linewidth": float(spec.get("linewidth", 2.0)),
            "linestyle": spec.get("linestyle", "-"),
            "band_alpha": float(spec.get("band_alpha", 0.10)),
            "zorder": int(spec.get("zorder", 2)),
            "show_seed_count": bool(spec.get("show_seed_count", False)),
            "delta": bool(spec.get("delta", False)),
            "delta_order": int(spec.get("delta_order", 0)),
            "model_warmup_repeats": _consistent_warmup_repeats(
                arm_configs, spec["label"]
            ),
        })
    return series, configs


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
                        "model_warmup_repeats": item["model_warmup_repeats"],
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
    parser.add_argument("--series-spec", help="JSON definition for an arbitrary set of seed groups")
    parser.add_argument("--model-onepush-dir", nargs="+")
    parser.add_argument("--model-twopush-dir", nargs="+")
    parser.add_argument("--model-label", default="Learned ranker")
    parser.add_argument("--comparison-onepush-dir")
    parser.add_argument("--comparison-twopush-dir")
    parser.add_argument("--comparison-label", default="Geometric heuristic")
    parser.add_argument("--random-onepush-dirs", nargs=3)
    parser.add_argument("--random-twopush-dirs", nargs=3)
    parser.add_argument("--onepush-key", default=str(eval_sets.ONEPUSH))
    parser.add_argument("--divisions", default=str(eval_sets.DIVISIONS))
    parser.add_argument("--expect-1push", type=int, default=eval_sets.EXPECTED["onepush_manifest_episodes"])
    parser.add_argument("--expect-2push", type=int, default=eval_sets.EXPECTED["pure2push_manifest_episodes"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hardware-label", default="1 core, pinned icelake")
    parser.add_argument("--curve-json", help="optional compact curve data for an interactive plot")
    parser.add_argument("--aggregate-json", help="optional fixed-budget tables for every plotted series")
    parser.add_argument("--delta-reference-key",
                        help="series key used as zero for the paired component-removal plot")
    parser.add_argument("--figure-title", help="override the combined figure title")
    parser.add_argument("--common-episode-set", action="store_true",
                        help="restrict every seed and method to the exact episode intersection")
    parser.add_argument("--x-axis", default="sims", choices=["sims", "time", "both"],
                        help="cost axis: simulator calls (default, works on any artifact), wall-clock "
                             "seconds (needs instrumented rows), or both")
    args = parser.parse_args()

    if args.series_spec:
        legacy_args = (
            args.model_onepush_dir, args.model_twopush_dir,
            args.comparison_onepush_dir, args.comparison_twopush_dir,
            args.random_onepush_dirs, args.random_twopush_dirs,
        )
        if any(legacy_args):
            raise RuntimeError("--series-spec cannot be mixed with the legacy series arguments")
        series, configs = _load_series_spec(args.series_spec, args)
    else:
        if not args.model_onepush_dir or not args.model_twopush_dir:
            raise RuntimeError("model directories are required unless --series-spec is used")
        if len(args.model_onepush_dir) != len(args.model_twopush_dir):
            raise RuntimeError("model 1push/2push seed counts differ")
        model_seeds = []
        model_configs = []
        for onepush_dir, twopush_dir in zip(args.model_onepush_dir, args.model_twopush_dir):
            tiered, config = _load_arm(onepush_dir, twopush_dir, args)
            model_seeds.append(tiered)
            model_configs.append(config)
        if bool(args.comparison_onepush_dir) != bool(args.comparison_twopush_dir):
            raise RuntimeError("comparison requires both 1push and 2push directories")
        if bool(args.random_onepush_dirs) != bool(args.random_twopush_dirs):
            raise RuntimeError("random comparison requires both 1push and 2push directories")
        comparison = []
        comparison_configs = []
        if args.comparison_onepush_dir:
            tiered, config = _load_arm(args.comparison_onepush_dir, args.comparison_twopush_dir, args)
            comparison.append(tiered)
            comparison_configs.append(config)
        random_seeds = []
        random_configs = []
        for onepush_dir, twopush_dir in zip(
            args.random_onepush_dirs or [], args.random_twopush_dirs or []
        ):
            tiered, config = _load_arm(onepush_dir, twopush_dir, args)
            random_seeds.append(tiered)
            random_configs.append(config)
        configs = model_configs + comparison_configs + random_configs
        series = [
            {"key": "model", "label": args.model_label, "arms": model_seeds,
             "color": COLORS["model"], "linewidth": 2.4, "linestyle": "-",
             "model_warmup_repeats": _consistent_warmup_repeats(
                 model_configs, args.model_label)},
        ]
        if comparison:
            series.append(
                {"key": "comparison", "label": args.comparison_label, "arms": comparison,
                 "color": COLORS["comparison"], "linewidth": 2.3, "linestyle": "-",
                 "model_warmup_repeats": _consistent_warmup_repeats(
                     comparison_configs, args.comparison_label)}
            )
        if random_seeds:
            series.append(
                {"key": "random", "label": "Random", "arms": random_seeds,
                 "color": COLORS["random"], "linewidth": 2.2, "linestyle": "--",
                 "model_warmup_repeats": _consistent_warmup_repeats(
                     random_configs, "Random")}
            )

    if not configs:
        raise RuntimeError("no series loaded")
    reference_config = configs[0]
    if any(not _same_config(reference_config, config) for config in configs[1:]):
        raise RuntimeError("leaf rows do not share one search configuration apart from prior")
    if reference_config.get("hmax") != 2:
        raise RuntimeError("expected hmax=2 leaf rows")
    if not reference_config.get("dedupe_noop") or not reference_config.get("prune_jam_depth"):
        raise RuntimeError("expected no-op dedupe and jam-depth pruning")
    sim_budget = int(reference_config["sim_budget"])
    all_arms = [arm for item in series for arm in item["arms"]]
    common = _restrict_to_common_episodes(all_arms) if args.common_episode_set else None
    if common:
        print("common episode set: " + ", ".join(
            f"{horizon}={len(common[horizon])}" for horizon in HORIZONS))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    metrics = ("sims", "time") if args.x_axis == "both" else (args.x_axis,)
    grids = {}
    for metric in metrics:
        # One grid shared by every arm and panel of a metric, so curves are directly comparable.
        grid = np.arange(1, sim_budget + 1) if metric == "sims" else _time_grid(all_arms)
        grids[metric] = grid
        for horizon in HORIZONS:
            _single_horizon(series, horizon, out_dir, metric, grid, sim_budget, args.hardware_label)
        _combined(series, out_dir, metric, grid, sim_budget, args.hardware_label,
                  args.figure_title)
    if args.delta_reference_key:
        if args.x_axis != "sims":
            raise RuntimeError("paired ablation effects are defined on simulator-call budgets")
        _delta_plot(series, args.delta_reference_key, out_dir)
    if args.curve_json:
        if set(grids) != {"sims", "time"}:
            raise RuntimeError("--curve-json requires --x-axis both")
        _write_curve_json(args.curve_json, series, grids, sim_budget, args.hardware_label)
    if args.aggregate_json:
        _write_aggregate_json(args.aggregate_json, series, common, args, reference_config)


if __name__ == "__main__":
    main()
