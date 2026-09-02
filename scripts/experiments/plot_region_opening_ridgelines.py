#!/usr/bin/env python3
"""Plot per-problem region-opening cost distributions behind the manuscript table."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from tabulate_region_opening_costs import (  # noqa: E402
    EXPECTED_COMMON,
    LEAVES,
    load_leaf,
    observation_median,
    per_problem,
    tie_preserving_cuts,
    tier_keys,
)


METHODS = ("HY5U", "Geometric", "Random")
COLORS = {"HY5U": "#2a78d6", "Geometric": "#1b9e77", "Random": "#e46c3a"}
TIER_ORDER = ("hard", "medium", "easy")
TIER_LABEL = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
METRICS = {"sims": "Simulator pushes", "t_wall": "Wall-clock time (s)"}
BACKGROUND = "#fbfbf9"
INK = "#171717"
MUTED = "#5b5b58"
GRID = "#deded9"


def load_costs(args):
    result = {}
    for horizon, leaf in LEAVES.items():
        methods = {
            "HY5U": [
                load_leaf(args.hy5u_root / f"HY5U_s{seed}" / leaf, require_warmup=True)
                for seed in (1, 2, 3)
            ],
            "Random": [
                load_leaf(args.random_root / f"rand_s{seed}" / leaf)
                for seed in (7000, 8000, 9000)
            ],
            "Geometric": [load_leaf(args.geometric_root / leaf)],
        }
        common = set.intersection(*(set(rows) for arm in methods.values() for rows in arm))
        if len(common) != EXPECTED_COMMON[horizon]:
            raise RuntimeError(
                f"{horizon}: common population {len(common)} != {EXPECTED_COMMON[horizon]}"
            )
        random_sims = per_problem(methods["Random"], common, "sims")
        cuts = tie_preserving_cuts(random_sims)
        groups = tier_keys(random_sims, cuts)
        result[horizon] = {
            "cuts": cuts,
            "groups": groups,
            "costs": {
                method: {
                    metric: per_problem(rows, common, metric)
                    for metric in METRICS
                }
                for method, rows in methods.items()
            },
        }
    return result


def smooth_log_density(values: list[float], grid_log: np.ndarray, bandwidth: float) -> np.ndarray:
    log_values = np.log10(np.asarray(values, dtype=np.float64))
    z = (grid_log[:, None] - log_values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).mean(axis=1)
    peak = float(density.max())
    return density / peak if peak > 0 else density


def format_median(value: float, metric: str) -> str:
    if metric == "sims":
        return str(int(value)) if value.is_integer() else f"{value:.1f}"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def panel(ax, horizon: str, metric: str, data: dict, x_limits: tuple[float, float]) -> None:
    ax.set_facecolor(BACKGROUND)
    left, right = x_limits
    grid_log = np.linspace(math.log10(left), math.log10(right), 520)
    grid = 10 ** grid_log
    bandwidth = 0.085 if metric == "sims" else 0.105
    y_positions = []
    y_labels = []

    for tier_index, tier in enumerate(TIER_ORDER):
        for method_index, method in enumerate(METHODS):
            y = tier_index * 4.0 + method_index
            y_positions.append(y)
            y_labels.append(f"{TIER_LABEL[tier]}  ·  {method}")
            observations = [data["costs"][method][metric][key] for key in data["groups"][tier]]
            solved_values = [value for solved, value in observations if solved and value is not None]
            unsolved_pct = 100.0 * (len(observations) - len(solved_values)) / len(observations)
            density = smooth_log_density(solved_values, grid_log, bandwidth)
            height = 0.78 * density
            color = COLORS[method]
            ax.fill_between(grid, y, y + height, color=color, alpha=0.38, linewidth=0)
            ax.plot(grid, y + height, color=color, linewidth=1.35)
            ax.hlines(y, left, right, color="#c8c8c3", linewidth=0.55, zorder=0)

            median_solved, median_value = observation_median(observations)
            if median_solved and median_value is not None:
                median_density = float(np.interp(math.log10(median_value), grid_log, height))
                ax.vlines(
                    median_value,
                    y,
                    y + max(0.24, median_density),
                    color=color,
                    linewidth=2.0,
                    zorder=5,
                )
                ax.text(
                    median_value,
                    y + max(0.24, median_density) + 0.08,
                    format_median(median_value, metric),
                    color=color,
                    fontsize=7.0,
                    ha="center",
                    va="bottom",
                )

            if unsolved_pct > 0:
                marker_x = 10 ** (math.log10(right) - 0.025 * (math.log10(right) - math.log10(left)))
                ax.scatter(
                    [marker_x],
                    [y + 0.2],
                    marker="x",
                    s=22 + 3.0 * unsolved_pct,
                    linewidth=1.4,
                    color=color,
                    zorder=6,
                )
                ax.text(
                    marker_x,
                    y + 0.57,
                    f"{unsolved_pct:.1f}%",
                    color=color,
                    fontsize=6.7,
                    ha="right",
                    va="bottom",
                )

        if tier_index < len(TIER_ORDER) - 1:
            ax.axhline(y + 1.55, color=GRID, linewidth=0.8)

    ax.set_xscale("log")
    ax.set_xlim(left, right)
    ax.set_ylim(-0.45, max(y_positions) + 1.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=7.7, color=INK)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", which="major", color=GRID, linewidth=0.65)
    ax.grid(axis="x", which="minor", color=GRID, linewidth=0.35, alpha=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    title = "One push" if horizon == "1push" else "Two pushes"
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)
    ax.set_xlabel(METRICS[metric] + " · log scale", fontsize=8.5, color=MUTED)


def main() -> None:
    scratch = Path(os.environ["NAMO_SCRATCH"])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hy5u-root",
        type=Path,
        default=scratch / "aquaman/round0/eval_walltime4k_warmup3",
    )
    parser.add_argument(
        "--random-root",
        type=Path,
        default=scratch / "aquaman/round0/eval_walltime4k",
    )
    parser.add_argument(
        "--geometric-root",
        type=Path,
        default=scratch / "aquaman/round0/eval_walltime4k/geometric_region_corrected_v1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/experiments/plots/region_opening_cost_ridgelines"),
        help="Output stem; writes both PNG and PDF.",
    )
    args = parser.parse_args()
    data = load_costs(args)

    solved_time = [
        value
        for horizon in LEAVES
        for method in METHODS
        for solved, value in data[horizon]["costs"][method]["t_wall"].values()
        if solved and value is not None
    ]
    time_limits = (0.1, 10 ** math.ceil(math.log10(max(solved_time))))
    limits = {"sims": (0.8, 5000.0), "t_wall": time_limits}

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 12.0))
    for column, horizon in enumerate(("1push", "2push")):
        panel(axes[0, column], horizon, "sims", data[horizon], limits["sims"])
        panel(axes[1, column], horizon, "t_wall", data[horizon], limits["t_wall"])

    legend = [
        Line2D([0], [0], color=COLORS[method], linewidth=3, label=method)
        for method in METHODS
    ]
    legend.append(Line2D([0], [0], color=INK, marker="x", linestyle="none", label="Unsolved mass"))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.06, top=0.87, wspace=0.27, hspace=0.19)
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.925))
    fig.suptitle(
        "Search effort distributions on the common region-opening test population",
        fontsize=14,
        fontweight="bold",
        color=INK,
        y=0.985,
    )
    fig.text(
        0.5,
        0.958,
        "Ridges show solved episodes; vertical ticks are the table medians; × marks censored mass. "
        "Difficulty is induced by Random's per-problem median simulator cost.",
        ha="center",
        va="top",
        fontsize=8.8,
        color=MUTED,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = args.out.with_suffix(suffix)
        fig.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
