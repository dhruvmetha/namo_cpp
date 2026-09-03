#!/usr/bin/env python3
"""Plot compact method-level ridgelines for region-opening search cost."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from tabulate_region_opening_costs import (  # noqa: E402
    EXPECTED_COMMON,
    LEAVES,
    load_leaf,
    per_problem,
    tie_preserving_cuts,
    tier_keys,
)


METHODS = ("Geometric", "Random", "HY5U")
COLORS = {"HY5U": "#4cc465", "Geometric": "#d9534f", "Random": "#909090"}
METRICS = {
    "sims": "Simulator pushes to first success (log scale)",
    "t_wall": "Wall-clock time to first success (s, log scale)",
}
LIMITS = {"sims": (0.8, 5000.0), "t_wall": (0.05, 2000.0)}
TICKS = {
    "sims": (1, 3, 10, 30, 100, 300, 1000, 3000),
    "t_wall": (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000),
}
INK = "#303030"
AXIS = "#c7c7c7"


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
    """Return a unit-area KDE over log10 cost, conditional on success."""
    log_values = np.log10(np.asarray(values, dtype=np.float64))
    z = (grid_log[:, None] - log_values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).mean(axis=1)
    peak = float(density.max())
    density[density < 0.0015 * peak] = 0.0
    area = float(np.trapezoid(density, grid_log))
    return density / area


def tick_label(value: float, _position: int) -> str:
    return f"{value:g}"


def panel(ax, horizon: str, metric: str, data: dict, *, show_method_labels: bool) -> None:
    left, right = LIMITS[metric]
    grid_log = np.linspace(math.log10(left), math.log10(right), 700)
    grid = 10 ** grid_log
    bandwidth = 0.11 if metric == "sims" else 0.13
    y_by_method = {"Geometric": 1.36, "Random": 0.68, "HY5U": 0.0}
    ridges = {}

    for method in METHODS:
        observations = list(data["costs"][method][metric].values())
        solved_values = [value for solved, value in observations if solved and value is not None]
        unsolved_pct = 100.0 * (len(observations) - len(solved_values)) / len(observations)
        density = smooth_log_density(solved_values, grid_log, bandwidth)
        ridges[method] = (density, unsolved_pct)

    height_scale = 1.12 / max(float(density.max()) for density, _ in ridges.values())
    for method in METHODS:
        y = y_by_method[method]
        density, unsolved_pct = ridges[method]
        height = height_scale * density
        visible = density > 0
        color = COLORS[method]

        ax.fill_between(
            grid,
            y,
            y + height,
            where=visible,
            interpolate=True,
            color=color,
            alpha=0.96,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            grid,
            np.where(visible, y + height, np.nan),
            color="white",
            linewidth=1.15,
            zorder=3,
        )
        if show_method_labels:
            ax.text(
                -0.045,
                y + 0.18,
                method,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9.8,
                fontweight="bold",
                color=INK,
                clip_on=False,
            )
        if unsolved_pct > 0:
            ax.text(
                0.985,
                y + 0.13,
                rf"$\times$ {unsolved_pct:.1f}% unsolved",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=6.8,
                color=INK,
                zorder=4,
            )

    ax.set_xscale("log")
    ax.set_xlim(left, right)
    ax.set_ylim(-0.16, 2.55)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(FixedLocator(TICKS[metric]))
    ax.xaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", colors="#505050", labelsize=7.6, length=4, width=0.8)
    ax.tick_params(axis="x", which="minor", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    title = "One push" if horizon == "1push" else "Two pushes"
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, pad=4)


def plot_metric(metric: str, data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), sharey=True)
    for ax, horizon in zip(axes, ("1push", "2push"), strict=True):
        panel(ax, horizon, metric, data[horizon], show_method_labels=horizon == "1push")

    fig.supxlabel(METRICS[metric], fontsize=9.5, color=INK, y=0.045)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.23, top=0.88, wspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=300 if suffix == ".png" else None, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


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
        help="Output stem; writes simulator-push outputs at the stem and a _wall_time variant.",
    )
    args = parser.parse_args()
    data = load_costs(args)
    plot_metric("sims", data, args.out)
    plot_metric("t_wall", data, args.out.with_name(args.out.name + "_wall_time"))


if __name__ == "__main__":
    main()
