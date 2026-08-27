from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from full_namo_sim_exp.experiment_io import Experiment
from full_namo_sim_exp.results import Outcome, load_all_results


MODEL_COLOR = "#009E73"
RANDOM_COLOR = "#999999"
GRID_COLOR = "#E5E5E5"
OUTPUT_DPI = 220
TIME_GRID_SIZE = 400
Metric = Literal["simulator_calls", "wall_time_seconds"]


@dataclass(frozen=True)
class Curves:
    thresholds: np.ndarray
    model: np.ndarray
    random_mean: np.ndarray


def set_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 17,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#444444",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.frameon": False,
            "legend.fontsize": 15,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _cost(outcome: Outcome, metric: Metric) -> float:
    if metric == "simulator_calls":
        return float(outcome.simulator_calls)
    return outcome.wall_time_seconds


def _thresholds(
    all_results: dict[str, dict[str, Outcome]],
    metric: Metric,
) -> np.ndarray:
    costs = np.asarray(
        [_cost(outcome, metric) for arm in all_results.values() for outcome in arm.values()],
        dtype=float,
    )
    positive = costs[costs > 0]
    if positive.size == 0:
        raise ValueError("logarithmic x-axis requires at least one positive terminal cost")
    if metric == "simulator_calls":
        return np.arange(0, int(costs.max()) + 1, dtype=float)
    return np.concatenate(
        (
            np.asarray([0.0]),
            np.geomspace(float(positive.min()), float(positive.max()), TIME_GRID_SIZE - 1),
        )
    )


def _empirical_curve(
    outcomes: dict[str, Outcome],
    thresholds: np.ndarray,
    metric: Metric,
) -> np.ndarray:
    successful_costs = np.asarray(
        [_cost(outcome, metric) for outcome in outcomes.values() if outcome.solved],
        dtype=float,
    )
    if successful_costs.size == 0:
        return np.zeros_like(thresholds)
    return 100.0 * np.sum(successful_costs[:, None] <= thresholds[None, :], axis=0) / len(
        outcomes
    )


def prepare_curves(experiment: Experiment, metric: Metric) -> Curves:
    all_results = load_all_results(experiment)
    thresholds = _thresholds(all_results, metric)
    model = _empirical_curve(all_results[experiment.model.name], thresholds, metric)
    random_seed_curves = np.stack(
        [
            _empirical_curve(all_results[arm.name], thresholds, metric)
            for arm in experiment.random_arms
        ]
    )
    return Curves(thresholds, model, random_seed_curves.mean(axis=0))


def _label_positions(model: float, random: float) -> tuple[float, float]:
    if abs(model - random) >= 7.0:
        return min(97.0, max(3.0, model)), min(97.0, max(3.0, random))
    midpoint = (model + random) / 2.0
    return min(97.0, midpoint + 3.5), max(3.0, midpoint - 3.5)


def _draw_panel(
    axis: plt.Axes,
    curves: Curves,
    *,
    title: str,
    xlabel: str,
    model_label: str,
    fractions: tuple[str, str],
) -> None:
    axis.plot(
        curves.thresholds,
        curves.model,
        color=MODEL_COLOR,
        linewidth=2.8,
        label=model_label,
        solid_capstyle="round",
        zorder=3,
    )
    axis.plot(
        curves.thresholds,
        curves.random_mean,
        color=RANDOM_COLOR,
        linewidth=2.8,
        label="Random (5-seed mean)",
        solid_capstyle="round",
        zorder=2,
    )
    positive = curves.thresholds[curves.thresholds > 0]
    lower = float(positive[0])
    upper = float(positive[-1])
    axis.set_xscale("log")
    axis.set_xlim(lower, upper if upper > lower else 10.0 * lower)
    axis.set_ylim(-2.0, 104.0)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    axis.set_title(title, pad=3)
    axis.set_xlabel(xlabel)
    axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", pad=2)

    model_y, random_y = _label_positions(float(curves.model[-1]), float(curves.random_mean[-1]))
    transform = axis.get_yaxis_transform()
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.8}
    for fraction, y, color in (
        (fractions[0], model_y, MODEL_COLOR),
        (fractions[1], random_y, RANDOM_COLOR),
    ):
        axis.text(
            0.985,
            y,
            fraction,
            transform=transform,
            ha="right",
            va="center",
            color=color,
            fontsize=10.0,
            fontweight="bold",
            bbox=label_box,
            zorder=5,
        )


def create_figure(experiment: Experiment) -> Figure:
    set_style()
    wall = prepare_curves(experiment, "wall_time_seconds")
    calls = prepare_curves(experiment, "simulator_calls")
    model_successes = round(len(experiment.population.scene_ids) * wall.model[-1] / 100.0)
    pooled_random = round(
        5 * len(experiment.population.scene_ids) * wall.random_mean[-1] / 100.0
    )
    n = len(experiment.population.scene_ids)
    fractions = (f"{model_successes}/{n}", f"{pooled_random}/{5 * n}")

    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.6), sharey=True)
    _draw_panel(
        axes[0],
        wall,
        title="(a) Wall-clock time",
        xlabel="Wall-clock time (s, log scale)",
        model_label=experiment.model.label,
        fractions=fractions,
    )
    _draw_panel(
        axes[1],
        calls,
        title="(b) Simulator calls",
        xlabel="Simulator calls (log scale)",
        model_label=experiment.model.label,
        fractions=fractions,
    )
    axes[0].set_ylabel("Success rate", labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.0,
        columnspacing=1.6,
        handletextpad=0.6,
    )
    figure.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.26, wspace=0.10)
    return figure


def render(experiment: Experiment) -> tuple[Path, Path]:
    experiment.plot_root.mkdir(parents=True, exist_ok=True)
    stem = experiment.plot_root / "full_namo_success_vs_cost"
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure = create_figure(experiment)
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    figure.savefig(png, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pdf, png
