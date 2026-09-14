from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator, PercentFormatter
from scipy.interpolate import PchipInterpolator

from full_namo_sim_exp.experiment_io import Experiment
from full_namo_sim_exp.results import (
    FROZEN_DIFFICULTIES, FROZEN_MODEL_ARM, FROZEN_RANDOM_ARMS,
    FROZEN_RANDOM_SEEDS, FrozenResults, Outcome, load_all_results, load_frozen_results,
)
from full_namo_sim_exp.figure_style import (
    DIFFICULTY_LINESTYLES, METHOD_COLORS, PUBLICATION_DPI,
)


MODEL_COLOR = METHOD_COLORS["PAVE"]
RANDOM_COLOR = METHOD_COLORS["Random"]
GRID_COLOR = "#E5E5E5"
OUTPUT_DPI = 220
TIME_GRID_SIZE = 400
Metric = Literal["simulator_calls", "wall_time_seconds"]
FROZEN_METHODS = ("PAVE", "Random")
FROZEN_CURVE_WIDTH = 3.4
SINGLE_COLUMN_CURVE_WIDTH = 2.2
SINGLE_COLUMN_SIZE = (3.5, 2.35)
LATEX_FONT_PREAMBLE = "\n".join((
    r"\usepackage{amsmath,amssymb,amsfonts}",
    r"\renewcommand{\rmdefault}{ptm}",
    r"\renewcommand{\sfdefault}{phv}",
    r"\renewcommand{\ttdefault}{pcr}",
    r"\AtBeginDocument{\rmfamily\bfseries\boldmath}",
))
DEFAULT_SMOOTH_ANCHORS = 32
SMOOTH_GRID_SIZE = 1500
EARLY_INTEGER_BUDGETS = 10
METRIC_STEMS = {
    "simulator_calls": "success_vs_simulator_budget",
    "wall_time_seconds": "success_vs_wall_clock_budget",
}


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
            "text.usetex": False,
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
    costs = np.asarray([_cost(row, metric) if row.solved else np.inf for row in outcomes.values()])
    return _success_at_budget(costs, thresholds)


def _success_at_budget(costs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Count successful first-solution costs, retaining failures in the denominator."""
    solved = np.sort(costs[np.isfinite(costs)])
    return 100.0 * np.searchsorted(solved, thresholds, side="right") / len(costs)


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


def prepare_frozen_curves(results: FrozenResults, metric: Metric, smooth_anchors: int) -> dict:
    """Build six exact CDFs and their smooth displays on common budget grids.

    Random's median is computed separately for each environment before any
    success-rate calculation. Failures sort as infinity, so at least three
    of its five seeds must solve an environment within a given budget.
    """
    costs = {}
    for difficulty in FROZEN_DIFFICULTIES:
        ids = [scene["geometry_id"] for scene in results.scenes if scene["difficulty"] == difficulty]
        arm_costs = {}
        for arm, outcomes in results.outcomes.items():
            arm_costs[arm] = np.asarray([
                _cost(outcomes[scene_id], metric) if outcomes[scene_id].solved else np.inf
                for scene_id in ids
            ])
        costs[difficulty, "PAVE"] = arm_costs[FROZEN_MODEL_ARM]
        costs[difficulty, "Random"] = np.sort(
            np.stack([arm_costs[arm] for arm in FROZEN_RANDOM_ARMS]), axis=0,
        )[len(FROZEN_RANDOM_ARMS) // 2]
    events = np.concatenate([values[np.isfinite(values)] for values in costs.values()])
    positive = events[events > 0]
    if not positive.size:
        raise ValueError(f"{metric}: a logarithmic budget axis requires a positive success cost")
    lower, upper = ((1.0, float(results.call_cap)) if metric == "simulator_calls"
                    else (float(positive.min()) * 0.8, float(positive.max()) * 1.05))
    exact_grid = np.unique(np.r_[lower, events[(events >= lower) & (events <= upper)], upper])
    anchors = np.geomspace(lower, upper, smooth_anchors)
    if metric == "simulator_calls":
        anchors = np.unique(np.r_[lower, np.arange(1, min(EARLY_INTEGER_BUDGETS, results.call_cap) + 1),
                                  np.rint(anchors), upper])
    display_grid = np.unique(np.r_[np.geomspace(lower, upper, SMOOTH_GRID_SIZE), anchors])
    exact = {key: _success_at_budget(values, exact_grid) for key, values in costs.items()}
    anchor_rates = {key: _success_at_budget(values, anchors) for key, values in costs.items()}
    display = {
        key: PchipInterpolator(np.log10(anchors), rates, extrapolate=False)(np.log10(display_grid))
        for key, rates in anchor_rates.items()
    }
    return {"costs": costs, "exact_grid": exact_grid, "exact": exact,
            "anchors": anchors, "anchor_rates": anchor_rates,
            "display_grid": display_grid, "display": display}


def set_frozen_style(*, single_column: bool = False) -> None:
    """Use the paper's LaTeX Times family and real bfseries at final figure size."""
    set_style()
    # The generic serif entry avoids Matplotlib's "Times" alias, which loads
    # mathptmx and changes the paper's math fonts. Select ptm explicitly in TeX.
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": LATEX_FONT_PREAMBLE,
                         "font.family": "serif", "font.serif": ["serif"],
                         "font.sans-serif": ["sans-serif"], "font.monospace": ["monospace"],
                         "font.size": 11, "axes.labelsize": 11.5, "axes.titlesize": 12,
                         "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.titleweight": "bold", "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "pdf.fonttype": 42})
    if single_column:
        plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5,
                             "axes.titlesize": 9, "xtick.labelsize": 7.5,
                             "ytick.labelsize": 7.5, "axes.linewidth": 0.8})


def _draw_frozen_panel(ax, results: FrozenResults, metric: Metric, curves: dict,
                       tiers: tuple[str, ...], *, overlay: bool = False,
                       linewidth: float = FROZEN_CURVE_WIDTH) -> None:
    """Draw unchanged smoothed costs on the shared metric-specific budget domain."""
    for tier in tiers:
        for method in FROZEN_METHODS:
            ax.plot(curves["display_grid"], curves["display"][tier, method],
                    color=METHOD_COLORS[method], linestyle=DIFFICULTY_LINESTYLES[tier] if overlay else "-",
                    linewidth=linewidth, label=f"{method} — {tier.capitalize()}" if overlay else method,
                    solid_capstyle="round", dash_capstyle="round",
                    zorder=3 if method == "PAVE" else 2)
    lower, upper = curves["display_grid"][[0, -1]]
    ticks = ([1, 10, 100, 1000, results.call_cap] if metric == "simulator_calls"
             else [0.1, 1, 10, 100, 1000, 10000])
    ticks = sorted({value for value in ticks if lower <= value <= upper})
    ax.set_xscale("log")
    ax.set_xlim(lower, upper)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{value / 1000:g}k" if value >= 1000 else f"{value:g}" for value in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, 102)
    # Plain tick text keeps numeric labels in ptm rather than ScalarFormatter's
    # automatic math mode, which would use Computer Modern math digits.
    success_ticks = [0, 20, 40, 60, 80, 100]
    ax.set_yticks(success_ticks, labels=[str(value) for value in success_ticks])
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)


def create_difficulty_figure(results: FrozenResults, metric: Metric, curves: dict,
                             difficulty: str | None = None) -> Figure:
    """Render one metric for a single difficulty, or overlay all three tiers."""
    set_frozen_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.6) if difficulty else (7.2, 5.1))
    tiers = (difficulty,) if difficulty else FROZEN_DIFFICULTIES
    _draw_frozen_panel(ax, results, metric, curves, tiers, overlay=difficulty is None)
    ax.set_ylabel(r"Success rate (\%)" if difficulty else r"Environments solved within difficulty (\%)")
    ax.set_xlabel("Simulated-push budget" if metric == "simulator_calls" else "Wall-clock planning budget (s)")
    ax.set_title(f"{difficulty.capitalize()} · 100 environments" if difficulty
                 else "Full NAMO · 100 environments per difficulty", pad=12)
    methods = [Line2D([], [], color=METHOD_COLORS[method], linewidth=FROZEN_CURVE_WIDTH, label=method) for method in FROZEN_METHODS]
    if difficulty:
        fig.legend(handles=methods, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                   ncol=2, fontsize=10, handlelength=2.5, columnspacing=1.8)
        fig.subplots_adjust(left=0.15, right=0.975, top=0.88, bottom=0.265)
        return fig
    difficulties = [Line2D([], [], color="#444444", linewidth=FROZEN_CURVE_WIDTH, linestyle=DIFFICULTY_LINESTYLES[tier],
                           label=tier.capitalize()) for tier in FROZEN_DIFFICULTIES]
    fig.legend(handles=methods, title="Method", loc="lower center", bbox_to_anchor=(0.29, 0.005),
               ncol=2, fontsize=10, title_fontsize=10, handlelength=2.5, columnspacing=1.3)
    fig.legend(handles=difficulties, title="Difficulty", loc="lower center", bbox_to_anchor=(0.74, 0.005),
               ncol=3, fontsize=10, title_fontsize=10, handlelength=2.5, columnspacing=1.3)
    fig.subplots_adjust(left=0.13, right=0.975, top=0.91, bottom=0.245)
    return fig


def create_hard_pair_figure(results: FrozenResults, by_metric: dict) -> Figure:
    """Compose the two Hard curves at single-column size with a shared y-axis."""
    set_frozen_style(single_column=True)
    fig, axes = plt.subplots(1, 2, figsize=SINGLE_COLUMN_SIZE, sharey=True)
    for ax, metric in zip(axes, METRIC_STEMS):
        _draw_frozen_panel(ax, results, metric, by_metric[metric], ("hard",),
                           linewidth=SINGLE_COLUMN_CURVE_WIDTH)
        ax.tick_params(axis="both", length=3, pad=2)
    axes[0].set_ylabel(r"Success rate (\%)", labelpad=3)
    axes[0].set_xlabel("(a) Simulator-push\nbudget", labelpad=4)
    axes[1].set_xlabel("(b) Wall-clock\nbudget (s)", labelpad=4)
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", left=False)
    count = len(by_metric["simulator_calls"]["costs"]["hard", "PAVE"])
    fig.suptitle(f"Hard · {count} environments", y=0.99, fontsize=9.5, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, fontsize=8.5, handlelength=2.5, columnspacing=1.8, handletextpad=0.6)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.855, bottom=0.37, wspace=0.17)
    return fig


def _write_rows(path: Path, fields: tuple[str, ...], rows) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_frozen(results: FrozenResults, out_dir: Path, smooth_anchors: int = DEFAULT_SMOOTH_ANCHORS,
                  dpi: int = PUBLICATION_DPI,
                  layout: Literal["overlaid", "separate", "hard-pair"] = "overlaid") -> tuple[Path, ...]:
    """Write standalone figures, exact/display CSVs, costs and provenance."""
    if smooth_anchors < 3 or dpi <= 0:
        raise ValueError("smooth_anchors must be at least 3 and dpi must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    by_metric = {metric: prepare_frozen_curves(results, metric, smooth_anchors) for metric in METRIC_STEMS}
    outputs = []
    if layout == "hard-pair":
        figures = [("success_vs_budget_hard", create_hard_pair_figure(results, by_metric))]
    else:
        figures = ((f"{METRIC_STEMS[metric]}_{difficulty or 'by_difficulty'}",
                    create_difficulty_figure(results, metric, curves, difficulty))
                   for metric, curves in by_metric.items()
                   for difficulty in (FROZEN_DIFFICULTIES if layout == "separate" else (None,)))
    for stem, figure in figures:
        for extension in ("png", "pdf"):
            path = out_dir / f"{stem}.{extension}"
            options = {"dpi": dpi} if extension == "png" else {"metadata": {"CreationDate": None, "ModDate": None}}
            figure.savefig(path, bbox_inches=None if layout == "hard-pair" else "tight", facecolor="white", **options)
            outputs.append(path)
        plt.close(figure)
    for kind in ("exact", "display"):
        _write_rows(out_dir / f"{kind}_curves.csv",
                    ("metric", "method", "difficulty", "budget", "success_rate_pct", "environments"),
                    ({"metric": metric, "method": method, "difficulty": tier, "budget": float(budget),
                      "success_rate_pct": float(rate), "environments": len(curves["costs"][tier, method])}
                     for metric, curves in by_metric.items() for (tier, method), rates in curves[kind].items()
                     for budget, rate in zip(curves[f"{kind}_grid"], rates)))
    _write_rows(out_dir / "per_environment_costs.csv",
                ("geometry_id", "difficulty", "template", "method", *METRIC_STEMS),
                ({"geometry_id": scene["geometry_id"], "difficulty": tier, "template": scene["template"], "method": method,
                  **{metric: float(curves["costs"][tier, method][index]) if np.isfinite(curves["costs"][tier, method][index]) else "unresolved"
                     for metric, curves in by_metric.items()}}
                 for tier in FROZEN_DIFFICULTIES
                 for index, scene in enumerate(scene for scene in results.scenes if scene["difficulty"] == tier)
                 for method in FROZEN_METHODS))
    root = Path(__file__).resolve().parent
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    metadata = {
        "campaign": results.campaign, "source": str(results.source), "source_sha256": results.source_sha256,
        "population": len(results.scenes), "counts": {tier: sum(s["difficulty"] == tier for s in results.scenes) for tier in FROZEN_DIFFICULTIES},
        "included_geometry_ids": [s["geometry_id"] for s in results.scenes], "excluded_geometry_ids": results.excluded_geometry_ids,
        "population_filter": "Frozen easy/medium/hard labels only; failed runs within each tier remain included.",
        "call_cap": results.call_cap, "methods": FROZEN_METHODS, "method_colors": METHOD_COLORS,
        "layout": layout, "figures": [path.name for path in outputs],
        "plotted_difficulties": ["hard"] if layout == "hard-pair" else FROZEN_DIFFICULTIES,
        "font_weight": "bold",
        "text_renderer": "LaTeX via Matplotlib text.usetex",
        "text_font_family": "Times Roman (ptm)",
        "latex_font_preamble": LATEX_FONT_PREAMBLE,
        "tex_tools": {tool: subprocess.run([tool, "--version"], capture_output=True, text=True,
                                           check=True).stdout.splitlines()[0] for tool in ("latex", "dvipng")},
        "curve_linewidth_pt": SINGLE_COLUMN_CURVE_WIDTH if layout == "hard-pair" else FROZEN_CURVE_WIDTH,
        "single_column_size_inches": SINGLE_COLUMN_SIZE if layout == "hard-pair" else None,
        "difficulty_linestyles": DIFFICULTY_LINESTYLES if layout == "overlaid" else {tier: "-" for tier in FROZEN_DIFFICULTIES},
        "random_seeds": FROZEN_RANDOM_SEEDS,
        "random_definition": "Median per environment across five seeds for each cost metric, with failures infinite; at least three seeds must solve the environment by a budget.",
        "smoothing": "Monotone PCHIP on common log-budget anchors, with integer push anchors and budgets 1 through 10 preserved. Display approximation between anchors; exact numerical rates in exact_curves.csv.",
        "smooth_anchors_requested": smooth_anchors, "dpi": dpi,
        "anchors": {metric: curves["anchors"].tolist() for metric, curves in by_metric.items()},
        "terminal_successes": {tier: {method: int(np.isfinite(by_metric["simulator_calls"]["costs"][tier, method]).sum()) for method in FROZEN_METHODS} for tier in FROZEN_DIFFICULTIES},
        "campaign_provenance": results.provenance, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("plot.py", "results.py", "figure_style.py")},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"population": metadata["counts"], "terminal_successes": metadata["terminal_successes"]}))
    return tuple(outputs)


def main() -> None:
    """Reproduce frozen-300 difficulty figures without simulation."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--snapshot", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "frozen400_timed_20260913.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="new output directory; existing outputs are not overwritten")
    parser.add_argument("--smooth-anchors", type=int, default=DEFAULT_SMOOTH_ANCHORS)
    parser.add_argument("--dpi", type=int, default=PUBLICATION_DPI)
    parser.add_argument("--layout", choices=("overlaid", "separate", "hard-pair"), default="overlaid",
                        help="two overlaid figures (default), six separate figures, or a single-column Hard comparison")
    args = parser.parse_args()
    for path in render_frozen(load_frozen_results(args.snapshot), args.out_dir, args.smooth_anchors, args.dpi, args.layout):
        print(path)


if __name__ == "__main__":
    main()
