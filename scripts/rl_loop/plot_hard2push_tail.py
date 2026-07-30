#!/usr/bin/env python3
"""Plot the exhaustive-GT hard-2push model tail beyond the random 900-call cap."""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np

from namo.paths import resolve


MODEL_COLOR = "#0072B2"
RANDOM_COLOR = "#D55E00"


def _read_jsonl(path_or_dir):
    paths = [path_or_dir] if Path(path_or_dir).is_file() else sorted(
        glob.glob(str(Path(path_or_dir) / "shard_*.jsonl"))
    )
    rows = []
    for path in paths:
        with open(path) as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def _key(row):
    return str(resolve(row["xml"])), row["object_id"], row.get("region")


def _hard_keys(path):
    raw = json.load(open(path))
    return {
        (str(resolve(xml)), row["object_id"], row.get("region"))
        for xml, records in raw.items()
        for row in records
        if row["division"] == "hard"
    }


def _curve(rows, grid):
    solved = np.sort([row["sims"] for row in rows if row["solved"]])
    return 100.0 * np.searchsorted(solved, grid, side="right") / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-spliced", required=True)
    parser.add_argument("--random-dirs", required=True, nargs=3)
    parser.add_argument("--divisions", required=True)
    parser.add_argument("--base-budget", type=int, default=900)
    parser.add_argument("--tail-budget", type=int, default=10000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    hard_keys = _hard_keys(args.divisions)
    model = _read_jsonl(args.model_spliced)
    random_seeds = [
        [row for row in _read_jsonl(directory) if _key(row) in hard_keys]
        for directory in args.random_dirs
    ]
    if len(model) != len(hard_keys) or any(len(rows) != len(hard_keys) for rows in random_seeds):
        raise RuntimeError("model/random rows do not match the exhaustive-GT hard tier")

    model_grid = np.arange(1, args.tail_budget + 1)
    random_grid = np.arange(1, args.base_budget + 1)
    model_curve = _curve(model, model_grid)
    seed_curves = np.vstack([_curve(rows, random_grid) for rows in random_seeds])
    random_mean = seed_curves.mean(axis=0)
    random_std = seed_curves.std(axis=0, ddof=1)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 12,
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
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.fill_between(
        random_grid,
        np.clip(random_mean - random_std, 0, 100),
        np.clip(random_mean + random_std, 0, 100),
        color=RANDOM_COLOR,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(random_grid, random_mean, color=RANDOM_COLOR, linewidth=2.3,
            label="Random (3 seeds, mean ± SD; capped at 900)")
    ax.scatter([args.base_budget], [random_mean[-1]], color=RANDOM_COLOR, s=28, zorder=3)
    ax.plot(model_grid, model_curve, color=MODEL_COLOR, linewidth=2.6,
            label="Learned ranker (tail extended to queue exhaustion or 10,000)")
    ax.axvline(args.base_budget, color="#777777", linestyle=":", linewidth=1.2)
    ax.text(args.base_budget * 1.05, 3, "original cap", color="#666666", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(1, args.tail_budget)
    ax.set_ylim(0, 103)
    ticks = (1, 2, 5, 10, 30, 100, 300, 900, 3000, 10000)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(("1", "2", "5", "10", "30", "100", "300",
                                                  "900", "3k", "10k")))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
    ax.set_xlabel("Simulator calls")
    ax.set_ylabel("Verified success (%)")
    ax.set_title("Hard 2push (<5% exhaustive-GT setups) · hmax=2", fontsize=15,
                 fontweight="semibold")
    ax.legend(loc="lower right")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.15)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out}.{{png,pdf}}")


if __name__ == "__main__":
    main()
