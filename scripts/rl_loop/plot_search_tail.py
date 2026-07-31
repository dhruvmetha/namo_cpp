#!/usr/bin/env python3
"""Plot the registered hard-2push natural-exhaustion tail from saved leaf rows."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np

from namo import eval_sets
from namo.paths import resolve


COLORS = {"model": "#0072B2", "random": "#D55E00"}


def _canonical_xml(path):
    return str(resolve(path))


def _hard_keys(path):
    raw = json.load(open(path))
    return {
        (_canonical_xml(xml), row["object_id"], row.get("region"))
        for xml, records in raw.items()
        for row in records
        if row["division"] == "hard"
    }


def _load(path, hard_keys, expected):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [
        row for row in rows
        if (_canonical_xml(row["xml"]), row["object_id"], row.get("region")) in hard_keys
    ]
    keys = [(_canonical_xml(row["xml"]), row["object_id"], row.get("region")) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate episode rows in {path}")
    if len(rows) != expected:
        raise RuntimeError(f"registered hard rows in {path}: {len(rows)} != {expected}")
    return rows


def _curve(rows, grid):
    solved = np.sort([int(row["sims"]) for row in rows if row["solved"]])
    return 100.0 * np.searchsorted(solved, grid, side="right") / len(rows)


def _first_at(curve, grid, target):
    hits = np.flatnonzero(curve >= target)
    return int(grid[hits[0]]) if hits.size else None


def _save(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--random", required=True, nargs=3)
    parser.add_argument("--divisions", default=str(eval_sets.DIVISIONS))
    parser.add_argument("--expect-hard", type=int, default=eval_sets.EXPECTED["divisions"]["hard"])
    parser.add_argument("--max-calls", type=int, default=10000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    hard_keys = _hard_keys(args.divisions)
    if len(hard_keys) != args.expect_hard:
        raise RuntimeError(f"registered hard keys: {len(hard_keys)} != {args.expect_hard}")
    model = _load(args.model, hard_keys, args.expect_hard)
    random_rows = [_load(path, hard_keys, args.expect_hard) for path in args.random]
    grid = np.arange(1, args.max_calls + 1)
    model_curve = _curve(model, grid)
    random_curves = np.vstack([_curve(rows, grid) for rows in random_rows])
    random_mean = random_curves.mean(axis=0)
    random_sd = random_curves.std(axis=0, ddof=1)

    thresholds = (50, 75, 90, 95, 99, 100)
    summary = {
        "n": args.expect_hard,
        "model": {
            "final_success": round(float(model_curve[-1]), 3),
            "solved": int(sum(row["solved"] for row in model)),
            "avg_calls_to_solve": round(float(np.mean([row["sims"] for row in model if row["solved"]])), 3),
            "calls_to_success": {str(target): _first_at(model_curve, grid, target) for target in thresholds},
            "last_solve_call": max(int(row["sims"]) for row in model if row["solved"]),
        },
        "random": {
            "final_success_mean": round(float(random_mean[-1]), 3),
            "final_success_sample_sd": round(float(random_sd[-1]), 3),
            "solved_per_seed": [int(sum(row["solved"] for row in rows)) for rows in random_rows],
            "calls_to_mean_success": {str(target): _first_at(random_mean, grid, target) for target in thresholds},
            "last_solve_call_per_seed": [max(int(row["sims"]) for row in rows if row["solved"]) for rows in random_rows],
        },
    }
    random_avg_calls = [float(np.mean([row["sims"] for row in rows if row["solved"]])) for rows in random_rows]
    summary["random"]["avg_calls_to_solve_mean"] = round(float(np.mean(random_avg_calls)), 3)
    summary["random"]["avg_calls_to_solve_sample_sd"] = round(float(np.std(random_avg_calls, ddof=1)), 3)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as stream:
        json.dump(summary, stream, indent=2)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.fill_between(grid, np.clip(random_mean - random_sd, 0, 100),
                    np.clip(random_mean + random_sd, 0, 100),
                    color=COLORS["random"], alpha=0.18, linewidth=0)
    ax.plot(grid, random_mean, color=COLORS["random"], linewidth=2.2,
            label="Random (3 seeds, mean ± SD)")
    ax.plot(grid, model_curve, color=COLORS["model"], linewidth=2.4, label="Learned ranker")
    ax.axvline(900, color="#777777", linestyle="--", linewidth=1.2, label="Original 900-call budget")
    ax.set_xscale("log")
    ax.set_xlim(1, args.max_calls)
    ax.set_ylim(0, 103)
    ticks = (1, 2, 5, 10, 30, 100, 300, 900, 3000, 10000)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(value) for value in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
    ax.set_xlabel("Simulator calls")
    ax.set_ylabel("Verified success (%)")
    ax.set_title(f"Hard 2push through natural queue exhaustion  ·  n={args.expect_hard}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
