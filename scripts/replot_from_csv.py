#!/usr/bin/env python3
"""Replot Success-vs-Time and Success-vs-Forward-Sims from the tidy CSV.

Reads the CSV produced by `export_eval_csv.py`, builds:
  - Difficulty buckets from the oracle (ref::) median pushes (tertiles).
  - Success @ time-cutoff and Success @ push-cutoff curves per model, per bucket.

Usage:
    python scripts/replot_from_csv.py --csv 1_push_eval/raw.csv --out 1_push_eval/replot --bench 1push
    python scripts/replot_from_csv.py --csv 2_push_eval/raw.csv --out 2_push_eval/replot --bench 2push
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BENCH_CFG = {
    "1push": dict(
        time_max_s=6.0, time_step_s=0.05,
        push_max=120, push_step=1,
        time_xlabel="Time cutoff (s)",
        push_xlabel="Forward simulations (verified push evaluations)",
    ),
    "2push": dict(
        time_max_s=100.0, time_step_s=0.5,
        push_max=2000, push_step=10,
        time_xlabel="Time cutoff (s)",
        push_xlabel="Forward simulations (verified push evaluations)",
    ),
}


def build_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'difficulty' column to non-ref rows via oracle-median tertiles."""
    ref = df[df["model"].str.startswith("ref::") & df["success"]]
    oracle = (
        ref.groupby(["env", "region", "object"])["pushes"]
        .median()
        .rename("oracle_pushes")
        .reset_index()
    )
    p33, p66 = oracle["oracle_pushes"].quantile([1 / 3, 2 / 3]).tolist()

    def bucket(x):
        if x <= p33:
            return "easy"
        if x <= p66:
            return "medium"
        return "hard"

    oracle["difficulty"] = oracle["oracle_pushes"].map(bucket)
    print(f"oracle push tertiles: p33={p33:.1f}  p66={p66:.1f}")
    print(oracle["difficulty"].value_counts().to_string())

    out = df.merge(oracle, on=["env", "region", "object"], how="left")
    # Only keep triplets where oracle solved (so all models compared on same set).
    out = out[out["oracle_pushes"].notna()].copy()
    return out


def success_at_cutoff(values: np.ndarray, success: np.ndarray, cutoffs: np.ndarray):
    """Fraction of N triplets with success=True AND value <= cutoff, vs cutoffs."""
    n = len(values)
    if n == 0:
        return np.zeros_like(cutoffs, dtype=float)
    # Only successful runs contribute to cumulative success.
    v = values[success]
    rates = np.array([(v <= c).sum() / n for c in cutoffs])
    return rates


def plot_curves(
    df: pd.DataFrame, x_col: str, cutoffs: np.ndarray,
    xlabel: str, title: str, out_path: Path,
    x_to_plot=None,
):
    """One figure, 3 panels (easy/medium/hard), one line per model."""
    models = [m for m in df["model"].unique() if not m.startswith("ref::")]
    categories = ["easy", "medium", "hard"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(models))))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    n_by_cat = {}
    for ax, cat in zip(axes, categories):
        sub_cat = df[df["difficulty"] == cat]
        # n triplets = unique (env, region, object) in this bucket
        n_triplets = sub_cat.drop_duplicates(["env", "region", "object"]).shape[0] // max(1, len(models))
        n_by_cat[cat] = n_triplets
        for i, m in enumerate(models):
            m_df = sub_cat[sub_cat["model"] == m]
            vals = m_df[x_col].to_numpy(dtype=float)
            succ = m_df["success"].to_numpy(dtype=bool)
            rates = success_at_cutoff(vals, succ, cutoffs)
            xs = x_to_plot if x_to_plot is not None else cutoffs
            ax.plot(xs, rates, label=m, color=colors[i], lw=2)
        ax.set_title(f"{cat.capitalize()}  (N={n_by_cat[cat]})", fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
        ax.grid(True, ls="--", alpha=0.5)
    axes[0].set_ylabel("Success rate")
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"saved {out_path}.{{pdf,png}}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--bench", choices=list(BENCH_CFG), required=True)
    args = ap.parse_args()

    cfg = BENCH_CFG[args.bench]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows, models: {df['model'].unique().tolist()}")
    df = build_difficulty(df)

    # Drop oracle rows for plotting — only compare planners.
    planners = df[~df["model"].str.startswith("ref::")].copy()

    # Time vs success — cutoffs in ms (data is ms in CSV), x-axis in seconds.
    t_cuts_s = np.arange(0, cfg["time_max_s"] + cfg["time_step_s"], cfg["time_step_s"])
    t_cuts_ms = t_cuts_s * 1000.0
    plot_curves(
        planners, "time_ms", t_cuts_ms,
        cfg["time_xlabel"], f"Success vs Time — {args.bench}",
        out_dir / "success_vs_time", x_to_plot=t_cuts_s,
    )

    # Pushes vs success — pushes are integers ("verified push evaluations").
    p_cuts = np.arange(0, cfg["push_max"] + cfg["push_step"], cfg["push_step"], dtype=float)
    plot_curves(
        planners, "pushes", p_cuts,
        cfg["push_xlabel"], f"Success vs Forward Sims — {args.bench}",
        out_dir / "success_vs_forward_sims",
    )


if __name__ == "__main__":
    main()
