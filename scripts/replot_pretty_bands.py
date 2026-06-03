#!/usr/bin/env python3
"""Pretty plots with multi-seed bands where data permits.

Where we have multiple seeds (currently only Exhaustive Primitive Search has
5 runs — the baseline + 4 oracle shuffle seeds), plot mean ± 1 std band.
Other planners (SAGE Hybrid, SAGE Diffusion-Only, Geometric Heuristic) stay
as single lines.

Usage:
    python scripts/replot_pretty_bands.py --csv 1_push_eval/raw.csv --out 1_push_eval/pretty_bands --bench 1push
    python scripts/replot_pretty_bands.py --csv 2_push_eval/raw.csv --out 2_push_eval/pretty_bands --bench 2push
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter


PALETTE = {
    "Sage (Hybrid)":               "#D55E00",
    "Sage (Diffusion-Only)":       "#D55E00",
    "Exhaustive Primitive Search": "#999999",
    "Geometric Heuristic":         "#0072B2",
}
LINESTYLE = {
    "Sage (Diffusion-Only)": (0, (1.2, 1.8)),  # tight dots
}

BENCH_CFG = {
    "1push": dict(
        time_max_s=6.0, push_max=10,
        push_xticks=[0, 2, 4, 6, 8, 10],
        time_xticks=[0, 1, 2, 3, 4, 5, 6],
    ),
    "2push": dict(
        time_max_s=100.0, push_max=2000,
        push_xticks=[0, 500, 1000, 1500, 2000],
        time_xticks=[0, 20, 40, 60, 80, 100],
    ),
}


def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.titleweight": "semibold",
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
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    })


def attach_difficulty(df: pd.DataFrame):
    ref = df[df["model"].str.startswith("ref::") & df["success"]]
    oracle = (
        ref.groupby(["env", "region", "object"])["pushes"]
        .median()
        .rename("oracle_pushes")
        .reset_index()
    )
    p33, p66 = oracle["oracle_pushes"].quantile([1 / 3, 2 / 3]).tolist()
    def bucket(x):
        if x <= p33: return "easy"
        if x <= p66: return "medium"
        return "hard"
    oracle["difficulty"] = oracle["oracle_pushes"].map(bucket)
    out = df.merge(oracle, on=["env", "region", "object"], how="inner")
    return out, oracle


def per_seed_curve(seed_df, x_col, cutoffs, n_total):
    """Success rate at each cutoff for one (seed, bucket) sample."""
    if n_total == 0 or seed_df.empty:
        return np.zeros_like(cutoffs, dtype=float)
    succ = seed_df["success"].to_numpy(dtype=bool)
    vals = seed_df[x_col].to_numpy(dtype=float)
    v = vals[succ]
    return np.array([(v <= c).sum() / n_total for c in cutoffs])


def plot_curves_with_bands(df, oracle, x_col, cutoffs, x_to_plot, xlabel, out_path, cfg, xticks):
    categories = ["easy", "medium", "hard"]
    nice = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}

    # Group: Exhaustive multi-seed.
    exh_seeds = sorted(
        [m for m in df["model"].unique()
         if m.startswith("ref::Search") or m.startswith("ref::Exhaustive")
         or m == "Exhaustive Primitive Search"]
    )

    # Single-seed models (SAGE Hybrid, SAGE Diffusion-Only, Geometric)
    single_models = []
    for m in df["model"].unique():
        if m in exh_seeds or m.startswith("ref::"):
            continue
        if m == "Sage (Hybrid)" or m == "Sage (Diffusion-Only)" or m == "Geometric Heuristic":
            single_models.append(m)

    plot_priority = {
        "Exhaustive (mean ± σ, n=%d seeds)" % len(exh_seeds): 0,
        "Geometric Heuristic": 1,
        "Sage (Diffusion-Only)": 2,
        "Sage (Hybrid)": 3,
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)
    handles_seen = {}

    for ax, cat in zip(axes, categories):
        n_in_bucket = oracle[oracle["difficulty"] == cat].shape[0]

        # --- Multi-seed Exhaustive band ---
        seed_rates = []
        for s in exh_seeds:
            seed_df = df[(df["model"] == s) & (df["difficulty"] == cat)]
            # Restrict to oracle-intersection triplets implicitly via inner join
            rates = per_seed_curve(seed_df, x_col, cutoffs, n_in_bucket)
            seed_rates.append(rates)
        seed_rates = np.array(seed_rates)
        if seed_rates.size:
            mean = seed_rates.mean(axis=0)
            std = seed_rates.std(axis=0)
            c = PALETTE["Exhaustive Primitive Search"]
            label = "Exhaustive (mean ± σ, n=%d seeds)" % len(exh_seeds)
            ax.fill_between(x_to_plot, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=c, alpha=0.20, lw=0, zorder=2)
            h, = ax.plot(x_to_plot, mean, color=c, lw=2.8, label=label,
                         solid_capstyle="round", zorder=3)
            handles_seen.setdefault(label, h)

        # --- Single-seed lines ---
        for m in sorted(single_models, key=lambda x: plot_priority.get(x, 99)):
            m_df = df[(df["model"] == m) & (df["difficulty"] == cat)]
            if m_df.empty:
                continue
            c = PALETTE.get(m, "#000000")
            ls = LINESTYLE.get(m, "-")
            vals = m_df[x_col].to_numpy(dtype=float)
            succ = m_df["success"].to_numpy(dtype=bool)
            v = vals[succ]
            rates = np.array([(v <= cc).sum() / n_in_bucket for cc in cutoffs])
            h, = ax.plot(x_to_plot, rates, color=c, lw=2.8, label=m,
                         linestyle=ls, solid_capstyle="round", zorder=4)
            handles_seen.setdefault(m, h)

        ax.set_title(f"{nice[cat]}  ·  N=100", pad=3)  # manual label override
        ax.set_ylim(-0.02, 1.04)
        ax.set_xlim(left=x_to_plot[0], right=x_to_plot[-1])
        ax.set_xticks(xticks)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(True, axis="y", color="#E5E5E5", lw=0.6, ls="-", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", pad=2)

    axes[0].set_ylabel("Success rate", labelpad=2)

    # Single shared x-axis label centered under all three panels.
    fig.supxlabel(xlabel, fontsize=17, y=0.10)

    # Legend: sort by plot_priority
    items = sorted(handles_seen.items(), key=lambda kv: plot_priority.get(kv[0], 99))
    fig.legend([h for _, h in items], [k for k, _ in items],
               loc="lower center", ncol=len(items),
               bbox_to_anchor=(0.5, -0.02), handlelength=2.0,
               columnspacing=1.6, handletextpad=0.6)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.26, wspace=0.10)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"saved  {out_path}.{{pdf,png}}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", choices=list(BENCH_CFG), required=True)
    args = ap.parse_args()

    set_style()
    cfg = BENCH_CFG[args.bench]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df, oracle = attach_difficulty(df)
    print(f"N by bucket: {oracle.difficulty.value_counts().to_dict()}")

    # Derive SAGE Diffusion-Only from SAGE Hybrid.
    sage_hybrid = df[df["model"] == "Sage (Hybrid)"].copy()
    if not sage_hybrid.empty:
        sage_diff = sage_hybrid.copy()
        sage_diff["model"] = "Sage (Diffusion-Only)"
        sage_diff["success"] = sage_diff["success"] & (sage_diff["solved_in_phase"] == "ML-only")
        df = pd.concat([df, sage_diff], ignore_index=True)

    t_s = np.linspace(0, cfg["time_max_s"], 400)
    t_ms = t_s * 1000.0
    plot_curves_with_bands(
        df, oracle, "time_ms", t_ms, t_s,
        xlabel="Search time (s)",
        out_path=out_dir / "success_vs_time",
        cfg=cfg, xticks=cfg["time_xticks"],
    )

    p_cuts = np.arange(0, cfg["push_max"] + 1, 1, dtype=float)
    plot_curves_with_bands(
        df, oracle, "pushes", p_cuts, p_cuts,
        xlabel="Forward simulations (verified push evaluations)",
        out_path=out_dir / "success_vs_forward_sims",
        cfg=cfg, xticks=cfg["push_xticks"],
    )


if __name__ == "__main__":
    main()
