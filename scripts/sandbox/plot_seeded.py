#!/usr/bin/env python3
"""3-seed best-first success curves (mean line + min-max band) from the full-set campaign dirs
(full{1,2}_s{1,2,3}_b900). Each row: {model, tier, n_sim, t_wall, solved}. Per (model,tier): a success
curve per seed -> mean + band. random's 3 seeds = rng 7/8/9. SIM=full set; TIME=identical-CPU nodes."""
import json, glob, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

EV = "/scratch/dm1487/eval/timebench"
COL = {"Hz": "#1f77b4", "NoHz": "#2ca02c", "random": "#d62728"}
TIERS = ["easy", "med", "hard"]


def nice(ax, cand, lo, hi):
    t = [x for x in cand if lo <= x <= hi]
    ax.xaxis.set_major_locator(FixedLocator(t)); ax.xaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in t]))
    ax.xaxis.set_minor_locator(NullLocator())


def load_seed(d):
    bt = defaultdict(list)
    for f in glob.glob(f"{EV}/{d}/shard_*.jsonl"):
        for l in open(f):
            r = json.loads(l); bt[(r["model"], r["tier"])].append((bool(r["solved"]), r["n_sim"], r["t_wall"]))
    return bt


def succ(data, grid, idx):                          # idx 1=n_sim, 2=t_wall
    n = len(data)
    return np.array([100.0 * sum(1 for d in data if d[0] and d[idx] <= X) / n for X in grid]) if n else np.zeros(len(grid))


def band(ax, x, curves, color, label):
    if not curves: return
    C = np.vstack(curves)
    ax.plot(x, C.mean(0), color=color, lw=2, label=f"{label} ({len(curves)} seeds)")
    if len(curves) > 1:
        ax.fill_between(x, C.min(0), C.max(0), color=color, alpha=0.18)


def make(title, seed_dirs, out):
    seeds = [load_seed(d) for d in seed_dirs]
    nseed = sum(1 for s in seeds if s)
    simgrid = np.unique(np.round(np.logspace(0, np.log10(900), 120)).astype(int))
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.8))
    for ci, T in enumerate(TIERS):
        st = sorted(tw for s in seeds for m in COL for (sv, ns, tw) in s.get((m, T), []) if sv)
        ttmin = max((st[0] * 0.9) if st else 0.1, 0.05); ttmax = (st[-1] * 1.1) if st else 1.0
        tg = np.logspace(np.log10(ttmin), np.log10(ttmax), 160)
        for m in COL:
            band(ax[0, ci], simgrid, [succ(s.get((m, T), []), simgrid, 1) for s in seeds if s.get((m, T))], COL[m], m)
            band(ax[1, ci], tg, [succ(s.get((m, T), []), tg, 2) for s in seeds if s.get((m, T))], COL[m], m)
        ax[0, ci].set_title(f"{T} — vs SIM budget"); ax[0, ci].set_xscale("log"); ax[0, ci].set_xlim(1, 900); ax[0, ci].set_xlabel("sim budget"); nice(ax[0, ci], [1, 2, 5, 10, 20, 50, 100, 200, 500, 900], 1, 900)
        ax[1, ci].set_title(f"{T} — vs WALL-TIME"); ax[1, ci].set_xscale("log"); ax[1, ci].set_xlim(ttmin, ttmax); ax[1, ci].set_xlabel("wall-clock budget (s)"); nice(ax[1, ci], [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], ttmin, ttmax)
    for a in ax.flat:
        a.set_ylim(0, 100); a.grid(alpha=0.3); a.legend(fontsize=8, loc="lower right"); a.set_ylabel("% solved")
    fig.suptitle(f"{title} — best-first, {nseed} seeds (mean line, min–max band). SIM=full set, TIME=identical-CPU exclusive nodes", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); print(f"  saved {out}")


if __name__ == "__main__":
    make("1-push (n=1323, hmax=1)", ["full1_s1_b900", "full1_s2_b900", "full1_s3_b900"], f"{EV}/curves_1push_3seed.png")
    make("2-push (34/tier sample ~102/seed, hmax=2)", ["full2_s1_b900", "full2_s2_b900", "full2_s3_b900"], f"{EV}/curves_2push_3seed.png")
