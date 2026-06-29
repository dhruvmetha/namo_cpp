#!/usr/bin/env python3
"""Best-first success curves SPLIT BY DIFFICULTY (easy/med/hard). Two figures (1-push, 2-push), each 2x3:
rows = {vs SIM budget (full set), vs WALL-TIME (same-node sample)}, cols = {easy, med, hard}.
Tier per episode from the key's solve-rate (hard<0.05, med<0.30, else easy)."""
import json, glob, os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


def nice_logticks(ax, cand, lo, hi):
    """Plain-number major ticks at meaningful values on a log axis (no 10^x, no minor clutter)."""
    ticks = [t for t in cand if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())


EV = "/scratch/dm1487/eval"
COL = {"Hz": "#1f77b4", "NoHz": "#2ca02c", "random": "#d62728"}
TIERS = ["easy", "med", "hard"]
KEY1 = "/scratch/dm1487/datasets/namo_testset_v1/labels/onepush_episodes.json"
KEY2 = "/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push.json"


def tier(sr): return "hard" if sr < 0.05 else ("med" if sr < 0.30 else "easy")


def tiermap(key):
    k = json.load(open(key)); m = {}
    for xml, recs in k.items():
        for r in recs:
            sr = r.get("solve_rate_first_push", r.get("solve_rate", 0.0))
            m[(xml, r["object_id"], r.get("region"))] = tier(sr)
    return m


def leaves_bytier(dirs, tm):                       # full-set sim: {tier: [(solved, sims)]}
    eps = {}
    for d in dirs:
        for f in glob.glob(f"{EV}/{d}/shard_*.jsonl"):
            for l in open(f):
                r = json.loads(l); eps[(r["xml"], r["object_id"], r.get("region"))] = (bool(r["solved"]), r["sims"])
    bt = defaultdict(list); miss = 0
    for k, v in eps.items():
        t = tm.get(k)
        if t: bt[t].append(v)
        else: miss += 1
    if miss: print(f"    (warn: {miss}/{len(eps)} leaf eps unmatched to tier)")
    return bt


def time_bytier(path, model, idx):                 # path = dir of pooled shards OR single file. idx 1=t_wall, 2=n_sim
    files = glob.glob(f"{path}/shard_*.jsonl") if os.path.isdir(path) else ([path] if os.path.exists(path) else [])
    bt = defaultdict(list)
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if r["model"] == model:
                bt[r["tier"]].append((bool(r["solved"]), r["t_wall"] if idx == 1 else r["n_sim"]))
    return bt


def has_time(path):
    return bool(glob.glob(f"{path}/shard_*.jsonl")) if os.path.isdir(path) else os.path.isfile(path)


def succ(data, grid):
    n = len(data)
    return [100.0 * sum(1 for (s, x) in data if s and x <= X) / n for X in grid] if n else [0] * len(grid)


def make_fig(title, simdirs, timefn, key, out):
    tm = tiermap(key)
    simbt = {m: leaves_bytier(simdirs[m], tm) for m in simdirs}
    simgrid = np.unique(np.round(np.logspace(0, np.log10(900), 120)).astype(int))
    have_time = has_time(timefn)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for ci, T in enumerate(TIERS):
        # row 0: success vs sim budget (full set; for models w/o full data, fall back to time-run sample n_sim, dashed)
        for m in COL:
            if m in simdirs:
                d = simbt[m].get(T, []); ax[0, ci].plot(simgrid, succ(d, simgrid), color=COL[m], lw=2, label=f"{m} (n={len(d)})")
            elif have_time:
                d = time_bytier(timefn, m, 2).get(T, [])
                if d: ax[0, ci].plot(simgrid, succ(d, simgrid), color=COL[m], lw=2, ls="--", label=f"{m} (sample n={len(d)})")
        # row 1: success vs wall-time — PER-TIER LOG x over the FULL solve range so curves reach their plateau AND easy/med stay readable
        tt = {m: (time_bytier(timefn, m, 1).get(T, []) if have_time else []) for m in COL}
        solved_t = sorted(v for m in COL for (s, v) in tt[m] if s)
        ttmin = max((solved_t[0] * 0.9) if solved_t else 0.1, 0.05)
        ttmax = (solved_t[-1] * 1.1) if solved_t else 1.0          # full range -> the curve reaches its true plateau
        ttg = np.logspace(np.log10(ttmin), np.log10(ttmax), 200)
        for m in COL:
            if tt[m]: ax[1, ci].plot(ttg, succ(tt[m], ttg), color=COL[m], lw=2, label=f"{m} (n={len(tt[m])})")
        ax[0, ci].set_title(f"{T} — vs SIM budget"); ax[0, ci].set_xscale("log"); ax[0, ci].set_xlim(1, 900); ax[0, ci].set_xlabel("sim budget")
        nice_logticks(ax[0, ci], [1, 2, 5, 10, 20, 50, 100, 200, 500, 900], 1, 900)
        ax[1, ci].set_title(f"{T} — vs WALL-TIME"); ax[1, ci].set_xscale("log"); ax[1, ci].set_xlim(ttmin, ttmax); ax[1, ci].set_xlabel("wall-clock budget (s)")
        nice_logticks(ax[1, ci], [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], ttmin, ttmax)
    for a in ax.flat:
        a.set_ylim(0, 100); a.grid(alpha=0.3); a.legend(fontsize=7, loc="lower right"); a.set_ylabel("% solved")
    fig.suptitle(f"{title} — best-first by difficulty. SIM=full set, TIME=same exclusive node (warm, interleaved)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight"); print(f"  saved {out}")


# TIME source: 100-env @900 sharded dirs (pooled, same CPU type) when present; else the budget-100 sample fallback.
T1 = f"{EV}/timebench/s100_1_b900" if has_time(f"{EV}/timebench/s100_1_b900") else f"{EV}/timebench/bf1_time_b100.jsonl"
T2 = f"{EV}/timebench/s100_2_b900" if has_time(f"{EV}/timebench/s100_2_b900") else f"{EV}/timebench/bf2_time_b100.jsonl"
make_fig("1-push (n=1323, hmax=1)", {"Hz": ["bf1_1push_hz"], "NoHz": ["bf1_1push_nohz"], "random": ["bf1_1push_rand"]},
         T1, KEY1, f"{EV}/timebench/curves_1push_strat.png")
make_fig("2-push (n=1018, hmax=2)", {"Hz": ["bfq_hz_v3_s1"], "NoHz": ["bfq_nohz_v3_s1"]},
         T2, KEY2, f"{EV}/timebench/curves_2push_strat.png")
