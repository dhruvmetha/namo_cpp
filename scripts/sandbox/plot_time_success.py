#!/usr/bin/env python3
"""Time-vs-success (anytime) curves from a SAME-NODE best-first timing jsonl.
success(T) = fraction of episodes solved within wall-clock T (time-to-solve = t_wall, since the search
breaks the instant the goal opens). All models interleaved on one exclusive node => same-machine."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

fn = sys.argv[1]; out = sys.argv[2]; node = sys.argv[3] if len(sys.argv) > 3 else "one exclusive node"
rows = [json.loads(l) for l in open(fn)]
by = defaultdict(lambda: defaultdict(list))   # tier -> model -> [(t_wall, solved)]
for r in rows:
    by[r["tier"]][r["model"]].append((r["t_wall"], bool(r["solved"])))

tiers = ["easy", "med", "hard"]
colors = {"Hz": "#1f77b4", "NoHz": "#2ca02c", "random": "#d62728"}
Tmax = max(t for r in rows for t in [r["t_wall"]])
grid = np.linspace(0, Tmax, 300)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, tier in zip(axes, tiers):
    for m in ["Hz", "NoHz", "random"]:
        eps = by[tier].get(m, [])
        if not eps:
            continue
        n = len(eps)
        succ = [100.0 * sum(1 for (tw, sv) in eps if sv and tw <= T) / n for T in grid]
        ax.plot(grid, succ, label=f"{m} (n={n})", color=colors[m], lw=2)
    ax.set_title(f"{tier}"); ax.set_xlabel("wall-clock budget T (s)"); ax.grid(alpha=0.3)
    ax.set_xlim(0, Tmax)
axes[0].set_ylabel("% solved within T"); axes[0].set_ylim(0, 100); axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle(f"Time-vs-success (best-first, hmax=2) — SAME node ({node}), warm, OMP=1, all models interleaved", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"  saved {out}  (Tmax={Tmax:.1f}s)")
