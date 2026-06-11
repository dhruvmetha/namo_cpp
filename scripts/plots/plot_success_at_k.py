#!/usr/bin/env python
"""success@k curves: random (reachability-aware, no-replacement floor) vs
diffusion (informative ep500) vs sharp (champion 1-push scorer).

Numbers are the measured eval points (k = 1,5,10,20) on the SAME clean
held-out test set, per-episode matched, re-binned by true solve_rate
(n = 413 hard / 491 med / 752 easy). Sources:
  sharp  : /scratch/dm1487/eval/robust/sharp_s1__epoch017-val_loss0.2713.json
  floor  : same json (floor block = without-replacement / hypergeometric)
  diff   : docs/experiments/informative_1push_results.md (informative ep500)

Run: /scratch/dm1487/envs/namo/bin/python scripts/plots/plot_success_at_k.py
Out : docs/experiments/figures/success_at_k_{hard,med,easy,combined}.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = [1, 5, 10, 20]

# success@k (%) — measured
DATA = {
    "hard": {
        "sharp":     [32.9, 62.5, 75.8, 88.1],
        "diffusion": [ 5.9, 22.2, 33.7, 55.2],
        "random":    [ 2.7, 13.3, 25.6, 47.4],   # no-replacement (hypergeometric) floor
    },
    "med": {
        "sharp":     [81.3, 94.5, 96.7, 98.6],
        "diffusion": [28.9, 68.6, 81.9, 94.5],
        "random":    [16.8, 58.7, 81.3, 95.1],
    },
    "easy": {
        "sharp":     [99.6, 99.9, 100.0, 100.0],
        "diffusion": [64.6, 95.1, 98.4, 99.9],
        "random":    [65.4, 97.8, 99.9, 100.0],
    },
}
N = {"hard": 413, "med": 491, "easy": 752}

STYLE = {
    "sharp":     dict(color="#1b6cff", marker="o", lw=2.6, ms=8, label="1-push scorer (ours)", zorder=3),
    "diffusion": dict(color="#ff7a18", marker="s", lw=2.2, ms=7, label="diffusion (informative)", zorder=2),
    "random":    dict(color="#888888", marker="^", lw=1.8, ms=7, ls="--",
                      label="random floor (reachability-aware)", zorder=1),
}

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "experiments", "figures")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)


def _draw(ax, bin_name, title=True):
    d = DATA[bin_name]
    for key in ("random", "diffusion", "sharp"):  # draw floor first, sharp on top
        ax.plot(K, d[key], **STYLE[key])
    ax.set_xticks(K)
    ax.set_xlim(0.3, 20.7)
    ax.set_ylim(0, 103)
    ax.set_xlabel("k (top-k pushes tried)")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(f"{bin_name}  (n={N[bin_name]})", fontsize=12, fontweight="bold")


# --- per-bin figures ---
for b in ("hard", "med", "easy"):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    _draw(ax, b)
    ax.set_ylabel("success@k  (%)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(OUT, f"success_at_k_{b}.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("wrote", p)

# --- combined 1x3 ---
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True)
for ax, b in zip(axes, ("hard", "med", "easy")):
    _draw(ax, b)
axes[0].set_ylabel("success@k  (%)")
axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)
fig.suptitle("1-push success@k: random vs diffusion vs scorer", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
p = os.path.join(OUT, "success_at_k_combined.png")
fig.savefig(p, dpi=160)
plt.close(fig)
print("wrote", p)
