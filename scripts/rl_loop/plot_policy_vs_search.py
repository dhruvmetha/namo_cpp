#!/usr/bin/env python3
"""Success vs simulator calls: greedy policy against best-first search, at matched depth and budget.

One panel per (horizon, difficulty). x is simulator calls, which both methods spend one of per push,
so the two curves sit on one budget axis. Bands are +/- sample SD across seeds.

COMMON EPISODE SET, for the same reason aquaman_agg_common.py exists: policy mode skips episodes the
search harness keeps (1310 vs 1328 one-push, 973 vs 992 two-push), so scoring each method on its own
shards compares rates over different populations. Every series here is scored on the intersection of
all of them, and the panel n is that one number.

Encoding, so the four series survive greyscale and colour-blind viewing: hue is the RANKER (blue
HY5U, orange uniform random) and dash is the METHOD (solid policy, dashed search). Both hues pass
the categorical validator on the light surface (worst adjacent pair dE 24.7 protan, 33.6 normal).

    python scripts/rl_loop/plot_policy_vs_search.py \
        --policy-root $NAMO_SCRATCH/eval/policy_v3_jamguard_20260822 \
        --search-root $NAMO_SCRATCH/eval/search_h10_b10_20260822 --out <dir>
"""
import argparse
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/rl_loop"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo import eval_sets  # noqa: E402
from agg_testset_reactive import load_divisions  # noqa: E402

KS = list(range(1, 11))
TIERS = ("easy", "medium", "hard", "all")
HORIZONS = ("1push", "2push")
LEG = {"1push": {"policy": "1push_policy", "search": "1push_hmax2", "tiers": lambda: eval_sets.ONEPUSH},
       "2push": {"policy": "2push_policy", "search": "2push", "tiers": lambda: eval_sets.DIVISIONS}}

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SERIES = [                       # (label, root-key, arm-prefix, colour, dash)
    ("HY5U policy", "policy", "HY5U", BLUE, "-"),
    ("HY5U search", "search", "HY5U", BLUE, "--"),
    ("random policy", "policy", "rand", ORANGE, "-"),
    ("random search", "search", "rand", ORANGE, "--"),
]


def read_leaves(leaf_dir, div, kind):
    """{(xml, obj, region): (tier, cost)} for one arm+leg. cost = simulator calls to success, 0 = never."""
    out = {}
    nomatch = 0
    for path in sorted(glob.glob(os.path.join(leaf_dir, "shard_*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["xml"], row["object_id"], row.get("region"))
                tier = div.get(key)
                if tier is None:
                    nomatch += 1
                    continue
                cost = row["opened_at"] if kind == "policy" else (row["sims"] if row["solved"] else 0)
                out[key] = (tier, cost)
    if nomatch:
        raise RuntimeError(f"{leaf_dir}: {nomatch} rows matched no tier record")
    return out


def curve(leaves, common):
    """{tier: [success% at k=1..10]} over the COMMON episode set only."""
    hit = {t: np.zeros(len(KS)) for t in TIERS}
    n = {t: 0 for t in TIERS}
    for key in common:
        tier, cost = leaves[key]
        for t in (tier, "all"):
            n[t] += 1
            if cost:
                hit[t] += np.array([1.0 if cost <= k else 0.0 for k in KS])
    return {t: 100.0 * hit[t] / max(1, n[t]) for t in TIERS}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-root", required=True)
    ap.add_argument("--search-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-seeds", default="s1,s2,s3")
    ap.add_argument("--random-seeds", default="s7000,s8000,s9000")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    roots = {"policy": a.policy_root, "search": a.search_root}

    data, counts = {}, {}
    for hz in HORIZONS:
        div = load_divisions(str(LEG[hz]["tiers"]()))
        # pass 1: read every arm, then intersect -- no series is scored on episodes another one lacks
        leaves = {}
        for label, kind, prefix, _c, _d in SERIES:
            seeds = a.model_seeds if prefix == "HY5U" else a.random_seeds
            for sd in seeds.split(","):
                leaves[(label, sd)] = read_leaves(
                    os.path.join(roots[kind], f"{prefix}_{sd}", LEG[hz][kind]), div, kind)
        common = set.intersection(*(set(v) for v in leaves.values()))
        print(f"{hz}: common episode set n={len(common)} "
              f"(per-arm {min(len(v) for v in leaves.values())}-{max(len(v) for v in leaves.values())})")
        # pass 2: score every series on that one set
        for label, kind, prefix, _c, _d in SERIES:
            seeds = a.model_seeds if prefix == "HY5U" else a.random_seeds
            per_seed = []
            for sd in seeds.split(","):
                cur, n = curve(leaves[(label, sd)], common)
                per_seed.append(cur)
                counts[hz] = n
            data[(hz, label)] = {t: (np.mean([p[t] for p in per_seed], axis=0),
                                     np.std([p[t] for p in per_seed], axis=0, ddof=1))
                                 for t in TIERS}

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for r, hz in enumerate(HORIZONS):
        for c, tier in enumerate(TIERS):
            ax = axes[r][c]
            ax.set_facecolor("#fcfcfb")
            for label, _kind, _prefix, colour, dash in SERIES:
                mean, sd = data[(hz, label)][tier]
                ax.fill_between(KS, mean - sd, mean + sd, color=colour, alpha=0.13, linewidth=0)
                ax.plot(KS, mean, color=colour, linestyle=dash, linewidth=2.0,
                        marker="o", markersize=4.5, markeredgecolor="#fcfcfb",
                        markeredgewidth=0.8, label=label if (r == 0 and c == 0) else None)
            ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.set_ylim(0, 102)
            ax.set_xlim(1, 10)
            ax.set_xticks([1, 2, 3, 5, 10])
            ax.tick_params(colors=MUTED, labelsize=9, length=3)
            n = counts[hz][tier]
            ax.set_title(f"{hz}  ·  {tier}  (n={n})", color=INK, fontsize=10.5, pad=8)
            if c == 0:
                ax.set_ylabel("episodes opened (%)", color=MUTED, fontsize=10)
            if r == 1:
                ax.set_xlabel("simulator calls", color=MUTED, fontsize=10)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=10.5, labelcolor=MUTED)
    fig.suptitle("Greedy policy vs best-first search at equal simulator budget  ·  hmax=10, "
                 "fixed-physics v3, 3 seeds, band = ±1 SD",
                 color=INK, fontsize=12.5, y=1.055)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"policy_vs_search_success_vs_sims.{ext}"),
                    dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    rows = {f"{hz}|{lab}|{t}": {"mean": [round(float(v), 1) for v in data[(hz, lab)][t][0]],
                                "sd": [round(float(v), 1) for v in data[(hz, lab)][t][1]]}
            for hz in HORIZONS for (lab, _k, _p, _c, _d) in SERIES for t in TIERS}
    json.dump({"ks": KS, "series": rows}, open(os.path.join(a.out, "policy_vs_search.json"), "w"), indent=1)
    print(f"wrote {a.out}/policy_vs_search_success_vs_sims.png")
    for hz in HORIZONS:
        for lab, _k, _p, _c, _d in SERIES:
            m = data[(hz, lab)]["all"][0]
            print(f"  {hz:6s} {lab:15s} all: " + " ".join(f"@{k}={m[i]:5.1f}" for i, k in enumerate(KS) if k in (1, 2, 3, 5, 10)))


if __name__ == "__main__":
    main()
