#!/usr/bin/env python3
"""Ten candidate views of the same speed-up result, rendered as contact sheets to choose from.

Every panel is drawn from the paired per-problem file (paired_keyhole_compare.py) plus, for the
overhead panel, the campaign's own t_sim/t_score split. Nothing here is a new measurement -- the
point is to see which framing carries the result best before one of them becomes a paper figure.

    python scripts/experiments/plot_speedup_options.py --data <keyhole dir> --out <dir> [--leg 2push]
"""
import argparse
import glob
import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RANKER, RANDOM, HARD = "#2a78d6", "#eb6834", "#1baf7a"
TIER_COLOR = {"easy": RANKER, "medium": RANDOM, "hard": HARD}
TIERS = ["easy", "medium", "hard"]
INK, MUTED, GRID, SUNK = "#0b0b0b", "#52514e", "#e6e6e2", "#f2f1ec"


def style(ax, title=None, sub=None):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title + ("\n" + sub if sub else ""), fontsize=10, color=INK, loc="left",
                     linespacing=1.4)


def med(v):
    return st.median(v) if v else float("nan")


def pctl(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


# ---- the ten panels ----------------------------------------------------------------------

def p_scatter(ax, R):
    """A. Paired scatter -- the per-problem proof, losses visible."""
    style(ax, "A · paired cost, one dot per problem", "below the line = ranker cheaper")
    cl = [r for r in R if r["clean"]]
    for grp, c in ((["win"], RANKER), (["lose"], RANDOM)):
        pts = [r for r in cl if (r["speedup_time"] >= 1) == (grp[0] == "win")]
        ax.scatter([r["rand_t"] for r in pts], [r["model_t"] for r in pts], s=8, c=c,
                   alpha=0.45, linewidths=0)
    lo = min(min(r["rand_t"], r["model_t"]) for r in cl) * 0.8
    hi = max(max(r["rand_t"], r["model_t"]) for r in cl) * 1.3
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1)
    ax.plot([lo, hi], [lo / 10, hi / 10], color=MUTED, lw=0.9, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("random: seconds", fontsize=8.5, color=MUTED)
    ax.set_ylabel("ranker: seconds", fontsize=8.5, color=MUTED)


def p_percentile(ax, R):
    """B. Every problem's own speed-up, sorted."""
    style(ax, "B · speed-up per problem, sorted", "shaded band = random was faster")
    for t in TIERS:
        v = sorted(r["speedup_time"] for r in R if r["tier"] == t and r["clean"])
        if not v:
            continue
        ax.plot([100 * (i + 0.5) / len(v) for i in range(len(v))], v, color=TIER_COLOR[t], lw=2)
        ax.text(101, v[-1], t, color=TIER_COLOR[t], fontsize=8.5, va="center")
    ax.axhspan(1e-3, 1, color=SUNK, zorder=0)
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.set_yscale("log"); ax.set_xlim(0, 100)
    ax.set_ylim(bottom=min(r["speedup_time"] for r in R if r["clean"]) * 0.8)
    ax.set_xlabel("problems, sorted (percentile)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("speed-up", fontsize=8.5, color=MUTED)


def p_scaling(ax, R):
    """C. Cost against how rare the answer is -- the scaling claim."""
    style(ax, "C · cost vs how rare a right push is", "random pays the full 1/p price; the ranker does not")
    rows = [r for r in R if r["clean"] and r.get("draws")]
    for key, c, lab in (("rand_t", RANDOM, "random"), ("model_t", RANKER, "ranker")):
        x = [r["draws"] for r in rows]
        y = [r[key] for r in rows]
        ax.scatter(x, y, s=5, c=c, alpha=0.13, linewidths=0)
        lx = [math.log10(v) for v in x]
        lo, hi = min(lx), max(lx)
        bx, by = [], []
        for i in range(9):
            a, b = lo + i * (hi - lo) / 9, lo + (i + 1) * (hi - lo) / 9
            g = [(xx, yy) for xx, yy, l in zip(x, y, lx) if a <= l <= b]
            if len(g) >= 8:
                bx.append(med([p[0] for p in g])); by.append(med([p[1] for p in g]))
        ax.plot(bx, by, color=c, lw=2.2, marker="o", ms=4, mec="#fcfcfb", mew=1, zorder=4)
        n = len(x)
        mx, my = sum(lx) / n, sum(math.log10(v) for v in y) / n
        slope = (sum((a - mx) * (math.log10(b) - my) for a, b in zip(lx, y))
                 / sum((a - mx) ** 2 for a in lx))
        ax.text(bx[-1] * 1.2, by[-1], f"{lab}\nslope {slope:.2f}", color=c, fontsize=8.5,
                va="center", linespacing=1.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(right=max(r["draws"] for r in rows) * 5)
    ax.set_xlabel("expected blind draws to a right push", fontsize=8.5, color=MUTED)
    ax.set_ylabel("seconds", fontsize=8.5, color=MUTED)


def p_anytime(ax, R):
    """D. Anytime curve -- what each arm delivers per second of budget."""
    style(ax, "D · solved within a time budget", "solid = ranker, dashed = random")
    ts = [10 ** (-1 + 0.05 * i) for i in range(80)]
    for t in TIERS:
        sel = [r for r in R if r["tier"] == t]
        if not sel:
            continue
        for key, ls in (("model", "-"), ("rand", "--")):
            y = [100 * sum(1 for r in sel if r[f"{key}_solved"] and r[f"{key}_t"] <= T) / len(sel)
                 for T in ts]
            ax.plot(ts, y, color=TIER_COLOR[t], lw=2 if ls == "-" else 1.4, ls=ls)
        ax.text(ts[-1] * 1.15, 100 - 6 * TIERS.index(t), t, color=TIER_COLOR[t], fontsize=8.5,
                va="center")
    ax.set_xscale("log"); ax.set_xlim(ts[0], ts[-1] * 6); ax.set_ylim(0, 103)
    ax.set_xlabel("seconds allowed per problem", fontsize=8.5, color=MUTED)
    ax.set_ylabel("% of problems solved", fontsize=8.5, color=MUTED)


def p_budget_bars(ax, R):
    """E. The same thing as three headline budgets -- the most quotable form."""
    style(ax, "E · solved inside a fixed budget", "the sentence a reviewer repeats")
    budgets = [1, 10, 60]
    w = 0.36
    for j, (key, c, lab) in enumerate((("model", RANKER, "ranker"), ("rand", RANDOM, "random"))):
        xs, ys = [], []
        for i, t in enumerate(TIERS):
            sel = [r for r in R if r["tier"] == t]
            for k, T in enumerate(budgets):
                xs.append(i * 4 + k + (j - 0.5) * w)
                ys.append(100 * sum(1 for r in sel if r[f"{key}_solved"] and r[f"{key}_t"] <= T)
                          / len(sel))
        ax.bar(xs, ys, width=w, color=c, label=lab)
    ax.set_xticks([i * 4 + k for i in range(3) for k in range(3)])
    ax.set_xticklabels([f"{T}s" for _ in TIERS for T in budgets], fontsize=8)
    for i, t in enumerate(TIERS):
        ax.text(i * 4 + 1, -14, t, ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_ylabel("% of problems solved", fontsize=8.5, color=MUTED)


def p_hist(ax, R):
    """F. Distribution of the per-problem speed-up."""
    style(ax, "F · how often, by how much", "mass left of 1× is where we lose")
    v = [r["speedup_time"] for r in R if r["clean"]]
    bins = [10 ** (-1 + 0.1 * i) for i in range(40)]
    ax.hist([x for x in v if x < 1], bins=bins, color=RANDOM, alpha=0.85)
    ax.hist([x for x in v if x >= 1], bins=bins, color=RANKER, alpha=0.85)
    ax.axvline(med(v), color=INK, lw=1.4)
    ax.text(med(v) * 1.1, ax.get_ylim()[1] * 0.86, f"median {med(v):.1f}×", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("speed-up on a problem", fontsize=8.5, color=MUTED)
    ax.set_ylabel("problems", fontsize=8.5, color=MUTED)


def p_cumulative(ax, R):
    """G. Total compute to clear the whole benchmark."""
    style(ax, "G · time to clear the whole set", "cumulative hours, cheapest problems first")
    for key, c, lab in (("model_t", RANKER, "ranker"), ("rand_t", RANDOM, "random")):
        v = sorted(r[key] for r in R if r["clean"])
        cum, tot = [], 0.0
        for x in v:
            tot += x / 3600
            cum.append(tot)
        ax.plot(range(1, len(cum) + 1), cum, color=c, lw=2.2)
        ax.text(len(cum) * 1.01, cum[-1], f"{lab}\n{cum[-1]:.1f} h", color=c, fontsize=8.5,
                va="center", linespacing=1.3)
    ax.set_xlim(0, len(R) * 1.22)
    ax.set_xlabel("problems solved (cheapest first)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("cumulative hours", fontsize=8.5, color=MUTED)


def p_dumbbell(ax, R):
    """H. Median cost per tier, both arms, as one connected pair."""
    style(ax, "H · the typical problem, tier by tier", "median cost each; the × is the per-problem median")
    for i, t in enumerate(TIERS):
        cl = [r for r in R if r["tier"] == t and r["clean"]]
        m, q = med([r["model_t"] for r in cl]), med([r["rand_t"] for r in cl])
        # The label must carry the PER-PROBLEM median speed-up. q/m is the ratio of medians, which
        # reads far higher (18.5x vs 10.2x on hard) and is the conflation the campaign card bans.
        sp = med([r["speedup_time"] for r in cl])
        ax.plot([m, q], [i, i], color=MUTED, lw=2, zorder=1)
        ax.scatter([m], [i], s=90, color=RANKER, zorder=3)
        ax.scatter([q], [i], s=90, color=RANDOM, zorder=3)
        ax.text(q * 1.25, i, f"{sp:.1f}× typical problem   ·   medians {m:.1f}s vs {q:.1f}s",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(TIERS))); ax.set_yticklabels(TIERS, fontsize=9)
    ax.set_xscale("log"); ax.set_xlim(right=max(med([r["rand_t"] for r in R if r["tier"] == t])
                                                for t in TIERS) * 9)
    ax.set_ylim(-0.6, len(TIERS) - 0.4)
    ax.set_xlabel("seconds (median problem)", fontsize=8.5, color=MUTED)


def p_margins(ax, R):
    """I. How the wins are distributed in size -- no ratio arithmetic required."""
    style(ax, "I · win margins by tier", "share of problems in each band")
    bands = [(0, 1, "random faster", RANDOM), (1, 2, "up to 2×", "#c9d3de"),
             (2, 10, "2–10×", "#7fa9dd"), (10, 1e9, "over 10×", RANKER)]
    bottoms = [0] * len(TIERS)
    for lo, hi, lab, c in bands:
        vals = []
        for i, t in enumerate(TIERS):
            cl = [r["speedup_time"] for r in R if r["tier"] == t and r["clean"]]
            vals.append(100 * sum(1 for x in cl if lo <= x < hi) / len(cl))
        ax.barh(range(len(TIERS)), vals, left=bottoms, color=c, height=0.6, label=lab)
        for i, v in enumerate(vals):
            if v > 7:
                ax.text(bottoms[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="#0b0b0b" if c != RANKER else "#ffffff")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_yticks(range(len(TIERS))); ax.set_yticklabels(TIERS, fontsize=9)
    ax.set_xlim(0, 100); ax.set_xlabel("% of problems", fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42))


def p_overhead(ax, split):
    """J. Where the ranker's own seconds go -- we charge ourselves for inference."""
    style(ax, "J · what the seconds are spent on", "median problem; scoring is our own cost")
    labels, sims, scores = [], [], []
    for t in TIERS:
        for arm in ("ranker", "random"):
            labels.append(f"{t}\n{arm}")
            sims.append(split[(t, arm)]["sim"]); scores.append(split[(t, arm)]["score"])
    x = range(len(labels))
    ax.bar(x, sims, color="#b7c8da", label="physics simulation")
    ax.bar(x, scores, bottom=sims, color=RANKER, label="model scoring")
    ax.set_xticks(list(x))
    ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=7.5, rotation=30,
                       ha="right")
    ax.set_yscale("log")
    ax.set_ylim(bottom=min(s for s in sims if s > 0) * 0.4)
    ax.set_ylabel("seconds (median problem)", fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5)


# ---- data --------------------------------------------------------------------------------

def load_pairs(d, leg):
    return [json.loads(l) for l in open(os.path.join(d, f"pairs_{leg}.jsonl"))]


def attach_draws(rows, leg):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
    from namo import eval_sets

    def suf(p, n=5):
        return "/".join(str(p).rstrip("/").split("/")[-n:])

    if leg == "1push":
        man = json.load(open(eval_sets.path("onepush_manifest")))
        dens = {(suf(x), e["object_id"]): 100 * len(e["valid"]) / max(len(e["tried"]), 1)
                for x, eps in man.items() for e in eps}
    else:
        man = json.load(open(eval_sets.path("pure2push_manifest")))
        dens = {(suf(x), e["object_id"]):
                100 * len(e["valid_first_push"]) / max(len(e["tried_1push"]), 1)
                for x, eps in man.items() for e in eps}
    for r in rows:
        d = dens.get((r["xml"], r["object_id"]))
        r["draws"] = 100.0 / d if d else None


def timing_split(scratch, leg, tiers):
    """(tier, arm) -> median physics seconds and median scoring seconds, from the campaign rows."""
    root = os.path.join(scratch, "aquaman/round0/eval_walltime4k")
    legdir = "1push_hmax2" if leg == "1push" else "2push"
    acc = {}
    for arm, dirs in (("ranker", ["HY5U_s1", "HY5U_s2", "HY5U_s3"]),
                      ("random", ["rand_s7000", "rand_s8000", "rand_s9000"])):
        for d in dirs:
            for f in glob.glob(os.path.join(root, d, legdir, "shard_*.jsonl")):
                for line in open(f):
                    r = json.loads(line)
                    key = ("/".join(r["xml"].split("/")[-5:]), r["object_id"])
                    t = tiers.get(key)
                    if not t or not r["solved"]:
                        continue
                    acc.setdefault((t, arm), []).append((r["t_sim"], r["t_score"]))
    return {k: {"sim": med([a for a, _ in v]), "score": med([b for _, b in v])}
            for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leg", default="2push", choices=["1push", "2push"])
    ap.add_argument("--scratch", default=os.environ.get("NAMO_SCRATCH"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = load_pairs(a.data, a.leg)
    attach_draws(R, a.leg)
    tiers = {(r["xml"], r["object_id"]): r["tier"] for r in R}
    split = timing_split(a.scratch, a.leg, tiers)

    sheets = [([p_scatter, p_percentile, p_scaling, p_anytime, p_budget_bars], "1"),
              ([p_hist, p_cumulative, p_dumbbell, p_margins, None], "2")]
    for panels, name in sheets:
        fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))
        flat = [ax for row in axes for ax in row]
        for ax, fn in zip(flat, panels + [None] * 6):
            if fn is None:
                ax.axis("off")
                continue
            fn(ax, split if fn is p_overhead else R)
        if name == "2":
            p_overhead(flat[4], split)
            flat[5].axis("off")
        fig.suptitle(f"Ten ways to show the speed-up — {a.leg}, sheet {name} of 2",
                     y=0.985, fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        out = os.path.join(a.out, f"options_sheet{name}_{a.leg}.png")
        fig.savefig(out, dpi=140)
        print("wrote", out)


if __name__ == "__main__":
    main()
