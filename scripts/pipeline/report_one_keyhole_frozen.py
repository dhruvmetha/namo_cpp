#!/usr/bin/env python3
"""Report the 2026-09-14 HY5U ablations on Tri-An's frozen one-keyhole 600.

Every table splits by horizon (1-push, 2-push) and certificate tier (easy, medium, hard).
Scoring matches report_full_namo_rerun.py: a failed run costs infinite simulator calls.
Arms group into families by dropping the _sN seed suffix. Family numbers pool the three
seeds; seed spreads come from per-seed values. Random is Tri-An's five uniform seeds on the
same problems (references-v1, with references-identity-repair-v2 for the 11 problems he
repaired).

The report refuses to run unless every arm has all 600 problems from the expected code.
Rows can come from several --results folders (the Amarel run plus the CS run of problems
whose starting state did not match their certificate on Amarel's build); one problem
saved twice for one arm is an error.

Usage:
  python scripts/pipeline/report_one_keyhole_frozen.py --problems <input/one_keyhole> \
      --results <amarel results> --results <cs results> --arms arms.json \
      --references <Tri-An campaign root> --expect-source-sha256 <sha> --out report.json --plot-dir <dir>
"""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

GROUPS = [(1, "easy"), (1, "medium"), (1, "hard"), (2, "easy"), (2, "medium"), (2, "hard")]
WITHIN = (1, 2, 5, 10, 30, 100, 300, 1000, 3000)
RANDOM_SEEDS = (7000, 8000, 9000, 10000, 11000)
LABEL = {
    "HY5U": "HY5U (full model)",
    "HY5": "no unreachable-cell rule (HY5)",
    "HY5U_no_family": "no family data",
    "HY5U_regression": "regression only",
    "HY5U_independent": "independent contacts",
    "HY5U_global": "global readout",
    "HY5U_no_local": "no local",
    "HY5U_no_edge": "no edge identity",
    "HY5U_rank_only_nofloor": "rank-only, no floor",
    "Random": "Random ordering",
}


def cost(row):
    return row["total_calls"] if row["solved"] else math.inf


def summarize(rows):
    median = statistics.median(cost(r) for r in rows)
    return dict(runs=len(rows), solved=sum(r["solved"] for r in rows),
                success_rate=sum(r["solved"] for r in rows) / len(rows),
                median_calls_until_success=median if math.isfinite(median) else None,
                solved_within={k: sum(cost(r) <= k for r in rows) / len(rows) for k in WITHIN})


def group_key(problem):
    return (problem["horizon"], problem["difficulty"])


def table(runs_by_seed, problems, keep=None):
    """runs_by_seed: seed -> problem_id -> row. Pooled and per-seed summaries for each group."""
    keep = keep or set(problems)
    names = [f"{h}push_{d}" for h, d in GROUPS] + ["1push_all", "2push_all", "all"]
    members = {f"{h}push_{d}": [p for p in keep if group_key(problems[p]) == (h, d)] for h, d in GROUPS}
    members.update({f"{h}push_all": [p for p in keep if problems[p]["horizon"] == h] for h in (1, 2)}, all=list(keep))
    out = {}
    for name in names:
        seeds = {seed: summarize([rows[p] for p in members[name]]) for seed, rows in runs_by_seed.items()}
        pooled = summarize([rows[p] for rows in runs_by_seed.values() for p in members[name]])
        out[name] = dict(pooled, seeds=seeds)
    return out


def compare(first, second, problems):
    """Per problem, the median cost over seeds of `first` against `second`: fewer, same or more calls."""
    counts = defaultdict(lambda: dict(fewer=0, same=0, more=0))
    for pid, problem in problems.items():
        a = statistics.median(cost(rows[pid]) for rows in first.values())
        b = statistics.median(cost(rows[pid]) for rows in second.values())
        key = "fewer" if a < b else "more" if a > b else "same"
        for name in (f"{problem['horizon']}push_{problem['difficulty']}", f"{problem['horizon']}push_all", "all"):
            counts[name][key] += 1
    return dict(counts)


def load_ours(folders, arms, problems, expect_sha):
    by_arm = defaultdict(dict)
    for folder in folders:
        for arm in arms:
            for path in sorted((folder / arm["name"]).glob("problem_*.jsonl")):
                row = json.loads(path.read_text().splitlines()[0])
                source = row["runtime_fingerprints"]["source_sha256"]
                if row["technical_error"] or source != expect_sha:
                    raise RuntimeError(f"{path}: technical_error={row['technical_error']} source_sha256={source}")
                pid = row["certification_problem_id"]
                if pid in by_arm[arm["name"]]:
                    raise RuntimeError(f"{arm['name']}: problem {row['index']} saved twice")
                by_arm[arm["name"]][pid] = row
    for arm in arms:
        missing = set(problems) - set(by_arm[arm["name"]])
        if missing:
            raise RuntimeError(f"{arm['name']}: {len(missing)} problems missing")
    return by_arm


def load_references(root, arm, problems):
    rows = {}
    for version in ("references-v1", "references-identity-repair-v2"):
        for path in sorted((root / version / arm).glob("one_keyhole/shard_*/scene_*/seed_*/outcomes.jsonl")):
            for line in path.read_text().splitlines():
                if line:
                    row = json.loads(line)
                    rows[row.get("certification_problem_id") or row["problem_id"]] = row  # v2 overrides v1
    if set(rows) != set(problems):
        raise RuntimeError(f"{arm}: references cover {len(set(rows) & set(problems))} of {len(problems)} problems")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path, action="append")
    parser.add_argument("--arms", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path, help="Tri-An's one_keyhole_frozen600 campaign root")
    parser.add_argument("--expect-source-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plot-dir", type=Path)
    args = parser.parse_args()

    problems = {row["problem_id"]: row for row in map(json.loads, (args.problems / "manifest.jsonl").read_text().splitlines())}
    arms = json.loads(args.arms.read_text())
    ours = load_ours(args.results, arms, problems, args.expect_source_sha256)
    families = defaultdict(dict)
    for arm in arms:
        family, seed = re.fullmatch(r"(.+)_s(\d+)", arm["name"]).groups()
        families[family][int(seed)] = ours[arm["name"]]
    families["Random"] = {seed: load_references(args.references, f"random_s{seed}", problems) for seed in RANDOM_SEEDS}
    tri_an_s2 = load_references(args.references, "HY5U_s2_search", problems)

    drift = {pid for rows in ours.values() for pid, row in rows.items() if not row["door_objects_match"]}
    report = dict(problems=len(problems), call_cap=3000, expect_source_sha256=args.expect_source_sha256,
                  families={f: table(runs, problems) for f, runs in families.items()},
                  families_without_door_drift={f: table(runs, problems, set(problems) - drift)["all"]
                                               for f, runs in families.items()},
                  door_drift_problems=sorted(problems[p]["index"] for p in drift),
                  versus_hy5u={f: compare(runs, families["HY5U"], problems) for f, runs in families.items() if f != "HY5U"},
                  hy5u_s2_vs_tri_an=dict(
                      problems=len(problems),
                      same_solved=sum(ours["HY5U_s2"][p]["solved"] == tri_an_s2[p]["solved"] for p in problems),
                      identical=sum((ours["HY5U_s2"][p]["solved"], ours["HY5U_s2"][p]["total_calls"]) ==
                                    (tri_an_s2[p]["solved"], tri_an_s2[p]["total_calls"]) for p in problems)))
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print_tables(report)
    if args.plot_dir:
        plot(args.plot_dir, families, problems)


def print_tables(report):
    columns = [f"{h}push_{d}" for h, d in GROUPS] + ["1push_all", "2push_all", "all"]
    header = "| | " + " | ".join(c.replace("push_", "-push ") for c in columns) + " |\n|---|" + "---:|" * len(columns)
    families = report["families"]

    def block(title, cell):
        print(f"\n{title}\n{header}")
        for family, groups in families.items():
            print(f"| {LABEL.get(family, family)} | " + " | ".join(cell(groups[c]) for c in columns) + " |")

    def spread(summary, value):
        values = [value(s) for s in summary["seeds"].values()]
        return f"{value(summary):.1f} [{min(values):.1f}-{max(values):.1f}]"

    block("Solved within 3000 calls, % of runs [worst-best seed]",
          lambda s: spread(s, lambda x: 100 * x["success_rate"]))
    block("Solved within 1 call, % of runs [worst-best seed]",
          lambda s: spread(s, lambda x: 100 * x["solved_within"][1]))
    block("Solved within 5 calls, % of runs [worst-best seed]",
          lambda s: spread(s, lambda x: 100 * x["solved_within"][5]))
    block("Solved within 30 calls, % of runs [worst-best seed]",
          lambda s: spread(s, lambda x: 100 * x["solved_within"][30]))
    block("Median calls until solved (a failed run counts as infinite)",
          lambda s: ">3000" if s["median_calls_until_success"] is None else f"{s['median_calls_until_success']:g}")
    print(f"\nPer problem against HY5U (median over seeds): fewer / same / more calls\n{header}")
    for family, counts in report["versus_hy5u"].items():
        print(f"| {LABEL.get(family, family)} | " + " | ".join(
            f"{counts[c]['fewer']} / {counts[c]['same']} / {counts[c]['more']}" for c in columns) + " |")
    check = report["hy5u_s2_vs_tri_an"]
    print(f"\nHY5U_s2 against Tri-An's HY5U_s2 run: same solved {check['same_solved']}/{check['problems']}, "
          f"identical solved and calls {check['identical']}/{check['problems']}")
    print(f"Problems where today's room finder sees different door objects: {report['door_drift_problems']}")
    print("Solved within 3000 calls over all problems, with and without them: " + ", ".join(
        f"{LABEL.get(f, f)} {100 * families[f]['all']['success_rate']:.1f} / "
        f"{100 * s['success_rate']:.1f}" for f, s in report["families_without_door_drift"].items()))


def plot(plot_dir, families, problems):
    """Success vs simulator calls per horizon and tier: an overview of every family, then each
    ablation against HY5U and Random with median lines and worst-to-best seed bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir.mkdir(parents=True, exist_ok=True)
    budgets = np.unique(np.geomspace(1, 3000, 300).round().astype(int))
    members = {g: [p for p in problems if group_key(problems[p]) == g] for g in GROUPS}

    def curves(runs, group):
        return np.array([100 * np.searchsorted(np.sort([cost(rows[p]) for p in members[group]]), budgets,
                                               side="right") / len(members[group]) for rows in runs.values()])

    def grid(title):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
        for ax, group in zip(axes.flat, GROUPS):
            ax.set_xscale("log")
            ax.set_title(f"{group[0]}-push {group[1]} ({len(members[group])} problems)")
            ax.grid(alpha=0.3)
        for ax in axes[1]:
            ax.set_xlabel("Simulator calls (log scale, cap 3000)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Runs solved (%)")
        fig.suptitle(title)
        return fig, axes

    colors = iter(["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"])
    fig, axes = grid("Frozen one-keyhole 600: median seed of every model")
    for family, runs in families.items():
        style = (dict(color="black", lw=2.8, zorder=10) if family == "HY5U" else
                 dict(color="#666666", ls="--", lw=2) if family == "Random" else dict(color=next(colors), lw=1.4))
        for ax, group in zip(axes.flat, GROUPS):
            ax.plot(budgets, np.median(curves(runs, group), axis=0), label=LABEL.get(family, family), **style)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(plot_dir / "success_vs_calls_all_models.png", dpi=150)
    plt.close(fig)

    for family, runs in families.items():
        if family in ("HY5U", "Random"):
            continue
        fig, axes = grid(f"Frozen one-keyhole 600: {LABEL.get(family, family)} vs HY5U and Random "
                         "(line = median seed, band = worst to best seed)")
        for name, style in ((family, dict(color="#4C72B0")), ("HY5U", dict(color="#C44E52")),
                            ("Random", dict(color="#666666", ls="--"))):
            for ax, group in zip(axes.flat, GROUPS):
                c = curves(families[name], group)
                ax.plot(budgets, np.median(c, axis=0), lw=2, label=LABEL.get(name, name), **style)
                ax.fill_between(budgets, c.min(axis=0), c.max(axis=0), color=style["color"], alpha=0.18, lw=0)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=11, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(plot_dir / f"{family}_vs_HY5U.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
