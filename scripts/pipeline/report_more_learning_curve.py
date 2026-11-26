#!/usr/bin/env python3
"""Learning curve: one network trained on growing prefixes of its collection, on a subset of Tri-An's frozen
one-keyhole 600 (EXP-2026-11-24-more-faithful, MORE's PushNet seed 1 on Amarel).

Each arm in --arms carries "search" (a label shared by the arms that use one search) and "training_calls"
(simulator calls in the collection prefix the network was trained on). Optional keys: "heldout", a
verify_more_eca.py output whose correlation with sampled labels becomes a column, and "checkpoint_sha256",
which every row of the arm must match. The arms with one search form one curve.

Every table splits by horizon (1-push, 2-push) and certificate tier (easy, medium, hard). Scoring matches
report_one_keyhole_frozen.py, whose loaders this script reuses: a failed run costs infinite simulator calls,
every arm must have every problem of the subset (--only-index) from an accepted code fingerprint, and Random
is Tri-An's five uniform seeds on the same problems. All rows must also come from one binding build.
Pairings stay inside this run. Each prefix is paired with the next smaller one under the same search, and
every arm with Random; nothing here pairs these rows with runs from another machine type.

Usage:
  python scripts/pipeline/report_more_learning_curve.py --problems <input/one_keyhole> --results <dir> \
      --arms arms.json --references <Tri-An campaign root> --expect-source-sha256 <sha> \
      --only-index 0 --only-index 1 ... --out report.json --plot-dir <dir> --runs-csv runs.csv
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_one_keyhole_frozen import GROUPS, RANDOM_SEEDS, compare, load_ours, load_references, table, write_runs_csv  # noqa: E402

COLUMNS = [f"{h}push_{d}" for h, d in GROUPS] + ["1push_all", "2push_all", "all"]


def heldout_correlation(path):
    return float(re.search(r"correlation with sampled labels: (-?[\d.]+)", Path(path).read_text()).group(1))


def millions(calls):
    return f"{calls / 1e6:.3g}M"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path, action="append")
    parser.add_argument("--arms", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path, help="Tri-An's one_keyhole_frozen600 campaign root")
    parser.add_argument("--expect-source-sha256", required=True, action="append")
    parser.add_argument("--only-index", required=True, type=int, action="append", help="the problem subset, by index")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--runs-csv", type=Path)
    args = parser.parse_args()

    every = {row["problem_id"]: row for row in map(json.loads, (args.problems / "manifest.jsonl").read_text().splitlines())}
    only = set(args.only_index)
    problems = {pid: p for pid, p in every.items() if p["index"] in only}
    if len(problems) != len(only):
        raise RuntimeError(f"{len(only)} indices asked, {len(problems)} found in the manifest")
    arms = json.loads(args.arms.read_text())
    order = {search: i for i, search in enumerate(dict.fromkeys(a["search"] for a in arms))}  # searches in file order
    arms.sort(key=lambda a: (order[a["search"]], a["training_calls"]))
    ours = load_ours(args.results, arms, problems, args.expect_source_sha256)
    for arm in arms:
        rows = ours[arm["name"]]
        extra = sorted(rows[p]["index"] for p in set(rows) - set(problems))
        if extra:
            raise RuntimeError(f"{arm['name']}: rows outside the subset, problems {extra[:10]}")
        if "checkpoint_sha256" in arm:
            other = {r["runtime_fingerprints"]["checkpoint_sha256"] for r in rows.values()} - {arm["checkpoint_sha256"]}
            if other:
                raise RuntimeError(f"{arm['name']}: rows from checkpoints {sorted(other)}")
    bindings = {r["runtime_fingerprints"]["binding_sha256"] for rows in ours.values() for r in rows.values()}
    if len(bindings) != 1:
        raise RuntimeError(f"rows come from {len(bindings)} binding builds: {sorted(bindings)}")

    families = {arm["name"]: {1: ours[arm["name"]]} for arm in arms}
    random = {seed: {p: r for p, r in load_references(args.references, f"random_s{seed}", every).items() if p in problems}
              for seed in RANDOM_SEEDS}
    curves = defaultdict(list)
    for arm in arms:
        curves[arm["search"]].append(arm)
    report = dict(
        problems=len(problems), problem_indices=sorted(only), call_cap=3000, binding_sha256=bindings.pop(),
        expect_source_sha256=args.expect_source_sha256,
        arms={arm["name"]: dict(arm, heldout_correlation=heldout_correlation(arm["heldout"]) if "heldout" in arm else None)
              for arm in arms},
        tables=dict({name: table(runs, problems) for name, runs in families.items()}, Random=table(random, problems)),
        larger_vs_next_smaller={f"{b['name']} vs {a['name']}": compare(families[b["name"]], families[a["name"]], problems)
                                for curve in curves.values() for a, b in zip(curve, curve[1:])},
        versus_random={name: compare(runs, random, problems) for name, runs in families.items()})
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print_tables(report, curves)
    if args.plot_dir:
        plot(args.plot_dir, report, curves)
    if args.runs_csv:
        lines = [(name, 1, "learning_curve", row) for name, runs in families.items() for row in runs[1].values()]
        lines += [("Random", seed, "tri_an_frozen", row) for seed, rows in random.items() for row in rows.values()]
        write_runs_csv(args.runs_csv, lines, problems)


def print_tables(report, curves):
    header = ("| search | trained on (simulator calls) | held-out correlation | "
              + " | ".join(c.replace("push_", "-push ") for c in COLUMNS) + " |\n|---|---:|---:|" + "---:|" * len(COLUMNS))
    tables, arms = report["tables"], report["arms"]

    def block(title, cell):
        print(f"\n{title}\n{header}")
        for curve in curves.values():
            for arm in curve:
                corr = arms[arm["name"]]["heldout_correlation"]
                print(f"| {arm['search']} | {millions(arm['training_calls'])} | {'' if corr is None else f'{corr:.3f}'} | "
                      + " | ".join(cell(tables[arm["name"]][c], False) for c in COLUMNS) + " |")
        print("| Random ordering (Tri-An's 5 seeds) | | | " + " | ".join(cell(tables["Random"][c], True) for c in COLUMNS) + " |")

    def percent(value):
        def cell(summary, spread):
            text = f"{100 * value(summary):.1f}"
            if spread:
                seeds = [100 * value(s) for s in summary["seeds"].values()]
                text += f" [{min(seeds):.1f}-{max(seeds):.1f}]"
            return text
        return cell

    for k in (1, 5, 30, 100):
        block(f"Solved within {k} call{'s' if k > 1 else ''}, % of runs (Random: pooled [worst-best seed])",
              percent(lambda s, k=k: s["solved_within"][k]))
    block("Solved within 3000 calls, % of runs (Random: pooled [worst-best seed])", percent(lambda s: s["success_rate"]))
    block("Median calls until solved (a failed run counts as infinite)",
          lambda s, spread: ">3000" if s["median_calls_until_success"] is None else f"{s['median_calls_until_success']:g}")

    pairs = "two-sided sign test p on fewer vs more"
    print(f"\nPer problem, each training prefix against the next smaller one with the same search: fewer / same / more "
          f"calls for the larger prefix, {pairs}\n| search | prefixes | "
          + " | ".join(c.replace("push_", "-push ") for c in COLUMNS) + " |\n|---|---|" + "---:|" * len(COLUMNS))
    for curve in curves.values():
        for a, b in zip(curve, curve[1:]):
            counts = report["larger_vs_next_smaller"][f"{b['name']} vs {a['name']}"]
            print(f"| {b['search']} | {millions(b['training_calls'])} vs {millions(a['training_calls'])} | "
                  + " | ".join(pair(counts[c]) for c in COLUMNS) + " |")
    print(f"\nPer problem against Random (median over its 5 seeds): fewer / same / more calls, {pairs}\n"
          "| search | trained on | " + " | ".join(c.replace("push_", "-push ") for c in COLUMNS) + " |\n|---|---:|"
          + "---:|" * len(COLUMNS))
    for curve in curves.values():
        for arm in curve:
            counts = report["versus_random"][arm["name"]]
            print(f"| {arm['search']} | {millions(arm['training_calls'])} | " + " | ".join(pair(counts[c]) for c in COLUMNS) + " |")
    print(f"\nAll {sum(len(c) for c in curves.values())} arms: {report['problems']} problems, one binding build "
          f"({report['binding_sha256'][:8]}).")


def pair(counts):
    return f"{counts['fewer']} / {counts['same']} / {counts['more']}, p={counts['sign_test_p']:.2g}"


def plot(plot_dir, report, curves):
    """Median calls and solved within 30 calls against training calls, one panel per horizon and tier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    tables = report["tables"]

    def median(summary):
        value = summary["median_calls_until_success"]
        return 3000 if value is None else value

    rows = [("Median calls until solved\n(3000 = half the runs fail)", median, "log"),
            ("Solved within 30 calls (%)", lambda s: 100 * s["solved_within"][30], "linear")]
    fig, axes = plt.subplots(2, 6, figsize=(20, 7.5), sharex=True, sharey="row")
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860"]
    ticks = sorted({a["training_calls"] for curve in curves.values() for a in curve})
    for i, ((ylabel, value, scale), row) in enumerate(zip(rows, axes)):
        for ax, (h, d) in zip(row, GROUPS):
            group = f"{h}push_{d}"
            ax.set_xscale("log")
            ax.set_yscale(scale)
            ax.grid(alpha=0.3)
            for color, (search, curve) in zip(colors, curves.items()):
                ax.plot([a["training_calls"] for a in curve], [value(tables[a["name"]][group]) for a in curve],
                        marker="o", lw=2, color=color, label=search)
            ax.axhline(value(tables["Random"][group]), color="#666666", ls="--", lw=1.5, label="Random ordering")
            ax.set_ylim(*((0.8, 4000) if scale == "log" else (-3, 103)))
            ax.set_xticks(ticks, [millions(t) for t in ticks])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(f"{h}-push {d}")
            else:
                ax.set_xlabel("Trained on (simulator calls)")
        row[0].set_ylabel(ylabel)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=len(curves) + 1, fontsize=11, frameon=False)
    fig.suptitle(f"Learning curve on {report['problems']} frozen one-keyhole problems (one training seed per point)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(plot_dir / "learning_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
