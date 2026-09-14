#!/usr/bin/env python3
"""Report the 2026-09-14 Full NAMO rerun against Tri-An's frozen Random runs.

Scoring copies eval_full_namo_walltime.summarize: a failed run costs infinite
simulator calls, so a median that lands on a failure prints as ">cap". The report
refuses to run unless every HY5U arm has every scene with the expected code
fingerprint.

Two side questions come from rows that already exist:
  * Random on the rerun's code vs Tri-An's Random on the frozen code, same scene and
    seed. The rerun saved these Random rows before Random was dropped.
  * HY5U_s2 search before the rendering-goal fix vs after, same scene and seed.

Usage:
  python scripts/pipeline/report_full_namo_rerun.py --manifest-dir <rerun>/input/full_namo \
      --results <rerun>/results --expect-source-sha256 <sha> \
      [--before-fix <folder of HY5U_s2 search outcomes.jsonl>]... --out report.json
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

TIERS = ("easy", "medium", "hard", "unresolved")
HY5U = ("HY5U_s1", "HY5U_s2", "HY5U_s3")
RANDOM_SEEDS = (7000, 8000, 9000, 10000, 11000)
WITHIN = (10, 30, 100, 300, 1000, 3000, 9000)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def cost(row):
    return row["total_calls"] if row["solved"] else math.inf


def summarize(rows):
    median = statistics.median(cost(r) for r in rows)
    return dict(runs=len(rows), solved=sum(r["solved"] for r in rows),
                success_rate=sum(r["solved"] for r in rows) / len(rows),
                median_calls_until_success=median if math.isfinite(median) else None,
                solved_within={k: sum(cost(r) <= k for r in rows) / len(rows) for k in WITHIN})


def by_tier(rows_of_scene, problems):
    """rows_of_scene: problem_id -> list of rows. Returns tier -> summary, plus 'all'."""
    table = {t: summarize([r for pid, rows in rows_of_scene.items() if problems[pid]["difficulty"] == t
                           for r in rows]) for t in TIERS}
    table["all"] = summarize([r for rows in rows_of_scene.values() for r in rows])
    return table


def compare(first, second, problems):
    """Per scene, the median cost of `first` against `second`: fewer, same or more calls."""
    counts = {t: dict(fewer=0, same=0, more=0) for t in (*TIERS, "all")}
    for pid in problems:
        a = statistics.median(cost(r) for r in first[pid])
        b = statistics.median(cost(r) for r in second[pid])
        key = "fewer" if a < b else "more" if a > b else "same"
        counts[problems[pid]["difficulty"]][key] += 1
        counts["all"][key] += 1
    return counts


def tier_of(rows):
    """The frozen400 rule (input/full_namo/summary.json): failures rank above successes;
    median <30 easy, <300 medium, otherwise hard if any seed solves; 0/5 is unresolved."""
    if not any(r["solved"] for r in rows):
        return "unresolved"
    median = sorted(cost(r) for r in rows)[len(rows) // 2]
    return "easy" if median < 30 else "medium" if median < 300 else "hard"


def agreement(new, old):
    """Same scene and seed on two code versions: how many rows agree on solved and calls?"""
    pairs = [(new[k], old[k]) for k in new]
    differ = [dict(problem_id=k[0], seed=k[1], new=[new[k]["solved"], new[k]["total_calls"]],
                   old=[old[k]["solved"], old[k]["total_calls"]])
              for k in new if (new[k]["solved"], new[k]["total_calls"]) != (old[k]["solved"], old[k]["total_calls"])]
    return dict(pairs=len(pairs), identical=len(pairs) - len(differ),
                same_solved=sum(n["solved"] == o["solved"] for n, o in pairs), differ=differ)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--expect-source-sha256", required=True)
    parser.add_argument("--random", type=Path, help="default: <manifest-dir>/provenance/random5-outcomes.jsonl")
    parser.add_argument("--before-fix", action="append", type=Path, default=[])
    parser.add_argument("--new-code-random", type=Path,
                        help="folder of random_s<seed>/ rows saved on the rerun code (default: --results)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plot", type=Path, help="also draw success vs simulator calls with seed bands (PNG)")
    parser.add_argument("--runs-csv", type=Path, help="also write one line per run: HY5U, Random and before-fix rows")
    args = parser.parse_args()

    problems = {}
    for scene in read_jsonl(args.manifest_dir / "manifest.jsonl"):
        ids = {run["problem_id"] for run in scene["runs"]}
        assert len(ids) == 1, scene["index"]
        problems[ids.pop()] = scene

    hy5u = defaultdict(list)
    for arm in HY5U:
        seen = set()
        for path in sorted((args.results / arm).glob("scene_*.json")):
            row = json.loads(path.read_text())
            source = row["runtime_fingerprints"]["source_sha256"]
            if row["technical_error"] or source != args.expect_source_sha256:
                raise RuntimeError(f"{path}: technical_error={row['technical_error']} source_sha256={source}")
            assert problems[row["problem_id"]]["index"] == row["scene"]["index"], path
            seen.add(row["problem_id"])
            hy5u[row["problem_id"]].append(row)
        if seen != set(problems):
            raise RuntimeError(f"{arm}: {len(set(problems) - seen)} scenes missing")

    random = defaultdict(list)
    random_by_seed = {}
    for row in read_jsonl(args.random or args.manifest_dir / "provenance/random5-outcomes.jsonl"):
        random[row["problem_id"]].append(row)
        random_by_seed[row["problem_id"], row["shuffle_seed"]] = row
    assert set(random) == set(problems), "Random outcomes do not cover the manifest"
    assert all(sorted(r["shuffle_seed"] for r in rows) == list(RANDOM_SEEDS) for rows in random.values())
    assert all(tier_of(rows) == problems[pid]["difficulty"] for pid, rows in random.items()), "tier rule drifted"

    report = dict(scenes=len(problems), call_cap=9000, expect_source_sha256=args.expect_source_sha256,
                  hy5u_pooled=by_tier(hy5u, problems), random_pooled=by_tier(random, problems),
                  hy5u_seeds={arm: by_tier({pid: [r for r in rows if r["arm"] == arm] for pid, rows in hy5u.items()},
                                           problems) for arm in HY5U},
                  random_seeds={seed: by_tier({pid: [r for r in rows if r["shuffle_seed"] == seed]
                                               for pid, rows in random.items()}, problems) for seed in RANDOM_SEEDS},
                  hy5u_vs_random_per_scene=compare(hy5u, random, problems))

    new_random = {}
    for seed in RANDOM_SEEDS:
        for path in sorted(((args.new_code_random or args.results) / f"random_s{seed}").glob("scene_*.json")):
            row = json.loads(path.read_text())
            if not row["technical_error"] and row["runtime_fingerprints"]["source_sha256"] == args.expect_source_sha256:
                new_random[row["problem_id"], seed] = row
    report["random_new_code_vs_frozen"] = agreement(new_random, random_by_seed)
    # Would the tiers move if they were recomputed from the rerun code's Random rows?
    moved = defaultdict(int)
    for pid in problems:
        rows = [new_random[pid, seed] for seed in RANDOM_SEEDS if (pid, seed) in new_random]
        if len(rows) == len(RANDOM_SEEDS):
            moved[f"{problems[pid]['difficulty']}->{tier_of(rows)}"] += 1
    report["tiers_from_new_code_random"] = dict(moved)

    before = {}
    if args.before_fix:
        for folder in args.before_fix:
            for path in folder.rglob("outcomes.jsonl"):
                for row in read_jsonl(path):
                    assert row["shuffle_seed"] == 7000 and row["method"] == "model", path
                    assert row["problem_id"] not in before, path
                    before[row["problem_id"]] = row
        if set(before) != set(problems):
            raise RuntimeError(f"before-fix rows cover {len(set(before) & set(problems))} of {len(problems)} scenes")
        after = {pid: next(r for r in rows if r["arm"] == "HY5U_s2") for pid, rows in hy5u.items()}
        report["hy5u_s2_before_fix"] = by_tier({pid: [row] for pid, row in before.items()}, problems)
        report["hy5u_s2_after_vs_before_per_scene"] = compare({p: [r] for p, r in after.items()},
                                                             {p: [r] for p, r in before.items()}, problems)
        report["hy5u_s2_after_vs_before_same_seed"] = agreement({(p, 7000): r for p, r in after.items()},
                                                               {(p, 7000): r for p, r in before.items()})

    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print_tables(report)
    if args.plot:
        plot_success_vs_calls(args.plot, hy5u, random, problems)
    if args.runs_csv:
        lines = [("HY5U", int(row["arm"].rsplit("_s", 1)[1]), "rerun", row) for rows in hy5u.values() for row in rows]
        lines += [("Random", row["shuffle_seed"], "tri_an_frozen", row) for rows in random.values() for row in rows]
        lines += [("HY5U_before_fix", 2, "tri_an_frozen", row) for row in before.values()]
        write_runs_csv(args.runs_csv, lines, problems)


def write_runs_csv(path, lines, problems):
    """One line per run, sorted by model, seed and scene; the per-run JSON keeps everything else."""
    import csv

    lines.sort(key=lambda line: (line[0], line[1], problems[line[3]["problem_id"]]["index"]))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "source", "scene_index", "difficulty", "template", "solved",
                         "total_calls", "outcome", "binding_sha256"])
        for model, seed, source, row in lines:
            scene = problems[row["problem_id"]]
            writer.writerow([model, seed, source, scene["index"], scene["difficulty"], scene["template"],
                             int(row["solved"]), row["total_calls"], row.get("outcome"),
                             (row.get("runtime_fingerprints") or {}).get("binding_sha256", "")[:8]])


def plot_success_vs_calls(path, hy5u, random, problems):
    """Share of runs solved within k simulator calls, one panel per tier: median seed, band from worst to best seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    budgets = np.unique(np.geomspace(1, 9000, 400).round().astype(int))
    arms = (("HY5U search, 3 seeds", hy5u, "arm", dict(color="#C44E52", ls="-")),
            ("Random ordering, 5 seeds", random, "shuffle_seed", dict(color="#666666", ls="--")))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, tier in zip(axes.flat, (*TIERS, "all")):
        pids = [pid for pid in problems if tier == "all" or problems[pid]["difficulty"] == tier]
        for label, rows_of_scene, seed_key, style in arms:
            costs = defaultdict(list)
            for pid in pids:
                for row in rows_of_scene[pid]:
                    costs[row[seed_key]].append(cost(row))
            curves = np.array([100 * np.searchsorted(np.sort(c), budgets, side="right") / len(c)
                               for c in costs.values()])
            ax.plot(budgets, np.median(curves, axis=0), lw=2, label=label, **style)
            ax.fill_between(budgets, curves.min(axis=0), curves.max(axis=0), color=style["color"], alpha=0.2, lw=0)
        ax.set_xscale("log")
        ax.set_title(f"{tier.capitalize()} ({len(pids)} scenes)")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Simulator calls (log scale, cap 9000)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Runs solved (%)")
    axes[1, 2].axis("off")
    axes[1, 2].legend(*axes[0, 0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=11,
                      title="line = median seed\nband = worst to best seed", title_fontsize=10)
    fig.suptitle("Full NAMO frozen400: share of runs solved within a simulator-call budget")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def print_tables(report):
    columns = (*TIERS, "all")

    def line(label, cells):
        print(f"| {label} | " + " | ".join(cells) + " |")

    def median_text(summary):
        value = summary["median_calls_until_success"]
        return f">{report['call_cap']}" if value is None else f"{value:g}"

    groups = [("HY5U, 3 seeds", report["hy5u_pooled"]),
              *[(arm, report["hy5u_seeds"][arm]) for arm in HY5U],
              ("Random, 5 seeds", report["random_pooled"])]
    if "hy5u_s2_before_fix" in report:
        groups.append(("HY5U_s2 before fix", report["hy5u_s2_before_fix"]))

    print("Solved (runs solved / runs)")
    line("", columns)
    line("---", ["---"] * len(columns))
    for label, table in groups:
        line(label, [f"{100 * table[t]['success_rate']:.1f}% ({table[t]['solved']}/{table[t]['runs']})"
                     for t in columns])
    print("\nMedian simulator calls until solved (a failed run counts as infinite)")
    line("", columns)
    line("---", ["---"] * len(columns))
    for label, table in groups:
        line(label, [median_text(table[t]) for t in columns])
    for k in (30, 300):
        print(f"\nSolved within {k} simulator calls")
        line("", columns)
        line("---", ["---"] * len(columns))
        for label, table in groups:
            line(label, [f"{100 * table[t]['solved_within'][k]:.1f}%" for t in columns])
    print("\nPer scene: HY5U median of 3 seeds vs Random median of 5 seeds")
    line("", columns)
    line("---", ["---"] * len(columns))
    counts = report["hy5u_vs_random_per_scene"]
    for key, label in (("fewer", "HY5U needs fewer calls"), ("same", "same"), ("more", "HY5U needs more calls")):
        line(label, [str(counts[t][key]) for t in columns])
    check = report["random_new_code_vs_frozen"]
    print(f"\nRandom, rerun code vs frozen code, same scene and seed: {check['identical']}/{check['pairs']} identical "
          f"(solved and calls), {check['same_solved']}/{check['pairs']} same solved")
    moved = report["tiers_from_new_code_random"]
    kept = sum(n for key, n in moved.items() if key.split("->")[0] == key.split("->")[1])
    print(f"Tiers recomputed from rerun-code Random (scenes with all 5 seeds): {kept}/{sum(moved.values())} unchanged; "
          + ", ".join(f"{key}: {n}" for key, n in sorted(moved.items()) if key.split("->")[0] != key.split("->")[1]))
    if "hy5u_s2_after_vs_before_per_scene" in report:
        counts = report["hy5u_s2_after_vs_before_per_scene"]
        print("\nPer scene: HY5U_s2 after fix vs before fix (same seed)")
        line("", columns)
        line("---", ["---"] * len(columns))
        for key, label in (("fewer", "after needs fewer calls"), ("same", "same"), ("more", "after needs more calls")):
            line(label, [str(counts[t][key]) for t in columns])


if __name__ == "__main__":
    main()
