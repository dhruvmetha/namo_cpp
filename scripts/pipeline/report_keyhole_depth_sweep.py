#!/usr/bin/env python3
"""Report the keyhole600 depth sweep: the 300 two-push problems with the search depth capped at 2 to 5.

Every table splits by certificate tier (easy, medium, hard). Within a tier there is one row per
arm family (the model, Random) and depth cap. Scoring is report_one_keyhole_frozen.py's: a failed
run costs infinite simulator calls, the three seeds pool for the headline number, and the bracket
is the lowest and highest single seed. Every problem here has a 2-push solution, so the plan-length
columns show how often a deeper cap returns a longer plan than the problem needs. The paired
columns compare each problem's calls at depth d with the same seed at depth 2.

Refuses to run unless every arm has every two-push problem exactly once, from an accepted code
fingerprint. Rows may come from several --results folders (Amarel plus arrakis for the problems
whose starting state does not match their certificate on Amarel's build).

Usage:
  python scripts/pipeline/report_keyhole_depth_sweep.py --problems <input/one_keyhole> \
      --results <amarel results> --results <arrakis results> --arms arms.json \
      --expect-source-sha256 <sha> --out report.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from report_one_keyhole_frozen import WITHIN, cost, summarize

TIERS = ("easy", "medium", "hard")
NAME = re.compile(r"^(?P<family>.+)_s(?P<seed>\d+)_h(?P<depth>\d+)$")
LABEL = {"nfle": "no family, no local, no edge identity", "random": "Random ordering"}


def load(folders, arms, problems, expect_sha):
    rows = {}
    for folder in folders:
        for arm in arms:
            for path in sorted((folder / arm["name"]).glob("problem_*.jsonl")):
                row = json.loads(path.read_text().splitlines()[0])
                if row["index"] not in problems:
                    continue
                key = (arm["name"], row["index"])
                if key in rows:
                    raise ValueError(f"{arm['name']} problem {row['index']} saved twice")
                sha = (row.get("runtime_fingerprints") or {}).get("source_sha256")
                if sha not in expect_sha:
                    raise ValueError(f"{path}: source_sha256 {sha} not accepted")
                if row["search"]["hmax"] != arm["hmax"]:
                    raise ValueError(f"{path}: hmax {row['search']['hmax']} but arm says {arm['hmax']}")
                rows[key] = row
    missing = [(a["name"], i) for a in arms for i in problems if (a["name"], i) not in rows]
    if missing:
        raise ValueError(f"{len(missing)} runs missing, first {missing[:5]}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path, action="append")
    parser.add_argument("--arms", required=True, type=Path)
    parser.add_argument("--expect-source-sha256", required=True, action="append")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    manifest = [json.loads(line) for line in (args.problems / "manifest.jsonl").read_text().splitlines() if line]
    problems = {r["index"]: r for r in manifest if r["horizon"] == 2}
    arms = json.loads(args.arms.read_text())
    rows = load(args.results, arms, problems, set(args.expect_source_sha256))

    by_cell = defaultdict(lambda: defaultdict(list))   # (tier, family, depth) -> seed -> rows
    for arm in arms:
        m = NAME.match(arm["name"])
        for index, problem in problems.items():
            by_cell[(problem["difficulty"], m["family"], int(m["depth"]))][m["seed"]].append(rows[(arm["name"], index)])

    report = dict(problems=len(problems), arms=[a["name"] for a in arms], cells=[])
    for (tier, family, depth), seeds in sorted(by_cell.items(), key=lambda kv: (TIERS.index(kv[0][0]), kv[0][1], kv[0][2])):
        pooled = [r for rs in seeds.values() for r in rs]
        cell = dict(tier=tier, family=family, depth=depth, **summarize(pooled))
        per_seed = {s: summarize(rs) for s, rs in seeds.items()}
        cell["solved_within_seed_range"] = {k: [min(v["solved_within"][k] for v in per_seed.values()),
                                                max(v["solved_within"][k] for v in per_seed.values())] for k in WITHIN}
        cell["plan_length"] = dict(sorted(Counter(r["plan_len"] for r in pooled if r["solved"]).items()))
        if depth != 2:
            base = by_cell[(tier, family, 2)]
            paired = Counter()
            for seed, rs in seeds.items():
                ref = {r["index"]: cost(r) for r in base[seed]}
                for r in rs:
                    d = cost(r) - ref[r["index"]] if cost(r) != ref[r["index"]] else 0
                    paired["fewer" if d < 0 else "more" if d > 0 else "same"] += 1
            cell["vs_depth2_calls"] = dict(paired)
        report["cells"].append(cell)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")

    show = (1, 2, 5, 10, 30, 100, 1000, 3000)
    for tier in TIERS:
        print(f"\n### 2-push {tier} (n={sum(p['difficulty'] == tier for p in problems.values())} problems x 3 seeds)\n")
        print("| arm | depth | " + " | ".join(f"@{k}" for k in show) + " | median calls | plan lengths | vs depth 2 fewer/same/more |")
        print("|---|---|" + "---|" * len(show) + "---|---|---|")
        for c in (c for c in report["cells"] if c["tier"] == tier):
            lo_hi = c["solved_within_seed_range"]
            vals = " | ".join(f"{100 * c['solved_within'][k]:.1f} [{100 * lo_hi[k][0]:.0f}-{100 * lo_hi[k][1]:.0f}]" for k in show)
            v = c.get("vs_depth2_calls")
            vs = f"{v.get('fewer', 0)}/{v.get('same', 0)}/{v.get('more', 0)}" if v else ""
            plans = ", ".join(f"{k}:{n}" for k, n in c["plan_length"].items())
            print(f"| {LABEL.get(c['family'], c['family'])} | {c['depth']} | {vals} | {c['median_calls_until_success']} | {plans} | {vs} |")


if __name__ == "__main__":
    main()
