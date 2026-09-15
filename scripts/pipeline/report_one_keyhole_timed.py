#!/usr/bin/env python3
"""Report timed runs on Tri-An's frozen one-keyhole 600 from run_one_keyhole_frozen.py --timed.

Every table splits by horizon (1-push, 2-push) and certificate tier (easy, medium, hard).
Scoring matches eval_full_namo_walltime.summarize: a failed run takes infinite time and
infinite simulator calls. Arms group into families by dropping the seed suffix (_sN for
models, random_sNNNN pools into Random); cells show the range over seeds.

The report refuses rows with a technical error, a different code fingerprint, a different CPU
model or more than one pinned CPU. Every arm must cover the same problems; problems absent
from every arm must be among those that do not start in their certified state on these
builds (40, 277, 373, 463, 504). Each timed row is checked against a reference run of the
same arm: our untimed rows for models, Tri-An's untimed rows for Random, and his timed rows
for geometric.

Usage:
  python scripts/pipeline/report_one_keyhole_timed.py --problems <input/one_keyhole> \
      --rows <timed rows> [--rows <more timed rows>] --untimed <untimed results> \
      --references <Tri-An one_keyhole_frozen600 untimed root> --geometric-references <his timed geometric raw> \
      --expect-source-sha256 <sha> --expect-cpu "AMD EPYC 7352 24-Core Processor" --out report.json
"""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from report_one_keyhole_frozen import GROUPS, load_references

REFUSED = {40, 277, 373, 463, 504}
SECONDS = (1, 5, 30)


def family(arm):
    return "Random" if arm.startswith("random_s") else re.sub(r"_s\d+$", "", arm)


def summarize(rows):
    def median_cost(key):
        value = statistics.median(r[key] if r["solved"] else math.inf for r in rows)
        return value if math.isfinite(value) else None
    return dict(runs=len(rows), success_rate=sum(r["solved"] for r in rows) / len(rows),
                median_seconds=median_cost("t_wall"), median_calls=median_cost("total_calls"),
                solved_within_seconds={s: sum(r["solved"] and r["t_wall"] <= s for r in rows) / len(rows) for s in SECONDS},
                scoring_share=sum(r["t_score"] for r in rows) / sum(r["t_wall"] for r in rows))


def load_rows(folders, expect_sha, expect_cpu):
    by_arm = defaultdict(dict)
    for folder in folders:
        for path in sorted(folder.glob("*/problem_*.jsonl")):
            row = json.loads(path.read_text().splitlines()[0])
            source = row["runtime_fingerprints"]["source_sha256"]
            if (row["technical_error"] or source != expect_sha or row["cpu_model"] != expect_cpu
                    or len(row["cpu_affinity"]) != 1 or "t_wall" not in row):
                raise RuntimeError(f"{path}: technical_error={row['technical_error']} source={source} "
                                   f"cpu={row['cpu_model']} affinity={row['cpu_affinity']}")
            arm, pid = path.parent.name, row["certification_problem_id"]
            if pid in by_arm[arm]:
                raise RuntimeError(f"{arm}: problem {row['index']} saved twice")
            by_arm[arm][pid] = row
    return by_arm


def reference_rows(arm, args, problems):
    if arm == "geometric":
        rows = {}
        for path in args.geometric_references.glob("shard_*/outcomes.jsonl"):
            for line in path.read_text().splitlines():
                if line:
                    row = json.loads(line)
                    rows[row["certification_problem_id"]] = row
        return rows
    if arm.startswith("random_s"):
        return load_references(args.references, arm, problems)
    return {json.loads(p.read_text().splitlines()[0])["certification_problem_id"]: json.loads(p.read_text().splitlines()[0])
            for p in (args.untimed / arm).glob("problem_*.jsonl")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path, action="append")
    parser.add_argument("--untimed", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--geometric-references", required=True, type=Path)
    parser.add_argument("--expect-source-sha256", required=True)
    parser.add_argument("--expect-cpu", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only-index", type=int, action="append", help="report only these problems (smoke checks)")
    args = parser.parse_args()

    manifest = [json.loads(line) for line in (args.problems / "manifest.jsonl").read_text().splitlines() if line]
    everything = {row["problem_id"]: row for row in manifest}
    problems = {pid: row for pid, row in everything.items() if not args.only_index or row["index"] in args.only_index}
    by_arm = load_rows(args.rows, args.expect_source_sha256, args.expect_cpu)
    by_arm = {arm: {pid: row for pid, row in rows.items() if pid in problems} for arm, rows in by_arm.items()}
    covered = set.intersection(*(set(rows) for rows in by_arm.values()))
    for arm, rows in by_arm.items():
        if set(rows) != covered:
            raise RuntimeError(f"{arm}: covers {len(rows)} problems, other arms cover {len(covered)}; run incomplete")
    excluded = sorted(problems[pid]["index"] for pid in set(problems) - covered)
    if not set(excluded) <= REFUSED:
        raise RuntimeError(f"problems missing from every arm beyond the known refusals: {sorted(set(excluded) - REFUSED)}")

    pairing = {}
    for arm, rows in sorted(by_arm.items()):
        ref = reference_rows(arm, args, everything)
        compared = [pid for pid in rows if pid in ref]
        differ = [problems[pid]["index"] for pid in compared
                  if (rows[pid]["solved"], rows[pid]["total_calls"]) != (ref[pid]["solved"], ref[pid]["total_calls"])]
        pairing[arm] = dict(compared=len(compared), same_result_and_calls=len(compared) - len(differ), differ=sorted(differ))

    members = {f"{h}push_{d}": [p for p in covered if (problems[p]["horizon"], problems[p]["difficulty"]) == (h, d)]
               for h, d in GROUPS}
    members.update({f"{h}push_all": [p for p in covered if problems[p]["horizon"] == h] for h in (1, 2)}, all=sorted(covered))
    members = {name: pids for name, pids in members.items() if pids}
    families = defaultdict(list)
    for arm in sorted(by_arm):
        families[family(arm)].append(arm)
    tables = {name: {fam: {arm: summarize([by_arm[arm][p] for p in pids]) for arm in arms}
                     for fam, arms in families.items()} for name, pids in members.items()}
    nodes = {fam: {host: statistics.median(r["t_wall"] for arm in arms for r in by_arm[arm].values() if r["host"] == host)
                   for host in sorted({r["host"] for arm in arms for r in by_arm[arm].values()})}
             for fam, arms in families.items()}
    report = dict(problems=len(covered), excluded_indices=excluded, pairing=pairing, tables=tables,
                  median_seconds_by_host=nodes)
    args.out.write_text(json.dumps(report, indent=1) + "\n")

    def span(values, fmt):
        finite = [v for v in values if v is not None]
        if not finite:
            return "not reached"
        low, high = fmt(min(finite)), fmt(max(finite))
        text = low if low == high else f"{low}-{high}"
        return text if len(finite) == len(values) else text + " (some seeds not reached)"

    print(f"{len(covered)} problems; excluded {excluded}")
    for arm, p in pairing.items():
        print(f"pairing {arm}: {p['same_result_and_calls']}/{p['compared']} same result and calls; differ {p['differ'][:12]}")
    for fam, hosts in nodes.items():
        print(f"median seconds by host, {fam}: " + ", ".join(f"{h.split('.')[0]} {v:.2f}" for h, v in hosts.items()))
    print("\n| group | family | solved % | median seconds | solved within 1 s % | 5 s % | 30 s % | median calls |")
    print("|---|---|---|---|---|---|---|---|")
    for name, fams in tables.items():
        for fam, seeds in fams.items():
            cells = list(seeds.values())
            print(f"| {name} | {fam} | {span([100 * c['success_rate'] for c in cells], lambda v: f'{v:.1f}')} "
                  f"| {span([c['median_seconds'] for c in cells], lambda v: f'{v:.2f}')} | "
                  + " | ".join(span([100 * c['solved_within_seconds'][s] for c in cells], lambda v: f"{v:.1f}") for s in SECONDS)
                  + f" | {span([c['median_calls'] for c in cells], lambda v: f'{v:g}')} |")


if __name__ == "__main__":
    main()
