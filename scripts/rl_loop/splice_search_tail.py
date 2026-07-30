#!/usr/bin/env python3
"""Splice longer reruns into one fixed-tier search-eval tail."""
import argparse
import glob
import json
from pathlib import Path

from namo.paths import resolve


def _rows(directories):
    rows = []
    for directory in directories:
        for path in sorted(glob.glob(str(Path(directory) / "shard_*.jsonl"))):
            with open(path) as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def _key(row):
    return str(resolve(row["xml"])), row["object_id"], row.get("region")


def _division_lookup(path):
    raw = json.load(open(path))
    return {
        (str(resolve(xml)), row["object_id"], row.get("region")): row["division"]
        for xml, records in raw.items()
        for row in records
    }


def _config(rows):
    configs = {json.dumps(row["search"], sort_keys=True) for row in rows}
    if len(configs) != 1:
        raise RuntimeError(f"expected one search config, got {len(configs)}")
    return json.loads(next(iter(configs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--tail-dirs", required=True, nargs="+")
    parser.add_argument("--divisions", required=True)
    parser.add_argument("--tier", default="hard")
    parser.add_argument("--base-budget", type=int, required=True)
    parser.add_argument("--tail-budget", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = _rows([args.base_dir])
    tail = _rows(args.tail_dirs)
    divisions = _division_lookup(args.divisions)
    tier_rows = [row for row in base if divisions.get(_key(row)) == args.tier]
    expected_tail = {
        _key(row) for row in tier_rows if not row["solved"] and row["sims"] == args.base_budget
    }
    tail_by_key = {_key(row): row for row in tail}
    if len(tail_by_key) != len(tail):
        raise RuntimeError("duplicate tail episode rows")
    if set(tail_by_key) != expected_tail:
        raise RuntimeError(
            f"tail keys differ: missing={len(expected_tail - set(tail_by_key))}, "
            f"extra={len(set(tail_by_key) - expected_tail)}"
        )

    base_config = _config(base)
    tail_config = _config(tail)
    if base_config.get("sim_budget") != args.base_budget:
        raise RuntimeError("base sim budget mismatch")
    if tail_config.get("sim_budget") != args.tail_budget:
        raise RuntimeError("tail sim budget mismatch")
    left = {k: v for k, v in base_config.items() if k != "sim_budget"}
    right = {k: v for k, v in tail_config.items() if k != "sim_budget"}
    if left != right:
        raise RuntimeError("base and tail search configs differ beyond sim_budget")
    bad_early = [row for row in tail if row["solved"] and row["sims"] <= args.base_budget]
    if bad_early:
        raise RuntimeError(f"{len(bad_early)} tail reruns changed outcome at or before base budget")

    spliced = []
    for row in tier_rows:
        replacement = tail_by_key.get(_key(row), row)
        replacement = {**replacement, "search": tail_config}
        spliced.append(replacement)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as stream:
        for row in spliced:
            stream.write(json.dumps(row) + "\n")

    solved = sum(bool(row["solved"]) for row in spliced)
    exhausted = sum(not row["solved"] and row["sims"] < args.tail_budget for row in tail)
    capped = sum(not row["solved"] and row["sims"] == args.tail_budget for row in tail)
    print(json.dumps({
        "tier": args.tier,
        "n": len(spliced),
        "base_solved": sum(bool(row["solved"]) for row in tier_rows),
        "tail_episodes": len(tail),
        "tail_solved": sum(bool(row["solved"]) for row in tail),
        "tail_exhausted": exhausted,
        "tail_capped": capped,
        "final_solved": solved,
        "final_solve_rate": solved / len(spliced),
    }, indent=2))


if __name__ == "__main__":
    main()
