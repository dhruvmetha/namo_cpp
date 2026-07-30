#!/usr/bin/env python3
"""Build an episode-key subset for unsolved search tails in one fixed difficulty tier."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from namo import eval_sets
from agg_search_eval import (
    _canonical_xml,
    _load_divisions,
    _normalize_tier,
    _read_jsonl,
    _row_sims,
    _search_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf-dir", required=True)
    parser.add_argument("--source-key", default=str(eval_sets.PURE2PUSH))
    parser.add_argument("--divisions", default=str(eval_sets.DIVISIONS))
    parser.add_argument("--tier", choices=("easy", "medium", "hard"), required=True)
    parser.add_argument("--min-sims", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = _read_jsonl(args.leaf_dir)
    config = _search_config(rows)
    divisions = _load_divisions(args.divisions)
    source = json.load(open(args.source_key))
    source_lookup = {}
    for xml, records in source.items():
        for record in records:
            key = (_canonical_xml(xml), record["object_id"], record.get("region"))
            if key in source_lookup:
                raise RuntimeError(f"duplicate source episode: {key}")
            source_lookup[key] = (xml, record)

    selected = {}
    tier_unsolved = 0
    for row in rows:
        key = (_canonical_xml(row["xml"]), row["object_id"], row.get("region"))
        division = divisions.get(key)
        if division is None:
            continue
        if _normalize_tier(division) != args.tier or row["solved"]:
            continue
        tier_unsolved += 1
        if _row_sims(row) < args.min_sims:
            continue
        if key not in source_lookup:
            raise RuntimeError(f"tail episode absent from source key: {key}")
        if key in selected:
            raise RuntimeError(f"duplicate tail episode: {key}")
        selected[key] = source_lookup[key]

    output = defaultdict(list)
    for _key, (xml, record) in sorted(selected.items()):
        output[xml].append(record)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(dict(output), stream, indent=2)
    print(json.dumps({
        "tier": args.tier,
        "tier_unsolved": tier_unsolved,
        "min_sims": args.min_sims,
        "selected": len(selected),
        "search": config,
        "out": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
