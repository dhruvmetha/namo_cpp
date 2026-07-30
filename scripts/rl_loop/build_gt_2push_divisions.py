#!/usr/bin/env python3
"""Replace sampled 2push tiers with exhaustive-GT setup-density tiers."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path


def _tier(setup_pct):
    if setup_pct is None:
        return "unknown"
    return "hard" if setup_pct < 5 else ("medium" if setup_pct < 30 else "easy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz-manifest", required=True)
    parser.add_argument("--source-divisions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.load(open(args.viz_manifest))
    arm = next(iter(manifest["index"]))
    gt_rows = manifest["index"][arm]
    gt = {(os.path.realpath(row["xml"]), row["object_id"]): row for row in gt_rows}
    if len(gt) != len(gt_rows):
        raise RuntimeError("duplicate exhaustive-GT episode keys")

    source = json.load(open(args.source_divisions))
    output = {}
    counts = Counter()
    seen = set()
    for xml, records in source.items():
        output[xml] = []
        for record in records:
            key = (os.path.realpath(xml), record["object_id"])
            if key in seen:
                raise RuntimeError(f"duplicate source episode key: {key}")
            seen.add(key)
            row = gt.get(key)
            if row is None:
                raise RuntimeError(f"episode absent from visualization manifest: {key}")
            division = _tier(row.get("setup_hardness_pct"))
            counts[division] += 1
            output[xml].append({
                "object_id": record["object_id"],
                "region": record.get("region"),
                "division": division,
                "division_source": "exhaustive_gt_setup_density",
                "setup_hardness_pct": row.get("setup_hardness_pct"),
                "n_setups_gt": row.get("n_setups"),
            })

    if len(seen) != len(gt_rows):
        raise RuntimeError(f"source has {len(seen)} episodes but manifest has {len(gt_rows)}")
    expected = sum(len(records) for records in source.values())
    if sum(counts.values()) != expected:
        raise RuntimeError(f"expected {expected} episodes, got {sum(counts.values())}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as stream:
        json.dump(output, stream, separators=(",", ":"))
    print(json.dumps(dict(sorted(counts.items())), indent=2))


if __name__ == "__main__":
    main()
