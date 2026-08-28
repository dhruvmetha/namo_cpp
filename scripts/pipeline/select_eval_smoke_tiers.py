#!/usr/bin/env python3
"""Select a reproducible, episode-level difficulty-balanced eval smoke key."""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_common import bin_of
from namo.paths import resolve


TIERS = ("easy", "medium", "hard")


def canonical_xml(path):
    return str(resolve(path))


def episode_key(xml, record):
    return canonical_xml(xml), record["object_id"], record.get("region")


def load_divisions(path):
    raw = json.load(open(path))
    rows = {
        episode_key(xml, record): record["division"]
        for xml, records in raw.items()
        for record in records
    }
    n_records = sum(len(records) for records in raw.values())
    if len(rows) != n_records:
        raise RuntimeError("duplicate (xml, object_id, region) records in divisions")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--divisions", default="", help="required for 2-push; omit for 1-push solve-rate bins")
    parser.add_argument("--per-tier", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    raw = json.load(open(args.key))
    divisions = load_divisions(args.divisions) if args.divisions else None
    by_tier = defaultdict(list)
    seen = set()
    for xml, records in raw.items():
        for record in records:
            key = episode_key(xml, record)
            if key in seen:
                raise RuntimeError(f"duplicate episode in key: {key}")
            seen.add(key)
            if divisions is None:
                tier = bin_of(float(record["solve_rate"]))
                tier = "medium" if tier == "med" else tier
            else:
                tier = divisions.get(key)
                if tier is None:
                    raise RuntimeError(f"episode missing from divisions: {key}")
            if tier in TIERS:
                by_tier[tier].append((xml, record))

    rng = random.Random(args.seed)
    selected = []
    for tier in TIERS:
        population = sorted(
            by_tier[tier],
            key=lambda item: episode_key(item[0], item[1]),
        )
        if len(population) < args.per_tier:
            raise RuntimeError(f"{tier} has {len(population)} episodes, needs {args.per_tier}")
        selected.extend((tier, *item) for item in rng.sample(population, args.per_tier))

    output = defaultdict(list)
    for tier, xml, record in sorted(selected, key=lambda item: (item[1], item[2]["object_id"], item[2].get("region") or "")):
        output[xml].append({"object_id": record["object_id"], "region": record.get("region")})

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as stream:
        json.dump(dict(output), stream, indent=2)
        stream.write("\n")

    counts = Counter(tier for tier, _xml, _record in selected)
    n_records = sum(len(records) for records in output.values())
    print(json.dumps({
        "out": str(output_path),
        "seed": args.seed,
        "per_tier": args.per_tier,
        "tier_counts": {tier: counts[tier] for tier in TIERS},
        "episodes": n_records,
        "rooms": len(output),
        "multi_episode_rooms": sum(len(records) > 1 for records in output.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
