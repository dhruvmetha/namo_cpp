#!/usr/bin/env python3
"""Aggregate a fixed-template keyhole composition run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


STATIC_REJECTIONS = {
    "static_junk",
    "static_error",
    "goal_not_in_free_space",
    "no_component_path",
    "wrong_hop_count",
    "k1_boundary_has_no_blocker",
    "k1_not_reachable",
    "k1_no_push_edges",
    "wrong_boundary_count",
    "wrong_blocker_order",
}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _aggregate(rows: list[dict]) -> dict:
    attempted = sum(row["attempted"] for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    rejections: Counter[str] = Counter()
    for row in rows:
        rejections.update(row.get("rejections", {}))
    static_rejected = sum(rejections[key] for key in STATIC_REJECTIONS)
    static_passed = attempted - static_rejected
    post_static_rejected = static_passed - accepted
    return {
        "tasks": len(rows),
        "attempted": attempted,
        "static_passed": static_passed,
        "static_pass_rate": _rate(static_passed, attempted),
        "post_static_rejected": post_static_rejected,
        "replay_failed": post_static_rejected,
        "accepted": accepted,
        "accepted_per_attempt": _rate(accepted, attempted),
        "accepted_per_static_pass": _rate(accepted, static_passed),
        "rejections": dict(sorted(rejections.items())),
    }


def summarize(root: Path) -> dict:
    summaries = []
    by_pair: dict[str, list[dict]] = defaultdict(list)
    by_template: dict[str, list[dict]] = defaultdict(list)
    manifest_rows = []
    for path in sorted(root.glob("*/*/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        pair = "_".join(row["tiers"])
        template = row["template"].replace("/", "_")
        row["summary_path"] = str(path.resolve())
        summaries.append(row)
        by_pair[pair].append(row)
        by_template[template].append(row)
        manifest = path.with_name("manifest.jsonl")
        lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != row["accepted"]:
            raise RuntimeError(f"{manifest}: {len(lines)} rows != accepted={row['accepted']}")
        manifest_rows.extend(json.loads(line) for line in lines)

    sequence_keys = [
        tuple(tuple(donor["episode_key"]) for donor in row["donors"])
        for row in manifest_rows
    ]
    duplicates = len(sequence_keys) - len(set(sequence_keys))
    return {
        "root": str(root.resolve()),
        "summary_files": len(summaries),
        "manifest_rows": len(manifest_rows),
        "duplicate_ordered_donor_sequences": duplicates,
        "overall": _aggregate(summaries),
        "by_pair": {key: _aggregate(rows) for key, rows in sorted(by_pair.items())},
        "by_template": {key: _aggregate(rows) for key, rows in sorted(by_template.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    payload = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
