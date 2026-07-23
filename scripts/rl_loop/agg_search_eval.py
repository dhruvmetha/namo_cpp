#!/usr/bin/env python3
"""Aggregate the canonical d20 1push and pure-2push search outputs by difficulty."""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from namo.paths import DATASETS, resolve

ONEPUSH_CUTS = (1, 5, 300)
TWOPUSH_CUTS = (2, 5, 10, 30, 900)
TIERS = ("easy", "medium", "hard", "all")


def _read_jsonl(directory):
    rows = []
    for path in sorted(glob.glob(str(Path(directory) / "shard_*.jsonl"))):
        with open(path) as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def _canonical_xml(path):
    return str(resolve(path))


def _load_divisions(path):
    raw = json.load(open(path))
    return {
        (_canonical_xml(xml), rec["object_id"], rec.get("region")): rec["division"]
        for xml, records in raw.items()
        for rec in records
    }


def _normalize_tier(tier):
    return "medium" if tier == "med" else tier


def _summarize(rows, cuts):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["division"]].append(row)
        grouped["all"].append(row)
    result = {}
    for tier in TIERS:
        tier_rows = grouped[tier]
        sims = np.asarray([row["sims"] for row in tier_rows], dtype=np.float64)
        solved_sims = np.asarray([row["sims"] for row in tier_rows if row["solved"]], dtype=np.float64)
        result[tier] = {
            "n": len(tier_rows),
            **{
                f"solve@{cut}": round(
                    100.0 * np.count_nonzero(solved_sims <= cut) / max(1, len(tier_rows)), 1
                )
                for cut in cuts
            },
            "avg_sims_all": round(float(sims.mean()), 1),
            "avg_sims_to_solve": round(float(solved_sims.mean()), 1) if solved_sims.size else None,
            "median_sims_to_solve": round(float(np.median(solved_sims)), 1) if solved_sims.size else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, help="directory containing 1push/ and 2push/")
    parser.add_argument("--onepush-dir", default=None)
    parser.add_argument("--twopush-dir", default=None)
    parser.add_argument(
        "--divisions",
        default=str(DATASETS / "namo_testset_v1/labels/pure2push_divisions.json"),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-1push", type=int, default=1323)
    parser.add_argument("--expect-2push", type=int, default=1018)
    args = parser.parse_args()

    onepush = _read_jsonl(args.onepush_dir or Path(args.eval_root) / "1push")
    twopush = _read_jsonl(args.twopush_dir or Path(args.eval_root) / "2push")
    if len(onepush) != args.expect_1push:
        raise RuntimeError(f"1push rows {len(onepush)} != expected {args.expect_1push}")
    if len(twopush) != args.expect_2push:
        raise RuntimeError(f"2push rows {len(twopush)} != expected {args.expect_2push}")

    onepush_keys = [(row.get("xml_full", row["xml"]), row["object_id"]) for row in onepush]
    twopush_keys = [(row["xml"], row["object_id"], row.get("region")) for row in twopush]
    if len(set(onepush_keys)) != len(onepush_keys):
        raise RuntimeError("duplicate 1push episode rows")
    if len(set(twopush_keys)) != len(twopush_keys):
        raise RuntimeError("duplicate 2push episode rows")

    onepush_rows = [
        {
            "division": _normalize_tier(row["tier"]),
            "solved": bool(row["solved"]),
            "sims": int(row["n_sim"]),
        }
        for row in onepush
    ]

    divisions = _load_divisions(args.divisions)
    twopush_rows = []
    for row in twopush:
        key = (_canonical_xml(row["xml"]), row["object_id"], row.get("region"))
        division = divisions.get(key)
        if division is None:
            raise RuntimeError(f"unmatched pure-2push episode: {key}")
        twopush_rows.append(
            {
                "division": _normalize_tier(division),
                "solved": bool(row["solved"]),
                "sims": int(row["sims"]),
            }
        )

    report = {
        "eval_root": str(Path(args.eval_root).resolve()),
        "1push": _summarize(onepush_rows, ONEPUSH_CUTS),
        "2push": _summarize(twopush_rows, TWOPUSH_CUTS),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
