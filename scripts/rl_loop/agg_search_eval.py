#!/usr/bin/env python3
"""Aggregate one canonical 1push + pure-2push best-first arm by difficulty."""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from namo.paths import DATASETS, resolve
from namo import eval_sets
from eval_common import bin_of

ONEPUSH_CUTS = (1, 2, 5, 10, 30, 100, 300, 900)
TWOPUSH_CUTS = (1, 2, 5, 10, 30, 100, 300, 900)
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


def _row_sims(row):
    value = row.get("n_sim", row.get("sims"))
    if value is None:
        raise RuntimeError(f"row has no n_sim/sims: {row}")
    return int(value)


def _search_config(rows):
    configs = {json.dumps(row.get("search"), sort_keys=True) for row in rows}
    if len(configs) != 1:
        raise RuntimeError(f"mixed search configs across rows: {len(configs)}")
    config = json.loads(next(iter(configs)))
    if config is None:
        raise RuntimeError("leaf rows do not record search config")
    return config


def _onepush_divisions(path):
    raw = json.load(open(path))
    rows = [
        {"xml": xml, "object_id": row["object_id"], "division": bin_of(row["solve_rate"])}
        for xml, records in raw.items()
        for row in records
    ]
    lookup = {
        (_canonical_xml(row["xml"]), row["object_id"]): row["division"]
        for row in rows
    }
    if len(lookup) != len(rows):
        raise RuntimeError("duplicate canonical 1push (xml, object_id) keys")
    return lookup


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
        default=str(eval_sets.DIVISIONS),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--onepush-key", default=str(eval_sets.ONEPUSH))
    parser.add_argument("--expect-1push", type=int, default=eval_sets.EXPECTED["onepush_manifest_episodes"])
    parser.add_argument("--expect-2push", type=int, default=eval_sets.EXPECTED["pure2push_manifest_episodes"])
    parser.add_argument("--require-hmax", type=int)
    parser.add_argument("--require-dedupe-noop", action="store_true")
    parser.add_argument("--require-prune-jam-depth", action="store_true")
    args = parser.parse_args()

    onepush = _read_jsonl(args.onepush_dir or Path(args.eval_root) / "1push")
    twopush = _read_jsonl(args.twopush_dir or Path(args.eval_root) / "2push")
    if len(onepush) != args.expect_1push:
        raise RuntimeError(f"1push rows {len(onepush)} != expected {args.expect_1push}")
    if len(twopush) != args.expect_2push:
        raise RuntimeError(f"2push rows {len(twopush)} != expected {args.expect_2push}")

    onepush_keys = [(_canonical_xml(row.get("xml_full", row["xml"])), row["object_id"]) for row in onepush]
    twopush_keys = [(row["xml"], row["object_id"], row.get("region")) for row in twopush]
    if len(set(onepush_keys)) != len(onepush_keys):
        raise RuntimeError("duplicate 1push episode rows")
    if len(set(twopush_keys)) != len(twopush_keys):
        raise RuntimeError("duplicate 2push episode rows")

    onepush_divisions = _onepush_divisions(args.onepush_key)
    onepush_rows = []
    for row in onepush:
        key = (_canonical_xml(row.get("xml_full", row["xml"])), row["object_id"])
        division = onepush_divisions.get(key)
        if division is None:
            raise RuntimeError(f"unmatched 1push episode: {key}")
        onepush_rows.append(
            {"division": _normalize_tier(division), "solved": bool(row["solved"]), "sims": _row_sims(row)}
        )

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
                "sims": _row_sims(row),
            }
        )

    onepush_config = _search_config(onepush)
    twopush_config = _search_config(twopush)
    if onepush_config != twopush_config:
        raise RuntimeError(f"1push/2push search config mismatch: {onepush_config} != {twopush_config}")
    if args.require_hmax is not None and onepush_config.get("hmax") != args.require_hmax:
        raise RuntimeError(f"hmax={onepush_config.get('hmax')} != required {args.require_hmax}")
    if args.require_dedupe_noop and not onepush_config.get("dedupe_noop"):
        raise RuntimeError("no-op dedupe was not enabled")
    if args.require_prune_jam_depth and not onepush_config.get("prune_jam_depth"):
        raise RuntimeError("jam-depth pruning was not enabled")

    report = {
        "eval_root": str(Path(args.eval_root).resolve()),
        "search": onepush_config,
        "onepush_difficulty": {"hard": "solve_rate < 0.05", "medium": "0.05 <= solve_rate < 0.30",
                               "easy": "solve_rate >= 0.30"},
        "1push": _summarize(onepush_rows, ONEPUSH_CUTS),
        "2push": _summarize(twopush_rows, TWOPUSH_CUTS),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
