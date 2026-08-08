#!/usr/bin/env python3
"""Aggregate one canonical 1push + pure-2push best-first arm by difficulty."""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from namo.paths import resolve
from namo import eval_sets
from eval_common import bin_of

ONEPUSH_CUTS = (1, 2, 5, 10, 30, 100, 300, 900)
TWOPUSH_CUTS = (1, 2, 5, 10, 30, 100, 300, 900)
# Wall-clock budgets in SECONDS, for the success-vs-time axis. Only emitted when every row carries
# t_wall (runs from the instrumented search); pre-instrumentation artifacts aggregate exactly as before.
# Times are comparable ONLY within one pinned-hardware campaign -- never pool across boxes.
TIME_CUTS = (0.5, 1, 2, 5, 10, 30, 60, 120, 300)
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


def _row_timing(row):
    """Per-episode wall-clock, or None for rows written before the search was instrumented."""
    if row.get("t_wall") is None:
        return None
    return {k: float(row[k]) for k in ("t_wall", "t_sim", "t_score") if row.get(k) is not None}


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
    tiers = TIERS[:-1] + (("unknown",) if grouped["unknown"] else ()) + ("all",)
    for tier in tiers:
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
        # Wall-clock twin of the block above, on the SAME episodes -- emitted only when the whole tier
        # carries timing, so a partially-instrumented mix can never be silently averaged.
        if all(row.get("timing") for row in tier_rows):
            wall = np.asarray([row["timing"]["t_wall"] for row in tier_rows], dtype=np.float64)
            solved_wall = np.asarray(
                [row["timing"]["t_wall"] for row in tier_rows if row["solved"]], dtype=np.float64
            )
            sim_t = np.asarray([row["timing"]["t_sim"] for row in tier_rows], dtype=np.float64)
            score_t = np.asarray([row["timing"]["t_score"] for row in tier_rows], dtype=np.float64)
            result[tier].update(
                {
                    **{
                        f"solve@{cut}s": round(
                            100.0 * np.count_nonzero(solved_wall <= cut) / max(1, len(tier_rows)), 1
                        )
                        for cut in TIME_CUTS
                    },
                    "avg_wall_all": round(float(wall.mean()), 2),
                    "avg_wall_to_solve": round(float(solved_wall.mean()), 2) if solved_wall.size else None,
                    "median_wall_to_solve": (
                        round(float(np.median(solved_wall)), 2) if solved_wall.size else None
                    ),
                    # where the time actually went: sim vs ranking overhead (score_frac ~0 for random)
                    "sim_frac": round(float(sim_t.sum() / max(wall.sum(), 1e-9)), 3),
                    "score_frac": round(float(score_t.sum() / max(wall.sum(), 1e-9)), 3),
                    "sec_per_sim": round(float(sim_t.sum() / max(sims.sum(), 1e-9)), 4),
                }
            )
    return result


def load_tiered_rows(onepush_dir, twopush_dir, onepush_key, divisions_path, expect_onepush, expect_twopush):
    """Load, validate, and attach the canonical fixed difficulty tier to each episode row."""
    onepush = _read_jsonl(onepush_dir)
    twopush = _read_jsonl(twopush_dir)
    onepush_keys = [(_canonical_xml(row.get("xml_full", row["xml"])), row["object_id"]) for row in onepush]
    twopush_keys = [
        (_canonical_xml(row["xml"]), row["object_id"], row.get("region"))
        for row in twopush
    ]
    if len(set(onepush_keys)) != len(onepush_keys):
        raise RuntimeError("duplicate 1push episode rows")
    if len(set(twopush_keys)) != len(twopush_keys):
        raise RuntimeError("duplicate 2push episode rows")

    onepush_divisions = _onepush_divisions(onepush_key)
    onepush_rows = []
    for row in onepush:
        key = (_canonical_xml(row.get("xml_full", row["xml"])), row["object_id"])
        division = onepush_divisions.get(key)
        if division is None:
            continue
        onepush_rows.append(
            {"division": _normalize_tier(division), "solved": bool(row["solved"]), "sims": _row_sims(row),
             "timing": _row_timing(row)}
        )

    divisions = _load_divisions(divisions_path)
    twopush_rows = []
    for row in twopush:
        key = (_canonical_xml(row["xml"]), row["object_id"], row.get("region"))
        division = divisions.get(key)
        if division is None:
            continue
        twopush_rows.append(
            {"division": _normalize_tier(division), "solved": bool(row["solved"]), "sims": _row_sims(row),
             "timing": _row_timing(row)}
        )

    if len(onepush_rows) != expect_onepush:
        raise RuntimeError(f"matched 1push rows {len(onepush_rows)} != expected {expect_onepush}")
    if len(twopush_rows) != expect_twopush:
        raise RuntimeError(f"matched 2push rows {len(twopush_rows)} != expected {expect_twopush}")

    onepush_config = _search_config(onepush)
    twopush_config = _search_config(twopush)
    if onepush_config != twopush_config:
        raise RuntimeError(f"1push/2push search config mismatch: {onepush_config} != {twopush_config}")
    return {"1push": onepush_rows, "2push": twopush_rows}, onepush_config


def load_twopush_rows(twopush_dir, divisions_path, expect_twopush):
    """Load and validate a standalone canonical 2push search arm."""
    twopush = _read_jsonl(twopush_dir)
    keys = [(_canonical_xml(row["xml"]), row["object_id"], row.get("region")) for row in twopush]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate 2push episode rows")
    divisions = _load_divisions(divisions_path)
    rows = []
    for row, key in zip(twopush, keys):
        division = divisions.get(key)
        if division is not None:
            rows.append({"division": _normalize_tier(division), "solved": bool(row["solved"]),
                         "timing": _row_timing(row),
                         "sims": _row_sims(row)})
    if len(rows) != expect_twopush:
        raise RuntimeError(f"matched 2push rows {len(rows)} != expected {expect_twopush}")
    return rows, _search_config(twopush)


def load_onepush_rows(onepush_dir, onepush_key, expect_onepush):
    """Load and validate a standalone canonical 1push search arm."""
    onepush = _read_jsonl(onepush_dir)
    keys = [(_canonical_xml(row.get("xml_full", row["xml"])), row["object_id"]) for row in onepush]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate 1push episode rows")
    divisions = _onepush_divisions(onepush_key)
    rows = []
    for row, key in zip(onepush, keys):
        division = divisions.get(key)
        if division is not None:
            rows.append({"division": _normalize_tier(division), "solved": bool(row["solved"]),
                         "timing": _row_timing(row),
                         "sims": _row_sims(row)})
    if len(rows) != expect_onepush:
        raise RuntimeError(f"matched 1push rows {len(rows)} != expected {expect_onepush}")
    return rows, _search_config(onepush)


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
    horizon = parser.add_mutually_exclusive_group()
    horizon.add_argument("--onepush-only", action="store_true",
                         help="aggregate only the canonical 1push arm")
    horizon.add_argument("--twopush-only", action="store_true",
                         help="aggregate only the canonical 2push arm")
    args = parser.parse_args()

    if args.onepush_only:
        onepush_rows, search_config = load_onepush_rows(
            args.onepush_dir or Path(args.eval_root) / "1push", args.onepush_key, args.expect_1push)
        tiered = {"1push": onepush_rows}
    elif args.twopush_only:
        twopush_rows, search_config = load_twopush_rows(
            args.twopush_dir or Path(args.eval_root) / "2push", args.divisions, args.expect_2push)
        tiered = {"2push": twopush_rows}
    else:
        tiered, search_config = load_tiered_rows(
            args.onepush_dir or Path(args.eval_root) / "1push",
            args.twopush_dir or Path(args.eval_root) / "2push",
            args.onepush_key,
            args.divisions,
            args.expect_1push,
            args.expect_2push,
        )
    if args.require_hmax is not None and search_config.get("hmax") != args.require_hmax:
        raise RuntimeError(f"hmax={search_config.get('hmax')} != required {args.require_hmax}")
    if args.require_dedupe_noop and not search_config.get("dedupe_noop"):
        raise RuntimeError("no-op dedupe was not enabled")
    if args.require_prune_jam_depth and not search_config.get("prune_jam_depth"):
        raise RuntimeError("jam-depth pruning was not enabled")

    report = {
        "eval_root": str(Path(args.eval_root).resolve()),
        "search": search_config,
        "onepush_difficulty": {"hard": "solve_rate < 0.05", "medium": "0.05 <= solve_rate < 0.30",
                               "easy": "solve_rate >= 0.30"},
        "twopush_divisions": str(Path(args.divisions).resolve()),
    }
    if not args.onepush_only:
        report["2push"] = _summarize(tiered["2push"], TWOPUSH_CUTS)
    if not args.twopush_only:
        report["1push"] = _summarize(tiered["1push"], ONEPUSH_CUTS)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
