#!/usr/bin/env python3
"""Cross-arm report for a showcase multi-hop pool, split by keyhole-1 difficulty metadata.

`compare_multihop_rankers.py` handles the paired two-arm statistics (McNemar, paired cost).
This script covers what that one structurally cannot: an N-arm cross-tab of solve counts and
solve@k on one pool, stratified by the `kh1_showcase_candidates.jsonl` metadata each scene
carries (`showcase_horizon`, `tier`).

Scenes are joined to their candidate row by `os.path.realpath`, never by basename. These pools
have far fewer unique basenames than scenes, so a basename join silently mislabels most of the
corpus — failure mode #5 in docs/pipeline/multi_episode_rooms.md.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

CUTS = [2, 5, 10, 30, 100, 300, 900]


def _canon(path: str) -> str:
    return os.path.realpath(path.replace("/scache/scratch/", "/scratch/", 1))


def _load_candidates(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load per-scene keyhole-1 labels.

    Accepts either `kh1_showcase_candidates.jsonl` (which carries `showcase_horizon`) or the
    full `kh1_scenes.jsonl` (which does not). When the field is absent it is derived the same
    way the showcase list defines it: a scene is `1push` when some single push opens keyhole 1,
    otherwise `2push`.
    """
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "showcase_horizon" not in row:
            row["showcase_horizon"] = "1push" if row.get("any_1push_solvable") else "2push"
        rows[_canon(row["xml_path"])] = row
    return rows


def _load_arm(root: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for solved, name in ((True, "solved.jsonl"), (False, "unsolved.jsonl")):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[_canon(row["xml_path"])] = {"solved": solved, "row": row}
    return rows


def _calls(item: Dict[str, Any]) -> Optional[int]:
    return item["row"].get("simulation_budget_used_total")


def _arm_block(arm: Dict[str, Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    solved_keys = [key for key in keys if arm[key]["solved"]]
    solved_calls = [_calls(arm[key]) for key in solved_keys]
    solved_calls = [value for value in solved_calls if value is not None]
    return {
        "evaluated": len(keys),
        "solved": len(solved_keys),
        "solve_rate": len(solved_keys) / len(keys) if keys else 0.0,
        "solve_at": {
            str(cut): sum(
                1
                for key in solved_keys
                if (_calls(arm[key]) is not None and _calls(arm[key]) <= cut)
            )
            for cut in CUTS
        },
        "median_calls_when_solved": statistics.median(solved_calls) if solved_calls else None,
        "failure_kinds": dict(
            sorted(
                Counter(
                    str(arm[key]["row"].get("failure_kind") or arm[key]["row"].get("outcome") or "unknown")
                    for key in keys
                    if not arm[key]["solved"]
                ).items()
            )
        ),
    }


def build(
    candidates: Dict[str, Dict[str, Any]],
    arms: Dict[str, Dict[str, Dict[str, Any]]],
    hop: int,
) -> Dict[str, Any]:
    populations = {name: set(arm) for name, arm in arms.items()}
    shared = set.intersection(*populations.values())
    keys = sorted(shared)
    joined = sum(1 for key in keys if key in candidates)

    strata: Dict[str, List[str]] = {"all": keys}
    for horizon in sorted({candidates[key]["showcase_horizon"] for key in keys if key in candidates}):
        strata[f"showcase_horizon={horizon}"] = [
            key for key in keys if key in candidates and candidates[key]["showcase_horizon"] == horizon
        ]
    for tier in sorted({candidates[key]["tier"] for key in keys if key in candidates}):
        strata[f"keyhole1_tier={tier}"] = [
            key for key in keys if key in candidates and candidates[key]["tier"] == tier
        ]

    return {
        "hop": hop,
        "join": {
            "arm_population_sizes": {name: len(pop) for name, pop in populations.items()},
            "shared_population": len(keys),
            "joined_to_candidates": joined,
            "unjoined": len(keys) - joined,
        },
        "budget_per_keyhole": sorted(
            {
                arm[key]["row"]["simulation_budget_limit_per_keyhole"]
                for arm in arms.values()
                for key in keys
                if "simulation_budget_limit_per_keyhole" in arm[key]["row"]
            }
        ),
        "strata": {
            label: {name: _arm_block(arm, stratum_keys) for name, arm in arms.items()}
            for label, stratum_keys in strata.items()
        },
        "stratum_sizes": {label: len(stratum_keys) for label, stratum_keys in strata.items()},
    }


def _markdown(result: Dict[str, Any], arm_order: List[str]) -> str:
    lines = [f"# Showcase {result['hop']}-hop: HY5U vs uniform random", ""]
    lines.append(
        f"Shared population {result['join']['shared_population']}, "
        f"joined to candidate metadata {result['join']['joined_to_candidates']}, "
        f"budget per keyhole {result['budget_per_keyhole']}."
    )
    lines.append("")
    for label, blocks in result["strata"].items():
        size = result["stratum_sizes"][label]
        lines += [f"## {label} (n={size})", "", "| arm | solved | rate | " + " | ".join(f"@{cut}" for cut in CUTS) + " | median calls |", "|---|---:|---:|" + "---:|" * (len(CUTS) + 1)]
        for name in arm_order:
            block = blocks[name]
            cells = " | ".join(str(block["solve_at"][str(cut)]) for cut in CUTS)
            lines.append(
                f"| {name} | {block['solved']} | {100.0 * block['solve_rate']:.2f}% | {cells} | {block['median_calls_when_solved']} |"
            )
        lines.append("")
        if label == "all":
            kinds = sorted({kind for block in blocks.values() for kind in block["failure_kinds"]})
            if kinds:
                lines += ["| arm | " + " | ".join(kinds) + " |", "|---|" + "---:|" * len(kinds)]
                for name in arm_order:
                    counts = blocks[name]["failure_kinds"]
                    lines.append(f"| {name} | " + " | ".join(str(counts.get(kind, 0)) for kind in kinds) + " |")
                lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--hop", type=int, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=AGGREGATE_DIR",
        help="Repeatable. First one listed is treated as the reference arm in the tables.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arm_order = [spec.split("=", 1)[0] for spec in args.arm]
    arms = {name: _load_arm(Path(spec.split("=", 1)[1])) for name, spec in zip(arm_order, args.arm)}
    result = build(_load_candidates(args.candidates), arms, args.hop)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = _markdown(result, arm_order)
    (args.output / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
