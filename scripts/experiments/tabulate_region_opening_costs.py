#!/usr/bin/env python3
"""Build the manuscript median-cost table on the matched Cascadelake population.

Difficulty is defined separately for each horizon from Random's per-problem median simulator cost.
Cut points are chosen at the cumulative cost-group boundaries nearest one and two thirds, so equal
Random costs are never split across tiers. For each multi-seed method, costs are first medianed across
seeds within an episode, with censored trials ordered after every success, and then medianed across
episodes. Geometry is deterministic and therefore contributes one observation per episode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path


LEAVES = {"1push": "1push_hmax2", "2push": "2push"}
TIERS = ("all", "easy", "medium", "hard")
EXPECTED_COMMON = {"1push": 1310, "2push": 973}
PROTOCOL = {
    "hmax": 2,
    "sim_budget": 4000,
    "agg": "mean5",
    "combine": "q",
    "discount": "off",
    "raw": True,
    "dedupe_noop": True,
    "prune_jam_depth": True,
}


def episode_key(row: dict) -> tuple[str, str, str]:
    return row["xml"], row["object_id"], row.get("region", "goal")


def load_leaf(path: Path, *, require_warmup: bool = False) -> dict[tuple[str, str, str], dict]:
    rows: dict[tuple[str, str, str], dict] = {}
    shards = sorted(path.glob("shard_*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no shard JSONL files under {path}")
    for shard in shards:
        with shard.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = episode_key(row)
                if key in rows:
                    raise RuntimeError(f"duplicate episode {key} in {path}")
                for field, expected in PROTOCOL.items():
                    if row["search"].get(field) != expected:
                        raise RuntimeError(
                            f"protocol mismatch in {shard}: {field}={row['search'].get(field)!r}, "
                            f"expected {expected!r}"
                        )
                if require_warmup and row["search"].get("model_warmup_repeats") != 3:
                    raise RuntimeError(f"missing three-pass model warmup in {shard}")
                rows[key] = row
    return rows


def observation_median(observations: list[tuple[bool, float | None]]) -> tuple[bool, float | None]:
    ordered = sorted(
        observations,
        key=lambda item: (not item[0], item[1] if item[1] is not None else math.inf),
    )
    lower = ordered[(len(ordered) - 1) // 2]
    upper = ordered[len(ordered) // 2]
    if not (lower[0] and upper[0]):
        return False, None
    return True, (lower[1] + upper[1]) / 2.0


def per_problem(rows_by_seed: list[dict], keys: set[tuple[str, str, str]], metric: str):
    return {
        key: observation_median([(bool(rows[key]["solved"]), float(rows[key][metric])) for rows in rows_by_seed])
        for key in keys
    }


def tie_preserving_cuts(random_cost: dict) -> tuple[float, float]:
    counts = Counter(value for solved, value in random_cost.values() if solved)
    cumulative = 0
    boundaries: list[tuple[float, int]] = []
    for value, count in sorted(counts.items()):
        cumulative += count
        if cumulative < len(random_cost):
            boundaries.append((value, cumulative))
    cuts = []
    for fraction in (1 / 3, 2 / 3):
        target = fraction * len(random_cost)
        cuts.append(min(boundaries, key=lambda item: (abs(item[1] - target), item[0]))[0])
    if cuts[0] >= cuts[1]:
        raise RuntimeError(f"invalid difficulty cuts: {cuts}")
    return cuts[0], cuts[1]


def tier_keys(random_cost: dict, cuts: tuple[float, float]):
    grouped = {tier: set() for tier in TIERS}
    grouped["all"] = set(random_cost)
    low, high = cuts
    for key, (solved, value) in random_cost.items():
        if solved and value <= low:
            grouped["easy"].add(key)
        elif solved and value <= high:
            grouped["medium"].add(key)
        else:
            grouped["hard"].add(key)
    return grouped


def summarize(method_rows: list[dict], groups: dict[str, set], metric: str):
    problem_cost = per_problem(method_rows, groups["all"], metric)
    result = {}
    for tier in TIERS:
        solved, value = observation_median([problem_cost[key] for key in groups[tier]])
        result[tier] = value if solved else None
    return result


def latex_number(value: float | None, metric: str) -> str:
    if value is None:
        return r"$>4000$" if metric == "sims" else r"--"
    if metric == "time":
        return f"{value:.2f}"
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


def latex_rows(report: dict, metric: str) -> list[str]:
    order = [(horizon, tier) for horizon in ("1push", "2push") for tier in TIERS]
    best = {
        key: min(
            report[key[0]]["methods"][method][metric][key[1]]
            for method in ("HY5U", "Random", "Geometric")
            if report[key[0]]["methods"][method][metric][key[1]] is not None
        )
        for key in order
    }
    lines = []
    for method in ("HY5U", "Random", "Geometric"):
        cells = []
        for horizon, tier in order:
            value = report[horizon]["methods"][method][metric][tier]
            cell = latex_number(value, metric)
            if value is not None and math.isclose(value, best[(horizon, tier)]):
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{method:<9} & " + " & ".join(cells) + r" \\")
    lines.append("RL        & " + " & ".join(["--"] * len(order)) + r" \\")
    return lines


def main() -> None:
    scratch = Path(os.environ["NAMO_SCRATCH"])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hy5u-root", type=Path,
        default=scratch / "aquaman/round0/eval_walltime4k_warmup3",
    )
    parser.add_argument(
        "--random-root", type=Path,
        default=scratch / "aquaman/round0/eval_walltime4k",
    )
    parser.add_argument(
        "--geometric-root", type=Path,
        default=scratch / "aquaman/round0/eval_walltime4k/geometric_region_corrected_v1",
    )
    args = parser.parse_args()

    report = {}
    for horizon, leaf in LEAVES.items():
        methods = {
            "HY5U": [load_leaf(args.hy5u_root / f"HY5U_s{seed}" / leaf, require_warmup=True)
                      for seed in (1, 2, 3)],
            "Random": [load_leaf(args.random_root / f"rand_s{seed}" / leaf)
                       for seed in (7000, 8000, 9000)],
            "Geometric": [load_leaf(args.geometric_root / leaf)],
        }
        common = set.intersection(*(set(rows) for method in methods.values() for rows in method))
        if len(common) != EXPECTED_COMMON[horizon]:
            raise RuntimeError(
                f"{horizon}: common population {len(common)} != expected {EXPECTED_COMMON[horizon]}"
            )
        random_cost = per_problem(methods["Random"], common, "sims")
        cuts = tie_preserving_cuts(random_cost)
        groups = tier_keys(random_cost, cuts)
        report[horizon] = {
            "population": len(common),
            "random_median_sims_cuts": list(cuts),
            "tier_sizes": {tier: len(groups[tier]) for tier in TIERS},
            "methods": {
                method: {
                    "sims": summarize(rows, groups, "sims"),
                    "time": summarize(rows, groups, "t_wall"),
                }
                for method, rows in methods.items()
            },
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    print("\n% Simulator-push rows")
    print("\n".join(latex_rows(report, "sims")))
    print("\n% Wall-clock rows")
    print("\n".join(latex_rows(report, "time")))


if __name__ == "__main__":
    main()
