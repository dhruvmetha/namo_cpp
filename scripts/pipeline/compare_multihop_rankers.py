#!/usr/bin/env python3
"""Paired comparison of two aggregated Full-NAMO ranker evaluations."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def _load_arm(root: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for solved, name in ((True, "solved.jsonl"), (False, "unsolved.jsonl")):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["xml_path"]] = {"solved": solved, "row": row}
    return rows


def _template_key(xml_path: str) -> str:
    parts = Path(xml_path).parts
    for index, part in enumerate(parts[:-1]):
        if part in {"set1", "set2"} and parts[index + 1].startswith("benchmark_"):
            return f"{part}/{parts[index + 1]}"
    return "unknown"


def _mcnemar_exact(model_only: int, random_only: int) -> float:
    discordant = model_only + random_only
    if discordant == 0:
        return 1.0
    lower = min(model_only, random_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def compare(model_root: Path, random_root: Path) -> Dict[str, Any]:
    model = _load_arm(model_root)
    random = _load_arm(random_root)
    if set(model) != set(random):
        raise ValueError("Model and random evaluations do not contain the same XML population")

    keys = sorted(model)
    both = [key for key in keys if model[key]["solved"] and random[key]["solved"]]
    model_only = [key for key in keys if model[key]["solved"] and not random[key]["solved"]]
    random_only = [key for key in keys if random[key]["solved"] and not model[key]["solved"]]
    neither = [key for key in keys if not model[key]["solved"] and not random[key]["solved"]]

    model_both_sims = [model[key]["row"]["simulation_budget_used_total"] for key in both]
    random_both_sims = [random[key]["row"]["simulation_budget_used_total"] for key in both]
    paired_deltas = [m - r for m, r in zip(model_both_sims, random_both_sims)]
    model_faster = sum(delta < 0 for delta in paired_deltas)
    ties = sum(delta == 0 for delta in paired_deltas)
    random_faster = sum(delta > 0 for delta in paired_deltas)

    cuts = [2, 5, 10, 30, 100, 300, 600, 900]

    def solve_at(arm: Dict[str, Dict[str, Any]], cut: int) -> int:
        return sum(
            item["solved"] and item["row"]["simulation_budget_used_total"] <= cut
            for item in arm.values()
        )

    templates = Counter(_template_key(key) for key in keys)
    by_template = {}
    for template in sorted(templates):
        template_keys = [key for key in keys if _template_key(key) == template]
        model_count = sum(model[key]["solved"] for key in template_keys)
        random_count = sum(random[key]["solved"] for key in template_keys)
        by_template[template] = {
            "count": len(template_keys),
            "hy5u_solved": model_count,
            "random_solved": random_count,
            "delta_solved": model_count - random_count,
        }

    model_solved_sims = [
        item["row"]["simulation_budget_used_total"] for item in model.values() if item["solved"]
    ]
    random_solved_sims = [
        item["row"]["simulation_budget_used_total"] for item in random.values() if item["solved"]
    ]
    n = len(keys)
    return {
        "population": n,
        "protocol": {
            "initial_path_length": 2,
            "hmax_per_keyhole": 2,
            "simulation_budget_per_keyhole": 300,
            "random_seed": 42,
        },
        "headline": {
            "hy5u_solved": len(both) + len(model_only),
            "random_solved": len(both) + len(random_only),
            "hy5u_solve_rate": (len(both) + len(model_only)) / n,
            "random_solve_rate": (len(both) + len(random_only)) / n,
            "solve_rate_delta_points": 100.0 * (len(model_only) - len(random_only)) / n,
        },
        "paired_outcomes": {
            "both_solved": len(both),
            "hy5u_only": len(model_only),
            "random_only": len(random_only),
            "neither_solved": len(neither),
            "mcnemar_exact_p": _mcnemar_exact(len(model_only), len(random_only)),
        },
        "both_solved_simulator_cost": {
            "count": len(both),
            "hy5u_median_sims": statistics.median(model_both_sims) if both else None,
            "random_median_sims": statistics.median(random_both_sims) if both else None,
            "median_paired_delta_hy5u_minus_random": statistics.median(paired_deltas) if both else None,
            "hy5u_faster": model_faster,
            "ties": ties,
            "random_faster": random_faster,
        },
        "solved_only_cost_unpaired": {
            "hy5u_median_sims": statistics.median(model_solved_sims),
            "random_median_sims": statistics.median(random_solved_sims),
        },
        "solve_at_total_simulator_calls": {
            str(cut): {
                "hy5u": solve_at(model, cut),
                "random": solve_at(random, cut),
            }
            for cut in cuts
        },
        "by_template": by_template,
    }


def _markdown(result: Dict[str, Any]) -> str:
    headline = result["headline"]
    paired = result["paired_outcomes"]
    cost = result["both_solved_simulator_cost"]
    lines = [
        "# HY5U vs random on exact-two-hop Full NAMO",
        "",
        "| arm | solved | solve rate |",
        "|---|---:|---:|",
        f"| HY5U | {headline['hy5u_solved']} | {100.0 * headline['hy5u_solve_rate']:.2f}% |",
        f"| random seed 42 | {headline['random_solved']} | {100.0 * headline['random_solve_rate']:.2f}% |",
        "",
        f"HY5U leads by {headline['solve_rate_delta_points']:.2f} percentage points.",
        "",
        f"Paired outcomes: both={paired['both_solved']}, HY5U-only={paired['hy5u_only']}, random-only={paired['random_only']}, neither={paired['neither_solved']}.",
        "",
        f"On the {cost['count']} scenes both solve: median calls HY5U={cost['hy5u_median_sims']}, random={cost['random_median_sims']}; HY5U faster/tied/slower={cost['hy5u_faster']}/{cost['ties']}/{cost['random_faster']}.",
        "",
        "| total simulator-call cutoff | HY5U solved | random solved |",
        "|---:|---:|---:|",
    ]
    for cut, row in result["solve_at_total_simulator_calls"].items():
        lines.append(f"| {cut} | {row['hy5u']} | {row['random']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hy5u", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.hy5u, args.random)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "comparison.md").write_text(_markdown(result), encoding="utf-8")
    print(_markdown(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
