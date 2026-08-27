from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from full_namo_sim_exp.experiment_io import Experiment
from full_namo_sim_exp.results import Outcome, load_all_results


def _mcnemar_exact(model_only: int, random_only: int) -> float:
    discordant = model_only + random_only
    if discordant == 0:
        return 1.0
    lower = min(model_only, random_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _cluster_bootstrap_ci(
    differences: np.ndarray,
    cluster_ids: tuple[str, ...],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    cluster_indices: dict[str, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(cluster_ids):
        cluster_indices[cluster_id].append(index)
    groups = tuple(np.asarray(indices, dtype=int) for indices in cluster_indices.values())
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = rng.integers(0, len(groups), size=len(groups))
        values = np.concatenate([differences[groups[index]] for index in selected])
        estimates[replicate] = 100.0 * float(values.mean())
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def _success_vector(
    scene_ids: tuple[str, ...],
    outcomes: dict[str, Outcome],
    *,
    metric: str | None = None,
    cutoff: float | None = None,
) -> np.ndarray:
    values = []
    for scene in scene_ids:
        outcome = outcomes[scene]
        success = outcome.solved
        if metric == "simulator_calls":
            success = success and outcome.simulator_calls <= cutoff
        elif metric == "wall_time_seconds":
            success = success and outcome.wall_time_seconds <= cutoff
        values.append(float(success))
    return np.asarray(values, dtype=float)


def _cutoff_rows(
    experiment: Experiment,
    all_results: dict[str, dict[str, Outcome]],
    cutoffs: Iterable[float],
    metric: str,
) -> dict[str, dict[str, float | int]]:
    scene_ids = experiment.population.scene_ids
    model = all_results[experiment.model.name]
    random = [all_results[arm.name] for arm in experiment.random_arms]
    rows: dict[str, dict[str, float | int]] = {}
    for cutoff in cutoffs:
        model_values = _success_vector(scene_ids, model, metric=metric, cutoff=cutoff)
        random_values = np.stack(
            [_success_vector(scene_ids, run, metric=metric, cutoff=cutoff) for run in random]
        )
        model_successes = int(model_values.sum())
        pooled_random = int(random_values.sum())
        key = str(int(cutoff)) if float(cutoff).is_integer() else str(cutoff)
        rows[key] = {
            "model_successes": model_successes,
            "model_rate_percent": 100.0 * float(model_values.mean()),
            "random_pooled_successes": pooled_random,
            "random_mean_rate_percent": 100.0 * float(random_values.mean()),
            "paired_difference_percentage_points": 100.0
            * float((model_values - random_values.mean(axis=0)).mean()),
        }
    return rows


def compute_statistics(experiment: Experiment) -> dict[str, object]:
    all_results = load_all_results(experiment)
    scene_ids = experiment.population.scene_ids
    n = len(scene_ids)
    model_values = _success_vector(scene_ids, all_results[experiment.model.name])
    random_values = np.stack(
        [_success_vector(scene_ids, all_results[arm.name]) for arm in experiment.random_arms]
    )
    model_successes = int(model_values.sum())
    random_successes = random_values.sum(axis=1).astype(int)
    random_rates = 100.0 * random_successes / n
    paired_differences = model_values - random_values.mean(axis=0)
    ci = _cluster_bootstrap_ci(
        paired_differences,
        experiment.population.cluster_ids,
        replicates=experiment.analysis.bootstrap_replicates,
        seed=experiment.analysis.bootstrap_seed,
    )

    mcnemar_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for arm, seed_values in zip(experiment.random_arms, random_values):
        model_only = int(np.sum((model_values == 1) & (seed_values == 0)))
        random_only = int(np.sum((model_values == 0) & (seed_values == 1)))
        p_value = _mcnemar_exact(model_only, random_only)
        raw_p_values.append(p_value)
        mcnemar_rows.append(
            {
                "seed": arm.seed,
                "model_only": model_only,
                "random_only": random_only,
                "exact_p": p_value,
            }
        )
    for row, adjusted in zip(mcnemar_rows, _holm_adjust(raw_p_values)):
        row["holm_adjusted_p"] = adjusted

    pooled_successes = int(random_successes.sum())
    summary: dict[str, object] = {
        "experiment": experiment.name,
        "population": experiment.population.name,
        "population_size": n,
        "model": {
            "name": experiment.model.name,
            "label": experiment.model.label,
            "successes": model_successes,
            "total": n,
            "fraction": f"{model_successes}/{n}",
            "success_rate_percent": 100.0 * model_successes / n,
        },
        "random": {
            "pooled_successes": pooled_successes,
            "pooled_total": 5 * n,
            "pooled_fraction": f"{pooled_successes}/{5 * n}",
            "mean_success_rate_percent": float(random_rates.mean()),
            "sample_sd_percentage_points": statistics.stdev(random_rates.tolist()),
            "seeds": [
                {
                    "seed": arm.seed,
                    "successes": int(successes),
                    "total": n,
                    "success_rate_percent": float(rate),
                }
                for arm, successes, rate in zip(
                    experiment.random_arms, random_successes, random_rates
                )
            ],
        },
        "paired_final_success": {
            "estimand": "model minus the per-scene mean of five Random seeds",
            "difference_percentage_points": 100.0 * float(paired_differences.mean()),
            "cluster_bootstrap_95_ci_points": [ci[0], ci[1]],
            "bootstrap_unit": "population cluster_id",
            "bootstrap_replicates": experiment.analysis.bootstrap_replicates,
            "bootstrap_seed": experiment.analysis.bootstrap_seed,
        },
        "paired_mcnemar_by_random_seed": mcnemar_rows,
        "at_simulator_call_cutoffs": _cutoff_rows(
            experiment,
            all_results,
            experiment.analysis.simulator_call_cutoffs,
            "simulator_calls",
        ),
        "at_wall_time_cutoffs_seconds": _cutoff_rows(
            experiment,
            all_results,
            experiment.analysis.wall_time_cutoffs_seconds,
            "wall_time_seconds",
        ),
        "independence_note": (
            "The pooled Random fraction is descriptive. Its 5N rows are repeated trials "
            "on N scenes and are not treated as 5N independent test examples."
        ),
    }
    return summary


def write_statistics(experiment: Experiment) -> Path:
    summary = compute_statistics(experiment)
    experiment.analysis_root.mkdir(parents=True, exist_ok=True)
    path = experiment.analysis_root / "full_namo_statistics.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model = summary["model"]
    random = summary["random"]
    paired = summary["paired_final_success"]
    ci_low, ci_high = paired["cluster_bootstrap_95_ci_points"]
    caption = (
        f"{model['label']} {model['fraction']} ({model['success_rate_percent']:.2f}%); "
        f"Random {random['pooled_fraction']} "
        f"(five-seed mean {random['mean_success_rate_percent']:.2f}%, "
        f"sample SD {random['sample_sd_percentage_points']:.2f} pp); "
        f"paired difference {paired['difference_percentage_points']:.2f} pp "
        f"(cluster-bootstrap 95% CI [{ci_low:.2f}, {ci_high:.2f}] pp).\n"
    )
    (experiment.analysis_root / "full_namo_caption_stats.txt").write_text(
        caption,
        encoding="utf-8",
    )
    return path
