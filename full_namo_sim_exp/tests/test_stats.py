from __future__ import annotations

from pathlib import Path

import pytest

from full_namo_sim_exp.experiment_io import load_experiment
from full_namo_sim_exp.stats import compute_statistics

from .conftest import write_aggregate


def test_statistics_report_five_seed_rates_and_paired_bootstrap(
    experiment_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)
    write_aggregate(experiment.aggregate_root(experiment.model), {0, 1, 2})
    seed_successes = ({0}, {0, 1}, {0, 1}, {0, 1}, {0, 1, 2})
    for arm, solved in zip(experiment.random_arms, seed_successes):
        write_aggregate(experiment.aggregate_root(arm), set(solved), call_offset=arm.seed - 100)

    summary = compute_statistics(experiment)

    assert summary["model"]["fraction"] == "3/4"
    assert summary["model"]["success_rate_percent"] == 75.0
    assert summary["random"]["pooled_fraction"] == "10/20"
    assert summary["random"]["mean_success_rate_percent"] == 50.0
    assert summary["random"]["sample_sd_percentage_points"] == pytest.approx(17.6776695297)
    assert [seed["success_rate_percent"] for seed in summary["random"]["seeds"]] == [
        25.0,
        50.0,
        50.0,
        50.0,
        75.0,
    ]
    assert summary["paired_final_success"]["difference_percentage_points"] == 25.0
    assert summary["paired_final_success"]["bootstrap_replicates"] == 200
    assert len(summary["paired_final_success"]["cluster_bootstrap_95_ci_points"]) == 2
    assert len(summary["paired_mcnemar_by_random_seed"]) == 5


def test_statistics_keep_unsolved_scenes_in_fixed_budget_denominator(
    experiment_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)
    write_aggregate(experiment.aggregate_root(experiment.model), {0, 1, 2})
    for arm in experiment.random_arms:
        write_aggregate(experiment.aggregate_root(arm), {0, 1})

    summary = compute_statistics(experiment)

    assert summary["at_simulator_call_cutoffs"]["2"]["model_successes"] == 1
    assert summary["at_simulator_call_cutoffs"]["2"]["model_rate_percent"] == 25.0
