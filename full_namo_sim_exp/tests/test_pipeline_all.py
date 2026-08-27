from __future__ import annotations

import json
from pathlib import Path

from full_namo_sim_exp.experiment_io import Arm, Experiment, load_experiment
from full_namo_sim_exp.pipeline import main

from .conftest import SCENES, terminal_row, write_json, write_jsonl, write_run_config


def write_raw_arm(experiment: Experiment, arm: Arm, solved_indices: set[int]) -> None:
    for shard, indices in enumerate(((0, 1), (2, 3))):
        root = experiment.raw_output(arm, shard)
        rows = [
            terminal_row(
                SCENES[index],
                solved=index in solved_indices,
                calls=arm.seed % 10 + 2 * (index + 1),
                seconds=0.25 * (arm.seed % 10 + index + 1),
            )
            for index in indices
        ]
        write_jsonl(root / "solved.jsonl", [row for row in rows if row["solved"]])
        write_jsonl(root / "unsolved.jsonl", [row for row in rows if not row["solved"]])
        write_json(
            root / "summary.json",
            {
                "input_env_count": len(indices),
                "selected_env_count": len(indices),
                "selection_error_count": 0,
            },
        )
        write_run_config(root, experiment, arm)


def test_all_command_writes_validated_statistics_and_figure(
    experiment_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)
    assert main(["validate", "--experiment", str(experiment_path)]) == 0
    write_raw_arm(experiment, experiment.model, {0, 1, 2})
    seed_successes = ({0}, {0, 1}, {0, 1}, {0, 1}, {0, 1, 2})
    for arm, solved in zip(experiment.random_arms, seed_successes):
        write_raw_arm(experiment, arm, set(solved))

    status = main(["all", "--experiment", str(experiment_path)])

    assert status == 0
    stats_path = experiment.analysis_root / "full_namo_statistics.json"
    stats = json.loads(stats_path.read_text())
    assert stats["model"]["fraction"] == "3/4"
    assert stats["random"]["pooled_fraction"] == "10/20"
    caption = (experiment.analysis_root / "full_namo_caption_stats.txt").read_text()
    assert "Random 10/20" in caption
    assert "mean 50.00%" in caption
    assert "sample SD 17.68 pp" in caption
    assert "paired difference 25.00 pp" in caption
    assert (experiment.plot_root / "full_namo_success_vs_cost.pdf").stat().st_size > 1000
    assert (experiment.plot_root / "full_namo_success_vs_cost.png").stat().st_size > 1000
