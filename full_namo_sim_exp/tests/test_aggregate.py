from __future__ import annotations

import json
from pathlib import Path

import pytest

from full_namo_sim_exp.aggregate import aggregate_arm
from full_namo_sim_exp.experiment_io import load_experiment

from .conftest import SCENES, terminal_row, write_json, write_jsonl, write_run_config


def write_raw_shards(experiment_path: Path) -> None:
    experiment = load_experiment(experiment_path)
    arm = experiment.model
    for shard, indices in enumerate(((0, 1), (2, 3))):
        shard_root = experiment.raw_output(arm, shard)
        rows = [
            terminal_row(
                SCENES[index],
                solved=index < 3,
                calls=2 * (index + 1),
                seconds=0.5 * (index + 1),
            )
            for index in indices
        ]
        write_jsonl(shard_root / "solved.jsonl", [row for row in rows if row["solved"]])
        write_jsonl(shard_root / "unsolved.jsonl", [row for row in rows if not row["solved"]])
        write_json(
            shard_root / "summary.json",
            {
                "input_env_count": 2,
                "selected_env_count": 2,
                "selection_error_count": 0,
            },
        )
        write_run_config(shard_root, experiment, arm)


def test_aggregate_arm_requires_and_writes_exact_frozen_population(
    experiment_path: Path,
) -> None:
    write_raw_shards(experiment_path)
    experiment = load_experiment(experiment_path)

    summary = aggregate_arm(experiment, experiment.model)

    assert summary["population_size"] == 4
    assert summary["solved_count"] == 3
    assert summary["unsolved_count"] == 1
    assert (experiment.aggregate_root(experiment.model) / "solved.jsonl").exists()


def test_aggregate_arm_rejects_infrastructure_failures(
    experiment_path: Path,
) -> None:
    write_raw_shards(experiment_path)
    experiment = load_experiment(experiment_path)
    bad_path = experiment.raw_output(experiment.model, 1) / "unsolved.jsonl"
    bad = json.loads(bad_path.read_text().splitlines()[0])
    bad["failure_kind"] = "runner_exception"
    bad_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runner_exception"):
        aggregate_arm(experiment, experiment.model)


def test_aggregate_arm_rejects_missing_terminal_time(
    experiment_path: Path,
) -> None:
    write_raw_shards(experiment_path)
    experiment = load_experiment(experiment_path)
    path = experiment.raw_output(experiment.model, 0) / "solved.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].pop("search_time_ms")
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="search_time_ms"):
        aggregate_arm(experiment, experiment.model)


def test_aggregate_arm_rejects_protocol_mismatch(experiment_path: Path) -> None:
    write_raw_shards(experiment_path)
    experiment = load_experiment(experiment_path)
    path = experiment.raw_output(experiment.model, 1) / "run_config.json"
    raw = json.loads(path.read_text())
    raw["simulation_budget"] = 300
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="simulation_budget.*expected 900.*received 300"):
        aggregate_arm(experiment, experiment.model)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rollout_samples_per_state", 10),
        ("region_frontier_beam_width", 5),
        ("max_push_steps", 4),
    ],
)
def test_aggregate_arm_rejects_low_level_protocol_mismatch(
    experiment_path: Path,
    field: str,
    value: object,
) -> None:
    write_raw_shards(experiment_path)
    experiment = load_experiment(experiment_path)
    path = experiment.raw_output(experiment.model, 1) / "run_config.json"
    raw = json.loads(path.read_text())
    raw[field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        aggregate_arm(experiment, experiment.model)


def test_aggregate_arm_rejects_scenes_in_the_wrong_shard(experiment_path: Path) -> None:
    write_raw_shards(experiment_path)
    experiment = load_experiment(experiment_path)
    first = experiment.raw_output(experiment.model, 0) / "solved.jsonl"
    second = experiment.raw_output(experiment.model, 1) / "solved.jsonl"
    first_rows = [json.loads(line) for line in first.read_text().splitlines()]
    second_rows = [json.loads(line) for line in second.read_text().splitlines()]
    first_rows[0]["xml_path"], second_rows[0]["xml_path"] = (
        second_rows[0]["xml_path"],
        first_rows[0]["xml_path"],
    )
    write_jsonl(first, first_rows)
    write_jsonl(second, second_rows)

    with pytest.raises(ValueError, match="shard_0000.*scene mismatch"):
        aggregate_arm(experiment, experiment.model)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timing_scope", "environment_plus_search", "timing_scope"),
        ("path_length_n", 3, "path_length_n"),
    ],
)
def test_aggregate_arm_rejects_wrong_terminal_protocol(
    experiment_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    write_raw_shards(experiment_path)
    experiment = load_experiment(experiment_path)
    path = experiment.raw_output(experiment.model, 0) / "solved.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match=message):
        aggregate_arm(experiment, experiment.model)
