from __future__ import annotations

from pathlib import Path
import subprocess

from full_namo_sim_exp.experiment_io import load_experiment
from full_namo_sim_exp import run_shard as shard_module
import pytest

from full_namo_sim_exp.run_shard import (
    build_runner_commands,
    ensure_final_shard_available,
    shard_scenes,
)


def option(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_shards_partition_the_population_without_prefiltering(experiment_path: Path) -> None:
    experiment = load_experiment(experiment_path)

    first = shard_scenes(experiment, 0)
    second = shard_scenes(experiment, 1)

    assert first + second == experiment.population.scene_ids


def test_runner_commands_interleave_all_six_arms_with_matched_protocol(
    experiment_path: Path,
    tmp_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)
    shard_manifest = tmp_path / "input.txt"
    shard_manifest.write_text("\n".join(shard_scenes(experiment, 1)) + "\n")

    commands = build_runner_commands(
        experiment,
        shard_index=1,
        shard_manifest=shard_manifest,
        python_executable="python3",
    )

    assert len(commands) == 6
    assert commands[0].arm.name == "random_s101"
    assert commands[-1].arm.name == "hy5u"
    assert all("--simulation-budget" in command.argv for command in commands)
    assert all("900" in command.argv for command in commands)
    assert [command.arm.prior for command in commands].count("uniform") == 5
    assert [command.arm.prior for command in commands].count("model") == 1
    assert {option(command.argv, "--seed") for command in commands} == {"42"}
    assert [option(command.argv, "--ordering-seed") for command in commands] == [
        "101",
        "102",
        "103",
        "104",
        "105",
        "42",
    ]


def test_existing_raw_output_makes_shard_immutable(
    experiment_path: Path,
    tmp_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)
    commands = build_runner_commands(
        experiment,
        shard_index=0,
        shard_manifest=tmp_path / "input.txt",
        python_executable="python3",
    )
    experiment.raw_shard_root(0).mkdir(parents=True)

    with pytest.raises(ValueError, match="refusing to overwrite raw shard output"):
        ensure_final_shard_available(experiment, 0)


def test_shard_is_published_only_after_all_arms_finish(
    experiment_path: Path,
    monkeypatch,
) -> None:
    experiment = load_experiment(experiment_path)
    monkeypatch.setattr(shard_module, "load_experiment", lambda path: experiment)
    monkeypatch.setattr(shard_module, "verify_frozen_experiment", lambda value: None)

    def complete_arm(argv, check):
        Path(option(tuple(argv), "--output-dir")).mkdir(parents=True)

    monkeypatch.setattr(shard_module.subprocess, "run", complete_arm)

    assert shard_module.main(
        ["--experiment", str(experiment_path), "--shard-index", "0"]
    ) == 0
    final = experiment.raw_shard_root(0)
    assert final.is_dir()
    assert {path.name for path in final.iterdir() if path.is_dir()} == {
        arm.name for arm in experiment.arms
    }


def test_interrupted_shard_leaves_retryable_staging_not_final_output(
    experiment_path: Path,
    monkeypatch,
) -> None:
    experiment = load_experiment(experiment_path)
    monkeypatch.setattr(shard_module, "load_experiment", lambda path: experiment)
    monkeypatch.setattr(shard_module, "verify_frozen_experiment", lambda value: None)
    attempts = 0

    def interrupt_second_arm(argv, check):
        nonlocal attempts
        attempts += 1
        Path(option(tuple(argv), "--output-dir")).mkdir(parents=True)
        if attempts == 2:
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(shard_module.subprocess, "run", interrupt_second_arm)

    with pytest.raises(subprocess.CalledProcessError):
        shard_module.main(
            ["--experiment", str(experiment_path), "--shard-index", "0"]
        )

    assert not experiment.raw_shard_root(0).exists()
    assert list((experiment.run_root / "staging").glob(".shard_0000_*"))
