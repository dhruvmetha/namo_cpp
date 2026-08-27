from __future__ import annotations

import json
from pathlib import Path

import pytest

from full_namo_sim_exp.experiment_io import load_experiment


def test_load_experiment_builds_one_model_and_five_random_arms(
    experiment_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)

    assert experiment.name == "full-namo-final"
    assert len(experiment.population.scenes) == 4
    assert experiment.population.cluster_ids == ("base_0", "base_0", "base_1", "base_1")
    assert [arm.name for arm in experiment.arms] == [
        "hy5u",
        "random_s101",
        "random_s102",
        "random_s103",
        "random_s104",
        "random_s105",
    ]
    assert [arm.prior for arm in experiment.arms] == [
        "model",
        "uniform",
        "uniform",
        "uniform",
        "uniform",
        "uniform",
    ]
    assert experiment.raw_shard_root(0) == experiment_path.parent / "run/raw/shard_0000"
    assert experiment.raw_output(experiment.arms[0], 0) == (
        experiment_path.parent / "run/raw/shard_0000/hy5u"
    )


def test_load_experiment_rejects_nonfive_or_duplicate_random_seeds(
    experiment_path: Path,
) -> None:
    raw = json.loads(experiment_path.read_text())
    raw["random_seeds"] = [1, 2, 3, 4, 4]
    experiment_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="five distinct Random seeds"):
        load_experiment(experiment_path)


def test_validate_launch_inputs_requires_checkpoint_and_population_files(
    experiment_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)
    experiment.model.checkpoint.unlink()

    with pytest.raises(ValueError, match="checkpoint does not exist"):
        experiment.validate_launch_inputs()
