from __future__ import annotations

from pathlib import Path

import pytest

from full_namo_sim_exp.experiment_io import load_experiment
from full_namo_sim_exp import provenance
from full_namo_sim_exp.provenance import freeze_experiment, verify_frozen_experiment


def test_frozen_provenance_detects_checkpoint_or_manifest_changes(
    experiment_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)
    lock = freeze_experiment(experiment)

    assert lock == experiment.run_root / "experiment.lock.json"
    verify_frozen_experiment(experiment)

    experiment.model.checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint SHA-256 changed"):
        verify_frozen_experiment(experiment)


def test_frozen_provenance_detects_experiment_config_changes(
    experiment_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)
    freeze_experiment(experiment)
    experiment_path.write_text(experiment_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="experiment config SHA-256 changed"):
        verify_frozen_experiment(experiment)


def test_freeze_refuses_to_replace_a_different_lock(experiment_path: Path) -> None:
    experiment = load_experiment(experiment_path)
    lock = freeze_experiment(experiment)
    lock.write_text('{"not": "the original lock"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to replace existing experiment lock"):
        freeze_experiment(experiment)


def test_frozen_provenance_hashes_scene_and_primitive_contents(
    experiment_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)
    freeze_experiment(experiment)
    scene = Path(experiment.population.scene_ids[0])
    scene.write_text(scene.read_text() + "<!-- changed -->\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scene XML contents changed"):
        verify_frozen_experiment(experiment)


def test_frozen_provenance_detects_primitive_profile_changes(
    experiment_path: Path,
) -> None:
    experiment = load_experiment(experiment_path)
    freeze_experiment(experiment)
    primitive = experiment.protocol.primitive_data_dir / "1x_car_d5_fixture.dat"
    primitive.write_bytes(b"changed primitive")

    with pytest.raises(ValueError, match="primitive profile contents changed"):
        verify_frozen_experiment(experiment)


def test_freeze_requires_a_clean_repository(
    experiment_path: Path,
    monkeypatch,
) -> None:
    experiment = load_experiment(experiment_path)
    monkeypatch.setattr(
        provenance,
        "_repository_state",
        lambda: ("test-commit", " M full_namo_sim_exp/plot.py\n"),
    )

    with pytest.raises(ValueError, match="repository working tree must be clean"):
        freeze_experiment(experiment)


def test_frozen_provenance_detects_runtime_changes(
    experiment_path: Path,
    monkeypatch,
) -> None:
    experiment = load_experiment(experiment_path)
    freeze_experiment(experiment)
    monkeypatch.setattr(
        provenance,
        "_runtime_state",
        lambda: {"namo_rl_sha256": "different-runtime", "sage_commit": "test-sage"},
    )

    with pytest.raises(ValueError, match="runtime fingerprint changed"):
        verify_frozen_experiment(experiment)
