from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from full_namo_sim_exp.experiment_io import Experiment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, status


def _repository_state() -> tuple[str, str]:
    return _git_state(Path(__file__).resolve().parents[1])


def _distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "torch", "torchvision", "lightning", "pytorch-lightning"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _linked_mujoco(namo_rl_path: Path) -> Path:
    output = subprocess.run(
        ["ldd", str(namo_rl_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in output.splitlines():
        if "libmujoco" not in line or "=>" not in line:
            continue
        resolved = line.split("=>", 1)[1].split("(", 1)[0].strip()
        path = Path(resolved)
        if path.is_file():
            return path.resolve()
    raise ValueError(f"could not resolve linked MuJoCo library for {namo_rl_path}")


def _sage_repository_root() -> Path:
    configured = os.environ.get("SAGE_REPO")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_dir():
            return path
    namo_root = Path(__file__).resolve().parents[1]
    common_dir = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=namo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common_path = Path(common_dir)
    if not common_path.is_absolute():
        common_path = namo_root / common_path
    candidate = common_path.resolve().parent.parent / "sage_learning"
    if candidate.is_dir():
        return candidate
    raise ValueError(
        f"SAGE_REPO does not exist and no main-worktree sibling was found: {configured!r}"
    )


def _runtime_state() -> dict[str, object]:
    spec = importlib.util.find_spec("namo_rl")
    if spec is None or spec.origin is None:
        raise ValueError("could not locate the namo_rl extension used by this environment")
    namo_rl_path = Path(spec.origin).resolve()
    mujoco_path = _linked_mujoco(namo_rl_path)

    sage_root = _sage_repository_root()
    sage_commit, sage_status = _git_state(sage_root)
    if sage_status:
        raise ValueError("Sage repository working tree must be clean")

    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "python_distributions": _distribution_versions(),
        "namo_rl_path": str(namo_rl_path),
        "namo_rl_sha256": _sha256(namo_rl_path),
        "mujoco_library_path": str(mujoco_path),
        "mujoco_library_sha256": _sha256(mujoco_path),
        "sage_repository_path": str(sage_root),
        "sage_commit": sage_commit,
        "sage_status_porcelain": sage_status,
    }


def _file_records(paths: list[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in sorted(paths, key=lambda item: str(item.resolve()))
    ]


def _current_record(experiment: Experiment) -> dict[str, object]:
    commit, status = _repository_state()
    if status:
        raise ValueError(
            "repository working tree must be clean before freezing or using an experiment"
        )
    scene_files = [Path(scene) for scene in experiment.population.scene_ids]
    primitive_files = [
        path
        for path in experiment.protocol.primitive_data_dir.iterdir()
        if path.is_file() and path.name.startswith(experiment.protocol.primitive_prefix)
    ]
    return {
        "experiment": experiment.name,
        "population": experiment.population.name,
        "population_size": len(experiment.population.scene_ids),
        "experiment_config_path": str(experiment.source),
        "experiment_config_sha256": _sha256(experiment.source),
        "population_manifest_path": str(experiment.population_path),
        "population_manifest_sha256": _sha256(experiment.population_path),
        "checkpoint_path": str(experiment.model.checkpoint),
        "checkpoint_sha256": _sha256(experiment.model.checkpoint),
        "namo_config_path": str(experiment.protocol.config_file),
        "namo_config_sha256": _sha256(experiment.protocol.config_file),
        "scene_xml_files": _file_records(scene_files),
        "primitive_profile_files": _file_records(primitive_files),
        "repository_commit": commit,
        "repository_status_porcelain": status,
        "runtime": _runtime_state(),
        "arms": [
            {"name": arm.name, "prior": arm.prior, "seed": arm.seed}
            for arm in experiment.arms
        ],
    }


def freeze_experiment(experiment: Experiment) -> Path:
    experiment.validate_launch_inputs()
    experiment.run_root.mkdir(parents=True, exist_ok=True)
    path = experiment.run_root / "experiment.lock.json"
    current = _current_record(experiment)
    if path.exists():
        try:
            frozen = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"refusing to replace existing experiment lock: {path}") from exc
        if frozen != current:
            raise ValueError(f"refusing to replace existing experiment lock: {path}")
        return path
    path.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def verify_frozen_experiment(experiment: Experiment) -> None:
    path = experiment.run_root / "experiment.lock.json"
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"experiment is not frozen; run pipeline validate first: {path}") from exc
    current = _current_record(experiment)
    checks = (
        ("experiment_config_sha256", "experiment config SHA-256 changed"),
        ("population_manifest_sha256", "population manifest SHA-256 changed"),
        ("checkpoint_sha256", "checkpoint SHA-256 changed"),
        ("namo_config_sha256", "NAMO config SHA-256 changed"),
        ("scene_xml_files", "scene XML contents changed"),
        ("primitive_profile_files", "primitive profile contents changed"),
        ("repository_commit", "repository commit changed"),
        ("repository_status_porcelain", "repository working tree changed"),
        ("runtime", "runtime fingerprint changed"),
    )
    for field, message in checks:
        if frozen.get(field) != current.get(field):
            raise ValueError(message)
