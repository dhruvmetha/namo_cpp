from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


RANDOM_RUN_COUNT = 5


def _path(raw: object, base: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a nonempty path string")
    value = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not value.is_absolute():
        value = base / value
    return value.resolve()


def _positive_int(raw: object, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return raw


def normalize_scene_id(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("scene xml_path must be a nonempty string")
    normalized = str(PurePosixPath(raw.replace("\\", "/")))
    if normalized == ".":
        raise ValueError("scene xml_path must identify an XML file")
    return normalized


@dataclass(frozen=True)
class Population:
    name: str
    scene_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]

    @property
    def scenes(self) -> tuple[str, ...]:
        return self.scene_ids


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    prior: str
    seed: int
    checkpoint: Path


@dataclass(frozen=True)
class Protocol:
    evaluation_seed: int
    path_length: int
    config_file: Path
    primitive_data_dir: Path
    primitive_prefix: str
    max_push_steps: int
    simulation_budget_per_keyhole: int
    region_max_chain_depth: int
    goals_per_region: int
    region_success_min_reachable: int
    n_shards: int
    slurm_partition: str
    hardware_constraint: str


@dataclass(frozen=True)
class AnalysisConfig:
    bootstrap_replicates: int
    bootstrap_seed: int
    simulator_call_cutoffs: tuple[int, ...]
    wall_time_cutoffs_seconds: tuple[float, ...]


@dataclass(frozen=True)
class Experiment:
    name: str
    source: Path
    population_path: Path
    population: Population
    run_root: Path
    model: Arm
    random_arms: tuple[Arm, ...]
    protocol: Protocol
    analysis: AnalysisConfig

    @property
    def arms(self) -> tuple[Arm, ...]:
        return (self.model, *self.random_arms)

    def raw_shard_root(self, shard_index: int) -> Path:
        return self.run_root / "raw" / f"shard_{shard_index:04d}"

    def raw_output(self, arm: Arm, shard_index: int) -> Path:
        return self.raw_shard_root(shard_index) / arm.name

    def aggregate_root(self, arm: Arm) -> Path:
        return self.run_root / "aggregate" / arm.name

    @property
    def analysis_root(self) -> Path:
        return self.run_root / "analysis"

    @property
    def plot_root(self) -> Path:
        return self.run_root / "plots"

    def validate_launch_inputs(self) -> None:
        required = (
            (self.population_path, "population manifest"),
            (self.model.checkpoint, "checkpoint"),
            (self.protocol.config_file, "NAMO config"),
            (self.protocol.primitive_data_dir, "primitive data directory"),
        )
        for path, label in required:
            if not path.exists():
                raise ValueError(f"{label} does not exist: {path}")
        missing_scenes = [
            scene for scene in self.population.scene_ids if not Path(scene).is_file()
        ]
        if missing_scenes:
            raise ValueError(f"scene XML does not exist: {missing_scenes[0]}")
        primitive_files = [
            path
            for path in self.protocol.primitive_data_dir.iterdir()
            if path.is_file() and path.name.startswith(self.protocol.primitive_prefix)
        ]
        if not primitive_files:
            raise ValueError(
                "primitive data directory has no files matching prefix "
                f"{self.protocol.primitive_prefix!r}"
            )
        if self.protocol.n_shards > len(self.population.scene_ids):
            raise ValueError("protocol.n_shards cannot exceed the frozen population size")


def _load_population(path: Path) -> Population:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"population manifest does not exist: {path}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("scenes"), list):
        raise ValueError("population manifest must contain a scenes list")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("population manifest must contain a nonempty name")
    scene_ids: list[str] = []
    cluster_ids: list[str] = []
    for index, item in enumerate(raw["scenes"]):
        if isinstance(item, str):
            scene = normalize_scene_id(item)
            cluster = scene
        elif isinstance(item, dict):
            scene = normalize_scene_id(item.get("xml_path"))
            cluster_raw = item.get("cluster_id", scene)
            if not isinstance(cluster_raw, str) or not cluster_raw.strip():
                raise ValueError(f"population scenes[{index}].cluster_id must be nonempty")
            cluster = cluster_raw
        else:
            raise ValueError(f"population scenes[{index}] must be a path or object")
        scene_path = Path(scene)
        if not scene_path.is_absolute():
            scene = normalize_scene_id(str((path.parent / scene_path).resolve()))
        scene_ids.append(scene)
        cluster_ids.append(cluster)
    if not scene_ids:
        raise ValueError("population manifest is empty")
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("population manifest contains duplicate scene IDs")
    return Population(name=name, scene_ids=tuple(scene_ids), cluster_ids=tuple(cluster_ids))


def _cutoffs(raw: object, label: str, integer: bool) -> tuple[Any, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"analysis.{label} must be a nonempty list")
    values: list[Any] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"analysis.{label} values must be positive")
        if integer and int(value) != value:
            raise ValueError(f"analysis.{label} values must be integers")
        values.append(int(value) if integer else float(value))
    if values != sorted(set(values)):
        raise ValueError(f"analysis.{label} must be sorted and duplicate-free")
    return tuple(values)


def load_experiment(path: Path) -> Experiment:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"experiment config does not exist: {source}") from exc
    if not isinstance(raw, dict):
        raise ValueError("experiment config root must be an object")
    base = source.parent
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("experiment name must be nonempty")
    population_path = _path(raw.get("population"), base, "population")
    population = _load_population(population_path)
    run_root = _path(raw.get("run_root"), base, "run_root")

    model_raw = raw.get("model")
    if not isinstance(model_raw, dict):
        raise ValueError("model must be an object")
    model_name = model_raw.get("name")
    model_label = model_raw.get("label")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model.name must be nonempty")
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError("model.label must be nonempty")
    model_seed = model_raw.get("seed")
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise ValueError("model.seed must be an integer")
    checkpoint = _path(model_raw.get("checkpoint"), base, "model.checkpoint")
    model = Arm(model_name, model_label, "model", model_seed, checkpoint)

    random_seeds = raw.get("random_seeds")
    if (
        not isinstance(random_seeds, list)
        or len(random_seeds) != RANDOM_RUN_COUNT
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in random_seeds)
        or len(set(random_seeds)) != RANDOM_RUN_COUNT
    ):
        raise ValueError("experiment requires exactly five distinct Random seeds")
    random_arms = tuple(
        Arm(f"random_s{seed}", "Random", "uniform", seed, checkpoint)
        for seed in random_seeds
    )

    protocol_raw = raw.get("protocol")
    if not isinstance(protocol_raw, dict):
        raise ValueError("protocol must be an object")
    primitive_prefix = protocol_raw.get("primitive_prefix")
    if not isinstance(primitive_prefix, str) or not primitive_prefix.strip():
        raise ValueError("protocol.primitive_prefix must be nonempty")
    slurm_partition = protocol_raw.get("slurm_partition")
    hardware_constraint = protocol_raw.get("hardware_constraint")
    if not isinstance(slurm_partition, str) or not slurm_partition.strip():
        raise ValueError("protocol.slurm_partition must be nonempty")
    if not isinstance(hardware_constraint, str) or not hardware_constraint.strip():
        raise ValueError("protocol.hardware_constraint must be nonempty")
    protocol = Protocol(
        evaluation_seed=_positive_int(
            protocol_raw.get("evaluation_seed"), "protocol.evaluation_seed"
        ),
        path_length=_positive_int(protocol_raw.get("path_length"), "protocol.path_length"),
        config_file=_path(protocol_raw.get("config_file"), base, "protocol.config_file"),
        primitive_data_dir=_path(
            protocol_raw.get("primitive_data_dir"), base, "protocol.primitive_data_dir"
        ),
        primitive_prefix=primitive_prefix,
        max_push_steps=_positive_int(
            protocol_raw.get("max_push_steps"), "protocol.max_push_steps"
        ),
        simulation_budget_per_keyhole=_positive_int(
            protocol_raw.get("simulation_budget_per_keyhole"),
            "protocol.simulation_budget_per_keyhole",
        ),
        region_max_chain_depth=_positive_int(
            protocol_raw.get("region_max_chain_depth"),
            "protocol.region_max_chain_depth",
        ),
        goals_per_region=_positive_int(
            protocol_raw.get("goals_per_region"), "protocol.goals_per_region"
        ),
        region_success_min_reachable=_positive_int(
            protocol_raw.get("region_success_min_reachable"),
            "protocol.region_success_min_reachable",
        ),
        n_shards=_positive_int(protocol_raw.get("n_shards"), "protocol.n_shards"),
        slurm_partition=slurm_partition,
        hardware_constraint=hardware_constraint,
    )
    if protocol.region_success_min_reachable > protocol.goals_per_region:
        raise ValueError("region_success_min_reachable cannot exceed goals_per_region")

    analysis_raw = raw.get("analysis")
    if not isinstance(analysis_raw, dict):
        raise ValueError("analysis must be an object")
    bootstrap_seed = analysis_raw.get("bootstrap_seed")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("analysis.bootstrap_seed must be an integer")
    analysis = AnalysisConfig(
        bootstrap_replicates=_positive_int(
            analysis_raw.get("bootstrap_replicates"), "analysis.bootstrap_replicates"
        ),
        bootstrap_seed=bootstrap_seed,
        simulator_call_cutoffs=_cutoffs(
            analysis_raw.get("simulator_call_cutoffs"),
            "simulator_call_cutoffs",
            True,
        ),
        wall_time_cutoffs_seconds=_cutoffs(
            analysis_raw.get("wall_time_cutoffs_seconds"),
            "wall_time_cutoffs_seconds",
            False,
        ),
    )
    return Experiment(
        name=name,
        source=source,
        population_path=population_path,
        population=population,
        run_root=run_root,
        model=model,
        random_arms=random_arms,
        protocol=protocol,
        analysis=analysis,
    )
