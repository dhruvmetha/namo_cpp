from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from full_namo_sim_exp.experiment_io import Arm, Experiment, load_experiment
from full_namo_sim_exp.provenance import verify_frozen_experiment


@dataclass(frozen=True)
class RunnerCommand:
    arm: Arm
    argv: tuple[str, ...]
    output_dir: Path


def shard_scenes(experiment: Experiment, shard_index: int) -> tuple[str, ...]:
    if shard_index < 0 or shard_index >= experiment.protocol.n_shards:
        raise ValueError(
            f"shard index {shard_index} outside [0, {experiment.protocol.n_shards})"
        )
    scenes = experiment.population.scene_ids
    start = shard_index * len(scenes) // experiment.protocol.n_shards
    end = (shard_index + 1) * len(scenes) // experiment.protocol.n_shards
    if start >= end:
        raise ValueError(f"shard {shard_index} is empty; reduce protocol.n_shards")
    return scenes[start:end]


def _rotated_arms(experiment: Experiment, shard_index: int) -> tuple[Arm, ...]:
    arms = experiment.arms
    offset = shard_index % len(arms)
    return arms[offset:] + arms[:offset]


def build_runner_commands(
    experiment: Experiment,
    *,
    shard_index: int,
    shard_manifest: Path,
    python_executable: str,
    output_root: Path | None = None,
) -> tuple[RunnerCommand, ...]:
    protocol = experiment.protocol
    commands: list[RunnerCommand] = []
    root = output_root or experiment.raw_shard_root(shard_index)
    for arm in _rotated_arms(experiment, shard_index):
        output = root / arm.name
        argv = (
            python_executable,
            "-m",
            "full_namo_sim_exp.runner",
            "--manifest",
            str(shard_manifest),
            "--path-length",
            str(protocol.path_length),
            "--output-dir",
            str(output),
            "--config-file",
            str(protocol.config_file),
            "--goal-strategy",
            "scorer",
            "--region-max-chain-depth",
            str(protocol.region_max_chain_depth),
            "--primitive-data-dir",
            str(protocol.primitive_data_dir),
            "--primitive-prefix",
            protocol.primitive_prefix,
            "--region-success-min-reachable",
            str(protocol.region_success_min_reachable),
            "--goals-per-region",
            str(protocol.goals_per_region),
            "--seed",
            str(protocol.evaluation_seed),
            "--ordering-seed",
            str(arm.seed),
            "--simulation-budget",
            str(protocol.simulation_budget_per_keyhole),
            "--simulation-budget-scope",
            "keyhole",
            "--local-search",
            "best_first",
            "--best-first-prior",
            arm.prior,
            "--scorer-ckpt",
            str(arm.checkpoint),
            "--ml-device",
            "cpu",
            "--workers",
            "1",
        )
        commands.append(RunnerCommand(arm=arm, argv=argv, output_dir=output))
    return tuple(commands)


def ensure_final_shard_available(experiment: Experiment, shard_index: int) -> None:
    final = experiment.raw_shard_root(shard_index)
    if final.exists():
        raise ValueError(f"refusing to overwrite raw shard output: {final}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one population shard through HY5U and five Random seeds sequentially."
    )
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment = load_experiment(args.experiment)
    experiment.validate_launch_inputs()
    verify_frozen_experiment(experiment)
    scenes = shard_scenes(experiment, args.shard_index)
    ensure_final_shard_available(experiment, args.shard_index)
    if args.dry_run:
        staging_root = (
            experiment.run_root / "staging" / f"DRY_RUN_shard_{args.shard_index:04d}"
        )
    else:
        staging_parent = experiment.run_root / "staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".shard_{args.shard_index:04d}_",
                dir=staging_parent,
            )
        )
    manifest = staging_root / "input.txt"
    if not args.dry_run:
        manifest.write_text("".join(f"{scene}\n" for scene in scenes), encoding="utf-8")
    commands = build_runner_commands(
        experiment,
        shard_index=args.shard_index,
        shard_manifest=manifest,
        python_executable=sys.executable,
        output_root=staging_root,
    )
    try:
        for command in commands:
            print(f"[{args.shard_index:04d}] {command.arm.name}: {' '.join(command.argv)}")
            if not args.dry_run:
                subprocess.run(command.argv, check=True)
    except BaseException:
        print(f"incomplete shard staging retained at {staging_root}", file=sys.stderr)
        raise
    if not args.dry_run:
        final = experiment.raw_shard_root(args.shard_index)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging_root.rename(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
