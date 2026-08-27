from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Sequence

from full_namo_sim_exp.aggregate import aggregate_all
from full_namo_sim_exp.experiment_io import load_experiment
from full_namo_sim_exp.plot import render
from full_namo_sim_exp.provenance import freeze_experiment, verify_frozen_experiment
from full_namo_sim_exp.stats import write_statistics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full NAMO final-experiment pipeline.")
    parser.add_argument(
        "command",
        choices=("validate", "launch-command", "aggregate", "stats", "plot", "all"),
    )
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args(argv)
    experiment = load_experiment(args.experiment)

    if args.command in {"validate", "launch-command"}:
        experiment.validate_launch_inputs()
    if args.command == "validate":
        lock = freeze_experiment(experiment)
        print(
            json.dumps(
                {
                    "experiment": experiment.name,
                    "population": experiment.population.name,
                    "population_size": len(experiment.population.scene_ids),
                    "arms": [arm.name for arm in experiment.arms],
                    "n_shards": experiment.protocol.n_shards,
                    "lock": str(lock),
                },
                indent=2,
            )
        )
        return 0
    verify_frozen_experiment(experiment)
    if args.command == "launch-command":
        script = Path(__file__).with_name("run_interleaved.slurm")
        print(
            "EXPERIMENT="
            + shlex.quote(str(experiment.source))
            + " sbatch --exclusive --array=0-"
            + str(experiment.protocol.n_shards - 1)
            + " --partition="
            + shlex.quote(experiment.protocol.slurm_partition)
            + " --constraint="
            + shlex.quote(experiment.protocol.hardware_constraint)
            + " "
            + shlex.quote(str(script))
        )
        return 0
    if args.command in {"aggregate", "all"}:
        summaries = aggregate_all(experiment)
        print(json.dumps(summaries, indent=2, sort_keys=True))
    if args.command in {"stats", "all"}:
        print(write_statistics(experiment))
    if args.command in {"plot", "all"}:
        for path in render(experiment):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
