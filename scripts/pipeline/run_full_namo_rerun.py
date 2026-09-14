#!/usr/bin/env python3
"""Run every requested arm on one frozen Full NAMO scene, side by side.

Built for the 2026-09-14 rerun after the local rendering-goal fix. The task
settings copy Tri-An's frozen400 reference helper
(tools/behavioral-v1/candidate_campaign.py:222-230) exactly: 9000 simulated
pushes shared across the whole problem, region depth 2, 100 goal samples with
20 reachable, goal clearance on. Two things differ on purpose: nothing is
timed, and per-run statistics are off, because this rerun reports solve rate
and simulator calls only.

Each arm writes one JSON row to <out>/<arm>/scene_<index>.json. An existing
row is skipped, so a resubmitted array task only redoes what is missing.

Usage (one SLURM array task per manifest index):
  python scripts/pipeline/run_full_namo_rerun.py --manifest-dir <input/full_namo> \
      --config <namo_config.yaml> --out <dir> --index 0 \
      --arm HY5U_s1:model:7000:<ckpt> --arm random_s7000:uniform:7000
"""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def parse_arm(text):
    """NAME:PRIOR:SHUFFLE_SEED[:CHECKPOINT]"""
    parts = text.split(":", 3)
    if len(parts) < 3 or parts[1] not in {"model", "uniform"}:
        raise argparse.ArgumentTypeError(f"arm must be NAME:model|uniform:SEED[:CKPT], got {text!r}")
    checkpoint = parts[3] if len(parts) == 4 else None
    if (parts[1] == "model") != (checkpoint is not None):
        raise argparse.ArgumentTypeError(f"model arms need a checkpoint, uniform arms must not: {text!r}")
    return dict(name=parts[0], prior=parts[1], seed=int(parts[2]), checkpoint=checkpoint)


def run_arm(xml_path, arm, config, primitives, destination, scene):
    from namo.solvability_runner import SolveTask, solve_environment_task

    task = SolveTask(
        xml_path=str(xml_path), path_length_n=2, config_path=config, goal_strategy="scorer",
        region_max_chain_depth=2, primitive_data_dir=primitives, primitive_prefix="1x_car_d5_",
        rollout_samples_per_state=None, region_frontier_beam_width=None,
        region_success_min_reachable=20, goals_per_region=100, seed=42, use_cpp_snapshot=True,
        simulation_budget=9000, simulation_budget_scope="full_problem", local_search="best_first",
        best_first_prior=arm["prior"], scorer_ckpt=arm["checkpoint"], ml_device="cpu", max_push_steps=5,
        shuffle_seed=arm["seed"], goal_clearance=True, exec_mode="search",
        record_statistics=False, record_timing=False,
    )
    started = time.monotonic()
    row = solve_environment_task(task)["row"]
    row.update(arm=arm["name"], scene=scene, host_wall_seconds=round(time.monotonic() - started, 3))
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, sort_keys=True, default=str) + "\n")
    temporary.replace(destination)
    return arm["name"], bool(row.get("solved")), row.get("total_calls"), row.get("technical_error")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--primitives", default="data")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--arm", action="append", required=True, type=parse_arm)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.manifest_dir / "manifest.jsonl").read_text().splitlines() if line]
    row = rows[args.index]
    assert row["index"] == args.index, (row["index"], args.index)
    xml_path = args.manifest_dir / row["xml_relative_path"]
    scene = dict(index=row["index"], difficulty=row["difficulty"], geometry_id=row["geometry_id"],
                 template=row["template"], xml_relative_path=row["xml_relative_path"])

    pending = []
    for arm in args.arm:
        destination = args.out / arm["name"] / f"scene_{args.index:04d}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(json.dumps(dict(event="skip_existing", arm=arm["name"], index=args.index)), flush=True)
        else:
            pending.append((arm, destination))

    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=max(1, len(pending)), mp_context=context) as pool:
        futures = [pool.submit(run_arm, xml_path, arm, args.config, args.primitives, destination, scene)
                   for arm, destination in pending]
        for future in as_completed(futures):
            name, solved, calls, error = future.result()
            print(json.dumps(dict(event="saved", arm=name, index=args.index, difficulty=row["difficulty"],
                                  solved=solved, calls=calls, technical_error=error)), flush=True)


if __name__ == "__main__":
    main()
