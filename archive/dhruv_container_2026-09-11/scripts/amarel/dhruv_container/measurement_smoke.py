#!/usr/bin/env python3
"""Finite instrumentation check: three saved fixtures, never a candidate campaign."""

import argparse
from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import namo_rl

from namo.core.xml_goal_parser import extract_goal_from_xml
from namo.planners import get_region_snapshot
from namo.planners.search_measurements import (
    file_digest, save_run_row, write_json_artifact,
)
from namo.solvability_runner import SolveTask, solve_environment_task
from namo.strategies.scorer_goal_strategy import _get_scorer


PARITY_FIELDS = ("run_id", "problem_id", "semantic_protocol_hash", "solved", "total_calls",
                 "failure_kind", "final_goal_reachable", "initialized_state_digest",
                 "terminal_state_digest", "execution_digest", "attempt_count", "attempt_digests", "commit_count")


def inspect(row, path):
    """Check actual artifact coverage and call conservation, including failed prefixes."""
    assert row["complete"] and not row.get("technical_error"), row
    assert sum(a["sim_call_end"] - a["sim_call_start"] for a in row["attempt_digests"]) == row["total_calls"]
    assert len(row["attempt_digests"]) == row["attempt_count"]
    assert bool("t_wall" in row) == row["measurement"]["record_timing"]
    assert bool("statistics_sidecar" in row) == row["measurement"]["record_statistics"]
    if row["measurement"]["record_statistics"]:
        sidecar = path.parent / row["statistics_sidecar"]["path"]
        assert file_digest(sidecar) == row["statistics_sidecar"]["sha256"]
        statistics = json.loads(gzip.decompress(sidecar.read_bytes()))
        assert len(statistics["attempts"]) == row["attempt_count"]
        assert len(statistics["commits"]) == row["commit_count"]
        assert all(c["state_digest"] in statistics["states"] for c in statistics["checkpoints"])
        assert any("region_cells" in snapshot for snapshot in statistics["snapshots"])
        print(json.dumps(dict(run_id=row["run_id"], states=len(statistics["states"]),
                              snapshots=len(statistics["snapshots"]), checkpoints=len(statistics["checkpoints"]),
                              calls=row["total_calls"], commits=row["commit_count"], solved=row["solved"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("all", "full_success", "full_failed_prefix", "one_pooled"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    config = args.artifacts / "config/namo_config.yaml"
    checkpoint = args.artifacts / "HY5U_s2.ckpt"
    base = SolveTask(xml_path=str(args.artifacts / "scenes/goal_clearance.xml"), path_length_n=2,
        config_path=str(config), goal_strategy="scorer", region_max_chain_depth=2,
        primitive_data_dir=str(args.artifacts / "primitives"), primitive_prefix="1x_car_d5_",
        rollout_samples_per_state=None, region_frontier_beam_width=None,
        region_success_min_reachable=20, goals_per_region=100, seed=42, use_cpp_snapshot=True,
        simulation_budget=64, local_search="best_first", best_first_prior="model", scorer_ckpt=str(checkpoint),
        ml_device="cpu", max_push_steps=5, shuffle_seed=7000, goal_clearance=True)
    cases = ("full_success", "full_failed_prefix", "one_pooled") if args.case == "all" else (args.case,)
    if any(case.startswith("full") for case in cases):
        _get_scorer(str(checkpoint), str(config), "cpu").warmup(repeats=1)
    summaries = {}
    for case in cases:
        reference = None
        for stats, timing in ((False, False), (True, False), (False, True), (True, True)):
            measurement = dict(schema_version=1, record_statistics=stats, record_timing=timing)
            path = args.output / case / f"s{int(stats)}t{int(timing)}" / "outcomes.jsonl"
            path.parent.mkdir(parents=True)
            if case.startswith("full"):
                task = replace(base, record_statistics=stats, record_timing=timing,
                               full_namo_max_iterations=1 if case == "full_failed_prefix" else None)
                row = solve_environment_task(task)["row"]
            else:
                import eval_bestfirst as sandbox
                sandbox.CFG = str(config)
                sandbox.DATA_DIR = str(args.artifacts / "primitives")
                xml = repo / "python/tests/data/two_movable_doorway_fixture.xml"
                env = namo_rl.RLEnvironment(str(xml), str(config), False)
                env.reset()
                goal = extract_goal_from_xml(str(xml))
                env.set_robot_goal(*goal)
                env.get_reachable_objects()
                snapshot = get_region_snapshot(env, goals_per_region=100, use_xml_goal=True, seed=42,
                                               include_region_cells=stats)
                rec = sandbox.pooled_boundary_tasks(snapshot, [{"region": snapshot["goal_label"]}])[0]
                planner, _, _ = sandbox._make_planner("uniform", "", 0)
                options = SimpleNamespace(key=str(xml), prior="uniform", ckpt="", seed_base=42, hmax=2,
                    sim_budget=64, agg="mean5", combine="q", raw=True, dive_bonus=0.0, discount="off",
                    gamma=0.65, tau=1.0, eps=0.001, w0_mode="one", free_strike_q=2.0, child_patience=1,
                    dedupe_noop=True, prune_jam_depth=True, success="region")
                row, _ = sandbox._evaluate_pooled_task(options, planner, env, str(xml), goal,
                    env.get_full_state(), snapshot, env.get_observation() if stats else None,
                    rec, {key: value for key, value in vars(options).items() if key != "key"},
                    measurement, None, None, None)
            row["collection"] = dict(hostname=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                                      purpose="measurement_smoke", inflation_margin="canonical_1mm")
            saved = save_run_row(path, row)
            inspect(saved, path)
            if case == "full_failed_prefix":
                assert not saved["solved"] and saved["commit_count"] > 0, saved
            else:
                assert saved["solved"], saved
            actual = {field: saved.get(field) for field in PARITY_FIELDS}
            if reference is None:
                reference = actual
            else:
                assert actual == reference, {key: (reference[key], actual[key]) for key in reference if actual[key] != reference[key]}
            write_json_artifact(path.parent / "complete.json", {"expected_run_count": 1, "accounted_run_count": 1})
        summaries[case] = reference
    write_json_artifact(args.output / "smoke-complete.json", summaries)
    print(json.dumps({case: {key: row[key] for key in ("solved", "total_calls", "commit_count", "execution_digest")}
                      for case, row in summaries.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
