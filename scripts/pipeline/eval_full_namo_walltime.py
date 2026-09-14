#!/usr/bin/env python3
"""Timed frozen-scene evaluation using the canonical Full-NAMO planner.

Machine paths, artifact hashes, arms and shard count live in campaign YAML.
Failures retain consumed costs but have censored (null) success costs.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess

import yaml


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def shard_rows(rows, shard, nshards):
    """Round-robin balances the frozen manifest's difficulty-sorted order."""
    if not 0 <= shard < nshards:
        raise ValueError("shard must be in [0, nshards)")
    return rows[shard::nshards]


def summarize(rows):
    """All-run success-cost medians: an unsuccessful run has infinite cost."""
    def median_success_cost(key):
        value = statistics.median(r[key] if r["solved"] else math.inf for r in rows)
        return value if math.isfinite(value) else None

    return {
        "count": len(rows), "solved": sum(r["solved"] for r in rows),
        "success_rate": sum(r["solved"] for r in rows) / len(rows),
        "median_calls_until_success": median_success_cost("total_calls"),
        "median_time_until_success": median_success_cost("t_wall"),
        "median_consumed_calls": statistics.median(r["total_calls"] for r in rows),
        "median_elapsed_terminal_time": statistics.median(r["t_wall"] for r in rows),
        "solve_at_1s": sum(r["solved"] and r["t_wall"] <= 1 for r in rows) / len(rows),
        "solve_at_5s": sum(r["solved"] and r["t_wall"] <= 5 for r in rows) / len(rows),
    }


def load_scenes(config):
    root = Path(config["testbed"])
    manifest = root / "manifest.jsonl"
    if digest(manifest) != config["manifest_sha256"]:
        raise ValueError("Frozen manifest checksum mismatch")
    scenes = read_rows(manifest)
    if len(scenes) != config["n_scenes"] or len({r["geometry_id"] for r in scenes}) != len(scenes):
        raise ValueError("Frozen membership count or uniqueness mismatch")
    return scenes


def preflight(config, config_path):
    """Reject mismatched hardware/artifacts before any measured planner call."""
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    if cpu != config["cpu_model"]:
        raise RuntimeError(f"CPU mismatch: {cpu!r} != {config['cpu_model']!r}")
    job = subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]], text=True)
    fields = dict(token.split("=", 1) for token in job.split() if "=" in token)
    # Amarel Slurm 23.02 reports --exclusive as OverSubscribe=NO.
    if (fields.get("Partition") != "main" or fields.get("NumNodes") != "1"
            or fields.get("OverSubscribe") != "NO" or fields.get("Features") != "icelake"):
        raise RuntimeError(f"Requires main, one exclusive node, constraint icelake: {job}")
    node = subprocess.check_output(["scontrol", "show", "node", "-o", fields["NodeList"]], text=True)
    node_fields = dict(token.split("=", 1) for token in node.split() if "=" in token)
    if fields["NumCPUs"] != node_fields["CPUTot"]:
        raise RuntimeError("The allocation must own every CPU on the node")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "OPENCV_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key} must be 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Timed campaign must be CPU-only")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != config["code_commit"] or subprocess.check_output(["git", "diff", "HEAD", "--"], text=True):
        raise RuntimeError("Runtime must be the clean pinned commit")
    for path, expected in config["artifact_sha256"].items():
        if digest(path) != expected:
            raise RuntimeError(f"Artifact checksum mismatch: {path}")
    import namo_rl
    if Path(namo_rl.__file__).resolve() != Path(config["binding"]).resolve():
        raise RuntimeError("Wrong simulator binding imported")
    from namo.runtime_profile import require_canonical_runtime_config
    require_canonical_runtime_config(config["namo_config"])
    if config["budget"] != 20000 or not config["goal_clearance"]:
        raise RuntimeError("This Full-NAMO campaign requires 20k calls and goal clearance")
    return dict(host=socket.gethostname(), cpu_model=cpu, code_commit=commit,
                slurm_job=os.environ["SLURM_JOB_ID"], slurm_allocation=fields,
                campaign_sha256=digest(config_path))


def validate_row(row, budget):
    if row.get("failure_kind") == "runner_exception":
        raise RuntimeError(f"Technical failure: {row.get('error_message')}")
    if not row["goal_clearance_enabled"] or row["solved"] != row["final_goal_reachable"]:
        raise RuntimeError("Success must equal full end-goal reachability, with goal clearance enabled")
    if not 0 <= row["total_calls"] <= budget:
        raise RuntimeError("Invalid full-problem simulator budget accounting")
    if any(not math.isfinite(row[k]) or row[k] < 0 for k in ("t_wall", "t_sim", "t_score", "n_score")):
        raise RuntimeError("Invalid timing counters")
    if row["t_sim"] + row["t_score"] > row["t_wall"] + 0.001:
        raise RuntimeError("Component timers exceed full-problem time")


def run(config, config_path, arm_name, shard, smoke):
    from dataclasses import asdict
    import cv2
    import torch
    from namo.solvability_runner import SolveTask, solve_environment_task
    from namo.strategies.scorer_goal_strategy import _get_scorer
    from namo.planners.search_measurements import measurement_options

    measurement = measurement_options(config.get("measurement", {"record_timing": True}))
    if measurement["record_statistics"] or not measurement["record_timing"]:
        raise ValueError("timed driver requires measurement.record_statistics=false and record_timing=true")
    hardware = preflight(config, config_path)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    arm = config["arms"][arm_name]
    if arm["prior"] not in {"uniform", "model"} or arm["exec_mode"] not in {"search", "greedy_dfs"}:
        raise ValueError("Unsupported campaign arm")
    scenes = load_scenes(config)
    selected = config["smoke_scenes"][arm_name] if smoke else shard_rows(scenes, shard, config["nshards"])
    out = Path(config["output"]) / ("smoke" if smoke else "raw") / arm_name / f"shard_{shard:04d}"
    out.mkdir(parents=True, exist_ok=False)
    # Same cached loader as the planner; setup and synthetic warmup are untimed.
    if arm["prior"] == "model":
        _get_scorer(arm["checkpoint"], config["namo_config"], "cpu").warmup(repeats=3)
    metadata = dict(hardware, campaign=config["campaign"], arm=arm_name, **arm)
    save_json(out / "run_config.json", dict(metadata, config=config, shard=shard,
              selected_ids=[r["geometry_id"] for r in selected]))
    print("WALLTIME start " + json.dumps(dict(metadata, budget=config["budget"],
          hmax=2, agg="mean5", combine="q", discount="off", goal_clearance=True,
          n_scenes=len(selected), shard=shard)), flush=True)
    compact = []
    with (out / "outcomes.jsonl").open("x", buffering=1) as handle:
        for scene in selected:
            xml = Path(scene["xml_path"]) if smoke else Path(config["testbed"]) / scene["xml_relative_path"]
            if digest(xml) != scene["xml_sha256"]:
                raise RuntimeError(f"Scene checksum mismatch: {xml}")
            task = SolveTask(xml_path=str(xml), path_length_n=2, config_path=config["namo_config"],
                goal_strategy="scorer", region_max_chain_depth=2, primitive_data_dir=config["primitive_data_dir"],
                primitive_prefix="1x_car_d5_", rollout_samples_per_state=None, region_frontier_beam_width=None,
                region_success_min_reachable=20, goals_per_region=100, seed=42, use_cpp_snapshot=True,
                simulation_budget=config["budget"], simulation_budget_scope="full_problem", local_search="best_first",
                best_first_prior=arm["prior"], scorer_ckpt=arm.get("checkpoint"), ml_device="cpu", max_push_steps=5,
                shuffle_seed=arm["seed_base"], goal_clearance=True, exec_mode=arm["exec_mode"], **measurement)
            result = solve_environment_task(task)
            row = dict(result["row"], **metadata, geometry_id=scene["geometry_id"],
                       difficulty=scene["difficulty"], template=scene["template"],
                       horizon_pattern=scene["horizon_pattern"], xml_sha256=scene["xml_sha256"], task=asdict(task))
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            validate_row(row, config["budget"])
            if smoke and (not row["solved"] or ("expected_calls" in scene and row["total_calls"] != scene["expected_calls"])):
                raise RuntimeError("Smoke did not reproduce the expected end-goal success/call count")
            compact.append({k: row[k] for k in ("solved", "total_calls", "t_wall")})
            print(f"RESULT {arm_name} {row['geometry_id']} solved={row['solved']} calls={row['total_calls']} wall={row['t_wall']:.3f}", flush=True)
    save_json(out / "summary.json", dict(metadata, **summarize(compact), integrity_ok=True))


def report(config, config_path):
    """Require complete paired coverage before publishing any campaign summary."""
    scenes = load_scenes(config)
    expected_ids = {r["geometry_id"] for r in scenes}
    campaign_sha = digest(config_path)
    result = {"campaign": config["campaign"], "cpu_model": config["cpu_model"],
              "campaign_sha256": campaign_sha, "manifest_sha256": config["manifest_sha256"],
              "code_commit": config["code_commit"], "comparability": "within this campaign only",
              "null_success_median": "censored; simulator-call median is 20000+", "arms": {}}
    for name in config["arms"]:
        rows = []
        for shard in range(config["nshards"]):
            folder = Path(config["output"]) / "raw" / name / f"shard_{shard:04d}"
            summary = json.loads((folder / "summary.json").read_text())
            if not summary["integrity_ok"]:
                raise RuntimeError(f"Incomplete shard: {folder}")
            with (folder / "outcomes.jsonl").open() as handle:
                for line in handle:
                    row = json.loads(line)
                    validate_row(row, config["budget"])
                    if row["cpu_model"] != config["cpu_model"] or row["code_commit"] != config["code_commit"]:
                        raise RuntimeError("Mixed hardware or source revisions; campaign invalid")
                    if (row["campaign_sha256"] != campaign_sha or row["arm"] != name
                            or any(row.get(k) != v for k, v in config["arms"][name].items())):
                        raise RuntimeError("Mixed campaign configuration or arm assignment")
                    rows.append({k: row[k] for k in ("geometry_id", "difficulty", "horizon_pattern", "template",
                                                    "solved", "total_calls", "t_wall")})
        if len(rows) != len(scenes) or {r["geometry_id"] for r in rows} != expected_ids:
            raise RuntimeError(f"Incomplete or duplicated arm: {name}")
        result["arms"][name] = {"method": config["arms"][name], "overall": summarize(rows)}
        for key in ("difficulty", "horizon_pattern", "template"):
            result["arms"][name][key] = {value: summarize([r for r in rows if r[key] == value])
                                        for value in sorted({r[key] for r in rows})}
    save_json(Path(config["output"]) / "report.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.report:
        report(config, args.config)
    else:
        run(config, args.config, args.arm, args.shard, args.smoke)


if __name__ == "__main__":
    main()
