#!/usr/bin/env python3
"""Run best-first search arms on Tri-An's frozen one-keyhole problems.

Built for the 2026-09-14 ablation rerun on one_keyhole_frozen600_20260912_v1. The problem
setup ports Tri-An's evaluate_frozen_one (archive/dhruv_container_2026-09-11/, v2 in his
campaign root): each problem's frozen target points and door objects, the identity check
against the certificate problem_id, and the certificate-v1 restore for the problems his run
file lists. One change: it keeps the frozen door objects when today's room finder sees
different ones, and records both, instead of stopping. The search is scripts/sandbox/eval_bestfirst.py with his settings: up to 2
pushes, 3000 simulator calls, mean5, raw q, discount off, no-op dedupe and jam pruning on,
solved when 20% of the target points are reachable. By default untimed with statistics on, as
in his runs. --timed turns statistics off and timing on (the search's own t_wall, t_sim, t_score)
and runs one unit at a time in this process, single-threaded, recording the host, CPU model,
pinned CPUs and load average around each unit.

Work units are (problem, arm) pairs dealt round-robin into --nshards shards, so every shard
gets a mix of problems and arms when the shard count is not a multiple of the arm count;
--shuffle-seed shuffles the units first, which mixes them for any shard count. Each unit writes
<out>/<arm>/problem_<index>.jsonl plus its statistics sidecar; an existing row is skipped, so a
resubmitted shard only redoes what is missing.

Usage (one SLURM array task per shard):
  python scripts/pipeline/run_one_keyhole_frozen.py --problems <input/one_keyhole> \
      --config <namo_config.yaml> --arms arms.json --post-restore-ids ids.json --out <dir> \
      --shard 0 --nshards 480 --workers 14 [--only-index 3 --only-index 9] [--expect-source-sha256 <sha>]
      [--timed --workers 1 --shuffle-seed <n>]
"""

import argparse
import json
import multiprocessing
import os
import random
import socket
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
PLANNERS = {}


def cpu_model():
    return next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))


def evaluate(problems, row, arm, config, primitives, post_restore, timed=False):
    """Tri-An's evaluate_frozen_one for exec_mode=search, reading the problems from `problems`."""
    import namo_rl
    import eval_bestfirst as sandbox
    from namo.core.xml_goal_parser import extract_goal_from_xml
    from namo.planners import get_region_snapshot
    from namo.planners.search_measurements import content_digest, file_digest, state_record
    from namo.strategies import PrimitiveGoalStrategy

    sandbox.CFG, sandbox.DATA_DIR = config, primitives
    if arm["checkpoint"] not in PLANNERS:
        planner, device, _ = sandbox._make_planner(arm["prior"], arm["checkpoint"] or "", 3)
        if arm["prior"] == "model" and str(device) != "cpu":
            raise ValueError(f"one-keyhole runs score on CPU, got {device}")
        planner.prim = PrimitiveGoalStrategy(data_dir=primitives, primitive_prefix="1x_car_d5_")
        PLANNERS[arm["checkpoint"]] = planner
    planner = PLANNERS[arm["checkpoint"]]

    xml = str(problems / row["xml_relative_path"])
    task_path = problems / row["resolved_task"]["relative_path"]
    if file_digest(xml) != row["xml_sha256"] or file_digest(task_path) != row["resolved_task"]["sha256"]:
        raise ValueError(f"frozen scene or task checksum mismatch: {row['problem_id']}")
    tasks = json.loads(task_path.read_text())["tasks"]
    if len(tasks) != 1 or any(tasks[0][k] != row[k] for k in
                              ("boundary_objects", "target_samples", "target_region", "object_scope")):
        raise ValueError(f"frozen boundary definition mismatch: {row['problem_id']}")
    task = tasks[0]

    env = namo_rl.RLEnvironment(xml, config, False)
    env.reset()
    goal = extract_goal_from_xml(xml)
    env.set_robot_goal(*goal)
    env.get_reachable_objects()
    snapshot = get_region_snapshot(env, goals_per_region=100, use_xml_goal=True, seed=42, include_region_cells=True)
    initial = env.get_full_state()
    restored_first = row["problem_id"] in post_restore
    if restored_first:
        certificate = problems / row["certificate"]["relative_path"]
        saved = json.loads(certificate.read_text())
        if (file_digest(certificate) != row["certificate"]["sha256"]
                or saved["certification_source_sha256"] != "b99e07604981c2408ce38b1f82bdeb95ecbac8275926a8751f0c70114e355d93"
                or saved["problem_id"] != row["problem_id"]):
            raise ValueError("post-restore initialization requires the archived v1 certificate")
        # Certificate v1 read its initial state after this canonical restore.
        fresh = state_record(initial)
        env.set_full_state(initial)
        initial = env.get_full_state()
        restored = state_record(initial)
        if restored["qpos"] != fresh["qpos"] or any(restored["qvel"]):
            raise ValueError("legacy initialization violated the canonical zero-qvel restore")
    problem = dict(xml_sha256=file_digest(xml), initial_state=state_record(initial),
                   **{k: task[k] for k in ("target_samples", "target_region", "boundary_objects", "object_scope")})
    if content_digest(problem) != row["problem_id"]:
        raise ValueError(f"initialized scene differs from frozen certificate: {row['problem_id']}")
    # Tri-An's helper stopped when today's room finder saw different door objects. That happens on
    # 4 of the 600 frozen problems (37, 46, 77, 348). The frozen task stays the problem definition,
    # as it is for his tiers and Random runs, so the search uses the frozen objects and the row
    # records what the current code sees.
    live_door_objects = sandbox.pooled_boundary_tasks(snapshot, [dict(region=task["target_region"],
                                                                      target_points=task["target_samples"])])[0]["boundary_objects"]

    options = SimpleNamespace(key=str(problems / "manifest.jsonl"), prior=arm["prior"], ckpt=arm["checkpoint"] or "",
        seed_base=arm["seed_base"], hmax=2, sim_budget=3000, agg="mean5", combine="q", raw=True, dive_bonus=0.0,
        discount="off", gamma=0.65, tau=1.0, eps=0.001, w0_mode="one", free_strike_q=2.0, child_patience=1,
        dedupe_noop=True, prune_jam_depth=True, success="region")
    params = {k: v for k, v in vars(options).items() if k not in {"key", "ckpt", "seed_base", "success"}}
    params.update(gtable=None, object_scope="boundary_pool", seed_semantics="explicit_per_problem_v1",
        success_predicate="region", target_fraction=0.2, snapshot_seed=42, model_warmup_repeats=3,
        exec_mode="search", evaluation_adapter_sha256=file_digest(__file__))
    if restored_first:
        params["initial_state_convention"] = "certificate_v1_post_restore"
    measurement = dict(schema_version=1, record_statistics=not timed, record_timing=timed)
    result, _ = sandbox._evaluate_pooled_task(options, planner, env, xml, goal, initial, snapshot,
                                              env.get_observation(), task, params, measurement, None, None, None)
    result.update(exec_mode="search", population="one_keyhole", horizon=row["horizon"],
                  horizon_pattern=f"{row['horizon']}push", difficulty=row["difficulty"], template=row["template"],
                  geometry_id=row["geometry_id"], certification_problem_id=row["problem_id"],
                  certification_status="complete", arm=arm["name"], index=row["index"],
                  live_door_objects=live_door_objects,
                  door_objects_match=live_door_objects == task["boundary_objects"])
    return result


def run_unit(problems, row, arm, config, primitives, post_restore, destination, expect_source_sha256, timed=False):
    from namo.planners.search_measurements import save_run_row

    load_before = os.getloadavg()
    result = evaluate(problems, row, arm, config, primitives, post_restore, timed)
    if timed:
        result.update(host=socket.gethostname(), cpu_model=cpu_model(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                      load_average_before=load_before, load_average_after=os.getloadavg())
    source = (result.get("runtime_fingerprints") or {}).get("source_sha256")
    if result.get("technical_error") or (expect_source_sha256 and source != expect_source_sha256):
        destination.with_suffix(".rejected").write_text(json.dumps(result, sort_keys=True, default=str) + "\n")
        raise RuntimeError(f"{arm['name']} problem {row['index']}: technical_error={result.get('technical_error')} "
                           f"error={result.get('error_message')} source_sha256={source}")
    temporary = destination.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    save_run_row(temporary, result)
    temporary.replace(destination)
    return arm["name"], row["index"], bool(result["solved"]), result["total_calls"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", required=True, type=Path, help="Tri-An's input/one_keyhole folder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--primitives", default=str(REPO / "data"))
    parser.add_argument("--arms", required=True, type=Path, help="JSON list of {name, prior, checkpoint, seed_base}")
    parser.add_argument("--post-restore-ids", required=True, type=Path, help="JSON list of problem_ids")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only-index", type=int, action="append", help="restrict to these problem indices")
    parser.add_argument("--expect-source-sha256", help="reject rows whose code fingerprint differs")
    parser.add_argument("--timed", action="store_true", help="timing on, statistics off, one unit at a time")
    parser.add_argument("--shuffle-seed", type=int, help="shuffle the units with this seed before dealing shards")
    args = parser.parse_args()
    if args.timed and args.workers != 1:
        parser.error("--timed runs one unit at a time in this process; pass --workers 1")

    # The scorer picks its render config when it is built (scripts/sandbox/scorer_beam.py:55), so it
    # must see the same 1 mm config as the search.
    os.environ["NAMO_CFG"] = args.config
    for path in (REPO / "scripts/sandbox", REPO / "python", REPO / "build_python"):
        sys.path.insert(0, str(path))

    rows = [json.loads(line) for line in (args.problems / "manifest.jsonl").read_text().splitlines() if line]
    assert all(row["index"] == i for i, row in enumerate(rows))
    arms = json.loads(args.arms.read_text())
    post_restore = set(json.loads(args.post_restore_ids.read_text()))
    indices = args.only_index or range(len(rows))
    units = [(rows[i], arm) for i in indices for arm in arms]
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(units)
    units = units[args.shard::args.nshards]

    pending = []
    for row, arm in units:
        destination = args.out / arm["name"] / f"problem_{row['index']:04d}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            pending.append((row, arm, destination))
    print(json.dumps(dict(event="start", shard=args.shard, units=len(units), pending=len(pending))), flush=True)

    failures = 0
    if args.timed:
        import cv2
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        for row, arm, destination in pending:
            try:
                name, index, solved, calls = run_unit(args.problems, row, arm, args.config, args.primitives,
                                                      post_restore, destination, args.expect_source_sha256, timed=True)
                print(json.dumps(dict(event="saved", arm=name, index=index, solved=solved, calls=calls)), flush=True)
            except Exception as exc:
                failures += 1
                print(json.dumps(dict(event="failed", arm=arm["name"], index=row["index"], error=str(exc))), flush=True)
        sys.exit(1 if failures else 0)
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(pending))), mp_context=context) as pool:
        futures = [pool.submit(run_unit, args.problems, row, arm, args.config, args.primitives, post_restore,
                               destination, args.expect_source_sha256) for row, arm, destination in pending]
        for future in as_completed(futures):
            try:
                name, index, solved, calls = future.result()
                print(json.dumps(dict(event="saved", arm=name, index=index, solved=solved, calls=calls)), flush=True)
            except Exception as exc:
                failures += 1
                print(json.dumps(dict(event="failed", error=str(exc))), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
