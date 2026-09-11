#!/usr/bin/env python3
"""Bounded integration checks and fixed-action replay; never launches a campaign."""

import argparse
import ctypes
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import time

import cv2
import namo_rl
import torch

from namo.core.xml_goal_parser import extract_goal_from_xml
from namo.planners import get_region_snapshot
from namo.planners.full_namo.full_namo_planner import FullNAMOPlanner
from namo.solvability_runner import SolveTask, build_full_namo_planner_config, serialize_action


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


class CountedEnvironment(namo_rl.RLEnvironment):
    def __init__(self, xml, config):
        super().__init__(str(xml), str(config), False)
        self.observed_calls = 0

    def step(self, action):
        self.observed_calls += 1
        return super().step(action)


def state_and_graph(env):
    before = bool(env.is_robot_goal_reachable())
    reachable = sorted(env.get_reachable_objects())
    snapshot = get_region_snapshot(env, goals_per_region=100, seed=42,
                                   use_cpp_unified=True, use_xml_goal=True,
                                   include_goal_clearance=True)
    state = env.get_full_state()
    return dict(qpos=list(state.qpos), qvel=list(state.qvel),
                authoritative_before_snapshot=before,
                authoritative_after_snapshot=bool(env.is_robot_goal_reachable()),
                reachable_objects=reachable,
                robot_label=snapshot.get("robot_label"), goal_label=snapshot.get("goal_label"),
                snapshot_goal_reachable=bool(snapshot.get("goal_reachable")),
                adjacency={k: sorted(v) for k, v in snapshot["adjacency"].items()},
                edge_objects={a: {b: sorted(v) for b, v in edges.items()}
                              for a, edges in snapshot["edge_objects"].items()},
                multi_object_edges={k: sorted(v) for k, v in snapshot.get("multi_object_edges", {}).items()},
                goal_clearance=json_value(snapshot.get("goal_clearance")))


def make_env(xml, config, initial_state=None):
    env = CountedEnvironment(xml, config)
    if initial_state:
        state = env.get_full_state()
        state.qpos = initial_state["qpos"]
        state.qvel = initial_state["qvel"]
        env.set_full_state(state)
        assert list(env.get_full_state().qpos) == initial_state["qpos"]
    goal = extract_goal_from_xml(str(xml))
    env.set_robot_goal(*goal)
    return env, goal


def deserialize_action(row):
    action = namo_rl.Action()
    action.object_id = row["object_id"]
    action.edge_idx = row["edge_idx"]
    action.depth = row["depth"]
    action.x, action.y, action.theta = row["target"]
    return action


def physics_replay(name, xml, config, actions, initial_state=None):
    env, _goal = make_env(xml, config, initial_state)
    states = [state_and_graph(env)]
    for row in actions:
        env.step(deserialize_action(row))
        states.append(state_and_graph(env))
    return dict(name=name, xml_sha256=digest(xml), config_sha256=digest(config),
                actions=actions, states=states, observed_calls=env.observed_calls,
                final_goal_reachable=bool(env.is_robot_goal_reachable()))


def task_for(xml, config, artifacts, prior="uniform", seed=7000):
    return SolveTask(xml_path=str(xml), path_length_n=2, config_path=str(config),
        goal_strategy="scorer", region_max_chain_depth=2,
        primitive_data_dir=str(artifacts / "primitives"), primitive_prefix="1x_car_d5_",
        rollout_samples_per_state=None, region_frontier_beam_width=None,
        region_success_min_reachable=20, goals_per_region=100, seed=42,
        use_cpp_snapshot=True, simulation_budget=20000, simulation_budget_scope="full_problem",
        local_search="best_first", best_first_prior=prior,
        scorer_ckpt=str(artifacts / "HY5U_s2.ckpt") if prior == "model" else None,
        ml_device="cpu", max_push_steps=5, shuffle_seed=seed,
        goal_clearance=True, exec_mode="search", record_timing=True)


def search_case(name, task, initial_state=None):
    env, goal = make_env(task.xml_path, task.config_path, initial_state)
    initial = state_and_graph(env)
    config = build_full_namo_planner_config(task)
    timing = dict(t_sim=0.0, t_score=0.0, n_score=0)
    config.algorithm_params["full_namo_timing"] = timing
    planner = FullNAMOPlanner(env, config)
    started = time.perf_counter()
    result = planner.search(goal)
    timing["t_wall"] = time.perf_counter() - started
    stats = result.algorithm_stats
    final = state_and_graph(env)
    calls = int(stats["simulation_budget_used"])
    row = dict(name=name, success=bool(result.success),
        final_goal_reachable=bool(env.is_robot_goal_reachable()),
        observed_calls=env.observed_calls, simulation_budget_used=calls,
        failure_kind=stats.get("failure_kind"), failure_subkind=stats.get("failure_subkind"),
        error_message=result.error_message, timing=timing,
        actions=[serialize_action(a) for a in result.action_sequence or []],
        stats=json_value(stats), initial=initial, final=final)
    row["checks"] = dict(authoritative_success=row["success"] == row["final_goal_reachable"],
        counted_budget=0 <= calls == env.observed_calls <= task.simulation_budget,
        cumulative_timing=timing["t_sim"] >= 0 and timing["t_score"] >= 0
            and timing["t_sim"] + timing["t_score"] <= timing["t_wall"] + .001)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("fixed", "search"), required=True)
    args = parser.parse_args()
    artifacts, out = args.artifacts.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(os.environ["NAMO_REPO"]).resolve()
    os.chdir(repo)
    assert "NAMO_DISABLE_MOVABLE_BLOB_EDGES" not in os.environ
    assert Path(namo_rl.__file__).resolve().parent == repo / "build_python"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    library_paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if "libmujoco.so" in line})
    assert len(library_paths) == 1, library_paths
    library = ctypes.CDLL(library_paths[0])
    library.mj_versionString.restype = ctypes.c_char_p
    header = Path(os.environ["MJ_PATH"], "include/mujoco/mujoco.h").read_text()
    header_version = int(re.search(r"#define\s+mjVERSION_HEADER\s+(\d+)", header)[1])
    assert header_version == library.mj_version() == 320
    assert platform.python_version() == "3.12.13"
    config = artifacts / "config/namo_config.yaml"
    inputs = [config, config.with_name("wavefront_inflation.yaml"), artifacts / "HY5U_s2.ckpt"]
    inputs += sorted((artifacts / "primitives").glob("*.dat"))
    inputs += sorted((artifacts / "scenes").glob("*.xml"))
    inputs += [artifacts / "fixed-replay-inputs.json", artifacts / "physics-source-record.json",
               artifacts / "saved-consistency-failure.json"]
    inputs += sorted(p for p in (artifacts / "sage_learning").rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for path in (artifacts / "primitives").glob("*.dat"):
        assert digest(repo / "data" / path.name) == digest(path)
    identity = dict(host=socket.gethostname(), python=platform.python_version(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        binding=str(Path(namo_rl.__file__).resolve()), binding_sha256=digest(namo_rl.__file__),
        linked_mujoco=library_paths[0], linked_mujoco_sha256=digest(library_paths[0]),
        linked_mujoco_version=library.mj_versionString().decode(), header_version=header_version,
        input_sha256={str(p.relative_to(artifacts)): digest(p) for p in inputs},
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        build_info=(repo / "build_python/BUILD_INFO").read_text(),
        state_restore="NAMO set_full_state restores qpos and zeroes qvel; no historical MuJoCo warmstart is restored.")
    assert not identity["source_status"], identity["source_status"]
    save(out / "identity.json", identity)
    fixed = json.loads((artifacts / "fixed-replay-inputs.json").read_text())
    source = json.loads((artifacts / "physics-source-record.json").read_text())
    failure = json.loads((artifacts / "saved-consistency-failure.json").read_text())["row"]
    saved_goal_state = source["historical_control"]["result"]["row"]["terminal_state"]
    if args.phase == "fixed":
        for case in fixed:
            xml = repo / case["repo_xml"] if "repo_xml" in case else artifacts / case["xml"]
            cfg = repo / case["repo_config"] if "repo_config" in case else config
            initial = saved_goal_state if case.get("start") == "goal_terminal" else None
            record = physics_replay(case["name"], xml, cfg, case["actions"], initial)
            save(out / (case["name"] + ".json"), record)
            if case["name"] == "pooled_doorway":
                a, b = [action["object_id"] for action in case["actions"]]
                root, after_a, after_b = record["states"]
                assert a != b and a in root["reachable_objects"] and b not in root["reachable_objects"]
                assert root["adjacency"]["robot"] == ["goal"]
                assert not root["authoritative_after_snapshot"]
                assert b in after_a["reachable_objects"] and not after_a["authoritative_after_snapshot"]
                assert after_b["authoritative_after_snapshot"]
            if case["name"] in {"pooled_doorway", "goal_scene", "goal_terminal"}:
                assert record["final_goal_reachable"], case["name"]
            print(json.dumps(dict(name=case["name"], goal=record["final_goal_reachable"],
                                  calls=record["observed_calls"])), flush=True)
        orders = {}
        for order in ("ASAS", "SASA"):
            env, _goal = make_env(artifacts / "scenes/consistency_failure.xml", config)
            env.is_robot_goal_reachable()
            state = env.get_full_state()
            state.qpos, state.qvel = failure["terminal_state"]["qpos"], failure["terminal_state"]["qvel"]
            env.set_full_state(state)
            probes = []
            for token in order:
                if token == "A":
                    value = bool(env.is_robot_goal_reachable())
                else:
                    snapshot = get_region_snapshot(env, seed=42, use_cpp_unified=True,
                        use_xml_goal=False, include_goal_clearance=True)
                    value = {k: snapshot[k] for k in ("goal_reachable", "robot_label", "goal_label")}
                assert list(env.get_full_state().qpos) == failure["terminal_state"]["qpos"]
                probes.append(dict(query=token, value=value))
            orders[order] = probes
        save(out / "saved_consistency_queries.json", dict(orders=orders,
            historical_final_goal_reachable=failure["final_goal_reachable"],
            historical_goal_diagnostics=failure["goal_diagnostics"]))
        return
    from namo.strategies.scorer_goal_strategy import _get_scorer
    _get_scorer(str(artifacts / "HY5U_s2.ckpt"), str(config), "cpu").warmup(repeats=3)
    cases = [
        ("pooled_search", task_for(repo / "python/tests/data/two_movable_doorway_fixture.xml", config, artifacts, seed=42), None),
        ("model_multihop", task_for(artifacts / "scenes/goal_clearance.xml", config, artifacts, prior="model"), None),
        ("goal_terminal_recovery", task_for(artifacts / "scenes/goal_clearance.xml", config, artifacts, prior="model"), saved_goal_state),
        ("saved_consistency_recovery", task_for(artifacts / "scenes/consistency_failure.xml", config, artifacts, seed=8000), failure["terminal_state"]),
    ]
    cases.insert(1, ("optimistic_edge_zero_budget", replace(cases[0][1], simulation_budget=0), None))
    all_checks = []
    for name, task, initial in cases:
        record = search_case(name, task, initial)
        save(out / (name + ".json"), record)
        all_checks.extend(record["checks"].values())
        if name == "optimistic_edge_zero_budget":
            assert not record["success"] and record["observed_calls"] == 0
        if name in {"pooled_search", "model_multihop", "goal_terminal_recovery"}:
            assert record["success"], record
        if name == "goal_terminal_recovery":
            assert any(row.get("target_kind") == "goal_clearance" for row in record["stats"]["iteration_trace"])
        print(json.dumps({k: record[k] for k in ("name", "success", "simulation_budget_used", "checks", "failure_kind")}), flush=True)
    assert all(all_checks)
    from namo.solvability_runner import solve_environment_task
    wrapped = solve_environment_task(task_for(artifacts / "scenes/goal_clearance.xml", config, artifacts, prior="model"))
    save(out / "canonical_timed_runner.json", wrapped)
    row = wrapped["row"]
    assert row["solved"] and row["final_goal_reachable"]
    assert 0 < row["total_calls"] <= 20000
    assert row["t_sim"] + row["t_score"] <= row["t_wall"] + .001


if __name__ == "__main__":
    main()
