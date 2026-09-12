import json
import gzip
from pathlib import Path
from types import SimpleNamespace

import pytest

from namo.environment_selection import RegionPathAnalysis
from namo.solvability_runner import run_exact_n_solvability
from namo.runtime_profile import CANONICAL_CONFIG, CANONICAL_PRIMITIVE_PREFIX


def test_completed_failed_prefix_survives_interrupted_shard(tmp_path, monkeypatch):
    from namo import solvability_runner as runner
    from namo.planners.search_measurements import file_digest

    paths = [str(tmp_path / name) for name in ("first.xml", "second.xml")]
    monkeypatch.setattr(runner, "get_xml_files", lambda **_k: paths)
    monkeypatch.setattr(runner, "analyze_environment_path_length", lambda path, *_a, **_k:
                        RegionPathAnalysis(path, 2, "robot", "goal", {}))
    statistics = {"attempts": [], "decisions": [], "commits": [{"actions": [{"object_id": "box"}]}],
                  "checkpoints": [{"state_digest": "saved"}], "states": {"saved": {"qpos": [1.0], "qvel": [0.0]}},
                  "snapshots": []}

    def interrupted(tasks, workers):
        yield {"kind": "unsolved", "row": {"xml_path": paths[0], "run_id": "first-run", "complete": True,
               "solved": False, "failure_kind": "region_path_exhausted", "statistics": statistics}}
        raise RuntimeError("worker lost")

    monkeypatch.setattr(runner, "_iter_solve_results", interrupted)
    output = tmp_path / "out"
    kwargs = dict(repo_root=Path.cwd(), input_dir=str(tmp_path), manifest_path=None, path_length=2,
                  output_dir=str(output), measurement={"record_statistics": True})
    with pytest.raises(RuntimeError, match="worker lost"):
        runner.run_exact_n_solvability(**kwargs)
    row = json.loads((output / "outcomes.jsonl").read_text())
    assert json.loads((output / "unsolved.jsonl").read_text()) == row
    assert "statistics" not in row
    sidecar = output / row["statistics_sidecar"]["path"]
    assert file_digest(sidecar) == row["statistics_sidecar"]["sha256"]
    assert json.loads(gzip.decompress(sidecar.read_bytes()))["states"] == statistics["states"]
    assert not (output / "complete.json").exists()
    with pytest.raises(FileExistsError):
        runner.run_exact_n_solvability(**kwargs)


def test_run_exact_n_solvability_writes_expected_manifests(tmp_path, monkeypatch):
    xml_paths = [
        str(tmp_path / "env_a.xml"),
        str(tmp_path / "env_b.xml"),
        str(tmp_path / "env_c.xml"),
        str(tmp_path / "env_d.xml"),
    ]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(Path(CANONICAL_CONFIG).read_text(encoding="utf-8"), encoding="utf-8")

    analyses = {
        xml_paths[0]: RegionPathAnalysis(
            xml_path=xml_paths[0],
            path_length_n=2,
            robot_label="robot",
            goal_label="goal",
            adjacency={"robot": {"a"}},
        ),
        xml_paths[1]: RegionPathAnalysis(
            xml_path=xml_paths[1],
            path_length_n=-1,
            robot_label=None,
            goal_label=None,
            adjacency={},
            selection_error="missing_goal_region",
        ),
        xml_paths[2]: RegionPathAnalysis(
            xml_path=xml_paths[2],
            path_length_n=3,
            robot_label="robot",
            goal_label="goal",
            adjacency={"robot": {"b"}},
        ),
        xml_paths[3]: RegionPathAnalysis(
            xml_path=xml_paths[3],
            path_length_n=2,
            robot_label="robot",
            goal_label="goal",
            adjacency={"robot": {"c"}},
        ),
    }

    monkeypatch.setattr("namo.solvability_runner.get_xml_files", lambda **_kwargs: list(xml_paths))
    monkeypatch.setattr(
        "namo.solvability_runner.analyze_environment_path_length",
        lambda xml_path, *_args, **_kwargs: analyses[xml_path],
    )

    def fake_solve(task):
        assert task.record_statistics is True
        assert task.record_timing is False
        if task.xml_path.endswith("env_a.xml"):
            return {
                "kind": "solved",
                "row": {
                    "xml_path": task.xml_path,
                    "path_length_n": task.path_length_n,
                    "solution_length": 1,
                    "solution": [
                        {
                            "object_id": "box",
                            "edge_idx": 4,
                            "depth": 1,
                            "target": [1.0, 2.0, 0.0],
                        }
                    ],
                },
            }
        return {
            "kind": "unsolved",
            "row": {
                "xml_path": task.xml_path,
                "path_length_n": task.path_length_n,
                "outcome": "planner_failure",
                "failure_kind": "opener_failure_not_boundary_exhausted",
                "failure_subkind": None,
                "error_message": "no opening found",
            },
        }

    monkeypatch.setattr("namo.solvability_runner.solve_environment_task", fake_solve)

    summary = run_exact_n_solvability(
        repo_root=tmp_path,
        input_dir=str(tmp_path),
        manifest_path=None,
        path_length=2,
        output_dir=str(tmp_path / "out"),
        config_file=str(config_path),
        primitive_prefix=CANONICAL_PRIMITIVE_PREFIX,
        seed=42,
        shuffle_seed=7000,
        workers=1,
        measurement={"record_statistics": True, "record_timing": False},
    )

    out_dir = tmp_path / "out"
    selected_envs = (out_dir / "selected_envs.txt").read_text(encoding="utf-8").splitlines()
    solved_rows = [
        json.loads(line)
        for line in (out_dir / "solved.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unsolved_rows = [
        json.loads(line)
        for line in (out_dir / "unsolved.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary_json = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    run_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))

    assert selected_envs == [xml_paths[0], xml_paths[3]]
    assert [row["xml_path"] for row in solved_rows] == [xml_paths[0]]
    assert {row["xml_path"] for row in unsolved_rows} == {xml_paths[1], xml_paths[3]}
    assert summary == summary_json
    assert summary_json["selected_env_count"] == 2
    assert summary_json["solved_count"] == 1
    assert summary_json["selection_error_count"] == 1
    assert summary_json["planner_failure_count"] == 1
    assert run_config["primitive_prefix"] == CANONICAL_PRIMITIVE_PREFIX
    assert run_config["goal_strategy"] == "random_rollout"
    assert run_config["seed"] == 42
    assert run_config["shuffle_seed"] == 7000
    assert run_config["measurement"] == {"schema_version": 1, "record_statistics": True, "record_timing": False}


@pytest.mark.parametrize("mode,solved", [("search", True), ("greedy_dfs", False)])
def test_timed_full_problem_keeps_failed_costs_and_execution_mode(tmp_path, monkeypatch, mode, solved):
    from namo import solvability_runner as runner
    from namo.core import PlannerResult
    from namo.planners import search_measurements

    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco><worldbody/></mujoco>")
    task = runner.SolveTask(
        xml_path=str(scene), path_length_n=2, config_path=CANONICAL_CONFIG,
        goal_strategy="scorer", region_max_chain_depth=2, primitive_data_dir="data",
        primitive_prefix=CANONICAL_PRIMITIVE_PREFIX, rollout_samples_per_state=None,
        region_frontier_beam_width=None, region_success_min_reachable=20,
        goals_per_region=100, seed=42, use_cpp_snapshot=True, simulation_budget=20000,
        local_search="best_first", best_first_prior="uniform", shuffle_seed=7000,
        goal_clearance=True, exec_mode=mode, record_timing=True,
    )
    env = SimpleNamespace(get_full_state=lambda: SimpleNamespace(qpos=[], qvel=[]),
                          is_robot_goal_reachable=lambda: solved)
    monkeypatch.setattr(runner.namo_rl, "RLEnvironment", lambda *_a: env)
    monkeypatch.setattr(runner, "extract_goal_from_xml", lambda *_a: (0, 0, 0))
    clock = iter((100.0, 103.0))
    monkeypatch.setattr(search_measurements, "perf_counter", lambda: next(clock))

    class Planner:
        def __init__(self, _env, config):
            assert config.algorithm_params["full_namo_exec_mode"] == mode
            assert config.algorithm_params["full_namo_goal_clearance"] is True
            self.timing = config.algorithm_params["full_namo_timing"]

        def search(self, _goal):
            self.timing.update(t_sim=1.25, t_score=0.5, n_score=2)
            return PlannerResult(success=solved, solution_found=solved, action_sequence=[],
                algorithm_stats={"simulation_budget_used": 7,
                                 "failure_kind": None if solved else "region_path_exhausted"})

    monkeypatch.setattr(runner, "FullNAMOPlanner", Planner)
    result = runner.solve_environment_task(task)
    assert result["row"].get("failure_kind") != "runner_exception", result["row"]
    assert result["kind"] == ("solved" if solved else "unsolved")
    row = result["row"]
    assert (row["total_calls"], row["t_wall"], row["t_sim"], row["t_score"], row["n_score"]) == (7, 3.0, 1.25, 0.5, 2)
    assert row["calls_until_success"] == (7 if solved else None)
    assert row["time_until_success"] == (3.0 if solved else None)
    assert row["censored"] is (not solved)
    assert row["final_goal_reachable"] is solved


def test_greedy_timing_accumulates_without_changing_commits(monkeypatch):
    from namo.planners.opening import best_first_search as search

    candidate = SimpleNamespace(x=0, y=0, theta=0, edge_idx=1, depth=0)
    monkeypatch.setattr(search, "candidates", lambda *_a, **_k: ([("box", candidate, 1.0)], 0, None))
    monkeypatch.setattr(search, "make_action", lambda *_a: candidate)
    clock = iter((0, 1, 1, 3, 3, 3, 10, 11, 11, 13, 13, 13))
    monkeypatch.setattr("namo.planners.search_measurements.perf_counter", lambda: next(clock))
    env = SimpleNamespace(set_full_state=lambda _s: None, step=lambda _a: None,
                          get_full_state=lambda: "moved")
    timing = {"t_score": 5.0, "t_sim": 7.0, "n_score": 3}
    for _ in range(2):
        result = search.run_greedy_commit(None, env, None, "scene.xml", "start",
            2, 20000, "model", "mean5", "q", None, is_open=lambda _e: True,
            dedupe_noop=False, timing=timing)
        assert result.simulations_used == 1 and result.opened
        assert result.resulting_state == "moved"
    assert timing == {"t_score": 7.0, "t_sim": 11.0, "n_score": 5, "t_local_verify": 0.0}


def test_region_openings_share_cumulative_budget_and_timing(monkeypatch):
    from namo.core import PlannerConfig
    from namo.planners.opening import best_first_region_opening as opening
    from namo.planners.utils import PushAttemptBudget

    class Env:
        def __init__(self):
            self.state = "baseline"

        def get_full_state(self):
            return self.state

        def set_full_state(self, state):
            self.state = state

        def count_reachable_points(self, _points):
            return 0, -1

    env = Env()
    timing = {"t_score": 1.0, "t_sim": 2.0, "n_score": 3}
    budget = PushAttemptBudget(limit=20000)
    planner = opening.BestFirstRegionOpeningPlanner(
        env,
        PlannerConfig(algorithm_params={
            "best_first_prior": "uniform",
            "full_namo_timing": timing,
            "push_budget": budget,
        }),
    )
    goal = SimpleNamespace(x=0.0, y=0.0, theta=0.0, edge_idx=1, depth=0)
    snapshot = {
        "robot_label": "robot",
        "region_labels": {},
        "adjacency": {"robot": {"a", "b"}},
        "region_goals": {
            name: SimpleNamespace(goals=[SimpleNamespace(x=1.0, y=2.0, theta=0.0)])
            for name in ("a", "b")
        },
        "edge_objects": {"robot": {"a": ["door_a"], "b": ["door_b"]}},
    }
    monkeypatch.setattr("namo.planners.get_region_snapshot", lambda *_a, **_k: snapshot)
    remaining_budgets = []
    timing_ids = []
    simulations = iter((3, 5))

    def fake_solve_scene(*args, timing, solution_out, **_kwargs):
        used = next(simulations)
        remaining_budgets.append(args[6])
        timing_ids.append(id(timing))
        timing["t_score"] += 0.25
        timing["t_sim"] += used / 2
        timing["n_score"] += 1
        solution_out.update(plan=[("door", goal)], state=f"state-{used}")
        return True, used, 1, [], "solved"

    monkeypatch.setattr(opening, "solve_scene", fake_solve_scene)
    first = planner.search((0.0, 0.0, 0.0), target_neighbor="a")
    second = planner.search((0.0, 0.0, 0.0), target_neighbor="b")

    assert first.success and second.success
    assert remaining_budgets == [20000, 19997]
    assert all(local_id != id(timing) for local_id in timing_ids)
    assert budget.used == 8
    assert {key: timing[key] for key in ("t_score", "t_sim", "n_score")} == {"t_score": 1.5, "t_sim": 6.0, "n_score": 5}
    assert timing["t_local_search"] >= 0


def test_walltime_summary_includes_failures_and_preserves_shard_membership():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts/pipeline/eval_full_namo_walltime.py"
    spec = importlib.util.spec_from_file_location("full_walltime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [dict(solved=True, total_calls=4, t_wall=0.5),
            dict(solved=False, total_calls=2, t_wall=0.2),
            dict(solved=False, total_calls=20000, t_wall=1000)]
    summary = module.summarize(rows)
    assert summary["success_rate"] == 1 / 3
    assert summary["median_calls_until_success"] is None
    assert summary["median_time_until_success"] is None
    assert summary["solve_at_1s"] == 1 / 3
    assert summary["median_consumed_calls"] == 4
    scenes = list(range(11))
    shards = [module.shard_rows(scenes, i, 4) for i in range(4)]
    assert sorted(item for shard in shards for item in shard) == scenes


@pytest.mark.parametrize("statistics,timing", [(False, False), (False, True), (True, False), (True, True)])
def test_independent_measurement_modes_keep_failed_outcomes(tmp_path, monkeypatch, statistics, timing):
    from namo import solvability_runner as runner
    from namo.core import PlannerResult

    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco><worldbody/></mujoco>")
    task = runner.SolveTask(
        xml_path=str(scene), path_length_n=2, config_path=CANONICAL_CONFIG,
        goal_strategy="scorer", region_max_chain_depth=2, primitive_data_dir="data",
        primitive_prefix=CANONICAL_PRIMITIVE_PREFIX, rollout_samples_per_state=None,
        region_frontier_beam_width=None, region_success_min_reachable=20,
        goals_per_region=100, seed=42, use_cpp_snapshot=True, simulation_budget=9000,
        local_search="best_first", best_first_prior="uniform", shuffle_seed=7000,
        goal_clearance=False, record_statistics=statistics, record_timing=timing,
    )
    state = SimpleNamespace(qpos=[0.1, 0.2], qvel=[0.0, 0.0])
    env = SimpleNamespace(get_full_state=lambda: state, is_robot_goal_reachable=lambda: False,
                          get_observation=lambda: {"robot_pose": [0.1, 0.2, 0.0]})
    monkeypatch.setattr(runner.namo_rl, "RLEnvironment", lambda *_a: env)
    monkeypatch.setattr(runner, "extract_goal_from_xml", lambda *_a: (0.0, 0.0, 0.0))

    class Planner:
        def __init__(self, _env, config):
            self.measurement = config.algorithm_params["search_measurements"]

        def search(self, _goal):
            self.measurement.event("attempt_result", local_end="exhausted", calls=7)
            self.measurement.append("attempts", {"attempt_id": 0, "local_end": "exhausted"})
            return PlannerResult(success=False, solution_found=False, action_sequence=[],
                algorithm_stats={"simulation_budget_used": 7, "failure_kind": "region_path_exhausted",
                                 "iteration_trace": [{"outcome": "boundary_exhausted"}]})

    monkeypatch.setattr(runner, "FullNAMOPlanner", Planner)
    row = runner.solve_environment_task(task)["row"]
    assert row["total_calls"] == 7 and row["solved"] is False
    assert row["calls_until_success"] is None and row["complete"] is True
    assert row["failure_kind"] == "region_path_exhausted"
    assert all(row[key] for key in ("problem_id", "run_id", "semantic_protocol_hash",
                                   "execution_digest", "terminal_state_digest", "runtime_fingerprints"))
    assert ("statistics" in row) is statistics
    assert ("iteration_trace" in row) is statistics
    assert ("terminal_state" in row) is statistics
    assert ("t_wall" in row) is timing
    assert ("time_until_success" in row) is timing
    if timing:
        assert row["time_until_success"] is None
