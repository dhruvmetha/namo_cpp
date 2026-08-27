from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from full_namo_sim_exp import runner


def test_runner_times_only_complete_full_namo_search(monkeypatch) -> None:
    class FakePlanner:
        def __init__(self, env, config):
            assert env == "environment"
            assert config == "planner-config"

        def search(self, goal):
            assert goal == "goal"
            return SimpleNamespace(
                success=True,
                algorithm_stats={"simulation_budget_used_total": 7},
                action_sequence=[],
                error_message=None,
            )

    task = SimpleNamespace(xml_path="scene.xml", config_path="config.yaml", path_length_n=2)
    monkeypatch.setattr(runner.base.namo_rl, "RLEnvironment", lambda *args: "environment")
    monkeypatch.setattr(
        runner.base,
        "build_full_namo_planner_config",
        lambda task: "planner-config",
    )
    monkeypatch.setattr(runner.base, "FullNAMOPlanner", FakePlanner)
    monkeypatch.setattr(runner.base, "extract_goal_from_xml", lambda path: "goal")
    times = iter((10.0, 10.25))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(times))

    result = runner.timed_solve_environment_task(task)

    assert result["kind"] == "solved"
    assert result["row"]["simulation_budget_used_total"] == 7
    assert result["row"]["search_time_ms"] == pytest.approx(250.0)
    assert result["row"]["timing_scope"] == "full_namo_planner_search"


def test_ordering_seed_does_not_change_evaluation_or_snapshot_seeds() -> None:
    config = SimpleNamespace(
        random_seed=42,
        algorithm_params={
            "region_snapshot_seed": 42,
            "ml_seed": 42,
            "shuffle_seed": 42,
        },
    )

    updated = runner.with_ordering_seed(config, 101)

    assert updated is config
    assert updated.random_seed == 42
    assert updated.algorithm_params["region_snapshot_seed"] == 42
    assert updated.algorithm_params["ml_seed"] == 42
    assert updated.algorithm_params["shuffle_seed"] == 101


def test_manifest_runner_executes_every_scene_without_path_length_filter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_tasks = []
    monkeypatch.setattr(runner.base, "_resolve_config_path", lambda *args: "config.yaml")
    monkeypatch.setattr(runner.base, "require_canonical_runtime_config", lambda path: None)
    monkeypatch.setattr(runner.base, "derive_max_push_steps", lambda path: 5)
    monkeypatch.setattr(
        runner.base, "require_canonical_primitive_profile", lambda prefix, steps: None
    )
    monkeypatch.setattr(
        runner.base,
        "get_xml_files",
        lambda **kwargs: ["declared_n2.xml", "would_have_been_filtered.xml"],
    )
    monkeypatch.setattr(
        runner.base,
        "analyze_environment_path_length",
        lambda *args, **kwargs: pytest.fail("manifest runs must not prefilter scenes"),
    )

    def fake_results(tasks, workers):
        captured_tasks.extend(tasks)
        return iter(
            {
                "kind": "solved",
                "row": {
                    "xml_path": task.xml_path,
                    "path_length_n": task.path_length_n,
                    "solved": True,
                },
            }
            for task in tasks
        )

    monkeypatch.setattr(runner.base, "_iter_solve_results", fake_results)
    summary = runner.run_manifest_without_prefilter(
        repo_root=tmp_path,
        input_dir=None,
        manifest_path="population.txt",
        path_length=2,
        output_dir=str(tmp_path / "raw"),
        config_file="config.yaml",
        goal_strategy="scorer",
        region_max_chain_depth=2,
        primitive_data_dir=str(tmp_path / "data"),
        primitive_prefix="1x_car_d5_",
        rollout_samples_per_state=None,
        region_frontier_beam_width=None,
        region_success_min_reachable=20,
        goals_per_region=100,
        seed=42,
        use_cpp_snapshot=True,
        simulation_budget=900,
        simulation_budget_scope="keyhole",
        local_search="best_first",
        best_first_prior="uniform",
        region_selection_strategy="ml_first",
        scorer_ckpt="model.ckpt",
        ml_device="cpu",
        workers=1,
        full_namo_max_iterations=None,
        audit_next_keyhole_reachability=False,
        preserve_next_keyhole_access=False,
        limit=None,
    )

    assert [task.xml_path for task in captured_tasks] == [
        "declared_n2.xml",
        "would_have_been_filtered.xml",
    ]
    assert summary["input_env_count"] == 2
    assert summary["selected_env_count"] == 2
    assert summary["selection_error_count"] == 0
