import json
from pathlib import Path

from namo.environment_selection import RegionPathAnalysis
from namo.solvability_runner import run_exact_n_solvability
from namo.runtime_profile import CANONICAL_CONFIG, CANONICAL_PRIMITIVE_PREFIX


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
