import json
import os
from pathlib import Path

import pytest


def _write_temp_config(stem: str, text: str) -> Path:
    path = Path("/tmp") / f"{stem}.yaml"
    path.write_text(text)
    return path


def _build_config_with_overrides(overrides: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    base_cfg = repo_root / "config" / "namo_config_car.yaml"
    text = base_cfg.read_text()
    if overrides:
        text += "\n" + overrides.strip() + "\n"
    return _write_temp_config("namo_failure_diag_test_cfg", text)


def _make_env(config_path: Path):
    try:
        import namo_rl  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"namo_rl not importable: {e}")

    repo_root = Path(__file__).resolve().parents[2]
    xml_path = repo_root / "test_xml" / "little-car-modeling-package" / "artifacts" / "nav_env.xml"
    if not xml_path.exists():
        pytest.skip(f"Missing XML fixture: {xml_path}")
    if not config_path.exists():
        pytest.skip(f"Missing config: {config_path}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    prev_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        env = namo_rl.RLEnvironment(str(xml_path), str(config_path), visualize=False)
        env.reset()
        return env
    finally:
        os.chdir(prev_cwd)


def _failing_invalid_edge_step(env):
    import namo_rl  # type: ignore

    obs = env.get_observation()
    pose = obs["obstacle_1_movable_pose"]
    action = namo_rl.Action()
    action.object_id = "obstacle_1_movable"
    action.x = float(pose[0] + 0.2)
    action.y = float(pose[1])
    action.theta = float(pose[2])
    action.edge_idx = 9999
    action.depth = 0
    return env.step(action)


def test_failure_diag_invalid_edge_precheck():
    cfg_path = _build_config_with_overrides("")
    env = _make_env(cfg_path)

    step_result = _failing_invalid_edge_step(env)
    assert not step_result.done
    assert step_result.info["failure_code"] == "requested_edge_not_reachable"
    assert "failure_diag_json" in step_result.info
    diag = json.loads(step_result.info["failure_diag_json"])
    assert step_result.info["failure_reason"] == diag["summary"]


def test_failure_diag_forced_navigation_timeout():
    cfg_path = _build_config_with_overrides(
        """
navigation:
  diff_drive:
    max_nav_steps: 1
"""
    )
    env = _make_env(cfg_path)

    import namo_rl  # type: ignore

    obs = env.get_observation()
    pose = obs["obstacle_1_movable_pose"]
    reachable_edges = env.get_reachable_edges("obstacle_1_movable")
    assert reachable_edges, "Expected at least one reachable edge"

    action = namo_rl.Action()
    action.object_id = "obstacle_1_movable"
    action.x = float(pose[0] + 0.6)
    action.y = float(pose[1])
    action.theta = float(pose[2])
    action.edge_idx = int(reachable_edges[0])
    action.depth = 0

    step_result = env.step(action)
    assert not step_result.done
    assert "Primitive step" not in step_result.info["failure_reason"]
    assert step_result.info["failure_code"] == "navigation_failed"
    assert "timeout" in step_result.info.get("failure_nav_reason", "").lower()
    assert step_result.info.get("failure_step_index") == "1"


def test_failure_trace_gating():
    env_no_trace = _make_env(_build_config_with_overrides(""))
    step_result_no_trace = _failing_invalid_edge_step(env_no_trace)
    assert "failure_trace_json" not in step_result_no_trace.info

    env_trace = _make_env(
        _build_config_with_overrides(
            """
skill:
  emit_failure_trace: true
  failure_trace_max_events: 4
"""
        )
    )
    step_result_trace = _failing_invalid_edge_step(env_trace)
    assert "failure_trace_json" in step_result_trace.info
    trace_diag = json.loads(step_result_trace.info["failure_trace_json"])
    assert "trace_events" in trace_diag
    assert len(trace_diag["trace_events"]) > 0
    assert len(trace_diag["trace_events"]) <= 4
