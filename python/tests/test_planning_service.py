"""Contract tests for the external NAMO planning facade."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from namo.runtime_profile import CANONICAL_CONFIG, CANONICAL_PRIMITIVE_PREFIX


def _write_canonical_config(path):
    path.write_text(Path(CANONICAL_CONFIG).read_text(encoding="utf-8"), encoding="utf-8")


def test_service_package_exports_public_contract():
    from namo.services import NAMOAction, NAMOPlanResult, NAMOPlanningService

    assert NAMOAction.__name__ == "NAMOAction"
    assert NAMOPlanResult.__name__ == "NAMOPlanResult"
    assert NAMOPlanningService.__name__ == "NAMOPlanningService"


def test_registered_planners_import_without_dataset_environment():
    env = os.environ.copy()
    env.pop("NAMO_SCRATCH", None)

    completed = subprocess.run(
        [sys.executable, "-c", "import namo.planners"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_plan_from_xml_returns_actions_and_statistics(monkeypatch, tmp_path):
    from namo.services import NAMOPlanningService
    from namo.services import planning_service as module

    events = []

    class FakeEnvironment:
        def __init__(self, xml_path, config_path, enable_viewer, defer_warmup):
            events.append(
                ("environment", xml_path, config_path, enable_viewer, defer_warmup)
            )

        def set_robot_pose(self, x, y, theta):
            events.append(("pose", x, y, theta))

        def warm_up(self):
            events.append(("warm_up",))

        def set_robot_goal(self, x, y, theta):
            events.append(("goal", x, y, theta))

    valid_action = SimpleNamespace(object_id="obstacle_1", edge_idx=8, depth=2)
    invalid_action = SimpleNamespace(object_id="navigation", edge_idx=-1, depth=-1)

    class FakePlanner:
        def search(self, robot_goal):
            events.append(("search", robot_goal))
            return SimpleNamespace(
                success=True,
                action_sequence=[valid_action, invalid_action],
                algorithm_stats={"total_primitives_attempted": 7},
            )

    def create_planner(name, env, config):
        events.append(("planner", name, env, config))
        return FakePlanner()

    monkeypatch.setattr(module.namo_rl, "RLEnvironment", FakeEnvironment)
    monkeypatch.setattr(module, "_create_planner", create_planner)

    config_path = tmp_path / "namo.yaml"
    _write_canonical_config(config_path)
    service = NAMOPlanningService(str(config_path), primitive_data_dir=str(tmp_path))

    result = service.plan_from_xml(
        xml_path="scene.xml",
        robot_goal=(1.0, 2.0, 0.5),
        algorithm="full_namo",
        starting_robot_pose=(0.1, 0.2, 0.3),
        primitive_prefix=CANONICAL_PRIMITIVE_PREFIX,
    )

    assert result.success is True
    assert [(a.object_id, a.edge_idx, a.depth) for a in result.actions] == [
        ("obstacle_1", 8, 2)
    ]
    assert result.algorithm_stats == {"total_primitives_attempted": 7}
    assert events[:3] == [
        ("environment", "scene.xml", str(config_path), False, True),
        ("pose", 0.1, 0.2, 0.3),
        ("warm_up",),
    ]
    assert ("goal", 1.0, 2.0, 0.5) in events
    assert ("search", (1.0, 2.0, 0.5)) in events


def test_reachability_teleports_before_warmup(monkeypatch, tmp_path):
    from namo.services import NAMOPlanningService
    from namo.services import planning_service as module

    events = []

    class FakeEnvironment:
        def __init__(self, xml_path, config_path, enable_viewer, defer_warmup):
            events.append(("environment", defer_warmup))

        def set_robot_pose(self, x, y, theta):
            events.append(("pose", x, y, theta))

        def warm_up(self):
            events.append(("warm_up",))

        def set_robot_goal(self, x, y, theta):
            events.append(("goal", x, y, theta))

        def get_reachability_summary(self, analysis_mode):
            events.append(("summary", analysis_mode))
            return {"goal_reachable": True, "analysis_mode": analysis_mode}

    monkeypatch.setattr(module.namo_rl, "RLEnvironment", FakeEnvironment)

    config_path = tmp_path / "namo.yaml"
    _write_canonical_config(config_path)
    service = NAMOPlanningService(str(config_path))
    result = service.analyze_reachability_from_xml(
        xml_path="scene.xml",
        robot_goal=(2.0, 3.0, 0.0),
        analysis_mode=True,
        starting_robot_pose=(0.4, 0.5, 0.6),
    )

    assert result["goal_reachable"] is True
    assert result["compute_time_ms"] >= 0.0
    assert events == [
        ("environment", True),
        ("pose", 0.4, 0.5, 0.6),
        ("warm_up",),
        ("goal", 2.0, 3.0, 0.0),
        ("summary", True),
    ]


def test_plan_failure_is_returned_as_contextual_result(monkeypatch, tmp_path):
    from namo.services import NAMOPlanningService
    from namo.services import planning_service as module

    def fail_environment(*args, **kwargs):
        raise RuntimeError("bad scene")

    monkeypatch.setattr(module.namo_rl, "RLEnvironment", fail_environment)

    config_path = tmp_path / "namo.yaml"
    _write_canonical_config(config_path)
    service = NAMOPlanningService(str(config_path))
    result = service.plan_from_xml("broken.xml", (1.0, 2.0, 0.0))

    assert result.success is False
    assert result.actions == []
    assert "bad scene" in result.error_message
    assert result.search_time_ms >= 0.0
