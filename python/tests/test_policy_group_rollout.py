"""A policy commits moving states and keeps jam feedback local to an object."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("first_jams", [False, True])
def test_group_policy_switches_objects_without_undo(monkeypatch, tmp_path, first_jams):
    points = [[0.123456789123, 0.9]]

    class Env:
        state = 0
        steps = []

        def set_robot_goal(self, *args):
            pass

        def get_reachable_objects(self):
            return ["a", "b"]

        def get_full_state(self):
            return self.state

        def set_full_state(self, state):
            assert state == self.state, "Policy undid an executed moving push"

        def get_observation(self):
            return {"state": self.state}

        def step(self, action):
            self.steps.append(action.object_id)
            if action.object_id == "a" and first_jams:
                return SimpleNamespace(info={"failure_reason": "jam"})
            self.state += 1
            return SimpleNamespace(info={})

    env = Env()
    goal = SimpleNamespace(x=0.0, y=0.0, theta=0.0, edge_idx=3, depth=0)

    def rank(*args, restrict_obj, region_samples, **kwargs):
        assert restrict_obj == ["a", "b"]
        assert region_samples == points
        return [("a", goal, 2), ("b", goal, 1)] if env.state == 0 else [("b", goal, 2)]

    def is_open(environment, target):
        assert target == points
        return environment.state == (1 if first_jams else 2)

    fake_modules = {
        "scorer_beam": dict(BeamPlanner=lambda **kw: None, make_env=lambda _: env,
                            make_action=lambda obj, g: SimpleNamespace(object_id=obj), FALLBACK_GOAL=(0, 0, 0)),
        "eval_m3": dict(rank_first_pushes_h2=rank, sample_goal_points=None, goal_open_pts=is_open),
        "namo.planners.opening.best_first_search": dict(
            _unmoved=lambda a, b, obj: a == b, solve_scene=None),
    }
    for name, attrs in fake_modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "scripts/rl_loop/eval_policy.py"
    spec = importlib.util.spec_from_file_location("policy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "extract_goal_with_fallback", lambda *a: (0, 0, 0))
    cohort = tmp_path / "groups.jsonl"
    cohort.write_text(json.dumps({"eligible": True, "room_realpath": "/fake/room.xml",
                                  "group_id": "test", "boundary_objects": ["a", "b"],
                                  "goal_label": "goal", "target_points": points}) + "\n")
    out = tmp_path / "result.json"
    leaf = tmp_path / "result.jsonl"
    monkeypatch.setattr(sys, "argv", [str(path), "--ckpt", "fake", "--groups", str(cohort),
                                      "--out", str(out), "--leaf-out", str(leaf), "--max-pushes", "2"])
    module.main()
    result = json.loads(leaf.read_text())
    assert env.steps == ["a", "b"]
    assert result["opened_at"] == 2
    assert result["n_push"] == 2
    assert result["n_noop"] == int(first_jams)
