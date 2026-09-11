"""Focused contracts for independent keyhole-horizon certification."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))

import certify_keyhole_horizons as certifier  # noqa: E402


OBJECT_ID = "obstacle_0_movable"
REACHABLE_EDGES = (3, 7)
TARGET_POINTS = [(float(index), 0.0) for index in range(100)]
WITNESS = ((3, 1), (7, 4))


class _FakeAction:
    object_id = ""
    edge_idx = -1
    depth = -1
    x = 0.0
    y = 0.0
    theta = 0.0


class _FakeEnv:
    def __init__(self, outcomes=None, *, final_goal=True):
        self.outcomes = outcomes or {}
        self.path = ()
        self.final_goal = final_goal
        self.restored_states = []

    def get_full_state(self):
        return self.path

    def set_full_state(self, state):
        self.path = tuple(state)
        self.restored_states.append(self.path)

    def get_reachable_objects(self):
        return [OBJECT_ID]

    def get_reachable_edges(self, object_id):
        assert object_id == OBJECT_ID
        return list(REACHABLE_EDGES)

    def step(self, action):
        key = self.path + ((int(action.edge_idx), int(action.depth)),)
        self.path = key
        outcome = self.outcomes.get(key, {})
        return SimpleNamespace(
            done=outcome.get("done", True),
            info=outcome.get("info", {}),
        )

    def count_reachable_points(self, _points):
        return self.outcomes.get(self.path, {}).get("point_count", 0), -1

    def is_robot_goal_reachable(self):
        return self.final_goal and len(self.path) == 4


@pytest.fixture(autouse=True)
def _fake_action(monkeypatch):
    monkeypatch.setattr(certifier.namo_rl, "Action", _FakeAction)


def test_gate_two_requires_exhaustive_onepush_miss_and_closed_open_witness():
    outcomes = {
        (WITNESS[0],): {"point_count": 0},
        WITNESS: {"point_count": 100},
    }
    env = _FakeEnv(outcomes)

    result = certifier.certify_gate(env, OBJECT_ID, TARGET_POINTS, WITNESS)

    assert result["tried_onepush_count"] == len(REACHABLE_EDGES) * certifier.CANONICAL_DEPTHS
    assert result["valid_onepush_count"] == 0
    assert result["witness"]["opened"] == [False, True]
    assert result["label"] == "2"
    assert env.restored_states[: len(REACHABLE_EDGES) * certifier.CANONICAL_DEPTHS] == [
        ()
    ] * (len(REACHABLE_EDGES) * certifier.CANONICAL_DEPTHS)


def test_failed_push_cannot_be_a_onepush_opener():
    outcomes = {
        ((3, 1),): {
            "done": False,
            "point_count": 100,
            "info": {
                "wall_collision": "true",
                "failure_reason": "Robot collision during push with static object: walls",
            },
        },
        ((7, 0),): {"point_count": 100},
    }
    env = _FakeEnv(outcomes)

    result = certifier.certify_gate(env, OBJECT_ID, TARGET_POINTS, WITNESS)

    assert [3, 1] not in result["valid_onepush"]
    assert [7, 0] in result["valid_onepush"]
    assert result["label"] == "1"


@pytest.mark.parametrize(
    ("labels", "bucket"),
    [
        (("1", "1"), "reject"),
        (("1", "2"), "12"),
        (("2", "1"), "21"),
        (("2", "2"), "22"),
        (("2", "?"), "2?"),
        (("?", "2"), "?2"),
        (("1", "?"), "reject"),
    ],
)
def test_scene_retention_requires_at_least_one_certified_two(labels, bucket):
    assert certifier.classify_scene(*labels) == bucket


def test_complete_witness_success_depends_only_on_end_goal_reachability():
    actions = (((1, 0), (3, 1)), ((5, 2), (7, 4)))
    env = _FakeEnv(final_goal=True)

    result = certifier.replay_complete_witness(
        env,
        ("obstacle_0_movable", "obstacle_1_movable"),
        actions,
    )

    assert result["success"] is True
    assert result["end_goal_reachable"] is True
    assert len(result["actions"]) == 4


def test_missing_witness_is_searched_only_after_complete_onepush_sweep():
    env = _FakeEnv({WITNESS: {"point_count": 100}})
    calls = []

    def discover():
        assert len(env.restored_states) >= len(REACHABLE_EDGES) * 5
        assert env.path == ()
        calls.append(True)
        return {"actions": WITNESS, "simulations": 12, "end": "solved"}

    result = certifier.certify_gate(
        env, OBJECT_ID, TARGET_POINTS, (), discover_witness=discover,
    )
    assert calls == [True]
    assert result["label"] == "2"
    assert result["witness_search"]["simulations"] == 12


def test_generated_k2_removes_only_first_boundary_objects(tmp_path):
    source = tmp_path / "scene.xml"
    original = ('<mujoco><worldbody><body name="walls"/><body name="car"/>'
                '<body name="gate_a"/><body name="gate_b"/>'
                '<body name="gate_c"/><body name="context"/>'
                '</worldbody></mujoco>')
    source.write_text(original)
    output = tmp_path / "k2.xml"
    certifier.write_removed_blockers_xml(source, ["gate_a", "gate_b"], output)
    root = certifier.ET.parse(output).getroot()
    assert {b.get("name") for b in root.findall("./worldbody/body")} == {
        "walls", "car", "gate_c", "context",
    }
    assert source.read_text() == original


def test_boundary_two_label_cannot_ignore_an_alternative_onepush_opener():
    assert certifier.boundary_label([{"label": "2"}, {"label": "1"}]) == "1"
    assert certifier.boundary_label([{"label": "2"}, {"label": "?"}]) == "2"
    assert certifier.boundary_label([{"label": "?"}]) == "?"


def test_generated_groups_keep_11_and_partial_certified_twos():
    assert certifier.generated_group("11") == "11"
    for pattern in ("12", "21", "22", "2?", "?2"):
        assert certifier.generated_group(pattern) == "has_2push"
    for pattern in ("1?", "?1", "??"):
        assert certifier.generated_group(pattern) is None


def test_generated_task_cli_separates_discovery_from_old_witness_tasks():
    args = certifier.parser().parse_args([
        "prepare-generated", "--source", "/tmp/generated", "--output", "/tmp/tasks.jsonl",
        "--witness-budget", "4000",
    ])
    assert args.witness_budget == 4000


def test_witness_search_cannot_continue_from_a_failed_setup():
    inner = _FakeEnv({((3, 1),): {"done": False, "point_count": 100}})
    env = certifier._CompletedPushEnv(inner)
    result = env.step(certifier._action(OBJECT_ID, 3, 1))
    assert not result.done and not env.completed
    assert inner.path == ()
