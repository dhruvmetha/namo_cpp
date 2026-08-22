from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))

import compose_keyhole_modules as composer  # noqa: E402


K1 = "obstacle_0_movable"
K2 = "obstacle_1_movable"


def _donor(index: int, edge: int) -> composer.Donor:
    return composer.Donor(
        xml_path=f"/donor_{index}.xml",
        object_id=f"source_{index}",
        region=f"region_{index}",
        object_center=(float(index), 0.0),
        object_theta=0.0,
        tier="easy",
        horizon="1push",
        template="set2/benchmark_5",
        valid_root=((edge, 0),),
    )


def _state(*, goal: bool, k1: bool, k2: bool, k1_edges=(10,), k2_edges=(20,), counts=(0, 0)):
    objects = [object_id for object_id, reachable in ((K1, k1), (K2, k2)) if reachable]
    return {
        "goal": goal,
        "objects": objects,
        "edges": {K1: list(k1_edges), K2: list(k2_edges)},
        "counts": list(counts),
    }


class _FakeEnv:
    def __init__(self, states, *, k1_done=True, k2_done=True):
        self.states = states
        self.current = "initial"
        self.k1_done = k1_done
        self.k2_done = k2_done

    def set_robot_goal(self, *_goal):
        return None

    def get_full_state(self):
        return self.current

    def set_full_state(self, state):
        self.current = state

    def get_reachable_objects(self):
        return self.states[self.current]["objects"]

    def get_reachable_edges(self, object_id):
        return self.states[self.current]["edges"][object_id]

    def is_robot_goal_reachable(self):
        return self.states[self.current]["goal"]

    def get_observation(self):
        offset = {"initial": 0.0, "post_k1": 1.0, "post_k2": 2.0}[self.current]
        return {
            f"{K1}_pose": [offset, 0.0, 0.0],
            f"{K2}_pose": [offset, 1.0, 0.0],
        }

    def count_reachable_points(self, points):
        index = 0 if points[0][0] < 15.0 else 1
        return self.states[self.current]["counts"][index], -1

    def step(self, action):
        if action.object_id == K1:
            self.current = "post_k1"
            return SimpleNamespace(done=self.k1_done)
        if action.object_id == K2:
            self.current = "post_k2"
            return SimpleNamespace(done=self.k2_done)
        raise AssertionError(f"unexpected object {action.object_id}")


def _snapshot():
    def goals(x):
        return SimpleNamespace(goals=[SimpleNamespace(x=x, y=float(index)) for index in range(100)])

    return {
        "adjacency": {"robot": {"middle"}, "middle": {"robot", "goal"}, "goal": {"middle"}},
        "edge_objects": {
            "robot": {"middle": {K1}},
            "middle": {"robot": {K1}, "goal": {K2}},
            "goal": {"middle": {K2}},
        },
        "robot_label": "robot",
        "goal_label": "goal",
        "goal_in_free_space": True,
        "region_goals": {"middle": goals(10.0), "goal": goals(20.0)},
    }


def _run(monkeypatch, states, *, k1_done=True, k2_done=True):
    env = _FakeEnv(states, k1_done=k1_done, k2_done=k2_done)
    monkeypatch.setattr(composer.namo_rl, "RLEnvironment", lambda *_args: env)
    monkeypatch.setattr(composer, "extract_goal_from_xml", lambda _xml: (1.0, 2.0, 0.0))
    monkeypatch.setattr(composer, "get_region_snapshot", lambda *_args, **_kwargs: _snapshot())
    return composer.replay_two_keyhole_goal_chain(
        "/composed.xml", "/config.yaml", [_donor(0, 10), _donor(1, 20)]
    )


def _valid_states(counts=(0, 0)):
    return {
        "initial": _state(goal=False, k1=True, k2=False, counts=counts),
        "post_k1": _state(goal=False, k1=True, k2=True, counts=counts),
        "post_k2": _state(goal=True, k1=True, k2=True, counts=counts),
    }


def test_goal_chain_accepts_valid_progression_and_records_contract(monkeypatch):
    result = _run(monkeypatch, _valid_states())

    assert result["status"] == "solved"
    assert result["failure_reason"] is None
    assert result["component_path"] == ["robot", "middle", "goal"]
    assert result["boundary_objects"] == [[K1], [K2]]
    assert [state["goal_reachable"] for state in result["reachability_trace"]] == [False, False, True]
    assert K2 not in result["reachability_trace"][0]["reachable_objects"]
    assert result["reachability_trace"][1]["reachable_edges"][K2] == [20]
    assert result["actions"] == [[[10, 0]], [[20, 0]]]
    assert result["final_object_poses"] == result["object_pose_trace"][-1]


@pytest.mark.parametrize(
    ("states", "k1_done", "k2_done", "reason"),
    [
        (
            {
                **_valid_states(),
                "initial": _state(goal=False, k1=True, k2=True),
            },
            True,
            True,
            "k2_reachable_at_t0",
        ),
        (
            {
                **_valid_states(),
                "post_k1": _state(goal=True, k1=True, k2=True),
            },
            True,
            True,
            "k1_reached_goal",
        ),
        (_valid_states(), False, True, "k1_push_failed"),
        (
            {
                **_valid_states(),
                "post_k1": _state(goal=False, k1=True, k2=False),
            },
            True,
            True,
            "k1_did_not_expose_k2",
        ),
        (
            {
                **_valid_states(),
                "post_k1": _state(goal=False, k1=True, k2=True, k2_edges=()),
            },
            True,
            True,
            "k2_no_push_edges_after_k1",
        ),
        (_valid_states(), True, False, "k2_push_failed"),
        (
            {
                **_valid_states(),
                "post_k2": _state(goal=False, k1=True, k2=True),
            },
            True,
            True,
            "final_goal_unreachable",
        ),
    ],
)
def test_goal_chain_reports_specific_progression_failure(
    monkeypatch, states, k1_done, k2_done, reason
):
    result = _run(monkeypatch, states, k1_done=k1_done, k2_done=k2_done)

    assert result["status"] != "solved"
    assert result["failure_reason"] == reason


def test_diagnostic_point_counts_do_not_gate_valid_goal_progression(monkeypatch):
    result = _run(monkeypatch, _valid_states(counts=(0, 0)))

    assert result["status"] == "solved"
    assert result["target_point_thresholds"] == [20, 20]
    assert result["target_point_trace"] == [[0, 0], [0, 0], [0, 0]]


@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        ({"goal_in_free_space": False}, "goal_not_in_free_space"),
        ({"no_path": True}, "no_component_path"),
        ({"hop_mismatch": True}, "wrong_hop_count"),
        ({"no_reachable_blocker": True}, "k1_not_reachable"),
        ({"no_pushable_blocker": True}, "k1_no_push_edges"),
    ],
)
def test_static_acceptance_reports_specific_defect(defect, reason):
    row = {
        "goal_in_free_space": True,
        "boundaries": [{"objects": [K1]}, {"objects": [K2]}],
        **defect,
    }

    assert composer.static_acceptance(row, 2) == (False, reason)
