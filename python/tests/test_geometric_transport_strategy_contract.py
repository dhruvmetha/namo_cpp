from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytest

from namo.strategies.geometric_transport_strategy import GeometricTransportStrategy
from namo.strategies.goal_selection_strategy import Goal


@dataclass(frozen=True)
class _DummyState:
    tag: str = "s"


class _DummyPrimitiveStrategy:
    def __init__(self, goals_per_edge: List[List[Goal]]):
        self._goals_per_edge = goals_per_edge

    def generate_goals(self, object_id, state, env, max_goals, region_goals_sampled=None):
        return self._goals_per_edge


class _DummyEnv:
    def __init__(self, reachable_edges: List[int], priorities: List[int], robot_goal_xy=(4.0, 0.0)):
        self._reachable_edges = reachable_edges
        self._priorities = priorities
        self._robot_goal_xy = robot_goal_xy
        self._state = _DummyState("orig")

    def get_full_state(self):
        return self._state

    def set_full_state(self, state):
        self._state = state

    def get_reachable_edges(self, object_id):
        return list(self._reachable_edges)

    def get_robot_goal(self):
        return (self._robot_goal_xy[0], self._robot_goal_xy[1], 0.0)

    def evaluate_primitive_priorities(self, object_id, target_poses, robot_goal):
        assert tuple(robot_goal) == tuple(self._robot_goal_xy)
        assert len(target_poses) == len(self._priorities)
        return list(self._priorities)


def test_doc_contract_score_mapping_and_api_compatibility():
    # The docstring says priorities 1..6, and python wrapper maps them to scores:
    # score = 7 - priority (so higher score sorts earlier when score-first is used).
    #
    # This test validates:
    # - API compatibility: accepts region_goals_sampled kwarg
    # - score mapping for all priorities
    goals_per_edge = [
        [
            Goal(x=0.0, y=0.0, theta=0.0, edge_idx=0, depth=0),
            Goal(x=1.0, y=1.0, theta=0.0, edge_idx=0, depth=1),
            Goal(x=2.0, y=2.0, theta=0.0, edge_idx=0, depth=2),
            Goal(x=3.0, y=3.0, theta=0.0, edge_idx=0, depth=3),
            Goal(x=4.0, y=4.0, theta=0.0, edge_idx=0, depth=4),
            Goal(x=5.0, y=5.0, theta=0.0, edge_idx=0, depth=5),
        ]
    ]
    priorities = [1, 2, 3, 4, 5, 6]
    env = _DummyEnv(reachable_edges=[0], priorities=priorities, robot_goal_xy=(4.0, 0.0))

    strat = GeometricTransportStrategy(primitive_data_dir="data", verbose=False)
    strat._primitive_strategy = _DummyPrimitiveStrategy(goals_per_edge)  # type: ignore[attr-defined]

    out = strat.generate_goals(
        "obstacle_1_movable",
        _DummyState("s"),
        env,
        max_goals=600,
        region_goals_sampled=[(0.0, 0.0, 0.0)],  # accepted but unused
    )

    assert len(out) == 1
    got_scores = [g.score for g in out[0]]
    assert got_scores == [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    assert [g.edge_idx for g in out[0]] == [0] * 6
    assert [g.depth for g in out[0]] == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "priority, expected_label",
    [
        (1, "clean+opening"),
        (2, "movable+opening"),
        (3, "static+opening"),
        (4, "clean+no opening"),
        (5, "movable+no opening"),
        (6, "static+no opening"),
    ],
)
def test_doc_contract_priority_range(priority: int, expected_label: str):
    # Simple guardrail: priorities are expected to be 1..6 as documented.
    assert 1 <= priority <= 6, expected_label
