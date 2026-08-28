"""One-state greedy commits filter invalid actions without branching children."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from conftest import _require_real_namo_rl

_require_real_namo_rl()

from namo.planners.opening.best_first_search import run_greedy_commit  # noqa: E402


OBJ = "obstacle_1_movable"
EDGES = (0, 1, 2)
DEPTHS = (0, 1, 2)
GOAL_M = (0.37, 0.67, 0.0)
XML = "fake.xml"


class _Goal:
    def __init__(self, edge_idx, depth):
        self.x = self.y = self.theta = 0.0
        self.edge_idx = edge_idx
        self.depth = depth


class _Prim:
    def generate_goals(self, _obj, _state, _env, max_goals=0):
        return [[_Goal(edge, depth) for depth in DEPTHS] for edge in EDGES]


class _Scorer:
    def score_state(self, _env, _obj, _goal, _xml, region_samples=None, h=1, raw=False):
        grid = np.zeros((60, 5), dtype=float)
        for edge in EDGES:
            for depth in DEPTHS:
                grid[edge, depth] = 1.0 - (0.1 * edge + 0.01 * depth)
        return grid


class _Env:
    def __init__(self, moves_on=(), failure_reason=""):
        self.moves_on = set(moves_on)
        self.failure_reason = failure_reason
        self.stepped = []
        self._pose = [0.0, 0.0, 0.0]

    def set_full_state(self, state):
        self._pose = list(state["pose"])

    def get_full_state(self):
        return {"pose": list(self._pose)}

    def get_reachable_objects(self):
        return [OBJ]

    def get_reachable_edges(self, _obj):
        return list(EDGES)

    def get_observation(self):
        return {f"{OBJ}_pose": list(self._pose), "robot_pose": [0.0, 0.0, 0.0]}

    def step(self, action):
        key = (int(action.edge_idx), int(action.depth))
        self.stepped.append(key)
        if key in self.moves_on:
            self._pose[0] += 1.0
        info = {"failure_reason": self.failure_reason} if self.failure_reason else {}
        return SimpleNamespace(info=info)


def _planner():
    return SimpleNamespace(prim=_Prim(), scorer=_Scorer())


def _run(env, *, sim_budget=20):
    return run_greedy_commit(
        _planner(), env, GOAL_M, XML, env.get_full_state(),
        h=2, sim_budget=sim_budget, prior="model", agg="mean5", combine="q",
        rng=np.random.default_rng(0), restrict_obj=OBJ,
        is_open=lambda _env: False,
    )


def test_blacklists_noop_jams_then_commits_the_first_moving_candidate():
    env = _Env(moves_on={(1, 0)}, failure_reason="OBJECT_STUCK")

    result = _run(env)

    assert env.stepped == [(0, 0), (1, 0)]
    assert result.action is not None
    assert (result.action.edge_idx, result.action.depth) == (1, 0)
    assert result.simulations_used == 2
    assert result.end == "committed"


def test_prunes_same_and_deeper_depths_after_a_noop_jam():
    env = _Env(failure_reason="OBJECT_STUCK")

    result = _run(env)

    assert env.stepped == [(edge, 0) for edge in EDGES]
    assert result.action is None
    assert result.end == "exhausted"


def test_keeps_a_state_that_moved_before_reporting_a_jam():
    env = _Env(moves_on={(0, 0)}, failure_reason="OBJECT_STUCK")

    result = _run(env)

    assert result.action is not None
    assert result.resulting_state["pose"][0] == 1.0
    assert result.rejections == []


def test_never_simulates_a_sibling_after_a_moving_child():
    env = _Env(moves_on={(0, 0), (1, 0)})

    result = _run(env)

    assert env.stepped == [(0, 0)]
    assert result.simulations_used == 1


def test_counts_rejections_against_the_simulation_budget():
    env = _Env()

    result = _run(env, sim_budget=2)

    assert result.action is None
    assert result.simulations_used == 2
    assert result.end == "budget"
