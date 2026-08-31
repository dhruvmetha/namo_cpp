"""Reactive is a pure argmax: rank the live state, return the top push, simulate nothing.

Second redefinition, Dhruv, 2026-08-31. The first form chained pushes
through simulated states and charged sim no-ops against hmax, which ended
hardware runs with the robot never moving (hmax2/hard_004, five runs). The
first fix kept a sim no-op filter in front of dispatch; Dhruv ruled that
the wrong behavior outright -- a mode defined by distrusting the simulator
does not let the simulator screen its choices. Jam handling now lives
entirely on the hardware side: the camera's stuck detection blacklists a
push that moved nothing, and the blacklist feeds the next call's pool.

Three properties, each with the failure it catches:

  reactive never simulates a push   the simulator regaining a veto by any
                                    path, the bug this file exists to stop
  the argmax is the decision        one call, one push, end "decided"
  an empty pool ends the run        no reachable candidate means exhausted,
                                    not a crash and not a fabricated push

Runs against a fake environment, so no binding physics, no checkpoint, no
scene.

To verify:
  cd namo_cpp && source env.ilab.sh
  python -m pytest python/tests/test_reactive_jam_guards.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import _require_real_namo_rl

# make_action builds a namo_rl.Action, so the binding has to import. Nothing
# below runs physics.
_require_real_namo_rl()

from namo.planners.opening.best_first_search import run_reactive  # noqa: E402


# ─── Named constants ────────────────────────────────────────────────────

OBJ = "obstacle_1_movable"
EDGES = (0, 1, 2)
DEPTHS = (0, 1, 2)
POOL_SIZE = len(EDGES) * len(DEPTHS)

# The scorer head is (60 contacts, 5 depths). Only the cells above are ever
# offered as goals; the rest of the grid exists so the shape is the real one.
GRID_CONTACTS = 60
GRID_DEPTHS = 5

# Enough budget that running out of it is never the reason a test ends.
AMPLE_SIMS = 50

GOAL_M = (0.37, 0.67, 0.0)
XML = "fake.xml"


# ─── Fakes ──────────────────────────────────────────────────────────────


class _Goal:
    def __init__(self, edge_idx, depth):
        self.x = self.y = self.theta = 0.0
        self.edge_idx = edge_idx
        self.depth = depth


class _Prim:
    """Enumerates the same goals every call, grouped by edge, as the real one does."""

    def generate_goals(self, _obj, _state, _env, max_goals=0):
        return [[_Goal(e, d) for d in DEPTHS] for e in EDGES]


class _Scorer:
    """A fixed grid, so the order reactive walks is known in advance.

    Descending by edge then depth: (0,0) first, (2,2) last. Pinning the order
    is what lets the pruning test name which pushes should never be reached.
    """

    def score_state(self, _env, _obj, _goal, _xml, region_samples=None, h=1, raw=False):
        grid = np.zeros((GRID_CONTACTS, GRID_DEPTHS), dtype=float)
        for e in EDGES:
            for d in DEPTHS:
                grid[e, d] = 1.0 - (0.1 * e + 0.01 * d)
        return grid


class _Env:
    """Records every push, and moves the object only when told to.

    ``moves_on`` names the pushes that actually displace something. Everything
    else is a no-op: same observation before and after, which is exactly what a
    jam looks like from outside.
    """

    def __init__(self, moves_on=(), failure_reason=""):
        self.moves_on = set(moves_on)
        self.failure_reason = failure_reason
        self.stepped = []
        self._pose = [0.0, 0.0, 0.0]

    # -- what the ranker reads --
    def set_full_state(self, _state):
        return None

    def get_full_state(self):
        return {"pose": list(self._pose)}

    def get_reachable_objects(self):
        return [OBJ]

    def get_reachable_edges(self, _obj):
        return list(EDGES)

    def get_observation(self):
        return {f"{OBJ}_pose": list(self._pose), "robot_pose": [0.0, 0.0, 0.0]}

    # -- what reactive calls --
    def step(self, action):
        key = (int(action.edge_idx), int(action.depth))
        self.stepped.append(key)
        if key in self.moves_on:
            self._pose[0] += 1.0
        info = {"failure_reason": self.failure_reason} if self.failure_reason else {}
        return SimpleNamespace(info=info)


def _planner():
    return SimpleNamespace(prim=_Prim(), scorer=_Scorer())


def _run(env, *, pushes=POOL_SIZE, **kwargs):
    """One reactive run that never opens, so it stops only on a guard or a budget."""
    return run_reactive(
        _planner(),
        env,
        GOAL_M,
        XML,
        {"pose": [0.0, 0.0, 0.0]},
        pushes,
        AMPLE_SIMS,
        "model",
        "mean5",
        "q",
        np.random.default_rng(0),
        is_open=lambda _e: False,
        **kwargs,
    )


# ─── Tests ──────────────────────────────────────────────────────────────


def test_reactive_never_simulates_a_push():
    """The whole point of the redefinition: env.step must never be called."""
    env = _Env()

    solved, sims, plan_len, _boards, end = _run(env, pushes=4)

    assert env.stepped == [], f"reactive stepped the simulator: {env.stepped}"
    assert sims == 0


def test_the_argmax_is_the_decision():
    """One call, one push: the top-priority candidate comes back, end decided."""
    env = _Env()
    out = {}

    solved, sims, plan_len, _boards, end = _run(env, pushes=4, solution_out=out)

    assert end == "decided" and not solved
    assert plan_len == 1
    obj, goal = out["plan"][0]
    assert obj == OBJ
    assert (int(goal.edge_idx), int(goal.depth)) == (EDGES[0], DEPTHS[0]), (
        "the decision must be the scorer's argmax"
    )


def test_an_empty_pool_ends_the_run():
    """No reachable candidate is exhausted, not a crash or an invented push."""
    env = _Env()
    env.get_reachable_objects = lambda: []

    solved, sims, plan_len, _boards, end = _run(env, pushes=4)

    assert end == "exhausted" and not solved
    assert sims == 0
