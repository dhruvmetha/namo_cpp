"""A jammed push must not be re-picked, because reactive cannot escape on its own.

A push that jams leaves the state exactly as it found it. Reactive re-ranks
that identical state, gets back the identical pool, and picks the identical
argmax. Nothing in the loop breaks that cycle: re-deciding from the world is
the whole idea of the reactive arm, and here the world is not changing.

Measured on 20 failed easy 2-push episodes without the guards: 2.2 distinct
pushes across 10 steps, 8.75 no-ops, and 45% of episodes picking one push all
ten times. In simulation that wastes calls. On the real robot it is the car
shoving a stuck block over and over until a person stops it, which is why these
are not optional and not caller-supplied.

The search never had this problem, so the fix is borrowed from it rather than
invented. ``dedupe_noop`` drops a push that moved nothing, ``prune_jam_depth``
drops deeper pushes on an edge already known to jam (push_steps = depth + 1 and
the controller runs one continuous push, so a deeper push is the same
trajectory continued into the same obstruction). ``_unmoved`` is imported from
the search, never re-implemented, so there is one definition of "moved
nothing".

Five properties, each with the failure it catches:

  no push is simulated twice     the lock-up itself, in its plainest form
  an empty pool ends the run     reactive burns its whole budget re-picking
                                 from a pool it has exhausted
  a jammed edge prunes deeper    depth 2 on an edge that jammed at depth 0 is
                                 the same trajectory into the same obstruction,
                                 so simulating it is a call spent to learn
                                 nothing
  the guards are on by default   a caller that forgets them gets the measured
                                 lock-up, so forgetting must not be possible
  a move clears the bans         over-correcting the other way: once the object
                                 moves the board is new, and a push banned at
                                 the old state has to be offered again or the
                                 reactive blinds itself to its best option

Runs against a fake environment, so no binding physics, no checkpoint, no
scene. The guards are control flow and this pins the control flow.

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


def test_a_jammed_push_is_never_simulated_twice():
    """The lock-up itself: nine no-ops must be nine different pushes."""
    env = _Env()

    _run(env)

    assert len(env.stepped) == len(set(env.stepped)), (
        f"reactive re-picked a push that moved nothing: {env.stepped}"
    )


def test_an_exhausted_pool_ends_the_run():
    """Every candidate banned means there is nothing left to decide between."""
    env = _Env()

    _solved, sims, _plan_len, _boards, end = _run(env, pushes=AMPLE_SIMS)

    assert end == "exhausted", f"expected an exhausted pool, got {end!r}"
    assert sims == POOL_SIZE, (
        f"the pool holds {POOL_SIZE} pushes; reactive spent {sims} simulations"
    )


def test_a_jammed_edge_prunes_its_deeper_pushes():
    """Depth 2 on an edge that jammed at depth 0 runs into the same obstruction."""
    env = _Env(failure_reason="OBJECT_STUCK")

    _run(env, pushes=AMPLE_SIMS)

    assert env.stepped == [(e, 0) for e in EDGES], (
        f"expected the shallowest push per edge and nothing deeper, got {env.stepped}"
    )


def test_the_guards_do_not_have_to_be_asked_for():
    """A caller that forgets them would get the measured lock-up, so they are on."""
    import inspect

    defaults = inspect.signature(run_reactive).parameters

    assert defaults["dedupe_noop"].default is True
    assert defaults["prune_jam_depth"].default is True


def test_a_push_that_moves_the_object_clears_the_bans():
    """A new state is a new board, so the bans from the old one do not carry.

    The counterweight to the other four. Bans that outlived the state they were
    recorded at would hide reactive's best push for the rest of the episode,
    which is a quieter failure than the lock-up and just as wrong.
    """
    best = (EDGES[0], DEPTHS[0])
    # The best push is a no-op once, then works. If the ban survived the move,
    # reactive could never come back to it.
    env = _Env()

    def step(action):
        key = (int(action.edge_idx), int(action.depth))
        env.stepped.append(key)
        # The second push moves the object, whatever it is.
        if len(env.stepped) == 2:
            env._pose[0] += 1.0
        return SimpleNamespace(info={})

    env.step = step
    _run(env, pushes=4)

    assert env.stepped[2] == best, (
        f"after the object moved reactive should re-offer its top push {best}, "
        f"got {env.stepped}"
    )
