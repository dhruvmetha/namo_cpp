"""A push the skill refuses must not be reported as a push that happened.

`build_scene_replay.push()` used to return True as soon as it found a goal with the asked-for
(edge, depth), ignoring what the simulator did with it. When the edge is unreachable from where the
robot stands the skill declines, `failure_reason` comes back set, and the board is left bit-identical
-- so the frame the builder snapshotted next was the START state carrying an "after push 1" caption.
76 of 1278 two-push replays in the two-movable gallery shipped that way: setup no-opped, finish ran
from the untouched root and opened, and the card animated one push while claiming two.

The fixture is v3_snap/med/rb_00026, the scene the bug was found on. Edge 1 at depth 0 is the dud
the sweep recorded; edge 41 at depth 3 is the finish that really runs.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "python", "tests", "data")
CFG = os.path.join(REPO, "config", "namo_config_complete_skill15_car_1x.yaml")
XML = os.path.join(DATA, "unreachable_edge_push_fixture.xml")
OBJ = "obstacle_1_movable"

pytest.importorskip("namo_rl")
import namo_rl  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "scripts", "viz"))
from build_scene_replay import push  # noqa: E402

from namo.strategies.primitive_goal_strategy import PrimitiveGoalStrategy  # noqa: E402


def _make_action(obj, g):
    a = namo_rl.Action()
    a.object_id = obj
    a.x, a.y, a.theta = g.x, g.y, g.theta
    a.edge_idx = int(g.edge_idx)
    a.depth = int(g.depth)
    return a


def _pose(env):
    return tuple(round(v, 9) for v in env.get_observation()[f"{OBJ}_pose"][:2])


@pytest.fixture()
def env():
    e = namo_rl.RLEnvironment(XML, CFG, False)
    e.reset()
    return e


def test_refused_push_returns_false_and_leaves_the_board_alone(env):
    prim = PrimitiveGoalStrategy()
    before = _pose(env)
    assert push(env, prim, _make_action, OBJ, 1, 0) is False
    assert _pose(env) == before, "a refused push must not be reported as motion"


def test_a_push_that_runs_still_returns_true(env):
    prim = PrimitiveGoalStrategy()
    before = _pose(env)
    assert push(env, prim, _make_action, OBJ, 41, 3) is True
    assert _pose(env) != before, "the control push has to actually move the block"


def test_a_goal_that_does_not_exist_still_returns_false(env):
    prim = PrimitiveGoalStrategy()
    assert push(env, prim, _make_action, OBJ, 999, 0) is False
