"""Reactive and search must make the same first choice, or comparing them is meaningless.

Both decision rules read one pool: ``rank_first_pushes_h2`` over the reachable
(object, edge, depth) triples at a state, ordered by ``priority(q, V, combine)``.
The search pushes that pool into a heap and pops the head. Reactive takes the
argmax and stops. At ``combine="q"``, the repo default, those are the same
element, so from an identical state the two rules must simulate the identical
push and reach the identical verdict on it.

That is not a nice property, it is the thing that makes the hardware arms
comparable. Reactive is being built as a comparison arm against search
precisely because the sim-to-real gap attacks lookahead, and the whole reading
of that table rests on the two arms differing only in lookahead. If they also
disagree about which push is best at a single state, then every difference the
table shows is confounded by a ranking disagreement nobody measured, and no
result survives.

The published pair is 1-push reactive open@1 = search solve@1 = 83.7 all-tier
(97.9 / 80.7 / 41.6 easy / medium / hard). Those come from two harnesses over
hundreds of episodes. This file cannot re-measure them in a unit test, and does
not try. It pins the mechanism those numbers rest on, per scene and exactly:
same state, same push, same verdict.

Four properties, each with the failure it catches:

  same push, uniform prior     the two rules disagree about the head of a pool
                               they both claim to read. Needs no checkpoint, so
                               this one runs anywhere the binding does
  same push, model prior       the checkpoint reorders the pool and only one of
                               the rules follows it, which is the drift that
                               would survive the uniform case
  same verdict on that push    both rules pick alike but grade differently,
                               so open@1 and solve@1 count different events
  the push is the pool head     both rules agree on something that is not the
                               argmax. Agreement between two broken rules reads
                               exactly like agreement between two correct ones,
                               so pinning the answer independently is what
                               separates them

Deliberately not pinned: which edge wins. That moves with the checkpoint and
with physics, and the point here is that the two rules move together.

One simulator call per rule per case, so this is seconds, not minutes.

To verify:
  cd namo_cpp && source env.ilab.sh
  python -m pytest python/tests/test_reactive_search_first_choice_parity.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import REPO_ROOT, _require_real_namo_rl

_require_real_namo_rl()


# ─── Named constants ────────────────────────────────────────────────────

# A captured car scene with one movable and a boundary the robot can reach.
# Lives in the robot_control checkout beside this one, which is also where the
# deploy path reads its scenes from.
SCENE = (
    REPO_ROOT.parent
    / "robot_control"
    / "real_test_envs"
    / "1push/1hop/env1"
    / "env.xml"
)

GOAL_M = (0.37, 0.67, 0.0)
# The scenes spawn the car wedged in the wall corner; from there nothing is
# reachable and the candidate pool is empty.
START_POSE_M = (0.25, 0.10, 0.0)

# The deployed checkpoint. Resolved under this box's scratch root rather than
# written absolute, so the file is portable; override to point at another.
_SCRATCH = os.environ.get("NAMO_SCRATCH", "")
SCORER_CKPT = Path(
    os.environ.get("NAMO_REACTIVE_TEST_CKPT", "")
    or (
        Path(_SCRATCH) / "amarel_pull_20260817/cache_aquaman0_ckpts_bfix/HY5U_s2.ckpt"
        if _SCRATCH
        else "/nonexistent"
    )
)

# hmax=1 is the anchor's regime: one push, so the search has no lookahead to
# spend and the two rules are being compared on ranking alone.
ONE_PUSH = 1

# The repo default (best_first_combine). At "q" the heap is ordered by the raw
# action value, so its head is the pool's argmax.
COMBINE = "q"
AGG = "mean5"
SEED = 42

pytestmark = pytest.mark.skipif(
    not SCENE.is_file(), reason=f"needs the captured scene at {SCENE}"
)


# ─── Helpers ────────────────────────────────────────────────────────────


class _RecordingEnv:
    """Delegates to the real environment and records what was asked of it.

    The action is what the two rules are being compared on, so it is read off
    the simulator call itself rather than from either rule's own reporting. A
    rule that returns a plan it did not simulate cannot pass this way.
    """

    def __init__(self, env):
        self._env = env
        self.stepped = []

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        self.stepped.append(
            (str(action.object_id), int(action.edge_idx), int(action.depth))
        )
        return self._env.step(action)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """One loaded scene, its start state, and both planner surfaces.

    Module-scoped: the model arm loads a 50 MB checkpoint, and every case here
    starts from the same state by construction.
    """
    import sys

    import namo_rl

    from namo.runtime_profile import CANONICAL_CONFIG, CANONICAL_PRIMITIVE_PREFIX
    from namo.strategies.primitive_goal_strategy import PrimitiveGoalStrategy

    sys.path.insert(0, str(REPO_ROOT.parent / "robot_control" / "src"))
    from robot_control.utils.scene_xml import portable_scene

    config = REPO_ROOT / CANONICAL_CONFIG
    # The captured scene carries the absolute include of the box that made it.
    xml = portable_scene(SCENE, tmp_path_factory.mktemp("scene"))

    env = namo_rl.RLEnvironment(str(xml), str(config), False)
    env.reset()
    env.set_robot_pose(*START_POSE_M)
    state = env.get_full_state()

    prim = PrimitiveGoalStrategy(
        data_dir=str(REPO_ROOT / "data"),
        primitive_prefix=CANONICAL_PRIMITIVE_PREFIX,
    )
    planners = {"uniform": SimpleNamespace(prim=prim, scorer=None)}
    if SCORER_CKPT.is_file():
        from namo.strategies.scorer_goal_strategy import _get_scorer

        planners["model"] = SimpleNamespace(
            prim=prim, scorer=_get_scorer(str(SCORER_CKPT), str(config), "cpu")
        )
    return {"env": env, "xml": str(xml), "state": state, "planners": planners}


def _run(scene, rule, prior):
    """Run one decision rule for a single push and report what it did.

    Both rules get their own generator seeded identically, because under the
    uniform prior the priority IS the draw: sharing a generator would let the
    order of the two calls decide the comparison.
    """
    import numpy as np

    from namo.planners.opening.best_first_search import run_reactive, solve_scene

    planner = scene["planners"][prior]
    env = _RecordingEnv(scene["env"])
    entry = {"search": solve_scene, "reactive": run_reactive}[rule]

    solved, sims, _plan_len, _boards, _end = entry(
        planner,
        env,
        GOAL_M,
        scene["xml"],
        scene["state"],
        ONE_PUSH,
        ONE_PUSH,
        prior,
        AGG,
        COMBINE,
        np.random.default_rng(SEED),
        is_open=lambda e: e.is_robot_goal_reachable(),
    )
    assert sims == 1, f"{rule} spent {sims} simulations on a one-push budget"
    return {"action": env.stepped[0], "solved": bool(solved)}


def _pool_head(scene, prior):
    """The argmax of the pool, computed here rather than taken from either rule."""
    import numpy as np

    from namo.planners.opening.best_first_search import candidates, priority

    pool, value, _grid = candidates(
        scene["planners"][prior],
        scene["env"],
        GOAL_M,
        scene["xml"],
        scene["state"],
        ONE_PUSH,
        prior,
        AGG,
        np.random.default_rng(SEED),
    )
    assert pool, "an empty pool tests nothing"
    obj, goal, score = max(pool, key=lambda c: priority(c[2], value, COMBINE))
    return (str(obj), int(goal.edge_idx), int(goal.depth))


def _priors():
    marks = [pytest.param("uniform")]
    marks.append(
        pytest.param(
            "model",
            marks=pytest.mark.skipif(
                not SCORER_CKPT.is_file(), reason=f"needs {SCORER_CKPT.name}"
            ),
        )
    )
    return marks


# ─── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prior", _priors())
def test_both_rules_simulate_the_same_first_push(scene, prior):
    """The anchor. Same state, same pool, so the same push goes to the sim."""
    search = _run(scene, "search", prior)
    reactive = _run(scene, "reactive", prior)

    assert reactive["action"] == search["action"], (
        f"first choice diverged under prior={prior}: "
        f"reactive {reactive['action']} vs search {search['action']}. "
        "The two arms no longer differ only in lookahead."
    )


@pytest.mark.parametrize("prior", _priors())
def test_both_rules_reach_the_same_verdict(scene, prior):
    """open@1 and solve@1 have to be counting the same event."""
    search = _run(scene, "search", prior)
    reactive = _run(scene, "reactive", prior)

    assert reactive["solved"] == search["solved"], (
        f"same push graded differently under prior={prior}: "
        f"reactive solved={reactive['solved']} vs search solved={search['solved']}"
    )


@pytest.mark.parametrize("prior", _priors())
def test_the_agreed_push_is_the_argmax_of_the_pool(scene, prior):
    """Both agreeing on the wrong push looks exactly like both being right."""
    head = _pool_head(scene, prior)

    assert _run(scene, "search", prior)["action"] == head
    assert _run(scene, "reactive", prior)["action"] == head
