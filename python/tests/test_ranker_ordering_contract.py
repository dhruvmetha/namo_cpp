"""What the ranker promises the search: a stable order it did not invent itself.

Section 10.3 item 6 asks whether the production candidate ordering matches the
evaluation harness. That question can no longer fail. Since 96b4c08 moved
best-first out of the sandbox, scripts/sandbox/eval_m3.py line 55 imports
rank_first_pushes_h2 from namo.planners.opening.best_first_search and re-exports
it, and eval_bestfirst.py takes it from there. One implementation, so a parity
assertion could only ever pass. Asserting it anyway would be theatre.

What is worth pinning is the part nothing covers.
test_best_first_sandbox_contract pins chains and simulation counts under a
UNIFORM prior, deliberately, so it runs anywhere the binding does without a
checkpoint. That leaves the model-scored path untested, which is the path the
robot actually runs.

Four properties, each with a failure it catches:

  the grid is 60 x 5      the head is (60 contacts, 5 depths, 51 bins) and the
                          bins collapse to one number per cell; a shape drift
                          here means the checkpoint and the code disagree
  order and values repeat  a ranker that answers differently on identical input
                          makes every downstream determinism claim meaningless
  scoring changes order    if the model's output never reordered the pool it
                          would not matter whether it loaded at all
  values land in [0, 1]    the head is HL-Gauss over [0, 1] and the usable
                          scalar is the bin expectation. An argmax or a
                          max-over-bins also returns a plausible-looking float,
                          which is the trap the ranking README names first

Deliberately not pinned: which edge wins. That moves with the checkpoint and
with physics, and re-recording pinned answers is worth it only where the answer
is the point.

Needs the deployed checkpoint, which lives outside both repos; skipped without
it.

To verify:
  cd namo_cpp && source env.robotlearning.sh
  python -m pytest python/tests/test_ranker_ordering_contract.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import REPO_ROOT, _require_real_namo_rl

_require_real_namo_rl()


# ─── Named constants ────────────────────────────────────────────────────

# Outside both checkouts, so this is a path and not an import.
SCORER_CKPT = Path("/home/dhruv/projects_dhruv/namo/ranking/models/HY5U_s2.ckpt")

# A captured car scene with one movable and a boundary the robot can reach.
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

# Chain budget the deployed search runs at, and the horizon the scores condition
# on. hmax=2 is the canonical eval protocol.
HORIZON = 2

# The head is (60 contacts, 5 depths, 51 value bins). rank_first_pushes_h2
# returns the bins already collapsed to their expectation, so the grid it hands
# back is one score per contact and depth.
EXPECTED_CONTACTS = 60
EXPECTED_DEPTHS = 5

pytestmark = pytest.mark.skipif(
    not SCORER_CKPT.is_file() or not SCENE.is_file(),
    reason=f"needs {SCORER_CKPT.name} and a captured scene",
)


# ─── Helpers ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ranking(tmp_path_factory):
    """One scored pool and one unscored pool over the same state.

    Module-scoped: building the scorer loads a 50 MB checkpoint.
    """
    import namo_rl
    from namo.planners.opening.best_first_search import rank_first_pushes_h2
    from namo.runtime_profile import CANONICAL_CONFIG, CANONICAL_PRIMITIVE_PREFIX
    from namo.strategies.primitive_goal_strategy import PrimitiveGoalStrategy
    from namo.strategies.scorer_goal_strategy import _get_scorer

    import sys

    sys.path.insert(0, str(REPO_ROOT.parent / "robot_control" / "src"))
    from robot_control.utils.scene_xml import portable_scene

    config = REPO_ROOT / CANONICAL_CONFIG
    tmp = tmp_path_factory.mktemp("scene")
    # The captured scene carries the absolute include of the box that made it.
    xml = portable_scene(SCENE, tmp)

    env = namo_rl.RLEnvironment(str(xml), str(config), False)
    env.reset()
    env.set_robot_pose(*START_POSE_M)
    state = env.get_full_state()

    planner = SimpleNamespace(
        prim=PrimitiveGoalStrategy(
            data_dir=str(REPO_ROOT / "data"),
            primitive_prefix=CANONICAL_PRIMITIVE_PREFIX,
        ),
        scorer=_get_scorer(str(SCORER_CKPT), str(config), "cpu"),
    )

    def rank(**kw):
        return rank_first_pushes_h2(
            planner, env, GOAL_M, str(xml), state, HORIZON, **kw
        )

    scored, grid = rank(score=True, return_grid=True)
    repeated, _ = rank(score=True, return_grid=True)
    unscored = rank(score=False)
    return {
        "scored": scored,
        "repeated": repeated,
        "unscored": unscored,
        "grid": grid,
    }


def _order(pool):
    return [(obj, goal.edge_idx, goal.depth) for obj, goal, _value in pool]


def _values(pool):
    return [value for _obj, _goal, value in pool]


# ─── Tests ──────────────────────────────────────────────────────────────


def test_the_score_grid_is_one_value_per_contact_and_depth(ranking):
    grid = ranking["grid"]

    assert grid is not None, "return_grid=True has to hand back the pass it made"
    assert len(grid) == EXPECTED_CONTACTS
    assert all(len(row) == EXPECTED_DEPTHS for row in grid)


def test_the_same_state_ranks_the_same_way_twice(ranking):
    """Identical input, identical answer, values included.

    Every determinism claim downstream of the ranker rests on this one.
    """
    assert _order(ranking["scored"]) == _order(ranking["repeated"])
    assert _values(ranking["scored"]) == _values(ranking["repeated"])


def test_scoring_is_what_orders_the_pool(ranking):
    """Same candidates either way; the model decides what comes first.

    Without this, a checkpoint that silently failed to load would still produce
    a plausible pool in enumeration order.
    """
    scored, unscored = ranking["scored"], ranking["unscored"]

    assert len(scored) == len(unscored), "scoring must not change the candidate set"
    assert set(_order(scored)) == set(_order(unscored))
    assert _order(scored) != _order(unscored)
    assert all(value == 0.0 for value in _values(unscored)), (
        "the random baseline must not touch the model"
    )


def test_the_values_are_bin_expectations_not_raw_head_output(ranking):
    """HL-Gauss over [0, 1], so the usable scalar lives in [0, 1].

    An argmax over the 51 bins, or a max-over-bins, also returns a float that
    looks reasonable. The range is what separates them.
    """
    values = _values(ranking["scored"])

    assert values, "an empty pool tests nothing"
    assert all(0.0 <= value <= 1.0 for value in values), (
        f"values outside [0, 1]: min={min(values)}, max={max(values)}"
    )
    assert values == sorted(values, reverse=True), "the pool is meant to arrive sorted"
