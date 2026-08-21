"""Characterization of RegionOpeningPlanner._search_bfs on the synchronous path.

Written against the pre-refactor implementation and left in place afterwards.
`_search_bfs` used to accept either a list of per-edge goal lists (sync) or an
AsyncGoalResult, and carried an ML polling/merge ladder for the async case. The
async producer was never selected by any config, so the ladder only ever ran in
its `async_result is None` form.

These tests pin what the sync path actually does -- the exact order pushes are
attempted in, and the exact shape and contents of the return value -- so that
removing the async ladder is provably a no-op rather than an assertion.

The environment is a recording fake: what is under test is the search's
ordering and bookkeeping, not physics.
"""

from types import SimpleNamespace

import pytest

from namo.core import PlannerConfig
from namo.planners.opening.region_opening import RegionOpeningPlanner
from namo.planners.utils import PushAttemptBudget
from namo.strategies.goal_selection_strategy import Goal


# Large enough that the budget never truncates these fixtures.
UNLIMITED_PUSH_BUDGET = 100

# Progress printing is time-based; push it far out so it never fires mid-test.
NEVER_PRINT_PROGRESS_SEC = 10_000.0

EDGES = (0, 1, 2)
DEPTHS = (0, 1)


class RecordingEnv:
    """Fake env that records every attempted push in order."""

    def __init__(self):
        self.pushes = []
        self.state = {"name": "baseline"}

    def set_full_state(self, state):
        self.state = state

    def get_full_state(self):
        return self.state

    def get_observation(self):
        return {}

    def get_reachable_objects(self):
        return []


    def step(self, action):
        self.pushes.append((action.object_id, action.edge_idx, action.depth))
        return SimpleNamespace(info={}, done=False, reward=0.0)


@pytest.fixture
def planner_and_env(monkeypatch):
    env = RecordingEnv()
    monkeypatch.setattr(RegionOpeningPlanner, "_setup_constraints", lambda self: None)
    monkeypatch.setattr(
        RegionOpeningPlanner,
        "_initialize_algorithm",
        lambda self: setattr(self, "goal_strategy", object()),
    )
    planner = RegionOpeningPlanner(
        env,
        PlannerConfig(
            verbose=False,
            algorithm_params={"push_budget": PushAttemptBudget(limit=UNLIMITED_PUSH_BUDGET)},
        ),
    )
    planner._progress_total_primitives = 0
    planner._progress_last_print_time = 0.0
    planner._progress_last_print_count = 0
    planner._progress_interval_sec = NEVER_PRINT_PROGRESS_SEC
    planner._rejection_stats = {}
    for name in (
        "_record_push_exec_timing",
        "_record_primitive_ranking_timing",
        "_focus_camera_on_object",
    ):
        monkeypatch.setattr(planner, name, lambda *_a, **_k: None)
    # No push opens the region: forces the search to exhaust every candidate,
    # which is what makes the ordering observable.
    monkeypatch.setattr(planner, "_validate_opening", lambda *_a, **_k: (False, 0, None, []))
    return planner, env


def _goals():
    """One goal per (edge, depth), scored so edge 0 sorts first."""
    return [
        [
            Goal(x=float(e), y=0.0, theta=0.0, score=1.0 - 0.1 * e, edge_idx=e, depth=d)
            for d in DEPTHS
        ]
        for e in EDGES
    ]


def _run(planner):
    return planner._search_bfs(
        goals_per_edge=_goals(),
        reachable_edge_indices=set(EDGES),
        baseline_state={"name": "baseline"},
        neighbour_label="goal",
        region_goals={},
        object_id="box",
    )


def test_every_candidate_is_attempted_in_score_then_depth_order(planner_and_env):
    planner, env = planner_and_env

    _run(planner)

    assert env.pushes == [
        ("box", 0, 0), ("box", 0, 1),
        ("box", 1, 0), ("box", 1, 1),
        ("box", 2, 0), ("box", 2, 1),
    ]


def test_exhausted_search_returns_its_documented_shape(planner_and_env):
    planner, _env = planner_and_env

    result = _run(planner)

    assert len(result) == 6
    successful, min_depth, frontier, wall_collision, movable_collisions, trial_log = result
    assert successful == []
    assert min_depth == 0
    assert frontier == []
    assert wall_collision is False
    assert movable_collisions == set()
    assert len(trial_log) == len(EDGES) * len(DEPTHS)


def test_trial_log_records_every_attempted_cell_once(planner_and_env):
    planner, _env = planner_and_env

    *_rest, trial_log = _run(planner)

    logged = [(entry["edge_idx"], entry["depth"]) for entry in trial_log]
    assert logged == [(e, d) for e in EDGES for d in DEPTHS]
    assert all(entry["success"] is False for entry in trial_log)


def test_budget_is_charged_once_per_attempted_push(planner_and_env):
    planner, env = planner_and_env

    _run(planner)

    assert planner.push_budget.used == len(env.pushes)
