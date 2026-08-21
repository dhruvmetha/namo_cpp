"""How an opening is graded, and how a caller pins the points it is graded against.

`_validate_opening` decides whether a region counts as opened. It samples the
target region fresh on every call, which is correct inside one planning call
and wrong across physical pushes: a push re-partitions free space, so the next
call samples different points and grades against a different target. Region
labels are ordinal and renumber for the same reason, so they cannot be used to
pin the target either.

The first four tests characterize the sampling behaviour and were verified
against the implementation *before* pinning existed. The rest cover the pinned
path, where a caller supplies the points once and every later call grades
against exactly those.
"""

import math
from types import SimpleNamespace

import pytest

from namo.core import PlannerConfig
from namo.planners.opening.region_opening import (
    CANONICAL_MIN_REACHABLE_FRACTION,
    RegionOpeningPlanner,
)

TARGET = "region_4"
SAMPLED_POINT_COUNT = 100
# 0.2 x 100; the canonical ">=20 of 100" rule.
CANONICAL_POINTS_NEEDED = 20


class _CountingEnv:
    """Env stub whose reachability answer is fixed and whose calls are recorded."""

    def __init__(self, reachable_count=0, first_idx=-1):
        self.reachable_count = reachable_count
        self.first_idx = first_idx
        self.graded_against = []

    def count_reachable_points(self, xy_points):
        self.graded_against.append(list(xy_points))
        return self.reachable_count, self.first_idx

    def get_full_state(self):
        return {"name": "baseline"}

    def set_full_state(self, _state):
        return None



def _bundle(n, x0=0.0):
    return SimpleNamespace(
        goals=[SimpleNamespace(x=x0 + i * 0.01, y=0.0, theta=0.0) for i in range(n)]
    )


def _planner(monkeypatch, env, **params):
    monkeypatch.setattr(RegionOpeningPlanner, "_setup_constraints", lambda self: None)
    monkeypatch.setattr(
        RegionOpeningPlanner,
        "_initialize_algorithm",
        lambda self: setattr(self, "goal_strategy", object()),
    )
    planner = RegionOpeningPlanner(env, PlannerConfig(algorithm_params=params))
    planner._record_opening_validation_timing = lambda *_a, **_k: None
    return planner


# --- characterization: sampled-points behaviour, unchanged --------------------

def test_grades_against_the_sampled_points_of_the_target_region(monkeypatch):
    env = _CountingEnv(reachable_count=CANONICAL_POINTS_NEEDED, first_idx=0)
    planner = _planner(monkeypatch, env)
    region_goals = {TARGET: _bundle(SAMPLED_POINT_COUNT)}

    success, count, _first, all_goals = planner._validate_opening(TARGET, region_goals)

    assert success is True
    assert count == CANONICAL_POINTS_NEEDED
    assert len(all_goals) == SAMPLED_POINT_COUNT
    assert len(env.graded_against[0]) == SAMPLED_POINT_COUNT


def test_one_point_short_of_the_bar_is_not_open(monkeypatch):
    env = _CountingEnv(reachable_count=CANONICAL_POINTS_NEEDED - 1, first_idx=0)
    planner = _planner(monkeypatch, env)

    success, _count, _first, _all = planner._validate_opening(
        TARGET, {TARGET: _bundle(SAMPLED_POINT_COUNT)}
    )

    assert success is False


def test_missing_or_empty_region_grades_closed(monkeypatch):
    planner = _planner(monkeypatch, _CountingEnv())

    assert planner._validate_opening(TARGET, {})[0] is False
    assert planner._validate_opening(TARGET, {TARGET: _bundle(0)})[0] is False


def test_denominator_is_the_points_actually_sampled(monkeypatch):
    """A region with few free cells yields fewer points; the bar scales with it."""
    env = _CountingEnv(reachable_count=2, first_idx=0)
    planner = _planner(monkeypatch, env)

    success, _c, _f, _a = planner._validate_opening(TARGET, {TARGET: _bundle(10)})

    assert math.ceil(CANONICAL_MIN_REACHABLE_FRACTION * 10) == 2
    assert success is True


# --- pinned points -----------------------------------------------------------

PINNED = [(0.30, 0.40), (0.31, 0.40), (0.30, 0.41), (0.32, 0.42), (0.33, 0.43)]


def test_pinned_points_replace_the_sampled_ones(monkeypatch):
    """The whole point: grade against the caller's list, not this snapshot's."""
    env = _CountingEnv(reachable_count=len(PINNED), first_idx=0)
    planner = _planner(monkeypatch, env, region_target_points=PINNED)

    _s, _c, _f, all_goals = planner._validate_opening(
        TARGET, {TARGET: _bundle(SAMPLED_POINT_COUNT, x0=99.0)}
    )

    assert env.graded_against[0] == PINNED
    assert [(x, y) for x, y, _t in all_goals] == PINNED


def test_pinned_points_survive_the_target_label_vanishing(monkeypatch):
    """Labels are ordinal and renumber after a push; the pinned points must not care."""
    env = _CountingEnv(reachable_count=len(PINNED), first_idx=0)
    planner = _planner(monkeypatch, env, region_target_points=PINNED)

    success, _c, _f, _a = planner._validate_opening("region_99_gone", {})

    assert success is True
    assert env.graded_against[0] == PINNED


def test_pinned_denominator_is_the_pinned_count(monkeypatch):
    needed = math.ceil(CANONICAL_MIN_REACHABLE_FRACTION * len(PINNED))
    assert needed == 1

    at_bar = _planner(monkeypatch, _CountingEnv(needed, 0), region_target_points=PINNED)
    below = _planner(monkeypatch, _CountingEnv(needed - 1, -1), region_target_points=PINNED)

    assert at_bar._validate_opening(TARGET, {})[0] is True
    assert below._validate_opening(TARGET, {})[0] is False


def test_absent_key_leaves_sampling_untouched(monkeypatch):
    planner = _planner(monkeypatch, _CountingEnv())

    assert planner._pinned_target_points is None


def test_empty_pinned_list_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="region_target_points"):
        _planner(monkeypatch, _CountingEnv(), region_target_points=[])


def test_pinned_points_are_normalised_to_float_pairs(monkeypatch):
    """They cross a config boundary, so tuples/lists/ints must all work."""
    planner = _planner(monkeypatch, _CountingEnv(), region_target_points=[[0, 1], (2, 3)])

    assert planner._pinned_target_points == [(0.0, 1.0), (2.0, 3.0)]


def test_best_first_shares_the_same_pinning_contract():
    """One parser, so the two openers cannot disagree about the pinned target."""
    from namo.planners.opening.best_first_region_opening import (
        BestFirstRegionOpeningPlanner,
    )

    class _StubEnv:
        def get_full_state(self):
            return {"name": "baseline"}

        def set_full_state(self, _s):
            return None


    planner = BestFirstRegionOpeningPlanner(
        _StubEnv(),
        PlannerConfig(
            algorithm_params={
                "best_first_prior": "uniform",
                "region_target_points": PINNED,
            }
        ),
    )

    assert planner._pinned_target_points == PINNED
