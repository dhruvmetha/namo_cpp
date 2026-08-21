"""Both local searches must grade an opening against the same bar.

region_bfs has always defaulted region_min_reachable_fraction to 0.2 -- the
">=20 of 100 sampled goal points" rule the model registry's numbers were
produced under. best_first defaulted it to 0.0, which collapses to the
absolute region_success_min_reachable count (>=1 of 100). A config that set
neither therefore graded the two arms differently, making any A/B between
them invalid and letting the robot accept a boundary as open on a single
reachable point.

These tests pin that an omitted key lands both openers on the canonical bar.
"""

import math

import pytest

import inspect

from namo.core import PlannerConfig
from namo.services.planning_service import NAMOPlanningService
from namo.planners.opening.region_opening import (
    CANONICAL_MIN_REACHABLE_FRACTION,
    RegionOpeningPlanner,
)
from namo.planners.opening.best_first_region_opening import BestFirstRegionOpeningPlanner

# The canonical pairing: 100 sampled points per region, 20% of them reachable.
CANONICAL_GOALS_PER_REGION = 100
CANONICAL_POINTS_NEEDED = 20


class _StubEnv:
    def get_full_state(self):
        return {"name": "baseline"}

    def set_full_state(self, _state):
        return None



def _best_first(**params):
    """Uniform prior needs no checkpoint, so this needs no model or GPU."""
    params.setdefault("best_first_prior", "uniform")
    return BestFirstRegionOpeningPlanner(_StubEnv(), PlannerConfig(algorithm_params=params))


def _region_bfs(monkeypatch, **params):
    monkeypatch.setattr(RegionOpeningPlanner, "_setup_constraints", lambda self: None)
    monkeypatch.setattr(
        RegionOpeningPlanner,
        "_initialize_algorithm",
        lambda self: setattr(self, "goal_strategy", object()),
    )
    return RegionOpeningPlanner(_StubEnv(), PlannerConfig(algorithm_params=params))


def test_canonical_bar_is_twenty_of_one_hundred():
    needed = math.ceil(CANONICAL_MIN_REACHABLE_FRACTION * CANONICAL_GOALS_PER_REGION)
    assert needed == CANONICAL_POINTS_NEEDED


def test_both_openers_default_to_the_same_bar(monkeypatch):
    assert _best_first().min_fraction == CANONICAL_MIN_REACHABLE_FRACTION
    assert _region_bfs(monkeypatch)._min_reachable_fraction == CANONICAL_MIN_REACHABLE_FRACTION


def test_best_first_requires_twenty_points_by_default():
    assert _best_first()._minimum_needed(CANONICAL_GOALS_PER_REGION) == CANONICAL_POINTS_NEEDED


def test_explicit_zero_still_falls_back_to_the_absolute_count():
    planner = _best_first(region_min_reachable_fraction=0.0, region_success_min_reachable=3)

    assert planner._minimum_needed(CANONICAL_GOALS_PER_REGION) == 3


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_out_of_range_fraction_is_rejected(bad):
    with pytest.raises(ValueError, match="region_min_reachable_fraction"):
        _best_first(region_min_reachable_fraction=bad)


def test_service_defers_the_sample_size_to_planner_config():
    """The bar is a fraction, so the sample size is half of it.

    plan_from_xml used to default goals_per_region to 10 while PlannerConfig
    uses 100. Combined with the 0.2 fraction that graded a service caller at
    ">=2 of 10 sampled points" instead of ">=20 of 100" -- the same fraction
    over a sample small enough to be noise. Omitting the argument now defers to
    PlannerConfig, so there is one canonical sample size.
    """
    default = inspect.signature(NAMOPlanningService.plan_from_xml).parameters[
        "goals_per_region"
    ].default

    assert default is None
    assert PlannerConfig().goals_per_region == CANONICAL_GOALS_PER_REGION
