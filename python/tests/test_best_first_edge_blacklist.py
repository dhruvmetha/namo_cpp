"""best_first must skip edges the caller reports as already failed.

region_bfs honours external_edge_blacklist by seeding its per-node
edge_min_stuck_depth map. best_first ignored the key entirely, so a push the
real robot had just physically attempted and failed stayed in the candidate
pool: it could be scored, simulated, ranked first again, and re-issued. With
one push executed per replan, that is a loop the robot cannot escape.

Filtering happens at enumeration -- the blacklisted candidate never enters
the heap, so it costs no simulation and no budget.
"""

from types import SimpleNamespace

import pytest

from namo.core import PlannerConfig
from namo.planners.opening.best_first_region_opening import (
    BestFirstRegionOpeningPlanner,
    _BlacklistedEdgeFilter,
)
from namo.strategies import PrimitiveGoalStrategy

TARGET = "obstacle_1_movable"
OTHER = "obstacle_2_movable"
EDGES = (0, 1, 2, 3)
DEPTHS_PER_EDGE = 2


class _StubEnv:
    def get_full_state(self):
        return {"name": "baseline"}

    def set_full_state(self, _state):
        return None



class _FakePrim:
    """Returns one goal group per edge, mimicking PrimitiveGoalStrategy's shape."""

    def __init__(self):
        self.calls = []

    def generate_goals(self, object_id, state, env, max_goals=0):
        self.calls.append(object_id)
        return [
            [SimpleNamespace(edge_idx=e, depth=d) for d in range(DEPTHS_PER_EDGE)]
            for e in EDGES
        ]

    def some_other_method(self):
        return "delegated"


def _edges_of(goals_per_edge):
    return sorted({g.edge_idx for group in goals_per_edge for g in group})


def _planner(**params):
    params.setdefault("best_first_prior", "uniform")
    return BestFirstRegionOpeningPlanner(_StubEnv(), PlannerConfig(algorithm_params=params))


def test_blacklisted_edges_never_reach_the_pool():
    filt = _BlacklistedEdgeFilter(_FakePrim(), {TARGET: {1, 3}})

    goals = filt.generate_goals(TARGET, None, None)

    assert _edges_of(goals) == [0, 2]


def test_other_objects_are_untouched():
    filt = _BlacklistedEdgeFilter(_FakePrim(), {TARGET: {1, 3}})

    goals = filt.generate_goals(OTHER, None, None)

    assert _edges_of(goals) == list(EDGES)


def test_every_depth_of_a_blacklisted_edge_is_dropped():
    """The blacklist is per-edge, not per (edge, depth) -- a stuck edge stays stuck."""
    filt = _BlacklistedEdgeFilter(_FakePrim(), {TARGET: {0}})

    goals = filt.generate_goals(TARGET, None, None)

    assert all(g.edge_idx != 0 for group in goals for g in group)
    assert len([g for group in goals for g in group]) == (len(EDGES) - 1) * DEPTHS_PER_EDGE


def test_unrelated_attributes_delegate_to_the_wrapped_strategy():
    filt = _BlacklistedEdgeFilter(_FakePrim(), {TARGET: {0}})

    assert filt.some_other_method() == "delegated"


def test_planner_wraps_its_enumerator_when_a_blacklist_is_supplied():
    planner = _planner(external_edge_blacklist={TARGET: [1, 3]})

    assert isinstance(planner._search_planner.prim, _BlacklistedEdgeFilter)
    assert planner.external_edge_blacklist == {TARGET: {1, 3}}


def test_planner_leaves_enumeration_alone_when_there_is_no_blacklist():
    planner = _planner()

    assert isinstance(planner._search_planner.prim, PrimitiveGoalStrategy)
    assert planner.external_edge_blacklist == {}


@pytest.mark.parametrize("supplied", [{TARGET: ["1", "3"]}, {TARGET: (1, 3)}])
def test_blacklist_is_normalised_to_ints(supplied):
    """robot_control sends these over a config boundary; strings must still work."""
    planner = _planner(external_edge_blacklist=supplied)

    assert planner.external_edge_blacklist == {TARGET: {1, 3}}
