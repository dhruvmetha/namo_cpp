"""best_first must default to the protocol its published numbers were measured under.

Two silent couplings used to break that:

* best_first_hmax fell back to region_max_chain_depth. NAMOPlanningService.
  plan_from_xml defaults that to 1, so a service caller who asked for
  best-first got a depth-1 search while every registered evaluation ran at 2.
* the per-keyhole simulation budget defaulted to 100, against 900 in the
  registry's canonical protocol.

Neither failed loudly. Both produced results that looked like the evaluated
search and were not.
"""

from types import SimpleNamespace

import pytest

from namo.core import PlannerConfig
from namo.planners.opening.best_first_region_opening import (
    CANONICAL_BEST_FIRST_HMAX,
    CANONICAL_KEYHOLE_SIMULATION_BUDGET,
    BestFirstRegionOpeningPlanner,
)
from namo.planners.utils import PushAttemptBudget

# plan_from_xml's own default, and the value that used to leak into hmax.
SERVICE_DEFAULT_MAX_CHAIN_DEPTH = 1


class _StubEnv:
    def get_full_state(self):
        return {"name": "baseline"}

    def set_full_state(self, _state):
        return None



def _planner(**params):
    params.setdefault("best_first_prior", "uniform")
    return BestFirstRegionOpeningPlanner(_StubEnv(), PlannerConfig(algorithm_params=params))


def test_canonical_protocol_matches_the_registry():
    assert CANONICAL_BEST_FIRST_HMAX == 2
    assert CANONICAL_KEYHOLE_SIMULATION_BUDGET == 900


def test_hmax_defaults_to_the_evaluated_depth():
    assert _planner().hmax == CANONICAL_BEST_FIRST_HMAX


def test_hmax_does_not_inherit_region_max_chain_depth():
    """The regression: a service caller's max_chain_depth must not set search depth."""
    planner = _planner(region_max_chain_depth=SERVICE_DEFAULT_MAX_CHAIN_DEPTH)

    assert planner.hmax == CANONICAL_BEST_FIRST_HMAX


def test_explicit_hmax_still_wins():
    assert _planner(best_first_hmax=3, region_max_chain_depth=1).hmax == 3


@pytest.mark.parametrize("bad", [0, -1])
def test_nonsensical_hmax_is_rejected(bad):
    with pytest.raises(ValueError, match="best_first_hmax"):
        _planner(best_first_hmax=bad)


def test_budget_defaults_to_the_evaluated_budget():
    assert _planner().push_budget.limit == CANONICAL_KEYHOLE_SIMULATION_BUDGET


def test_explicit_budget_still_wins():
    assert _planner(full_namo_keyhole_simulation_budget=300).push_budget.limit == 300


def test_caller_supplied_budget_object_takes_precedence():
    budget = PushAttemptBudget(limit=7)

    planner = _planner(push_budget=budget, full_namo_keyhole_simulation_budget=300)

    assert planner.push_budget is budget


def test_candidate_acceptor_keeps_searching_after_first_local_opening(monkeypatch):
    class CandidateEnv(_StubEnv):
        def __init__(self):
            self.reachable = False
            self.state = {"name": "baseline"}

        def get_full_state(self):
            return self.state

        def set_full_state(self, state):
            self.state = state

        def count_reachable_points(self, points):
            return (len(points) if self.reachable else 0), -1

    env = CandidateEnv()
    planner = BestFirstRegionOpeningPlanner(
        env,
        PlannerConfig(
            algorithm_params={
                "best_first_prior": "uniform",
                "region_success_min_reachable": 1,
                "region_min_reachable_fraction": 0.0,
            }
        ),
    )
    snapshot = {
        "robot_label": "robot",
        "region_labels": {},
        "adjacency": {"robot": {"middle"}},
        "region_goals": {
            "middle": SimpleNamespace(goals=[SimpleNamespace(x=1.0, y=2.0, theta=0.0)])
        },
        "edge_objects": {"robot": {"middle": ["door"]}},
    }
    monkeypatch.setattr("namo.planners.get_region_snapshot", lambda *_args, **_kwargs: snapshot)

    def fake_solve_scene(*_args, is_open, solution_out, **_kwargs):
        env.reachable = True
        assert is_open(env) is False
        assert is_open(env) is True
        solution_out["plan"] = []
        solution_out["state"] = {"name": "accepted"}
        return True, 2, 0, [], "solved"

    monkeypatch.setattr(
        "namo.planners.opening.best_first_region_opening.solve_scene",
        fake_solve_scene,
    )
    decisions = iter(
        [
            (False, {"failure_reasons": ["next_contact_edges_lost"]}),
            (True, {"failure_reasons": []}),
        ]
    )

    result = planner.search(
        (0.0, 0.0, 0.0),
        target_neighbor="middle",
        candidate_acceptor=lambda _env: next(decisions),
    )

    assert result.success is True
    stats = result.algorithm_stats["future_interface"]
    assert stats["checks"] == 2
    assert stats["rejected"] == 1
    assert stats["accepted"] == 1
    assert stats["rejection_reasons"] == {"next_contact_edges_lost": 1}
