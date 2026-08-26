"""The decision rule you asked for is the one that runs, or the call refuses.

``solve_boundary_from_xml`` forwards anything it does not name straight into
``algorithm_params``, which the planners treat as an opaque bag. A key no
planner reads is dropped without a word. That is not hypothetical: on
2026-08-21 a hardware run went out with ``--local-search best_first
--scorer-ckpt HY5U_s2`` aimed at a planner that reads neither, swept 2,959
primitives over 394 s with no ranker loaded, and read exactly like a hard scene
instead of a misrouted flag. robot_control built ``check_search_reaches_planner``
against that.

A reactive mode is the sharper version of the same trap. It is a comparison arm,
so a run recorded under the wrong arm does not fail, it produces a number, and
that number goes in a table next to a number from the other arm. Silence here
does not cost an afternoon of debugging, it corrupts a result.

So ``mode`` is a named parameter with a closed vocabulary, checked before
anything is constructed, never a key riding into the bag.

Five properties, each with the failure it catches:

  mode is named, not forwarded    a typo lands in algorithm_params, is read by
                                  nobody, and the run silently does the default
  the default is unchanged        adding the switch quietly re-points every
                                  caller that never asked for it
  an unknown mode refuses         "policy", "argmax", "reactve" all become the
                                  search arm without a word
  reactive needs the ranker pool  region_bfs sweeps every edge and depth and
                                  builds no ranked pool, so there is no argmax
                                  to take; asking for one has to say so
  the planner receives it         the service validates, then drops it on the
                                  floor, which passes every check above

Pure argument plumbing, so no binding physics, no checkpoint, no scene.

To verify:
  cd namo_cpp && source env.ilab.sh
  python -m pytest python/tests/test_boundary_mode_routing.py -v
"""

from __future__ import annotations

import inspect

import pytest

from namo.core import PlannerConfig
from namo.planners.opening.best_first_region_opening import BestFirstRegionOpeningPlanner
from namo.services.planning_service import (
    BOUNDARY_MODES,
    DEFAULT_BOUNDARY_MODE,
    NAMOPlanningService,
)


# ─── Named constants ────────────────────────────────────────────────────

MODE_SEARCH = "search"
MODE_REACTIVE = "reactive"

# Enough to satisfy solve_boundary_from_xml's own argument checks, so a test
# reaches the mode validation rather than tripping on something earlier.
A_POINT = [(0.3, 0.4)]
A_GOAL = (0.37, 0.67, 0.0)


class _StubEnv:
    def get_full_state(self):
        return {"name": "baseline"}

    def set_full_state(self, _state):
        return None


def _best_first(**params):
    """Uniform prior needs no checkpoint, so this needs no model or GPU."""
    params.setdefault("best_first_prior", "uniform")
    return BestFirstRegionOpeningPlanner(_StubEnv(), PlannerConfig(algorithm_params=params))


def _solve(**kwargs):
    service = NAMOPlanningService.__new__(NAMOPlanningService)
    return NAMOPlanningService.solve_boundary_from_xml(
        service, "scene.xml", A_GOAL, A_POINT, **kwargs
    )


# ─── Tests ──────────────────────────────────────────────────────────────


def test_mode_is_a_named_parameter():
    """Named, so a typo is a TypeError rather than an ignored bag key."""
    assert "mode" in inspect.signature(
        NAMOPlanningService.solve_boundary_from_xml
    ).parameters


def test_the_vocabulary_is_closed_and_defaults_to_search():
    """Every existing caller keeps searching without being edited."""
    assert set(BOUNDARY_MODES) == {MODE_SEARCH, MODE_REACTIVE}
    assert DEFAULT_BOUNDARY_MODE == MODE_SEARCH
    assert (
        inspect.signature(NAMOPlanningService.solve_boundary_from_xml)
        .parameters["mode"]
        .default
        == MODE_SEARCH
    )


@pytest.mark.parametrize("mode", ["policy", "argmax", "reactve", "", "Search"])
def test_an_unknown_mode_refuses(mode):
    """Anything outside the vocabulary is a mistake, not a synonym for the default."""
    with pytest.raises(ValueError, match="mode"):
        _solve(mode=mode, local_search="best_first")


def test_reactive_refuses_the_planner_that_has_no_ranked_pool():
    """region_bfs sweeps every edge and depth; there is no argmax to take.

    The message has to name the way out, because a caller reading only the
    refusal should not have to find the pairing rule themselves.
    """
    with pytest.raises(ValueError) as excinfo:
        _solve(mode=MODE_REACTIVE, local_search="region_bfs")

    message = str(excinfo.value)
    assert "best_first" in message, f"the refusal must name the way out: {message}"


@pytest.mark.parametrize("mode", [MODE_SEARCH, MODE_REACTIVE])
def test_the_planner_reads_the_mode_it_was_given(mode):
    """Validating and then dropping it would pass every check above."""
    assert _best_first(decision_rule=mode).decision_rule == mode


def test_the_planner_defaults_to_search():
    assert _best_first().decision_rule == MODE_SEARCH


def test_the_planner_refuses_an_unknown_decision_rule():
    """The service is not the only door into the planner."""
    with pytest.raises(ValueError, match="decision_rule"):
        _best_first(decision_rule="policy")
