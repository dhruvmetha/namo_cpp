"""Selection contract for `goal_strategy` in RegionOpeningPlanner.

Two things are pinned here:

1. Each accepted name selects the strategy class it has always selected. This
   is a characterization test -- it exists so that removing a dead branch from
   the chain cannot silently change which class a *surviving* name resolves to.

2. An unrecognised name raises. Before this guard the chain ended in a bare
   ``else`` that fell through to PrimitiveGoalStrategy, so a typo or a stale
   config value ran a different experiment than the one requested and said
   nothing about it.

The strategy classes are replaced with markers because constructing the real
ones loads motion primitives from disk (and, for the scorer, torch). What is
under test is the dispatch, not the strategies.
"""

import pytest

from namo.core import PlannerConfig
from namo.planners.opening import region_opening as ro


# Strategy classes the chain can construct, by attribute name in the module.
PATCHED_STRATEGY_CLASSES = (
    "PrimitiveGoalStrategy",
    "RandomRolloutGoalStrategy",
    "MLPrimitiveGoalStrategy",
    "ScorerGoalStrategy",
    "GeometricTransportStrategy",
)

# A model path is only needed to get past the ml branch's own validation.
ML_PARAMS = {"ml_goal_model_path": "/nonexistent/model"}

# (goal_strategy value, expected class attribute name, extra algorithm_params)
SELECTION_CASES = [
    (None, "PrimitiveGoalStrategy", {}),
    ("primitive", "PrimitiveGoalStrategy", {}),
    ("PRIMITIVE", "PrimitiveGoalStrategy", {}),
    ("random_rollout", "RandomRolloutGoalStrategy", {}),
    ("random", "RandomRolloutGoalStrategy", {}),
    ("geometric", "GeometricTransportStrategy", {}),
    ("geometric_transport", "GeometricTransportStrategy", {}),
    ("scorer", "ScorerGoalStrategy", {}),
    ("f_scorer", "ScorerGoalStrategy", {}),
    ("ml", "MLPrimitiveGoalStrategy", ML_PARAMS),
    ("ml_primitive", "MLPrimitiveGoalStrategy", ML_PARAMS),
]


def _marker(name):
    """Stand-in for a strategy class that records which class was chosen."""

    class _Marker:
        selected_class_name = name

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    _Marker.__name__ = name
    return _Marker


@pytest.fixture
def patched_strategies(monkeypatch):
    for attr in PATCHED_STRATEGY_CLASSES:
        monkeypatch.setattr(ro, attr, _marker(attr))


def _select(algorithm_params):
    """Run only the strategy-selection logic, with no environment."""
    planner = object.__new__(ro.RegionOpeningPlanner)
    planner.algorithm_params = algorithm_params
    planner.config = PlannerConfig(verbose=False)
    planner._initialize_algorithm()
    return planner.goal_strategy


@pytest.mark.parametrize("name,expected,extra", SELECTION_CASES)
def test_accepted_name_selects_its_strategy(patched_strategies, name, expected, extra):
    params = dict(extra)
    if name is not None:
        params["goal_strategy"] = name

    assert _select(params).selected_class_name == expected


def test_unknown_name_raises_instead_of_defaulting(patched_strategies):
    with pytest.raises(ValueError) as excinfo:
        _select({"goal_strategy": "ml_asynk"})

    message = str(excinfo.value)
    assert "ml_asynk" in message
    assert "primitive" in message


def test_every_accepted_name_is_declared_valid():
    """The parametrized cases must be a subset of the declared name set."""
    tested = {name.lower() for name, _expected, _extra in SELECTION_CASES if name}

    assert tested <= ro.VALID_GOAL_STRATEGIES


def test_declared_names_are_all_dispatchable(patched_strategies):
    """Nothing in VALID_GOAL_STRATEGIES may fall through to the guard."""
    for name in sorted(ro.VALID_GOAL_STRATEGIES):
        params = {"goal_strategy": name, **ML_PARAMS}
        _select(params)  # must not raise
