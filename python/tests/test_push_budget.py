from types import SimpleNamespace

from namo.core import PlannerConfig
from namo.planners.opening.region_opening import RegionOpeningPlanner
from namo.planners.utils import PushAttemptBudget, PushBudgetExceeded
from namo.strategies.goal_selection_strategy import Goal


class _BudgetEnv:
    def __init__(self):
        self.step_calls = 0
        self.state = {"name": "baseline"}

    def set_full_state(self, state):
        self.state = state

    def get_observation(self):
        return {}

    def get_reachable_objects(self):
        return []

    def step(self, action):
        self.step_calls += 1
        return SimpleNamespace(info={}, done=False, reward=0.0)

    def get_full_state(self):
        return self.state

    def set_collision_checking(self, _enabled):
        return None


def _make_planner(monkeypatch, env, *, budget_limit):
    monkeypatch.setattr(RegionOpeningPlanner, "_setup_constraints", lambda self: None)
    monkeypatch.setattr(
        RegionOpeningPlanner,
        "_initialize_algorithm",
        lambda self: setattr(self, "goal_strategy", object()),
    )
    return RegionOpeningPlanner(
        env,
        PlannerConfig(
            verbose=False,
            algorithm_params={"push_budget": PushAttemptBudget(limit=budget_limit)},
        ),
    )


def test_collect_chain_observations_consumes_budget_before_env_step(monkeypatch):
    env = _BudgetEnv()
    planner = _make_planner(monkeypatch, env, budget_limit=1)
    baseline = {"name": "baseline"}
    goal_chain = [
        Goal(x=0.0, y=0.0, theta=0.0, score=1.0, edge_idx=0, depth=0),
        Goal(x=1.0, y=0.0, theta=0.0, score=1.0, edge_idx=1, depth=0),
    ]

    try:
        planner._collect_chain_observations("box", goal_chain, baseline)
        assert False, "Expected PushBudgetExceeded"
    except PushBudgetExceeded as exc:
        assert exc.limit == 1
        assert exc.used == 1

    assert env.step_calls == 1
    assert planner.push_budget.used == 1


def test_search_reports_simulation_budget_exhausted(monkeypatch):
    env = _BudgetEnv()
    planner = _make_planner(monkeypatch, env, budget_limit=3)

    def fake_explore(_state, level=0, target_neighbor=None):
        raise PushBudgetExceeded(limit=3, used=3)

    monkeypatch.setattr(planner, "_explore_from_state", fake_explore)

    result = planner.search((0.0, 0.0, 0.0), target_neighbor="goal")

    assert result.success is False
    assert result.algorithm_stats["failure_kind"] == "simulation_budget_exhausted"
    assert result.algorithm_stats["simulation_budget_limit"] == 3
    assert result.algorithm_stats["simulation_budget_used"] == 0


def test_search_bfs_propagates_budget_exhaustion_instead_of_swallowing(monkeypatch):
    env = _BudgetEnv()
    planner = _make_planner(monkeypatch, env, budget_limit=0)
    planner._progress_total_primitives = 0
    planner._progress_last_print_time = 0.0
    planner._progress_last_print_count = 0
    planner._progress_interval_sec = 10_000.0
    planner._rejection_stats = {}
    monkeypatch.setattr(planner, "_record_push_exec_timing", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(planner, "_record_primitive_ranking_timing", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(planner, "_focus_camera_on_object", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(planner, "_validate_opening", lambda *_args, **_kwargs: (False, 0, None, []))

    try:
        planner._search_bfs(
            goals_per_edge=[[Goal(x=0.0, y=0.0, theta=0.0, score=1.0, edge_idx=0, depth=0)]],
            reachable_edge_indices={0},
            baseline_state={"name": "baseline"},
            neighbour_label="goal",
            region_goals={},
            object_id="box",
        )
        assert False, "Expected PushBudgetExceeded"
    except PushBudgetExceeded as exc:
        assert exc.limit == 0
        assert exc.used == 0
