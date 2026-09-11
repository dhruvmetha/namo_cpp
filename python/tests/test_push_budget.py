from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("chain_length", [1, 2])
def test_bfs_success_at_budget_limit_needs_no_replay(monkeypatch, chain_length):
    from namo.planners.opening.region_opening import AttemptResult

    class SearchEnv(_BudgetEnv):
        def __init__(self):
            super().__init__()
            self.state = 0

        def get_observation(self):
            return {"box_pose": [self.state, 0, 0], "robot_pose": [self.state, 1, 0]}

        def get_reachable_edges(self, _object):
            return [0]

        def get_reachable_objects(self):
            return ["box", f"visible_{self.state}"]

        def step(self, action):
            self.step_calls += 1
            assert self.step_calls <= chain_length, "solution reconstruction must not replay pushes"
            self.state += 1
            return SimpleNamespace(info={"wall_collision": "true", "movable_collisions": "other"})

    env = SearchEnv()
    planner = _make_planner(monkeypatch, env, budget_limit=chain_length)
    planner.max_chain_depth = chain_length
    planner.selection_strategy = "cost_first"
    goal = Goal(x=1, y=0, theta=0, edge_idx=0, depth=0)
    planner.goal_strategy = SimpleNamespace(generate_goals=lambda *_a, **_k: [[goal]])
    monkeypatch.setattr(planner, "_focus_camera_on_object", lambda *_a: None)
    monkeypatch.setattr(planner, "_validate_opening", lambda *_a: (
        env.state >= chain_length, int(env.state >= chain_length), None, []))
    captured = []

    def explore(state, **_kwargs):
        rows, *_ = planner._search_with_chaining_bfs(
            "box", state, "goal", {}, max_solutions_to_collect=1)
        captured.extend(rows)
        return [AttemptResult(success=bool(rows), neighbour_region_label="goal")]

    monkeypatch.setattr(planner, "_explore_from_state", explore)
    result = planner.search((0, 0, 0), target_neighbor="goal")
    assert result.success
    assert env.step_calls == planner.push_budget.used == chain_length
    row = captured[0]
    assert len(row[0]) == chain_length
    assert [obs["box_pose"][0] for obs in row[1]] == list(range(chain_length))
    assert [obs["box_pose"][0] for obs in row[2]] == list(range(1, chain_length + 1))
    assert row[3] == chain_length
    assert row[6] == [["box", f"visible_{i}"] for i in range(chain_length)]
    assert row[7] == [["box", f"visible_{i}"] for i in range(1, chain_length + 1)]
    assert row[8] == row[9] == chain_length
    assert row[11:] == (True, 1)
    assert planner._get_runtime_timing_summary()["chain_observation_replay_calls"] == 0


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


@pytest.mark.parametrize(
    "object_dx,robot_dx,info,expected_attempts,expected_children",
    [
        (0.0, 0.0, {}, 2, 0),
        (5e-7, 0.0, {}, 2, 0),
        (2e-6, 0.0, {}, 2, 2),
        (0.0, 0.01, {}, 2, 2),
        (0.01, 0.0, {"collision_object": "wall"}, 2, 2),
        (0.01, 0.0, {"stuck": "true"}, 2, 2),
        (0.01, 0.0, {"failure_reason": "stuck", "stuck": "true"}, 1, 1),
        (0.0, 0.01, {"failure_reason": "placement failed"}, 1, 1),
        (0.0, 0.0, {"failure_reason": "stuck"}, 1, 0),
        (0.01, 0.0, {"wall_collision": "true"}, 2, 2),
    ],
)
def test_bfs_and_best_first_prune_the_same_root_attempts_and_children(
    monkeypatch, object_dx, robot_dx, info, expected_attempts, expected_children
):
    import random
    import namo.planners.opening.best_first_search as best_first
    from namo.planners.opening.region_opening import ChainNode

    class PruningEnv(_BudgetEnv):
        def __init__(self):
            super().__init__()
            self.state = (0.0, 0.0, -1)
            self.attempts = []

        def get_observation(self):
            return {"box_pose": [self.state[0], 0, 0], "robot_pose": [self.state[1], 0, 0]}

        def get_reachable_objects(self):
            return ["box"]

        def step(self, action):
            self.attempts.append((action.edge_idx, action.depth))
            self.state = (object_dx, robot_dx, action.depth)
            return SimpleNamespace(info=info)

    goals = [Goal(x=0, y=0, theta=0, edge_idx=0, depth=d) for d in range(2)]
    bfs_env = PruningEnv()
    planner = _make_planner(monkeypatch, bfs_env, budget_limit=20)
    planner._progress_total_primitives = 0
    planner._progress_last_print_time = 0.0
    planner._progress_last_print_count = 0
    planner._progress_interval_sec = 10_000.0
    planner._rejection_stats = {}
    monkeypatch.setattr(planner, "_focus_camera_on_object", lambda *_a: None)
    monkeypatch.setattr(planner, "_validate_opening", lambda *_a: (False, 0, None, []))
    root = ChainNode(state=bfs_env.state, goal=None, edge_idx=-1, depth=0)
    _, _, bfs_children, *_ = planner._search_bfs(
        [goals], {0}, root.state, "goal", {}, "box", parent_node=root, collect_frontier=True
    )

    best_env = PruningEnv()
    best_children = []

    def candidates(_planner, _env, _goal, _xml, state, *_a, **_kw):
        if state[2] == -1:
            return [("box", goal, 2.0 - goal.depth) for goal in goals], 1.0, None
        best_children.append(state)
        return [], 0.0, None

    monkeypatch.setattr(best_first, "candidates", candidates)
    best_first.solve_scene(
        None, best_env, (0, 0, 0), "fake.xml", best_env.state, 2, 20,
        "model", "mean5", "q", random.Random(0), restrict_obj="box", is_open=lambda _e: False
    )
    assert bfs_env.attempts == best_env.attempts
    assert len(bfs_env.attempts) == expected_attempts
    assert [node.state for node in bfs_children] == best_children
    assert len(bfs_children) == expected_children
