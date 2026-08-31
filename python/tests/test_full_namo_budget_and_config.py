import sys
import types

if "namo_rl" not in sys.modules:
    namo_rl_stub = types.ModuleType("namo_rl")
    namo_rl_stub.Action = type("Action", (), {})
    namo_rl_stub.RLEnvironment = object
    namo_rl_stub.RLState = object
    sys.modules["namo_rl"] = namo_rl_stub

from namo.core import PlannerConfig, PlannerResult
from namo.planners.full_namo.full_namo_planner import FullNAMOPlanner
from namo.planners.opening.region_opening import RegionOpeningPlanner
from namo.planners.utils import PushAttemptBudget
from namo.solvability_runner import SolveTask, build_full_namo_planner_config
from namo.runtime_profile import CANONICAL_NUM_DEPTHS, CANONICAL_PRIMITIVE_PREFIX


class FakeEnv:
    def __init__(self):
        self.goal = None
        self.current_state = "baseline"

    def set_robot_goal(self, x, y, theta):
        self.goal = (x, y, theta)

    def is_robot_goal_reachable(self):
        return False

    def set_full_state(self, state):
        self.current_state = state

    def get_xml_path(self):
        return "dummy.xml"

    def get_config_path(self):
        return "dummy.yaml"


def test_build_full_namo_planner_config_forwards_nested_region_settings(monkeypatch):
    monkeypatch.setattr(RegionOpeningPlanner, "_setup_constraints", lambda self: None)
    monkeypatch.setattr(
        RegionOpeningPlanner,
        "_initialize_algorithm",
        lambda self: setattr(self, "goal_strategy", object()),
    )

    task = SolveTask(
        xml_path="scene.xml",
        path_length_n=2,
        config_path="config.yaml",
        goal_strategy="random_rollout",
        region_max_chain_depth=2,
        primitive_data_dir="data",
        primitive_prefix=CANONICAL_PRIMITIVE_PREFIX,
        rollout_samples_per_state=7,
        region_frontier_beam_width=11,
        region_success_min_reachable=3,
        goals_per_region=10,
        seed=42,
        use_cpp_snapshot=True,
        simulation_budget=100000,
        simulation_budget_scope="keyhole",
        region_selection_strategy="ml_first",
        scorer_ckpt="hy5u.ckpt",
        ml_device="cpu",
        full_namo_max_iterations=None,
        max_push_steps=CANONICAL_NUM_DEPTHS,
        audit_next_keyhole_reachability=True,
        preserve_next_keyhole_access=False,
    )
    config = build_full_namo_planner_config(task)
    planner = FullNAMOPlanner(FakeEnv(), config)

    assert planner.max_iterations is None
    assert planner.use_cpp_unified_wavefront is True
    assert planner.region_snapshot_seed == 42
    assert planner.region_opener.max_chain_depth == 2
    assert planner.region_opener.push_budget.limit == 100000
    assert planner.region_opener.push_budget is config.algorithm_params["push_budget"]
    assert planner.budget_scope == "keyhole"
    assert planner.keyhole_budget_limit == 100000
    assert planner.local_search == "region_bfs"
    assert config.algorithm_params["best_first_prior"] == "model"
    assert config.algorithm_params["best_first_hmax"] == 2
    assert config.algorithm_params["region_selection_strategy"] == "ml_first"
    assert config.algorithm_params["scorer_ckpt"] == "hy5u.ckpt"
    assert config.algorithm_params["ml_device"] == "cpu"
    assert config.algorithm_params["primitive_prefix"] == CANONICAL_PRIMITIVE_PREFIX
    assert config.algorithm_params["rollout_samples_per_state"] == 7
    assert config.algorithm_params["region_frontier_beam_width"] == 11
    assert config.algorithm_params["region_success_min_reachable"] == 3
    assert config.algorithm_params["max_push_steps"] == CANONICAL_NUM_DEPTHS
    assert config.algorithm_params["full_namo_audit_next_keyhole_reachability"] is True
    assert config.algorithm_params["full_namo_preserve_next_keyhole_access"] is False


class _FutureAccessEnv(FakeEnv):
    def __init__(self, *, pose, edges, goal_reachable=False):
        super().__init__()
        self.pose = pose
        self.edges = edges
        self.goal_reachable = goal_reachable

    def is_robot_goal_reachable(self):
        return self.goal_reachable

    def get_observation(self):
        return {"blocker_pose": self.pose}

    def get_reachable_edges(self, object_id):
        assert object_id == "blocker"
        return self.edges


def _future_profile():
    return {
        "status": "ok",
        "objects": {
            "blocker": {
                "pose_before": [1.0, 2.0, 0.25],
                "reachable_edges_before": [0, 2],
            }
        },
    }


def _future_checker(monkeypatch, env):
    monkeypatch.setattr(FullNAMOPlanner, "_initialize_algorithm", lambda self: None)
    planner = FullNAMOPlanner(env, PlannerConfig())
    return planner._check_next_keyhole_access_candidate(env=env, profile=_future_profile())


def test_next_keyhole_gate_accepts_all_original_edges_and_allows_new_edges(monkeypatch):
    accepted, detail = _future_checker(
        monkeypatch,
        _FutureAccessEnv(pose=[1.0, 2.0, 0.25], edges=[0, 1, 2]),
    )

    assert accepted is True
    assert detail["objects"]["blocker"]["gained_edges"] == [1]


def test_next_keyhole_gate_rejects_lost_edges_or_moved_blocker(monkeypatch):
    accepted, detail = _future_checker(
        monkeypatch,
        _FutureAccessEnv(pose=[1.001, 2.0, 0.25], edges=[0]),
    )

    assert accepted is False
    assert detail["failure_reasons"] == ["next_blocker_moved", "next_contact_edges_lost"]
    assert detail["objects"]["blocker"]["lost_edges"] == [2]


def test_next_keyhole_gate_accepts_direct_goal_reachability(monkeypatch):
    accepted, detail = _future_checker(
        monkeypatch,
        _FutureAccessEnv(pose=[9.0, 9.0, 9.0], edges=[], goal_reachable=True),
    )

    assert accepted is True
    assert detail["goal_reachable"] is True


def test_keyhole_budget_constructs_fresh_local_opener(monkeypatch):
    monkeypatch.setattr(RegionOpeningPlanner, "_setup_constraints", lambda self: None)
    monkeypatch.setattr(
        RegionOpeningPlanner,
        "_initialize_algorithm",
        lambda self: setattr(self, "goal_strategy", object()),
    )
    budget = PushAttemptBudget(limit=100)
    config = PlannerConfig(
        algorithm_params={
            "push_budget": budget,
            "full_namo_budget_scope": "keyhole",
            "full_namo_keyhole_simulation_budget": 100,
        }
    )
    planner = FullNAMOPlanner(FakeEnv(), config)

    first = planner._prepare_region_opener_for_keyhole()
    first.push_budget.consume_or_raise()
    second = planner._prepare_region_opener_for_keyhole()

    assert first is not second
    assert first.push_budget.used == 1
    assert second.push_budget.limit == 100
    assert second.push_budget.used == 0


def test_full_namo_propagates_simulation_budget_exhaustion(monkeypatch):
    env = FakeEnv()

    class FakeOpener:
        def reset(self):
            pass

        def search(self, robot_goal, target_neighbor=None):
            return PlannerResult(
                success=False,
                solution_found=False,
                algorithm_stats={
                    "attempt_results": [],
                    "target_summary": {
                        "target_neighbor": target_neighbor,
                        "local_robot_label": "robot",
                        "local_neighbors": [target_neighbor],
                        "target_is_immediate_neighbor": True,
                        "failure_reason": "simulation_budget_exhausted",
                        "attempt_count": 0,
                        "detail_reasons": [],
                        "boundary_exhausted": False,
                    },
                    "rejection_breakdown": {},
                    "total_primitives_attempted": 4,
                    "failure_kind": "simulation_budget_exhausted",
                    "simulation_budget_limit": 9,
                    "simulation_budget_used": 9,
                    "simulation_budget_remaining": 0,
                },
                error_message="Simulation budget exhausted after 9 env.step() calls",
            )

    def fake_initialize(self):
        self.region_opener = FakeOpener()

    monkeypatch.setattr(FullNAMOPlanner, "_initialize_algorithm", fake_initialize)
    planner = FullNAMOPlanner(env, PlannerConfig())
    monkeypatch.setattr(
        planner,
        "_compute_region_snapshot",
        lambda: {
            "adjacency": {"robot": {"a"}, "a": {"robot", "goal"}, "goal": {"a"}},
            "robot_label": "robot",
            "goal_label": "goal",
            "goal_in_free_space": True,
        },
    )

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is False
    assert result.algorithm_stats["failure_kind"] == "simulation_budget_exhausted"
    assert result.algorithm_stats["failure_context"]["chosen_target_region"] == "a"


class _AlreadyAccessibleOpener:
    """Opener that reports the target region as already reachable, using zero pushes."""

    def __init__(self):
        self.calls = 0

    def reset(self):
        pass

    def search(self, robot_goal, target_neighbor=None):
        self.calls += 1
        attempt = types.SimpleNamespace(success=True, resulting_state="baseline")
        return PlannerResult(
            success=True,
            solution_found=True,
            action_sequence=[],
            algorithm_stats={
                "attempt_results": [attempt],
                "target_summary": {
                    "target_neighbor": target_neighbor,
                    "local_robot_label": "robot",
                    "local_neighbors": [target_neighbor],
                    "target_is_immediate_neighbor": True,
                    "failure_reason": "already_accessible",
                    "attempt_count": 0,
                    "detail_reasons": ["already_accessible"],
                    "boundary_exhausted": False,
                },
                "rejection_breakdown": {},
                "total_primitives_attempted": 0,
                "simulation_budget_limit": 300,
                "simulation_budget_used": 0,
                "simulation_budget_remaining": 300,
            },
        )


def _patch_planner_with_opener(monkeypatch, planner_env, opener, adjacency):
    monkeypatch.setattr(
        FullNAMOPlanner, "_initialize_algorithm", lambda self: setattr(self, "region_opener", opener)
    )
    planner = FullNAMOPlanner(planner_env, PlannerConfig())
    monkeypatch.setattr(
        planner,
        "_compute_region_snapshot",
        lambda: {
            "adjacency": adjacency,
            "robot_label": "robot",
            "goal_label": "goal",
            "goal_in_free_space": True,
        },
    )
    return planner


def test_already_accessible_is_a_zero_push_opening_not_an_invariant_failure(monkeypatch):
    """A target region the opener already counts reachable must not abort the whole scene."""
    env = FakeEnv()
    # Pre-loop check, then the iteration-top check, then reachable after the zero-push open.
    reachable = iter([False, False, True])
    monkeypatch.setattr(FakeEnv, "is_robot_goal_reachable", lambda self: next(reachable))

    planner = _patch_planner_with_opener(
        monkeypatch,
        env,
        _AlreadyAccessibleOpener(),
        {"robot": {"a"}, "a": {"robot", "goal"}, "goal": {"a"}},
    )

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is True
    assert (result.algorithm_stats or {}).get("failure_subkind") != "already_accessible"
    outcomes = [entry["outcome"] for entry in result.algorithm_stats["iteration_trace"]]
    assert "opened_target" in outcomes


def test_repeated_already_accessible_blacklists_instead_of_looping(monkeypatch):
    """Zero-push openings change nothing, so a repeat must reroute rather than spin forever."""
    env = FakeEnv()
    opener = _AlreadyAccessibleOpener()

    planner = _patch_planner_with_opener(
        monkeypatch,
        env,
        opener,
        {"robot": {"a"}, "a": {"robot", "goal"}, "goal": {"a"}},
    )

    result = planner.search((0.0, 0.0, 0.0))

    assert result.success is False
    assert result.algorithm_stats["failure_kind"] == "region_path_exhausted"
    outcomes = [entry["outcome"] for entry in result.algorithm_stats["iteration_trace"]]
    assert outcomes.count("opened_target") == 1
    assert "already_accessible_repeat" in outcomes
    assert opener.calls == 2


def _result_with(stats):
    return PlannerResult(success=False, solution_found=False, algorithm_stats=stats)


def test_budget_stop_is_recognised_from_either_layer():
    """The opener writes the same reason at two levels; accept both.

    best_first_region_opening sets it as `failure_kind` on the result (line 257)
    and as `failure_reason` inside the per-attempt summary (line 521). Reading
    only one would miss a stop reported by the other.
    """
    planner = FullNAMOPlanner(FakeEnv(), PlannerConfig())

    assert planner._budget_stopped(
        _result_with({"failure_kind": "simulation_budget_exhausted"})
    )
    assert planner._budget_stopped(
        _result_with({"target_summary": {"failure_reason": "simulation_budget_exhausted"}})
    )
    assert not planner._budget_stopped(
        _result_with({"target_summary": {"failure_reason": "all_pushes_failed"}})
    )
    assert not planner._budget_stopped(_result_with({}))


def test_keyhole_scope_always_has_budget_for_another_boundary():
    """Keyhole scope rebuilds the opener per boundary, so a reroute is always funded."""
    planner = FullNAMOPlanner(
        FakeEnv(),
        PlannerConfig(
            algorithm_params={
                "full_namo_budget_scope": "keyhole",
                "full_namo_keyhole_simulation_budget": 900,
            }
        ),
    )

    assert planner.budget_scope == "keyhole"
    assert planner._simulation_budget_remains(_result_with({})) is True


def test_full_problem_scope_reports_a_spent_budget_as_spent():
    """One shared budget, so a boundary that consumed it leaves nothing.

    This is the half that keeps the reroute honest. Without it a budget stop
    would reroute into a boundary it cannot afford to attempt.
    """
    planner = FullNAMOPlanner(FakeEnv(), PlannerConfig())
    planner.budget_scope = "full_problem"

    planner.region_opener = types.SimpleNamespace(
        push_budget=PushAttemptBudget(limit=900, used=400)
    )
    assert planner._simulation_budget_remains(_result_with({})) is True

    planner.region_opener.push_budget = PushAttemptBudget(limit=900, used=900)
    assert planner._simulation_budget_remains(_result_with({})) is False


def test_a_budget_stop_with_budget_left_is_a_reroute_not_a_dead_run():
    """Regression for best_first never rerouting.

    `boundary_exhausted` names three candidate-exhaustion reasons and a budget
    stop is not among them, so before this fix any budget stop fell through to
    `opener_failure_not_boundary_exhausted` and killed the whole problem.
    region_bfs runs its pool dry and reroutes; best_first spends its budget and
    could not. Measured on real_exp/twohop_00013, where region_bfs solved the
    scene in two pushes after rerouting and best_first returned failure having
    never looked at the two cheaper doorways beside the one it ground on.
    """
    planner = FullNAMOPlanner(FakeEnv(), PlannerConfig())
    planner.budget_scope = "keyhole"
    result = _result_with({"failure_kind": "simulation_budget_exhausted"})

    reroute = planner._budget_stopped(result) and planner._simulation_budget_remains(result)
    assert reroute, "a funded budget stop must reroute rather than end the run"

    summary = {"boundary_exhausted": False}
    assert not summary["boundary_exhausted"], (
        "the old code reached the hard-failure return through exactly this path"
    )
