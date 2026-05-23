"""Unit tests for v1 chain mode of UniformRolloutSampler.

These exercise the chain-mode code path directly without spinning up MuJoCo —
focuses on the schema disambiguation, knob validation, and ChainGoal record
shape. Integration tests that actually run pushes live in
test_uniform_rollout_sampler_integration.py.
"""

from __future__ import annotations

import pytest

from namo.core import PlannerConfig
from namo.data_collection.modular_parallel_collection import (
    _filter_trial_log_to_chain_depth_zero,
)
from namo.planners.sampling.uniform_rollout_sampler import (
    UniformRolloutSampler,
    _ChainGoal,
)


def _make_config(**overrides):
    """Build a chain-mode algorithm_params dict (no env actually exercised)."""
    algo = {
        "max_chain_depth": 2,
        "num_rollouts_per_env": 10,
        "max_solutions_per_env": 1,
        "num_depths": 10,
        "primitive_data_dir": "data",
        "primitive_prefix": "",
        "seed": 42,
        "env_idx": 0,
        "region_goal_samples_per_neighbor": 5,
    }
    algo.update(overrides)
    return PlannerConfig(
        max_depth=5,
        max_goals_per_object=5,
        max_terminal_checks=5000,
        max_search_time_seconds=300.0,
        goals_per_region=5,
        verbose=False,
        collect_stats=True,
        algorithm_params=algo,
    )


# ---------------------------------------------------------------------------
# Schema disambiguation: primitive_trial_log filter
# ---------------------------------------------------------------------------


def test_filter_v0_records_pass_through():
    v0 = [
        {"edge_idx": 3, "depth": 5, "success": True},
        {"edge_idx": 4, "depth": 2, "success": False},
    ]
    assert _filter_trial_log_to_chain_depth_zero(v0) == v0


def test_filter_none_and_empty():
    assert _filter_trial_log_to_chain_depth_zero(None) is None
    assert _filter_trial_log_to_chain_depth_zero([]) == []


def test_filter_keeps_only_chain_depth_zero():
    mixed = [
        {"edge_idx": 3, "depth": 5, "success": True, "chain_depth": 0},
        {"edge_idx": 7, "depth": 2, "success": False, "chain_depth": 1},
        {"edge_idx": 8, "depth": 4, "success": True, "chain_depth": 0},
    ]
    out = _filter_trial_log_to_chain_depth_zero(mixed)
    assert len(out) == 2
    assert all(e["chain_depth"] == 0 for e in out)


def test_filter_rejects_invalid_chain_depth():
    with pytest.raises(ValueError):
        _filter_trial_log_to_chain_depth_zero(
            [{"edge_idx": 1, "depth": 1, "chain_depth": -1}]
        )
    with pytest.raises(ValueError):
        _filter_trial_log_to_chain_depth_zero(
            [{"edge_idx": 1, "depth": 1, "chain_depth": "two"}]
        )


# ---------------------------------------------------------------------------
# Knob validation
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_env():
    """Minimal env stub that satisfies BasePlanner's __init__ requirements."""
    class _StubEnv:
        def get_action_constraints(self):
            class _C: pass
            c = _C()
            c.min_distance = 0.0
            c.max_distance = 1.0
            c.theta_min = 0.0
            c.theta_max = 6.28
            return c
        # Not exercised but kept so attribute access doesn't blow up.
        def get_object_info(self): return {}
        def get_observation(self): return {}
        def get_reachable_objects(self): return []
    return _StubEnv()


def test_max_chain_depth_must_be_in_range(fake_env):
    for bad in [-1, 0, 11, 999]:
        cfg = _make_config(max_chain_depth=bad)
        with pytest.raises(ValueError):
            UniformRolloutSampler(fake_env, cfg)


def test_num_rollouts_must_be_at_least_1(fake_env):
    cfg = _make_config(num_rollouts_per_env=0)
    with pytest.raises(ValueError):
        UniformRolloutSampler(fake_env, cfg)


def test_max_solutions_must_be_at_least_1(fake_env):
    cfg = _make_config(max_solutions_per_env=0)
    with pytest.raises(ValueError):
        UniformRolloutSampler(fake_env, cfg)


def test_per_env_seeding(fake_env):
    """seed = seed_base + env_idx (the v0 'constant seed' bug fix)."""
    cfg1 = _make_config(seed=42, env_idx=0)
    p1 = UniformRolloutSampler(fake_env, cfg1)
    assert p1.seed == 42

    cfg2 = _make_config(seed=42, env_idx=7)
    p2 = UniformRolloutSampler(fake_env, cfg2)
    assert p2.seed == 49

    cfg3 = _make_config(seed=100, env_idx=3)
    p3 = UniformRolloutSampler(fake_env, cfg3)
    assert p3.seed == 103


def test_default_knobs_are_sane(fake_env):
    """v1 defaults from the spec."""
    cfg = _make_config()
    p = UniformRolloutSampler(fake_env, cfg)
    assert p.max_chain_depth == 2
    assert p.num_rollouts_per_env == 10  # overridden in test fixture; spec is 10000
    assert p.max_solutions_per_env == 1


def test_n1_passes_validation(fake_env):
    """n=1 is the v0 fallback path; must initialize cleanly."""
    cfg = _make_config(max_chain_depth=1)
    p = UniformRolloutSampler(fake_env, cfg)
    assert p.max_chain_depth == 1


# ---------------------------------------------------------------------------
# _ChainGoal record shape (worker reads via attribute access)
# ---------------------------------------------------------------------------


def test_chain_goal_from_record():
    rec = {
        "object_id": "obstacle_3_movable",
        "edge_idx": 17,
        "push_depth_idx": 4,
        "target_pose": (1.5, -2.3, 0.7),
        "chain_depth": 1,
    }
    g = _ChainGoal.from_record(rec)
    assert g.x == pytest.approx(1.5)
    assert g.y == pytest.approx(-2.3)
    assert g.theta == pytest.approx(0.7)
    assert g.edge_idx == 17
    assert g.depth == 4  # cookbook push_depth_idx, not chain_depth


def test_chain_goal_has_worker_required_fields():
    """The worker reads .x, .y, .theta, .edge_idx, .depth via attribute access."""
    g = _ChainGoal(x=0.1, y=0.2, theta=0.3, edge_idx=5, depth=7)
    for attr in ("x", "y", "theta", "edge_idx", "depth"):
        assert hasattr(g, attr)
