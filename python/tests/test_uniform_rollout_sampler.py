"""Tests for UniformRolloutSampler."""

import namo.planners.sampling.uniform_rollout_sampler  # noqa: F401 — registers on import
from namo.core import PlannerFactory


def test_uniform_rollout_sampler_is_registered():
    available = PlannerFactory.list_available_planners()
    assert "uniform_rollout_sampler" in available
