"""Uniform Rollout Sampler — fresh 1-push F-characterization with chain-extendable schema.

v0 collects depth-0 exhaustive data only. The schema (TransitionRecord, EnvMetadata,
SamplerAttemptResult) is designed so a follow-up spec can append depth-1 / depth-2
records without breaking existing readers. See docs/superpowers/specs/
2026-05-19-uniform-rollout-sampler-design.md for the design.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import namo_rl
from namo.core import BasePlanner, PlannerConfig, PlannerFactory, PlannerResult


class UniformRolloutSampler(BasePlanner):
    """Exhaustively executes every reachable push primitive at the initial scene.

    Does no search. Logs every outcome. Output flows through the existing
    region_opening-style worker branch by emitting one AttemptResult per
    (object, neighbor) pair.
    """

    def __init__(self, env: namo_rl.RLEnvironment, config: PlannerConfig):
        super().__init__(env, config)

    def _setup_constraints(self) -> None:
        # No constraints needed: every reachable primitive is enumerated.
        pass

    def _initialize_algorithm(self) -> None:
        # v0 has no internal algorithm state — every search() call is independent.
        pass

    def reset(self) -> None:
        pass

    @property
    def algorithm_name(self) -> str:
        return "uniform_rollout_sampler"

    @property
    def algorithm_version(self) -> str:
        return "0.1.0"

    def search(self, robot_goal: Tuple[float, float, float]) -> PlannerResult:
        # Skeleton: implemented in later tasks. Returns empty result for now.
        return PlannerResult(
            success=False,
            solution_found=False,
            action_sequence=None,
            algorithm_stats={"attempt_results": []},
        )


PlannerFactory.register_planner("uniform_rollout_sampler", UniformRolloutSampler)
