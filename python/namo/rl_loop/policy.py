"""Policy = candidate scorer for the rollout.

Two arms, one interface:
  - arm A / gen-0 : uniform pi0 — NO model is loaded; every reachable candidate scores 0
    (the sampler then treats them uniformly). We still need PrimitiveGoalStrategy to
    ENUMERATE the executable (edge, depth) candidates, so we build a stub planner that has
    `.prim` but no `.scorer`.
  - arm B / gen>0 : a trained ckpt — full BeamPlanner (LiveScorer + PrimitiveGoalStrategy).

Both delegate candidate enumeration + scoring to the validated sandbox primitive
`eval_m3.rank_first_pushes_h2` (object-restricted, budget-h, score on/off), so the
loop's scoring is byte-identical to the deploy/eval path (eval_reactive_argmax).
"""
from typing import List, Optional, Tuple

from ._bootstrap import ensure_paths
ensure_paths()
from scorer_beam import BeamPlanner, DATA_DIR, PRIM_PREFIX          # noqa: E402
from eval_m3 import rank_first_pushes_h2                            # noqa: E402
from namo.strategies import PrimitiveGoalStrategy                   # noqa: E402


class _UniformPlanner:
    """Minimal planner surface for the uniform arm: enumerate candidates, no model."""
    def __init__(self):
        self.prim = PrimitiveGoalStrategy(data_dir=DATA_DIR, primitive_prefix=PRIM_PREFIX)
        self.scorer = None


class Policy:
    def __init__(self, ckpt: Optional[str] = None, score_h: int = 1):
        self.uniform = ckpt is None
        self.score_h = score_h
        self.ckpt = ckpt
        self.planner = _UniformPlanner() if self.uniform else BeamPlanner(ckpt=ckpt)

    def score_pool(self, env, robot_goal, xml: str, s_cur, restrict_obj: Optional[str]
                   ) -> List[Tuple[str, object, float]]:
        """[(obj, Goal, score)] over reachable (obj, edge, depth) at s_cur, desc by score.
        Leaves env at an arbitrary state (caller must set_full_state(s_cur) before stepping).
        Uniform arm returns score=0 for every candidate (sampler => uniform)."""
        return rank_first_pushes_h2(
            self.planner, env, robot_goal, xml, s_cur, self.score_h,
            restrict_obj=restrict_obj, score=(not self.uniform),
        )
