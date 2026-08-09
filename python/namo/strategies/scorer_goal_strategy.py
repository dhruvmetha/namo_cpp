"""Scorer-guided goal strategy for Region Opening.

Wraps the champion 1-push F-scorer (`sharp` EdgeCrossAttn) as a RO goal strategy: it reuses
PrimitiveGoalStrategy's executable-Goal enumeration, then sets each Goal.score to the scorer's
P(this push opens the NEIGHBOUR region), so RO's `_search_bfs` tries high-P pushes first and finds
the opening in far fewer forward simulations.

Key correctness point: the scorer's `goal_sample_region` channel MUST be rendered from the
neighbour region samples (`region_goals_sampled`), not the robot goal — that matches how the scorer
was trained (per-adjacency RO openings). We pass them straight through.

n-push hook: `generate_goals` accepts an optional `remaining_depth`. Phase 1 uses the 1-push opener
V1 at every level (correct within-level ranker for the LAST push of any chain). Phase 2 will slot a
depth-conditioned value V_d (trained on v3 2-push data) here for first-push / frontier ranking.

NOTE: imports the LiveScorer bridge from scripts/sandbox lazily (TODO: promote live_scorer +
load_scorer into the package once this lever is validated).
"""
from pathlib import Path
from typing import List, Optional, Tuple

import namo_rl

from .goal_selection_strategy import Goal
from .primitive_goal_strategy import PrimitiveGoalStrategy

_SANDBOX = str(Path(__file__).resolve().parents[3] / "scripts/sandbox")
_DEFAULT_CKPT_RELATIVE = Path(
    "sage_outputs/scorer/sharp_s1/namo-classifier/9yizg6i8/checkpoints/"
    "epoch017-val_loss0.2713.ckpt"
)

# Process-level cache: load the (heavy) scorer model + renderer ONCE per (ckpt, config), so
# running RO across many scenes (modular_parallel_collection) doesn't reload it every scene.
_SCORER_CACHE = {}


def _default_checkpoint() -> str:
    """Resolve the historical scorer default only when that strategy is used."""
    from namo.paths import SCRATCH

    return str(SCRATCH / _DEFAULT_CKPT_RELATIVE)


def _get_scorer(ckpt, namo_config_path, device):
    key = (ckpt, namo_config_path, device)
    sc = _SCORER_CACHE.get(key)
    if sc is None:
        if _SANDBOX not in __import__("sys").path:
            __import__("sys").path.insert(0, _SANDBOX)
        from live_scorer import LiveScorer  # noqa: E402  (heavy: torch + visualizer)
        kw = {}
        if namo_config_path:
            kw["render_config"] = namo_config_path
        if device:
            kw["device"] = device
        sc = LiveScorer(ckpt=ckpt, **kw)
        _SCORER_CACHE[key] = sc
    return sc


class ScorerGoalStrategy(PrimitiveGoalStrategy):
    """PrimitiveGoalStrategy + champion scorer for goal.score (RO candidate ranking)."""

    def __init__(self, ckpt: Optional[str] = None, namo_config_path: Optional[str] = None,
                 xml_path: Optional[str] = None, device: Optional[str] = None,
                 data_dir: str = "data", primitive_prefix: str = "", verbose: bool = False,
                 max_push_steps: Optional[int] = None):
        super().__init__(data_dir=data_dir, verbose=verbose,
                         primitive_prefix=primitive_prefix, max_push_steps=max_push_steps)
        resolved_ckpt = ckpt or _default_checkpoint()
        self.ckpt = resolved_ckpt
        self.xml_path = xml_path
        # Reuse a process-cached scorer (loaded once per ckpt/config) — avoids per-scene model reload.
        self._scorer = _get_scorer(resolved_ckpt, namo_config_path, device)

    @property
    def strategy_name(self) -> str:
        return "scorer"

    def generate_goals(self,
                       object_id: str,
                       state: namo_rl.RLState,
                       env: namo_rl.RLEnvironment,
                       max_goals: int,
                       region_goals_sampled: Optional[List[Tuple[float, float, float]]] = None,
                       remaining_depth: Optional[int] = None) -> List[List[Goal]]:
        # 1) executable primitive goals (edge/depth/x/y/theta), score=0 — reuses shape selection etc.
        goals_per_edge = super().generate_goals(object_id, state, env, max_goals, region_goals_sampled)
        if not goals_per_edge:
            return goals_per_edge

        # 2) score this state with the champion scorer, conditioned on the NEIGHBOUR region.
        orig = env.get_full_state()
        try:
            env.set_full_state(state)
            P = self._scorer.score_state(
                env, object_id, robot_goal=None, xml_file=self.xml_path,
                region_samples=region_goals_sampled,
            )  # (60, num_depths) in [0,1]
        finally:
            env.set_full_state(orig)

        # 3) write scores onto the goals; RO's _search_bfs sorts by these (high P tried first).
        nd = P.shape[1]
        for edge_goals in goals_per_edge:
            for g in edge_goals:
                e, d = int(g.edge_idx), int(g.depth)
                g.score = float(P[e, d]) if (0 <= e < P.shape[0] and 0 <= d < nd) else 0.0
        return goals_per_edge
