"""One forward-only rollout = the data engine.

From s0, repeat [ score candidates -> sample a push -> act -> region open? ] up to
max_depth, early-stop on open. NO branching, NO mid-rollout resets: set_full_state is used
only to return to s0 between whole rollouts. This is depth-1-MCTS-for-data-collection.

Action sampling per step:
  - forced first push (pidx==1, if forced_action given): pin (edge,depth) to that action.
  - otherwise: eps-greedy over the reachable pool — with prob epsilon pick UNIFORMLY over
    reachable candidates; else sample from softmax(score / temperature). Uniform-arm scores
    are all 0, so softmax is uniform there too.

Reward / early-stop: goal_open_pts(env, gp, open_frac) — region OPEN iff >= open_frac of the
s0-sampled goal points are reachable (the canonical region criterion, matches collection +
eval_reactive_argmax exactly).

Each step stores the pre-action state (qpos/qvel) + the chosen (edge,depth) + the reachable
mask, enough to render the model ctx later (build_train_h5) without re-simulating the rollout.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import math
import random

from ._bootstrap import ensure_paths
ensure_paths()
import namo_rl                                                     # noqa: E402
from scorer_beam import make_action                               # noqa: E402
from eval_m3 import goal_open_pts                                  # noqa: E402


@dataclass
class StepRecord:
    qpos: List[float]              # pre-action full state (qvel below; set_full_state zeroes qvel anyway)
    qvel: List[float]
    edge: int                      # chosen edge (0-59)
    depth: int                     # chosen depth (0-based)
    x: float                       # executed push target (SE2) — lets revalidation re-execute the action
    y: float
    theta: float
    reachable_edges: List[int]     # reachable edges of the labeled object at this state (the mask)
    score: float                   # policy score of the chosen action (diagnostic)

    @property
    def action_id(self) -> int:
        return self.edge * 5 + self.depth


@dataclass
class RolloutRecord:
    xml_key: str
    object_id: str
    region: Optional[str]
    object_center: Tuple[float, float]
    difficulty: str
    horizon: str
    arm: str
    generation: int
    solved: bool
    T: Optional[int]               # solve length in pushes (None if not solved)
    n_pushes: int
    forced: bool
    forced_action: Optional[Tuple[int, int]]
    steps: List[StepRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["object_center"] = list(self.object_center)
        d["forced_action"] = list(self.forced_action) if self.forced_action else None
        return d


def _rlstate(qpos, qvel) -> "namo_rl.RLState":
    s = namo_rl.RLState()
    s.qpos = list(qpos)
    s.qvel = list(qvel)
    return s


def _sample_action(pool, cfg, rng: random.Random):
    """eps-greedy: eps -> uniform over reachable; else softmax(score/T)."""
    if rng.random() < cfg.epsilon:
        return rng.choice(pool)
    scores = [p[2] for p in pool]
    t = max(cfg.temperature, 1e-6)
    m = max(scores)
    w = [math.exp((s - m) / t) for s in scores]     # softmax over P/T (P in [0,1]); T->0 => argmax
    z = sum(w)
    r = rng.random() * z
    acc = 0.0
    for p, wi in zip(pool, w):
        acc += wi
        if r <= acc:
            return p
    return pool[-1]


def _pick_forced(pool, forced_action: Tuple[int, int]):
    """Return the pool entry matching (edge,depth); fall back to same-edge, then argmax."""
    e, d = forced_action
    for p in pool:
        if int(p[1].edge_idx) == e and int(p[1].depth) == d:
            return p
    for p in pool:
        if int(p[1].edge_idx) == e:
            return p
    return pool[0]


def run_rollout(env, ep, policy, cfg, s0, gp, robot_goal, rng: random.Random,
                forced_action: Optional[Tuple[int, int]] = None) -> RolloutRecord:
    """Run ONE rollout from s0. env must already be constructed with s0 restorable."""
    restrict = ep.object_id if cfg.restrict_to_labeled_object else None
    steps: List[StepRecord] = []
    opened_at = 0
    env.set_full_state(s0)
    for pidx in range(1, cfg.max_depth + 1):
        s_cur = env.get_full_state()
        pool = policy.score_pool(env, robot_goal, ep.xml, s_cur, restrict)
        if not pool:
            break
        reachable_edges = sorted({int(p[1].edge_idx) for p in pool})
        if forced_action is not None and pidx == 1:
            obj, g, score = _pick_forced(pool, forced_action)
        else:
            obj, g, score = _sample_action(pool, cfg, rng)
        steps.append(StepRecord(
            qpos=list(s_cur.qpos), qvel=list(s_cur.qvel),
            edge=int(g.edge_idx), depth=int(g.depth),
            x=float(g.x), y=float(g.y), theta=float(g.theta),
            reachable_edges=reachable_edges, score=float(score),
        ))
        env.set_full_state(s_cur)
        env.step(make_action(obj, g))
        if goal_open_pts(env, gp, cfg.open_frac):
            opened_at = pidx
            break
    solved = opened_at > 0
    return RolloutRecord(
        xml_key=ep.xml_key, object_id=ep.object_id, region=ep.region,
        object_center=ep.object_center, difficulty=ep.difficulty, horizon=ep.horizon,
        arm=cfg.arm, generation=cfg.generation, solved=solved,
        T=(opened_at if solved else None), n_pushes=len(steps),
        forced=(forced_action is not None), forced_action=forced_action, steps=steps,
    )
