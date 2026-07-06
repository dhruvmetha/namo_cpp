"""Rollout collector — the data engine driver.

For each episode: build env, sample the goal region once at s0, run `rollouts_per_episode`
forward rollouts. Hard episodes lacking setup diversity (fewer than
`forced_min_distinct_first` distinct successful first actions known so far) ALTERNATE
ordinary/forced rollouts; the forced action is the least-tried (this generation) of the
policy's top-`forced_top_initial` initial candidates that is absent from the success buffer
(<= `forced_max_attempts_per_action` attempts/action/gen).

Parallelism mirrors modular_parallel_collection: a multiprocessing.Pool over episode
shards, one Policy (model load) per worker, per-worker output pickle. n_workers=1 runs
serially (smoke/debug).
"""
from dataclasses import asdict
from typing import Dict, List, Optional, Set, Tuple
import os
import pickle
import random

from ._bootstrap import ensure_paths
ensure_paths()
from scorer_beam import make_env, FALLBACK_GOAL                    # noqa: E402
from eval_m3 import sample_goal_points, goal_open_pts              # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402

from .config import LoopConfig
from .episodes import EpisodeSpec
from .policy import Policy
from .rollout import run_rollout, RolloutRecord


def _choose_forced(top_initial, distinct_first: Set[int], attempts: Dict[int, int],
                   cfg: LoopConfig) -> Optional[Tuple[int, int, int]]:
    cands = [t for t in top_initial
             if t[0] not in distinct_first
             and attempts.get(t[0], 0) < cfg.forced_max_attempts_per_action]
    if not cands:
        return None
    cands.sort(key=lambda t: attempts.get(t[0], 0))
    return cands[0]


def collect_episode(ep: EpisodeSpec, cfg: LoopConfig, policy: Policy,
                    buf_first_actions: Set[int], rng: random.Random) -> List[RolloutRecord]:
    records: List[RolloutRecord] = []
    env = make_env(ep.xml)
    goal = extract_goal_with_fallback(ep.xml, FALLBACK_GOAL)
    env.set_robot_goal(*goal)
    env.get_reachable_objects()                 # warm wavefront
    s0 = env.get_full_state()
    gp = sample_goal_points(env)
    if not gp or goal_open_pts(env, gp, cfg.open_frac):
        return records                           # no goal region, or already open -> not an episode

    distinct_first: Set[int] = set(buf_first_actions)
    use_forced = (cfg.forced_enable and ep.difficulty == "hard"
                  and len(distinct_first) < cfg.forced_min_distinct_first)

    top_initial: List[Tuple[int, int, int]] = []   # (action_id, edge, depth), score-desc, distinct
    if use_forced:
        restrict = ep.object_id if cfg.restrict_to_labeled_object else None
        pool0 = policy.score_pool(env, goal, ep.xml, s0, restrict)
        seen: Set[int] = set()
        for (_o, g, _s) in pool0:
            aid = int(g.edge_idx) * 5 + int(g.depth)
            if aid in seen:
                continue
            seen.add(aid)
            top_initial.append((aid, int(g.edge_idx), int(g.depth)))
            if len(top_initial) >= cfg.forced_top_initial:
                break

    attempts: Dict[int, int] = {}
    for i in range(cfg.rollouts_per_episode):
        forced_action = None
        if use_forced and (i % 2 == 1):          # alternate ordinary / forced
            fa = _choose_forced(top_initial, distinct_first, attempts, cfg)
            if fa is not None:
                forced_action = (fa[1], fa[2])
                attempts[fa[0]] = attempts.get(fa[0], 0) + 1
        rec = run_rollout(env, ep, policy, cfg, s0, gp, goal, rng, forced_action)
        records.append(rec)
        if rec.solved and rec.steps:
            distinct_first.add(rec.steps[0].action_id)
    return records


def _worker(args) -> str:
    shard_specs, cfg_dict, buf_first, out_path, seed = args
    cfg = LoopConfig(**cfg_dict)
    policy = Policy(ckpt=cfg.ckpt, score_h=cfg.score_h)
    rng = random.Random(seed)
    out: List[dict] = []
    for ep in shard_specs:
        fa = set(buf_first.get(ep.key, set()))
        try:
            recs = collect_episode(ep, cfg, policy, fa, rng)
        except Exception as e:                    # one bad scene must not kill the shard
            print(f"  [collector] skip {ep.xml_key} {ep.object_id}: {e}", flush=True)
            continue
        out.extend(r.to_dict() for r in recs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    return out_path


def run_collection(specs: List[EpisodeSpec], cfg: LoopConfig,
                   buf_first_actions: Dict, out_dir: str,
                   n_workers: int = 1, seed: int = 7000) -> List[str]:
    """Shard episodes across workers, collect, write per-worker pkls. Returns pkl paths."""
    os.makedirs(out_dir, exist_ok=True)
    n_workers = max(1, n_workers)
    shards: List[List[EpisodeSpec]] = [specs[i::n_workers] for i in range(n_workers)]
    cfg_dict = asdict(cfg)
    tasks = [(shards[i], cfg_dict, buf_first_actions,
              os.path.join(out_dir, f"rollouts_shard{i}.pkl"), seed + i)
             for i in range(n_workers) if shards[i]]
    if n_workers == 1:
        return [_worker(tasks[0])] if tasks else []
    import multiprocessing as mp
    with mp.Pool(processes=n_workers) as pool:
        return list(pool.imap_unordered(_worker, tasks))
