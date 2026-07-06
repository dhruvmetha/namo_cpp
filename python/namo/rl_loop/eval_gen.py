"""Per-generation dev eval: greedy open@1/2/5/10, stratified by difficulty x horizon.

Deploys the trained pi head GREEDILY (argmax at the live state, object-restricted, re-ranked
each push, early-stop on region open) on held-out DEV rooms — the same protocol as
eval_reactive_argmax (opened_at -> cumulative open@k), reusing its exact primitives so the
number is comparable to the registry's reactive 40.7. Reported never horizon-only: every cell
is (horizon, difficulty).
"""
from collections import defaultdict
from statistics import median
from typing import List

from ._bootstrap import ensure_paths
ensure_paths()
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL   # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback            # noqa: E402

from .config import LoopConfig
from .episodes import EpisodeSpec


def greedy_open_ks(pi_ckpt: str, specs: List[EpisodeSpec], cfg: LoopConfig,
                   limit: int = 0) -> dict:
    pl = BeamPlanner(ckpt=pi_ckpt)
    K = cfg.eval_max_pushes
    ks = list(cfg.eval_open_ks)
    bins = defaultdict(lambda: {"n": 0, "opened": [0] * (K + 1)})
    done = 0
    for ep in specs:
        if limit and done >= limit:
            break
        try:
            env = make_env(ep.xml)
            goal = extract_goal_with_fallback(ep.xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects()
            s0 = env.get_full_state()
            gp = sample_goal_points(env)
        except Exception:
            continue
        if not gp or goal_open_pts(env, gp, cfg.open_frac):
            continue
        restrict = ep.object_id if cfg.restrict_to_labeled_object else None
        pool = rank_first_pushes_h2(pl, env, goal, ep.xml, s0, cfg.score_h,
                                    restrict_obj=restrict, score=True)
        if not pool:
            continue
        done += 1
        b = bins[(ep.horizon, ep.difficulty)]; b["n"] += 1
        s_cur = s0; opened_at = 0
        for pidx in range(1, K + 1):
            if pidx > 1:
                pool = rank_first_pushes_h2(pl, env, goal, ep.xml, s_cur, 1,
                                            restrict_obj=restrict, score=True)
                if not pool:
                    break
            o, g, _q = pool[0]                      # argmax (greedy deploy)
            env.set_full_state(s_cur); env.step(make_action(o, g))
            if goal_open_pts(env, gp, cfg.open_frac):
                opened_at = pidx; break
            s_cur = env.get_full_state()
        if opened_at:
            b["opened"][opened_at] += 1
    report = {}
    for (hor, diff), b in bins.items():
        cum = {k: sum(b["opened"][1:k + 1]) for k in ks}
        report[f"{hor}/{diff}"] = {"n": b["n"],
                                   **{f"open@{k}": round(100 * cum[k] / max(1, b["n"]), 1) for k in ks}}
    return report


# --------------------------------------------------------------------------------------------------
# SETUP-RANKING (the USER headline curve): how deep in the policy's first-push ranking does a
# finishable/solving setup appear? Adapts the Phase-0 k-sweep (phase0_ksweep.py, committed 6368c45),
# reusing the same oracle-finish primitive so the number is comparable to the Phase-0 anchor
# (setup-hit@1 = 54.0, @8 = 82.5; hard 36.8 / 70.4).
# --------------------------------------------------------------------------------------------------
KRANK = 8          # sweep the policy's top-KRANK setups
FIN_CAP = 25       # cap oracle-finish tries per setup
SETUP_KS = (1, 2, 4, 8)


def _finishable(pl, env, goal, xml, s0, obj, g_setup, gp, frac) -> bool:
    """Teacher-forced (oracle) finish: exec the setup, does ANY reachable 2nd push open the region?"""
    env.set_full_state(s0); env.step(make_action(obj, g_setup))
    if goal_open_pts(env, gp, frac):
        return True
    s1 = env.get_full_state()
    fp = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj, score=True)
    for (_o, g2, _v) in fp[:FIN_CAP]:
        env.set_full_state(s1); env.step(make_action(obj, g2))
        if goal_open_pts(env, gp, frac):
            return True
    return False


def setup_ranking(pi_ckpt: str, specs: List[EpisodeSpec], cfg: LoopConfig, limit: int = 0) -> dict:
    """Per (horizon, difficulty): setup-hit@k (a finishable setup in the policy's top-k first pushes)
    + median rank of the first finishable setup. sim = oracle-finish (no GT); key = GT valid_setups
    (conservative lower bound), reported where the episode carries a GT key."""
    pl = BeamPlanner(ckpt=pi_ckpt)
    bins = defaultdict(lambda: {"n": 0, "sim_at": [], "key_at": [], "n_key": 0})
    done = 0
    for ep in specs:
        if limit and done >= limit:
            break
        try:
            env = make_env(ep.xml)
            goal = extract_goal_with_fallback(ep.xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects()
            s0 = env.get_full_state()
            gp = sample_goal_points(env)
        except Exception:
            continue
        if not gp or goal_open_pts(env, gp, cfg.open_frac):
            continue
        restrict = ep.object_id if cfg.restrict_to_labeled_object else None
        pool0 = rank_first_pushes_h2(pl, env, goal, ep.xml, s0, cfg.score_h,
                                     restrict_obj=restrict, score=True)
        if not pool0:
            continue
        done += 1
        b = bins[(ep.horizon, ep.difficulty)]; b["n"] += 1
        top = pool0[:KRANK]
        sim_at = 0; key_at = 0
        for r, (_o, g, _sc) in enumerate(top, 1):
            ed = (int(g.edge_idx), int(g.depth))
            if key_at == 0 and ed in ep.valid_setups:
                key_at = r
            if sim_at == 0 and _finishable(pl, env, goal, ep.xml, s0, ep.object_id, g, gp, cfg.open_frac):
                sim_at = r
            if sim_at and (key_at or not ep.valid_setups):
                break
        b["sim_at"].append(sim_at)
        if ep.valid_setups:
            b["n_key"] += 1
            b["key_at"].append(key_at)
    report = {}
    for (hor, diff), b in bins.items():
        sim_found = [a for a in b["sim_at"] if a > 0]
        key_found = [a for a in b["key_at"] if a > 0]
        cell = {"n": b["n"]}
        for k in SETUP_KS:
            cell[f"setup_hit@{k}"] = round(100 * sum(1 for a in b["sim_at"] if 0 < a <= k) / max(1, b["n"]), 1)
        cell["median_setup_rank"] = (median(sim_found) if sim_found else None)
        if b["n_key"]:
            cell["key_hit@1"] = round(100 * sum(1 for a in b["key_at"] if a == 1) / b["n_key"], 1)
            cell["key_hit@8"] = round(100 * sum(1 for a in b["key_at"] if 0 < a <= 8) / b["n_key"], 1)
            cell["median_key_rank"] = (median(key_found) if key_found else None)
        report[f"{hor}/{diff}"] = cell
    return report
