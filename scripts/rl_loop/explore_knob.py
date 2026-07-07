#!/usr/bin/env python3
"""Pick the arm-B rollout temperature from the ACTUAL score distribution.

The reviewer flagged that softmax-over-P is near-uniform at T=1 (P in [0.5,0.73] -> <=1.26x
spread). This scores ~N pool s0 states with the arm-B ckpt, measures the real per-state P
spread, and reports the top/median softmax weight ratio across a temperature grid so we can
pick T giving a ~5-20x ratio (the exploration the collector actually needs). Arm A is uniform
and unaffected.

Reuses namo.rl_loop.policy.Policy.score_pool (byte-identical to the collector's scoring).
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop._bootstrap import ensure_paths      # noqa: E402
ensure_paths()
from namo.rl_loop.config import LoopConfig             # noqa: E402
from namo.rl_loop.episodes import load_pool            # noqa: E402
from namo.rl_loop.splits import load_split, episodes_in  # noqa: E402
from namo.rl_loop.policy import Policy                 # noqa: E402
from scorer_beam import make_env, FALLBACK_GOAL        # noqa: E402
from eval_m3 import sample_goal_points, goal_open_pts  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="arm-B policy ckpt (NoHz-v3)")
    ap.add_argument("--pool-key", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = LoopConfig(ckpt=a.ckpt, pool_key=a.pool_key, split_file=a.split_file)
    specs = load_pool(a.pool_key)
    split = load_split(a.split_file)
    train = episodes_in(specs, split, "train")
    rng = random.Random(a.seed)
    rng.shuffle(train)

    pol = Policy(ckpt=a.ckpt, score_h=cfg.score_h)
    per_state = []   # (pmax, pmed, pmin, n_cand)
    n_done = 0
    for ep in train:
        if n_done >= a.n_states:
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
        pool = pol.score_pool(env, goal, ep.xml, s0, restrict)
        if not pool or len(pool) < 2:
            continue
        scores = sorted((p[2] for p in pool), reverse=True)
        per_state.append((scores[0], median(scores), scores[-1], len(scores)))
        n_done += 1

    # score histogram (all candidate P across all states) — sample pmax/pmed spread
    spreads = [pm - pmd for (pm, pmd, _, _) in per_state]
    def ratio_top_med(T):
        # median over states of exp((pmax - pmed)/T)
        rs = [math.exp((pm - pmd) / max(T, 1e-6)) for (pm, pmd, _, _) in per_state]
        return median(rs) if rs else None
    def ratio_top_min(T):
        rs = [math.exp((pm - pmn) / max(T, 1e-6)) for (pm, _, pmn, _) in per_state]
        return median(rs) if rs else None

    Tgrid = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03]
    table = {f"{T:.2f}": {"median_ratio_top_over_median": round(ratio_top_med(T), 3) if ratio_top_med(T) else None,
                          "median_ratio_top_over_min": round(ratio_top_min(T), 3) if ratio_top_min(T) else None}
             for T in Tgrid}
    # pick T whose top/median ratio lands in [5,20], preferring ~10
    chosen = None
    best_d = 1e9
    for T in Tgrid:
        r = ratio_top_med(T)
        if r is None:
            continue
        if 5.0 <= r <= 20.0 and abs(r - 10.0) < best_d:
            best_d = abs(r - 10.0); chosen = T
    if chosen is None:   # fall back: closest to 10x from below
        cand = [(abs((ratio_top_med(T) or 0) - 10.0), T) for T in Tgrid if ratio_top_med(T)]
        chosen = min(cand)[1] if cand else 0.1

    def hist(vals, edges):
        h = [0] * (len(edges) - 1)
        for v in vals:
            for i in range(len(edges) - 1):
                if edges[i] <= v < edges[i + 1]:
                    h[i] += 1; break
        return h
    pmax_vals = [x[0] for x in per_state]; pmed_vals = [x[1] for x in per_state]
    out = {
        "ckpt": a.ckpt, "n_states": len(per_state),
        "pmax": {"mean": round(sum(pmax_vals) / max(1, len(pmax_vals)), 4),
                 "median": round(median(pmax_vals), 4) if pmax_vals else None,
                 "min": round(min(pmax_vals), 4) if pmax_vals else None,
                 "max": round(max(pmax_vals), 4) if pmax_vals else None},
        "pmedian_per_state": {"mean": round(sum(pmed_vals) / max(1, len(pmed_vals)), 4) if pmed_vals else None},
        "spread_pmax_minus_pmed": {"mean": round(sum(spreads) / max(1, len(spreads)), 4) if spreads else None,
                                   "median": round(median(spreads), 4) if spreads else None,
                                   "max": round(max(spreads), 4) if spreads else None},
        "spread_hist_edges": [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.01],
        "spread_hist": hist(spreads, [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.01]),
        "temperature_grid_median_weight_ratio": table,
        "chosen_temperature": chosen,
        "target": "top/median weight ratio in [5,20], prefer ~10x",
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n>>> CHOSEN TEMPERATURE = {chosen}  (top/median ratio "
          f"{ratio_top_med(chosen):.2f}x) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
