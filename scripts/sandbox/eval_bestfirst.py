#!/usr/bin/env python3
"""VALUE-GUIDED GREEDY BEST-FIRST search — objective = MIN TOTAL SIMULATED PUSHES to open the path.

The design we converged on (not MCTS, not admissible-A*, not budget-layered):
  * One priority queue of UNSIMULATED candidate pushes. Pop the most-promising, SIMULATE it (the only
    expensive op, ~1s), check goal (free), stop on first open. NO pruning -> complete within Hmax.
  * Objective is min SIMS, not min depth -> NO layering, NO g-term. Values are compared ACROSS depths:
    a deep-but-near-solution node outranks a fresh shallow first push, because it needs fewer more sims.
  * Q (per-action prior, PRE-sim) guides EXPANSION (which pushes to add). V=mean_top5(Q(s,.)) (per-state,
    POST-sim leaf value) guides SELECTION (which branch to chase). priority = combine(Q(s,a), V(s)):
    "trust a high push-score only if the state it sits in is robustly promising" — a shrinkage that hedges
    the value head's miscalibration on OOD post-push states (maxP 24.6 vs mean_top5 34.5 in our data).
  * sim_budget = the single reactive<->search dial (tiny -> reactive single best path; larger -> search).

Baseline: --prior uniform = identical loop, RANDOM order, no value -> proves the guidance is worth it.
Metric: solve-rate within sim_budget + avg sims-to-solve. Sweep --sim-budget for the reactive->search curve.

  python scripts/sandbox/eval_bestfirst.py --ckpt <ckpt> --manifest <pure2push.txt> --hmax 2 \
      --sim-budget 30 --prior model --agg mean5 --combine blend --start 0 --end 985 --out <json>
"""
import sys, os, json, time, argparse, random, heapq
REPO = "/cache/home/dm1487/projects/namo/namo_cpp"
SAGE = "/cache/home/dm1487/projects/namo/sage_learning"
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from scorer_beam import BeamPlanner, make_env, make_action, read_manifest, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2  # noqa: E402  -> [(obj, Goal, q)] desc by Q(state,.,h)
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402

PURE2PUSH = "/scratch/dm1487/manifests/test_pure2_fromkey.txt"


def candidates(planner, env, goal, xml, state, h, prior, agg, rng, restrict_obj=None):
    """Reachable pushes from `state` (restricted to restrict_obj = the labeled object) with a priority-base
    value + the state value V. model: q = Q(state,a,h); V = agg of top Q (mean5 robust, or max). uniform: random q, V=0."""
    pool = rank_first_pushes_h2(planner, env, goal, xml, state, h, restrict_obj=restrict_obj)   # [(obj, g, q)]
    if not pool:
        return [], 0.0
    if prior == "uniform":
        out = [(o, g, rng.random()) for (o, g, _q) in pool]
        return out, 0.0                                              # no state value for the random baseline
    qs = sorted((q for (_o, _g, q) in pool), reverse=True)
    V = (sum(qs[:5]) / min(5, len(qs))) if agg == "mean5" else qs[0]
    return pool, V


def priority(q, V, combine):
    if combine == "q":       return q                 # Layer-1: raw action-value (perfect-model baseline)
    if combine == "product": return q * V             # both must be high
    return 0.5 * q + 0.5 * V                           # blend (default): action score tempered by state value


def solve_scene(planner, env, goal, xml, s0, hmax, sim_budget, prior, agg, combine, rng, restrict_obj=None):
    """Greedy best-first ON THE LABELED OBJECT (restrict_obj). Returns (solved, sims_used, plan_len|None)."""
    heap = []; ctr = 0; sims = 0
    pool, V0 = candidates(planner, env, goal, xml, s0, hmax, prior, agg, rng, restrict_obj=restrict_obj)
    for (obj, g, q) in pool:
        heapq.heappush(heap, (-priority(q, V0, combine), ctr,
                              {"obj": obj, "g": g, "from": s0, "ndone": 0, "plan": [(obj, g)]})); ctr += 1
    while heap and sims < sim_budget:
        _negpr, _c, it = heapq.heappop(heap)
        env.set_full_state(it["from"]); env.step(make_action(it["obj"], it["g"])); sims += 1
        if env.is_robot_goal_reachable():
            return True, sims, len(it["plan"])
        ndone = it["ndone"] + 1
        if ndone < hmax:                                  # room for another push -> expand the reached state
            s_new = env.get_full_state()
            h = hmax - ndone
            pool, V = candidates(planner, env, goal, xml, s_new, h, prior, agg, rng, restrict_obj=restrict_obj)
            for (obj2, g2, q2) in pool:
                heapq.heappush(heap, (-priority(q2, V, combine), ctr,
                                      {"obj": obj2, "g": g2, "from": s_new, "ndone": ndone,
                                       "plan": it["plan"] + [(obj2, g2)]})); ctr += 1
    return False, sims, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=PURE2PUSH)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=985)
    ap.add_argument("--hmax", type=int, default=2, help="max pushes (search depth bound; model saw H in {1,2})")
    ap.add_argument("--sim-budget", type=int, default=30, help="max sims/scene = the reactive<->search dial")
    ap.add_argument("--prior", default="model", choices=["model", "uniform"])
    ap.add_argument("--agg", default="mean5", choices=["mean5", "max"], help="state-value aggregate (selection)")
    ap.add_argument("--combine", default="blend", choices=["q", "blend", "product"])
    ap.add_argument("--key", default="/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push.json",
                    help="GROUND-TRUTH key (per (object,goal) records). The search is CONSTRAINED to the labeled "
                         "object → true k-push problem, one-to-one w/ GT. Eval is per-EPISODE (record), not per-scene.")
    ap.add_argument("--out", default="/scratch/dm1487/eval/bestfirst.json")
    ap.add_argument("--leaf-out", default="/scratch/dm1487/eval/bestfirst.jsonl")
    a = ap.parse_args()

    import os as _os
    key = json.load(open(a.key)); keyrp = {_os.path.realpath(k): v for k, v in key.items()}
    planner = BeamPlanner(ckpt=a.ckpt)
    print(f"device={planner.scorer.device} hmax={a.hmax} sim_budget={a.sim_budget} prior={a.prior} "
          f"agg={a.agg} combine={a.combine} key={_os.path.basename(a.key)}", flush=True)
    xmls = read_manifest(a.manifest, None)[a.start:a.end]
    n = n_solved = n_already = n_norec = sims_tot = sims_solved = 0; t0 = time.time()
    lf = open(a.leaf_out, "w")
    for xi, xml in enumerate(xmls):
        try:
            recs = key.get(xml) or keyrp.get(_os.path.realpath(xml))
            if not recs:
                n_norec += 1; continue
            env = make_env(xml)
            goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects()
            if env.is_robot_goal_reachable():
                n_already += 1; continue
            s0 = env.get_full_state()
            for ri, rec in enumerate(recs):                       # one EPISODE per (object,goal) record
                rng = random.Random(7000 + xi * 17 + ri)
                obj = rec.get("object_id")
                solved, sims, plen = solve_scene(planner, env, goal, xml, s0, a.hmax, a.sim_budget,
                                                 a.prior, a.agg, a.combine, rng, restrict_obj=obj)
                n += 1; sims_tot += sims; n_solved += int(solved); sims_solved += sims if solved else 0
                lf.write(json.dumps({"xml": xml, "object_id": obj, "region": rec.get("region"),
                                     "solved": solved, "sims": sims, "plan_len": plen}) + "\n")
            if xi % 20 == 0:
                print(f"  [{xi}/{len(xmls)}] episodes={n} solved={n_solved} avg_sims={sims_tot/max(n,1):.1f} "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
        except Exception as ex:
            print(f"  scene {xi} err: {ex}", file=sys.stderr); continue
    lf.close()
    res = {"ckpt": a.ckpt, "hmax": a.hmax, "sim_budget": a.sim_budget, "prior": a.prior, "agg": a.agg,
           "combine": a.combine, "key": _os.path.basename(a.key), "n_episodes": n,
           "n_already_open": n_already, "n_no_record": n_norec,
           "solve_rate": round(100.0 * n_solved / max(n, 1), 1),
           "avg_sims_all": round(sims_tot / max(n, 1), 2),
           "avg_sims_to_solve": round(sims_solved / max(n_solved, 1), 2)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
