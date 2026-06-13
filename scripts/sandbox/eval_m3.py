#!/usr/bin/env python3
"""M3 — ZERO-SIM H=2 foresight eval (the horizon-Q headline).

On pure-2-push scenes (no single push opens the region), rank the FIRST pushes by Q(s0, ., H=2) — the
budget-Q model's value at remaining budget 2, ONE forward pass, ZERO simulations for the decision — then
VERIFY the top-k by simulation (grading only, offline) to check whether the picked first push is a real
SETUP (leads to a state a second push opens). This is apples-to-apples with fpv_m2b's first-push number
(75.2@1) which used ~49 sims to rank; here the ranking costs 0 sims.

Bars: 34.5 @1 (registered: old champion + 49-sim beam) | 75.2 @1 (fpv_m2b, M2b + 49 sims).

  PYTHONPATH=...build_python:...python:...scripts:...scripts/sandbox:...sage_learning \
  python scripts/sandbox/eval_m3.py --ckpt <qfull.ckpt> --start 0 --end 985 --h 2 --topk 10 \
      --out /scratch/dm1487/eval/m3_<name>.json --leaf-out /scratch/dm1487/eval/m3_<name>.jsonl
"""
import sys, os, json, time, argparse
REPO = "/cache/home/dm1487/projects/namo/namo_cpp"
SAGE = "/cache/home/dm1487/projects/namo/sage_learning"
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402
from scorer_beam import BeamPlanner, make_env, make_action, read_manifest, FALLBACK_GOAL  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402

PURE2PUSH = "/scratch/dm1487/manifests/test_pure2push_combined.txt"


def rank_first_pushes_h2(planner, env, robot_goal, xml, s0, h):
    """Rank ALL reachable (obj, edge, depth) first pushes by Q(s0, ., h). ZERO sims.
    Returns [(obj, Goal, value)] sorted desc. Mirrors BeamPlanner._candidates' reachability pooling
    but scores at budget h (the foresight query)."""
    env.set_full_state(s0)
    reach_objs = list(env.get_reachable_objects())          # warms wavefront
    redges = {o: set(env.get_reachable_edges(o)) for o in reach_objs}
    pool = []
    for obj in reach_objs:
        if not redges[obj]:
            continue
        P = planner.scorer.score_state(env, obj, robot_goal, xml, h=h)   # (60,5) at budget h
        env.set_full_state(s0)                                            # score_state may move state
        goals_per_edge = planner.prim.generate_goals(obj, s0, env, max_goals=0)
        for edge_goals in goals_per_edge:
            for g in edge_goals:
                if g is None:
                    continue
                e = int(getattr(g, "edge_idx", -1)); d = int(getattr(g, "depth", -1))
                if e in redges[obj] and 0 <= d < P.shape[1]:
                    pool.append((obj, g, float(P[e, d])))
    pool.sort(key=lambda x: -x[2])
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=PURE2PUSH)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=25)
    ap.add_argument("--h", type=int, default=2, help="budget to query (2 = foresight; 1 = reactive-1push control)")
    ap.add_argument("--topk", type=int, default=10, help="grade hit@k for k up to this; verify-by-sim budget per first push")
    ap.add_argument("--second-k", type=int, default=15, help="2nd-push verify budget when checking a first push is a setup")
    ap.add_argument("--grade", default="sim", choices=["sim", "key"],
                    help="sim = verify each top-k first push by simulation (comparable to the 75.2/34.5 sim bars). "
                         "key = grade vs the exhaustive pure2push.json setups (ZERO sims, fast feeler; ground-truth).")
    ap.add_argument("--key", default="/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push.json")
    ap.add_argument("--out", default="/scratch/dm1487/eval/m3.json")
    ap.add_argument("--leaf-out", default="/scratch/dm1487/eval/m3.jsonl")
    a = ap.parse_args()

    planner = BeamPlanner(ckpt=a.ckpt)
    print(f"device={planner.scorer.device}  ckpt={os.path.basename(a.ckpt)}  H={a.h}  grade={a.grade}", flush=True)
    xmls = read_manifest(a.manifest, None)[a.start:a.end]
    # key grading: posmap[realpath(xml)][object_id] = set of (e,d) that are setups (valid_first_push ∪ valid_1push,
    # i.e. opens-within-2). ZERO sims — the exhaustive truth from the 2-push validset.
    posmap = {}
    if a.grade == "key":
        kf = json.load(open(a.key))
        for kx, recs in kf.items():
            rp = os.path.realpath(kx)
            d = {}
            for r in recs:
                s = {tuple(t) for t in r.get("valid_first_push", [])} | {tuple(t) for t in r.get("valid_1push", [])}
                d[r["object_id"]] = s
            posmap[rp] = d; posmap[kx] = d
    KS = [1, 3, 5, 10]
    hit = {k: 0 for k in KS}; n = 0; n_already = n_onepush = 0; t0 = time.time()
    lf = open(a.leaf_out, "w")
    setup_cache = {}  # (xml, obj, e, d) -> is it a real setup (sim-verified)? memoize within a scene

    for xi, xml in enumerate(xmls):
        try:
            env = make_env(xml)
            goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects()
            if env.is_robot_goal_reachable():
                n_already += 1; continue
            s0 = env.get_full_state()
            # ZERO-SIM RANKING: order first pushes by Q(s0, ., h)
            pool = rank_first_pushes_h2(planner, env, goal, xml, s0, a.h)
            if not pool:
                continue
            # filter out 1-push openers up front (pure-2push: there shouldn't be any, but guard) and
            # grade hit@k by VERIFYING the top-k first pushes are setups (sim only the graded top-k)
            def is_setup(obj, g1):
                key = (xml, obj, int(g1.edge_idx), int(g1.depth))
                if key in setup_cache:
                    return setup_cache[key]
                env.set_full_state(s0); env.step(make_action(obj, g1))
                if env.is_robot_goal_reachable():            # opens in 1 -> not a pure setup; treat as success-for-2 too
                    setup_cache[key] = True; return True
                s1 = env.get_full_state()
                pool2 = planner._candidates(env, goal, xml, s1)
                ok = False
                for (o2, g2, _p) in pool2[:a.second_k]:
                    env.set_full_state(s1); env.step(make_action(o2, g2))
                    if env.is_robot_goal_reachable():
                        ok = True; break
                setup_cache[key] = ok; return ok

            def is_setup_key(obj, g1):
                pm = posmap.get(xml) or posmap.get(os.path.realpath(xml)) or {}
                return (int(g1.edge_idx), int(g1.depth)) in pm.get(obj, set())

            graded = is_setup_key if a.grade == "key" else is_setup
            n += 1
            found_rank = None
            for rank, (obj, g1, val) in enumerate(pool[:max(KS)]):
                if graded(obj, g1):
                    found_rank = rank; break
            for k in KS:
                if found_rank is not None and found_rank < k:
                    hit[k] += 1
            lf.write(json.dumps({"xml": xml, "n_cand": len(pool), "found_rank": found_rank}) + "\n")
            if xi % 20 == 0:
                el = time.time() - t0
                print(f"  [{xi}/{len(xmls)}] graded={n} hit@1={hit[1]} ({el:.0f}s)", file=sys.stderr, flush=True)
        except Exception as ex:
            print(f"  scene {xi} err: {ex}", file=sys.stderr)
            continue
    lf.close()
    res = {"ckpt": a.ckpt, "H": a.h, "grade": a.grade, "n_graded": n, "n_already_open": n_already,
           "hit_at_k": {str(k): (100.0 * hit[k] / n if n else 0.0) for k in KS},
           "bars": ({"old_champ_49sim": 34.5, "fpv_m2b_49sim": 75.2} if a.grade == "sim"
                    else {"note": "key-graded vs exhaustive setups (NOT comparable to the 75.2/34.5 sim bars); "
                                  "compare H=2 vs H=1 grade=key on the SAME scenes"})}
    json.dump(res, open(a.out, "w"))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
