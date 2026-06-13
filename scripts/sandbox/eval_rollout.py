#!/usr/bin/env python3
"""ROLLOUT eval — the faithful, deploy-matched horizon-Q solve-rate, REACTIVE *and* SEARCH.

The model drives; we simulate only what the policy explores and grade by env.is_robot_goal_reachable()
(NO precomputed key). This includes the real post-push state s1 (the OOD v2 targets), so it can't
over-credit like the first-push proxy (eval_m3). One script spans the whole regime via flags:

  --prior  q | uniform   rank candidates by Q (the model) or RANDOM (brute-force baseline the Q must beat)
  --flat-h1              rank EVERY push at H=1 (foresight OFF) — the reactive control; default = budget-aware
                         (push at step t ranks at H=remaining; first push at H=B exercises the foresight)
  --w2 W                 2nd-push beam width: 1 = greedy reactive; >1 = SEARCH (sim top-W seconds per first)
  --topk K               1st-push width for solve@k (try top-k first pushes)

Per scene, budget B (B=2 pure2push; model saw H in {1,2}). Sims spent = the deploy/search cost; reported.
  solve@k = ANY of the top-k first pushes leads to an opened path (greedy or w2-beam on the 2nd push).

Matrix to characterize a ckpt on pure2push (B=2):
  reactive Q      : --prior q                 (foresight ON,  0-search anchor)
  reactive H1     : --prior q --flat-h1       (foresight OFF control)
  reactive random : --prior uniform           (random-policy floor)
  search Q        : --prior q --w2 10         (Q-guided search curve)
  search random   : --prior uniform --w2 10   (brute-force search baseline)

  python scripts/sandbox/eval_rollout.py --ckpt <ckpt> --manifest <pure2push.txt> --budget 2 \
      --prior q --w2 1 --start 0 --end 985 --out <json> --leaf-out <jsonl>
"""
import sys, os, json, time, argparse, random
REPO = "/cache/home/dm1487/projects/namo/namo_cpp"
SAGE = "/cache/home/dm1487/projects/namo/sage_learning"
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from scorer_beam import BeamPlanner, make_env, make_action, read_manifest, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402

PURE2PUSH = "/scratch/dm1487/manifests/test_pure2_fromkey.txt"


def ranked(planner, env, goal, xml, s, h, prior, rng):
    """Candidate first pushes from state s, ordered by prior. ZERO sims."""
    pool = rank_first_pushes_h2(planner, env, goal, xml, s, h)   # sorted desc by Q(s,.,h)
    if prior == "uniform":
        rng.shuffle(pool)                                        # same candidate SET, random order
    return pool


def finish(planner, env, goal, xml, s, remaining, w2, prior, rng, flat_h1):
    """Roll out the remaining `remaining` pushes from s. Beam width w2 on the LAST push, greedy before.
    Returns (solved, sims_used). Commits 1 sim per simulated push."""
    used = 0
    while remaining > 0:
        h = 1 if flat_h1 else remaining
        pool = ranked(planner, env, goal, xml, s, h, prior, rng)
        if not pool:
            return False, used
        width = w2 if remaining == 1 else 1                      # search only at the final push
        for (obj, g, _v) in pool[:width]:
            env.set_full_state(s); env.step(make_action(obj, g)); used += 1
            if env.is_robot_goal_reachable():
                return True, used
        if remaining == 1:
            return False, used
        s = env.get_full_state(); remaining -= 1                 # greedy-advance with pool[0] (already simmed)
    return False, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=PURE2PUSH)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=985)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10, help="1st-push width for solve@k")
    ap.add_argument("--w2", type=int, default=1, help="2nd-push beam width (1=greedy reactive; >1=search)")
    ap.add_argument("--prior", default="q", choices=["q", "uniform"])
    ap.add_argument("--flat-h1", action="store_true", help="rank every push at H=1 (foresight-off control)")
    ap.add_argument("--out", default="/scratch/dm1487/eval/rollout.json")
    ap.add_argument("--leaf-out", default="/scratch/dm1487/eval/rollout.jsonl")
    a = ap.parse_args()

    planner = BeamPlanner(ckpt=a.ckpt)
    print(f"device={planner.scorer.device} ckpt={os.path.basename(a.ckpt)} budget={a.budget} "
          f"prior={a.prior} w2={a.w2} flat_h1={a.flat_h1}", flush=True)
    xmls = read_manifest(a.manifest, None)[a.start:a.end]
    KS = [1, 3, 5, 10]
    solve = {k: 0 for k in KS}; n = 0; n_already = 0; sims = 0; t0 = time.time()
    lf = open(a.leaf_out, "w")

    for xi, xml in enumerate(xmls):
        try:
            rng = random.Random(1000 + xi)                       # deterministic per scene (uniform prior)
            env = make_env(xml)
            goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects()
            if env.is_robot_goal_reachable():
                n_already += 1; continue
            s0 = env.get_full_state()
            h0 = 1 if a.flat_h1 else a.budget
            pool0 = ranked(planner, env, goal, xml, s0, h0, a.prior, rng)
            if not pool0:
                continue
            n += 1
            solved_rank = None
            for rank in range(min(a.topk, len(pool0))):
                obj, g1, _v = pool0[rank]
                env.set_full_state(s0); env.step(make_action(obj, g1)); sims += 1
                if env.is_robot_goal_reachable():
                    solved_rank = rank; break
                s1 = env.get_full_state()
                ok, used = finish(planner, env, goal, xml, s1, a.budget - 1, a.w2, a.prior, rng, a.flat_h1)
                sims += used
                if ok:
                    solved_rank = rank; break
            for k in KS:
                if solved_rank is not None and solved_rank < k:
                    solve[k] += 1
            lf.write(json.dumps({"xml": xml, "n_cand": len(pool0), "solved_rank": solved_rank}) + "\n")
            if xi % 20 == 0:
                print(f"  [{xi}/{len(xmls)}] n={n} solve@1={solve[1]} sims={sims} ({time.time()-t0:.0f}s)",
                      file=sys.stderr, flush=True)
        except Exception as ex:
            print(f"  scene {xi} err: {ex}", file=sys.stderr); continue
    lf.close()
    res = {"ckpt": a.ckpt, "mode": ("search" if a.w2 > 1 else "reactive"), "prior": a.prior,
           "flat_h1": bool(a.flat_h1), "budget": a.budget, "w2": a.w2,
           "n_solvable": n, "n_already_open": n_already, "sims_total": sims,
           "sims_per_scene": round(sims / max(n, 1), 1),
           "solve_at_k": {str(k): round(100.0 * solve[k] / n, 1) if n else 0.0 for k in KS}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
