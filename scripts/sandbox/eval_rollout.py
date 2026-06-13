#!/usr/bin/env python3
"""REACTIVE ROLLOUT eval — the faithful, deploy-matched horizon-Q number.

Unlike eval_m3 (which only grades the FIRST push against the exhaustive key), this actually RUNS the
model's policy and asks the only question that matters: did it OPEN THE PATH? The model drives; we
simulate only the pushes it COMMITS to (the deploy cost, ~budget sims/scene), and grade by
env.is_robot_goal_reachable() — NO precomputed key needed. This includes the real post-push state s1
(the OOD that v2 targets), so it can't over-credit like the first-push proxy.

Per scene, budget B (B=2 for pure2push; B=1 for 1-push; B=3 general — but the model only saw H∈{1,2}):
  GREEDY (@1): a1 = argmax Q(s0,.,B) -> sim -> s1; if open done; else a2 = argmax Q(s1,.,B-1) -> sim ...
  @k: try the top-k FIRST pushes (ranked at H=B); greedy for the remaining B-1 after each; solve@k =
      ANY of the top-k first-push lines opens the path. Sims bounded by k*B/scene = the deploy budget.
Budget-AWARE ranking (push at step t uses H = remaining budget) — this is what exercises the foresight;
ranking everything at H=1 would be the reactive-1push control (eval_m3 H=1).

  python scripts/sandbox/eval_rollout.py --ckpt <qfull.ckpt> --manifest <pure2push.txt> \
      --budget 2 --topk 10 --start 0 --end 985 --out /scratch/dm1487/eval/rollout_<name>.json
"""
import sys, os, json, time, argparse
REPO = "/cache/home/dm1487/projects/namo/namo_cpp"
SAGE = "/cache/home/dm1487/projects/namo/sage_learning"
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from scorer_beam import BeamPlanner, make_env, make_action, read_manifest, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402

PURE2PUSH = "/scratch/dm1487/manifests/test_pure2push_combined.txt"


def greedy_finish(planner, env, goal, xml, s, remaining):
    """Greedy rollout of `remaining` pushes from state s (budget-aware: each push ranks at H=remaining).
    Returns (solved, steps_used_within_this_call). Commits 1 sim per step."""
    used = 0
    for r in range(remaining, 0, -1):
        pool = rank_first_pushes_h2(planner, env, goal, xml, s, r)
        if not pool:
            return False, used
        obj, g, _v = pool[0]
        env.set_full_state(s); env.step(make_action(obj, g)); used += 1
        if env.is_robot_goal_reachable():
            return True, used
        s = env.get_full_state()
    return False, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=PURE2PUSH)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=985)
    ap.add_argument("--budget", type=int, default=2, help="push budget B (2 for pure2push; model saw H in {1,2})")
    ap.add_argument("--topk", type=int, default=10, help="max k for solve@k (try top-k first pushes, greedy after)")
    ap.add_argument("--out", default="/scratch/dm1487/eval/rollout.json")
    ap.add_argument("--leaf-out", default="/scratch/dm1487/eval/rollout.jsonl")
    a = ap.parse_args()

    planner = BeamPlanner(ckpt=a.ckpt)
    print(f"device={planner.scorer.device} ckpt={os.path.basename(a.ckpt)} budget={a.budget} reactive", flush=True)
    xmls = read_manifest(a.manifest, None)[a.start:a.end]
    KS = [1, 3, 5, 10]
    solve = {k: 0 for k in KS}; n = 0; n_already = 0; sims = 0; t0 = time.time()
    lf = open(a.leaf_out, "w")

    for xi, xml in enumerate(xmls):
        try:
            env = make_env(xml)
            goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects()
            if env.is_robot_goal_reachable():
                n_already += 1; continue
            s0 = env.get_full_state()
            pool0 = rank_first_pushes_h2(planner, env, goal, xml, s0, a.budget)   # rank first push at H=B
            if not pool0:
                continue
            n += 1
            solved_rank = None
            for rank in range(min(a.topk, len(pool0))):
                obj, g1, _v = pool0[rank]
                env.set_full_state(s0); env.step(make_action(obj, g1)); sims += 1
                if env.is_robot_goal_reachable():            # first push alone opened it
                    solved_rank = rank; break
                s1 = env.get_full_state()
                ok, used = greedy_finish(planner, env, goal, xml, s1, a.budget - 1)
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
    res = {"ckpt": a.ckpt, "mode": "reactive", "budget": a.budget, "n_solvable": n, "n_already_open": n_already,
           "sims_total": sims, "sims_per_scene": round(sims / max(n, 1), 1),
           "solve_at_k": {str(k): round(100.0 * solve[k] / n, 1) if n else 0.0 for k in KS}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
