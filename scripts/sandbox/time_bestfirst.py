#!/usr/bin/env python3
"""WALL-CLOCK best-first SEARCH timing — Hz / NoHz / random, SAME node, interleaved, warm-only.

The search-regime counterpart of time_benchmark.py (reactive@2). Reuses the EXACT search (candidates/priority
from eval_bestfirst, combine='q' which the user confirmed). Now that render ~0.1s (< a sim), this shows the
model's sim-savings converting to WALL-TIME: the model solves in fewer sims => less wall-clock than random's
deeper search. Per (episode,model): t_score (renders+NN), t_sim (env.step), n_sim (=sims used), t_wall, solved."""
import sys, os, json, time, argparse, random, heapq
from collections import defaultdict
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL          # noqa: E402
from eval_bestfirst import candidates, priority                                    # noqa: E402  (same search)
from eval_m3 import sample_goal_points, goal_open_pts                              # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback                   # noqa: E402
from namo.paths import resolve                                                     # noqa: E402

S = "/scratch/dm1487/sage_outputs/scorer"
HZ = f"{S}/qfull_v3_v4hq_s1/namo-classifier/qkfk0slk/checkpoints/epoch011-val_loss0.6571.ckpt"
NOHZ = f"{S}/qfull_nohz_v3_v4hq_s1/namo-classifier/wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt"
KEY = "/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push.json"
PC = time.perf_counter


def tier(sr):
    return "hard" if sr < 0.05 else ("med" if sr < 0.30 else "easy")


def stratified(n_per):
    k = json.load(open(KEY)); b = defaultdict(list)
    for xml, recs in k.items():
        for r in recs:
            b[tier(r.get("solve_rate_first_push", 0.0))].append((xml, r["object_id"]))
    out = []
    for t in ("easy", "med", "hard"):
        out += [(x, o, t) for (x, o) in b[t][:n_per]]
    return out


def timed_bf(pl, env, goal, xmlp, obj, s0, gp, prior, agg, combine, budget, rng):
    """Greedy best-first on the labeled object (hmax=2), timed. Mirrors eval_bestfirst.solve_scene."""
    isopen = lambda e: goal_open_pts(e, gp)
    heap = []; ctr = 0; sims = 0; tsc = 0.0; tsim = 0.0; nsc = 0; t0 = PC()
    env.set_full_state(s0)
    t = PC(); pool, V0 = candidates(pl, env, goal, xmlp, s0, 2, prior, agg, rng, restrict_obj=obj); tsc += PC() - t; nsc += 1
    for (o, g, q) in pool:
        heapq.heappush(heap, (-priority(q, V0, combine), ctr, {"obj": o, "g": g, "from": s0, "ndone": 0})); ctr += 1
    solved = False
    while heap and sims < budget:
        _n, _c, it = heapq.heappop(heap)
        env.set_full_state(it["from"]); t = PC(); env.step(make_action(it["obj"], it["g"])); tsim += PC() - t; sims += 1
        if isopen(env):
            solved = True; break
        ndone = it["ndone"] + 1
        if ndone < 2:
            s_new = env.get_full_state()
            t = PC(); pool, V = candidates(pl, env, goal, xmlp, s_new, 2 - ndone, prior, agg, rng, restrict_obj=obj); tsc += PC() - t; nsc += 1
            for (o2, g2, q2) in pool:
                heapq.heappush(heap, (-priority(q2, V, combine), ctr, {"obj": o2, "g": g2, "from": s_new, "ndone": ndone})); ctr += 1
    return {"t_score": tsc, "t_sim": tsim, "n_score": nsc, "n_sim": sims, "t_wall": PC() - t0, "solved": solved}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tier", type=int, default=50)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = random.Random(7)
    print("loading scorers...", flush=True)
    pl_hz = BeamPlanner(ckpt=HZ); pl_nz = BeamPlanner(ckpt=NOHZ)
    models = [("Hz", pl_hz, "model"), ("NoHz", pl_nz, "model"), ("random", pl_nz, "uniform")]
    samp = stratified(a.n_per_tier)
    print(f"  {len(samp)} episodes; budget={a.budget}; warming up {a.warmup}/model (untimed)...", flush=True)
    x0, o0, _ = samp[0]; xp0 = str(resolve(x0)); e0 = make_env(xp0); g0 = extract_goal_with_fallback(xp0, FALLBACK_GOAL)
    e0.set_robot_goal(*g0); e0.get_reachable_objects(); s00 = e0.get_full_state()
    for _ in range(a.warmup):
        for pl in (pl_hz, pl_nz):
            candidates(pl, e0, g0, xp0, s00, 2, "model", "mean5", rng, restrict_obj=o0)
    fh = open(a.out, "w"); t_start = PC()
    for i, (xml, obj, t) in enumerate(samp):
        try:
            xmlp = str(resolve(xml)); env = make_env(xmlp); goal = extract_goal_with_fallback(xmlp, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state(); gp = sample_goal_points(env)
        except Exception:
            continue
        if not gp or goal_open_pts(env, gp):
            continue
        for name, pl, prior in models:                       # interleaved: all 3 on the SAME s0
            r = timed_bf(pl, env, goal, xmlp, obj, s0, gp, prior, "mean5", "q", a.budget, rng)
            r.update({"model": name, "tier": t, "xml": os.path.basename(xml), "object_id": obj})
            fh.write(json.dumps(r) + "\n")
        if i % 10 == 0:
            fh.flush(); print(f"  [{i}/{len(samp)}] {PC() - t_start:.0f}s", file=sys.stderr, flush=True)
    fh.close()
    print(f"DONE {a.out}  ({PC() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
