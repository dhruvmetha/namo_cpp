#!/usr/bin/env python3
"""RETIRED 2026-08-06 for the canonical wall-clock protocol — DO NOT use for paper numbers.

timed_bf() below is a private COPY of the search that predates the two adopted pruning rules
(dedupe_noop, prune_jam_depth, both default-on in eval_bestfirst.solve_scene), so it times a
different — slower — search than the one behind every number in RESULTS.md. eval_bestfirst.py
now carries the timing itself (t_wall/t_sim/t_score/n_score per episode in --leaf-out), so a
timed run IS a canonical run. Kept only for the historical Hz/NoHz reactive comparison.

WALL-CLOCK best-first SEARCH timing — Hz / NoHz / random, SAME node, interleaved, warm-only.

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
from namo import eval_sets                                                          # noqa: E402

S = "/scratch/dm1487/sage_outputs/scorer"
HZ = f"{S}/qfull_v3_v4hq_s1/namo-classifier/qkfk0slk/checkpoints/epoch011-val_loss0.6571.ckpt"
NOHZ = f"{S}/qfull_nohz_v3_v4hq_s1/namo-classifier/wl8k6iyv/checkpoints/epoch012-val_loss0.6896.ckpt"
KEY = str(eval_sets.PURE2PUSH)
PC = time.perf_counter


def tier(sr):
    return "hard" if sr < 0.05 else ("med" if sr < 0.30 else "easy")


def stratified(n_per, key_path):
    k = json.load(open(key_path)); b = defaultdict(list)
    for xml, recs in k.items():
        for r in recs:
            sr = r.get("solve_rate_first_push", r.get("solve_rate", 0.0))   # 2-push vs 1-push key field
            b[tier(sr)].append((xml, r["object_id"]))
    out = []
    for t in ("easy", "med", "hard"):
        out += [(x, o, t) for (x, o) in b[t][:n_per]]
    return out


def full_episodes(key_path):
    """ALL episodes in a stable order (for sharded full-set runs), each tagged with its difficulty tier."""
    k = json.load(open(key_path)); out = []
    for xml in sorted(k):
        for r in k[xml]:
            sr = r.get("solve_rate_first_push", r.get("solve_rate", 0.0))
            out.append((xml, r["object_id"], tier(sr)))
    return out


def timed_bf(pl, env, goal, xmlp, obj, s0, gp, prior, agg, combine, budget, rng, hmax):
    """Greedy best-first on the labeled object, timed. Mirrors eval_bestfirst.solve_scene (hmax = push depth)."""
    isopen = lambda e: goal_open_pts(e, gp)
    heap = []; ctr = 0; sims = 0; tsc = 0.0; tsim = 0.0; nsc = 0; t0 = PC()
    depth_hist = defaultdict(int); solve_ranks = None            # a/b: sims by tree-depth; c: rank-path of the winning plan
    env.set_full_state(s0)
    t = PC(); pool, V0 = candidates(pl, env, goal, xmlp, s0, hmax, prior, agg, rng, restrict_obj=obj); tsc += PC() - t; nsc += 1
    for rank, (o, g, q) in enumerate(sorted(pool, key=lambda x: -priority(x[2], V0, combine))):   # rank = priority order in the sibling pool
        heapq.heappush(heap, (-priority(q, V0, combine), ctr, {"obj": o, "g": g, "from": s0, "ndone": 0, "ranks": [rank]})); ctr += 1
    solved = False
    while heap and sims < budget:
        _n, _c, it = heapq.heappop(heap)
        depth_hist[it["ndone"]] += 1                             # this sim expands a node at push-depth ndone (0=first push, 1=second)
        env.set_full_state(it["from"]); t = PC(); env.step(make_action(it["obj"], it["g"])); tsim += PC() - t; sims += 1
        if isopen(env):
            solved = True; solve_ranks = it["ranks"]; break       # ranks of each push in the winning plan
        ndone = it["ndone"] + 1
        if ndone < hmax:
            s_new = env.get_full_state()
            t = PC(); pool, V = candidates(pl, env, goal, xmlp, s_new, hmax - ndone, prior, agg, rng, restrict_obj=obj); tsc += PC() - t; nsc += 1
            for rank, (o2, g2, q2) in enumerate(sorted(pool, key=lambda x: -priority(x[2], V, combine))):
                heapq.heappush(heap, (-priority(q2, V, combine), ctr, {"obj": o2, "g": g2, "from": s_new, "ndone": ndone, "ranks": it["ranks"] + [rank]})); ctr += 1
    return {"t_score": tsc, "t_sim": tsim, "n_score": nsc, "n_sim": sims, "t_wall": PC() - t0, "solved": solved,
            "depth_hist": dict(depth_hist), "solve_ranks": solve_ranks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tier", type=int, default=50)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--hmax", type=int, default=2)
    ap.add_argument("--key", default=KEY)
    ap.add_argument("--start", type=int, default=-1)   # shard range [start,end) over the base list
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--strat", action="store_true")    # shard the STRATIFIED list (n-per-tier) instead of full set
    ap.add_argument("--hz-ckpt", default=HZ)            # seed-variant checkpoints
    ap.add_argument("--nohz-ckpt", default=NOHZ)
    ap.add_argument("--rng-seed", type=int, default=7)  # random baseline seed (random uses the uniform pick)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--models", default="Hz,NoHz,random", help="comma subset of Hz,NoHz,random (run random-only for the 10-seed floor without recomputing the deterministic model)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = a.models.split(",")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = random.Random(a.rng_seed)
    print(f"loading scorers (hz={a.hz_ckpt.split('/scorer/')[-1].split('/')[0]} nohz={a.nohz_ckpt.split('/scorer/')[-1].split('/')[0]} rng={a.rng_seed})...", flush=True)
    need_hz = "Hz" in want; need_nz = "NoHz" in want or "random" in want   # random reuses a planner shell (prim only, no scoring)
    pl_hz = BeamPlanner(ckpt=a.hz_ckpt) if need_hz else None
    pl_nz = BeamPlanner(ckpt=a.nohz_ckpt) if need_nz else None
    allm = {"Hz": ("Hz", pl_hz, "model"), "NoHz": ("NoHz", pl_nz, "model"), "random": ("random", pl_nz, "uniform")}
    models = [allm[m] for m in want]
    if a.end >= 0:
        base = stratified(a.n_per_tier, a.key) if a.strat else full_episodes(a.key)
        samp = base[a.start:a.end]
    else:
        samp = stratified(a.n_per_tier, a.key)
    print(f"  {len(samp)} episodes; budget={a.budget} hmax={a.hmax} key={os.path.basename(a.key)}; warming up {a.warmup}/model...", flush=True)
    x0, o0, _ = samp[0]; xp0 = str(resolve(x0)); e0 = make_env(xp0); g0 = extract_goal_with_fallback(xp0, FALLBACK_GOAL)
    e0.set_robot_goal(*g0); e0.get_reachable_objects(); s00 = e0.get_full_state()
    if "Hz" in want or "NoHz" in want:          # warmup only matters for the (scored) model path; random needs none
        for _ in range(a.warmup):
            for pl in (pl_hz, pl_nz):
                if pl is not None:
                    candidates(pl, e0, g0, xp0, s00, a.hmax, "model", "mean5", rng, restrict_obj=o0)
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
            r = timed_bf(pl, env, goal, xmlp, obj, s0, gp, prior, "mean5", "q", a.budget, rng, a.hmax)
            r.update({"model": name, "tier": t, "xml": os.path.basename(xml), "xml_full": xml, "object_id": obj})
            fh.write(json.dumps(r) + "\n")
        if i % 10 == 0:
            fh.flush(); print(f"  [{i}/{len(samp)}] {PC() - t_start:.0f}s", file=sys.stderr, flush=True)
    fh.close()
    print(f"DONE {a.out}  ({PC() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
