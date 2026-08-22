#!/usr/bin/env python3
"""PURE REACTIVE MPC = repeat[ argmax push -> step -> opened? ] up to --max-pushes times  (region criterion, object-constrained).

Zero search: each push is the model's current top pick (or a RANDOM pick for the floor) at the LIVE state, then we act
for real and check if the region opened. No simulate-and-undo — MPC can't take a push back, but it CAN keep pushing.
  - push i = argmax Q(s_{i-1}, ., H)  (the model's #1 push at the live state, object-constrained to the labeled object)
       query budget H = --h for push 1 (2=foresight / 1=told-you-have-1-push), H=1 for pushes 2..k
       (NoHz models ignore H, and the RANDOM pool is H-independent -> the depth-k result does not depend on --h)
  - graded by goal_open_pts (>=20% of s0-sampled goal-region points reachable) = the canonical region criterion.
  - EARLY STOP: region opens (record push index it opened at) OR the candidate pool empties.
  - JAM GUARD (--dedupe-noop / --jam-prune, both ON by default = what the canonical search runs):
       Without it the policy LOCKS UP. A push that jams leaves the state untouched, so re-ranking that
       identical state returns the identical argmax, so the policy re-picks the same jammed push until
       the budget runs out. Measured on 20 failed easy-2push episodes: 2.2 distinct pushes across 10
       steps, 8.75 no-ops, 45% of episodes picking one push all ten times. The search cannot stall this
       way -- it pops each candidate once, drops no-op children (dedupe_noop) and skips deeper pushes on
       a jammed edge (prune_jam_depth). Ported here as a per-state ban list, cleared the moment the
       object actually moves, which is the search opening a fresh child board.
Each leaf records `opened_at` in 1..k (or 0 = never), so cumulative open@1..open@k for every k<=max_pushes come from ONE run.
Default --max-pushes 2 is EXACTLY the old setup+finish (argmax setup@H -> argmax finish@H1) -> backward compatible."""
import sys, os, json, argparse, random, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", f"{REPO}/scripts/pipeline", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts  # noqa: E402
from namo.planners.opening.best_first_search import _unmoved  # noqa: E402 - ONE definition of "moved nothing"
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import DATASETS, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", default=str(DATASETS / "namo_testset_v1/labels/pure2push.json"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = to end (xml-index shard)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaf-out", default="", help="optional per-episode jsonl: {xml,object_id,region,opened_at,open1..openK}")
    ap.add_argument("--h", type=int, default=2, help="query budget for the FIRST push (2=foresight; 1=told-you-have-1-push); pushes 2..k query H=1")
    ap.add_argument("--max-pushes", type=int, default=2,
                    help="depth-k reactive MPC budget: keep pushing (argmax/random) up to k times, early-stop on open or empty pool. Default 2 = old setup+finish (backward compatible).")
    ap.add_argument("--prior", default="q", choices=["q", "uniform"], help="q=argmax(model); uniform=RANDOM pick, no model")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--no-dedupe-noop", dest="dedupe_noop", action="store_false", default=True,
                    help="legacy: let the policy re-pick a push that moved nothing (it then loops until budget)")
    ap.add_argument("--no-jam-prune", dest="jam_prune", action="store_false", default=True,
                    help="legacy: do not skip deeper pushes on an edge already known to jam at this state")
    a = ap.parse_args()
    K = max(1, a.max_pushes)
    rng = random.Random(a.seed)
    pl = BeamPlanner(ckpt=a.ckpt)
    key = json.load(open(a.key))
    xmls = list(key); xmls = xmls[a.start:(a.end if a.end else len(xmls))]
    n = 0; skip = 0; leaf = []
    t_all = 0.0; t_solved = 0.0   # wall-time (ranking+sims only, excl env build) over counted / solved episodes
    opened = [0] * (K + 1)   # opened[k] = #episodes that opened AT push k (index 1..K); opened[0] unused
    for xi, xml in enumerate(xmls):
        for rec in key[xml]:
            obj = rec["object_id"]; reg = rec.get("region")
            try:
                xmlp = str(resolve(xml)); env = make_env(xmlp); goal = extract_goal_with_fallback(xmlp, FALLBACK_GOAL)
                env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state()
                gp = sample_goal_points(env)
            except Exception:
                skip += 1; continue
            if not gp or goal_open_pts(env, gp):
                skip += 1; continue
            # push 1 decides skip semantics: no candidate for the labeled object -> not a valid episode
            t0 = time.time()   # time the decision+sim work (ranking + pushes), NOT the fixed env-build overhead
            pool = rank_first_pushes_h2(pl, env, goal, xml, s0, a.h, restrict_obj=obj, score=(a.prior == "q"))
            if not pool:
                skip += 1; continue
            n += 1
            opened_at = 0
            n_push = 0                                                         # pushes SIMULATED = sims spent
            n_noop = 0
            banned = set()          # (edge, depth) already tried AT THIS STATE
            jam_at = {}             # edge -> shallowest depth known to jam AT THIS STATE
            s_cur = s0
            for pidx in range(1, K + 1):
                if pidx > 1:  # re-rank at the LIVE state; finishes query H=1
                    pool = rank_first_pushes_h2(pl, env, goal, xml, s_cur, 1, restrict_obj=obj, score=(a.prior == "q"))
                live = [(o, g, q) for (o, g, q) in pool
                        if (int(g.edge_idx), int(g.depth)) not in banned
                        and not (a.jam_prune and jam_at.get(int(g.edge_idx)) is not None
                                 and int(g.depth) >= jam_at[int(g.edge_idx)])]
                if not live:
                    break  # candidate pool empty -> stop early
                _o, g, _q = live[0] if a.prior == "q" else rng.choice(live)    # ARGMAX or RANDOM push
                env.set_full_state(s_cur)
                obs_before = env.get_observation() if a.dedupe_noop else None
                res = env.step(make_action(obj, g)); n_push += 1
                if goal_open_pts(env, gp):
                    opened_at = pidx; break                                    # region opened -> stop
                edge, depth = int(g.edge_idx), int(g.depth)
                if a.jam_prune and (res.info or {}).get("failure_reason"):
                    jd = jam_at.get(edge)
                    if jd is None or depth < jd:
                        jam_at[edge] = depth       # this edge's trajectory jams from here on down
                if a.dedupe_noop and _unmoved(obs_before, env.get_observation(), obj):
                    n_noop += 1
                    banned.add((edge, depth))      # same state, same pool -> never re-offer this push
                    continue                       # s_cur unchanged; do NOT clear the ban list
                banned.clear(); jam_at.clear()     # the object moved: a new state is a new board
                s_cur = env.get_full_state()
            dt = time.time() - t0
            t_all += dt
            if opened_at:
                opened[opened_at] += 1
                t_solved += dt
            r_leaf = {"xml": xml, "object_id": obj, "region": reg, "opened_at": opened_at,
                      "n_push": n_push, "n_noop": n_noop, "t_ep": round(dt, 4)}
            for k in range(1, max(K, 2) + 1):                                  # open1,open2 always present (old aggregator) + open3..openK
                r_leaf[f"open{k}"] = int(0 < opened_at <= k)
            leaf.append(r_leaf)
        if xi % 25 == 0:
            print(f"  [{xi}/{len(xmls)}] n={n} open@{K}={sum(opened[1:])}", file=sys.stderr, flush=True)
    cum = {k: sum(opened[1:k + 1]) for k in range(1, K + 1)}                   # cumulative open@k
    out = {"ckpt": os.path.basename(a.ckpt), "n": n, "skip": skip, "max_pushes": K, "h_first": a.h,
           "dedupe_noop": a.dedupe_noop, "jam_prune": a.jam_prune,
           "opened_at_hist": {str(k): opened[k] for k in range(1, K + 1)},
           "avg_t_ep_ms": round(1000 * t_all / max(n, 1), 1),
           "avg_t_solve_ms": round(1000 * t_solved / max(sum(opened[1:]), 1), 1),
           "total_t_s": round(t_all, 1),
           "open1": cum[1], "open2": cum.get(2, cum[1])}
    for k in range(1, K + 1):
        out[f"open{k}_total"] = cum[k]
        out[f"reactive_argmax@{k}"] = round(100 * cum[k] / max(n, 1), 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    if a.leaf_out:
        os.makedirs(os.path.dirname(a.leaf_out), exist_ok=True)
        with open(a.leaf_out, "w") as fh:
            for r in leaf:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
