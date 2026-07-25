#!/usr/bin/env python3
"""VALUE-GUIDED GREEDY BEST-FIRST search — objective = MIN TOTAL SIMULATED PUSHES to open the path.

The design we converged on (not MCTS, not admissible-A*, not budget-layered):
  * One priority queue of UNSIMULATED candidate pushes. Pop the most-promising, SIMULATE it (the only
    expensive op, ~1s), check goal (free), stop on first open. NO pruning -> complete within Hmax.
  * Objective is min SIMS, not min depth -> NO layering, NO g-term. Values are compared ACROSS depths:
    a deep-but-near-solution node outranks a fresh shallow first push, because it needs fewer more sims.
  * Q (per-action prior, PRE-sim) guides EXPANSION (which pushes to add). V=mean_top5(Q(s,.)) (per-state,
    POST-sim leaf value) guides SELECTION (which branch to chase). priority = combine(Q(s,a), V(s)).
  * sim_budget = the single reactive<->search dial (tiny -> reactive single best path; larger -> search).

FAILURE-DISCOUNT extension (--discount, default off => bit-identical to the static queue):
  Per-BOARD weight w(b). Effective priority of candidate a on board b = combine(q,V)*w(b).
  Root board (depth 0): w=1 ALWAYS. Child board (depth 1, post-setup finish state): w starts 1, and is
  updated ONLY by FAILED sims of that board's OWN candidates (a failed finish push demotes its siblings ->
  children of wrong roots stop flooding the queue). Lazy stale-reinsert on pop (w only decreases). w floored
  at EPS (never pruned). Modes: off | gamma (w*=GAMMA/fail) | fitted (w=g_table[min(k,kmax)]) | conf
  (w*=(1-q_failed)^TAU). LIFETIME LOGGING is always on when --lifetime-out is set (cheap): per-board records.

Baseline: --prior uniform = identical loop, RANDOM order, no value -> proves the guidance is worth it.

  python scripts/sandbox/eval_bestfirst.py --ckpt <ckpt> --manifest <pure2push.txt> --hmax 2 \
      --sim-budget 900 --prior model --agg mean5 --combine q --discount off --start 0 --end 985 --out <json>
"""
import sys, os, json, time, argparse, random, heapq
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
from scorer_beam import BeamPlanner, make_env, make_action, read_manifest, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import MANIFESTS, DATASETS, SCRATCH  # noqa: E402

PURE2PUSH = str(MANIFESTS / "test_pure2_fromkey.txt")


def candidates(planner, env, goal, xml, state, h, prior, agg, rng, restrict_obj=None, raw=False):
    """Reachable pushes from `state` (restricted to restrict_obj = the labeled object) with a priority-base
    value + the state value V. model: q = Q(state,a,h); V = agg of top Q (mean5 robust, or max). uniform: random q, V=0."""
    pool = rank_first_pushes_h2(planner, env, goal, xml, state, h, restrict_obj=restrict_obj,
                                score=(prior != "uniform"), raw=raw)          # uniform: skip the model forward pass
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


def _update_w_on_fail(board, q_failed, discount, gamma, tau, g_table, gkmax, eps):
    """A candidate of `board` was simulated and DID NOT open the goal. Always bump k_failed (for the lifetime
    log). Demote w ONLY for child boards (depth>=1) and ONLY when a discount mode is active; root w frozen=1.
    free_strikes (per board, default 0): that many initial failures are ignored by the demotion (patience)."""
    board["k_failed"] += 1
    if board["depth"] < 1 or discount == "off":
        return
    k = board["k_failed"] - board.get("free_strikes", 0)
    if k <= 0:
        return
    if discount == "gamma":
        board["w"] *= gamma
    elif discount == "conf":
        board["w"] *= (1.0 - q_failed) ** tau
    elif discount == "fitted":
        board["w"] = board.get("w0", 1.0) * g_table[min(k, gkmax)]
    if board["w"] < eps:
        board["w"] = eps


def solve_scene(planner, env, goal, xml, s0, hmax, sim_budget, prior, agg, combine, rng, restrict_obj=None,
                is_open=lambda e: e.is_robot_goal_reachable(), raw=False, dive_bonus=0.0,
                discount="off", gamma=0.65, tau=1.0, g_table=None, eps=1e-3,
                w0_mode="one", free_strike_q=2.0):
    """Greedy best-first ON THE LABELED OBJECT (restrict_obj). Returns (solved, sims, plan_len|None, boards, end).
    boards = per-board lifetime records; end in {solved, budget, exhausted}. w(b) via --discount (off=static)."""
    gkmax = (max(g_table) if g_table else 0)
    heap = []; sims = 0
    boards = []                                                    # index == board_id
    ctr = [0]

    def next_ctr():
        ctr[0] += 1; return ctr[0] - 1

    def new_board(depth, npool, w0=1.0, free_strikes=0):
        w0 = min(max(w0, eps), 1.0)
        boards.append({"board_id": len(boards), "depth": depth, "n_candidates": npool,
                       "k_failed": 0, "w": w0, "w0": w0, "free_strikes": free_strikes, "tries": []})
        return boards[-1]

    def push(item, bp, board):
        item["bp"] = bp; item["board_id"] = board["board_id"]
        item["se"] = bp * board["w"]
        heapq.heappush(heap, (-item["se"], next_ctr(), item))

    pool, V0 = candidates(planner, env, goal, xml, s0, hmax, prior, agg, rng, restrict_obj=restrict_obj, raw=raw)
    root = new_board(0, len(pool))
    for (obj, g, q) in pool:                              # roots: ndone=0
        push({"obj": obj, "g": g, "from": s0, "ndone": 0, "plan": [(obj, g)], "q": q},
             priority(q, V0, combine), root)
    while heap and sims < sim_budget:
        _negpr, _cc, it = heapq.heappop(heap)
        board = boards[it["board_id"]]
        cur = it["bp"] * board["w"]                       # lazy stale-reinsert (w only decreases)
        if cur < it["se"] - 1e-12:
            it["se"] = cur; heapq.heappush(heap, (-cur, next_ctr(), it)); continue
        env.set_full_state(it["from"]); env.step(make_action(it["obj"], it["g"])); sims += 1
        opened = bool(is_open(env))
        board["tries"].append((len(board["tries"]) + 1, float(it["q"]), opened))   # (within-board try#, q, opened)
        if opened:
            return True, sims, len(it["plan"]), boards, "solved"
        _update_w_on_fail(board, float(it["q"]), discount, gamma, tau, g_table, gkmax, eps)
        ndone = it["ndone"] + 1
        if ndone < hmax:                                  # room for another push -> expand the reached state
            s_new = env.get_full_state()
            h = hmax - ndone
            pool2, V = candidates(planner, env, goal, xml, s_new, h, prior, agg, rng,
                                  restrict_obj=restrict_obj, raw=raw)
            child = new_board(ndone, len(pool2),
                              w0=(V if w0_mode == "v" else 1.0),
                              free_strikes=(1 if float(it["q"]) >= free_strike_q else 0))
            for (obj2, g2, q2) in pool2:                  # children: +dive_bonus*ndone bias (kept for parity)
                push({"obj": obj2, "g": g2, "from": s_new, "ndone": ndone,
                      "plan": it["plan"] + [(obj2, g2)], "q": q2},
                     priority(q2, V, combine) + dive_bonus * ndone, child)
    end = "budget" if sims >= sim_budget else "exhausted"
    return False, sims, None, boards, end


def _finalize_boards(boards, ep):
    """Turn raw board dicts into lifetime JSONL rows: terminal status + winner try for the g(k) fit.
    status: winner (a try opened) | exhausted (all n_candidates tried, none opened) | censored (untried remain)."""
    rows = []
    for b in boards:
        tries = b["tries"]; n_tried = len(tries)
        win_try = next((t[0] for t in tries if t[2]), None)
        if win_try is not None:
            status = "winner"
        elif n_tried >= b["n_candidates"]:
            status = "exhausted"
        else:
            status = "censored"
        rows.append({**ep, "board_id": b["board_id"], "depth": b["depth"],
                     "n_candidates": b["n_candidates"], "k_failed": b["k_failed"],
                     "n_tried": n_tried, "status": status, "winner_try": win_try,
                     "tries": [[t[0], round(t[1], 6), t[2]] for t in tries]})
    return rows


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
    ap.add_argument("--discount", default="off", choices=["off", "gamma", "fitted", "conf"],
                    help="per-board failure demotion. off=static queue (BIT-IDENTICAL baseline).")
    ap.add_argument("--gamma", type=float, default=0.65, help="--discount gamma: w *= gamma per failed sim")
    ap.add_argument("--tau", type=float, default=1.0, help="--discount conf: w *= (1-q_failed)^tau")
    ap.add_argument("--gtable", default="", help="--discount fitted: JSON {k: g_norm} (g_norm[0]=1)")
    ap.add_argument("--eps", type=float, default=1e-3, help="floor on w (never prune)")
    ap.add_argument("--w0-mode", default="one", choices=["one", "v"],
                    help="child board starting credibility: one=1.0 (default) | v=board value V (model prior)")
    ap.add_argument("--free-strike-q", type=float, default=2.0,
                    help="boards reached via a setup with q >= this get 1 free strike (2.0 = disabled)")
    ap.add_argument("--key", default=str(DATASETS / "namo_testset_v1/labels/pure2push.json"),
                    help="key: {xml: [ {object_id, region, ...} ]}. Search restricted to object_id per record.")
    ap.add_argument("--seed-base", type=int, default=7000,
                    help="RNG base for the uniform baseline; model is deterministic so only matters for --prior uniform.")
    ap.add_argument("--raw", action="store_true", help="use raw HL-Gauss E[bin] (no sigmoid) for the priority")
    ap.add_argument("--dive-bonus", type=float, default=0.0, help="CASCADE dive bonus per push-already-done (default 0)")
    ap.add_argument("--success", default="region", choices=["region", "point"])
    ap.add_argument("--out", default=str(SCRATCH / "eval/bestfirst.json"))
    ap.add_argument("--leaf-out", default=str(SCRATCH / "eval/bestfirst.jsonl"))
    ap.add_argument("--lifetime-out", default="", help="per-board lifetime JSONL (one row per board).")
    a = ap.parse_args()

    import os as _os
    g_table = None
    if a.discount == "fitted":
        raw = json.load(open(a.gtable)); g_table = {int(k): float(v) for k, v in raw.items()}
    key = json.load(open(a.key)); keyrp = {_os.path.realpath(k): v for k, v in key.items()}
    planner = BeamPlanner(ckpt=a.ckpt)
    print(f"device={planner.scorer.device} hmax={a.hmax} sim_budget={a.sim_budget} prior={a.prior} "
          f"agg={a.agg} combine={a.combine} discount={a.discount} gamma={a.gamma} key={_os.path.basename(a.key)} "
          f"success={a.success}", flush=True)
    xmls = read_manifest(a.manifest, None)[a.start:a.end]
    n = n_solved = n_already = n_norec = sims_tot = sims_solved = 0; t0 = time.time()
    lf = open(a.leaf_out, "w")
    ltf = open(a.lifetime_out, "w") if a.lifetime_out else None
    ep_ctr = 0
    for xi, xml in enumerate(xmls):
        try:
            recs = key.get(xml) or keyrp.get(_os.path.realpath(xml))
            if not recs:
                n_norec += 1; continue
            env = make_env(xml)
            goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects()
            if a.success == "region":
                gp = sample_goal_points(env)
                is_open = (lambda e, p=gp: goal_open_pts(e, p))
            else:
                is_open = (lambda e: e.is_robot_goal_reachable())
            if is_open(env):
                n_already += 1; continue
            s0 = env.get_full_state()
            for ri, rec in enumerate(recs):
                rng = random.Random(a.seed_base + xi * 17 + ri)
                obj = rec.get("object_id")
                solved, sims, plen, boards, end = solve_scene(
                    planner, env, goal, xml, s0, a.hmax, a.sim_budget, a.prior, a.agg, a.combine, rng,
                    restrict_obj=obj, is_open=is_open, raw=a.raw, dive_bonus=a.dive_bonus,
                    discount=a.discount, gamma=a.gamma, tau=a.tau, g_table=g_table, eps=a.eps, w0_mode=a.w0_mode, free_strike_q=a.free_strike_q)
                n += 1; sims_tot += sims; n_solved += int(solved); sims_solved += sims if solved else 0
                lf.write(json.dumps({"xml": xml, "object_id": obj, "region": rec.get("region"),
                                     "solved": solved, "sims": sims, "plan_len": plen}) + "\n")
                if ltf is not None:
                    ep = {"ep": ep_ctr, "xml": xml, "object_id": obj, "region": rec.get("region"),
                          "solved": solved, "sims": sims, "end": end}
                    for row in _finalize_boards(boards, ep):
                        ltf.write(json.dumps(row) + "\n")
                ep_ctr += 1
            if xi % 20 == 0:
                print(f"  [{xi}/{len(xmls)}] episodes={n} solved={n_solved} avg_sims={sims_tot/max(n,1):.1f} "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
        except Exception as ex:
            print(f"  scene {xi} err: {ex}", file=sys.stderr); continue
    lf.close()
    if ltf is not None:
        ltf.close()
    res = {"ckpt": a.ckpt, "hmax": a.hmax, "sim_budget": a.sim_budget, "prior": a.prior, "agg": a.agg,
           "combine": a.combine, "discount": a.discount, "gamma": a.gamma, "dive_bonus": a.dive_bonus,
           "key": _os.path.basename(a.key), "n_episodes": n,
           "n_already_open": n_already, "n_no_record": n_norec,
           "solve_rate": round(100.0 * n_solved / max(n, 1), 1),
           "avg_sims_all": round(sims_tot / max(n, 1), 2),
           "avg_sims_to_solve": round(sims_solved / max(n_solved, 1), 2)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
