"""The canonical best-first region-opening search.

Moved here from scripts/sandbox/eval_bestfirst.py, eval_m3.py and
scorer_beam.py, unchanged. It used to live in the sandbox, and both production
and the evaluation harness reached it by inserting scripts/sandbox on sys.path
at call time. That satisfied parity, since there was only ever one copy, but it
left the robot's behaviour resting on a directory whose purpose is being edited
freely, and it dragged NAMO_SCRATCH into production because eval_bestfirst
imports namo.paths for its CLI.

The dependency now runs the other way. This module owns the search and imports
nothing from scripts/; the sandbox scripts import it. Parity still holds by
construction, and test_best_first_sandbox_contract.py pins the exact chains and
simulation counts so the move itself can be shown not to have changed them.

The function bodies are the originals, character for character, with one
exception: make_pop is imported lazily inside the tracing branch, because
viz.trace_schema lives under scripts/ and only visualization callers pass
trace_out.
"""

from __future__ import annotations

import heapq
import time

import namo_rl


def _make_pop(*args, **kwargs):
    """Trace-row builder, resolved on use: viz lives in scripts/, not the package."""
    from viz.trace_schema import make_pop

    return make_pop(*args, **kwargs)


def make_action(obj, goal):
    a = namo_rl.Action()
    a.object_id = obj
    a.x = float(goal.x)
    a.y = float(goal.y)
    a.theta = float(goal.theta)
    a.edge_idx = int(goal.edge_idx)   # MUST be >= 0
    a.depth = int(goal.depth)         # MUST be >= 0
    return a

def rank_first_pushes_h2(planner, env, robot_goal, xml, s0, h, restrict_obj=None, score=True, raw=False,
                          return_grid=False, region_samples=None):
    """Rank reachable (obj, edge, depth) first pushes by Q(s0, ., h). ZERO sims.
    Returns [(obj, Goal, value)] sorted desc. restrict_obj (per-episode invariant): if set, consider ONLY
    that object (or those objects, when given a collection) — the search must push a boundary blocker,
    not an unrelated reachable object. Mirrors BeamPlanner._candidates' reachability pooling at budget h.
    score=False: skip the model forward pass entirely, return q=0.0 for every candidate (the RANDOM baseline
    must not touch the model — same candidate SET, no scores; caller assigns random priority).
    return_grid (default False, opt-in — existing callers' return shape is unaffected): also return the last
    computed (60,5) score grid P as a plain nested list (None if score=False or no object was scored) — lets
    callers that need the grid (e.g. a search trace) reuse this pass instead of paying a second forward one."""
    env.set_full_state(s0)
    reach_objs = list(env.get_reachable_objects())          # warms wavefront
    if restrict_obj is not None:
        allowed = {restrict_obj} if isinstance(restrict_obj, str) else set(restrict_obj)
        reach_objs = [o for o in reach_objs if o in allowed]       # ONLY the boundary object(s)
    redges = {o: set(env.get_reachable_edges(o)) for o in reach_objs}
    pool = []
    grid = None
    for obj in reach_objs:
        if not redges[obj]:
            continue
        if score:
            P = planner.scorer.score_state(
                env, obj, robot_goal, xml, region_samples=region_samples, h=h, raw=raw
            )   # (60,5) at budget h
            env.set_full_state(s0)                                            # score_state may move state
            ndepth = P.shape[1]
            if return_grid:
                grid = P.tolist()
        else:
            P = None; ndepth = 5                                              # uniform baseline: no model call
        goals_per_edge = planner.prim.generate_goals(obj, s0, env, max_goals=0)
        for edge_goals in goals_per_edge:
            for g in edge_goals:
                if g is None:
                    continue
                e = int(getattr(g, "edge_idx", -1)); d = int(getattr(g, "depth", -1))
                if e in redges[obj] and 0 <= d < ndepth:
                    pool.append((obj, g, float(P[e, d]) if score else 0.0))
    pool.sort(key=lambda x: -x[2])
    if return_grid:
        return pool, grid
    return pool

def candidates(planner, env, goal, xml, state, h, prior, agg, rng, restrict_obj=None, raw=True,
               want_grid=False, region_samples=None):
    """Reachable pushes from `state` (restricted to restrict_obj = the labeled object) with a priority-base
    value + the state value V. model: q = Q(state,a,h); V = agg of top Q (mean5 robust, or max). uniform: random q, V=0.
    want_grid (viz only, default False = no extra cost): also return the model's (60,5) score grid for `state`,
    reusing the forward pass rank_first_pushes_h2 already made — None when prior=uniform or the pool is empty."""
    if want_grid:
        pool, grid = rank_first_pushes_h2(planner, env, goal, xml, state, h, restrict_obj=restrict_obj,
                                          score=(prior != "uniform"), raw=raw, return_grid=True,
                                          region_samples=region_samples)
    else:
        pool = rank_first_pushes_h2(planner, env, goal, xml, state, h, restrict_obj=restrict_obj,
                                    score=(prior != "uniform"), raw=raw,
                                    region_samples=region_samples)  # uniform: skip the model forward pass
        grid = None
    if not pool:
        return [], 0.0, None
    if prior == "uniform":
        out = [(o, g, rng.random()) for (o, g, _q) in pool]
        return out, 0.0, grid                                        # no state value for the random baseline (grid is None)
    qs = sorted((q for (_o, _g, q) in pool), reverse=True)
    V = (sum(qs[:5]) / min(5, len(qs))) if agg == "mean5" else qs[0]
    return pool, V, grid


def priority(q, V, combine):
    if combine == "q":       return q                 # Layer-1: raw action-value (perfect-model baseline)
    if combine == "product": return q * V             # both must be high
    return 0.5 * q + 0.5 * V                           # blend (default): action score tempered by state value


def _update_w_on_fail(board, q_failed, discount, gamma, tau, g_table, gkmax, eps, child_patience=1):
    """A candidate of `board` was simulated and DID NOT open the goal. Always bump k_failed (for the lifetime
    log). Demote w ONLY for child boards (depth>=1) and ONLY when a discount mode is active; root w frozen=1.
    free_strikes (per board, default 0): that many initial failures are ignored by the demotion (patience)."""
    board["k_failed"] += 1
    if board["depth"] < 1 or discount == "off":
        return
    k = board["k_failed"] - board.get("free_strikes", 0)
    if k <= 0:
        return
    if discount == "gamma" and k % child_patience == 0:
        board["w"] *= gamma
    elif discount == "conf":
        board["w"] *= (1.0 - q_failed) ** tau
    elif discount == "fitted":
        board["w"] = board.get("w0", 1.0) * g_table[min(k, gkmax)]
    if board["w"] < eps:
        board["w"] = eps


_FAILTYPE = {"0": "NONE", "1": "ROBOT_PLACEMENT_COLLISION", "2": "OBJECT_COLLISION_DURING_PUSH",
             "3": "OBJECT_STUCK", "4": "NO_REACHABLE_EDGES", "5": "NO_PLAN_FOUND",
             "6": "ITERATION_LIMIT_REACHED", "7": "INVALID_PARAMETERS", "8": "UNKNOWN"}


def _unmoved(before, after, obj, tol=1e-6):
    """Did this push leave BOTH the pushed object and the robot exactly where they were?"""
    if before is None:
        return False
    for key in (f"{obj}_pose", "robot_pose"):
        a, b = before[key], after[key]
        if any(abs(float(a[i]) - float(b[i])) > tol for i in range(3)):
            return False
    return True


def solve_scene(planner, env, goal, xml, s0, hmax, sim_budget, prior, agg, combine, rng, restrict_obj=None,
                is_open=lambda e: e.is_robot_goal_reachable(), raw=True, dive_bonus=0.0,
                discount="off", gamma=0.65, tau=1.0, g_table=None, eps=1e-3,
                w0_mode="one", free_strike_q=2.0, child_patience=1, dedupe_noop=True,
                prune_jam_depth=True, trace_out=None, capture=None, timing=None,
                stop_on_open=True, win_bar=5, max_wins=64, region_samples=None,
                solution_out=None):
    """Greedy best-first ON THE LABELED OBJECT (restrict_obj). Returns (solved, sims, plan_len|None, boards, end).
    boards = per-board lifetime records; end in {solved, budget, exhausted}. w(b) via --discount (off=static).
    timing (dict, optional): filled with t_score/t_sim/t_wall/n_score for the wall-clock protocol. Always
    accumulated (perf_counter is ~50 ns against a ~1 s sim) so the TIMED search is literally THE canonical
    search -- never a fork of it. The retired scripts/sandbox/time_bestfirst.py kept its own copy of this
    loop, which predated dedupe_noop/prune_jam_depth and so silently timed a different, slower search.
    trace_out (list, viz only): every pop is appended as a make_pop row and every board also carries its full
    candidate pool + the model grid. None (default) = not one extra op anywhere.
    capture (viz only): capture(state) -> (geom, regions), recorded on EVERY pop (the state that push reached)
    and on every board (the state its candidates were generated at), so the page can redraw the scene where
    the search actually was instead of at the start state. Captured for the pop BEFORE the early return on
    success, so a solved episode's winning push -- which creates no board at all -- is recorded too; the child
    board spawned by a failed pop reuses that same capture rather than paying for the identical state twice."""
    gkmax = (max(g_table) if g_table else 0)
    # (board, edge) -> shallowest depth known to jam there. push_steps = depth+1 and the controller runs
    # ONE continuous push, so depth k+1 is depth k's trajectory continued: if the robot jams partway
    # through k it hits the same obstruction at the same tick for every deeper k'. Verified on a full
    # arm: 1214 of 1215 such pairs held (the one exception is the sim's known ~0.3mm warmstart jitter).
    # Pruning upward only -- a SHORTER push may stop before the obstruction, so those stay.
    jam_at = {}
    tm = timing if timing is not None else {}          # wall-clock accumulators (caller-owned; {} = discarded)
    tm["t_score"] = tm.get("t_score", 0.0); tm["t_sim"] = tm.get("t_sim", 0.0); tm["n_score"] = tm.get("n_score", 0)
    _t_wall0 = time.perf_counter()
    tracing = trace_out is not None
    heap = []; sims = 0
    boards = []                                                    # index == board_id
    ctr = [0]

    def next_ctr():
        ctr[0] += 1; return ctr[0] - 1

    def new_board(depth, npool, w0=1.0, free_strikes=0, parent_edge=-1, parent_depth=-1,
                  pool_rows=None, grid=None, state=None, geom=None, regions=None):
        w0 = min(max(w0, eps), 1.0)
        if geom is None and capture is not None:      # already captured by the pop that reached this state?
            geom, regions = capture(state)
        boards.append({"board_id": len(boards), "depth": depth, "n_candidates": npool,
                       "k_failed": 0, "w": w0, "w0": w0, "free_strikes": free_strikes, "tries": [],
                       "parent_edge": parent_edge, "parent_depth": parent_depth,
                       "pool": pool_rows, "grid": grid, "geom": geom, "regions": regions})
        return boards[-1]

    def push(item, bp, board):
        item["bp"] = bp; item["board_id"] = board["board_id"]
        item["se"] = bp * board["w"]
        heapq.heappush(heap, (-item["se"], next_ctr(), item))

    def trace_rows(cand):                                          # every candidate of a board, popped or not
        return [{"obj": o, "edge": int(g.edge_idx), "depth": int(g.depth), "q": float(q)} for (o, g, q) in cand]

    _t = time.perf_counter()
    pool, V0, grid0 = candidates(planner, env, goal, xml, s0, hmax, prior, agg, rng, restrict_obj=restrict_obj,
                                 raw=raw, want_grid=tracing, region_samples=region_samples)
    tm["t_score"] += time.perf_counter() - _t; tm["n_score"] += 1
    root = new_board(0, len(pool), pool_rows=(trace_rows(pool) if tracing else None), grid=grid0, state=s0)
    for (obj, g, q) in pool:                              # roots: ndone=0
        push({"obj": obj, "g": g, "from": s0, "ndone": 0, "plan": [(obj, g)], "q": q},
             priority(q, V0, combine), root)
    while heap and sims < sim_budget:
        _negpr, _cc, it = heapq.heappop(heap)
        board = boards[it["board_id"]]
        cur = it["bp"] * board["w"]                       # lazy stale-reinsert (w only decreases)
        if cur < it["se"] - 1e-12:
            it["se"] = cur; heapq.heappush(heap, (-cur, next_ctr(), it)); continue
        _jk = (it["board_id"], int(it["g"].edge_idx))
        _jd = jam_at.get(_jk)
        if prune_jam_depth and _jd is not None and int(it["g"].depth) >= _jd:
            continue                                      # same trajectory, already known to jam -- no sim
        env.set_full_state(it["from"])
        obs_before = env.get_observation() if dedupe_noop else None
        _t = time.perf_counter()
        step_res = env.step(make_action(it["obj"], it["g"])); sims += 1
        tm["t_sim"] += time.perf_counter() - _t
        opened = bool(is_open(env))
        # WHY the push failed, not just that it did. The skill maps causes into failure_type and
        # collision_object (namo_push_skill.cpp:183-196): a wall collision and a jam against another
        # movable object are completely different situations, and reason alone cannot tell them apart.
        _i = step_res.info or {}
        fail = None
        if _i.get("failure_reason"):
            fail = {"reason": _i["failure_reason"], "type": _FAILTYPE.get(str(_i.get("failure_type")), str(_i.get("failure_type"))),
                    "collision": _i.get("collision_object") or "", "movable": _i.get("movable_collisions") or "",
                    "wall": str(_i.get("wall_collision")).lower() == "true"}
        # VIZ ONLY. The state this push REACHED, read while the sim is still standing in it and before the
        # `opened` return, so the winning push of a solved episode is recorded like any other. Nothing has
        # touched the sim since env.step() -- is_open only reads.
        s_after = pop_geom = pop_regions = None
        if capture is not None:
            s_after = env.get_full_state()
            pop_geom, pop_regions = capture(s_after)
        if tracing:
            trace_out.append(_make_pop(sims, it["board_id"], it["obj"], int(it["g"].edge_idx),
                                      int(it["g"].depth), float(it["q"]), float(it["bp"]),
                                      float(board["w"]), opened, geom=pop_geom, regions=pop_regions,
                                      fail=fail))
        if fail is not None:
            _d0 = int(it["g"].depth)
            if _jd is None or _d0 < _jd:
                jam_at[_jk] = _d0
        board["tries"].append((len(board["tries"]) + 1, float(it["q"]), opened))   # (within-board try#, q, opened)
        if opened:
            tm["t_wall"] = time.perf_counter() - _t_wall0
            if solution_out is not None:
                solution_out["plan"] = list(it["plan"])
                solution_out["state"] = env.get_full_state()
            if stop_on_open:
                return True, sims, len(it["plan"]), boards, "solved"
            # multi-solution collection mode (round-3 doctrine): record the win, keep mining unless
            # the model met the deploy bar on its FIRST win (nothing to learn) or the win-cap hit.
            _wins = boards[0].setdefault("_wins", [])
            _wins.append((sims, len(it["plan"])))
            if (len(_wins) == 1 and sims <= win_bar) or len(_wins) >= max_wins:
                return True, sims, len(it["plan"]), boards, "solved"
            continue
        _update_w_on_fail(board, float(it["q"]), discount, gamma, tau, g_table, gkmax, eps, child_patience)
        ndone = it["ndone"] + 1
        # A push that moved NOTHING reaches the state it started from, so the child board would be a
        # byte-identical duplicate of this one -- same pool, same scores -- and would simply re-offer the
        # pushes we just tried. Measured: 27.4% of all sims are spent on such duplicates, and no solved
        # episode has ever won from one. Adopted on by default; --no-dedupe-noop restores the legacy path.
        if dedupe_noop and _unmoved(obs_before, env.get_observation(), it["obj"]):
            continue
        if ndone < hmax:                                  # room for another push -> expand the reached state
            s_new = env.get_full_state() if s_after is None else s_after
            h = hmax - ndone
            _t = time.perf_counter()
            pool2, V, grid2 = candidates(planner, env, goal, xml, s_new, h, prior, agg, rng,
                                         restrict_obj=restrict_obj, raw=raw, want_grid=tracing,
                                         region_samples=region_samples)
            tm["t_score"] += time.perf_counter() - _t; tm["n_score"] += 1
            child = new_board(ndone, len(pool2),
                              w0=(V if w0_mode == "v" else 1.0),
                              free_strikes=(1 if float(it["q"]) >= free_strike_q else 0),
                              parent_edge=(int(it["g"].edge_idx) if tracing else -1),
                              parent_depth=(int(it["g"].depth) if tracing else -1),
                              pool_rows=(trace_rows(pool2) if tracing else None),
                              grid=grid2, state=s_new, geom=pop_geom, regions=pop_regions)
            for (obj2, g2, q2) in pool2:                  # children: +dive_bonus*ndone bias (kept for parity)
                push({"obj": obj2, "g": g2, "from": s_new, "ndone": ndone,
                      "plan": it["plan"] + [(obj2, g2)], "q": q2},
                     priority(q2, V, combine) + dive_bonus * ndone, child)
    end = "budget" if sims >= sim_budget else "exhausted"
    tm["t_wall"] = time.perf_counter() - _t_wall0
    _wins = boards[0].get("_wins") if boards else None
    if _wins:
        return True, _wins[0][0], _wins[0][1], boards, "solved_mined"
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
