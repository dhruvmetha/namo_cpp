"""MORE's own search, run unmodified against our simulator through a thin adapter.

MORE is Huang, Guo, Boularias and Yu, "Interleaving Monte Carlo Tree Search and Self-Supervised
Learning for Object Retrieval in Clutter" (ICRA 2022), code at github.com/arc-l/more. We import its
tree code at runtime from a pinned clone and never copy it: the repo has no license file. The
clone is `$MORE_REF`, default `$NAMO_SCRATCH/reference/arc-l_more`, and must sit at revision
70402c0 with the imported files unmodified; `load_more()` refuses anything else.

What is theirs, unmodified:
  mcts/nodes.py, mcts/search.py              collection search (plain UCT, uniform rollouts)
  mcts_network/nodes.py, mcts_network/search.py  deploy search (network prior, greedy rollouts)
  constants.py                                MCTS_DISCOUNT, MCTS_TOP, MCTS_UCT_RATIO, MCTS_EARLY_ROLLOUTS
Those files import only math, numpy, tqdm and constants. The PyBullet world lives in their
push.py (PushState) and mcts_utils.py (MCTSHelper); this module replaces exactly those two:

  NamoMove      PushMove      mcts_network/push.py:208-226   identity (object, edge, depth); q_value = prior
  MoreContext   MCTSHelper    mcts_utils.py:39-67, :635-669  caches, action library, simulator, call count
  NamoPushState PushState     mcts/push.py:22-153            reward, terminal test, cached transitions

The four approved substitutions live here and nowhere else: the (object, edge, depth) action
library for MORE's pixel pushes, the reachable goal-sample fraction for its grasp Q inside
push_result, a verified opening for "graspable", and a simulator-call budget where one is wanted.

One runtime change to their modules, display only: their loops are wrapped in tqdm progress
bars (mcts/search.py:11). We rebind that name to a disabled tqdm unless MORE_TQDM=1, because a
collection campaign runs millions of iterations. No logic is touched.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from namo.planners.search_measurements import clock_finish, clock_start

MORE_PIN = "70402c001c30908e7a70b93f8de3c31abdd26fdb"
_VENDOR_FILES = ("constants.py", "mcts/nodes.py", "mcts/search.py",
                 "mcts_network/nodes.py", "mcts_network/search.py")
_LOADED: Dict[str, SimpleNamespace] = {}


def more_ref_path() -> Path:
    ref = os.environ.get("MORE_REF")
    if ref:
        return Path(ref)
    scratch = os.environ.get("NAMO_SCRATCH")
    if not scratch:
        from namo.paths import SCRATCH as scratch
    return Path(scratch) / "reference" / "arc-l_more"


def _check_pin(ref: Path) -> None:
    head = subprocess.run(["git", "-C", str(ref), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if head != MORE_PIN:
        raise RuntimeError(f"MORE clone {ref} is at {head}, expected {MORE_PIN}")
    dirty = subprocess.run(["git", "-C", str(ref), "diff", "--quiet", MORE_PIN, "--", *_VENDOR_FILES])
    if dirty.returncode != 0:
        raise RuntimeError(f"MORE clone {ref} has local edits to {_VENDOR_FILES}")


def _exec(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_more(ref: Optional[Path] = None) -> SimpleNamespace:
    """Import MORE's four search modules and constants from the pinned clone, unmodified.

    Their nodes.py does `from constants import ...`, so their constants.py is registered as the
    top-level module `constants` only while those files execute, then the previous binding (if
    any) is restored. Our modules never see their names.
    """
    ref = Path(ref) if ref is not None else more_ref_path()
    key = str(ref.resolve())
    if key in _LOADED:
        return _LOADED[key]
    _check_pin(ref)
    prior = sys.modules.get("constants")
    constants = _exec("_more_ref_constants", ref / "constants.py")
    sys.modules["constants"] = constants
    try:
        mods = SimpleNamespace(
            ref=ref, constants=constants,
            uct_nodes=_exec("_more_ref_mcts_nodes", ref / "mcts" / "nodes.py"),
            uct_search=_exec("_more_ref_mcts_search", ref / "mcts" / "search.py"),
            net_nodes=_exec("_more_ref_mcts_network_nodes", ref / "mcts_network" / "nodes.py"),
            net_search=_exec("_more_ref_mcts_network_search", ref / "mcts_network" / "search.py"),
        )
    finally:
        if prior is None:
            del sys.modules["constants"]
        else:
            sys.modules["constants"] = prior
    if os.environ.get("MORE_TQDM", "0") != "1":
        import tqdm as _tqdm
        quiet = functools.partial(_tqdm.tqdm, disable=True)
        mods.uct_search.tqdm = quiet
        mods.net_search.tqdm = quiet
    mods.UctNode = mods.uct_nodes.PushSearchNode
    mods.UctSearch = mods.uct_search.MonteCarloTreeSearch
    mods.NetNode = mods.net_nodes.PushSearchNode
    mods.NetSearch = mods.net_search.MonteCarloTreeSearch
    _LOADED[key] = mods
    return mods


class MoreSolved(Exception):
    """Protocol stop: a simulated push verified an opening (anywhere, rollouts included)."""

    def __init__(self, plan, state):
        self.plan, self.state = plan, state


class MoreBudgetExhausted(Exception):
    """The simulator-call cap is reached; raised before the call that would exceed it."""


class NamoMove:
    """PushMove (mcts_network/push.py:208-226). MORE compares moves by geometry; ours by action id."""

    __slots__ = ("obj", "goal", "edge", "depth", "q_value")

    def __init__(self, obj, goal, q_value=0.0):
        self.obj, self.goal = obj, goal
        self.edge, self.depth = int(goal.edge_idx), int(goal.depth)
        self.q_value = float(q_value)

    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.obj, self.edge, self.depth)

    def __str__(self):
        return f"{self.obj}_{self.edge}_{self.depth}"

    def __repr__(self):
        return f"NamoMove({self.obj}, edge {self.edge}, depth {self.depth}, q {self.q_value:.3f})"

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)


class MoreContext:
    """MCTSHelper for our simulator.

    move_recorder and simulation_recorder are MORE's two caches (mcts_utils.py:61-66), with the same
    keys: a state's uid, and uid + str(move). A failed move is not recorded, as in MORE
    (mcts/push.py:90-95 returns before recording); the caller removes it from the state's action
    list instead (remove_action), which is what keeps it from being tried again.

    actions(state)  -> [(obj, Goal, q)]  the legal pushes at a simulator state, any order
    verifier(env)   -> (opened, frac)     the verified-opening test and the reachable fraction
    """

    def __init__(self, env, actions: Callable, verifier: Callable, *, dedupe_noop=True,
                 prune_jam_depth=True, sim_budget: Optional[int] = None, stop_on_open=False,
                 timing=None, measurements=None):
        self.env, self._actions, self._verifier = env, actions, verifier
        self.dedupe_noop, self.prune_jam_depth = dedupe_noop, prune_jam_depth
        self.sim_budget, self.stop_on_open = sim_budget, stop_on_open
        self.timing, self.measurements = timing, measurements
        self.calls = 0                      # every env.step, across resets
        self.first_open_call: Optional[int] = None
        self.first_open_plan: Optional[List[NamoMove]] = None
        self.legal: Dict[str, List[Tuple[int, int]]] = {}     # "search:uid" -> legal (edge, depth), before removals
        self.search_index = 0
        self.boards: List[Dict[str, Any]] = []
        self._board_of: Dict[str, int] = {}
        self.reset()

    def reset(self, search_index=None):
        """mcts_utils.py:64-66, called by MORE after every executed push (more_main.py:524).

        uids restart at "root" in every search, so anything kept across searches is keyed by
        `search_index` too.
        """
        if search_index is not None:
            self.search_index = search_index
        self.move_recorder: Dict[str, List[NamoMove]] = {}
        self.simulation_recorder: Dict[str, Tuple[Any, float, bool]] = {}
        self._jam_at: Dict[Tuple[str, str, int], int] = {}

    # -- the action library (MORE: sample_actions + the PPN read, mcts_network/push.py:140-160) --
    def actions_for(self, state: "NamoPushState") -> List[NamoMove]:
        t = clock_start(self.timing)
        pool = self._actions(state)
        clock_finish(self.timing, "t_score", t)
        if self.timing is not None:
            self.timing["n_score"] = self.timing.get("n_score", 0) + 1
        # Prior-independent order, as MORE's contour sampling is independent of its PPN. The order
        # decides MORE's tie-breaks (children are popped from the end), so it must not leak the prior.
        pool = sorted(pool, key=lambda t3: (str(t3[0]), int(t3[1].edge_idx), int(t3[1].depth)))
        self.legal[f"{self.search_index}:{state.uid}"] = [(int(g.edge_idx), int(g.depth)) for (_o, g, _q) in pool]
        board_id = len(self.boards)
        self._board_of[state.uid] = board_id
        parent = self._board_of.get(state.parent_uid) if state.parent_uid is not None else None
        last = state.plan[-1] if state.plan else None
        self.boards.append({"board_id": board_id, "depth": state.depth, "n_candidates": len(pool),
                            "k_failed": 0, "w": 1.0, "w0": 1.0, "free_strikes": 0, "tries": [],
                            "parent_edge": last.edge if last else -1,
                            "parent_depth": last.depth if last else -1,
                            "parent_board_id": parent, "pool": None, "grid": None, "geom": None,
                            "regions": None})
        if self.measurements is not None:
            self.measurements.board(board_id, parent, state.depth, pool)
        return [NamoMove(o, g, q) for (o, g, q) in pool]

    # -- one physics attempt (mcts_utils.py:635-669) --
    def simulate(self, state: "NamoPushState", move: NamoMove):
        from namo.planners.opening.best_first_search import _record_jam_depth, _unmoved, make_action
        jk = (state.uid, move.obj, move.edge)
        jd = self._jam_at.get(jk)
        if self.prune_jam_depth and jd is not None and move.depth >= jd:
            # Our action library's depth axis: a deeper push from the same contact retraces the
            # jammed shallower one, so it fails the same way. No call, same outcome as a failure.
            return None
        if self.sim_budget is not None and self.calls >= self.sim_budget:
            raise MoreBudgetExhausted()
        env = self.env
        env.set_full_state(state.sim_state)
        before = env.get_observation() if self.dedupe_noop else None
        action = make_action(move.obj, move.goal)
        t = clock_start(self.timing)
        step = env.step(action)
        self.calls += 1
        clock_finish(self.timing, "t_sim", t)
        t = clock_start(self.timing)
        opened, frac = self._verifier(env)
        clock_finish(self.timing, "t_local_verify", t)
        info = step.info or {}
        board_id = self._board_of.get(state.uid)
        if self.measurements is not None:
            self.measurements.simulated(
                board_id=board_id,
                parent_board_id=self._board_of.get(state.parent_uid) if state.parent_uid else None,
                chain_depth=state.depth + 1, action=action, opened=opened, info=info)
        if board_id is not None:
            b = self.boards[board_id]
            b["tries"].append((len(b["tries"]) + 1, float(frac), bool(opened)))
            if not opened:
                b["k_failed"] += 1
        _record_jam_depth(self._jam_at, jk, move.depth, info)
        after = env.get_full_state()
        if opened:
            plan = list(state.plan) + [move]
            if self.first_open_call is None:
                self.first_open_call, self.first_open_plan = self.calls, plan
            if self.stop_on_open:
                raise MoreSolved(plan, after)
            return after, float(frac), True
        if self.dedupe_noop and _unmoved(before, env.get_observation(), move.obj):
            return None
        return after, float(frac), False


class NamoPushState:
    """PushState (mcts/push.py:22-153) over our simulator.

    level / max_level are MORE's per-search depth (root is 0). `base_depth` is how many pushes the
    episode has already executed before this search's root, so `depth = base_depth + level` is the
    push count from s0, which is what the action library and the boards want.
    """

    def __init__(self, uid, sim_state, frac, opened, level, ctx: MoreContext, max_level,
                 max_q=1.0, plan: Sequence[NamoMove] = (), parent_uid=None, base_depth=0):
        self.uid, self.sim_state, self.frac, self.opened = uid, sim_state, float(frac), bool(opened)
        self.level, self.ctx, self.max_level, self.max_q = level, ctx, max_level, max_q
        self.plan, self.parent_uid, self.base_depth = tuple(plan), parent_uid, base_depth

    @property
    def depth(self):
        return self.base_depth + self.level

    @property
    def push_result(self):
        """mcts/push.py:60-69 with the reachable goal-sample fraction for MORE's grasp q.

        MORE: 0.2*clamp(q, 0, 1), plus 1 when q > GRASP_Q_PUSH_THRESHOLD. Our "q clears the bar" is
        the verifier's opening, so push_result >= 1 exactly when opened (0.2*frac <= 0.2 < 1), which
        is what both search.py early-stop tests compare against max_q = 1.0.
        """
        r = 0.2 * min(max(self.frac, 0.0), 1.0)
        return r + 1.0 if self.opened else r

    def is_push_over(self):
        """mcts/push.py:71-83: max level, the opening (their q > max_q), or no legal action left."""
        if self.level == self.max_level or self.opened:
            return True
        moves = self.ctx.move_recorder.get(self.uid)
        return moves is not None and len(moves) == 0

    def move(self, move: NamoMove, is_consecutive_move=False):
        """mcts/push.py:85-131. `is_consecutive_move` is MORE's skip-the-restore shortcut; we always restore."""
        key = self.uid + str(move)
        rec = self.ctx.simulation_recorder
        if key not in rec:
            res = self.ctx.simulate(self, move)
            if res is None:
                return None
            rec[key] = res
            in_recorder = False
        else:
            in_recorder = True
        sim_state, frac, opened = rec[key]
        child = NamoPushState(f"{self.uid}.{self.level}-{move}", sim_state, frac, opened,
                              self.level + 1, self.ctx, self.max_level, self.max_q,
                              plan=self.plan + (move,), parent_uid=self.uid, base_depth=self.base_depth)
        return child, None, None, in_recorder

    def get_actions(self, color_image=None, mask_image=None):
        """mcts/push.py:133-146; the move list is cached per uid and shared, as MORE's is."""
        if self.uid not in self.ctx.move_recorder:
            self.ctx.move_recorder[self.uid] = self.ctx.actions_for(self)
        return self.ctx.move_recorder[self.uid]

    def remove_action(self, move: NamoMove):
        """mcts/push.py:148-153."""
        moves = self.ctx.move_recorder.get(self.uid)
        if moves is not None and move in moves:
            moves.remove(move)


# ------------------------------------------------------------------------------------------------
# Shared plumbing
# ------------------------------------------------------------------------------------------------
def make_verifier(is_open, region_samples):
    """(opened, frac) from one state: the verifier's boolean and the reachable goal-sample fraction."""
    pts = [tuple(p[:2]) for p in (region_samples or [])]

    def verify(env):
        if not pts:
            return bool(is_open(env)), 0.0
        rc, _first = env.count_reachable_points(pts)
        return bool(is_open(env)), rc / float(len(pts))
    return verify


def make_action_source(planner, env, goal, xml, hmax, prior, agg, rng, restrict_obj, raw,
                       region_samples, timing):
    """Legal pushes at a state: the same `candidates()` pool best-first searches."""
    from namo.planners.opening.best_first_search import candidates

    def source(state: NamoPushState):
        pool, _V, _grid = candidates(planner, env, goal, xml, state.sim_state,
                                     max(1, hmax - state.depth), prior, agg, rng,
                                     restrict_obj=restrict_obj, raw=raw,
                                     region_samples=region_samples, timing=timing)
        return pool
    return source


def walk(root):
    """Every node of a MORE tree, breadth first (the order of save_mcts_data, mcts_main.py:183-207)."""
    out, queue = [], [root]
    while queue:
        node = queue.pop(0)
        out.append(node)
        queue.extend(node.children)
    return out


def tree_rows(root, search_index=0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """save_mcts_data (mcts_main.py:180-212): one row per child of every node with children.

    label = max(child.q) (:193), num_visits = child.n (:200). The parent's simulator state is
    returned separately, keyed like the rows, so pictures render in a later pass. Every child of
    MORE's collection tree was simulated (mcts/nodes.py:47-63), so every row is physics evidence.
    """
    rows, parents = [], {}
    for node in walk(root):
        if not node.children:
            continue
        st = node.state
        key = f"{search_index}:{st.uid}"
        parents[key] = st
        for c in node.children:
            m = c.prev_move
            rows.append({"key": key, "search": search_index, "uid": st.uid, "depth": st.depth,
                         "obj": m.obj, "edge": m.edge, "push_depth": m.depth,
                         "label": float(max(c.q)), "num_visits": int(c.n),
                         "child_opened": bool(c.state.opened) if c.state is not None else False})
    return rows, parents


def run_search(search, root, iterations, eval_mode, more):
    """search.best_action, guarded for one failure mode of MORE's own code.

    If every root action fails, MORE's tree ends with no root child and best_child_top
    (mcts/nodes.py:129-132, mcts_network/nodes.py:141-143) takes np.argmax of an empty list and
    raises. MORE never hits it on its benchmark; we report "no push possible" instead.
    """
    try:
        return search.best_action(iterations, more.constants.MCTS_EARLY_ROLLOUTS, eval=eval_mode)
    except ValueError:
        if root.children:
            raise
        return None


def _root_state(ctx, s0, hmax_level, base_depth, verifier, env):
    env.set_full_state(s0)
    opened, frac = verifier(env)
    return NamoPushState("root", s0, frac, opened, 0, ctx, hmax_level, base_depth=base_depth)


# ------------------------------------------------------------------------------------------------
# Collection: MORE's System 2 (mcts_main.py), unchanged semantics
# ------------------------------------------------------------------------------------------------
def collect_episode(env, s0, *, hmax, iterations, action_source, verifier, dedupe_noop=True,
                    prune_jam_depth=True, rng_seed=None, timing=None, measurements=None,
                    more=None):
    """One MORE collection episode: search, save the tree, execute best_child_top, repeat.

    mcts_main.py:422-562 with its knobs as MORE runs them: `test` False, so no early stop and the
    whole tree is saved (:354, :483, :551); best_action(300) per search (paper IV-B, constants.py:42
    comment); caches reset after every executed push (:561). The loop ends on an opening or when
    the episode has executed hmax pushes. MORE's own loop runs until the grasp; hmax is our
    horizon substitution, so each search's max_level is the pushes still left.

    The executed push is the tree's own cached transition of best_child_top, so executing it costs
    no extra call (MORE executes it on its separate real env, mcts_main.py:507-519). Returns a dict
    with the true call count per search and per episode.
    """
    import numpy as np
    import time
    more = more or load_more()
    if rng_seed is not None:
        np.random.seed(rng_seed)            # mcts/nodes.py:135 rollouts draw from numpy's global RNG
    ctx = MoreContext(env, action_source, verifier, dedupe_noop=dedupe_noop,
                      prune_jam_depth=prune_jam_depth, timing=timing, measurements=measurements)
    state, executed, searches, rows, parents = s0, [], [], [], {}
    solved, t0 = False, time.time()
    for step in range(hmax):
        ctx.reset(step)
        root_state = _root_state(ctx, state, hmax - step, step, verifier, env)
        root_state.plan = tuple(executed)
        root = more.UctNode(root_state)
        calls0, ts = ctx.calls, time.time()
        search = more.UctSearch(root)
        best = None
        root_has_actions = bool(root_state.get_actions())
        if root_has_actions:
            best = run_search(search, root, iterations, False, more)
        nodes = walk(root)
        r, p = tree_rows(root, step)
        rows.extend(r)
        parents.update(p)
        searches.append({"search": step, "calls": ctx.calls - calls0, "iterations": iterations if root_has_actions else 0,
                         "tree_nodes": len(nodes), "rows": len(r), "seconds": round(time.time() - ts, 3),
                         "best": list(best.prev_move.key) if best is not None else None})
        if best is None or not root.children:
            break
        child = best.state
        executed.append(best.prev_move)
        state = child.sim_state
        if child.opened:
            solved = True
            break
    return {"solved": solved, "calls": ctx.calls, "first_open_call": ctx.first_open_call,
            "plan": [m.key for m in executed], "final_state": state, "searches": searches,
            "rows": rows, "parents": parents, "legal": dict(ctx.legal),
            "seconds": round(time.time() - t0, 3)}


# ------------------------------------------------------------------------------------------------
# Deploy: MORE's System 1+2 (more_main.py) in two stopping modes
# ------------------------------------------------------------------------------------------------
STOP_MODES = ("protocol", "native")


def _unexpanded_left(root) -> bool:
    """Can the tree still produce a call? Only a non-terminal node's unsimulated child can.

    mcts_network/search.py:27-40 descends only through non-terminal nodes and simulates a child
    only when it is selected with state None; a terminal node's rollout never moves.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_terminal_node():
            continue
        for c in node.children:
            if c.state is None:
                return True
            stack.append(c)
    return False


def solve_scene_more(planner, env, goal, xml, s0, hmax, sim_budget, prior, agg, combine, rng,
                     restrict_obj=None, is_open=lambda e: e.is_robot_goal_reachable(), raw=True,
                     region_samples=None, dedupe_noop=True, prune_jam_depth=True, timing=None,
                     measurements=None, solution_out=None, stop_mode="protocol", num_iter=50,
                     chunk=500, stall_iters=5000, max_iters=1_000_000, more=None, **_ignored):
    """MORE's deploy search (mcts_network/), same contract as best_first_search.solve_scene.

    stop_mode="protocol": one search, stopped at the first verified opening anywhere, rollouts
      included, under the simulator-call cap. MORE's own early-stop rule is off (eval=False) because
      ours replaces it. The search also ends when no non-terminal node has an unsimulated child left
      ("exhausted"), or when `stall_iters` iterations in a row neither call the simulator nor expand
      a node ("stalled": MORE's greedy selection can starve a child whose clamped prior is 0).
    stop_mode="native": more_main.py:393-525. Each decision runs best_action(num_iter, -1, eval=True)
      (MORE's stop rule, mcts_network/search.py:16-23), executes best_child_top, resets the caches and
      searches again from the new state; success is an executed push that opens. Every search's calls
      are charged, and so is executing a child the tree never simulated. Ends after hmax executed
      pushes ("horizon") or at the call cap ("budget").

    Returns (solved, sims, plan_len|None, boards, end).
    """
    if stop_mode not in STOP_MODES:
        raise ValueError(f"stop_mode must be one of {STOP_MODES}, got {stop_mode!r}")
    more = more or load_more()
    tm = timing
    if tm is not None:
        for key in ("t_score", "t_sim", "n_score"):
            tm.setdefault(key, 0)
    t_wall = clock_start(tm)
    verifier = make_verifier(is_open, region_samples)
    source = make_action_source(planner, env, goal, xml, hmax, prior, agg, rng, restrict_obj, raw,
                                region_samples, tm)
    ctx = MoreContext(env, source, verifier, dedupe_noop=dedupe_noop, prune_jam_depth=prune_jam_depth,
                      sim_budget=sim_budget, stop_on_open=(stop_mode == "protocol"), timing=tm,
                      measurements=measurements)
    solved, plan, final, end = False, None, None, "exhausted"
    try:
        if stop_mode == "protocol":
            root = more.NetNode(_root_state(ctx, s0, hmax, 0, verifier, env))
            search = more.NetSearch(root)              # pre_expands the root: one scoring pass
            done, quiet = 0, 0
            while root.children:
                calls0, nodes0 = ctx.calls, len(walk(root))
                if run_search(search, root, chunk, False, more) is None:
                    end = "exhausted"
                    break
                done += chunk
                if not _unexpanded_left(root):
                    end = "exhausted"
                    break
                quiet = quiet + chunk if (ctx.calls == calls0 and len(walk(root)) == nodes0) else 0
                if quiet >= stall_iters:
                    end = "stalled"
                    break
                if done >= max_iters:
                    end = "iter_cap"
                    break
        else:
            state, executed = s0, []
            end = "horizon"
            for step in range(hmax):
                ctx.reset(step)
                root_state = _root_state(ctx, state, hmax - step, step, verifier, env)
                root_state.plan = tuple(executed)
                root = more.NetNode(root_state)
                search = more.NetSearch(root)
                best = run_search(search, root, num_iter, True, more) if root.children else None
                if best is None:
                    end = "exhausted"
                    break
                child = best.state if best.state is not None else root_state.move(best.prev_move)
                executed.append(best.prev_move)
                if child is None:
                    continue                        # the push failed: MORE re-plans from where it is
                child = child[0] if isinstance(child, tuple) else child
                state = child.sim_state
                if child.opened:
                    solved, plan, final, end = True, list(executed), state, "solved"
                    break
    except MoreSolved as hit:
        solved, plan, final, end = True, list(hit.plan), hit.state, "solved"
    except MoreBudgetExhausted:
        end = "budget"
    sims = ctx.calls
    if solution_out is not None:
        solution_out["total_sims"] = ctx.calls
        if solved:
            solution_out["plan"] = [(m.obj, m.goal) for m in plan]
            solution_out["state"] = final
    clock_finish(tm, "t_wall", t_wall)
    return solved, sims, (len(plan) if solved else None), ctx.boards, end
