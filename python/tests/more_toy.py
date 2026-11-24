"""Toy simulator, reference PushState and an old/new driver for the MORE tests.

FakeEnv: a state is the tuple of pushes applied so far. `opens` names the push chains that open
the goal (50/100 goal samples reachable), `reach` gives other states' reachable fraction, `noops`
are (state, action) pairs that move nothing (MORE's failed move), `fails` are states reached with
a controller failure_reason.

RefPushState / RefHelper: MORE's mcts/push.py PushState and MCTSHelper transliterated line for
line onto FakeEnv, with only the approved substitutions (fraction for grasp q, opened for
"q > max_q"). Driving MORE's unmodified tree classes with these is the reference the adapter is
checked against.

The driver functions run a collection on whichever MORE implementation is importable: the
vendored adapter (namo.planners.opening.more_vendor) or the 13763e54 port (more_mcts), so the
regression tests can show a failure on the old code and a pass on the new.
"""
from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager

PTS = [(0.0, 0.0, 0.0)] * 100


class Goal:
    def __init__(self, edge_idx, depth):
        self.x = self.y = self.theta = 0.0
        self.edge_idx = int(edge_idx)
        self.depth = int(depth)


class _Step:
    def __init__(self, info=None):
        self.info = info or {}


class FakeEnv:
    def __init__(self, opens=(), noops=(), reach=None, fails=()):
        self.state = ()
        self.opens = {tuple(c) for c in opens}
        self.noops = {tuple(n) for n in noops}
        self.reach = reach or {}
        self.fails = {tuple(f) for f in fails}
        self.steps = 0
        self.log = []

    def set_full_state(self, s):
        self.state = s

    def get_full_state(self):
        return self.state

    def get_observation(self):
        moved = len(self.state)
        return {"obj_pose": [moved, 0.0, 0.0], "robot_pose": [moved, 0.0, 0.0]}

    def step(self, action):
        self.steps += 1
        key = (action.object_id, int(action.edge_idx), int(action.depth))
        self.log.append((self.state, key))
        if (self.state, key) in self.noops:
            return _Step({"failure_reason": ""})
        self.state = self.state + (key,)
        return _Step({"failure_reason": "stuck"} if self.state in self.fails else {})

    def count_reachable_points(self, pts):
        if self.state in self.opens:
            return int(0.5 * len(pts)), 0
        return int(round(self.reach.get(self.state, 0.02) * len(pts))), 0

    def is_robot_goal_reachable(self):
        return self.state in self.opens


def goal_open_pts(env, pts, frac=0.2):
    """Verbatim rule of scripts/sandbox/eval_m3.py:45-52."""
    if not pts:
        return False
    rc, _first = env.count_reachable_points(pts)
    return rc >= max(1, math.ceil(frac * len(pts)))


def stable_prior(state, key):
    """A deterministic pseudo-random prior in [-0.2, 1.4], so the clamp in best_child matters."""
    h = hashlib.sha1(repr((state, key)).encode()).hexdigest()
    return -0.2 + 1.6 * int(h[:8], 16) / 0xFFFFFFFF


def pool_factory(edges=4, depths=2, prior=None, reverse=True):
    """candidates() stand-in: every (edge, depth) of one object, reversed by default so the adapter's
    own ordering is exercised. The 13763e54 port keeps the pool order, so its driver passes
    reverse=False to reproduce the audit's scenes exactly."""
    def cands(planner, env, goal, xml, state, h, prior_name, agg, rng, **kw):
        pool = [("obj", Goal(e, d), (prior or (lambda s, k: 10 - k[1] - 0.1 * k[2]))(state, ("obj", e, d)))
                for e in range(edges) for d in range(depths)]
        return (pool[::-1] if reverse else pool), 0.0, None
    return cands


def action_source(edges, depths, prior=None):
    """Adapter action source over the toy: (obj, Goal, q) for every action, any order."""
    def source(state):
        return [("obj", Goal(e, d), (prior or (lambda s, k: 0.0))(state.sim_state, ("obj", e, d)))
                for e in range(edges) for d in range(depths)][::-1]
    return source


def verifier(is_open=None):
    is_open = is_open or (lambda e: e.state in e.opens)

    def verify(env):
        rc, _ = env.count_reachable_points(PTS)
        return bool(is_open(env)), rc / float(len(PTS))
    return verify


# ------------------------------------------------------------------------------------------------
# Reference: mcts/push.py transliterated onto FakeEnv
# ------------------------------------------------------------------------------------------------
class RefMove:
    """PushMove: identity and str() from the action; q_value is the prior (mcts_network/push.py)."""

    def __init__(self, key, q_value=0.0):
        self.key, self.q_value = key, q_value
        self.obj, self.edge, self.depth = key

    def __str__(self):
        return f"{self.obj}_{self.edge}_{self.depth}"

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)


class _Act:
    def __init__(self, key):
        self.object_id, self.edge_idx, self.depth = key


class RefHelper:
    """MCTSHelper (mcts_utils.py:39-67, :635-669) for FakeEnv."""

    def __init__(self, env, edges, depths, verify, prior=None):
        self.env, self.edges, self.depths, self.verify = env, edges, depths, verify
        self.prior = prior or (lambda s, k: 0.0)
        self.reset()

    def reset(self):
        self.move_recorder = {}
        self.simulation_recorder = {}

    def sample_actions(self, object_states):
        keys = sorted(("obj", e, d) for e in range(self.edges) for d in range(self.depths))
        return [RefMove(k, self.prior(object_states, k)) for k in keys]

    def simulate(self, object_states, key):
        env = self.env
        env.set_full_state(object_states)
        before = env.get_observation()
        env.step(_Act(key))
        if env.get_observation() == before:            # nothing moved: MORE's failed push
            return None
        opened, frac = self.verify(env)
        return env.get_full_state(), frac, opened


class RefPushState:
    """mcts/push.py:22-153, line for line, with frac for q_value and `opened` for q > max_q."""

    def __init__(self, uid, object_states, q_value, opened, level, mcts_helper, max_q=1.0, max_level=2):
        self.uid = uid
        self.object_states = object_states
        self.q_value = q_value
        self.opened = opened
        self.level = level
        self.mcts_helper = mcts_helper
        self.max_q = max_q
        self.max_level = max_level

    @property
    def push_result(self):
        result = self.q_value
        result = min(result, 1)
        result = max(result, 0)
        result *= 0.2
        if self.opened:
            result += 1
        return result

    def is_push_over(self):
        if self.level == self.max_level or self.opened:
            return True
        if self.uid in self.mcts_helper.move_recorder:
            if len(self.mcts_helper.move_recorder[self.uid]) == 0:
                return True
        return False

    def _move_result(self, move, is_consecutive_move=False):
        key = self.uid + str(move)
        if key not in self.mcts_helper.simulation_recorder:
            result = self.mcts_helper.simulate(self.object_states, move.key)
            if result is None:
                return None
            object_states, new_q, opened = result
            self.mcts_helper.simulation_recorder[key] = object_states, new_q, opened
            in_recorder = False
        else:
            object_states, new_q, opened = self.mcts_helper.simulation_recorder[key]
            in_recorder = True
        return object_states, new_q, opened, in_recorder

    def move(self, move, is_consecutive_move=False):
        result = self._move_result(move, is_consecutive_move=is_consecutive_move)
        if result is None:
            return None
        object_states, new_q, opened, in_recorder = result
        return (RefPushState(f"{self.uid}.{self.level}-{move}", object_states, new_q, opened,
                             self.level + 1, self.mcts_helper, max_q=self.max_q, max_level=self.max_level),
                None, None, in_recorder)

    def get_actions(self, color_image=None, mask_image=None):
        key = self.uid
        if key not in self.mcts_helper.move_recorder:
            self.mcts_helper.move_recorder[key] = self.mcts_helper.sample_actions(self.object_states)
        return self.mcts_helper.move_recorder[key]

    def remove_action(self, move):
        key = self.uid
        if key in self.mcts_helper.move_recorder:
            moves = self.mcts_helper.move_recorder[key]
            if move in moves:
                moves.remove(move)


# ------------------------------------------------------------------------------------------------
# Old/new driver for the regression tests
# ------------------------------------------------------------------------------------------------
try:
    from namo.planners.opening import more_vendor as NEW
except ImportError:                     # the 13763e54 tree: only the port exists
    NEW = None
    from namo.planners.opening import more_mcts as OLD
API = "new" if NEW is not None else "old"


class FirstRng:
    def randrange(self, n):
        return 0

    def random(self):
        return 0.5


def parse_path(uid):
    """'root.0-obj_1_0.1-obj_0_0' (new) or 'r.0-obj_1_0' (old) -> (('obj', 1, 0), ('obj', 0, 0))."""
    out = []
    for tok in uid.split(".")[1:]:
        _lvl, act = tok.split("-", 1)
        obj, e, d = act.rsplit("_", 2)
        out.append((obj, int(e), int(d)))
    return tuple(out)


def collect(monkeypatch, env, *, edges, depths=1, iterations, calls, hmax=2, is_open=None):
    """One collection on the current implementation. rows = first search's rows, normalised.

    new: MORE's collection episode with `iterations` per search; old: the port's uct mode with a
    `calls` simulator budget. Rollouts take index 0 on both sides.
    """
    import numpy as np
    if API == "new":
        monkeypatch.setattr(np.random, "randint", lambda n: 0)
        res = NEW.collect_episode(env, (), hmax=hmax, iterations=iterations,
                                  action_source=action_source(edges, depths),
                                  verifier=verifier(is_open))
        rows = [{"path": parse_path(r["uid"]), "key": (r["obj"], r["edge"], r["push_depth"]),
                 "label": r["label"], "n": r["num_visits"]} for r in res["rows"] if r["search"] == 0]
        return {"rows": rows, "total_calls": res["calls"], "solved": res["solved"]}
    monkeypatch.setattr(OLD, "candidates", pool_factory(edges, depths, reverse=False))
    rows, sol = [], {}
    solved, sims, _plen, _b, _end = OLD.solve_scene_mcts(
        planner=None, env=env, goal=None, xml="x.xml", s0=(), hmax=hmax, sim_budget=calls,
        prior="uniform", agg="mean5", combine="q", rng=FirstRng(), region_samples=PTS,
        is_open=is_open or (lambda e: e.state in e.opens), solution_out=sol, record_out=rows,
        mode="uct", stop_on_open=False)
    rows = [{"path": parse_path(r["uid"]), "key": (r["obj"], r["edge"], r["push_depth"]),
             "label": r["label"], "n": r["num_visits"]} for r in rows]
    return {"rows": rows, "total_calls": sol.get("total_sims"), "solved": solved}


def row(rows, path, key):
    hit = [r for r in rows if r["path"] == tuple(path) and r["key"] == key]
    assert len(hit) == 1, (path, key, rows)
    return hit[0]


@contextmanager
def backups(monkeypatch):
    """Record (path, value) for every backpropagate call, on either implementation."""
    seen = []
    if API == "new":
        cls = NEW.load_more().UctNode
        real = cls.backpropagate

        def spy(self, result):
            seen.append((parse_path(self.state.uid), result))
            return real(self, result)
    else:
        cls = OLD._Node
        real = cls.backpropagate

        def spy(self, result):
            seen.append((parse_path(self.uid), result))
            return real(self, result)
    monkeypatch.setattr(cls, "backpropagate", spy)
    yield seen
