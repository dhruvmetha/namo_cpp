"""End-to-end control flow of the MORE adaptation, on a fake simulator.

The physics is faked so the accounting can be checked exactly. These are the
properties an external-baseline comparison stands or falls on: one simulator call
per physics attempt and none for a cache hit, a stop at the first verified opening
wherever it happens, and a search that terminates instead of spinning once every
action inside hmax has been tried.
"""

import pytest

from namo.planners.opening import more_mcts as mm


class _Goal:
    def __init__(self, edge_idx, depth):
        self.x = self.y = self.theta = 0.0
        self.edge_idx = int(edge_idx)
        self.depth = int(depth)


class _Step:
    def __init__(self, info=None):
        self.info = info or {}


class FakeEnv:
    """Deterministic toy dynamics: a state is the tuple of pushes applied so far.

    `opens` names the push chains that open the goal. `noops` names pushes that move
    nothing. `reach` gives the reachable-sample fraction for a state, defaulting low.
    """

    def __init__(self, opens=(), noops=(), reach=None, fails=()):
        self.state = ()
        self.opens = {tuple(c) for c in opens}
        self.noops = {tuple(n) for n in noops}
        self.reach = reach or {}
        self.fails = {tuple(f) for f in fails}
        self.steps = 0

    def set_full_state(self, s):
        self.state = s

    def get_full_state(self):
        return self.state

    def get_observation(self):
        moved = len([p for p in self.state])
        return {"obj_pose": [moved, 0.0, 0.0], "robot_pose": [moved, 0.0, 0.0]}

    def step(self, action):
        self.steps += 1
        key = (action.object_id, int(action.edge_idx), int(action.depth))
        if (self.state, key) in self.noops:
            return _Step({"failure_reason": ""})
        self.state = self.state + (key,)
        return _Step({"failure_reason": "stuck"} if self.state in self.fails else {})

    def count_reachable_points(self, pts):
        if self.state in self.opens:
            return int(0.5 * len(pts)), 0
        return int(self.reach.get(self.state, 0.02) * len(pts)), 0

    def is_robot_goal_reachable(self):
        return self.state in self.opens


def _pool_factory(edges=4, depths=2):
    """A fixed action library, scored so edge 0 ranks highest and edge n-1 lowest."""
    def _candidates(planner, env, goal, xml, state, h, prior, agg, rng, **kw):
        pool = [("obj", _Goal(e, d), float(10 - e - 0.1 * d))
                for e in range(edges) for d in range(depths)]
        return pool, 0.0, None
    return _candidates


def _run(env, monkeypatch, *, hmax=2, sim_budget=100, edges=4, depths=2, **kw):
    monkeypatch.setattr(mm, "candidates", _pool_factory(edges, depths))
    solution = {}
    out = mm.solve_scene_mcts(
        planner=None, env=env, goal=None, xml="x.xml", s0=(), hmax=hmax,
        sim_budget=sim_budget, prior="model", agg="mean5", combine="q", rng=None,
        region_samples=[(0.0, 0.0, 0.0)] * 100,
        is_open=lambda e: e.state in e.opens,
        solution_out=solution, **kw)
    return out, solution


def test_stops_at_the_first_verified_opening_and_reports_the_chain(monkeypatch):
    """Edge 0 depth 0 is the top-ranked action and it opens, so this costs one attempt."""
    env = FakeEnv(opens=[(("obj", 0, 0),)])
    (solved, sims, plen, _boards, end), solution = _run(env, monkeypatch)

    assert solved is True and end == "solved"
    assert sims == 1 and env.steps == 1 and plen == 1
    assert [(o, g.edge_idx, g.depth) for o, g in solution["plan"]] == [("obj", 0, 0)]


def test_two_push_chain_is_found_and_the_plan_has_both_pushes(monkeypatch):
    """Only the setup-then-finish pair opens, so the search must go a level deep."""
    env = FakeEnv(opens=[(("obj", 0, 0), ("obj", 1, 0))])
    (solved, sims, plen, _b, end), solution = _run(env, monkeypatch)

    assert solved is True and end == "solved" and plen == 2
    assert [g.edge_idx for _o, g in solution["plan"]] == [0, 1]
    assert sims == env.steps


def test_reported_sims_equal_actual_step_calls(monkeypatch):
    """The comparison axis is physics attempts, so the count we report must be the real one.

    Checked on a run that ends at the budget and on one that solves, because the two
    exits leave the loop by different paths.
    """
    starved = FakeEnv(opens=[(("obj", 3, 1), ("obj", 3, 1))])
    (solved, sims, _p, _b, end), _sol = _run(starved, monkeypatch, sim_budget=20)
    assert solved is False and end == "budget"
    assert sims == starved.steps == 20

    winner = FakeEnv(opens=[(("obj", 0, 0), ("obj", 1, 0))])
    (solved, sims, _p, _b, end), _sol = _run(winner, monkeypatch, sim_budget=200)
    assert solved is True and sims == winner.steps


def test_a_transition_is_never_simulated_twice(monkeypatch):
    """MORE's simulation_recorder: revisiting a branch replays from cache, free.

    Without this the rollout phase would re-push its way down the tree on every
    iteration and MORE's attempt count would balloon for reasons the paper never
    intended, which would make the baseline look worse than the published method is.
    """
    env = FakeEnv()
    seen = []
    real_step = FakeEnv.step

    def _record(self, action):
        seen.append((self.state, (action.object_id, int(action.edge_idx), int(action.depth))))
        return real_step(self, action)

    monkeypatch.setattr(FakeEnv, "step", _record)
    (_solved, sims, _p, _b, _end), _sol = _run(env, monkeypatch, sim_budget=10_000,
                                               edges=3, depths=2)

    assert len(seen) == len(set(seen)), "the same (state, action) was simulated twice"
    assert sims == len(seen)


def test_search_terminates_when_no_chain_opens(monkeypatch):
    """Nothing opens. With hmax=2 and 8 actions the tree is finite, so this must end."""
    env = FakeEnv()
    (solved, sims, plen, boards, end), _sol = _run(env, monkeypatch, sim_budget=10_000)

    assert solved is False and plen is None
    assert end == "exhausted"
    assert sims == env.steps and sims < 10_000
    assert len(boards) >= 1


def test_budget_exhaustion_is_reported_as_budget(monkeypatch):
    env = FakeEnv()
    (solved, sims, _p, _b, end), _sol = _run(env, monkeypatch, sim_budget=5,
                                             edges=30, depths=5)

    assert solved is False and end == "budget"
    assert sims == 5 and env.steps == 5


def test_a_noop_push_costs_its_attempt_then_leaves_the_tree(monkeypatch):
    """dedupe_noop parity with best-first: the attempt is paid for, the child is dropped."""
    noop = ((), ("obj", 0, 0))
    env = FakeEnv(opens=[(("obj", 1, 0),)], noops=[noop])
    # depths=1 so the pool orders purely by edge: the no-op is tried first, the winner second.
    (solved, sims, plen, _b, _end), solution = _run(env, monkeypatch, depths=1)

    assert solved is True and plen == 1
    assert [g.edge_idx for _o, g in solution["plan"]] == [1]
    assert sims == 2 and env.steps == 2           # the no-op, then the winner


def test_the_ranker_ordering_decides_which_push_is_tried_first(monkeypatch):
    """With no sampled evidence the prior alone orders the root, so the top score goes first."""
    env = FakeEnv()
    seen = []
    base = _pool_factory(4, 1)

    def _spy(*a, **k):
        pool, v, g = base(*a, **k)
        return pool, v, g
    monkeypatch.setattr(mm, "candidates", _spy)

    real_step = FakeEnv.step

    def _record(self, action):
        seen.append(int(action.edge_idx))
        return real_step(self, action)
    monkeypatch.setattr(FakeEnv, "step", _record)

    mm.solve_scene_mcts(planner=None, env=env, goal=None, xml="x.xml", s0=(), hmax=1,
                        sim_budget=4, prior="model", agg="mean5", combine="q", rng=None,
                        region_samples=[(0.0, 0.0, 0.0)] * 100,
                        is_open=lambda e: e.state in e.opens)

    assert seen[0] == 0, "highest-scored action must be attempted first"


def test_training_records_cover_the_tried_actions(monkeypatch):
    """Stage-2 supervision: rows carry action identity, a label and a visit count."""
    env = FakeEnv()
    rows = []
    monkeypatch.setattr(mm, "candidates", _pool_factory(3, 1))
    mm.solve_scene_mcts(planner=None, env=env, goal=None, xml="x.xml", s0=(), hmax=2,
                        sim_budget=200, prior="model", agg="mean5", combine="q", rng=None,
                        region_samples=[(0.0, 0.0, 0.0)] * 100,
                        is_open=lambda e: e.state in e.opens, record_out=rows)

    assert rows, "a completed search must leave training evidence"
    assert {"obj", "edge", "push_depth", "label", "num_visits"} <= set(rows[0])
    assert all(r["num_visits"] >= 1 for r in rows)
    assert all(0.0 <= r["label"] <= 1.2 for r in rows)
