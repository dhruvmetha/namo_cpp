"""The plain UCT collector, checked against MORE's `mcts/` (not `mcts_network/`).

This is the search that generates MORE's training data, and the paper is explicit
that it runs with no network: `mcts_main.py:480` builds the root state without a push
net. So this variant has to work from a cold start with prior="uniform", or MORE
cannot bootstrap and the whole self-supervised claim collapses.

Everything is pinned to revision 70402c001c30908e7a70b93f8de3c31abdd26fdb.
"""

import math

import pytest

from namo.planners.opening import more_mcts as mm
from test_more_mcts_search import FakeEnv, _pool_factory


class _Goal:
    def __init__(self, edge_idx, depth):
        self.x = self.y = self.theta = 0.0
        self.edge_idx = int(edge_idx)
        self.depth = int(depth)


def _child(results, n, prior=0.0, mode="uct"):
    c = mm._Node(None, "obj", _Goal(0, 0), prior, None, 1, "u", mode=mode)
    c.results = list(results)
    c.n = n
    return c


def test_uct_node_starts_cold():
    """mcts/nodes.py:18 is n=0, q=[]. The guided node is n=1, q=[0]. Not the same node."""
    cold = mm._Node(None, "o", _Goal(0, 0), 0.0, None, 1, "u", mode="uct")
    warm = mm._Node(None, "o", _Goal(0, 0), 0.0, None, 1, "u", mode="guided")
    assert cold.n == 0 and cold.results == []
    assert warm.n == 1 and warm.results == [0.0]


def test_uct_selection_matches_the_published_formula():
    """mcts/nodes.py:121 = sum(sorted(q)[-10:]) / min(n, 10) + sqrt(2)*sqrt(2*ln(parent.n)/n)."""
    parent = mm._Node(None, None, None, 0.0, None, 0, "r", mode="uct")
    parent.n = 9
    a = _child([1.0, 0.0, 0.0, 0.0], n=4)      # good once, tried a lot
    b = _child([0.4], n=1)                      # mediocre, barely tried
    parent.children = [a, b]

    def w(c):
        return (sum(sorted(c.results)[-mm.MCTS_TOP_UCT:]) / min(c.n, mm.MCTS_TOP_UCT)
                + mm.MCTS_UCT_RATIO * math.sqrt(2 * math.log(parent.n) / c.n))

    assert w(a) == pytest.approx(0.25 + math.sqrt(2) * math.sqrt(2 * math.log(9) / 4))
    assert w(b) == pytest.approx(0.40 + math.sqrt(2) * math.sqrt(2 * math.log(9) / 1))
    chosen = parent.best_child("uct")
    assert chosen is (a if w(a) > w(b) else b)
    assert chosen is b, "the barely-tried child should win on the exploration term"


def test_uct_divisor_is_a_capped_mean_not_the_visit_count():
    """The two searches divide differently, and swapping them changes the ordering."""
    parent = mm._Node(None, None, None, 0.0, None, 0, "r", mode="uct")
    parent.n = 4
    c = _child([1.0] * 20, n=20)
    parent.children = [c]
    capped = sum(sorted(c.results)[-mm.MCTS_TOP_UCT:]) / min(c.n, mm.MCTS_TOP_UCT)
    assert capped == pytest.approx(1.0)                      # mean of the best ten
    assert sum(sorted(c.results)[-mm.MCTS_TOP:]) / c.n == pytest.approx(0.15)   # guided


def test_uct_skips_children_with_no_backup_yet():
    """A freshly expanded UCT child has n=0, so its score is undefined and it is skipped."""
    parent = mm._Node(None, None, None, 0.0, None, 0, "r", mode="uct")
    parent.n = 3
    fresh = _child([], n=0)
    seen = _child([0.5], n=1)
    parent.children = [fresh, seen]
    assert parent.best_child("uct") is seen


def _run_uct(env, monkeypatch, *, hmax=2, sim_budget=200, edges=4, depths=2, seed=3, **kw):
    import random
    monkeypatch.setattr(mm, "candidates", _pool_factory(edges, depths))
    solution = {}
    out = mm.solve_scene_mcts(
        planner=None, env=env, goal=None, xml="x.xml", s0=(), hmax=hmax,
        sim_budget=sim_budget, prior="uniform", agg="mean5", combine="q",
        rng=random.Random(seed), region_samples=[(0.0, 0.0, 0.0)] * 100,
        is_open=lambda e: e.state in e.opens, solution_out=solution, mode="uct", **kw)
    return out, solution


def test_uct_collects_with_no_model_at_all(monkeypatch):
    """prior='uniform' means candidates() never touches a network. Cold start must work."""
    env = FakeEnv(opens=[(("obj", 2, 0), ("obj", 1, 0))])
    (solved, sims, plen, boards, end), solution = _run_uct(env, monkeypatch)

    assert solved is True and end == "solved" and plen == 2
    assert sims == env.steps
    assert [g.edge_idx for _o, g in solution["plan"]] == [2, 1]


def test_uct_sim_accounting_is_exact(monkeypatch):
    env = FakeEnv()
    (solved, sims, _p, _b, end), _s = _run_uct(env, monkeypatch, sim_budget=25, edges=30, depths=5)
    assert solved is False and end == "budget"
    assert sims == env.steps == 25


def test_uct_terminates_when_nothing_opens(monkeypatch):
    env = FakeEnv()
    (solved, sims, _p, _b, end), _s = _run_uct(env, monkeypatch, sim_budget=10_000)
    assert solved is False and end == "exhausted"
    assert sims == env.steps and sims < 10_000


def test_a_rollout_never_strands_a_tree_child(monkeypatch):
    """Regression: rollouts must walk states, not mark tree children as expanded.

    MORE's rollout moves through PushStates and creates no PushSearchNode. An earlier
    version of this module let the rollout set `state` on real tree children. Under
    uct that leaves them at n=0, where best_child skips them forever, so the search
    silently deletes candidates it never actually ruled out. The invariant: every
    child carrying a state has been backed up at least once.
    """
    env = FakeEnv()
    rows = []
    _out, _s = _run_uct(env, monkeypatch, sim_budget=400, record_out=rows)

    assert rows, "the search must leave training evidence"
    stranded = [r for r in rows if r["expanded"] and r["num_visits"] == 0]
    assert not stranded, f"{len(stranded)} expanded children were never backed up"


def test_uct_rollouts_are_random_not_greedy(monkeypatch):
    """mcts/nodes.py:134 picks uniformly. Different seeds must explore differently."""
    seen = set()
    for seed in (1, 2, 3, 4, 5):
        env = FakeEnv()
        order = []
        real = FakeEnv.step

        def rec(self, action, _o=order):
            _o.append(int(action.edge_idx))
            return real(self, action)

        monkeypatch.setattr(FakeEnv, "step", rec)
        _run_uct(env, monkeypatch, sim_budget=12, edges=8, depths=1, seed=seed)
        seen.add(tuple(order))
        monkeypatch.undo()
    assert len(seen) > 1, "a uniform-random rollout policy must vary with the seed"
