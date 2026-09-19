"""The MORE adaptation's tree mechanics, checked against the published originals.

Every assertion here is pinned to github.com/arc-l/more at revision
70402c001c30908e7a70b93f8de3c31abdd26fdb, the one the manuscript audit records. The
point is not that the numbers look sensible. It is that selection, backup and the
rollout policy compute what MORE's nodes.py computes, so a later result can be
called a MORE-inspired adaptation without hand-waving.

The environment is faked. These tests are about the search, not the physics.
"""

import math

import pytest

from namo.planners.opening import more_mcts as mm


class _Goal:
    def __init__(self, edge_idx, depth):
        self.x = self.y = self.theta = 0.0
        self.edge_idx = edge_idx
        self.depth = depth


def _node(prior, results, n):
    node = mm._Node(None, "obj", _Goal(0, 0), prior, None, 1, "u")
    node.results = list(results)
    node.n = n
    return node


def test_best_child_matches_more_weight_formula():
    """nodes.py best_child: (sum(sorted(c.q)[-3:]) + clamp(c.nq, 0, 1.2)) / c.n."""
    parent = mm._Node(None, None, None, 0.0, None, 0, "r")
    parent.children = [
        _node(prior=0.1, results=[0.0, 1.0, 1.0, 1.0, 0.9], n=5),
        _node(prior=0.9, results=[0.0], n=1),
    ]
    chosen = parent.best_child()
    expected = [(sum(sorted(c.results)[-3:]) + max(0.0, min(c.prior, 1.2))) / c.n
                for c in parent.children]
    assert parent.children.index(chosen) == expected.index(max(expected))
    assert expected[0] == pytest.approx((1.0 + 1.0 + 1.0 + 0.1) / 5)
    assert expected[1] == pytest.approx((0.0 + 0.9) / 1)


def test_unexpanded_child_scores_exactly_its_prior():
    """A fresh child has n=1 and results=[0], so MORE's weight collapses to the prior.

    This is the load-bearing property of the whole comparison: with no sampled
    evidence yet, the network ordering IS the search ordering, which is what makes
    MORE a learned-ranking method rather than a blind tree search.
    """
    parent = mm._Node(None, None, None, 0.0, None, 0, "r")
    parent.children = [_node(prior=p, results=[0.0], n=1) for p in (0.2, 0.7, 0.5)]
    chosen = parent.best_child()
    assert parent.children.index(chosen) == 1 and chosen.prior == 0.7


def test_prior_clamp_ceiling_is_one_point_two():
    parent = mm._Node(None, None, None, 0.0, None, 0, "r")
    parent.children = [_node(prior=50.0, results=[0.0], n=1),
                       _node(prior=1.2, results=[0.0], n=1)]
    weights = [(sum(sorted(c.results)[-3:]) + max(0.0, min(c.prior, mm.PRIOR_CLAMP))) / c.n
               for c in parent.children]
    assert weights[0] == weights[1] == pytest.approx(1.2)


def test_backpropagate_discounts_by_half_per_level():
    """nodes.py backpropagate: parent gets result * MCTS_DISCOUNT, recursively."""
    root = mm._Node(None, None, None, 0.0, None, 0, "r")
    mid = mm._Node(None, "o", _Goal(1, 0), 0.0, root, 1, "r.0")
    leaf = mm._Node(None, "o", _Goal(2, 0), 0.0, mid, 2, "r.1")
    root.children, mid.children = [mid], [leaf]

    leaf.backpropagate(1.0)

    assert leaf.results == [0.0, 1.0] and leaf.n == 2
    assert mid.results == [0.0, 0.5] and mid.n == 2
    assert root.results == [0.0, 0.25] and root.n == 2
    assert mm.MCTS_DISCOUNT == 0.5


def test_push_result_keeps_more_reward_shape():
    """push.py push_result: 0.2 * clamp(q,0,1), plus 1 when the threshold is cleared.

    The opening bonus has to dominate the shaped term, or rollouts would chase
    partial reachability instead of actual openings.
    """
    assert mm._push_result(0.0, False) == pytest.approx(0.0)
    assert mm._push_result(1.0, False) == pytest.approx(0.2)
    assert mm._push_result(0.25, True) == pytest.approx(1.05)
    assert mm._push_result(0.0, True) - mm._push_result(1.0, False) > 0.7


def test_prior_rescale_preserves_ranker_order():
    """minmax exists so our ranker's out-of-range scores survive MORE's 1.2 clamp."""
    pool = [("o", _Goal(i, 0), q) for i, q in enumerate([8.0, 2.0, 5.0, -1.0])]
    scaled = mm._scale_priors(pool, "minmax")
    assert max(scaled) == pytest.approx(mm.PRIOR_CLAMP) and min(scaled) == pytest.approx(0.0)
    order_raw = sorted(range(4), key=lambda i: pool[i][2], reverse=True)
    order_scaled = sorted(range(4), key=lambda i: scaled[i], reverse=True)
    assert order_raw == order_scaled

    flat = [("o", _Goal(i, 0), 3.0) for i in range(3)]
    assert mm._scale_priors(flat, "minmax") == [mm.PRIOR_CLAMP * 0.5] * 3
    assert mm._scale_priors(pool, "raw") == [8.0, 2.0, 5.0, -1.0]


def test_training_record_labels_are_max_return_weighted_by_visits():
    """mcts_main.py save_mcts_data: label = max(child.q), weight = child.n."""
    root = mm._Node("s0", None, None, 0.0, None, 0, "r")
    a = mm._Node("s1", "o", _Goal(3, 1), 0.4, root, 1, "r.a")
    b = mm._Node(None, "o", _Goal(4, 2), 0.1, root, 1, "r.b")
    a.results, a.n = [0.0, 0.5, 1.0, 0.25], 4
    root.children = [a, b]

    rows = []
    mm._collect_records(root, rows)

    assert len(rows) == 2
    row_a = next(r for r in rows if r["edge"] == 3)
    assert row_a["label"] == pytest.approx(1.0) and row_a["num_visits"] == 4
    assert row_a["push_depth"] == 1 and row_a["expanded"] is True
    row_b = next(r for r in rows if r["edge"] == 4)
    assert row_b["label"] == pytest.approx(0.0) and row_b["num_visits"] == 1
    assert row_b["expanded"] is False
