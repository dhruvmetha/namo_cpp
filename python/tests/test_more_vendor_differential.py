"""Differential: MORE's unmodified search driven by our adapter vs by a transliterated PushState.

Both sides run MORE's own mcts/ and mcts_network/ classes, imported unmodified from the pinned
clone. The reference side drives them with RefPushState (more_toy.py), a line-for-line copy of
mcts/push.py onto the toy simulator; the candidate side drives them with the real adapter
(NamoPushState, MoreContext, more_vendor.py). Same numpy seed, same scenes. Required: identical
simulator-call sequences, identical trees (every node's uid, visit count n and backed returns q),
identical returned action. Scenes include 1-push openers, 2-push setup/finish pairs, failed
(no-op) moves at the root and below, controller failures, and states at exactly 20/100.

Jam-depth pruning is OFF on the adapter here: it is our action library's substitution (SPEC row
16) and MORE has no counterpart; test_more_vendor_adapter.py covers it separately.
"""
import random

import numpy as np
import pytest

from more_toy import (FakeEnv, RefHelper, RefPushState, action_source, goal_open_pts, PTS,
                      stable_prior, verifier)

more_vendor = pytest.importorskip("namo.planners.opening.more_vendor")
MORE = more_vendor.load_more()


def scene(seed):
    r = random.Random(seed)
    edges, depths = r.randint(2, 5), r.randint(1, 3)
    acts = [("obj", e, d) for e in range(edges) for d in range(depths)]
    opens = set()
    if r.random() < 0.4:
        opens.add((r.choice(acts),))                                  # 1-push openers
    for _ in range(r.randint(0, 3)):
        opens.add((r.choice(acts), r.choice(acts)))                   # 2-push setup + finish
    noops = {((), r.choice(acts)) for _ in range(r.randint(0, 2))}    # failed moves at the root
    for _ in range(r.randint(0, 2)):
        a = r.choice(acts)
        noops.add(((a,), r.choice(acts)))                             # and one level down
    reach = {(r.choice(acts),): r.choice([0.1, 0.19, 0.2, 0.21]) for _ in range(r.randint(0, 3))}
    fails = {(r.choice(acts),) for _ in range(r.randint(0, 2))}
    iters = r.choice([1, 2, 5, 20, 60, 150, 300])
    return dict(edges=edges, depths=depths, opens=opens, noops=noops, reach=reach, fails=fails,
                iters=iters, eval_mode=r.random() < 0.5)


def is_open(e):
    return goal_open_pts(e, PTS)


def env_of(sc):
    return FakeEnv(opens=sc["opens"], noops=sc["noops"], reach=sc["reach"], fails=sc["fails"])


def tree_signature(root):
    out = []
    for node in more_vendor.walk(root):
        st = node.state
        out.append((st.uid if st is not None else None,
                    str(node.prev_move) if node.prev_move is not None else None,
                    node.n, tuple(round(v, 12) for v in node.q)))
    return out


def run_reference(sc, network, seed):
    env = env_of(sc)
    prior = stable_prior if network else None
    helper = RefHelper(env, sc["edges"], sc["depths"], verifier(is_open), prior=prior)
    root_state = RefPushState("root", (), 0.02, False, 0, helper, max_level=2)
    Node, Search = (MORE.NetNode, MORE.NetSearch) if network else (MORE.UctNode, MORE.UctSearch)
    root = Node(root_state)
    np.random.seed(seed)
    search = Search(root)
    best = more_vendor.run_search(search, root, sc["iters"], sc["eval_mode"], MORE)
    return env.log, tree_signature(root), (str(best.prev_move) if best is not None else None)


def run_adapter(sc, network, seed):
    env = env_of(sc)
    prior = stable_prior if network else None
    ctx = more_vendor.MoreContext(env, action_source(sc["edges"], sc["depths"], prior),
                                  verifier(is_open), dedupe_noop=True, prune_jam_depth=False)
    root_state = more_vendor.NamoPushState("root", (), 0.02, False, 0, ctx, max_level=2)
    Node, Search = (MORE.NetNode, MORE.NetSearch) if network else (MORE.UctNode, MORE.UctSearch)
    root = Node(root_state)
    np.random.seed(seed)
    search = Search(root)
    best = more_vendor.run_search(search, root, sc["iters"], sc["eval_mode"], MORE)
    assert ctx.calls == env.steps
    return env.log, tree_signature(root), (str(best.prev_move) if best is not None else None)


@pytest.mark.parametrize("network", [False, True], ids=["collection_uct", "deploy_network"])
@pytest.mark.parametrize("seed", range(150))
def test_adapter_matches_reference(seed, network):
    sc = scene(seed)
    ref = run_reference(sc, network, seed)
    ada = run_adapter(sc, network, seed)
    assert ada[0] == ref[0], "simulator call sequences differ"
    assert ada[1] == ref[1], "trees differ (uid, move, visits, backed returns)"
    assert ada[2] == ref[2], "returned action differs"


# ------------------------------------------------------------------------------------------------
# The outer loops: collection (mcts_main.py:422-562) and native deploy (more_main.py:393-525)
# ------------------------------------------------------------------------------------------------
def reference_collection(sc, seed, hmax=2):
    """mcts_main.py:422-562 on the reference state: search, save, execute best_child_top, reset."""
    env = env_of(sc)
    helper = RefHelper(env, sc["edges"], sc["depths"], verifier(is_open))
    np.random.seed(seed)
    state, frac, plan, per_search, rows = (), 0.02, [], [], []
    for step in range(hmax):
        helper.reset()
        root = MORE.UctNode(RefPushState("root", state, frac, False, 0, helper, max_level=hmax - step))
        steps0 = env.steps
        search = MORE.UctSearch(root)
        best = more_vendor.run_search(search, root, sc["iters"], False, MORE) \
            if root.state.get_actions() else None
        for node in more_vendor.walk(root):
            for c in node.children:
                rows.append((step, node.state.uid, str(c.prev_move), max(c.q), c.n))
        per_search.append(env.steps - steps0)
        if best is None or not root.children:
            break
        plan.append(str(best.prev_move))
        state, frac = best.state.object_states, best.state.q_value
        if best.state.opened:
            break
    return env.steps, plan, per_search, rows


@pytest.mark.parametrize("seed", range(60))
def test_collection_episode_matches_reference(seed):
    sc = scene(seed)
    ref_calls, ref_plan, ref_per, ref_rows = reference_collection(sc, seed)
    env = env_of(sc)
    res = more_vendor.collect_episode(env, (), hmax=2, iterations=sc["iters"],
                                      action_source=action_source(sc["edges"], sc["depths"]),
                                      verifier=verifier(is_open), prune_jam_depth=False, rng_seed=seed)
    rows = [(r["search"], r["uid"], f'{r["obj"]}_{r["edge"]}_{r["push_depth"]}', r["label"], r["num_visits"])
            for r in res["rows"]]
    assert res["calls"] == env.steps == ref_calls
    assert ["{}_{}_{}".format(*k) for k in res["plan"]] == ref_plan
    assert [s["calls"] for s in res["searches"]] == ref_per
    assert rows == ref_rows


def reference_native(sc, seed, num_iter, hmax=2):
    """more_main.py:393-525: search (eval=True), execute best_child_top, reset, repeat."""
    env = env_of(sc)
    helper = RefHelper(env, sc["edges"], sc["depths"], verifier(is_open), prior=stable_prior)
    np.random.seed(seed)
    state, frac, plan = (), 0.02, []
    for step in range(hmax):
        helper.reset()
        root = MORE.NetNode(RefPushState("root", state, frac, False, 0, helper, max_level=hmax - step))
        search = MORE.NetSearch(root)
        best = more_vendor.run_search(search, root, num_iter, True, MORE) if root.children else None
        if best is None:
            return env.steps, plan, False
        child = best.state if best.state is not None else root.state.move(best.prev_move)
        plan.append(str(best.prev_move))
        if child is None:
            continue
        child = child[0] if isinstance(child, tuple) else child
        state, frac = child.object_states, child.q_value
        if child.opened:
            return env.steps, plan, True
    return env.steps, plan, False


@pytest.mark.parametrize("seed", range(60))
def test_native_deploy_matches_reference(seed):
    sc = scene(seed)
    ref_calls, ref_plan, ref_solved = reference_native(sc, seed, num_iter=max(sc["iters"], 5))
    env = env_of(sc)
    from more_toy import pool_factory
    import namo.planners.opening.best_first_search as bfs
    real = bfs.candidates
    bfs.candidates = pool_factory(sc["edges"], sc["depths"], prior=stable_prior)
    try:
        np.random.seed(seed)
        sol = {}
        solved, sims, plen, _b, end = more_vendor.solve_scene_more(
            None, env, None, "x", (), 2, 10_000, "model", "mean5", "q", None, is_open=is_open,
            region_samples=PTS, prune_jam_depth=False, solution_out=sol, stop_mode="native",
            num_iter=max(sc["iters"], 5))
    finally:
        bfs.candidates = real
    assert sims == env.steps == ref_calls
    assert solved == ref_solved
    if solved:
        assert ["{}_{}_{}".format(o, g.edge_idx, g.depth) for o, g in sol["plan"]] == ref_plan


def test_the_differential_catches_a_reward_without_the_opening_bonus(monkeypatch):
    """Guard check: break the adapter's push_result the way BUG-1 broke the port; it must show."""
    monkeypatch.setattr(more_vendor.NamoPushState, "push_result",
                        property(lambda self: 0.2 * min(max(self.frac, 0.0), 1.0)))
    caught = sum(run_adapter(scene(s), net, s) != run_reference(scene(s), net, s)
                 for net in (False, True) for s in range(60))
    assert caught > 0


def test_the_differential_catches_a_prior_dependent_action_order(monkeypatch):
    """Guard check: MORE's tie-breaks follow the action list order, so an unsorted list must show."""
    def unsorted(self, state):
        return [more_vendor.NamoMove(o, g, q) for (o, g, q) in self._actions(state)]
    monkeypatch.setattr(more_vendor.MoreContext, "actions_for", unsorted)
    caught = sum(run_adapter(scene(s), net, s) != run_reference(scene(s), net, s)
                 for net in (False, True) for s in range(60))
    assert caught > 0
