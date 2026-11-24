"""The adapter around MORE's unmodified search: pinning, the PushState contract, both deploy modes.

What MORE's own classes do is checked in test_more_vendor_differential.py against a transliterated
PushState. This file checks what is ours: the clone pin, our substitutions (reward input, opening,
jam pruning, call budget), the two stopping modes, the collection outer loop and its records, and
the harness contract.
"""
import shutil
import subprocess
import sys

import numpy as np
import pytest

from more_toy import FakeEnv, Goal, PTS, action_source, goal_open_pts, pool_factory, verifier

mv = pytest.importorskip("namo.planners.opening.more_vendor")


# -- the clone ------------------------------------------------------------------------------------
def _clone(tmp_path, rev):
    dst = tmp_path / "more"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(mv.more_ref_path()), str(dst)], check=True)
    subprocess.run(["git", "-C", str(dst), "checkout", "-q", rev], check=True)
    return dst


def test_load_more_imports_their_unmodified_search_classes():
    more = mv.load_more()
    assert more.UctNode.__module__ == "_more_ref_mcts_nodes"
    assert more.NetNode.__module__ == "_more_ref_mcts_network_nodes"
    assert more.constants.MCTS_DISCOUNT == 0.5 and more.constants.MCTS_TOP == 10
    assert more.constants.MCTS_EARLY_ROLLOUTS == -1
    assert "constants" not in sys.modules or sys.modules["constants"] is not more.constants


def test_load_more_refuses_another_revision(tmp_path):
    ref = _clone(tmp_path, "8fdb3ea")
    with pytest.raises(RuntimeError, match="expected 70402c0"):
        mv.load_more(ref)


def test_load_more_refuses_an_edited_clone(tmp_path):
    ref = _clone(tmp_path, mv.MORE_PIN)
    with open(ref / "mcts" / "nodes.py", "a") as fh:
        fh.write("\n# edited\n")
    with pytest.raises(RuntimeError, match="local edits"):
        mv.load_more(ref)


# -- PushState contract ---------------------------------------------------------------------------
def _ctx(env, edges=2, depths=1, **kw):
    return mv.MoreContext(env, action_source(edges, depths), verifier(lambda e: goal_open_pts(e, PTS)), **kw)


def test_push_result_is_more_shape_with_the_opening_as_the_threshold():
    ctx = _ctx(FakeEnv())
    closed = mv.NamoPushState("u", (), 0.5, False, 0, ctx, 2)
    opened = mv.NamoPushState("u", (), 0.5, True, 0, ctx, 2)
    assert closed.push_result == pytest.approx(0.1)          # 0.2 * clamp(q, 0, 1)
    assert opened.push_result == pytest.approx(1.1)          # + 1 when "graspable"
    assert mv.NamoPushState("u", (), 3.0, False, 0, ctx, 2).push_result == pytest.approx(0.2)
    assert opened.push_result >= opened.max_q > closed.push_result


def test_terminal_test_is_level_opening_or_no_action_left():
    ctx = _ctx(FakeEnv())
    assert mv.NamoPushState("u", (), 0.0, False, 2, ctx, 2).is_push_over()
    assert mv.NamoPushState("u", (), 0.0, True, 0, ctx, 2).is_push_over()
    s = mv.NamoPushState("u", (), 0.0, False, 0, ctx, 2)
    assert not s.is_push_over()
    for m in list(s.get_actions()):
        s.remove_action(m)
    assert s.is_push_over()


def test_actions_are_in_a_prior_independent_order_and_the_legal_set_is_recorded():
    ctx = mv.MoreContext(FakeEnv(), action_source(3, 2, prior=lambda s, k: -k[1]), verifier())
    s = mv.NamoPushState("root", (), 0.0, False, 0, ctx, 2)
    assert [m.key for m in s.get_actions()] == sorted(("obj", e, d) for e in range(3) for d in range(2))
    assert ctx.legal["0:root"] == [(e, d) for e in range(3) for d in range(2)]


def test_a_cached_transition_costs_no_call():
    env = FakeEnv()
    ctx = _ctx(env)
    s = mv.NamoPushState("root", (), 0.0, False, 0, ctx, 2)
    m = s.get_actions()[0]
    a, _, _, rec_a = s.move(m)
    b, _, _, rec_b = s.move(m)
    assert env.steps == ctx.calls == 1 and not rec_a and rec_b and a.sim_state == b.sim_state


def test_a_deeper_push_after_a_jam_costs_no_call():
    env = FakeEnv(fails=[(("obj", 0, 0),)])                 # depth 0 through edge 0 jams
    ctx = _ctx(env, edges=1, depths=3)
    s = mv.NamoPushState("root", (), 0.0, False, 0, ctx, 2)
    moves = {m.key: m for m in s.get_actions()}
    assert s.move(moves[("obj", 0, 0)]) is not None
    assert s.move(moves[("obj", 0, 2)]) is None and env.steps == 1


def test_a_noop_is_a_failed_move_and_is_charged():
    env = FakeEnv(noops=[((), ("obj", 0, 0))])
    ctx = _ctx(env)
    s = mv.NamoPushState("root", (), 0.0, False, 0, ctx, 2)
    assert s.move(s.get_actions()[0]) is None and env.steps == 1


# -- collection -----------------------------------------------------------------------------------
def _collect(env, iters=40, edges=3, depths=1, **kw):
    return mv.collect_episode(env, (), hmax=2, iterations=iters, action_source=action_source(edges, depths),
                              verifier=verifier(), rng_seed=0, **kw)


def test_collection_runs_every_iteration_after_an_opening_and_counts_every_call():
    env = FakeEnv(opens=[(("obj", 2, 0),)])
    res = _collect(env)
    assert res["solved"] and res["plan"] == [("obj", 2, 0)]
    assert res["calls"] == env.steps == sum(s["calls"] for s in res["searches"])
    assert res["first_open_call"] == 1 and res["calls"] > 1     # kept searching, as test=False does
    assert res["searches"][0]["iterations"] == 40


def test_collection_re_searches_from_the_executed_push_with_fresh_caches():
    env = FakeEnv(opens=[(("obj", 0, 0), ("obj", 1, 0))])
    res = _collect(env, iters=60)
    assert len(res["searches"]) == 2 and res["plan"][-1] == ("obj", 1, 0) and res["solved"]
    assert {r["search"] for r in res["rows"]} == {0, 1}
    assert all(r["depth"] >= 1 for r in res["rows"] if r["search"] == 1)


def test_collection_rows_are_every_simulated_child_with_max_q_and_visits():
    env = FakeEnv(opens=[(("obj", 0, 0), ("obj", 1, 0))])
    more = mv.load_more()
    real = more.UctSearch.best_action
    trees = []

    def keep(self, *a, **k):
        trees.append(self.root)
        return real(self, *a, **k)
    mp = pytest.MonkeyPatch()
    mp.setattr(more.UctSearch, "best_action", keep)
    try:
        res = _collect(env, iters=30)
    finally:
        mp.undo()
    expect = [(node.state.uid, str(c.prev_move), max(c.q), c.n)
              for node in mv.walk(trees[0]) for c in node.children]
    got = [(r["uid"], f'{r["obj"]}_{r["edge"]}_{r["push_depth"]}', r["label"], r["num_visits"])
           for r in res["rows"] if r["search"] == 0]
    assert got == expect and all(r["num_visits"] >= 1 for r in res["rows"])
    assert set(res["parents"]) == {r["key"] for r in res["rows"]}
    assert set(res["parents"]) <= set(res["legal"])


def test_collection_stops_cleanly_when_every_root_push_fails():
    env = FakeEnv(noops=[((), ("obj", e, 0)) for e in range(3)])
    res = _collect(env)
    assert not res["solved"] and res["plan"] == [] and res["calls"] == env.steps == 3


# -- deploy ---------------------------------------------------------------------------------------
def _deploy(env, monkeypatch, mode, edges=4, depths=2, budget=1000, prior=None, num_iter=50, **kw):
    import namo.planners.opening.best_first_search as bfs
    monkeypatch.setattr(bfs, "candidates", pool_factory(edges, depths, prior=prior))
    np.random.seed(0)
    sol = {}
    out = mv.solve_scene_more(None, env, None, "x", (), 2, budget, "model", "mean5", "q", None,
                              is_open=lambda e: goal_open_pts(e, PTS), region_samples=PTS,
                              solution_out=sol, stop_mode=mode, num_iter=num_iter, **kw)
    return out, sol


RANKED = (lambda s, k: 1.0 - 0.2 * k[1])     # priors inside MORE's [0, 1.2] clamp, edge 0 best


def test_protocol_stops_at_an_opening_found_by_a_rollout(monkeypatch):
    """Call 1 expands the top-prior push; its greedy rollout pushes the top prior again and opens.

    The priors must sit inside [0, 1.2]: MORE clamps them in selection (mcts_network/nodes.py:117),
    so priors that all exceed 1.2 tie and its first-max rule picks the LAST action in the list
    (pre_expand pops from the end, :54-57), while the rollout still ranks them unclamped (:145-153).
    """
    env = FakeEnv(opens=[(("obj", 0, 0), ("obj", 0, 0))])
    (solved, sims, plen, _b, end), sol = _deploy(env, monkeypatch, "protocol", depths=1, prior=RANKED)
    assert solved and end == "solved" and sims == env.steps == 2 and plen == 2
    assert sol["total_sims"] == 2


def test_protocol_reports_the_budget(monkeypatch):
    env = FakeEnv()
    (solved, sims, _p, _b, end), _s = _deploy(env, monkeypatch, "protocol", edges=30, depths=5, budget=7)
    assert not solved and end == "budget" and sims == env.steps == 7


def test_protocol_ends_when_nothing_can_be_simulated(monkeypatch):
    env = FakeEnv()
    (solved, sims, _p, _b, end), _s = _deploy(env, monkeypatch, "protocol", edges=3, depths=1)
    assert not solved and end == "exhausted" and sims == env.steps == 3 + 9


def test_protocol_detects_more_greedy_starvation(monkeypatch):
    """Zero priors everywhere: MORE's greedy selection keeps re-visiting the first expanded subtree
    (its returns are > 0) and never selects a sibling whose score is a clamped 0. It must end."""
    env = FakeEnv()
    (solved, sims, _p, _b, end), _s = _deploy(env, monkeypatch, "protocol", edges=4, depths=1,
                                              prior=lambda s, k: -1.0, stall_iters=2000)
    assert not solved and end == "stalled" and sims == env.steps < 4 + 16


def test_native_executes_and_re_plans_until_the_push_that_opens(monkeypatch):
    env = FakeEnv(opens=[(("obj", 0, 0), ("obj", 0, 0))])
    (solved, sims, plen, _b, end), sol = _deploy(env, monkeypatch, "native", depths=1, num_iter=10)
    assert solved and end == "solved" and plen == 2 and sims == env.steps
    assert [(o, g.edge_idx) for o, g in sol["plan"]] == [("obj", 0), ("obj", 0)]


def test_native_charges_every_re_search(monkeypatch):
    """Protocol stops inside the first search; native pays for the second search from scratch."""
    scene = dict(opens=[(("obj", 0, 0), ("obj", 0, 0))])
    (_s1, protocol_calls, *_), _ = _deploy(FakeEnv(**scene), monkeypatch, "protocol", depths=1, prior=RANKED)
    (_s2, native_calls, *_), _ = _deploy(FakeEnv(**scene), monkeypatch, "native", depths=1, num_iter=10,
                                         prior=RANKED)
    assert native_calls > protocol_calls


def test_native_stops_at_the_horizon(monkeypatch):
    env = FakeEnv()
    (solved, sims, plen, _b, end), _s = _deploy(env, monkeypatch, "native", depths=1, num_iter=5)
    assert not solved and end == "horizon" and plen is None and sims == env.steps


def test_ppn_executes_the_top_prior_push_at_each_live_state(monkeypatch):
    """No search: two executed pushes, two calls, each the highest prior (edge 0, depth 0)."""
    env = FakeEnv(opens=[(("obj", 0, 0), ("obj", 0, 0))])
    (solved, sims, plen, _b, end), sol = _deploy(env, monkeypatch, "ppn", depths=2)
    assert solved and end == "solved" and sims == env.steps == 2 and plen == 2
    assert [(o, g.edge_idx, g.depth) for o, g in sol["plan"]] == [("obj", 0, 0), ("obj", 0, 0)]


def test_ppn_drops_a_failed_push_and_tries_the_next_best_from_the_same_state(monkeypatch):
    """The top push is a no-op at the start: it is charged and uses a push; the next best,
    edge 0 depth 1, then opens from the same state."""
    env = FakeEnv(opens=[(("obj", 0, 1),)], noops=[((), ("obj", 0, 0))])
    (solved, sims, plen, _b, end), sol = _deploy(env, monkeypatch, "ppn", depths=2)
    assert solved and end == "solved" and sims == env.steps == 2 and plen == 2
    assert env.log == [((), ("obj", 0, 0)), ((), ("obj", 0, 1))]


def test_ppn_stops_at_the_horizon(monkeypatch):
    env = FakeEnv()
    (solved, sims, plen, _b, end), _s = _deploy(env, monkeypatch, "ppn", depths=1)
    assert not solved and end == "horizon" and plen is None and sims == env.steps == 2


def test_measurement_hooks_see_every_call(monkeypatch):
    class Meas:
        def __init__(self):
            self.calls, self.boards = 0, 0

        def board(self, board_id, parent_board_id, depth, pool):
            self.boards += 1

        def simulated(self, **kw):
            self.calls += 1
    m = Meas()
    env = FakeEnv(opens=[(("obj", 1, 1), ("obj", 2, 0))])
    (solved, sims, _p, boards, _e), _s = _deploy(env, monkeypatch, "protocol", measurements=m)
    assert m.calls == sims == env.steps and m.boards == len(boards) >= 1
    keys = {"board_id", "depth", "n_candidates", "k_failed", "tries", "parent_edge", "parent_depth",
            "pool", "grid", "w0", "free_strikes", "geom", "regions"}
    assert all(keys <= set(b) for b in boards)


def test_unknown_stop_mode_is_refused(monkeypatch):
    with pytest.raises(ValueError):
        _deploy(FakeEnv(), monkeypatch, "fast")
