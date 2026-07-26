import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from viz.index_metrics import rank_of_best_green, top1_truth  # noqa: E402


def _pool(*triples):
    return [{"obj": "o", "edge": e, "depth": d, "q": q} for (e, d, q) in triples]


def test_rank_one_when_the_model_tops_a_green():
    pool = _pool((5, 0, 0.9), (7, 1, 0.4))
    assert rank_of_best_green(pool, {(5, 0)}) == 1


def test_rank_counts_position_in_descending_q_order():
    pool = _pool((5, 0, 0.9), (7, 1, 0.8), (9, 2, 0.7))
    assert rank_of_best_green(pool, {(9, 2)}) == 3


def test_rank_is_none_when_nothing_is_green():
    assert rank_of_best_green(_pool((5, 0, 0.9)), set()) is None


def test_ties_break_deterministically_on_edge_then_depth():
    pool = _pool((9, 0, 0.5), (2, 0, 0.5))
    assert rank_of_best_green(pool, {(9, 0)}) == 2
    assert rank_of_best_green(pool, {(2, 0)}) == 1


def test_empty_pool():
    assert rank_of_best_green([], {(1, 1)}) is None
    assert top1_truth([], set(), set()) == "empty"


def test_top1_truth_labels_the_highest_scoring_candidate():
    pool = _pool((5, 0, 0.9), (7, 1, 0.4))
    assert top1_truth(pool, {(5, 0)}, set()) == "opener"
    assert top1_truth(pool, set(), {(5, 0)}) == "setup"
    assert top1_truth(pool, {(7, 1)}, set()) == "dead"


from viz.build_manifest import index_row, _tier_lookup  # noqa: E402


def _trace(pool, solved=True, sims=3):
    return {"meta": {"xml": "/x/a.xml", "object_id": "o"},
            "boards": [{"board_id": 0, "depth": 0, "pool": pool}],
            "result": {"solved": solved, "sims": sims}}


def test_index_row_uses_the_root_board_ordering():
    # Root (depth 0) pool: best-green (7, 1) sits second -> rank 2, "dead".
    root_pool = [{"obj": "o", "edge": 5, "depth": 0, "q": 0.9},
                 {"obj": "o", "edge": 7, "depth": 1, "q": 0.4}]
    # A non-root board placed FIRST in the list: its top candidate IS the
    # best-green edge, so grabbing boards[0] unconditionally would score
    # rank 1 / "opener" instead of the root's rank 2 / "dead".
    first_pool = [{"obj": "o", "edge": 7, "depth": 1, "q": 0.99}]
    # A non-root board placed LAST: its pool doesn't contain the best-green
    # edge at all, so grabbing boards[-1] unconditionally would score
    # rank None instead of the root's rank 2.
    last_pool = [{"obj": "o", "edge": 3, "depth": 1, "q": 0.5}]
    trace = {
        "meta": {"xml": "/x/a.xml", "object_id": "o"},
        "boards": [
            {"board_id": 1, "depth": 1, "pool": first_pool},
            {"board_id": 0, "depth": 0, "pool": root_pool},
            {"board_id": 2, "depth": 1, "pool": last_pool},
        ],
        "result": {"solved": True, "sims": 3},
    }
    gt = {"root": {"openers": [[7, 1]], "setups": []}}
    row = index_row(trace, gt, "hard")
    assert row["rank_best_green"] == 2
    assert row["top1"] == "dead"
    assert row["tier"] == "hard" and row["has_gt"] is True
    assert row["key"].startswith("a__o__")


def test_index_row_without_gt_is_marked_and_has_no_rank():
    pool = [{"obj": "o", "edge": 5, "depth": 0, "q": 0.9}]
    row = index_row(_trace(pool), None, "easy")
    assert row["has_gt"] is False
    assert row["rank_best_green"] is None
    assert row["top1"] == "unknown"


def test_tier_lookup_joins_on_realpath_not_basename():
    # Two different scene directories that happen to share an xml basename,
    # carrying DIFFERENT tiers for the same object_id. A basename-only join
    # would collide these into one entry and silently mis-tier one of them.
    div = {
        "/rooms/set1/room.xml": [{"object_id": "obstacle_1", "division": "easy"}],
        "/rooms/set2/room.xml": [{"object_id": "obstacle_1", "division": "hard"}],
    }
    lookup = _tier_lookup(div)

    assert lookup[(os.path.realpath("/rooms/set1/room.xml"), "obstacle_1")] == "easy"
    assert lookup[(os.path.realpath("/rooms/set2/room.xml"), "obstacle_1")] == "hard"

    # An episode whose xml never appears in the divisions data at all: the
    # lookup simply has no entry for it. main()'s consumer does
    # tiers.get((realpath, object_id), "unknown"), so that's the behavior we
    # pin here too -- this test doesn't change what "unknown" means, only
    # documents it.
    missing_key = (os.path.realpath("/rooms/set3/room.xml"), "obstacle_1")
    assert missing_key not in lookup
    assert lookup.get(missing_key, "unknown") == "unknown"
