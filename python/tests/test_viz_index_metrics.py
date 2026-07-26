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


from viz.build_manifest import index_row  # noqa: E402


def _trace(pool, solved=True, sims=3):
    return {"meta": {"xml": "/x/a.xml", "object_id": "o"},
            "boards": [{"board_id": 0, "depth": 0, "pool": pool}],
            "result": {"solved": solved, "sims": sims}}


def test_index_row_uses_the_root_board_ordering():
    pool = [{"obj": "o", "edge": 5, "depth": 0, "q": 0.9},
            {"obj": "o", "edge": 7, "depth": 1, "q": 0.4}]
    gt = {"root": {"openers": [[7, 1]], "setups": []}}
    row = index_row(_trace(pool), gt, "hard")
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
