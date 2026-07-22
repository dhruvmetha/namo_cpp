import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("eval_scorer_under_test", REPO / "scripts/eval_scorer.py")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def _scores(best_edge, best_depth):
    scores = np.zeros((60, 5), dtype=np.float32)
    scores[best_edge, best_depth] = 1.0
    return scores


def test_diagnostic_record_separates_edge_and_depth_errors():
    valid = {(2, 1), (2, 2)}
    tried = {(e, d) for e in (2, 4) for d in range(5)}
    pool = [e * 5 + d for e in (2, 4) for d in range(5)]

    success = EVAL.diagnostic_record(_scores(2, 1), valid, tried, pool, {"sr": 0.01})
    wrong_depth = EVAL.diagnostic_record(_scores(2, 4), valid, tried, pool, {"sr": 0.01})
    wrong_edge = EVAL.diagnostic_record(_scores(4, 0), valid, tried, pool, {"sr": 0.01})

    assert success["cat"] == "success"
    assert wrong_depth["cat"] == "right_edge_wrong_depth"
    assert wrong_depth["depth_right"] is False
    assert wrong_edge["cat"] == "wrong_edge"
    assert wrong_edge["depth_right"] is None


def test_aggregate_records_preserves_three_way_failure_percentages():
    valid = {(2, 1)}
    tried = {(e, d) for e in (2, 4) for d in range(5)}
    pool = [e * 5 + d for e in (2, 4) for d in range(5)]
    records = [
        EVAL.diagnostic_record(_scores(2, 1), valid, tried, pool, {"sr": 0.01}),
        EVAL.diagnostic_record(_scores(2, 4), valid, tried, pool, {"sr": 0.01}),
        EVAL.diagnostic_record(_scores(4, 0), valid, tried, pool, {"sr": 0.01}),
    ]

    hard = EVAL.aggregate_records(records)["hard"]
    assert hard["failure_decomp_at1_pct"] == {
        "success": 33.3,
        "right_edge_wrong_depth": 33.3,
        "wrong_edge": 33.3,
        "not_reachable": 0.0,
    }
    assert hard["valid_missing_from_pool"] == 0
