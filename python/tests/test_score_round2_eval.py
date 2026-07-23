import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/rl_loop/score_round2_eval.py"
SPEC = importlib.util.spec_from_file_location("score_round2_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_first_positive_rank_uses_only_known_comparison_cells():
    scores = np.array([0.99, 0.8, 0.7, 0.6])
    positive = np.array([False, False, True, False])
    candidates = np.array([False, True, True, True])
    assert MODULE.first_positive_rank(scores, positive, candidates) == 2


def test_rank_summary_reports_hit_rates_and_first_rank():
    summary = MODULE.summarize_ranks([1, 2, 6, 10])
    assert summary["hit_at_1_pct"] == 25.0
    assert summary["hit_at_5_pct"] == 50.0
    assert summary["median_first_rank"] == 4.0
