"""The live ranker exposes an explicit device warmup."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SANDBOX = Path(__file__).resolve().parents[2] / "scripts" / "sandbox"
if str(SANDBOX) not in sys.path:
    sys.path.insert(0, str(SANDBOX))

from live_scorer import LiveScorer  # noqa: E402
import eval_bestfirst  # noqa: E402


def test_warmup_runs_three_synthetic_h2_forward_passes():
    scorer = LiveScorer.__new__(LiveScorer)
    calls = []

    def score_ctx(ctx, contact_px, *, h, raw):
        calls.append((ctx.copy(), contact_px.copy(), h, raw))
        return np.zeros((60, 5), dtype=np.float32)

    scorer.score_ctx = score_ctx

    scorer.warmup()

    assert len(calls) == 3
    for ctx, contact_px, horizon, raw in calls:
        assert ctx.shape == (5, 64, 64)
        assert contact_px.shape == (60, 2)
        assert horizon == 2
        assert raw is True


def test_best_first_eval_warms_model_immediately_after_loading(monkeypatch):
    events = []

    class FakeScorer:
        device = "cpu"

        def warmup(self, *, repeats):
            events.append(("warmup", repeats))

    class FakeBeamPlanner:
        def __init__(self, *, ckpt):
            events.append(("load", ckpt))
            self.scorer = FakeScorer()

    monkeypatch.setattr(eval_bestfirst, "BeamPlanner", FakeBeamPlanner)

    _planner, device, applied = eval_bestfirst._make_planner("model", "HY5U.ckpt", 3)

    assert events == [("load", "HY5U.ckpt"), ("warmup", 3)]
    assert device == "cpu"
    assert applied == 3
