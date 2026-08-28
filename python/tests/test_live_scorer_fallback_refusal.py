"""The ranker must never score on the legacy-BFS fallback masks.

The visualizer's fallback fires when the unified wavefront throws, and its
own message says the legacy path may use the wrong robot size. Until
2026-08-28 that warning was captured into a discarded buffer and recorded in
a flag no caller read, so a degraded render scored silently. These tests pin
the refusal: a render whose captured output contains the fallback text
raises, with the captured output in the message so the root cause survives,
and a clean render still returns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sandbox"))

import live_scorer as m  # noqa: E402


class _Viz:
    def __init__(self, prints, result="ok"):
        self._prints = prints
        self._result = result

    def generate_all_masks_highres(self, _ep, tight_crop_size_meters=None, fast_scorer=True):
        print(self._prints)
        if self._result == "ok":
            lt = {k: np.zeros((8, 8)) for k in m.TIGHT}
            return {"local_tight": lt, "local_tight_metadata": {}}
        return None


def _scorer(viz):
    s = object.__new__(m.LiveScorer)
    s.viz = viz
    s.crop_m = 1.0
    s.last_fell_back = False
    return s


def _fake_env():
    return SimpleNamespace(
        get_observation=lambda: {"obj_pose": [0, 0, 0], "robot_pose": [0, 0, 0]},
        get_object_info=lambda: {},
        get_reachable_objects=lambda: [],
    )


def _render(s):
    return m.LiveScorer.render_ctx(
        s, _fake_env(), "obj", (0.0, 0.0, 0.0), "fake.xml", region_samples=[(0.0, 0.0)]
    )


def test_a_fallback_render_refuses_and_carries_the_root_cause(monkeypatch):
    monkeypatch.setattr(
        m.LiveScorer, "_episode_data", lambda self, *a, **k: {}, raising=True
    )
    s = _scorer(_Viz("wavefront exploded: ValueError(...)\nfalling back to legacy BFS"))
    with pytest.raises(RuntimeError) as exc:
        _render(s)
    assert "wrong robot size" in str(exc.value)
    assert "wavefront exploded" in str(exc.value)  # the swallowed output resurfaces


def test_a_clean_render_still_returns(monkeypatch):
    monkeypatch.setattr(
        m.LiveScorer, "_episode_data", lambda self, *a, **k: {}, raising=True
    )
    s = _scorer(_Viz("rendered 5 channels"))
    ctx, meta = _render(s)
    assert ctx.shape[0] == len(m.TIGHT)
    assert s.last_fell_back is False
