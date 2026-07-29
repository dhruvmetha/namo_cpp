"""Loader safety net for the canonical eval-set pointer.

If config/eval_sets.yaml points at a wrong/missing/renamed file, THESE fail loud —
which is exactly what lets scripts migrate to `namo.eval_sets` safely. Counts are
asserted against the yaml's own `expected_counts`, so both stay in lockstep.

Run (needs the box env for namo.paths):  source env.<machine>.sh && pytest python/tests/test_eval_sets.py
"""
import json

import h5py
import pytest

from namo import eval_sets as E
from eval_common import bin_of


def _n_episodes(p):
    d = json.load(open(p))
    return sum(len(v) for v in d.values())


def test_all_files_exist():
    for name in ("onepush_manifest", "pure2push_manifest", "pure2push_divisions",
                 "twopush_source", "twopush_gt_h5"):
        p = E.path(name)
        assert p.exists(), f"{name} resolves to a missing file: {p}"


def test_episode_counts_match_expected():
    exp = E.EXPECTED
    assert _n_episodes(E.ONEPUSH) == exp["onepush_manifest_episodes"]
    assert _n_episodes(E.PURE2PUSH) == exp["pure2push_manifest_episodes"]
    assert _n_episodes(E.TWOPUSH_SOURCE) == exp["twopush_source_episodes"]


def test_division_tiers_match_expected():
    d = json.load(open(E.DIVISIONS))
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for eps in d.values():
        for e in eps:
            counts[e["division"]] += 1
    assert counts == E.EXPECTED["divisions"]


def test_onepush_fixed_difficulty_counts_match_expected():
    d = json.load(open(E.ONEPUSH))
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for eps in d.values():
        for e in eps:
            tier = bin_of(e["solve_rate"])
            counts["medium" if tier == "med" else tier] += 1
    assert counts == E.EXPECTED["onepush_divisions"]


def test_gt_h5_node_count():
    with h5py.File(E.TWOPUSH_GT_H5, "r") as f:
        assert f["node_kind"].shape[0] == E.EXPECTED["twopush_gt_h5_nodes"]


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        E.path("does_not_exist")
