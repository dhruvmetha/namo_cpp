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
    for name in ("onepush_source", "onepush_manifest", "pure2push_source", "pure2push_manifest", "pure2push_divisions",
                 "pure2push_gt_divisions_source",
                 "pure2push_sampled_divisions",
                 "twopush_source", "twopush_gt_h5"):
        p = E.path(name)
        assert p.exists(), f"{name} resolves to a missing file: {p}"


def test_episode_counts_match_expected():
    exp = E.EXPECTED
    assert _n_episodes(E.ONEPUSH_SOURCE) == exp["onepush_source_episodes"]
    assert _n_episodes(E.ONEPUSH) == exp["onepush_manifest_episodes"]
    assert _n_episodes(E.PURE2PUSH_SOURCE) == exp["pure2push_source_episodes"]
    assert _n_episodes(E.PURE2PUSH) == exp["pure2push_manifest_episodes"]
    assert _n_episodes(E.TWOPUSH_SOURCE) == exp["twopush_source_episodes"]
    assert _n_episodes(E.GT_DIVISIONS_SOURCE) == exp["pure2push_gt_divisions_source_episodes"]


def test_division_tiers_match_expected():
    d = json.load(open(E.DIVISIONS))
    counts = {}
    for eps in d.values():
        for e in eps:
            tier = e["division"]
            counts[tier] = counts.get(tier, 0) + 1
    assert counts == E.EXPECTED["divisions"]


def test_legacy_sampled_division_tiers_match_expected():
    d = json.load(open(E.LEGACY_SAMPLED_DIVISIONS))
    counts = {}
    for eps in d.values():
        for e in eps:
            tier = e["division"]
            counts[tier] = counts.get(tier, 0) + 1
    assert counts == E.EXPECTED["sampled_divisions"]


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


def test_canonical_random_baseline_is_three_seed_hmax2():
    baseline = E.RANDOM_SEARCH_BASELINE
    assert baseline["status"] == "COMPLETE_v3_population"
    assert baseline["population_episodes"] == {
        "1push": E.EXPECTED["onepush_manifest_episodes"],
        "2push": E.EXPECTED["pure2push_manifest_episodes"],
    }
    assert baseline["ranker"] == "uniform_random"
    assert baseline["seeds"] == [7000, 8000, 9000]
    assert baseline["report"] == "mean_plus_sample_sd"
    expected_search = {
        "agg": "mean5",
        "child_patience": 1,
        "combine": "q",
        "dedupe_noop": True,
        "discount": "off",
        "dive_bonus": 0.0,
        "eps": 0.001,
        "free_strike_q": 2.0,
        "gamma": 0.65,
        "gtable": None,
        "hmax": 2,
        "prior": "uniform",
        "prune_jam_depth": True,
        "raw": True,
        "sim_budget": 900,
        "tau": 1.0,
        "w0_mode": "one",
    }
    assert baseline["search"] == expected_search
    for seed, artifact in E.baseline_paths("random_search_hmax2").items():
        assert artifact.exists(), f"random seed {seed} artifact is missing: {artifact}"
        result = json.load(open(artifact))
        assert result["search"] == expected_search
        assert result["1push"]["all"]["n"] == E.EXPECTED["onepush_manifest_episodes"]
        assert result["2push"]["all"]["n"] == E.EXPECTED["pure2push_manifest_episodes"]
        assert result["2push"]["hard"]["n"] == E.EXPECTED["divisions"]["hard"]


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        E.path("does_not_exist")
