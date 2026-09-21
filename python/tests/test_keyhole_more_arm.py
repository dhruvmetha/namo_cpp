"""Adding a MORE arm to the keyhole runner must not disturb the best-first arms.

The keyhole600 best-first results are registered (`hy5u-ablations-keyhole600-1mm-v1`),
and every output row carries the `params` dict describing the protocol that produced it.
If a new arm type silently added a key there, arms whose search did not change would
still record a different protocol, and the registry rule "reuse an entry only when the
full protocol matches" would stop being checkable by comparing those dicts.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "pipeline" / "run_one_keyhole_frozen.py"

# Exactly what a best-first arm recorded before the MORE arm existed.
BESTFIRST_PARAM_KEYS = {
    "prior", "hmax", "sim_budget", "agg", "combine", "raw", "dive_bonus", "discount",
    "gamma", "tau", "eps", "w0_mode", "free_strike_q", "child_patience",
    "dedupe_noop", "prune_jam_depth",
}


def _load():
    """Import the runner without executing its CLI."""
    spec = importlib.util.spec_from_file_location("run_one_keyhole_frozen", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                      # pragma: no cover - env-dependent imports
        pytest.skip(f"runner not importable here: {exc}")
    return mod


BESTFIRST_ARM = {"name": "HY5U_s1", "prior": "model", "checkpoint": "/ckpt.ckpt", "seed_base": 7000}
MORE_ARM = {"name": "more_s1", "prior": "model", "checkpoint": "/ckpt.ckpt", "seed_base": 7000,
            "planner": "more"}


def test_bestfirst_arm_records_exactly_the_registered_protocol_keys():
    mod = _load()
    _options, params = mod.search_options(BESTFIRST_ARM, Path("/problems"))
    assert set(params) == BESTFIRST_PARAM_KEYS


def test_bestfirst_arm_carries_no_planner_attribute():
    """eval_bestfirst reads the planner with getattr, so absence must mean best-first."""
    mod = _load()
    options, _params = mod.search_options(BESTFIRST_ARM, Path("/problems"))
    assert not hasattr(options, "planner")
    assert getattr(options, "planner", "bestfirst") == "bestfirst"


def test_more_arm_selects_the_mcts_and_records_it():
    mod = _load()
    options, params = mod.search_options(MORE_ARM, Path("/problems"))
    assert options.planner == "more"
    assert params["planner"] == "more"
    assert params["more_prior_scale"] == "minmax"      # default, stated in the row
    assert "more_records_out" not in params, "a scratch path is not part of the protocol"


def test_more_arm_keeps_every_shared_search_setting_identical():
    """The two arms must differ ONLY by the planner, or the comparison means nothing."""
    mod = _load()
    _o1, bf = mod.search_options(BESTFIRST_ARM, Path("/problems"))
    _o2, more = mod.search_options(MORE_ARM, Path("/problems"))
    assert set(more) - set(bf) == {"planner", "more_prior_scale"}
    for key in bf:
        assert bf[key] == more[key], f"{key} differs between the arms"
    assert bf["hmax"] == 2 and bf["sim_budget"] == 3000 and bf["discount"] == "off"
