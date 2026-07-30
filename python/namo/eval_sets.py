"""Canonical eval / test-set pointer — the single source of truth for test-set paths.

All eval code imports its test-set paths from here instead of hardcoding
``namo_testset_v1/labels/...``. The paths themselves live in ``config/eval_sets.yaml``
(relative to ``$NAMO_SCRATCH``); this module resolves them onto the current box via
``namo.paths``. Change the test set = edit the yaml only; every reader follows.

    from namo import eval_sets
    key = eval_sets.PURE2PUSH          # resolved absolute Path on this box
    key = eval_sets.path("onepush_manifest")

Companion doc: docs/experiments/eval_set_registry.md. Lives at ``namo.eval_sets``
(not ``namo.core.*``) so pure-Python eval/agg scripts import it without pulling in
the compiled ``namo_rl`` binding.
"""
import yaml
from pathlib import Path

from namo.paths import SCRATCH

# repo config/: this file is python/namo/eval_sets.py → parents[2] is the repo root.
_CFG_PATH = Path(__file__).resolve().parents[2] / "config" / "eval_sets.yaml"


def _load():
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


_CFG = _load()
_FILES = _CFG["files"]
EXPECTED = _CFG.get("expected_counts", {})
EXCLUSIONS = _CFG.get("search_eval_exclusions", [])
TESTSET = _CFG["testset"]


def path(name):
    """Resolve a named eval-set file to an absolute Path on this box.

    ``name`` is a key under ``files:`` in config/eval_sets.yaml. Raises KeyError
    (loud, not a silent wrong path) if the name is unknown.
    """
    if name not in _FILES:
        raise KeyError(f"unknown eval-set {name!r}; known: {sorted(_FILES)}")
    return SCRATCH / _FILES[name]


# Convenience attributes — the canonical set, resolved.
ONEPUSH = path("onepush_manifest")
PURE2PUSH = path("pure2push_manifest")
DIVISIONS = path("pure2push_divisions")
SAMPLED_DIVISIONS = path("pure2push_sampled_divisions")
TWOPUSH_SOURCE = path("twopush_source")
TWOPUSH_GT_H5 = path("twopush_gt_h5")


if __name__ == "__main__":
    # CLI for slurm/sh: resolve a named eval-set path single-source.
    #   python -m namo.eval_sets <name>   → one absolute path, no trailing junk
    #   python -m namo.eval_sets --list   → all names, one per line
    import sys

    if len(sys.argv) != 2:
        sys.exit("usage: python -m namo.eval_sets <name>|--list")
    arg = sys.argv[1]
    if arg == "--list":
        for name in _FILES:
            print(name)
    else:
        print(path(arg))
