"""Central path resolver — the single source of truth for machine-specific roots.

Every script/module imports its roots from here instead of hardcoding
``/scratch/dm1487``, so the SAME committed file works on Amarel, ilab, and any
new box with zero edits. The only machine-specific input is the environment,
set once per box via ``source env.<machine>.sh`` (env.amarel.sh / env.ilab.sh).
This is what makes the repo portable through git: no per-checkout edits to
tracked files, so ``git pull``/``push`` stay clean on every box.

Lives at ``namo.paths`` (NOT ``namo.core.paths``) on purpose: ``namo.core``
eagerly imports the compiled ``namo_rl`` binding, but ``namo.__init__`` is lazy,
so ``from namo.paths import ...`` is cheap and works in pure-Python scripts
(plots, manifest builders) that never touch physics.

Roots come from env vars. ``NAMO_SCRATCH`` is the base; the rest derive from it
unless individually overridden. We FAIL LOUD if a required root is unset rather
than defaulting to ``/scratch/dm1487`` — a silent default is exactly how
portability bugs hide (it "works" on Amarel and silently writes to the wrong
place everywhere else).

    from namo.paths import DATASETS, OUTPUTS, H5, resolve
    key_json = DATASETS / "namo_testset_v1/labels/pure2push.json"
    xml = resolve(label_key)   # map a legacy-baked absolute path onto this box
"""
import os
from pathlib import Path

_HINT = "Run `source env.<machine>.sh` (env.amarel.sh / env.ilab.sh) first — see docs/PORTABILITY.md."


def _require(name):
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"${name} is not set. {_HINT}")
    return Path(v)


def _derive(name, base, *parts):
    """Env override if present, else derived from a resolved base root."""
    v = os.environ.get(name)
    return Path(v) if v else base.joinpath(*parts)


# Base + derived data roots. NAMO_SCRATCH is the keystone — required eagerly,
# since every data dir derives from it.
SCRATCH = _require("NAMO_SCRATCH")
DATASETS = _derive("NAMO_DATASETS", SCRATCH, "datasets")
H5 = _derive("NAMO_H5", SCRATCH, "h5")
MANIFESTS = _derive("NAMO_MANIFESTS", SCRATCH, "manifests")
OUTPUTS = _derive("NAMO_OUTPUTS", SCRATCH, "outputs")
LOGS = _derive("NAMO_LOGS", SCRATCH, "logs")

GLOBAL_SEED = int(os.environ.get("NAMO_GLOBAL_SEED", "42"))

# SAGE_REPO / MJ_PATH are resolved LAZILY (module __getattr__): a script that
# imports only data dirs shouldn't fail for lack of them — they fail loud only
# when actually imported/accessed.
_LAZY = ("SAGE_REPO", "MJ_PATH")


def __getattr__(name):
    if name in _LAZY:
        return _require(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# The absolute prefix older data artifacts (label-JSON xml keys, manifests) were
# written with, on the original Amarel box. resolve() maps it onto this box.
_LEGACY_SCRATCH = "/scratch/dm1487"


def resolve(path):
    """Map a possibly-legacy absolute path onto this box's NAMO_SCRATCH.

    Older artifacts (e.g. namo_testset_v1 label JSON keys) bake in
    ``/scratch/dm1487/...`` absolute paths. Call this at load time so the data
    itself stays path-free and portable. No-op when the path carries no legacy
    prefix (already-relative or already-correct paths pass through unchanged).
    """
    s = str(path)
    # Match the prefix only on a path boundary (exact, or followed by "/") so
    # "/scratch/dm1487-old/..." doesn't get mangled.
    if str(SCRATCH) != _LEGACY_SCRATCH and (s == _LEGACY_SCRATCH or s.startswith(_LEGACY_SCRATCH + "/")):
        return Path(str(SCRATCH) + s[len(_LEGACY_SCRATCH):])
    return Path(s)
