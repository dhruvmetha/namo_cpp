"""sys.path bootstrap so rl_loop can import the sandbox rollout primitives and sage.

The rollout/eval primitives we reuse (scorer_beam, eval_m3, live_scorer) live in
scripts/sandbox and are NOT importable as a package — the whole sandbox uses raw
sys.path insertion (every file there does the same). We mirror that here once so every
rl_loop module can `from scorer_beam import ...` etc.

Worktree note: a forked worktree may have no build_python/*.so of its own (C++ is
unchanged), so we point the build_python entry at the main checkout when the local one
is missing. Everything else is resolved relative to THIS repo.
"""
import os
import sys
from pathlib import Path

# python/namo/rl_loop/_bootstrap.py -> repo root is parents[3]
REPO = Path(__file__).resolve().parents[3]
SAGE = os.environ.get("SAGE_REPO") or str(REPO.parent / "sage_learning")


def _build_python_dir() -> str:
    local = REPO / "build_python"
    if (local / "namo_rl.cpython-311-x86_64-linux-gnu.so").exists() or any(local.glob("namo_rl*.so")):
        return str(local)
    # worktree fallback: the sibling main checkout keeps the built bindings.
    main = REPO.parents[0] if REPO.name.startswith("agent-") else None
    # generic fallback: the canonical checkout path used across the CS estate.
    canonical = Path("/common/home/dm1487/robotics_research/ktamp/namo/build_python")
    if any(canonical.glob("namo_rl*.so")):
        return str(canonical)
    return str(local)


def ensure_paths() -> None:
    entries = [
        _build_python_dir(),
        str(REPO / "python"),
        str(REPO / "scripts"),
        str(REPO / "scripts/sandbox"),
        str(REPO / "scripts/pipeline"),
        SAGE,
    ]
    for p in entries:
        if p and p not in sys.path:
            sys.path.insert(0, p)


ensure_paths()
