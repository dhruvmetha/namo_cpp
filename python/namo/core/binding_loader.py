"""Canonical namo_rl binding loader utilities.

Strict policy:
- Only <repo>/build_python is accepted as namo_rl source.
- No fallback to alternate build directories.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Tuple


def _build_instructions() -> str:
    return (
        "  cmake -S . -B build_python -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON\n"
        "  cmake --build build_python --target namo_rl -j$(nproc)"
    )


def resolve_repo_root(anchor_file: Path, levels_up: int = 2) -> Path:
    """Resolve NAMO repo root from an anchor file path."""
    return anchor_file.resolve().parents[levels_up]


def ensure_python_and_build_paths(repo_root: Path) -> Path:
    """Ensure canonical python and build paths are present on sys.path."""
    python_dir = repo_root / "python"
    build_dir = (repo_root / "build_python").resolve()

    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))

    if not build_dir.is_dir() or not any(build_dir.glob("namo_rl*.so")):
        raise RuntimeError(
            "Canonical namo_rl build missing at build_python. Build with:\n"
            f"{_build_instructions()}"
        )

    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))

    return build_dir


def assert_canonical_namo_rl(module: object, expected_build_dir: Path) -> Path:
    """Assert imported namo_rl module is loaded from canonical build dir."""
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if expected_build_dir not in module_path.parents:
        raise RuntimeError(
            "Loaded namo_rl from non-canonical path.\n"
            f"  loaded:   {module_path}\n"
            f"  expected: {expected_build_dir}\n"
            "Fix PYTHONPATH so build_python is first."
        )
    return module_path


def load_canonical_namo_rl(repo_root: Path) -> Tuple[ModuleType, Path, Path]:
    """Load namo_rl from canonical build directory and validate provenance."""
    expected_build_dir = ensure_python_and_build_paths(repo_root).resolve()
    namo_rl = importlib.import_module("namo_rl")
    module_path = assert_canonical_namo_rl(namo_rl, expected_build_dir)
    return namo_rl, module_path, expected_build_dir
