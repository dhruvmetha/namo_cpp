#!/usr/bin/env python3
"""Smoke check that namo_rl loads from canonical build_python."""

from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
python_dir = repo_root / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

from namo.core.binding_loader import load_canonical_namo_rl


def main() -> int:
    namo_rl, module_path, expected_build = load_canonical_namo_rl(repo_root)
    payload = {
        "repo_root": str(repo_root.resolve()),
        "expected_build": str(expected_build),
        "loaded_namo_rl": str(module_path),
        "module_name": getattr(namo_rl, "__name__", "namo_rl"),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
