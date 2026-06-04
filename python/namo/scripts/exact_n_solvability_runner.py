#!/usr/bin/env python3
"""CLI wrapper for the exact-n Full NAMO solvability runner."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "python"))

from namo.core.binding_loader import load_canonical_namo_rl

namo_rl, module_path, expected_build = load_canonical_namo_rl(project_root)

from namo.solvability_runner import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
