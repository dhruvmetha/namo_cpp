#!/usr/bin/env python3
"""Compatibility entry point for the maintained policy evaluator."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "rl_loop" / "eval_policy.py"),
                   run_name="__main__")
