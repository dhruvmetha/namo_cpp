#!/usr/bin/env python3
"""Enforce strict canonical namo_rl binding policy in active source paths."""

from __future__ import annotations

import re
from pathlib import Path

BANNED_LITERAL_PATTERNS = [
    "build_python_local",
    "build_python_mjxrl",
    "build_python_mjxrl_",
]

BANNED_REGEX_PATTERNS = [
    re.compile(r"build_python\*"),
    re.compile(r"glob\([^\n\)]*build_python[^\n\)]*\*"),
]

# Only scan active source and docs where runtime policy is documented.
TARGET_GLOBS = [
    "python/**/*.py",
    "python/**/*.md",
    "python/setup.py",
    "build_python_bindings.sh",
    "DATA_COLLECTION_GUIDE.md",
    "IDFS_DATA_PIPELINE.md",
    "MCTS_DATA_PIPELINE.md",
]

EXCLUDE_SUBSTRINGS = [
    "/build/",
    "/build_python/",
    "/build_python_local/",
    "/build_wavefront_",
    "/.git/",
    "/python/namo.egg-info/",
]


def should_skip(path: Path) -> bool:
    text = str(path)
    return any(token in text for token in EXCLUDE_SUBSTRINGS)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []

    seen: set[Path] = set()
    for pattern in TARGET_GLOBS:
        for file_path in repo_root.glob(pattern):
            if file_path in seen or not file_path.is_file() or should_skip(file_path):
                continue
            if file_path.name == "check_canonical_binding_policy.py":
                continue
            seen.add(file_path)

            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for banned in BANNED_LITERAL_PATTERNS:
                if banned in content:
                    offenders.append(f"{file_path}: contains '{banned}'")
            for regex in BANNED_REGEX_PATTERNS:
                if regex.search(content):
                    offenders.append(f"{file_path}: matches /{regex.pattern}/")

    if offenders:
        print("Canonical binding policy violations found:")
        for item in offenders:
            print(f"  - {item}")
        return 1

    print("Canonical binding policy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
