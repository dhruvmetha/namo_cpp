#!/usr/bin/env python3
"""Validate unified reachability summary against legacy per-query APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _default_paths(repo_root: Path) -> tuple[str, str]:
    return (
        str(repo_root / "data" / "test_scene.xml"),
        str(repo_root / "config" / "benchmark_config.yaml"),
    )


def _inject_python_paths(repo_root: Path) -> None:
    if str(repo_root / "python") not in sys.path:
        sys.path.insert(0, str(repo_root / "python"))

    build_dirs = [repo_root / "build_python"]
    build_dirs.extend(sorted(repo_root.glob("build_python*")))
    for build_dir in build_dirs:
        if not build_dir.is_dir():
            continue
        if any(build_dir.glob("namo_rl*.so")) and str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
            return


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check unified wavefront reachability summary consistency."
    )
    parser.add_argument("--xml", type=str, default=None, help="Path to XML scene")
    parser.add_argument("--config", type=str, default=None, help="Path to NAMO config")
    parser.add_argument("--goal-x", type=float, default=3.0, help="Goal X (meters)")
    parser.add_argument("--goal-y", type=float, default=2.4, help="Goal Y (meters)")
    parser.add_argument("--goal-theta", type=float, default=0.0, help="Goal theta (radians)")
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Also request analysis-mode summary with reachable edge indices",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    default_xml, default_config = _default_paths(repo_root)
    xml_path = args.xml or default_xml
    config_path = args.config or default_config

    # Ensure local package/module resolution from source tree.
    _inject_python_paths(repo_root)

    import namo_rl  # pylint: disable=import-error

    env = namo_rl.RLEnvironment(xml_path, config_path, False)
    env.set_robot_goal(args.goal_x, args.goal_y, args.goal_theta)

    summary = env.get_reachability_summary(False)
    analysis_summary = env.get_reachability_summary(True) if args.analysis else None

    mismatches: list[str] = []
    legacy_goal = env.is_robot_goal_reachable()
    if bool(summary.get("goal_reachable", False)) != bool(legacy_goal):
        mismatches.append(
            f"goal_reachable mismatch: summary={summary.get('goal_reachable')} legacy={legacy_goal}"
        )

    objects = summary.get("objects", {})
    for obj_name, stats in objects.items():
        legacy_edges = env.get_reachable_edges(obj_name)
        summary_edges = int(stats.get("reachable_edges", -1))
        if len(legacy_edges) != summary_edges:
            mismatches.append(
                f"{obj_name}: reachable_edges mismatch summary={summary_edges} legacy={len(legacy_edges)}"
            )

        summary_reachable = bool(stats.get("reachable", False))
        if summary_reachable != (len(legacy_edges) > 0):
            mismatches.append(
                f"{obj_name}: reachable flag mismatch summary={summary_reachable} legacy={len(legacy_edges) > 0}"
            )

    if analysis_summary is not None:
        for obj_name, stats in analysis_summary.get("objects", {}).items():
            if "reachable_edge_indices" not in stats:
                mismatches.append(f"{obj_name}: analysis summary missing reachable_edge_indices")

    result = {
        "xml_path": xml_path,
        "config_path": config_path,
        "goal": [args.goal_x, args.goal_y, args.goal_theta],
        "goal_reachable": summary.get("goal_reachable", False),
        "object_count": len(objects),
        "reachable_objects": sum(1 for s in objects.values() if s.get("reachable", False)),
        "mismatches": mismatches,
    }

    print(json.dumps(result, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
