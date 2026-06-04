"""Helpers for selecting environments by initial region-path length."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import namo_rl

from namo.planners import get_region_snapshot


@dataclass
class RegionPathAnalysis:
    """Selection metadata for one XML environment."""

    xml_path: str
    path_length_n: int
    robot_label: Optional[str]
    goal_label: Optional[str]
    adjacency: Dict[str, Set[str]]
    selection_error: Optional[str] = None


def find_shortest_path_length(
    adjacency: Dict[str, Set[str]],
    start: str,
    end: str,
) -> int:
    """Return shortest edge count between two region labels, or -1 if unreachable."""
    if start == end:
        return 0
    if start not in adjacency:
        return -1

    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        current, depth = queue.popleft()
        for neighbor in adjacency.get(current, set()):
            if neighbor == end:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return -1


def get_xml_files(
    input_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """Return XML file paths from a directory or manifest."""
    xml_files: List[str] = []

    if manifest_path:
        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        base_dir = manifest.parent
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                entry = line.strip()
                if not entry or not entry.endswith(".xml"):
                    continue
                if os.path.isabs(entry):
                    xml_files.append(entry)
                else:
                    xml_files.append(str((base_dir / entry).resolve()))
    elif input_dir:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        xml_files = sorted(str(path.resolve()) for path in input_path.glob("*.xml"))
    else:
        raise ValueError("Must provide either input_dir or manifest_path")

    if limit is not None:
        xml_files = xml_files[:limit]

    return xml_files


def analyze_environment_path_length(
    xml_path: str,
    config_path: str,
    *,
    use_cpp_snapshot: bool = True,
) -> RegionPathAnalysis:
    """Compute the initial robot-to-goal region path length for one environment."""
    try:
        env = namo_rl.RLEnvironment(xml_path, config_path, False)
        snapshot = get_region_snapshot(
            env,
            goals_per_region=0,
            local_info_only=False,
            use_cpp_unified=use_cpp_snapshot,
            use_xml_goal=True,
        )
        adjacency = {
            str(region): set(neighbors)
            for region, neighbors in snapshot.get("adjacency", {}).items()
        }
        robot_label = snapshot.get("robot_label") or None
        goal_label = snapshot.get("goal_label") or None

        if not robot_label:
            return RegionPathAnalysis(
                xml_path=xml_path,
                path_length_n=-1,
                robot_label=None,
                goal_label=goal_label,
                adjacency=adjacency,
                selection_error="missing_robot_region",
            )

        if not goal_label:
            return RegionPathAnalysis(
                xml_path=xml_path,
                path_length_n=-1,
                robot_label=robot_label,
                goal_label=None,
                adjacency=adjacency,
                selection_error="missing_goal_region",
            )

        return RegionPathAnalysis(
            xml_path=xml_path,
            path_length_n=find_shortest_path_length(adjacency, robot_label, goal_label),
            robot_label=robot_label,
            goal_label=goal_label,
            adjacency=adjacency,
        )
    except Exception as exc:
        return RegionPathAnalysis(
            xml_path=xml_path,
            path_length_n=-1,
            robot_label=None,
            goal_label=None,
            adjacency={},
            selection_error=str(exc),
        )
