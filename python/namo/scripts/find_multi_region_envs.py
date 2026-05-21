#!/usr/bin/env python3
"""Find XML environments with at least N regions between robot and goal.

This script scans XML files and identifies environments where the robot
must traverse through multiple regions to reach the goal. Useful for
finding challenging test cases for visualization and debugging.

Usage:
    python python/namo/scripts/find_multi_region_envs.py \
        --input-dir /path/to/xml/files \
        --output-file multi_region_envs.txt \
        --min-regions 2

    # Or with a manifest file:
    python python/namo/scripts/find_multi_region_envs.py \
        --manifest /path/to/manifest.txt \
        --output-file multi_region_envs.txt \
        --min-regions 2
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "python"))
from namo.core.binding_loader import load_canonical_namo_rl
from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter

namo_rl, module_path, expected_build = load_canonical_namo_rl(project_root)


def find_shortest_path_length(
    adjacency: Dict[str, Set[str]], start: str, end: str
) -> int:
    """Find shortest path length between two regions using BFS.

    Returns the number of regions traversed (edges), or -1 if unreachable.
    A direct connection returns 1, going through one intermediate region returns 2, etc.
    """
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

    return -1  # Unreachable


def analyze_environment(
    xml_path: str,
    config_path: str,
    resolution: float = 0.02,
) -> Tuple[int, Dict[str, Set[str]], str, str]:
    """Analyze an environment and return the region path length from robot to goal.

    Returns:
        Tuple of (path_length, adjacency_dict, robot_label, goal_label)
        path_length is -1 if unreachable, 0 if same region, N for N regions to traverse
    """
    try:
        # Create environment (visualize=False for headless)
        env = namo_rl.RLEnvironment(xml_path, config_path, False)

        # Create exporter and build snapshot
        exporter = WavefrontSnapshotExporter(env, resolution=resolution)
        snapshot = exporter.build_snapshot(
            xml_path=xml_path,
            config_path=config_path,
            goal_radius=None,  # auto: sqrt(hx^2 + hy^2) + tier1_margin
            goals_per_region=0,
            use_current_state=False,
        )

        # Get adjacency and region labels
        adjacency: Dict[str, Set[str]] = {
            region: set(neighbours) for region, neighbours in snapshot.adjacency.items()
        }
        region_labels = dict(snapshot.region_labels)

        # Find robot and goal labels
        robot_label = None
        goal_label = None
        for label in region_labels.values():
            if "robot" in label.lower():
                robot_label = label
            if "goal" in label.lower():
                goal_label = label

        if robot_label is None or goal_label is None:
            return -1, adjacency, robot_label or "unknown", goal_label or "unknown"

        # Find shortest path
        path_length = find_shortest_path_length(adjacency, robot_label, goal_label)

        return path_length, adjacency, robot_label, goal_label

    except Exception as e:
        print(f"  Error analyzing {xml_path}: {e}")
        return -1, {}, "error", "error"


def get_xml_files(
    input_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """Get list of XML files to analyze."""
    xml_files = []

    if manifest_path:
        # Read from manifest file
        manifest = Path(manifest_path)
        if not manifest.exists():
            print(f"Error: Manifest file not found: {manifest_path}")
            sys.exit(1)

        base_dir = manifest.parent
        with open(manifest, "r") as f:
            for line in f:
                line = line.strip()
                if line and line.endswith(".xml"):
                    # Handle both absolute and relative paths
                    if os.path.isabs(line):
                        xml_files.append(line)
                    else:
                        xml_files.append(str(base_dir / line))

    elif input_dir:
        # Scan directory for XML files
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)

        xml_files = sorted([str(f) for f in input_path.glob("*.xml")])

    else:
        print("Error: Must provide either --input-dir or --manifest")
        sys.exit(1)

    if limit:
        xml_files = xml_files[:limit]

    return xml_files


def main():
    parser = argparse.ArgumentParser(
        description="Find XML environments with multiple regions between robot and goal"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing XML environment files",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to manifest file listing XML files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="multi_region_envs.txt",
        help="Output file to write matching environment paths (default: multi_region_envs.txt)",
    )
    parser.add_argument(
        "--min-regions",
        type=int,
        default=2,
        help="Minimum number of regions between robot and goal (default: 2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/namo_config_complete_skill15.yaml",
        help="Path to NAMO config file (default: config/namo_config_complete_skill15.yaml)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.02,
        help="Grid resolution for region analysis (default: 0.02)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to analyze (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each environment",
    )

    args = parser.parse_args()

    # Get config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(project_root / config_path)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Get XML files to analyze
    xml_files = get_xml_files(args.input_dir, args.manifest, args.limit)
    print(f"Found {len(xml_files)} XML files to analyze")

    # Analyze each environment
    matching_envs = []
    stats = {
        "total": 0,
        "errors": 0,
        "unreachable": 0,
        "by_path_length": {},
    }

    for i, xml_path in enumerate(xml_files):
        if (i + 1) % 100 == 0 or args.verbose:
            print(f"Analyzing {i + 1}/{len(xml_files)}: {os.path.basename(xml_path)}")

        stats["total"] += 1
        path_length, adjacency, robot_label, goal_label = analyze_environment(
            xml_path, config_path, args.resolution
        )

        if path_length == -1:
            if robot_label == "error":
                stats["errors"] += 1
            else:
                stats["unreachable"] += 1
            continue

        # Track path length distribution
        stats["by_path_length"][path_length] = stats["by_path_length"].get(path_length, 0) + 1

        if args.verbose:
            print(f"  Path length: {path_length} regions")
            print(f"  Adjacency: {adjacency}")

        if path_length >= args.min_regions:
            matching_envs.append((xml_path, path_length, len(adjacency)))
            if args.verbose:
                print(f"  ✓ MATCHES (>= {args.min_regions} regions)")

    # Sort by path length (descending) then by number of regions
    matching_envs.sort(key=lambda x: (-x[1], -x[2]))

    # Write output
    output_path = args.output_file
    if not os.path.isabs(output_path):
        output_path = str(project_root / output_path)

    with open(output_path, "w") as f:
        for xml_path, path_length, num_regions in matching_envs:
            f.write(f"{xml_path}\n")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total environments analyzed: {stats['total']}")
    print(f"Errors: {stats['errors']}")
    print(f"Unreachable (no path): {stats['unreachable']}")
    print(f"\nPath length distribution:")
    for length in sorted(stats["by_path_length"].keys()):
        count = stats["by_path_length"][length]
        marker = " ✓" if length >= args.min_regions else ""
        print(f"  {length} region(s): {count} environments{marker}")
    print(f"\nMatching environments (>= {args.min_regions} regions): {len(matching_envs)}")
    print(f"Output written to: {output_path}")

    # Print a few examples
    if matching_envs:
        print(f"\nTop 5 examples:")
        for xml_path, path_length, num_regions in matching_envs[:5]:
            print(f"  {os.path.basename(xml_path)}: {path_length} regions, {num_regions} total regions")


if __name__ == "__main__":
    main()
