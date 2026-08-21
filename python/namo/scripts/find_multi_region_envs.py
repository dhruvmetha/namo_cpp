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
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "python"))
from namo.core.binding_loader import load_canonical_namo_rl
from namo.environment_selection import analyze_environment_path_length, get_xml_files

namo_rl, module_path, expected_build = load_canonical_namo_rl(project_root)


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
        default="config/namo_config_complete_skill15_car_1x.yaml",
        help="Path to the canonical car 1x d5 NAMO config",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.02,
        help="Deprecated compatibility flag; unified snapshot path ignores this value",
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
        analysis = analyze_environment_path_length(
            xml_path,
            config_path,
            use_cpp_snapshot=True,
        )
        path_length = analysis.path_length_n
        adjacency = analysis.adjacency
        robot_label = analysis.robot_label
        goal_label = analysis.goal_label

        if analysis.selection_error:
            stats["errors"] += 1
            if args.verbose:
                print(f"  Error: {analysis.selection_error}")
            continue

        if path_length == -1:
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
