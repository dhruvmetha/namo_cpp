#!/usr/bin/env python3
"""
Generate Filtered Manifest from Oracle Results

Creates a manifest with region:object skip entries based on oracle (search) results.

Usage:
    # Skip failed triplets (default)
    python generate_filtered_manifest.py \
        --base-manifest /path/to/manifest.txt \
        --results-dir /path/to/oracle/results \
        --output /path/to/filtered_manifest.txt

    # For 2-push evaluation: only 2-push envs, skip everything that's not 2-push success
    python generate_filtered_manifest.py \
        --base-manifest /path/to/manifest.txt \
        --results-dir /path/to/oracle/results \
        --output /path/to/filtered_manifest.txt \
        --only-2push-envs --skip-non-2push

Output format (tab-separated):
    /path/to/env.xml\tregion1:obj1,region2:obj2,...

Options:
    --only-2push-envs: Only include envs with at least one 2-push solution
    --skip-non-2push: Skip everything that's NOT a 2-push success (1-push + failed)
"""

import argparse
import pickle
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def load_base_manifest(manifest_path: str) -> List[str]:
    """Load environment paths from base manifest."""
    envs = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Take only the path (ignore any existing skip entries)
            parts = line.split('\t')
            envs.append(parts[0])
    return envs


@dataclass
class TripletResult:
    """Result for a single triplet."""
    success: bool
    chain_depth: int


def load_oracle_results(
    results_dir: str,
) -> Dict[str, Dict[str, TripletResult]]:
    """
    Load oracle results and extract success/chain_depth per (env, region, object) triplet.

    Returns:
        {xml_file: {region:object: TripletResult}}
    """
    results: Dict[str, Dict[str, TripletResult]] = defaultdict(dict)

    pkl_files = glob(f"{results_dir}/**/*.pkl", recursive=True)
    print(f"Found {len(pkl_files)} pkl files in {results_dir}")

    for file in pkl_files:
        if 'collection_summary' in file:
            continue

        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)

            episode_results = data.get('episode_results', [])

            for ep in episode_results:
                xml_file = ep.get('xml_file', '')
                if not xml_file:
                    continue

                alg_stats = ep.get('algorithm_stats', {})
                region_label = alg_stats.get('neighbour_region_label')
                object_id = alg_stats.get('chosen_object_id', '')

                if region_label is None:
                    continue

                chain_depth = alg_stats.get('chain_depth', 0)
                success = ep.get('solution_found', False)

                # Handle no_reachable_objects case (no object_id means skip entire region)
                if not object_id:
                    key = region_label  # Just region, no object - means skip whole region
                else:
                    key = f"{region_label}:{object_id}"

                # Store result (first occurrence wins if duplicates)
                if key not in results[xml_file]:
                    results[xml_file][key] = TripletResult(success=success, chain_depth=chain_depth)

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    return dict(results)


def filter_2push_envs(
    oracle_results: Dict[str, Dict[str, TripletResult]]
) -> Tuple[Dict[str, Dict[str, TripletResult]], int, int]:
    """
    Filter to only include environments that have at least one 2-push solution.

    Returns:
        (filtered_results, num_envs_removed, num_1push_only_envs)
    """
    filtered: Dict[str, Dict[str, TripletResult]] = {}
    envs_removed = 0

    for xml_file, triplets in oracle_results.items():
        has_2push = any(
            r.success and r.chain_depth >= 2
            for r in triplets.values()
        )
        if has_2push:
            filtered[xml_file] = triplets
        else:
            envs_removed += 1

    return filtered, envs_removed, envs_removed


def generate_skip_entries(
    oracle_results: Dict[str, Dict[str, TripletResult]],
    skip_non_2push: bool = False,
) -> Dict[str, List[str]]:
    """
    Generate skip entries for each environment.

    Args:
        oracle_results: {xml_file: {region:object: TripletResult}}
        skip_non_2push: If True, skip everything that's NOT a 2-push success.
                        If False, only skip failed triplets.

    Returns:
        {xml_file: [region:object or region, ...]} for triplets/regions to skip
        - "region:object" skips specific object in region
        - "region" (no colon) skips entire region
    """
    skip_entries: Dict[str, List[str]] = {}

    for xml_file, triplets in oracle_results.items():
        to_skip = []

        if skip_non_2push:
            # Skip everything that's NOT a successful 2-push
            # Group triplets by region
            region_triplets: Dict[str, List[Tuple[str, TripletResult]]] = defaultdict(list)
            region_only_failures: List[str] = []  # Regions with no reachable objects

            for key, result in triplets.items():
                if ':' in key:
                    region, obj = key.split(':', 1)
                    region_triplets[region].append((obj, result))
                else:
                    # Region-only key (no object) - means no_reachable_objects
                    if not result.success:
                        region_only_failures.append(key)

            # Add region-only failures first
            to_skip.extend(region_only_failures)

            for region, objs in region_triplets.items():
                # Find 2-push successes (the only ones we want to keep)
                two_push_objs = [obj for obj, r in objs if r.success and r.chain_depth >= 2]
                non_2push_objs = [obj for obj, r in objs if not (r.success and r.chain_depth >= 2)]

                if len(two_push_objs) == 0:
                    # NO 2-push successes -> skip entire region
                    to_skip.append(region)
                elif len(non_2push_objs) > 0:
                    # Some 2-push, some not -> skip specific non-2push objects
                    for obj in non_2push_objs:
                        to_skip.append(f"{region}:{obj}")
        else:
            # Skip failed triplets only
            to_skip = [key for key, result in triplets.items() if not result.success]

        if to_skip:
            skip_entries[xml_file] = sorted(to_skip)

    return skip_entries


def write_filtered_manifest(
    base_envs: List[str],
    skip_entries: Dict[str, List[str]],
    oracle_results: Dict[str, Dict[str, TripletResult]],
    output_path: str,
    include_only_in_oracle: bool = True,
):
    """Write the filtered manifest."""

    lines_written = 0
    envs_with_skips = 0
    total_skips = 0

    with open(output_path, 'w') as f:
        for env_path in base_envs:
            # Optionally filter to only envs that appear in oracle results
            if include_only_in_oracle and env_path not in oracle_results:
                continue

            skips = skip_entries.get(env_path, [])

            if skips:
                f.write(f"{env_path}\t{','.join(skips)}\n")
                envs_with_skips += 1
                total_skips += len(skips)
            else:
                f.write(f"{env_path}\n")

            lines_written += 1

    return lines_written, envs_with_skips, total_skips


def print_summary(
    oracle_results: Dict[str, Dict[str, TripletResult]],
    skip_entries: Dict[str, List[str]],
    lines_written: int,
    envs_with_skips: int,
    total_skips: int,
    skip_non_2push: bool = False,
    envs_removed: int = 0,
):
    """Print summary statistics."""
    total_triplets = sum(len(v) for v in oracle_results.values())
    successful_triplets = sum(
        sum(1 for r in v.values() if r.success)
        for v in oracle_results.values()
    )
    failed_triplets = total_triplets - successful_triplets

    # Count by chain depth
    depth_1_success = sum(
        sum(1 for r in v.values() if r.success and r.chain_depth == 1)
        for v in oracle_results.values()
    )
    depth_2_success = sum(
        sum(1 for r in v.values() if r.success and r.chain_depth >= 2)
        for v in oracle_results.values()
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Oracle results:")
    print(f"  Environments:      {len(oracle_results)}")
    if envs_removed > 0:
        print(f"  (Removed {envs_removed} envs with no 2-push solutions)")
    print(f"  Total triplets:    {total_triplets}")
    print(f"  Successful:        {successful_triplets} ({successful_triplets/total_triplets*100:.1f}%)")
    print(f"    - 1-push:        {depth_1_success}")
    print(f"    - 2-push:        {depth_2_success}")
    print(f"  Failed:            {failed_triplets} ({failed_triplets/total_triplets*100:.1f}%)")
    print(f"\nOutput manifest:")
    print(f"  Environments:      {lines_written}")
    print(f"  With skip entries: {envs_with_skips}")
    print(f"  Total skip items:  {total_skips}")
    if skip_non_2push:
        print(f"  (Skipping non-2push: 1-push + failed → evaluate only 2-push successes)")
    else:
        print(f"  (Skipping failed triplets only)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate filtered manifest from oracle results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--base-manifest", "-b",
        type=str,
        required=True,
        help="Path to base manifest file"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        required=True,
        help="Directory containing oracle result pkl files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for filtered manifest"
    )
    parser.add_argument(
        "--all-envs",
        action="store_true",
        help="Include all envs from base manifest (not just those in oracle results)"
    )
    parser.add_argument(
        "--only-2push-envs",
        action="store_true",
        help="Only include environments that have at least one 2-push solution"
    )
    parser.add_argument(
        "--skip-non-2push",
        action="store_true",
        help="Skip everything that's NOT a 2-push success (1-push + failed) - for 2-push only evaluation"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.base_manifest).exists():
        raise FileNotFoundError(f"Base manifest not found: {args.base_manifest}")
    if not Path(args.results_dir).exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    # Load data
    print(f"Loading base manifest: {args.base_manifest}")
    base_envs = load_base_manifest(args.base_manifest)
    print(f"  Found {len(base_envs)} environments")

    print(f"\nLoading oracle results: {args.results_dir}")
    oracle_results = load_oracle_results(args.results_dir)
    print(f"  Found results for {len(oracle_results)} environments")

    # Filter to only 2-push envs if requested
    envs_removed = 0
    if args.only_2push_envs:
        print(f"\nFiltering to only environments with 2-push solutions...")
        oracle_results, envs_removed, _ = filter_2push_envs(oracle_results)
        print(f"  Kept {len(oracle_results)} environments (removed {envs_removed})")

    # Generate skip entries
    skip_entries = generate_skip_entries(oracle_results, skip_non_2push=args.skip_non_2push)

    # Write output
    print(f"\nWriting filtered manifest: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    lines_written, envs_with_skips, total_skips = write_filtered_manifest(
        base_envs,
        skip_entries,
        oracle_results,
        args.output,
        include_only_in_oracle=not args.all_envs,
    )

    print_summary(
        oracle_results,
        skip_entries,
        lines_written,
        envs_with_skips,
        total_skips,
        skip_non_2push=args.skip_non_2push,
        envs_removed=envs_removed,
    )

    print(f"Done! Filtered manifest saved to: {args.output}")


if __name__ == "__main__":
    main()
