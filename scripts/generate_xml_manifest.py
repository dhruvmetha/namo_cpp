#!/usr/bin/env python3
"""Generate a manifest file listing all XML environment files.

This creates a sorted, shuffled (with fixed seed) list of XML files that can be
used by modular_parallel_collection.py for consistent ordering across runs.

Usage:
    python generate_xml_manifest.py --input-dir /path/to/xmls --output manifest.txt
    python generate_xml_manifest.py --input-dir /path/to/xmls --output manifest.txt --no-shuffle
    python generate_xml_manifest.py --input-dir /path/to/xmls --output manifest.txt --check-unique
"""

import argparse
import glob
import os
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


def check_not_temp(xml_path: str) -> str:
    """Return path if not a temp file, else None."""
    if xml_path.endswith('_temp.xml'):
        return None
    return xml_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest of XML environment files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Base directory containing XML files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output manifest file path"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the file list (keep sorted order)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for filtering (default: CPU count)"
    )
    parser.add_argument(
        "--check-unique",
        action="store_true",
        help="Check for duplicate filenames and report them"
    )

    args = parser.parse_args()

    # Find all XML files
    xml_pattern = os.path.join(args.input_dir, "**", "*.xml")
    print(f"Scanning for XML files in {args.input_dir}...")

    # Stream discovery
    all_files = []
    for xml_path in tqdm(glob.iglob(xml_pattern, recursive=True), desc="Finding files", unit="file"):
        if not xml_path.endswith('_temp.xml'):
            all_files.append(xml_path)

    print(f"Found {len(all_files)} XML files (excluding _temp.xml)")

    if not all_files:
        print("No XML files found!")
        return 1

    # Check for duplicate filenames
    if args.check_unique:
        print("Checking for duplicate filenames...")
        basename_to_paths = defaultdict(list)
        for path in all_files:
            basename_to_paths[os.path.basename(path)].append(path)

        duplicates = {k: v for k, v in basename_to_paths.items() if len(v) > 1}
        if duplicates:
            print(f"\nWARNING: Found {len(duplicates)} duplicate filenames:")
            for basename, paths in sorted(duplicates.items())[:10]:  # Show first 10
                print(f"  {basename}: {len(paths)} occurrences")
                for p in paths[:3]:  # Show first 3 paths
                    print(f"    - {p}")
                if len(paths) > 3:
                    print(f"    ... and {len(paths) - 3} more")
            if len(duplicates) > 10:
                print(f"  ... and {len(duplicates) - 10} more duplicate filenames")
            print(f"\nTotal unique filenames: {len(basename_to_paths)}")
        else:
            print(f"All {len(all_files)} files have unique names")

    # Sort for consistent base ordering
    print("Sorting files...")
    all_files.sort()

    # Shuffle with fixed seed for reproducibility
    if not args.no_shuffle:
        print(f"Shuffling with seed={args.seed}...")
        random.seed(args.seed)
        random.shuffle(all_files)

    # Write manifest
    print(f"Writing manifest to {args.output}...")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        for xml_path in tqdm(all_files, desc="Writing manifest", unit="file"):
            f.write(xml_path + '\n')

    print(f"\nManifest created: {args.output}")
    print(f"Total files: {len(all_files)}")

    return 0


if __name__ == "__main__":
    exit(main())
