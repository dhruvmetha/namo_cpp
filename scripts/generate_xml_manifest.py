#!/usr/bin/env python3
"""Generate a manifest file listing all XML environment files.

This creates a sorted, shuffled (with fixed seed) list of XML files that can be
used by modular_parallel_collection.py for consistent ordering across runs.

CLI Usage:
    python generate_xml_manifest.py --input-dir /path/to/xmls --output manifest.txt
    python generate_xml_manifest.py --input-dir /path/to/xmls --output manifest.txt --no-shuffle
    python generate_xml_manifest.py --input-dir /path/to/xmls --output manifest.txt --check-unique

Programmatic Usage:
    from scripts.generate_xml_manifest import create_manifest, discover_xml_files

    # From a list of files:
    files = ['/path/to/env1.xml', '/path/to/env2.xml', ...]
    result = create_manifest(files, '/path/to/manifest.txt')

    # Or discover files from a directory:
    files = discover_xml_files('/path/to/xmls')
    result = create_manifest(files, '/path/to/manifest.txt')
"""

import argparse
import glob
import os
import random
from collections import Counter

from tqdm import tqdm


def discover_xml_files(input_dir: str, verbose: bool = True) -> list:
    """Discover all XML files in a directory recursively.

    Args:
        input_dir: Base directory to scan
        verbose: Whether to print progress

    Returns:
        List of XML file paths (excludes _temp.xml files)
    """
    xml_pattern = os.path.join(input_dir, "**", "*.xml")
    if verbose:
        print(f"Scanning for XML files in {input_dir}...")

    all_files = []
    iterator = glob.iglob(xml_pattern, recursive=True)
    if verbose:
        iterator = tqdm(iterator, desc="Finding files", unit="file")

    for xml_path in iterator:
        all_files.append(xml_path)

    if verbose:
        print(f"Found {len(all_files)} XML files")

    return all_files


def create_manifest(
    files: list,
    output_path: str,
    seed: int = 42,
    shuffle: bool = True,
    check_unique: bool = True,
    verbose: bool = True
) -> dict:
    """Create a manifest file from a list of files.

    Args:
        files: List of file paths to include in the manifest
        output_path: Path to write the manifest file
        seed: Random seed for shuffling (default: 42)
        shuffle: Whether to shuffle the file list (default: True)
        check_unique: Whether to check for duplicate filenames (default: True)
        verbose: Whether to print progress messages (default: True)

    Returns:
        dict with keys:
            - 'total_files': Number of files in manifest
            - 'duplicates': Dict of duplicate filenames (if check_unique=True)
            - 'output_path': Path to the created manifest
    """
    result = {
        'total_files': 0,
        'duplicates': {},
        'output_path': output_path
    }

    # Filter out temp files
    all_files = [f for f in files if not f.endswith('_temp.xml')]

    if verbose:
        print(f"Processing {len(all_files)} files (excluding _temp.xml)")

    if not all_files:
        if verbose:
            print("No files to process!")
        return result

    # Check for duplicate full paths and deduplicate
    if check_unique:
        if verbose:
            print("Checking for duplicate paths...")
        path_counts = Counter(all_files)
        duplicates = {p: count for p, count in path_counts.items() if count > 1}
        result['duplicates'] = duplicates

        if duplicates and verbose:
            print(f"\nWARNING: Found {len(duplicates)} duplicate paths:")
            for path, count in sorted(duplicates.items())[:10]:
                print(f"  {path}: {count} occurrences")
            if len(duplicates) > 10:
                print(f"  ... and {len(duplicates) - 10} more duplicates")

        # Deduplicate the list
        all_files = list(dict.fromkeys(all_files))
        if verbose:
            print(f"After deduplication: {len(all_files)} unique files")

    # Sort for consistent base ordering
    if verbose:
        print("Sorting files...")
    all_files.sort()

    # Shuffle with fixed seed for reproducibility
    if shuffle:
        if verbose:
            print(f"Shuffling with seed={seed}...")
        random.seed(seed)
        random.shuffle(all_files)

    # Write manifest
    if verbose:
        print(f"Writing manifest to {output_path}...")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        for file_path in all_files:
            f.write(file_path + '\n')

    result['total_files'] = len(all_files)

    if verbose:
        print(f"\nManifest created: {output_path}")
        print(f"Total files: {len(all_files)}")

    return result


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
        "--check-unique",
        action="store_true",
        help="Check for duplicate paths and deduplicate"
    )

    args = parser.parse_args()

    # Discover files
    all_files = discover_xml_files(args.input_dir, verbose=True)
    if not all_files:
        print("No XML files found!")
        return 1

    # Create manifest
    result = create_manifest(
        files=all_files,
        output_path=args.output,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        check_unique=args.check_unique,
        verbose=True
    )

    return 0 if result['total_files'] > 0 else 1


if __name__ == "__main__":
    exit(main())
