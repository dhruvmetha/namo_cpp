#!/usr/bin/env python3
"""Update physics parameters in nov28 XML files to match aug9 physics.

Changes applied:
- Obstacle friction: 1.0 0.005 0.001 -> 0.0 0.005 0.001 (frictionless)
- Obstacle mass: 0.2 -> 0.1
- Robot torsional friction: 0.001 -> 0.0001
- Obstacle height (z-size): 0.2 -> 0.3
- Obstacle z-position: 0.2 -> 0.3
- Robot z-position: 0.2 -> 0.15

Usage:
    python update_xml_physics.py --input-dir /path/to/nov28 --dry-run
    python update_xml_physics.py --input-dir /path/to/nov28
    python update_xml_physics.py --input-dir /path/to/nov28 --workers 16
"""

import argparse
import glob
import os
import re
import tempfile
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


def update_xml_physics(xml_content: str) -> str:
    """Update physics parameters in XML content to match aug9 physics.

    Args:
        xml_content: Original XML file content

    Returns:
        Updated XML content with aug9 physics parameters
    """
    updated = xml_content

    # 1. Update obstacle friction: 1.0 0.005 0.001 -> 0.0 0.005 0.001
    # Match movable obstacle geoms and update their friction
    # Pattern matches: friction="1.0 0.005 0.001" in obstacle lines
    updated = re.sub(
        r'(<geom name="obstacle_\d+_movable"[^>]*friction=")1\.0 0\.005 0\.001(")',
        r'\g<1>0.0 0.005 0.001\2',
        updated
    )

    # 2. Update obstacle mass: 0.2 -> 0.1
    updated = re.sub(
        r'(<geom name="obstacle_\d+_movable"[^>]*mass=")0\.2(")',
        r'\g<1>0.1\2',
        updated
    )

    # 3. Update robot torsional friction: 1.0 0.005 0.001 -> 1.0 0.005 0.0001
    # Match robot geom specifically
    updated = re.sub(
        r'(<geom name="robot"[^>]*friction=")1\.0 0\.005 0\.001(")',
        r'\g<1>1.0 0.005 0.0001\2',
        updated
    )

    # 4. Update obstacle z-size: 0.2 -> 0.3
    # This is trickier because size has 3 components: x y z
    # We need to change the last component from 0.2 to 0.3
    def update_obstacle_size(match):
        prefix = match.group(1)
        x_size = match.group(2)
        y_size = match.group(3)
        suffix = match.group(4)
        return f'{prefix}{x_size} {y_size} 0.3{suffix}'

    updated = re.sub(
        r'(<geom name="obstacle_\d+_movable"[^>]*size=")([0-9.]+) ([0-9.]+) 0\.2(")',
        update_obstacle_size,
        updated
    )

    # 5. Update obstacle z-position: 0.2 -> 0.3
    def update_obstacle_pos(match):
        prefix = match.group(1)
        x_pos = match.group(2)
        y_pos = match.group(3)
        suffix = match.group(4)
        return f'{prefix}{x_pos} {y_pos} 0.3{suffix}'

    updated = re.sub(
        r'(<geom name="obstacle_\d+_movable"[^>]*pos=")([0-9.-]+) ([0-9.-]+) 0\.2(")',
        update_obstacle_pos,
        updated
    )

    # 6. Update robot z-position: 0.2 -> 0.15
    # Robot geom has pos="x y 0.2" - update the z component
    def update_robot_pos(match):
        prefix = match.group(1)
        x_pos = match.group(2)
        y_pos = match.group(3)
        suffix = match.group(4)
        return f'{prefix}{x_pos} {y_pos} 0.15{suffix}'

    updated = re.sub(
        r'(<geom name="robot"[^>]*pos=")([0-9.-]+) ([0-9.-]+) 0\.2(")',
        update_robot_pos,
        updated
    )

    return updated


def process_xml_file(args: tuple) -> dict:
    """Process a single XML file (multiprocessing compatible).

    Args:
        args: Tuple of (xml_path, dry_run, input_dir, output_dir)
              If output_dir is None, modifies files in place.
              If output_dir is set, writes updated files there preserving directory structure.

    Returns:
        Dict with processing results
    """
    xml_path, dry_run, input_dir, output_dir = args

    with open(xml_path, 'r') as f:
        original_content = f.read()

    updated_content = update_xml_physics(original_content)

    changed = original_content != updated_content

    if changed and not dry_run:
        if output_dir:
            # Write to output directory, preserving relative path structure
            rel_path = os.path.relpath(xml_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)

            # Create parent directories if needed
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, 'w') as f:
                f.write(updated_content)
        else:
            # In-place modification using safe write
            import shutil
            fd, temp_path = tempfile.mkstemp(suffix='.xml')
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(updated_content)
                shutil.copy2(temp_path, xml_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    return {
        'path': xml_path,
        'changed': changed,
        'dry_run': dry_run
    }


def main():
    parser = argparse.ArgumentParser(
        description="Update physics parameters in XML files to match aug9 physics"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/common/users/shared/robot_learning/dm1487/namo/mj_env_configs/nov28",
        help="Directory containing XML files to update"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes, just report what would be changed"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details for each file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPUs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for updated files (preserves directory structure). If not set, modifies files in place."
    )

    args = parser.parse_args()

    # Setup
    xml_pattern = os.path.join(args.input_dir, "**", "*.xml")

    if args.dry_run:
        print("DRY RUN - no files will be modified")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

    print("\nPhysics changes to apply:")
    print("  - Obstacle friction: 1.0 0.005 0.001 -> 0.0 0.005 0.001")
    print("  - Obstacle mass: 0.2 -> 0.1")
    print("  - Robot torsional friction: 1.0 0.005 0.001 -> 1.0 0.005 0.0001")
    print("  - Obstacle z-size: 0.2 -> 0.3")
    print("  - Obstacle z-position: 0.2 -> 0.3")
    print("  - Robot z-position: 0.2 -> 0.15")
    print()

    # Determine number of workers
    num_workers = args.workers if args.workers else cpu_count()
    print(f"Using {num_workers} parallel workers")
    print("Streaming file discovery and processing...\n")

    # Process files in parallel with streaming file discovery
    changed_count = 0
    unchanged_count = 0
    total_count = 0
    changed_files = []

    # Use a generator to stream task arguments
    def task_generator():
        for xml_path in glob.iglob(xml_pattern, recursive=True):
            yield (xml_path, args.dry_run, args.input_dir, args.output_dir)

    pool = None
    try:
        pool = Pool(processes=num_workers)
        # Use imap_unordered with chunksize for better performance with many files
        chunksize = 100  # Process files in batches for efficiency
        with tqdm(desc="Processing XML files", unit="file") as pbar:
            for result in pool.imap_unordered(process_xml_file, task_generator(), chunksize=chunksize):
                total_count += 1
                if result['changed']:
                    changed_count += 1
                    changed_files.append(result['path'])
                else:
                    unchanged_count += 1
                pbar.update(1)
                pbar.set_postfix(changed=changed_count, unchanged=unchanged_count)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\n\nInterrupted! Cleaning up workers...")
        if pool is not None:
            pool.terminate()
            pool.join()
        print(f"Processed {total_count} files before interrupt")
        print(f"  Changed: {changed_count}, Unchanged: {unchanged_count}")
        return 1
    finally:
        if pool is not None:
            pool.terminate()

    if total_count == 0:
        print(f"No XML files found in {args.input_dir}")
        return 1

    # Print verbose output after parallel processing
    if args.verbose and changed_files:
        action = "Would modify" if args.dry_run else "Modified"
        for path in sorted(changed_files):
            print(f"  {action}: {path}")

    # Summary
    print(f"\nSummary:")
    print(f"  Files processed: {total_count}")
    print(f"  Files {'that would be ' if args.dry_run else ''}changed: {changed_count}")
    print(f"  Files unchanged: {unchanged_count}")

    if args.dry_run and changed_count > 0:
        print(f"\nRun without --dry-run to apply changes")

    return 0


if __name__ == "__main__":
    exit(main())
