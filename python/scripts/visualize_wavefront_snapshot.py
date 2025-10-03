#!/usr/bin/env python3
"""Render a composite visualisation for a saved wavefront snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
from namo.visualization.wavefront_viewer import visualize_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="wavefront_snapshots",
        help="Directory containing the exported snapshot artefacts (default: wavefront_snapshots)",
    )
    parser.add_argument(
        "--prefix",
        default="snapshot",
        help="Filename prefix used during export (default: snapshot)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening an interactive window; only write the summary image to disk",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the generated summary figure (PNG). Defaults to <prefix>_summary.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    directory = Path(args.input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {directory}")

    save_target = Path(args.output) if args.output else None
    output_path = visualize_snapshot(directory, prefix=args.prefix, show=not args.no_show, save_path=save_target)

    print("Snapshot visualisation written to:", output_path)


if __name__ == "__main__":
    main()
