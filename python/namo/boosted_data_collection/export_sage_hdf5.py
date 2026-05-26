#!/usr/bin/env python3
"""Export boosted manifests to sage-learning-compatible HDF5."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from namo.boosted_data_collection.episode_builder import iter_training_episodes
from namo.boosted_data_collection.schema import SCHEMA_NAME


def _discover_manifest_files(input_dir: str, pattern: str) -> List[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def _load_manifest(path: Path) -> Mapping[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        manifest = pickle.load(f)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Boosted manifest at {path} is not a mapping")
    return manifest


def _load_visualization_components():
    from namo.visualization.mask_generation import NAMODataVisualizer
    from namo.visualization.mask_generation.batch_collection import HDF5Writer, process_episode

    return NAMODataVisualizer, HDF5Writer, process_episode


def export_manifests_to_hdf5(
    manifest_files: Iterable[Path],
    output_hdf5: str,
    *,
    local_only: bool,
    local_crop_size: float,
    sample_policy: str,
    max_horizon: Optional[int],
    max_cells_per_object: Optional[int],
    max_samples_per_cell: Optional[int],
    verbose: bool,
) -> Dict[str, Any]:
    visualizer_cls, hdf5_writer_cls, process_episode_fn = _load_visualization_components()
    visualizer = visualizer_cls(figsize=(10, 8))
    written = 0
    seen_manifests = 0
    summary: Dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "input_manifests": [],
        "episodes_written": 0,
        "skipped_samples": 0,
    }

    with hdf5_writer_cls(output_hdf5) as writer:
        for manifest_path in manifest_files:
            manifest = _load_manifest(manifest_path)
            seen_manifests += 1
            summary["input_manifests"].append(str(manifest_path))

            for episode in iter_training_episodes(
                manifest,
                sample_policy=sample_policy,
                max_horizon=max_horizon,
                max_cells_per_object=max_cells_per_object,
                max_samples_per_cell=max_samples_per_cell,
            ):
                masks, metadata = process_episode_fn(
                    episode,
                    visualizer,
                    generate_local=True,
                    local_only=local_only,
                    local_crop_size=local_crop_size,
                    use_highres=True,
                )
                if not masks or metadata is None:
                    summary["skipped_samples"] += 1
                    continue

                writer.add_sample(masks, metadata)
                written += 1
                if verbose:
                    print(f"[sample] {episode.get('episode_id')} -> written")

    summary["episodes_written"] = written
    summary["manifest_count"] = seen_manifests
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export boosted manifests to sage-learning HDF5")
    parser.add_argument("--input-dir", required=True, help="Directory containing boosted manifests")
    parser.add_argument("--output-hdf5", required=True, help="Output HDF5 path")
    parser.add_argument(
        "--pattern",
        default="*_boosted_manifest.pkl*",
        help="Recursive filename pattern for boosted manifests",
    )
    parser.add_argument(
        "--sample-policy",
        choices=["canonical", "all"],
        default="canonical",
        help="How many provenance chains to emit per opened cell",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=2,
        help="Maximum action-chain horizon to export (default: 2)",
    )
    parser.add_argument(
        "--max-cells-per-object",
        type=int,
        default=None,
        help="Cap the number of opened cells exported per object",
    )
    parser.add_argument(
        "--max-samples-per-cell",
        type=int,
        default=1,
        help="Cap the number of provenance chains exported per opened cell",
    )
    parser.add_argument(
        "--local-crop-size",
        type=float,
        default=5.0,
        help="Crop size in meters for local mask generation",
    )
    parser.add_argument(
        "--include-global",
        action="store_true",
        help="Include global masks in addition to local masks",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on input manifest count")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    manifest_files = _discover_manifest_files(args.input_dir, args.pattern)
    if args.max_files is not None:
        manifest_files = manifest_files[: max(0, int(args.max_files))]
    if not manifest_files:
        raise FileNotFoundError(
            f"No boosted manifests found under {args.input_dir} matching pattern {args.pattern!r}"
        )

    summary = export_manifests_to_hdf5(
        manifest_files,
        args.output_hdf5,
        local_only=not bool(args.include_global),
        local_crop_size=float(args.local_crop_size),
        sample_policy=str(args.sample_policy),
        max_horizon=args.max_horizon,
        max_cells_per_object=args.max_cells_per_object,
        max_samples_per_cell=args.max_samples_per_cell,
        verbose=bool(args.verbose),
    )

    summary_path = Path(args.output_hdf5).with_suffix(Path(args.output_hdf5).suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Exported {summary['episodes_written']} samples from {summary['manifest_count']} manifests "
        f"to {args.output_hdf5}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
