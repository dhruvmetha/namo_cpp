#!/usr/bin/env python3
"""Export wavefront-style occupancy grids and region metadata for visualisation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, cast

from PIL import Image  # type: ignore[import]

import namo_rl  # type: ignore[import]

from environment_selection import visualize_environment  # type: ignore[attr-defined]

visualize_environment_fn = cast(Any, visualize_environment)
from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True, help="Path to the MuJoCo XML environment file")
    parser.add_argument("--config", required=True, help="Path to the NAMO configuration YAML file")
    parser.add_argument(
        "--output-dir",
        default="wavefront_snapshots",
        help="Directory where snapshot artefacts will be written (default: wavefront_snapshots)",
    )
    parser.add_argument(
        "--snapshot-prefix",
        default="snapshot",
        help="Prefix used for generated files (default: snapshot)",
    )
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=0.15,
        help="Radius in metres used to paint goal cells when labelling regions (default: 0.15)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Override grid resolution in metres (default: match WavefrontGrid at 0.01)",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip generating a top-down environment image from the XML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initialising NAMO RL environment...")
    rl_env_cls: Any = getattr(namo_rl, "RLEnvironment")
    env: Any = rl_env_cls(args.xml, args.config, visualize=False)

    exporter = WavefrontSnapshotExporter(env, resolution=args.resolution)
    print("Building occupancy grids and region graph...")
    snapshot = exporter.build_snapshot(
        xml_path=str(Path(args.xml).resolve()),
        config_path=str(Path(args.config).resolve()),
        goal_radius=args.goal_radius,
    )

    print("Saving snapshot artefacts...")
    saved_paths: Dict[str, Path] = snapshot.save(output_dir, prefix=args.snapshot_prefix)

    if not args.skip_render:
        print("Rendering environment overview image...")
        raw_image: Any = visualize_environment_fn(args.xml)
        if raw_image is not None:
            image_path = output_dir / f"{args.snapshot_prefix}_environment.png"
            save_method = getattr(raw_image, "save", None)
            if callable(save_method):
                save_method(image_path)
                saved_paths["environment_image"] = image_path
            else:
                print("Warning: visualisation utility did not return a PIL Image; skipping image export.")
        else:
            print("Warning: environment render returned no image; skipping image export.")

    print("\nExport complete. Files written:")
    for label, path in saved_paths.items():
        print(f"  {label:>18}: {path}")


if __name__ == "__main__":
    main()
