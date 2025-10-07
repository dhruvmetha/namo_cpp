"""Console helper for inspecting wavefront region connectivity.

Example:
    python -m scripts.snapshot_connectivity_cli \
        --xml /path/to/env.xml \
        --config python/config/namo_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Set, cast

try:
    import namo_rl  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - surfaced immediately to the user
    raise SystemExit(
        "Failed to import namo_rl. Ensure the MuJoCo virtualenv is active and PYTHONPATH is set."
    ) from exc

from namo.planners import snapshot_region_connectivity

namo_rl = cast(Any, namo_rl)


def _format_set(values: Iterable[str]) -> str:
    items = sorted(values)
    return ", ".join(items) if items else "<none>"


def _print_region_summary(
    adjacency: Mapping[str, Set[str]],
    edge_objects: Mapping[str, MutableMapping[str, Set[str]]],
    *,
    max_regions: int,
    verbose: bool,
) -> None:
    region_names: Sequence[str] = sorted(adjacency.keys())
    total = len(region_names)
    if total == 0:
        print("No regions discovered.")
        return

    display_names = region_names if max_regions < 0 else region_names[:max_regions]
    for region in display_names:
        neighbours = adjacency.get(region, set())
        print(f"- {region}: neighbours -> [{_format_set(neighbours)}]")
        if not verbose:
            continue
        edges = edge_objects.get(region, {})
        for neighbour in sorted(edges.keys()):
            objects = edges[neighbour]
            print(f"    edge {region} <-> {neighbour}: blockers -> [{_format_set(objects)}]")

    if 0 <= max_regions < total:
        print(f"... ({total - max_regions} additional region(s) omitted; use --max-regions -1 to show all)")


def _print_goal_samples(region_goals: Mapping[str, Any]) -> None:
    if not region_goals:
        print("[goals] no samples generated.")
        return

    print("[goals] sampled positions per region:")
    for region in sorted(region_goals.keys()):
        bundle = region_goals[region]
        samples = getattr(bundle, "goals", [])
        blocking_default: Set[str] = set()
        blockers = cast(Iterable[str], getattr(bundle, "blocking_objects", blocking_default))
        formatted_samples = ", ".join(
            f"({getattr(sample, 'x', 0.0):.3f}, {getattr(sample, 'y', 0.0):.3f})"
            for sample in samples
        )
        if not formatted_samples:
            formatted_samples = "<none>"
        print(f"  - {region}: goals -> [{formatted_samples}] blockers -> [{_format_set(blockers)}]")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect wavefront region connectivity.")
    parser.add_argument(
        "--xml",
        required=True,
        help="Path to the MuJoCo XML environment file",
    )
    parser.add_argument(
        "--config",
        default="python/config/namo_config.yaml",
        help="Path to the NAMO configuration YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Launch the environment viewer instead of running headless",
    )
    parser.add_argument(
        "--goal-radius",
        type=float,
        default=0.15,
        help="Goal radius used when building connectivity (default: %(default)s)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Optional override for the snapshot grid resolution",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=15,
        help="Maximum regions to print (-1 for all, default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print blocking objects for each edge as well",
    )
    parser.add_argument(
        "--goal-samples",
        type=int,
        default=0,
        help="Number of random goal samples per region to compute (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    print("[setup] constructing RLEnvironment")
    env_cls = getattr(namo_rl, "RLEnvironment")
    env = env_cls(args.xml, args.config, args.render)

    print("[info] exporting region connectivity via snapshot helper")
    adjacency, edge_objects, region_labels, region_goals, _ = snapshot_region_connectivity(
        env,
        args.xml,
        args.config,
        resolution=args.resolution,
        goal_radius=args.goal_radius,
        include_snapshot=False,
        goals_per_region=args.goal_samples,
    )

    print(f"[result] regions discovered: {len(region_labels)}")
    _print_region_summary(
        adjacency,
        edge_objects,
        max_regions=args.max_regions,
        verbose=args.verbose,
    )

    if args.goal_samples > 0:
        _print_goal_samples(region_goals)

    if hasattr(env, "close"):
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
