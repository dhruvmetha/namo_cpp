"""Helpers for computing region connectivity via the fast snapshot exporter."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from namo.visualization.wavefront_snapshot import WavefrontSnapshot, WavefrontSnapshotExporter

RegionAdjacency = Dict[str, Set[str]]
RegionEdgeObjects = Dict[str, Dict[str, Set[str]]]
RegionLabels = Dict[int, str]


def snapshot_region_connectivity(
    env: Any,
    xml_path: str,
    config_path: str,
    *,
    resolution: Optional[float] = None,
    goal_radius: float = 0.15,
    include_snapshot: bool = False,
) -> Tuple[RegionAdjacency, RegionEdgeObjects, RegionLabels, Optional[WavefrontSnapshot]]:
    """Compute region connectivity using the snapshot exporter.

    Args:
        env: Active NAMO RL environment (or compatible interface) to sample state from.
        xml_path: Path to the MuJoCo XML file used to initialise ``env``.
        config_path: Path to the NAMO YAML configuration file.
        resolution: Optional override for grid resolution (defaults to exporter default).
        goal_radius: Radius (metres) used for goal region when building connectivity.
        include_snapshot: When ``True`` also return the full :class:`WavefrontSnapshot` for
            callers that need raw grids or metadata.

    Returns:
        Tuple containing ``(adjacency, edge_objects, region_labels, snapshot_or_none)``. The
        snapshot entry is ``None`` unless ``include_snapshot`` is set.

    Notes:
        This helper mirrors the data returned by the C++ ``WavefrontGrid`` connectivity logic but
        executes much faster by leveraging pre-existing NumPy rasterisation in the snapshot
        exporter. The exporter will temporarily reset the environment to obtain geometry, so make
        sure to cache any planner state that depends on the current simulation before invoking it.
    """

    exporter = WavefrontSnapshotExporter(env, resolution=resolution)
    snapshot = exporter.build_snapshot(
        xml_path=xml_path,
        config_path=config_path,
        goal_radius=goal_radius,
    )

    adjacency: RegionAdjacency = {
        region: set(neighbours) for region, neighbours in snapshot.adjacency.items()
    }
    edge_objects: RegionEdgeObjects = {
        region: {neighbour: set(objs) for neighbour, objs in neighbour_map.items()}
        for region, neighbour_map in snapshot.edge_objects.items()
    }
    region_labels: RegionLabels = dict(snapshot.region_labels)

    return adjacency, edge_objects, region_labels, snapshot if include_snapshot else None
