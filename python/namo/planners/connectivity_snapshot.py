"""Helpers for computing region connectivity via the fast snapshot exporter."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from namo.visualization.wavefront_snapshot import (
    RegionGoalBundle,
    RegionGoalSample,
    WavefrontSnapshot,
    WavefrontSnapshotExporter,
)

RegionAdjacency = Dict[str, Set[str]]
RegionEdgeObjects = Dict[str, Dict[str, Set[str]]]
RegionLabels = Dict[int, str]
RegionGoalSamples = Dict[str, RegionGoalBundle]


def find_robot_label(region_labels: RegionLabels) -> Optional[str]:
    for label in region_labels.values():
        if "robot" in label:
            return label
    return None


def restrict_to_local_regions(
    adjacency: RegionAdjacency,
    edge_objects: RegionEdgeObjects,
    region_labels: RegionLabels,
    robot_label: Optional[str],
) -> Tuple[RegionAdjacency, RegionEdgeObjects, RegionLabels]:
    if not robot_label:
        return adjacency, edge_objects, region_labels

    neighbours = adjacency.get(robot_label, set())
    filtered_adjacency: RegionAdjacency = {robot_label: set(neighbours)}
    for neighbour in neighbours:
        filtered_adjacency[neighbour] = {robot_label}

    robot_edges = edge_objects.get(robot_label, {})
    filtered_edge_objects: RegionEdgeObjects = {
        robot_label: {
            neighbour: set(robot_edges.get(neighbour, set()))
            for neighbour in neighbours
        }
    }
    for neighbour in neighbours:
        neighbour_edges = edge_objects.get(neighbour, {})
        filtered_edge_objects[neighbour] = {
            robot_label: set(neighbour_edges.get(robot_label, set()))
        }

    allowed_labels = {robot_label, *neighbours}
    filtered_labels: RegionLabels = {
        idx: label for idx, label in region_labels.items() if label in allowed_labels
    }

    return filtered_adjacency, filtered_edge_objects, filtered_labels


def clone_goal_bundle(bundle: Any) -> RegionGoalBundle:
    return RegionGoalBundle(
        goals=[RegionGoalSample(sample.x, sample.y, sample.theta) for sample in bundle.goals],
        blocking_objects=set(bundle.blocking_objects),
    )


def snapshot_region_connectivity(
    env: Any,
    xml_path: str,
    config_path: str,
    *,
    resolution: Optional[float] = None,
    goal_radius: float = 0.15,
    include_snapshot: bool = False,
    goals_per_region: int = 0,
    generate_training_data: bool = False,
    local_info_only: bool = False,
) -> Tuple[
    RegionAdjacency,
    RegionEdgeObjects,
    RegionLabels,
    RegionGoalSamples,
    Optional[WavefrontSnapshot],
]:
    """Compute region connectivity using the snapshot exporter.

    Args:
        env: Active NAMO RL environment (or compatible interface) to sample state from.
        xml_path: Path to the MuJoCo XML file used to initialise ``env``.
        config_path: Path to the NAMO YAML configuration file.
        resolution: Optional override for grid resolution (defaults to exporter default).
        goal_radius: Radius (metres) used for goal region when building connectivity.
        include_snapshot: When ``True`` also return the full :class:`WavefrontSnapshot` for
            callers that need raw grids or metadata.
        goals_per_region: Maximum number of goal samples to draw per region. Ignored unless
            ``generate_training_data`` is ``True``.
        generate_training_data: When ``True``, include sampled goal bundles for each non-robot
            region in the returned data.
        local_info_only: When ``True``, restrict adjacency and edge-object outputs to the robot
            region and its immediate neighbours.

    Returns:
    Tuple containing ``(adjacency, edge_objects, region_labels, region_goals, snapshot_or_none)``.
    The snapshot entry is ``None`` unless ``include_snapshot`` is set.

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
        goals_per_region=goals_per_region if generate_training_data else 0,
    )

    adjacency: RegionAdjacency = {
        region: set(neighbours) for region, neighbours in snapshot.adjacency.items()
    }
    edge_objects: RegionEdgeObjects = {
        region: {neighbour: set(objs) for neighbour, objs in neighbour_map.items()}
        for region, neighbour_map in snapshot.edge_objects.items()
    }
    region_labels: RegionLabels = dict(snapshot.region_labels)

    robot_label = find_robot_label(region_labels)

    if local_info_only:
        adjacency, edge_objects, region_labels = restrict_to_local_regions(
            adjacency,
            edge_objects,
            region_labels,
            robot_label,
        )

    region_goals: RegionGoalSamples = {}
    if generate_training_data and goals_per_region > 0:
        region_goals = {
            region: clone_goal_bundle(bundle)
            for region, bundle in snapshot.region_goals.items()
            if "robot" not in region.lower()
        }

    if include_snapshot:
        if local_info_only:
            snapshot.adjacency = {key: set(value) for key, value in adjacency.items()}
            snapshot.edge_objects = {
                key: {nbr: set(objs) for nbr, objs in value.items()}
                for key, value in edge_objects.items()
            }
            snapshot.region_labels = dict(region_labels)
        if not generate_training_data:
            snapshot.region_goals = {}
        else:
            snapshot.region_goals = {
                region: clone_goal_bundle(bundle)
                for region, bundle in region_goals.items()
            }

    return (
        adjacency,
        edge_objects,
        region_labels,
        region_goals,
        snapshot if include_snapshot else None,
    )
