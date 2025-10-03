from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt  # type: ignore[import]
import networkx as nx  # type: ignore[import]
import numpy as np
from matplotlib.colors import ListedColormap  # type: ignore[import]
from matplotlib.figure import Figure  # type: ignore[import]
import matplotlib.patheffects as patheffects  # type: ignore[import]
from numpy.typing import NDArray

plt = cast(Any, plt)
nx = cast(Any, nx)


GridArray = NDArray[np.int_]


@dataclass
class WavefrontSnapshotData:
    resolution: float
    bounds: Tuple[float, float, float, float]
    uninflated_grid: GridArray
    static_grid: GridArray
    dynamic_grid: GridArray
    region_map: GridArray
    region_labels: Dict[int, str]
    adjacency: Dict[str, List[str]]
    edge_objects: Dict[str, Dict[str, List[str]]]
    robot_pose: Tuple[float, float, float]
    goal_pose: Optional[Tuple[float, float, float]]
    robot_half_extent: Tuple[float, float]
    movable_objects: Sequence[Dict[str, float]]
    environment_image: Optional[Path]

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return self.bounds


def load_snapshot(directory: Path, prefix: str = "snapshot") -> WavefrontSnapshotData:
    directory = directory.expanduser().resolve()
    metadata_path = directory / f"{prefix}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    def _load_array(name: str) -> GridArray:
        return np.load(directory / f"{prefix}_{name}.npy")

    environment_image = directory / f"{prefix}_environment.png"
    if not environment_image.exists():
        environment_image = None

    return WavefrontSnapshotData(
        resolution=float(metadata["resolution"]),
        bounds=tuple(metadata["bounds"]),
        uninflated_grid=_load_array("uninflated_grid"),
        static_grid=_load_array("static_grid"),
        dynamic_grid=_load_array("dynamic_grid"),
        region_map=_load_array("region_map"),
        region_labels={int(k): str(v) for k, v in metadata["region_labels"].items()},
        adjacency={
            str(region): list(neighbors) for region, neighbors in metadata.get("adjacency", {}).items()
        },
        edge_objects={
            str(region): {str(neighbor): list(objs) for neighbor, objs in neighbors.items()}
            for region, neighbors in metadata.get("adjacency_objects", {}).items()
        },
        robot_pose=tuple(metadata.get("robot_pose", [0.0, 0.0, 0.0])),
        goal_pose=tuple(metadata["goal_pose"]) if metadata.get("goal_pose") else None,
        robot_half_extent=tuple(metadata.get("robot_half_extent", [0.2, 0.2])),
        movable_objects=metadata.get("movable_objects", []),
        environment_image=environment_image,
    )


def _plot_environment(ax: Any, data: WavefrontSnapshotData) -> None:
    ax.set_title("Environment Layout")
    if data.environment_image and data.environment_image.exists():
        image_data = cast(Any, plt).imread(str(data.environment_image))
        ax.imshow(image_data)
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "No render available", ha="center", va="center", fontsize=12)
        ax.axis("off")
    text_effects = [patheffects.withStroke(linewidth=2, foreground="white")]
    ax.scatter([data.robot_pose[0]], [data.robot_pose[1]], c="#0d47a1", marker="*", s=140, label="Robot")
    if data.goal_pose:
        ax.scatter([data.goal_pose[0]], [data.goal_pose[1]], c="#2e7d32", marker="o", s=90, label="Goal")
    for obj in data.movable_objects:
        ax.scatter([obj["x"]], [obj["y"]], c="#fbc02d", marker="s", s=40)
        ax.text(
            obj["x"],
            obj["y"] + 0.1,
            obj["name"],
            ha="center",
            va="bottom",
            fontsize=7,
            color="#212121",
            path_effects=text_effects,
        )
    ax.legend(loc="upper right", fontsize=7)


def _plot_heatmap(ax: Any, data: WavefrontSnapshotData) -> None:
    ax.set_title("Dynamic Grid (inflated)")
    width, height = data.dynamic_grid.shape
    codes = np.zeros((width, height), dtype=np.uint8)

    static_obstacles = data.static_grid == -2
    dynamic_obstacles = (data.dynamic_grid == -2) & ~static_obstacles

    codes[static_obstacles] = 1  # walls
    codes[dynamic_obstacles] = 2  # movable obstacles

    goal_region_ids = [rid for rid, label in data.region_labels.items() if "goal" in label]
    if goal_region_ids:
        goal_mask = np.isin(data.region_map, goal_region_ids)
        codes[goal_mask] = 3

    cmap = ListedColormap([
        "#bdbdbd",  # free space
        "#ffffff",  # walls/static obstacles
        "#fbc02d",  # movable obstacles
        "#2e7d32",  # goal cells
    ])

    extent = (data.bounds[0], data.bounds[1], data.bounds[2], data.bounds[3])
    ax.imshow(codes.T, origin="lower", extent=extent, cmap=cmap)
    ax.set_xlim(data.bounds[0], data.bounds[1])
    ax.set_ylim(data.bounds[2], data.bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])

    robot_x, robot_y, _ = data.robot_pose
    ax.scatter([robot_x], [robot_y], c="#0d47a1", marker="*", s=140, label="Robot")

    if data.goal_pose:
        goal_x, goal_y, _ = data.goal_pose
        ax.scatter([goal_x], [goal_y], c="#2e7d32", marker="o", edgecolors="white", linewidths=0.8, s=90, label="Goal")

    region_effects = [patheffects.withStroke(linewidth=2, foreground="white")]
    for region_id, label in data.region_labels.items():
        coords = np.argwhere(data.region_map == region_id)
        if coords.size == 0:
            continue
        mean_x = coords[:, 0].mean()
        mean_y = coords[:, 1].mean()
        world_x = data.bounds[0] + mean_x * data.resolution
        world_y = data.bounds[2] + mean_y * data.resolution
        ax.text(
            world_x,
            world_y,
            label,
            fontsize=7,
            ha="center",
            va="center",
            color="#212121" if "goal" not in label else "white",
            path_effects=region_effects,
        )

    for obj in data.movable_objects:
        ax.scatter([obj["x"]], [obj["y"]], c="#fbc02d", marker="s", s=40)
        ax.text(
            obj["x"],
            obj["y"] + 0.1,
            obj["name"],
            fontsize=7,
            ha="center",
            va="bottom",
            color="#212121",
            path_effects=region_effects,
        )

    ax.legend(loc="upper right", fontsize=7)


def _plot_region_graph(ax: Any, data: WavefrontSnapshotData) -> None:
    ax.set_title("Region Connectivity")
    graph = cast(Any, nx).Graph()
    for region, neighbors in data.adjacency.items():
        graph.add_node(region)
        for neighbor in neighbors:
            graph.add_edge(region, neighbor)

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No adjacency data", ha="center", va="center")
        ax.axis("off")
        return

    layout = cast(Any, nx).spring_layout(graph, seed=42)
    node_colors = ["gold" if region == "robot_goal" else "skyblue" if region == "robot" else "lightgreen"
                   if region == "goal" else "lightgrey" for region in graph.nodes()]

    cast(Any, nx).draw_networkx(
        graph,
        pos=layout,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        node_size=1200,
        font_size=8,
        font_weight="bold",
        edge_color="#555555",
    )

    if data.edge_objects:
        edge_labels: Dict[Tuple[str, str], str] = {}
        for region, neighbor_map in data.edge_objects.items():
            for neighbor, objects in neighbor_map.items():
                if region not in graph or neighbor not in graph:
                    continue
                if not graph.has_edge(region, neighbor):
                    continue
                key = tuple(sorted((region, neighbor)))
                if key in edge_labels:
                    continue
                if objects:
                    display = ", ".join(objects[:3])
                    if len(objects) > 3:
                        display += ", …"
                    edge_labels[(region, neighbor)] = display
        if edge_labels:
            cast(Any, nx).draw_networkx_edge_labels(
                graph,
                pos=layout,
                edge_labels=edge_labels,
                font_size=7,
                ax=ax,
                label_pos=0.5,
            )
    ax.axis("off")


def create_figure(data: WavefrontSnapshotData) -> Figure:
    subplot_result = cast(Any, plt).subplots(1, 3, figsize=(18, 6))
    fig, axes = subplot_result
    _plot_environment(axes[0], data)
    _plot_heatmap(axes[1], data)
    _plot_region_graph(axes[2], data)
    fig.tight_layout()
    return fig


def visualize_snapshot(
    directory: Path,
    prefix: str = "snapshot",
    show: bool = True,
    save_path: Optional[Path] = None,
) -> Path:
    data = load_snapshot(directory, prefix)
    fig = create_figure(data)

    if save_path is None:
        save_path = directory / f"{prefix}_summary.png"
    cast(Any, fig).savefig(str(save_path), dpi=300)

    if show:
        cast(Any, plt).show()
    else:
        cast(Any, plt).close(fig)

    return save_path


__all__ = [
    "WavefrontSnapshotData",
    "load_snapshot",
    "create_figure",
    "visualize_snapshot",
]
