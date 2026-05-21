from __future__ import annotations

import json
import math
import hashlib
import itertools
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt  # type: ignore[import]
import networkx as nx  # type: ignore[import]
import numpy as np
from matplotlib.figure import Figure  # type: ignore[import]
import matplotlib.patheffects as patheffects  # type: ignore[import]
from matplotlib.patches import Polygon as MplPolygon  # type: ignore[import]
from numpy.typing import NDArray

plt = cast(Any, plt)
nx = cast(Any, nx)

COLOR_BACKGROUND = "#FFFFFF"
COLOR_WALL = "#A0A0A0"  # static obstacles
COLOR_MOVABLE = "#FFFF00"  # movable (non-target)
COLOR_TARGET = "#00FFFF"
COLOR_ROBOT_REGION = "#FF0000"
COLOR_GOAL_REGION = "#00FF00"
COLOR_PRED_GOAL = "#4B0082"
COLOR_CROP_BOX = "#000000"
COLOR_OUTLINE = "#000000"

# Markers
COLOR_ROBOT_MARKER = "#0000FF"  # blue fill
COLOR_ROBOT_MARKER_OUTLINE = "#FFFFFF"
COLOR_GOAL_MARKER_FILL = "#FF0000"  # red fill
COLOR_GOAL_MARKER_OUTLINE = "#000000"
COLOR_GOAL_MARKER_HALO = "#FFFFFF"

# Regions not explicitly called out (kept light so obstacles/markers remain readable).
COLOR_OTHER_REGION = "#E6E6FF"

# Non-robot / non-goal region colors (avoid confusion with:
# - movable yellow (#FFFF00)
# - goal green (#00FF00)
# - target cyan (#00FFFF)
# - walls gray (#A0A0A0)
PASTEL_REGION_COLORS = [
    "#CFE9FF",  # light blue
    "#FFE3C4",  # light orange
    "#D7F8D7",  # light green (far from goal green, but clearly distinct)
    "#FAD1E6",  # light pink
    "#E7D6FF",  # light lavender
    "#D6FFF2",  # light aqua/mint (distinct from cyan)
    "#FFF0D6",  # light beige
    "#FFD3B6",  # light peach
]


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
    tier1_inflation_margin_m: float
    movable_objects: Sequence[Dict[str, float]]
    environment_image: Optional[Path]
    xml_path: Optional[str] = None
    target_object: Optional[str] = None

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
        tier1_inflation_margin_m=float(metadata.get("tier1_inflation_margin_m", 0.005)),
        movable_objects=metadata.get("movable_objects", []),
        environment_image=environment_image,
        xml_path=str(metadata.get("xml_path")) if metadata.get("xml_path") else None,
        target_object=str(metadata.get("target_object")) if metadata.get("target_object") else None,
    )

def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    color = hex_color.lstrip("#")
    return (
        int(color[0:2], 16) / 255.0,
        int(color[2:4], 16) / 255.0,
        int(color[4:6], 16) / 255.0,
    )

def _hex_to_rgb_u8(hex_color: str) -> Tuple[int, int, int]:
    color = hex_color.lstrip("#")
    return (
        int(color[0:2], 16),
        int(color[2:4], 16),
        int(color[4:6], 16),
    )


def _snapshot_inflation_radius(data: WavefrontSnapshotData) -> float:
    hx = abs(float(data.robot_half_extent[0]))
    hy = abs(float(data.robot_half_extent[1]))
    margin = float(data.tier1_inflation_margin_m)
    if margin < 0.0:
        margin = 0.005
    radius = math.sqrt(hx * hx + hy * hy)
    if radius <= 0.0:
        radius = 0.15
    return radius + margin


def _region_label_color(label: str) -> str:
    lower = label.lower()
    if "robot" in lower:
        return COLOR_ROBOT_REGION
    if "goal" in lower:
        return COLOR_GOAL_REGION
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return PASTEL_REGION_COLORS[digest[0] % len(PASTEL_REGION_COLORS)]

def _compute_region_color_map(data: WavefrontSnapshotData) -> Dict[str, str]:
    """Assign light colors to regions so adjacent regions differ as much as possible."""

    all_regions = sorted({str(lbl) for lbl in data.region_labels.values()})
    adjacency: Dict[str, List[str]] = {r: [] for r in all_regions}
    for r, neigh in data.adjacency.items():
        adjacency.setdefault(str(r), [])
        for n in neigh:
            adjacency.setdefault(str(n), [])
            adjacency[str(r)].append(str(n))

    color_map: Dict[str, str] = {}
    for region in all_regions:
        lower = region.lower()
        if "robot" in lower:
            color_map[region] = COLOR_ROBOT_REGION
        elif "goal" in lower:
            color_map[region] = COLOR_GOAL_REGION

    # When there are only a handful of regions, use a small, high-contrast pastel set.
    non_special = [r for r in all_regions if r not in color_map]
    if len(non_special) <= 4:
        # Pick a set that is visually far apart so e.g. r4 and r6 never look "almost the same".
        palette = list(PASTEL_REGION_COLORS[:4])
    else:
        palette = list(PASTEL_REGION_COLORS)
    palette_rgb = {c: _hex_to_rgb_u8(c) for c in palette}
    usage: Dict[str, int] = {c: 0 for c in palette}

    def color_distance(a: str, b: str) -> float:
        ra, ga, ba = palette_rgb.get(a, _hex_to_rgb_u8(a))
        rb, gb, bb = palette_rgb.get(b, _hex_to_rgb_u8(b))
        return float((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2)

    # If there are only a few non-(robot/goal) regions, keep them all distinct.
    enforce_unique = len(non_special) <= len(palette)

    # Greedy coloring, highest degree first, choosing colors far from neighbor colors.
    order = sorted(
        non_special,
        key=lambda r: (-len(set(adjacency.get(r, []))), r),
    )
    for region in order:
        neighbor_colors = [color_map[n] for n in adjacency.get(region, []) if n in color_map]
        used_non_special = {color_map[r] for r in color_map.keys() if r not in {"robot", "goal", "robot_goal"}}

        best_color = palette[0]
        best_score = -1.0
        best_usage = 10 ** 9
        found_non_conflict = False
        for candidate in palette:
            if enforce_unique and candidate in used_non_special:
                continue
            conflict = candidate in neighbor_colors
            if neighbor_colors:
                min_dist = min(color_distance(candidate, nc) for nc in neighbor_colors)
            else:
                min_dist = 1e12
            score = min_dist
            cand_usage = usage[candidate]
            if (not conflict) and (not found_non_conflict):
                # Prefer any non-conflicting color over conflicting ones.
                best_color = candidate
                best_score = score
                best_usage = cand_usage
                found_non_conflict = True
                continue
            if conflict and found_non_conflict:
                continue
            if score > best_score or (math.isclose(score, best_score) and cand_usage < best_usage):
                best_color = candidate
                best_score = score
                best_usage = cand_usage

        color_map[region] = best_color
        usage[best_color] += 1

    # Ensure the displayed labels r3 and r4 (which correspond to region_6 and region_5 after remap)
    # are clearly distinct in common paper figures.
    def _hex_to_rgb_any(hex_color: str) -> Tuple[int, int, int]:
        c = hex_color.lstrip("#")
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))

    def _dist(a: str, b: str) -> float:
        ra, ga, ba = _hex_to_rgb_any(a)
        rb, gb, bb = _hex_to_rgb_any(b)
        return float((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2)

    a, b = "region_6", "region_5"
    if a in color_map and b in color_map:
        if _dist(color_map[a], color_map[b]) < 9000.0:
            # Re-pick color for 'a' to maximize distance to 'b' and its neighbors.
            neighbors = [color_map[n] for n in adjacency.get(a, []) if n in color_map]
            target = color_map[b]
            best = color_map[a]
            best_score = -1.0
            for candidate in palette:
                if candidate == color_map[b]:
                    continue
                score = _dist(candidate, target)
                if neighbors:
                    score = min(score, *( _dist(candidate, nc) for nc in neighbors))
                if score > best_score:
                    best_score = score
                    best = candidate
            color_map[a] = best

    return color_map


def region_color_map(data: WavefrontSnapshotData) -> Dict[str, str]:
    """Public helper: pastel-but-distinct region color assignment."""
    return _compute_region_color_map(data)


def _short_region_label(label: str) -> str:
    if label == "robot_goal":
        return "rg"
    if label.startswith("region_"):
        suffix = label.split("_", 1)[1]
        return f"r{suffix}"
    return label


def _compute_between_obstacles_region_label_map(data: WavefrontSnapshotData) -> Dict[str, str]:
    """Map raw region node names (e.g. 'region_3') -> display labels (e.g. 'r2') using geometry.

    This is a display-only mapping used to keep region names consistent with paper figures even when
    underlying region IDs change. If geometry data is missing, returns an empty mapping.
    """

    # Candidate region centroids (world coords) keyed by raw region label string.
    region_centroids: Dict[str, Tuple[float, float]] = {}
    res = float(data.resolution)
    b0 = float(data.bounds[0])
    b2 = float(data.bounds[2])
    for region_id, label in data.region_labels.items():
        label_s = str(label)
        if not label_s.startswith("region_"):
            continue
        coords = np.argwhere(data.region_map == int(region_id))
        if coords.size == 0:
            continue
        mean_gx = float(coords[:, 0].mean())
        mean_gy = float(coords[:, 1].mean())
        wx = b0 + (mean_gx + 0.5) * res
        wy = b2 + (mean_gy + 0.5) * res
        region_centroids[label_s] = (wx, wy)

    if not region_centroids:
        return {}

    # Movable object centers keyed by short label 'oN'.
    obj_xy: Dict[str, Tuple[float, float]] = {}
    for obj in data.movable_objects:
        name = str(obj.get("name", ""))
        short = _short_object_label(name)
        if re.match(r"^o\d+$", short):
            obj_xy[short] = (float(obj.get("x", 0.0)), float(obj.get("y", 0.0)))

    # Desired display mapping (semantic) for this figure set.
    desired: List[Tuple[str, Tuple[str, str]]] = [
        ("r2", ("o1", "o2")),
        ("r3", ("o2", "o6")),
        ("r4", ("o4", "o5")),
        ("r5", ("o6", "o7")),
    ]

    targets: List[Tuple[str, Tuple[float, float]]] = []
    for display, (a, b) in desired:
        if a not in obj_xy or b not in obj_xy:
            continue
        (ax, ay) = obj_xy[a]
        (bx, by) = obj_xy[b]
        targets.append((display, ((ax + bx) * 0.5, (ay + by) * 0.5)))

    if not targets:
        return {}

    regions = list(region_centroids.keys())
    k = min(len(targets), len(regions))
    targets = targets[:k]

    def dist2(p: Tuple[float, float], q: Tuple[float, float]) -> float:
        dx = float(p[0]) - float(q[0])
        dy = float(p[1]) - float(q[1])
        return dx * dx + dy * dy

    best_total = float("inf")
    best_perm: Optional[Tuple[str, ...]] = None
    for subset in itertools.combinations(regions, k):
        for perm in itertools.permutations(subset):
            total = 0.0
            for idx, region_label in enumerate(perm):
                total += dist2(region_centroids[region_label], targets[idx][1])
            if total < best_total:
                best_total = total
                best_perm = tuple(str(x) for x in perm)

    if best_perm is None:
        return {}

    mapping: Dict[str, str] = {}
    for idx, region_label in enumerate(best_perm):
        mapping[str(region_label)] = str(targets[idx][0])
    return mapping


def _display_region_label(data: WavefrontSnapshotData, raw_label: str, *, mapping: Dict[str, str]) -> str:
    raw = str(raw_label)
    if raw in mapping:
        return mapping[raw]
    return _short_region_label(raw)


def _short_object_label(name: str) -> str:
    match = re.match(r"^obstacle_(\d+)_movable$", name)
    if match:
        return f"o{match.group(1)}"
    match = re.match(r"^obstacle_(\d+)$", name)
    if match:
        return f"o{match.group(1)}"
    return name


def _target_inflated_mask(data: WavefrontSnapshotData) -> NDArray[np.bool_]:
    if not data.target_object:
        return np.zeros_like(data.dynamic_grid, dtype=bool)

    target = None
    for obj in data.movable_objects:
        if str(obj.get("name", "")) == data.target_object:
            target = obj
            break
    if target is None:
        return np.zeros_like(data.dynamic_grid, dtype=bool)

    width, height = data.dynamic_grid.shape
    mask = np.zeros((width, height), dtype=bool)

    cx = float(target.get("x", 0.0))
    cy = float(target.get("y", 0.0))
    yaw = float(target.get("theta", 0.0))
    half_w = float(target.get("half_extent_x", 0.0))
    half_h = float(target.get("half_extent_y", 0.0))
    if half_w <= 0.0 or half_h <= 0.0:
        return mask

    inflate_r = _snapshot_inflation_radius(data)
    half_w += inflate_r
    half_h += inflate_r

    cos_a = math.cos(yaw)
    sin_a = math.sin(yaw)

    # Conservative AABB in grid coords from rotated corners
    corners_local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    corners_world: List[Tuple[float, float]] = []
    for lx, ly in corners_local:
        corners_world.append((cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a))

    def world_to_grid_x(wx: float) -> int:
        return int(math.floor((wx - data.bounds[0]) / float(data.resolution)))

    def world_to_grid_y(wy: float) -> int:
        return int(math.floor((wy - data.bounds[2]) / float(data.resolution)))

    min_gx = max(0, min(world_to_grid_x(wx) for wx, _ in corners_world))
    max_gx = min(width - 1, max(world_to_grid_x(wx) for wx, _ in corners_world))
    min_gy = max(0, min(world_to_grid_y(wy) for _, wy in corners_world))
    max_gy = min(height - 1, max(world_to_grid_y(wy) for _, wy in corners_world))

    res = float(data.resolution)
    for gx in range(min_gx, max_gx + 1):
        world_x = data.bounds[0] + gx * res + 0.5 * res
        dx = world_x - cx
        for gy in range(min_gy, max_gy + 1):
            world_y = data.bounds[2] + gy * res + 0.5 * res
            dy = world_y - cy

            local_x = dx * cos_a + dy * sin_a
            local_y = -dx * sin_a + dy * cos_a
            if abs(local_x) <= half_w and abs(local_y) <= half_h:
                mask[gx, gy] = True

    return mask


def _target_uninflated_mask(data: WavefrontSnapshotData) -> NDArray[np.bool_]:
    if not data.target_object:
        return np.zeros_like(data.dynamic_grid, dtype=bool)

    target: Optional[Dict[str, float]] = None
    for obj in data.movable_objects:
        if str(obj.get("name", "")) == data.target_object:
            target = cast(Dict[str, float], obj)
            break
    if target is None:
        return np.zeros_like(data.dynamic_grid, dtype=bool)

    width, height = data.dynamic_grid.shape
    mask = np.zeros((width, height), dtype=bool)

    cx = float(target.get("x", 0.0))
    cy = float(target.get("y", 0.0))
    yaw = float(target.get("theta", 0.0))
    half_w = float(target.get("half_extent_x", 0.0))
    half_h = float(target.get("half_extent_y", 0.0))
    if half_w <= 0.0 or half_h <= 0.0:
        return mask

    cos_a = math.cos(yaw)
    sin_a = math.sin(yaw)

    corners_local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    corners_world: List[Tuple[float, float]] = []
    for lx, ly in corners_local:
        corners_world.append((cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a))

    def world_to_grid_x(wx: float) -> int:
        return int(math.floor((wx - data.bounds[0]) / float(data.resolution)))

    def world_to_grid_y(wy: float) -> int:
        return int(math.floor((wy - data.bounds[2]) / float(data.resolution)))

    min_gx = max(0, min(world_to_grid_x(wx) for wx, _ in corners_world))
    max_gx = min(width - 1, max(world_to_grid_x(wx) for wx, _ in corners_world))
    min_gy = max(0, min(world_to_grid_y(wy) for _, wy in corners_world))
    max_gy = min(height - 1, max(world_to_grid_y(wy) for _, wy in corners_world))

    res = float(data.resolution)
    for gx in range(min_gx, max_gx + 1):
        world_x = data.bounds[0] + gx * res + 0.5 * res
        dx = world_x - cx
        for gy in range(min_gy, max_gy + 1):
            world_y = data.bounds[2] + gy * res + 0.5 * res
            dy = world_y - cy

            local_x = dx * cos_a + dy * sin_a
            local_y = -dx * sin_a + dy * cos_a
            if abs(local_x) <= half_w and abs(local_y) <= half_h:
                mask[gx, gy] = True

    return mask


def _binary_erode_3x3(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    m = mask.astype(bool)
    p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    out = p[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= p[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]]
    return out


def _binary_dilate_3x3(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    m = mask.astype(bool)
    p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    out = np.zeros_like(m, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= p[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]]
    return out


def _outline_mask(mask: NDArray[np.bool_], *, thickness_px: int = 2) -> NDArray[np.bool_]:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    er = mask.astype(bool)
    for _ in range(max(1, int(thickness_px))):
        er = _binary_erode_3x3(er)
    border = mask & ~er
    out = border
    for _ in range(max(0, int(thickness_px) - 1)):
        out = _binary_dilate_3x3(out)
    return out


def _distance_to_obstacles(obstacles: NDArray[np.bool_]) -> NDArray[np.int_]:
    """Multi-source BFS distance using 8-neighborhood."""
    width, height = obstacles.shape
    dist = np.full((width, height), 10 ** 9, dtype=np.int32)
    q: deque[Tuple[int, int]] = deque()
    xs, ys = np.nonzero(obstacles)
    for gx, gy in zip(xs.tolist(), ys.tolist()):
        dist[gx, gy] = 0
        q.append((gx, gy))
    if not q:
        dist[:, :] = 10 ** 6
        return dist
    while q:
        gx, gy = q.popleft()
        base = int(dist[gx, gy])
        nd = base + 1
        for dx, dy in (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ):
            nx = gx + dx
            ny = gy + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if nd < int(dist[nx, ny]):
                dist[nx, ny] = nd
                q.append((nx, ny))
    return dist


def _rasterize_movable_object_mask(data: WavefrontSnapshotData, obj: Dict[str, float], *, inflated: bool) -> NDArray[np.bool_]:
    width, height = data.dynamic_grid.shape
    mask = np.zeros((width, height), dtype=bool)

    cx = float(obj.get("x", 0.0))
    cy = float(obj.get("y", 0.0))
    yaw = float(obj.get("theta", 0.0))
    half_w = float(obj.get("half_extent_x", 0.0))
    half_h = float(obj.get("half_extent_y", 0.0))
    if half_w <= 0.0 or half_h <= 0.0:
        return mask

    if inflated:
        inflate_r = _snapshot_inflation_radius(data)
        half_w += inflate_r
        half_h += inflate_r

    cos_a = math.cos(yaw)
    sin_a = math.sin(yaw)

    corners_local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    corners_world: List[Tuple[float, float]] = []
    for lx, ly in corners_local:
        corners_world.append((cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a))

    res = float(data.resolution)
    bx0, _, by0, _ = data.bounds

    def world_to_grid_x(wx: float) -> int:
        return int(math.floor((wx - bx0) / res))

    def world_to_grid_y(wy: float) -> int:
        return int(math.floor((wy - by0) / res))

    min_gx = max(0, min(world_to_grid_x(wx) for wx, _ in corners_world))
    max_gx = min(width - 1, max(world_to_grid_x(wx) for wx, _ in corners_world))
    min_gy = max(0, min(world_to_grid_y(wy) for _, wy in corners_world))
    max_gy = min(height - 1, max(world_to_grid_y(wy) for _, wy in corners_world))

    # Treat a cell as occupied if the object overlaps the cell square, not just if the center is inside.
    # This reduces "dotted" edges at high export sizes where center-only tests can create gaps.
    # We approximate overlap by expanding the half extents by the cell half-diagonal.
    cell_margin = 0.5 * res * math.sqrt(2.0)

    for gx in range(min_gx, max_gx + 1):
        world_x = data.bounds[0] + gx * res + 0.5 * res
        dx = world_x - cx
        for gy in range(min_gy, max_gy + 1):
            world_y = data.bounds[2] + gy * res + 0.5 * res
            dy = world_y - cy

            local_x = dx * cos_a + dy * sin_a
            local_y = -dx * sin_a + dy * cos_a
            if abs(local_x) <= (half_w + cell_margin) and abs(local_y) <= (half_h + cell_margin):
                mask[gx, gy] = True

    return mask


def environment_label_positions(data: WavefrontSnapshotData) -> List[Tuple[float, float, str]]:
    """Return (world_x, world_y, label) for movable object labels in the environment panel."""
    labels: List[Tuple[float, float, str]] = []
    for obj in data.movable_objects:
        labels.append((float(obj.get("x", 0.0)), float(obj.get("y", 0.0)), _short_object_label(str(obj.get("name", "")))))
    return labels


def inflated_label_positions(data: WavefrontSnapshotData) -> List[Tuple[float, float, str]]:
    """Return (world_x, world_y, label) for region + object labels in the inflated panel."""
    width, height = data.dynamic_grid.shape
    res = float(data.resolution)
    region_display_map = _compute_between_obstacles_region_label_map(data)
    inflation_band = (data.dynamic_grid == -1) & ~(data.uninflated_grid == -1)

    occupied_uninflated = data.uninflated_grid == -1
    movable_uninflated = _movable_mask_uninflated(data)
    wall_uninflated = occupied_uninflated & ~movable_uninflated
    obstacle_mask = (wall_uninflated | movable_uninflated).astype(bool)
    dist = _distance_to_obstacles(obstacle_mask)

    placed: List[Tuple[float, float, str]] = []
    used_xy: List[Tuple[float, float]] = []
    min_sep_m = 0.15

    # Region labels (pick a cell deep inside region, then separate).
    for region_id, label in data.region_labels.items():
        coords = np.argwhere(data.region_map == region_id)
        if coords.size == 0:
            continue
        display = _display_region_label(data, str(label), mapping=region_display_map)

        coords_to_consider = coords
        if display == "r4":
            # Avoid placing r4 on the inflation whitespace (rendered white) when possible.
            keep = ~inflation_band[coords[:, 0], coords[:, 1]]
            if bool(np.any(keep)):
                coords_to_consider = coords[keep]
        best_gx = int(coords[0, 0])
        best_gy = int(coords[0, 1])
        best_d = -1
        for gx, gy in coords_to_consider:
            d = int(dist[int(gx), int(gy)])
            if d > best_d:
                best_d = d
                best_gx = int(gx)
                best_gy = int(gy)
        world_x = float(data.bounds[0]) + (best_gx + 0.5) * res
        world_y = float(data.bounds[2]) + (best_gy + 0.5) * res
        if str(label).lower().startswith("robot"):
            world_y = max(float(data.bounds[2]), world_y - 0.25)
        if display == "r4":
            world_x = min(float(data.bounds[1]) - 1e-6, world_x + 0.08)

        ok = True
        for ux, uy in used_xy:
            if math.hypot(world_x - ux, world_y - uy) < min_sep_m:
                ok = False
                break
        if not ok:
            continue
        used_xy.append((world_x, world_y))
        placed.append((world_x, world_y, display))

    # Object labels (placed in nearby free space).
    for obj in data.movable_objects:
        cx = float(obj.get("x", 0.0))
        cy = float(obj.get("y", 0.0))
        placed.append((cx, cy, _short_object_label(str(obj.get("name", "")))))

    return placed


def _movable_mask_uninflated(data: WavefrontSnapshotData) -> NDArray[np.bool_]:
    width, height = data.uninflated_grid.shape
    out = np.zeros((width, height), dtype=bool)
    for obj in data.movable_objects:
        out |= _rasterize_movable_object_mask(data, obj, inflated=False)
    return out


def movable_mask_uninflated(data: WavefrontSnapshotData) -> NDArray[np.bool_]:
    """Public helper for scripts that want the uninflated movable mask."""
    return _movable_mask_uninflated(data)


def _plot_environment(ax: Any, data: WavefrontSnapshotData) -> None:
    ax.set_title("Environment Layout")

    # Render directly from the uninflated grid for crisp corners (no antialiased MuJoCo render).
    width, height = data.uninflated_grid.shape
    occupied = data.uninflated_grid == -1
    movable_mask = _movable_mask_uninflated(data)
    wall_occupied = occupied & ~movable_mask

    rgb = np.empty((width, height, 3), dtype=np.float32)
    rgb[:, :, :] = _hex_to_rgb(COLOR_BACKGROUND)
    rgb[wall_occupied] = _hex_to_rgb(COLOR_WALL)

    # Movables: target cyan, others yellow.
    target_mask = np.zeros_like(movable_mask, dtype=bool)
    if data.target_object:
        for obj in data.movable_objects:
            if str(obj.get("name", "")) == data.target_object:
                target_mask = _rasterize_movable_object_mask(data, obj, inflated=False)
                break
    rgb[movable_mask & ~target_mask] = _hex_to_rgb(COLOR_MOVABLE)
    rgb[target_mask] = _hex_to_rgb(COLOR_TARGET)

    # Thin black borders around movable objects only (not walls).
    movable_outline = _outline_mask(movable_mask.astype(bool), thickness_px=1)
    rgb[movable_outline] = _hex_to_rgb(COLOR_OUTLINE)

    extent = (data.bounds[0], data.bounds[1], data.bounds[2], data.bounds[3])
    ax.imshow(rgb.transpose(1, 0, 2), origin="lower", extent=extent, interpolation="nearest")
    ax.set_axis_off()

    text_effects = [patheffects.withStroke(linewidth=2, foreground="white")]
    _draw_robot_footprint(ax, data, label="Robot")
    if data.goal_pose:
        ax.scatter([data.goal_pose[0]], [data.goal_pose[1]], c=COLOR_GOAL_MARKER_HALO, marker="*", s=210, label=None)
        ax.scatter(
            [data.goal_pose[0]],
            [data.goal_pose[1]],
            c=COLOR_GOAL_MARKER_FILL,
            marker="*",
            s=140,
            edgecolors=COLOR_GOAL_MARKER_OUTLINE,
            linewidths=1.4,
            label="Goal",
        )
    for obj in data.movable_objects:
        ax.text(
            float(obj["x"]),
            float(obj["y"]),
            _short_object_label(str(obj["name"])),
            ha="center",
            va="center",
            fontsize=7,
            color="#212121",
            path_effects=text_effects,
        )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=7,
    )


def environment_rgb_u8(data: WavefrontSnapshotData) -> NDArray[np.uint8]:
    """Return an environment RGB image (uint8) in PIL layout (top-left origin)."""
    width, height = data.uninflated_grid.shape
    occupied = data.uninflated_grid == -1
    movable_mask = _movable_mask_uninflated(data)
    wall_occupied = occupied & ~movable_mask

    rgb = np.empty((width, height, 3), dtype=np.float32)
    rgb[:, :, :] = _hex_to_rgb(COLOR_BACKGROUND)
    rgb[wall_occupied] = _hex_to_rgb(COLOR_WALL)

    target_mask = np.zeros_like(movable_mask, dtype=bool)
    if data.target_object:
        for obj in data.movable_objects:
            if str(obj.get("name", "")) == data.target_object:
                target_mask = _rasterize_movable_object_mask(data, obj, inflated=False)
                break
    rgb[movable_mask & ~target_mask] = _hex_to_rgb(COLOR_MOVABLE)
    rgb[target_mask] = _hex_to_rgb(COLOR_TARGET)

    # Match matplotlib's imshow(rgb.transpose(1,0,2), origin="lower"), then convert to PIL top-left.
    img = (rgb.transpose(1, 0, 2) * 255.0).clip(0, 255).astype(np.uint8)
    return np.flipud(img)


def _plot_heatmap(ax: Any, data: WavefrontSnapshotData) -> None:
    ax.set_title("Regions + Obstacles (inflated)")
    width, height = data.dynamic_grid.shape

    # Emulate the "white inflation border" look:
    # - Regions are from the inflated grid (region_map already respects inflated obstacles).
    # - Obstacles are drawn using UNINFLATED footprints only, so the inflation-only band stays white.
    occupied_uninflated = data.uninflated_grid == -1
    movable_uninflated = _movable_mask_uninflated(data)
    wall_uninflated = occupied_uninflated & ~movable_uninflated

    target_mask = np.zeros_like(movable_uninflated, dtype=bool)
    if data.target_object:
        for obj in data.movable_objects:
            if str(obj.get("name", "")) == data.target_object:
                target_mask = _rasterize_movable_object_mask(data, obj, inflated=False)
                break

    rgb = np.empty((width, height, 3), dtype=np.float32)
    rgb[:, :, :] = _hex_to_rgb(COLOR_BACKGROUND)

    # Color all regions (computed on inflated grid), so it's easy to see where regions stop.
    region_colors = _compute_region_color_map(data)
    for region_id, label in data.region_labels.items():
        mask = data.region_map == region_id
        if np.any(mask):
            rgb[mask] = _hex_to_rgb(region_colors.get(str(label), _region_label_color(str(label))))

    # Obstacles override regions (UNINFLATED only).
    rgb[wall_uninflated] = _hex_to_rgb(COLOR_WALL)
    rgb[movable_uninflated & ~target_mask] = _hex_to_rgb(COLOR_MOVABLE)
    rgb[target_mask] = _hex_to_rgb(COLOR_TARGET)

    # Border the inflation-only band so it's easy to see what was removed by inflation.
    inflated_occ = data.dynamic_grid == -1
    inflation_band = inflated_occ & ~(data.uninflated_grid == -1)
    inflation_outline = _outline_mask(inflation_band.astype(bool), thickness_px=1)
    rgb[inflation_outline] = _hex_to_rgb(COLOR_OUTLINE)

    # Borders are thin and not inflated: outline UNINFLATED movable footprints only.
    movable_outline = _outline_mask(movable_uninflated.astype(bool), thickness_px=1)
    rgb[movable_outline] = _hex_to_rgb(COLOR_OUTLINE)

    extent = (data.bounds[0], data.bounds[1], data.bounds[2], data.bounds[3])
    ax.imshow(rgb.transpose(1, 0, 2), origin="lower", extent=extent, interpolation="nearest")
    ax.set_xlim(data.bounds[0], data.bounds[1])
    ax.set_ylim(data.bounds[2], data.bounds[3])
    ax.set_axis_off()

    _draw_robot_footprint(ax, data, label="Robot")

    if data.goal_pose:
        goal_x, goal_y, _ = data.goal_pose
        ax.scatter([goal_x], [goal_y], c=COLOR_GOAL_MARKER_HALO, marker="*", s=210, label=None)
        ax.scatter(
            [goal_x],
            [goal_y],
            c=COLOR_GOAL_MARKER_FILL,
            marker="*",
            edgecolors=COLOR_GOAL_MARKER_OUTLINE,
            linewidths=1.4,
            s=140,
            label="Goal",
        )

    region_effects = [patheffects.withStroke(linewidth=2, foreground="white")]
    obstacle_mask = (wall_uninflated | movable_uninflated).astype(bool)
    dist = _distance_to_obstacles(obstacle_mask)
    region_display_map = _compute_between_obstacles_region_label_map(data)

    used: List[Tuple[float, float]] = []
    min_sep_m = 0.15
    for region_id, label in data.region_labels.items():
        coords = np.argwhere(data.region_map == region_id)
        if coords.size == 0:
            continue
        display = _display_region_label(data, str(label), mapping=region_display_map)

        coords_to_consider = coords
        if display == "r4":
            inflated_occ = data.dynamic_grid == -1
            inflation_band = inflated_occ & ~(data.uninflated_grid == -1)
            keep = ~inflation_band[coords[:, 0], coords[:, 1]]
            if bool(np.any(keep)):
                coords_to_consider = coords[keep]

        # Choose a cell deep inside the region.
        best_gx = int(coords[0, 0])
        best_gy = int(coords[0, 1])
        best_d = -1
        for gx, gy in coords_to_consider:
            d = int(dist[int(gx), int(gy)])
            if d > best_d:
                best_d = d
                best_gx = int(gx)
                best_gy = int(gy)
        world_x = data.bounds[0] + (best_gx + 0.5) * data.resolution
        world_y = data.bounds[2] + (best_gy + 0.5) * data.resolution
        if str(label).lower().startswith("robot"):
            world_y = max(float(data.bounds[2]), world_y - 0.25)
        if display == "r4":
            world_x = min(float(data.bounds[1]) - 1e-6, float(world_x) + 0.08)

        # Basic separation between region labels.
        ok = True
        for ux, uy in used:
            if math.hypot(world_x - ux, world_y - uy) < min_sep_m:
                ok = False
                break
        if not ok:
            continue
        used.append((world_x, world_y))

        ax.text(
            world_x,
            world_y,
            display,
            fontsize=7,
            ha="center",
            va="center",
            color="#212121",
            path_effects=region_effects,
        )

    # Object labels (centered on the objects).
    obj_effects = [patheffects.withStroke(linewidth=2, foreground="white")]
    for obj in data.movable_objects:
        cx = float(obj.get("x", 0.0))
        cy = float(obj.get("y", 0.0))
        ax.text(
            cx,
            cy,
            _short_object_label(str(obj.get("name", ""))),
            fontsize=7,
            ha="center",
            va="center",
            color="#212121",
            path_effects=obj_effects,
        )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=7,
    )


def inflated_rgb_u8(data: WavefrontSnapshotData) -> NDArray[np.uint8]:
    """Return an inflated-panel RGB image (uint8) in PIL layout (top-left origin)."""
    width, height = data.dynamic_grid.shape

    occupied_uninflated = data.uninflated_grid == -1
    movable_uninflated = _movable_mask_uninflated(data)
    wall_uninflated = occupied_uninflated & ~movable_uninflated

    target_mask = np.zeros_like(movable_uninflated, dtype=bool)
    if data.target_object:
        for obj in data.movable_objects:
            if str(obj.get("name", "")) == data.target_object:
                target_mask = _rasterize_movable_object_mask(data, obj, inflated=False)
                break

    rgb = np.empty((width, height, 3), dtype=np.float32)
    rgb[:, :, :] = _hex_to_rgb(COLOR_BACKGROUND)

    region_colors = _compute_region_color_map(data)
    for region_id, label in data.region_labels.items():
        mask = data.region_map == region_id
        if np.any(mask):
            rgb[mask] = _hex_to_rgb(region_colors.get(str(label), _region_label_color(str(label))))

    rgb[wall_uninflated] = _hex_to_rgb(COLOR_WALL)
    rgb[movable_uninflated & ~target_mask] = _hex_to_rgb(COLOR_MOVABLE)
    rgb[target_mask] = _hex_to_rgb(COLOR_TARGET)

    inflated_occ = data.dynamic_grid == -1
    inflation_band = inflated_occ & ~(data.uninflated_grid == -1)
    inflation_outline = _outline_mask(inflation_band.astype(bool), thickness_px=1)
    rgb[inflation_outline] = _hex_to_rgb(COLOR_OUTLINE)

    img = (rgb.transpose(1, 0, 2) * 255.0).clip(0, 255).astype(np.uint8)
    return np.flipud(img)


def _plot_region_graph(ax: Any, data: WavefrontSnapshotData) -> None:
    ax.set_title("Region Connectivity")
    graph = cast(Any, nx).Graph()
    for region, neighbors in data.adjacency.items():
        region_s = str(region)
        graph.add_node(region_s)
        for neighbor in neighbors:
            graph.add_edge(region_s, str(neighbor))

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No adjacency data", ha="center", va="center")
        ax.axis("off")
        return

    region_display_map = _compute_between_obstacles_region_label_map(data)

    def _find_node(predicate: Any) -> Optional[str]:
        for node in graph.nodes():
            if predicate(str(node)):
                return str(node)
        return None

    robot_node = _find_node(lambda s: s.lower() in {"robot", "robot_goal"}) or _find_node(lambda s: "robot" in s.lower())
    goal_node = _find_node(lambda s: s.lower() == "goal") or _find_node(lambda s: ("goal" in s.lower()) and ("robot" not in s.lower()))

    # If we can identify all key nodes, use the requested compact "box perimeter" layout.
    by_display: Dict[str, str] = {}
    for node in graph.nodes():
        disp = _display_region_label(data, str(node), mapping=region_display_map)
        if disp in {"robot", "goal", "r2", "r3", "r4", "r5"}:
            by_display[disp] = str(node)

    layout: Dict[str, Tuple[float, float]] = {}
    used: set[str] = set()
    if {"robot", "goal", "r2", "r3", "r4", "r5"}.issubset(set(by_display.keys())):
        # Tight box: nodes are aligned to a rectangle perimeter and packed closely.
        width_m = float(data.bounds[1]) - float(data.bounds[0])
        height_m = float(data.bounds[3]) - float(data.bounds[2])
        aspect = 1.0
        if width_m > 1e-9 and height_m > 1e-9:
            aspect = max(0.5, min(2.5, height_m / width_m))

        a = 0.34
        b = a * aspect
        layout = {
            by_display["robot"]: (-a, b),   # top-left
            by_display["r2"]: (a, b),       # top-right
            by_display["r3"]: (-a, 0.0),    # mid-left
            by_display["r4"]: (-a, -b),     # bottom-left
            by_display["r5"]: (a, 0.0),     # mid-right
            by_display["goal"]: (a, -b),    # bottom-right
        }
        used = set(layout.keys())

    main_path: List[str] = []
    if robot_node is not None and goal_node is not None and cast(Any, nx).has_path(graph, robot_node, goal_node):
        main_path = cast(List[str], cast(Any, nx).shortest_path(graph, robot_node, goal_node))

    # Prefer a deterministic "unit-edge" layout when the robot->goal component is a simple chain.
    # This keeps all edges exactly the same length for the common region-graph case (a path graph),
    # and avoids nodes getting pulled sideways just to satisfy equal-length constraints.
    if (not layout) and robot_node is not None and goal_node is not None and cast(Any, nx).has_path(graph, robot_node, goal_node):
        component = cast(Any, nx).node_connected_component(graph, robot_node)
        if goal_node in component:
            sub = graph.subgraph(component)
            degrees = [deg for _, deg in sub.degree()]
            if sub.number_of_edges() == sub.number_of_nodes() - 1 and (max(degrees) if degrees else 0) <= 2:
                path = cast(List[str], cast(Any, nx).shortest_path(sub, robot_node, goal_node))
                step = 1.0
                y_top = 0.5 * step * (len(path) - 1)
                for idx, node in enumerate(path):
                    layout[str(node)] = (0.0, float(y_top - idx * step))
                used.update(layout.keys())

    # Fallback: compact, anchored spring layout (still tries to keep edge lengths uniform).
    if not layout:
        initial_pos: Dict[str, Tuple[float, float]] = {}
        fixed: List[str] = []
        # Anchor the robot->goal path as a vertical column for readability.
        if main_path and len(main_path) >= 2:
            denom = max(1, (len(main_path) - 1))
            for idx, node in enumerate(main_path):
                y = 1.0 - 2.0 * float(idx) / float(denom)
                initial_pos[str(node)] = (0.0, float(y))
                if str(node) not in fixed:
                    fixed.append(str(node))
        else:
            if robot_node is not None:
                initial_pos[robot_node] = (0.0, 1.0)
                fixed.append(robot_node)
            if goal_node is not None and goal_node != robot_node:
                initial_pos[goal_node] = (0.0, -1.0)
                fixed.append(goal_node)

        isolated = [str(n) for n in graph.nodes() if graph.degree[n] == 0 and str(n) not in fixed]
        for idx, node in enumerate(sorted(isolated)):
            initial_pos[node] = (0.8 + 0.2 * (idx % 2), 0.8 - 0.25 * (idx // 2))

        num_nodes = max(1, graph.number_of_nodes())
        k = 0.35 / math.sqrt(num_nodes)
        raw = cast(Any, nx).spring_layout(
            graph,
            seed=42,
            k=k,
            iterations=200,
            pos=initial_pos or None,
            fixed=fixed or None,
        )

        xs = [pos[0] for pos in raw.values()]
        ys = [pos[1] for pos in raw.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(1e-9, max_x - min_x)
        height = max(1e-9, max_y - min_y)
        scale = 1.6 / max(width, height)
        for node, (x, y) in list(raw.items()):
            raw[node] = ((x - (min_x + max_x) * 0.5) * scale, (y - (min_y + max_y) * 0.5) * scale)
        if robot_node is not None:
            raw[robot_node] = (0.0, 1.0)
        if goal_node is not None and goal_node != robot_node:
            raw[goal_node] = (0.0, -1.0)
        for node in fixed:
            if node in initial_pos:
                raw[node] = initial_pos[node]

        fixed_nodes = set(fixed)
        target_len = 0.55
        points: Dict[str, NDArray[np.float64]] = {
            str(node): np.array([float(x), float(y)], dtype=np.float64)
            for node, (x, y) in raw.items()
        }
        edges = [(str(u), str(v)) for u, v in graph.edges()]
        for _ in range(250):
            for u, v in edges:
                pu = points[u]
                pv = points[v]
                diff = pv - pu
                dist = float(np.hypot(diff[0], diff[1]))
                if dist < 1e-9:
                    continue
                direction = diff / dist
                delta = (dist - target_len) * 0.5
                if u not in fixed_nodes:
                    points[u] = pu + direction * delta
                if v not in fixed_nodes:
                    points[v] = pv - direction * delta
            for node, p in points.items():
                if node in fixed_nodes:
                    continue
                p[0] *= 0.9

        layout = {node: (float(p[0]) * 0.35, float(p[1])) for node, p in points.items()}
        used = set(layout.keys())

    # Place any remaining nodes (other components) near the main column.
    remaining = [str(n) for n in graph.nodes() if str(n) not in used]
    if remaining:
        base_y = min((y for _, y in layout.values()), default=-1.0) - 1.2
        for idx, node in enumerate(sorted(remaining)):
            layout[node] = (1.0 + 0.2 * (idx % 2), float(base_y - 0.9 * (idx // 2)))
    region_colors = _compute_region_color_map(data)
    node_colors = [region_colors.get(str(region), _region_label_color(str(region))) for region in graph.nodes()]
    node_labels = {str(node): _display_region_label(data, str(node), mapping=region_display_map) for node in graph.nodes()}

    cast(Any, nx).draw_networkx(
        graph,
        pos=layout,
        ax=ax,
        labels=node_labels,
        with_labels=True,
        node_color=node_colors,
        node_size=1200,
        font_size=8,
        font_weight="bold",
        edge_color="#555555",
        linewidths=1.2,
        edgecolors="white",
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
                    display = ", ".join(_short_object_label(obj) for obj in objects[:3])
                    if len(objects) > 3:
                        display += ", …"
                    edge_labels[(region, neighbor)] = display
        if edge_labels:
            # Keep edge labels horizontal for readability (networkx defaults to rotating with edges).
            try:
                cast(Any, nx).draw_networkx_edge_labels(
                    graph,
                    pos=layout,
                    edge_labels=edge_labels,
                    font_size=7,
                    ax=ax,
                    label_pos=0.5,
                    rotate=False,
                )
            except TypeError:
                cast(Any, nx).draw_networkx_edge_labels(
                    graph,
                    pos=layout,
                    edge_labels=edge_labels,
                    font_size=7,
                    ax=ax,
                    label_pos=0.5,
                )
    ax.axis("off")
    if {"robot", "goal", "r2", "r3", "r4", "r5"}.issubset(set(by_display.keys())):
        m = 0.12 * max(a, b)
        ax.set_xlim(-(a + m), (a + m))
        ax.set_ylim(-(b + m), (b + m))
        ax.set_aspect("equal", adjustable="box")


def _draw_robot_footprint(ax: Any, data: WavefrontSnapshotData, label: Optional[str] = None) -> None:
    hx = abs(float(data.robot_half_extent[0]))
    hy = abs(float(data.robot_half_extent[1]))
    x, y, theta = float(data.robot_pose[0]), float(data.robot_pose[1]), float(data.robot_pose[2])

    # Fallback marker for malformed metadata.
    if hx <= 0.0 or hy <= 0.0:
        ax.scatter(
            [x],
            [y],
            c=COLOR_ROBOT_MARKER,
            marker="o",
            s=90,
            edgecolors=COLOR_ROBOT_MARKER_OUTLINE,
            linewidths=1.2,
            label=label,
        )
        return

    corners_local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    c = math.cos(theta)
    s = math.sin(theta)
    corners_world: List[Tuple[float, float]] = []
    for lx, ly in corners_local:
        wx = x + c * lx - s * ly
        wy = y + s * lx + c * ly
        corners_world.append((wx, wy))

    footprint = MplPolygon(
        corners_world,
        closed=True,
        facecolor=COLOR_ROBOT_MARKER,
        edgecolor=COLOR_ROBOT_MARKER_OUTLINE,
        linewidth=1.2,
        label=label,
        zorder=5,
    )
    ax.add_patch(footprint)

    # Heading cue from center to front face.
    front_x = x + c * hx
    front_y = y + s * hx
    ax.plot(
        [x, front_x],
        [y, front_y],
        color=COLOR_ROBOT_MARKER_OUTLINE,
        linewidth=1.2,
        zorder=6,
    )


def create_figure(data: WavefrontSnapshotData) -> Figure:
    subplot_result = cast(Any, plt).subplots(1, 3, figsize=(18, 6))
    fig, axes = subplot_result
    _plot_environment(axes[0], data)
    _plot_heatmap(axes[1], data)
    _plot_region_graph(axes[2], data)
    fig.tight_layout()
    return fig


def create_panel_figures(data: WavefrontSnapshotData) -> Dict[str, Figure]:
    figures: Dict[str, Figure] = {}

    fig_env, ax_env = cast(Any, plt).subplots(1, 1, figsize=(7, 7))
    _plot_environment(ax_env, data)
    cast(Any, fig_env).tight_layout()
    figures["environment"] = fig_env

    fig_heat, ax_heat = cast(Any, plt).subplots(1, 1, figsize=(7, 7))
    _plot_heatmap(ax_heat, data)
    cast(Any, fig_heat).tight_layout()
    figures["inflated"] = fig_heat

    fig_graph, ax_graph = cast(Any, plt).subplots(1, 1, figsize=(7, 7))
    _plot_region_graph(ax_graph, data)
    cast(Any, fig_graph).tight_layout()
    figures["graph"] = fig_graph

    return figures


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


def visualize_snapshot_panels(
    directory: Path,
    prefix: str = "snapshot",
    *,
    show: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    data = load_snapshot(directory, prefix)
    figures = create_panel_figures(data)

    out_dir = Path(output_dir) if output_dir is not None else Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}
    for name, fig in figures.items():
        out = out_dir / f"{prefix}_panel_{name}.png"
        cast(Any, fig).savefig(str(out), dpi=300, bbox_inches="tight", pad_inches=0.02)
        outputs[name] = out
        if show:
            cast(Any, plt).show()
        else:
            cast(Any, plt).close(fig)

    return outputs


__all__ = [
    "WavefrontSnapshotData",
    "load_snapshot",
    "create_figure",
    "create_panel_figures",
    "visualize_snapshot",
    "visualize_snapshot_panels",
]
