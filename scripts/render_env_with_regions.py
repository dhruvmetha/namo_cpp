"""Render an environment, its wavefront regions, and its C++ region graph."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib import colormaps
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAMO_CONFIG = PROJECT_ROOT / "config" / "namo_config_car.yaml"
sys.path.insert(0, str(PROJECT_ROOT / "python"))

import namo_rl  # noqa: E402
from namo.planners import get_region_snapshot  # noqa: E402
from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter  # noqa: E402

# Reuse the parsing helpers from the existing template renderer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_template_images import (  # noqa: E402
    parse_walls, parse_robot_pose, parse_obstacles, parse_goal, bounds_from_walls,
)


def render_one(xml_path: Path, out_path: Path, namo_config: Path,
               resolution: float = 0.01):
    walls = parse_walls(xml_path)
    robot = parse_robot_pose(xml_path)
    obstacles = parse_obstacles(xml_path)
    goal = parse_goal(xml_path)
    xmin, xmax, ymin, ymax = bounds_from_walls(walls)

    # The Python exporter provides the per-cell raster needed for the middle panel.
    # It reads robot size and inflation margin from the same config as C++.
    env = namo_rl.RLEnvironment(str(xml_path), str(namo_config), visualize=False)
    exporter = WavefrontSnapshotExporter(env, resolution=resolution)
    rng = np.random.default_rng(0)
    raster_snapshot = exporter.build_snapshot(
        xml_path=str(xml_path),
        config_path=str(namo_config),
        goal_radius=None,
        goals_per_region=0,
        rng=rng,
    )
    region_map = raster_snapshot.region_map
    raster_region_labels = raster_snapshot.region_labels

    # Planning uses the C++ snapshot. Use it as the source of truth for the graph rather
    # than drawing the Python exporter's independently reconstructed connectivity.
    graph_snapshot = get_region_snapshot(
        env,
        goals_per_region=0,
        goal_radius=None,
        local_info_only=False,
        use_cpp_unified=True,
        use_xml_goal=True,
    )
    region_labels = graph_snapshot["region_labels"]
    adjacency = graph_snapshot["adjacency"]
    edge_objects = graph_snapshot["edge_objects"]
    robot_label = graph_snapshot["robot_label"]
    goal_label = graph_snapshot["goal_label"]

    raster_label_names = set(raster_region_labels.values())
    cpp_label_names = set(region_labels.values())
    if raster_label_names != cpp_label_names or raster_snapshot.adjacency != adjacency:
        print("  [WARN] Python raster connectivity differs from the C++ planning snapshot")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # ===== Left panel: environment =====
    ax = axes[0]
    ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=True, facecolor="#f5f0e1",
                                    edgecolor="black", linewidth=1.5))
    for cx, cy, hx, hy in walls:
        ax.add_patch(patches.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                        fill=True, facecolor="#666",
                                        edgecolor="black", linewidth=0.5))
    for cx, cy, hx, hy, yaw_deg in obstacles:
        rect = patches.Rectangle((-hx, -hy), 2 * hx, 2 * hy,
                                 fill=True, facecolor="gold", edgecolor="black",
                                 linewidth=0.5, alpha=0.85)
        rect.set_transform(Affine2D().rotate_deg(yaw_deg).translate(cx, cy) + ax.transData)
        ax.add_patch(rect)
    if goal is not None:
        ax.add_patch(plt.Circle((goal[0], goal[1]), goal[2],
                                fill=True, facecolor="red", edgecolor="darkred",
                                alpha=0.5, linewidth=1.5, linestyle="--"))
    if robot is not None:
        ax.plot(robot[0], robot[1], "o", color="green", markersize=10)
    ax.set_xlim(xmin - 0.1, xmax + 0.1)
    ax.set_ylim(ymin - 0.1, ymax + 0.1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Environment")

    # ===== Compute robot/goal regions first (so we can color them in both panels) =====
    rm = np.asarray(region_map)
    W, H = rm.shape  # x_dim, y_dim

    # Use the wavefront's own world bounds, not bounds derived from walls — those can
    # differ by half a wall-thickness and shift the cell lookup by one.
    wf_bounds = exporter.bounds  # (xmin, xmax, ymin, ymax)
    wxmin, wxmax, wymin, wymax = wf_bounds

    def world_to_grid_pre(px, py):
        gx = int((px - wxmin) / (wxmax - wxmin) * W)
        gy = int((py - wymin) / (wymax - wymin) * H)
        return max(0, min(W - 1, gx)), max(0, min(H - 1, gy))

    def lookup_region(px, py, label):
        gx, gy = world_to_grid_pre(px, py)
        v = int(rm[gx, gy])
        # Probe a 3x3 neighborhood if direct cell isn't a labeled region — the world point
        # may snap to a wall cell or an inflated boundary.
        if v <= 0:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        nv = int(rm[nx, ny])
                        if nv > 0:
                            v = nv
                            break
                if v > 0:
                    break
        if v <= 0:
            print(f"  [WARN] {label} at ({px:.3f},{py:.3f}) → grid ({gx},{gy}) = {int(rm[gx,gy])} (no labeled region nearby)")
            return None
        return v

    robot_region_id = lookup_region(*robot, "robot") if robot is not None else None
    goal_region_id = lookup_region(goal[0], goal[1], "goal") if goal is not None else None

    # ===== Middle panel: region map (with robot=red, goal=green override) =====
    ax = axes[1]
    unique_labels = sorted(set(int(x) for x in rm.flatten()))

    # Sanity check: every positive value in region_map should appear in region_labels.
    # (Value 0 = occupied cells, never visited by bfs; this is not a mismatch.)
    rmap_positive = sorted([l for l in unique_labels if l > 0])
    rlbl_keys = sorted(int(k) for k in raster_region_labels.keys())
    only_in_map = [l for l in rmap_positive if l not in rlbl_keys]
    only_in_lbl = [l for l in rlbl_keys if l not in rmap_positive]
    if only_in_map or only_in_lbl:
        print(f"  [WARN] region_map ↔ region_labels mismatch (positive values only):")
        print(f"         only in region_map:    {only_in_map}")
        print(f"         only in region_labels: {only_in_lbl}")

    # Palette without red/green so non-robot/non-goal regions don't visually clash.
    OTHER_PALETTE = [
        "#1F77B4",  # blue
        "#FF7F0E",  # orange
        "#9467BD",  # purple
        "#8C564B",  # brown
        "#E377C2",  # pink
        "#17BECF",  # cyan
        "#BCBD22",  # olive
        "#7F7F7F",  # mid-gray
        "#AEC7E8",  # light blue
        "#FFBB78",  # light orange
        "#C5B0D5",  # light purple
        "#C49C94",  # light brown
        "#F7B6D2",  # light pink
        "#9EDAE5",  # light cyan
        "#DBDB8D",  # light olive
    ]
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)
    palette = [hex_to_rgb(c) for c in OTHER_PALETTE]

    labeled_ids = {int(k) for k in raster_region_labels.keys()}
    label_to_color = {}
    color_idx = 0
    for lbl in unique_labels:
        if lbl == -2 or lbl == 0:
            # In region_map: 0 = bfs never visited (occupied cells stay at 0). -2 = legacy.
            label_to_color[lbl] = (0.55, 0.55, 0.55)      # gray
        elif lbl == -1 or lbl not in labeled_ids:
            label_to_color[lbl] = (1.0, 1.0, 1.0)         # rare unlabeled free → white
        elif lbl == robot_region_id:
            label_to_color[lbl] = (1.00, 0.255, 0.212)    # red (#FF4136)
        elif lbl == goal_region_id:
            label_to_color[lbl] = (0.180, 0.800, 0.251)   # green (#2ECC40)
        else:
            label_to_color[lbl] = palette[color_idx % len(palette)]
            color_idx += 1

    rgb_xy = np.zeros((W, H, 3), dtype=float)
    for lbl, color in label_to_color.items():
        rgb_xy[rm == lbl] = color
    # Transpose first two axes so imshow shows (rows=y, cols=x).
    ax.imshow(rgb_xy.transpose(1, 0, 2), origin="lower",
              extent=[xmin, xmax, ymin, ymax], aspect="equal",
              interpolation="nearest")

    n_regions = sum(1 for lbl in unique_labels if lbl > 0)
    ax.set_xlim(xmin - 0.1, xmax + 0.1)
    ax.set_ylim(ymin - 0.1, ymax + 0.1)
    ax.set_aspect("equal")
    ax.set_title(f"Wavefront regions ({n_regions})")

    # ===== Right panel: clean adjacency graph (networkx layout) =====
    ax = axes[2]

    import networkx as nx

    G = nx.Graph()
    for label in region_labels.values():
        G.add_node(label)
    for label, neighbors in adjacency.items():
        for neighbor in neighbors:
            if label == neighbor:
                continue
            G.add_edge(label, neighbor)

    # robot_region_id / goal_region_id were computed before the middle panel.

    # Pick a layout.
    if len(G) > 0:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=0, iterations=200)

        # Color nodes same as the wavefront panel (label_to_color built earlier),
        # with robot/goal regions overridden to green/red.
        node_colors = []
        edge_colors = []
        node_sizes = []
        for n in G.nodes():
            if n == robot_label:
                node_colors.append("#FF4136"); edge_colors.append("#7A1009"); node_sizes.append(900)
            elif n == goal_label:
                node_colors.append("#2ECC40"); edge_colors.append("#0E5C18"); node_sizes.append(900)
            else:
                node_colors.append("#AEC7E8")
                edge_colors.append("#222"); node_sizes.append(600)

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#333", width=1.5, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               edgecolors=edge_colors, linewidths=2, node_size=node_sizes)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

        # Edge labels = short obstacle names (e.g., 'o3, o7') for the obstacles that
        # separate the connected pair of regions. Pull from snapshot.edge_objects.
        import re as _re
        def short_obstacle(name: str) -> str:
            m = _re.search(r"obstacle_(\d+)", name)
            return f"o{m.group(1)}" if m else name

        edge_labels = {}
        for u, v in G.edges():
            objs = (edge_objects.get(u, {}).get(v, set())
                    | edge_objects.get(v, {}).get(u, set()))
            if objs:
                edge_labels[(u, v)] = ", ".join(sorted(short_obstacle(o) for o in objs))
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                         font_size=7, font_color="#222",
                                         bbox=dict(boxstyle="round,pad=0.2",
                                                   facecolor="white",
                                                   edgecolor="#666", alpha=0.9))

    # Legend.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF4136",
                   markeredgecolor="#7A1009", markersize=12,
                   label=f"robot region ({robot_label or '?'})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ECC40",
                   markeredgecolor="#0E5C18", markersize=12,
                   label=f"goal region ({goal_label or '?'})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    n_regions = G.number_of_nodes()
    n_edges = G.number_of_edges()
    ax.set_title(f"Region graph ({n_regions} nodes, {n_edges} edges)", fontsize=10)
    ax.set_axis_off()

    plt.suptitle(str(xml_path.name), fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmls", nargs="+", help="Env XML(s) to render")
    ap.add_argument("--namo-config",
                    type=Path,
                    default=DEFAULT_NAMO_CONFIG)
    ap.add_argument("--out-dir", default="env_region_images")
    ap.add_argument("--resolution", type=float, default=0.01)
    args = ap.parse_args()

    namo_config = args.namo_config.resolve()
    if not namo_config.is_file():
        ap.error(f"NAMO config does not exist: {namo_config}")

    out_dir = Path(args.out_dir)
    for xml in args.xmls:
        xml = Path(xml)
        out_path = out_dir / (xml.stem + "_regions.png")
        print(f"Rendering {xml.name} -> {out_path}")
        try:
            render_one(xml, out_path, namo_config, args.resolution)
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    main()
