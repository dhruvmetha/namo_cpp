"""Standard motion primitive visualization.

For each edge point, shows the object's final pose at ALL depths (1-10),
connected as a trajectory. Each trajectory starts at the origin and extends
outward, with a small rotated rectangle at each depth showing the object's
final pose.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAMO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np

from generate_car_primitives import OBJECT_CONFIGS, SCALE


def load_primitives(path: Path) -> list[tuple[float, float, float, int, int]]:
    with open(path, "rb") as f:
        count = struct.unpack("I", f.read(4))[0]
        prims = []
        for _ in range(count):
            dx, dy, dtheta, edge_idx, push_steps = struct.unpack("fffBB", f.read(14))
            prims.append((dx, dy, dtheta, edge_idx, push_steps))
    return prims


def draw_rotated_rect(ax, x, y, theta, half_sx, half_sy, **kwargs):
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    corners = [
        (-half_sx, -half_sy), (half_sx, -half_sy),
        (half_sx, half_sy), (-half_sx, half_sy), (-half_sx, -half_sy)
    ]
    xs = [x + cx * cos_t - cy * sin_t for cx, cy in corners]
    ys = [y + cx * sin_t + cy * cos_t for cx, cy in corners]
    ax.plot(xs, ys, **kwargs)


def plot_shape(obj_config, primitives, output_dir: Path, points_per_face: int = 15):
    half_sx = obj_config.half_size_x * 1000  # mm
    half_sy = obj_config.half_size_y * 1000

    # Group: edge_idx -> [(push_steps, dx, dy, dtheta), ...] sorted by push_steps
    by_edge = {}
    for dx, dy, dtheta, eidx, ps in primitives:
        if eidx not in by_edge:
            by_edge[eidx] = []
        by_edge[eidx].append((ps, dx * 1000, dy * 1000, dtheta))
    for eidx in by_edge:
        by_edge[eidx].sort(key=lambda x: x[0])

    face_colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']  # red blue green orange
    face_names = ['+x face (push -x)', '+y face (push -y)', '-x face (push +x)', '-y face (push +y)']

    # ── Plot 1: Full primitive fan (all edges, all depths) ──────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title(f"{obj_config.name} primitives — all edges, all depths\n"
                 f"Object: {obj_config.half_size_x*200:.1f}x{obj_config.half_size_y*200:.1f}cm", fontsize=13)

    # Draw initial object
    draw_rotated_rect(ax, 0, 0, 0, half_sx, half_sy, color='black', linewidth=2.5, zorder=10)

    for eidx, entries in sorted(by_edge.items()):
        face = eidx // points_per_face
        color = face_colors[face % 4]

        # Trajectory: origin → depth1 → depth2 → ... → depth10
        xs = [0.0]
        ys = [0.0]
        for ps, dx, dy, dtheta in entries:
            xs.append(dx)
            ys.append(dy)

        # Draw trajectory line
        ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.7, zorder=1)

        # Draw small object outline at each depth
        for ps, dx, dy, dtheta in entries:
            scale = 0.4  # draw smaller rectangles for clarity
            draw_rotated_rect(ax, dx, dy, dtheta,
                             half_sx * scale, half_sy * scale,
                             color=color, linewidth=0.4, alpha=0.2)

        # Endpoint marker at max depth
        last = entries[-1]
        ax.plot(last[1], last[2], 'o', color=color, markersize=2.5, alpha=0.6, zorder=5)

    for c, n in zip(face_colors, face_names):
        ax.plot([], [], color=c, linewidth=2, label=n)
    ax.legend(fontsize=10, loc='upper left')

    lim = 550
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    plt.tight_layout()
    path = output_dir / f"primitives_fan_{obj_config.name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path.name}")

    # ── Plot 2: Per-face detail (4 subplots) ────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f"{obj_config.name} primitives — per face detail\n"
                 f"Object: {obj_config.half_size_x*200:.1f}x{obj_config.half_size_y*200:.1f}cm", fontsize=14)

    for face_idx in range(4):
        ax = axes[face_idx // 2][face_idx % 2]
        ax.set_title(f"{face_names[face_idx]}", fontsize=11)
        color = face_colors[face_idx]

        # Draw initial object
        draw_rotated_rect(ax, 0, 0, 0, half_sx, half_sy, color='black', linewidth=2, zorder=10)

        edge_start = face_idx * points_per_face
        edge_end = edge_start + points_per_face

        for eidx in range(edge_start, edge_end):
            if eidx not in by_edge:
                continue
            entries = by_edge[eidx]

            # Point index within face (0=one extreme, 14=other extreme)
            local_idx = eidx - edge_start
            is_center = (local_idx == points_per_face // 2)
            is_extreme = (local_idx == 0 or local_idx == points_per_face - 1)

            lw = 2.0 if is_center else (1.2 if is_extreme else 0.6)
            alpha = 0.9 if is_center else (0.7 if is_extreme else 0.3)

            xs = [0.0]
            ys = [0.0]
            for ps, dx, dy, dtheta in entries:
                xs.append(dx)
                ys.append(dy)

            ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw, zorder=3)

            # Draw final object outline at max depth
            last = entries[-1]
            rect_alpha = 0.6 if (is_center or is_extreme) else 0.15
            rect_lw = 1.5 if is_center else (1.0 if is_extreme else 0.3)
            draw_rotated_rect(ax, last[1], last[2], last[3],
                             half_sx * 0.5, half_sy * 0.5,
                             color=color, linewidth=rect_lw, alpha=rect_alpha)

            # Label extremes and center
            if is_center or is_extreme:
                label = "center" if is_center else ("edge 0" if local_idx == 0 else f"edge {points_per_face-1}")
                # Small dot at max depth endpoint
                ax.plot(last[1], last[2], 'o', color=color, markersize=4, alpha=0.8)
                ax.annotate(label, (last[1], last[2]), fontsize=7, alpha=0.7,
                           textcoords="offset points", xytext=(5, 5))

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

    plt.tight_layout()
    path = output_dir / f"primitives_per_face_{obj_config.name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path.name}")

    # ── Plot 3: Single face with full object outlines at each depth ──
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    face_idx = 2  # -x face (pushing +x) — most intuitive to view
    ax.set_title(f"{obj_config.name} — {face_names[face_idx]}, full object outlines at each depth", fontsize=12)

    draw_rotated_rect(ax, 0, 0, 0, half_sx, half_sy, color='black', linewidth=2.5, zorder=10)

    edge_start = face_idx * points_per_face
    # Show 5 representative edges: 0, 3, 7 (center), 11, 14
    representative = [0, 3, points_per_face // 2, points_per_face - 4, points_per_face - 1]
    line_colors = ['#e41a1c', '#ff7f00', '#000000', '#377eb8', '#4daf4a']
    line_labels = ['extreme bottom', 'off-center bottom', 'center', 'off-center top', 'extreme top']

    for rep_idx, (local_idx, lc, ll) in enumerate(zip(representative, line_colors, line_labels)):
        eidx = edge_start + local_idx
        if eidx not in by_edge:
            continue
        entries = by_edge[eidx]

        xs = [0.0]
        ys = [0.0]
        for ps, dx, dy, dtheta in entries:
            xs.append(dx)
            ys.append(dy)

        ax.plot(xs, ys, color=lc, linewidth=2, alpha=0.8, label=ll, zorder=5)

        # Draw object outline at each depth
        for ps, dx, dy, dtheta in entries:
            alpha = 0.15 + 0.07 * ps  # more opaque at higher depths
            draw_rotated_rect(ax, dx, dy, dtheta, half_sx, half_sy,
                             color=lc, linewidth=0.8, alpha=alpha)

        # Endpoint
        last = entries[-1]
        ax.plot(last[1], last[2], 'D', color=lc, markersize=5, zorder=6)

    ax.legend(fontsize=10, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    # Auto-scale to content
    ax.autoscale()
    margin = 50
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] - margin, xlim[1] + margin)
    ax.set_ylim(ylim[0] - margin, ylim[1] + margin)

    plt.tight_layout()
    path = output_dir / f"primitives_outlines_{obj_config.name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path.name}")


def main():
    output_dir = PROJECT_ROOT / "artifacts" / "object_paths"
    output_dir.mkdir(parents=True, exist_ok=True)

    for obj in OBJECT_CONFIGS:
        dat_path = NAMO_ROOT / "data" / f"car_motion_primitives_15_{obj.name}.dat"
        if not dat_path.exists():
            print(f"Skipping {obj.name}: {dat_path} not found")
            continue

        primitives = load_primitives(dat_path)
        print(f"{obj.name} ({len(primitives)} primitives):")
        plot_shape(obj, primitives, output_dir)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
