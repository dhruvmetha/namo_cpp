"""Visualize generated car motion primitives as displacement vectors."""
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
import numpy as np


def load_primitives(path: Path) -> list[tuple[float, float, float, int, int]]:
    """Load primitives from binary .dat file."""
    with open(path, "rb") as f:
        count = struct.unpack("I", f.read(4))[0]
        primitives = []
        for _ in range(count):
            dx, dy, dtheta, edge_idx, push_steps = struct.unpack("fffBB", f.read(14))
            primitives.append((dx, dy, dtheta, edge_idx, push_steps))
    return primitives


def plot_primitives(shape: str, primitives: list, half_sx: float, half_sy: float,
                    output_path: Path, points_per_face: int = 15):
    """Plot displacement vectors for all primitives of one shape."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Car Push Primitives: {shape} ({half_sx*200:.1f}x{half_sy*200:.1f}cm)", fontsize=14)

    # Group by push_steps
    by_depth = {}
    for dx, dy, dtheta, edge_idx, push_steps in primitives:
        if push_steps not in by_depth:
            by_depth[push_steps] = []
        by_depth[push_steps].append((dx, dy, dtheta, edge_idx))

    # Plot 1: Object footprint at end of push, only left/center/right edge per face
    from matplotlib.transforms import Affine2D
    ax = axes[0]
    shown_depths = [d for d in (1, 5, 10) if d in by_depth]
    # Per-face local indices to keep: 0 (left), 14 (right) for points_per_face=15
    selected_local = {0: 'L', points_per_face - 1: 'R'}
    # Restrict to a single face for clarity (face 0 = +x, 1 = +y, 2 = -x, 3 = -y)
    shown_faces = {0}
    face_names = {0: '+x', 1: '+y', 2: '-x', 3: '-y'}
    face_label = ', '.join(f"face {f} ({face_names[f]})" for f in sorted(shown_faces))
    ax.set_title(f"Object pose after push — {face_label}, L/R, depths {shown_depths}")
    # Initial footprint at origin
    rect = patches.Rectangle((-half_sx, -half_sy), 2*half_sx, 2*half_sy,
                              linewidth=2, edgecolor='black', facecolor='lightyellow', alpha=0.5)
    ax.add_patch(rect)

    depth_colors = {1: 'tab:blue', 5: 'tab:orange', 10: 'tab:red'}
    for depth in shown_depths:
        color = depth_colors[depth]
        for dx, dy, dtheta, edge_idx in by_depth[depth]:
            local = edge_idx % points_per_face
            face = edge_idx // points_per_face
            if face not in shown_faces or local not in selected_local:
                continue
            footprint = patches.Rectangle((-half_sx, -half_sy), 2*half_sx, 2*half_sy,
                                          linewidth=1.0, edgecolor=color,
                                          facecolor='none', alpha=0.75)
            t = (Affine2D().rotate(dtheta).translate(dx, dy) + ax.transData)
            footprint.set_transform(t)
            ax.add_patch(footprint)
            # Label deepest push only, to keep clutter down
            if depth == max(shown_depths):
                face = edge_idx // points_per_face
                ax.annotate(f"f{face}{selected_local[local]}", (dx, dy),
                            fontsize=6, ha='center', va='center', color=color)
        ax.plot([], [], color=color, linewidth=2, label=f'depth={depth}')
    ax.legend(loc='upper left', fontsize=8)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # Plot 2: Displacement magnitude by edge
    ax = axes[1]
    ax.set_title("Displacement by edge index")
    for depth in [1, 5, 10]:
        if depth not in by_depth:
            continue
        entries = by_depth[depth]
        edges = [e[3] for e in entries]
        mags = [math.sqrt(e[0]**2 + e[1]**2) * 1000 for e in entries]
        ax.plot(edges, mags, 'o-', markersize=3, label=f'depth={depth}', alpha=0.7)

    ax.set_xlabel('Edge index')
    ax.set_ylabel('Displacement (mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Displacement by face at max depth
    ax = axes[2]
    ax.set_title(f"Push directions by face (depth={max(by_depth.keys())})")
    rect = patches.Rectangle((-half_sx, -half_sy), 2*half_sx, 2*half_sy,
                              linewidth=2, edgecolor='black', facecolor='lightyellow', alpha=0.5)
    ax.add_patch(rect)

    max_depth_entries = by_depth[max(by_depth.keys())]
    face_colors = ['red', 'blue', 'green', 'orange']
    face_names = ['+x face', '+y face', '-x face', '-y face']

    for dx, dy, dtheta, edge_idx in max_depth_entries:
        face = edge_idx // points_per_face
        color = face_colors[face % 4]
        ax.arrow(0, 0, dx, dy, head_width=0.005, head_length=0.003,
                 fc=color, ec=color, alpha=0.6, linewidth=0.8)

    # Legend
    for i, (color, name) in enumerate(zip(face_colors, face_names)):
        ax.plot([], [], color=color, linewidth=2, label=name)
    ax.legend(loc='upper left', fontsize=8)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    SCALE = 0.07 / 0.30
    shapes = {
        "square": (0.35 * SCALE, 0.35 * SCALE),
        "wide":   (0.45 * SCALE, 0.25 * SCALE),
        "tall":   (0.25 * SCALE, 0.45 * SCALE),
    }

    output_dir = PROJECT_ROOT / "artifacts" / "primitives"
    output_dir.mkdir(parents=True, exist_ok=True)

    for shape, (half_sx, half_sy) in shapes.items():
        dat_path = NAMO_ROOT / "data" / f"car_motion_primitives_15_{shape}.dat"
        if not dat_path.exists():
            print(f"Skipping {shape}: {dat_path} not found")
            continue

        primitives = load_primitives(dat_path)
        print(f"{shape}: loaded {len(primitives)} primitives")

        # Stats
        mags = [math.sqrt(dx**2 + dy**2) for dx, dy, _, _, _ in primitives]
        rotations = [abs(dtheta) for _, _, dtheta, _, _ in primitives]
        print(f"  Displacement: {min(mags)*1000:.1f} - {max(mags)*1000:.1f} mm (mean {np.mean(mags)*1000:.1f} mm)")
        print(f"  Rotation: {min(rotations)*180/math.pi:.2f} - {max(rotations)*180/math.pi:.2f} deg")

        plot_primitives(shape, primitives, half_sx, half_sy,
                       output_dir / f"car_primitives_{shape}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
