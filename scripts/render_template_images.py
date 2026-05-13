"""Render top-down 2D images of template XMLs (walls + floor outline + robot pose)."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def parse_walls(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    walls = []
    for body in root.iter("body"):
        if body.get("name") == "walls":
            for geom in body.findall("geom"):
                pos = [float(v) for v in geom.get("pos").split()]
                size = [float(v) for v in geom.get("size").split()]
                walls.append((pos[0], pos[1], size[0], size[1]))  # cx, cy, hx, hy
    return walls


def parse_obstacles(xml_path: Path):
    """Return list of (cx, cy, hx, hy, yaw_deg) for each movable obstacle."""
    import re
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obs = []
    for body in root.iter("body"):
        name = body.get("name", "")
        if not re.match(r"obstacle_\d+_movable", name):
            continue
        for geom in body.findall("geom"):
            pos = [float(v) for v in geom.get("pos", "0 0 0").split()]
            size = [float(v) for v in geom.get("size", "0 0 0").split()]
            euler = geom.get("euler", "0 0 0").split()
            yaw = float(euler[2]) if len(euler) >= 3 else 0.0
            obs.append((pos[0], pos[1], size[0], size[1], yaw))
    return obs


def parse_goal(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for site in root.iter("site"):
        if site.get("name") == "goal":
            pos = [float(v) for v in site.get("pos", "0 0 0").split()]
            sz = float(site.get("size", "0.1").split()[0])
            return (pos[0], pos[1], sz)
    return None


def parse_robot_pose(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Try point robot first
    for body in root.iter("body"):
        if body.get("name") == "robot":
            for geom in body.findall("geom"):
                if geom.get("name") == "robot":
                    pos = [float(v) for v in geom.get("pos").split()]
                    return (pos[0], pos[1])
    # Fall back to car body (uses freejoint, pose is on the body element itself)
    for body in root.iter("body"):
        if body.get("name") == "car":
            pos = [float(v) for v in body.get("pos", "0 0 0").split()]
            return (pos[0], pos[1])
    return None


def bounds_from_walls(walls):
    if not walls:
        return (-3, 3, -3, 3)
    xs = [cx - hx for cx, cy, hx, hy in walls] + [cx + hx for cx, cy, hx, hy in walls]
    ys = [cy - hy for cx, cy, hx, hy in walls] + [cy + hy for cx, cy, hx, hy in walls]
    return (min(xs), max(xs), min(ys), max(ys))


def render(xml_path: Path, out_path: Path, label: str | None = None):
    from matplotlib.transforms import Affine2D
    walls = parse_walls(xml_path)
    robot = parse_robot_pose(xml_path)
    obstacles = parse_obstacles(xml_path)
    goal = parse_goal(xml_path)
    xmin, xmax, ymin, ymax = bounds_from_walls(walls)

    fig, ax = plt.subplots(figsize=(6, 6))
    # Floor
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           fill=True, facecolor="#f5f0e1", edgecolor="black", linewidth=1.5))
    # Walls
    for cx, cy, hx, hy in walls:
        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                fill=True, facecolor="#666", edgecolor="black", linewidth=0.5))
    # Obstacles (movable, yellow rotated rectangles)
    for cx, cy, hx, hy, yaw_deg in obstacles:
        rect = Rectangle((-hx, -hy), 2 * hx, 2 * hy,
                         fill=True, facecolor="gold", edgecolor="black",
                         linewidth=0.5, alpha=0.85)
        rect.set_transform(Affine2D().rotate_deg(yaw_deg).translate(cx, cy) + ax.transData)
        ax.add_patch(rect)
    # Goal site (red dashed circle)
    if goal is not None:
        ax.add_patch(plt.Circle((goal[0], goal[1]), goal[2],
                                fill=True, facecolor="red", edgecolor="darkred",
                                alpha=0.5, linewidth=1.5, linestyle="--", label="goal"))
    # Robot
    if robot is not None:
        ax.plot(robot[0], robot[1], "o", color="green", markersize=10, label="robot")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlim(xmin - 0.2, xmax + 0.2)
    ax.set_ylim(ymin - 0.2, ymax + 0.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if label:
        ax.set_title(label, fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-root", default="templates/aug9")
    ap.add_argument("--out-root", default="templates/aug9_images")
    ap.add_argument("--grid-png", default="templates/aug9_grid.png",
                    help="Output path for combined grid summary image")
    args = ap.parse_args()

    root = Path(args.templates_root)
    out_root = Path(args.out_root)
    xmls = sorted(root.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"no benchmark_*.xml under {root}")

    print(f"Rendering {len(xmls)} templates...")
    for xml in xmls:
        rel = xml.relative_to(root).with_suffix(".png")
        out = out_root / rel
        label = str(xml.relative_to(root))
        render(xml, out, label=label)
        print(f"  {out}")

    # Combined grid: flat layout, ceil(sqrt(N)) cols
    import math
    from matplotlib.transforms import Affine2D
    n = len(xmls)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
    for k, xml in enumerate(xmls):
        ax = axes[k // cols][k % cols]
        walls = parse_walls(xml)
        robot = parse_robot_pose(xml)
        obstacles = parse_obstacles(xml)
        goal = parse_goal(xml)
        xmin, xmax, ymin, ymax = bounds_from_walls(walls)
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=True, facecolor="#f5f0e1", edgecolor="black", linewidth=1))
        for cx, cy, hx, hy in walls:
            ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                   fill=True, facecolor="#555"))
        for cx, cy, hx, hy, yaw_deg in obstacles:
            rect = Rectangle((-hx, -hy), 2 * hx, 2 * hy,
                             fill=True, facecolor="gold", edgecolor="black",
                             linewidth=0.4, alpha=0.85)
            rect.set_transform(Affine2D().rotate_deg(yaw_deg).translate(cx, cy) + ax.transData)
            ax.add_patch(rect)
        if goal is not None:
            ax.add_patch(plt.Circle((goal[0], goal[1]), goal[2],
                                    fill=True, facecolor="red", edgecolor="darkred",
                                    alpha=0.5, linewidth=1, linestyle="--"))
        if robot is not None:
            ax.plot(robot[0], robot[1], "o", color="green", markersize=6)
        ax.set_xlim(xmin - 0.2, xmax + 0.2)
        ax.set_ylim(ymin - 0.2, ymax + 0.2)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(str(xml.relative_to(root)), fontsize=7)
    # Hide unused cells
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_axis_off()
    plt.tight_layout()
    Path(args.grid_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.grid_png, dpi=120)
    plt.close()
    print(f"\nGrid summary: {args.grid_png}")


if __name__ == "__main__":
    main()
