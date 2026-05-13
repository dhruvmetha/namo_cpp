#!/usr/bin/env python3
"""
Comprehensive F-characterization analysis.
Loads exhaustive trial logs, computes statistics, generates visualizations.

Usage:
    python analyze_F.py --data-dir /tmp/f_char_50/modular_data_westeros
"""

import os
import sys
import pickle
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

# Add project root so we can import visualize_environment
NAMO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NAMO_ROOT / "python"))
from environment_selection import visualize_environment

OUT_DIR = Path(__file__).resolve().parent


# ── Data loading ─────────────────────────────────────────────────────────

def load_all_instances(data_dir):
    """Load all pkl files, deduplicate by (xml, object, region), return list of dicts."""
    instances = []
    seen = set()

    pkl_files = sorted(Path(data_dir).glob("*_results.pkl"))
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        for ep in data["episode_results"]:
            stats = ep["algorithm_stats"]
            tlog = stats.get("primitive_trial_log")
            if not tlog:
                continue

            xml_file = ep["xml_file"]
            obj_id = stats.get("chosen_object_id", "?")
            region = stats.get("neighbour_region_label", "?")
            key = (xml_file, obj_id, region)

            if key in seen:
                continue
            seen.add(key)

            # Build 60x10 grid (edge_idx x depth)
            grid = np.full((60, 10), np.nan)  # nan = not evaluated
            for trial in tlog:
                ei, d = trial["edge_idx"], trial["depth"]
                grid[ei, d] = 1.0 if trial["success"] else 0.0

            evaluated = ~np.isnan(grid)
            R = int(evaluated.sum())
            F = int(np.nansum(grid))
            ratio = F / R if R > 0 else 0.0

            # Wall collision stats among feasible (handle overlap)
            wall_in_F = sum(1 for t in tlog if t["success"] and t["wall_collision"])
            mov_in_F = sum(1 for t in tlog if t["success"] and t["movable_collisions"])
            any_contact_in_F = sum(1 for t in tlog if t["success"] and
                                   (t["wall_collision"] or t["movable_collisions"]))
            clean_in_F = F - any_contact_in_F

            instances.append({
                "xml_file": xml_file,
                "object_id": obj_id,
                "region": region,
                "grid": grid,
                "R": R,
                "F": F,
                "ratio": ratio,
                "wall_in_F": wall_in_F,
                "movable_in_F": mov_in_F,
                "any_contact_in_F": any_contact_in_F,
                "clean_in_F": clean_in_F,
                "trial_log": tlog,
                "env_name": Path(xml_file).stem,
                "pkl_name": pkl_path.stem,
            })

    # Filter out F=0 instances — those are multi-push problems, not 1-push
    multi_push = [i for i in instances if i["F"] == 0]
    instances = [i for i in instances if i["F"] > 0]
    if multi_push:
        print(f"  Excluded {len(multi_push)} instances with F=0 (multi-push problems)")

    instances.sort(key=lambda x: x["ratio"])
    return instances


def classify_difficulty(ratio):
    if ratio < 0.05:
        return "very_hard"
    elif ratio < 0.15:
        return "hard"
    elif ratio < 0.40:
        return "medium"
    elif ratio < 0.70:
        return "easy"
    else:
        return "very_easy"


# ── Clustering analysis ──────────────────────────────────────────────────

def count_clusters(grid):
    """Count connected components in the success grid (4-connected)."""
    binary = (grid == 1.0).astype(int)
    visited = np.zeros_like(binary, dtype=bool)
    clusters = []

    def bfs(r, c):
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if visited[cr, cc]:
                continue
            visited[cr, cc] = True
            cells.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < binary.shape[0] and 0 <= nc < binary.shape[1]:
                    if binary[nr, nc] == 1 and not visited[nr, nc]:
                        stack.append((nr, nc))
        return cells

    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            if binary[i, j] == 1 and not visited[i, j]:
                cells = bfs(i, j)
                clusters.append(cells)

    return clusters


def cluster_stats(inst):
    """Compute clustering statistics for an instance."""
    clusters = count_clusters(inst["grid"])
    n_clusters = len(clusters)
    sizes = [len(c) for c in clusters]
    max_size = max(sizes) if sizes else 0
    # Fragmentation: 1 - (largest_cluster / F)
    frag = 1.0 - (max_size / inst["F"]) if inst["F"] > 0 else 0.0

    # Direction extent per cluster: how many unique edge_idxs
    dir_extents = []
    depth_extents = []
    for cl in clusters:
        edges = set(r for r, c in cl)
        depths = set(c for r, c in cl)
        dir_extents.append(len(edges))
        depth_extents.append(len(depths))

    return {
        "n_clusters": n_clusters,
        "cluster_sizes": sizes,
        "max_cluster_size": max_size,
        "fragmentation": frag,
        "dir_extents": dir_extents,
        "depth_extents": depth_extents,
    }


# ── Direction analysis ───────────────────────────────────────────────────

DIRECTION_MAP = {
    "push_down":  list(range(0, 30, 2)),   # even 0-28
    "push_up":    list(range(1, 30, 2)),   # odd 1-29
    "push_left":  list(range(30, 60, 2)),  # even 30-58
    "push_right": list(range(31, 60, 2)),  # odd 31-59
}

DIRECTION_LABELS = ["Push Down", "Push Up", "Push Left", "Push Right"]
DIRECTION_KEYS = ["push_down", "push_up", "push_left", "push_right"]
DIRECTION_COLORS = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]


def direction_breakdown(grid):
    """How many F in each push direction."""
    counts = {}
    for dname, edge_list in DIRECTION_MAP.items():
        sub = grid[edge_list, :]
        counts[dname] = int(np.nansum(sub))
    return counts


# ── Visualizations ───────────────────────────────────────────────────────

def plot_heatmap_4dir(inst, ax=None, title=None, show_colorbar=True):
    """Plot F heatmap organized by 4 push directions."""
    grid = inst["grid"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Build direction-grouped display: 4 blocks of 15 rows each
    display = np.full((60, 10), np.nan)
    row_labels = []
    row = 0
    for dname, dkey in zip(DIRECTION_LABELS, DIRECTION_KEYS):
        edges = DIRECTION_MAP[dkey]
        for ei in edges:
            display[row, :] = grid[ei, :]
            row += 1

    # Color: green=success, gray=fail, black=unreachable
    cmap = ListedColormap(["#444444", "#2ecc71"])
    masked = np.ma.masked_invalid(display)
    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest")

    # Mark unreachable as black
    unreachable = np.isnan(display)
    black_overlay = np.zeros((*display.shape, 4))
    black_overlay[unreachable] = [0, 0, 0, 1]
    ax.imshow(black_overlay, aspect="auto", interpolation="nearest")

    # Direction separators
    for sep in [15, 30, 45]:
        ax.axhline(sep - 0.5, color="white", linewidth=2)

    ax.set_xlabel("Push Depth (0–9)", fontsize=9)
    ax.set_yticks([7, 22, 37, 52])
    ax.set_yticklabels(DIRECTION_LABELS, fontsize=8, fontweight="bold")
    ax.set_xticks(range(10))

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", label=f"Feasible (F={inst['F']})"),
        mpatches.Patch(facecolor="#444444", label=f"Reachable, failed"),
        mpatches.Patch(facecolor="black", label="Unreachable"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
              framealpha=0.9)

    return ax


def render_scene(xml_file, resolution=600, highlight_object=None):
    """Render scene with optional target object highlighting.

    Uses the project's visualize_environment() as base, then overlays
    the target object in red/orange if specified.
    """
    if not os.path.exists(xml_file):
        return None

    import xml.etree.ElementTree as ET
    from PIL import Image, ImageDraw
    import math

    # Parse XML to get bounds and all objects
    tree = ET.parse(xml_file)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    if worldbody is None:
        return visualize_environment(xml_file, resolution=resolution)

    # Collect all geoms with positions
    geoms_data = []
    for geom in worldbody.iter('geom'):
        name = geom.get('name', '')
        gtype = geom.get('type', 'sphere')
        if gtype == 'plane':
            continue
        pos = [float(x) for x in geom.get('pos', '0 0 0').split()]
        size = [float(x) for x in geom.get('size', '0.1 0.1 0.1').split()]
        euler = [float(x) for x in geom.get('euler', '0 0 0').split()]
        yaw = math.radians(euler[2]) if len(euler) > 2 else 0
        geoms_data.append({
            'name': name, 'type': gtype, 'x': pos[0], 'y': pos[1],
            'size': size, 'yaw': yaw
        })

    sites_data = []
    for site in worldbody.iter('site'):
        name = site.get('name', '')
        if name == 'goal':
            pos = [float(x) for x in site.get('pos', '0 0 0').split()]
            size = [float(x) for x in site.get('size', '0.25 0.25 0.25').split()]
            sites_data.append({'name': name, 'x': pos[0], 'y': pos[1], 'radius': size[0]})

    if not geoms_data:
        return visualize_environment(xml_file, resolution=resolution)

    # Compute bounds
    all_x = []
    all_y = []
    for g in geoms_data:
        if g['type'] in ['box', 'capsule', 'cylinder']:
            w, h = g['size'][0], g['size'][1] if len(g['size']) > 1 else g['size'][0]
            corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
            cos_y, sin_y = math.cos(g['yaw']), math.sin(g['yaw'])
            for dx, dy in corners:
                all_x.append(g['x'] + dx*cos_y - dy*sin_y)
                all_y.append(g['y'] + dx*sin_y + dy*cos_y)
        else:
            r = g['size'][0]
            all_x.extend([g['x'] - r, g['x'] + r])
            all_y.extend([g['y'] - r, g['y'] + r])
    for s in sites_data:
        r = s['radius']
        all_x.extend([s['x'] - r, s['x'] + r])
        all_y.extend([s['y'] - r, s['y'] + r])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    pad = 0.05 * max(max_x - min_x, max_y - min_y)
    min_x -= pad; max_x += pad; min_y -= pad; max_y += pad

    scale = (resolution * 0.9) / max(max_x - min_x, max_y - min_y)
    margin = resolution * 0.05

    def w2p(x, y):
        return (x - min_x) * scale + margin, (max_y - y) * scale + margin

    # Colors
    BG = (30, 30, 30)
    WALL = (180, 180, 180)
    OBSTACLE = (255, 212, 0)
    TARGET = (255, 69, 0)       # Red-orange for target
    ROBOT = (0, 120, 255)
    GOAL = (0, 220, 0)

    img = Image.new('RGB', (resolution, resolution), BG)
    draw = ImageDraw.Draw(img)

    def draw_box(g, color):
        w, h = g['size'][0], g['size'][1] if len(g['size']) > 1 else g['size'][0]
        corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        cos_y, sin_y = math.cos(g['yaw']), math.sin(g['yaw'])
        pix = []
        for dx, dy in corners:
            wx = g['x'] + dx*cos_y - dy*sin_y
            wy = g['y'] + dx*sin_y + dy*cos_y
            pix.append(w2p(wx, wy))
        draw.polygon(pix, fill=color, outline=(0, 0, 0))

    # Z-order: walls first, then obstacles, then robot/goal
    walls = [g for g in geoms_data if 'wall' in g['name'].lower()]
    obstacles = [g for g in geoms_data if 'obstacle' in g['name'].lower()]
    robot_geoms = [g for g in geoms_data if 'robot' in g['name'].lower()]

    for g in walls:
        draw_box(g, WALL)
    for g in obstacles:
        is_target = (highlight_object and g['name'] == highlight_object)
        draw_box(g, TARGET if is_target else OBSTACLE)
    for g in robot_geoms:
        if g['type'] == 'sphere':
            cx, cy = w2p(g['x'], g['y'])
            r = g['size'][0] * scale
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=ROBOT)
    for s in sites_data:
        cx, cy = w2p(s['x'], s['y'])
        r = s['radius'] * scale
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=GOAL)

    return img


def plot_scene_and_F(inst, out_path=None, figsize=(10, 10)):
    """Two-row plot: scene on top, F heatmap on bottom."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.25)

    # Row 1: Scene rendering
    ax_scene = fig.add_subplot(gs[0])
    img = render_scene(inst["xml_file"], highlight_object=inst["object_id"])
    if img is not None:
        ax_scene.imshow(img)
        ax_scene.set_title(
            f"Scene: {inst['env_name']}\n"
            f"Object: {inst['object_id']}  |  Region: {inst['region']}",
            fontsize=10, fontweight="bold"
        )
    else:
        ax_scene.text(0.5, 0.5, f"Scene not found:\n{inst['xml_file']}",
                      ha="center", va="center", fontsize=9)
        ax_scene.set_title("Scene (unavailable)", fontsize=10)
    ax_scene.axis("off")

    # Row 2: F heatmap
    ax_heat = fig.add_subplot(gs[1])
    diff = classify_difficulty(inst["ratio"])
    plot_heatmap_4dir(
        inst, ax=ax_heat,
        title=f"|F|/|R| = {inst['F']}/{inst['R']} = {inst['ratio']:.1%}  "
              f"[{diff.upper().replace('_',' ')}]"
    )

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out_path}")
    return fig


# ── Summary statistics ───────────────────────────────────────────────────

def print_difficulty_distribution(instances):
    """Print difficulty distribution table."""
    buckets = defaultdict(list)
    for inst in instances:
        d = classify_difficulty(inst["ratio"])
        buckets[d].append(inst)

    order = ["very_hard", "hard", "medium", "easy", "very_easy"]
    labels = ["Very Hard (<5%)", "Hard (5-15%)", "Medium (15-40%)",
              "Easy (40-70%)", "Very Easy (>70%)"]

    print("\n" + "=" * 70)
    print("DIFFICULTY DISTRIBUTION")
    print("=" * 70)
    print(f"{'Category':<22} {'Count':>6} {'Avg |F|':>8} {'Avg |R|':>8} {'Avg |F|/|R|':>12} {'Wall%':>7}")
    print("-" * 70)
    for cat, label in zip(order, labels):
        insts = buckets.get(cat, [])
        n = len(insts)
        if n == 0:
            print(f"{label:<22} {0:>6}")
            continue
        avg_F = np.mean([i["F"] for i in insts])
        avg_R = np.mean([i["R"] for i in insts])
        avg_ratio = np.mean([i["ratio"] for i in insts])
        wall_pct = np.mean([
            i["wall_in_F"] / i["F"] * 100 if i["F"] > 0 else 0
            for i in insts
        ])
        print(f"{label:<22} {n:>6} {avg_F:>8.1f} {avg_R:>8.1f} {avg_ratio:>11.1%} {wall_pct:>6.1f}%")

    print("-" * 70)
    print(f"{'TOTAL':<22} {len(instances):>6}")
    print()


def print_clustering_analysis(instances):
    """Print clustering statistics by difficulty."""
    print("=" * 70)
    print("CLUSTERING ANALYSIS (Hypothesis 2 & 3)")
    print("=" * 70)
    print(f"{'Category':<22} {'Avg Clusters':>13} {'Avg Frag':>10} "
          f"{'Avg MaxCluster':>15} {'Dir Extent':>11} {'Depth Extent':>13}")
    print("-" * 70)

    order = ["very_hard", "hard", "medium", "easy", "very_easy"]
    labels = ["Very Hard (<5%)", "Hard (5-15%)", "Medium (15-40%)",
              "Easy (40-70%)", "Very Easy (>70%)"]

    for cat, label in zip(order, labels):
        insts = [i for i in instances if classify_difficulty(i["ratio"]) == cat]
        if not insts:
            print(f"{label:<22} {'(none)':>13}")
            continue

        all_stats = [cluster_stats(i) for i in insts]
        avg_n = np.mean([s["n_clusters"] for s in all_stats])
        avg_frag = np.mean([s["fragmentation"] for s in all_stats])
        avg_max = np.mean([s["max_cluster_size"] for s in all_stats])
        avg_dir = np.mean([np.mean(s["dir_extents"]) if s["dir_extents"] else 0
                           for s in all_stats])
        avg_dep = np.mean([np.mean(s["depth_extents"]) if s["depth_extents"] else 0
                           for s in all_stats])
        print(f"{label:<22} {avg_n:>13.1f} {avg_frag:>10.2f} "
              f"{avg_max:>15.1f} {avg_dir:>11.1f} {avg_dep:>13.1f}")
    print()


def print_direction_analysis(instances):
    """Print direction breakdown by difficulty (Hypothesis 4)."""
    print("=" * 70)
    print("DIRECTION ANALYSIS (Hypothesis 4: depth vs direction)")
    print("=" * 70)

    order = ["very_hard", "hard", "medium", "easy", "very_easy"]
    labels = ["Very Hard", "Hard", "Medium", "Easy", "Very Easy"]

    print(f"{'Category':<15}", end="")
    for dl in DIRECTION_LABELS:
        print(f" {dl:>12}", end="")
    print(f" {'Dominant':>12}")
    print("-" * 75)

    for cat, label in zip(order, labels):
        insts = [i for i in instances if classify_difficulty(i["ratio"]) == cat]
        if not insts:
            continue
        totals = {dk: 0 for dk in DIRECTION_KEYS}
        for inst in insts:
            bd = direction_breakdown(inst["grid"])
            for dk in DIRECTION_KEYS:
                totals[dk] += bd[dk]
        total_F = sum(totals.values())
        print(f"{label:<15}", end="")
        for dk in DIRECTION_KEYS:
            pct = totals[dk] / total_F * 100 if total_F > 0 else 0
            print(f" {pct:>11.1f}%", end="")
        dom = max(totals, key=totals.get)
        print(f" {dom:>12}")
    print()


def print_wall_collision_analysis(instances):
    """Print wall collision statistics (Hypothesis 5)."""
    print("=" * 70)
    print("WALL COLLISION ANALYSIS (Hypothesis 5: walls create F)")
    print("=" * 70)

    order = ["very_hard", "hard", "medium", "easy", "very_easy"]
    labels = ["Very Hard (<5%)", "Hard (5-15%)", "Medium (15-40%)",
              "Easy (40-70%)", "Very Easy (>70%)"]

    print(f"{'Category':<22} {'F with wall':>12} {'F with mov':>12} {'F any coll':>11} {'F clean':>10} {'Contact%':>9}")
    print("-" * 78)

    for cat, label in zip(order, labels):
        insts = [i for i in instances if classify_difficulty(i["ratio"]) == cat]
        if not insts:
            continue
        total_F = sum(i["F"] for i in insts)
        total_wall = sum(i["wall_in_F"] for i in insts)
        total_mov = sum(i["movable_in_F"] for i in insts)
        total_any = sum(i["any_contact_in_F"] for i in insts)
        total_clean = sum(i["clean_in_F"] for i in insts)
        contact_pct = total_any / total_F * 100 if total_F > 0 else 0
        print(f"{label:<22} {total_wall:>12} {total_mov:>12} {total_any:>11} {total_clean:>10} {contact_pct:>8.1f}%")
    print()


# ── Summary figures ──────────────────────────────────────────────────────

def plot_difficulty_spectrum(instances, out_path=None):
    """Multi-panel figure: one representative per difficulty level."""
    order = ["very_hard", "hard", "medium", "easy", "very_easy"]
    labels = ["Very Hard\n(<5%)", "Hard\n(5-15%)", "Medium\n(15-40%)",
              "Easy\n(40-70%)", "Very Easy\n(>70%)"]

    buckets = defaultdict(list)
    for inst in instances:
        buckets[classify_difficulty(inst["ratio"])].append(inst)

    # Pick median instance from each bucket
    reps = []
    for cat in order:
        insts = buckets.get(cat, [])
        if insts:
            insts_sorted = sorted(insts, key=lambda x: x["ratio"])
            reps.append(insts_sorted[len(insts_sorted) // 2])
        else:
            reps.append(None)

    n_panels = sum(1 for r in reps if r is not None)
    fig, axes = plt.subplots(2, n_panels, figsize=(4 * n_panels, 8),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.3})
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    col = 0
    for rep, label in zip(reps, labels):
        if rep is None:
            continue

        # Scene
        img = render_scene(rep["xml_file"], resolution=400, highlight_object=rep["object_id"])
        if img is not None:
            axes[0, col].imshow(img)
        axes[0, col].axis("off")
        axes[0, col].set_title(f"{label}\n{rep['env_name']}", fontsize=9, fontweight="bold")

        # Heatmap
        plot_heatmap_4dir(rep, ax=axes[1, col],
                          title=f"|F|/|R|={rep['ratio']:.1%}")
        col += 1

    fig.suptitle("F Characterization: Difficulty Spectrum", fontsize=14, fontweight="bold", y=1.02)

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")
    return fig


def plot_ratio_histogram(instances, out_path=None):
    """Histogram of |F|/|R| across all instances."""
    ratios = [i["ratio"] for i in instances]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 1.05, 0.05)
    colors_map = {
        "very_hard": "#e74c3c",
        "hard": "#e67e22",
        "medium": "#f1c40f",
        "easy": "#2ecc71",
        "very_easy": "#3498db",
    }

    # Stack by difficulty
    for cat, color in colors_map.items():
        cat_ratios = [r for r in ratios if classify_difficulty(r) == cat]
        if cat_ratios:
            ax.hist(cat_ratios, bins=bins, alpha=0.7, color=color,
                    label=f"{cat.replace('_',' ').title()} ({len(cat_ratios)})",
                    edgecolor="white", linewidth=0.5)

    ax.set_xlabel("|F| / |R|", fontsize=11)
    ax.set_ylabel("Number of instances", fontsize=11)
    ax.set_title("Distribution of Feasibility Ratio |F|/|R|", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")
    return fig


def plot_scatter_F_vs_R(instances, out_path=None):
    """Scatter: |F| vs |R|, colored by difficulty."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors_map = {
        "very_hard": "#e74c3c", "hard": "#e67e22", "medium": "#f1c40f",
        "easy": "#2ecc71", "very_easy": "#3498db",
    }

    for inst in instances:
        cat = classify_difficulty(inst["ratio"])
        c = colors_map[cat]
        axes[0].scatter(inst["R"], inst["F"], c=c, s=40, alpha=0.7, edgecolors="black", linewidth=0.5)
        axes[1].scatter(inst["R"], inst["ratio"], c=c, s=40, alpha=0.7, edgecolors="black", linewidth=0.5)

    axes[0].set_xlabel("|R| (reachable primitives)")
    axes[0].set_ylabel("|F| (feasible primitives)")
    axes[0].set_title("|F| vs |R|", fontweight="bold")
    axes[0].plot([0, 600], [0, 600], "k--", alpha=0.3, label="|F|=|R|")
    axes[0].legend()

    axes[1].set_xlabel("|R| (reachable primitives)")
    axes[1].set_ylabel("|F| / |R|")
    axes[1].set_title("Feasibility Ratio vs Reachable Set Size", fontweight="bold")

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=c, label=k.replace("_", " ").title())
        for k, c in colors_map.items()
    ]
    axes[1].legend(handles=legend_elements, fontsize=8)

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")
    return fig


def plot_clustering_vs_difficulty(instances, out_path=None):
    """Scatter: fragmentation and cluster count vs |F|/|R|."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ratios = []
    n_clusters_list = []
    frag_list = []
    max_cluster_list = []

    for inst in instances:
        if inst["F"] == 0:
            continue
        cs = cluster_stats(inst)
        ratios.append(inst["ratio"])
        n_clusters_list.append(cs["n_clusters"])
        frag_list.append(cs["fragmentation"])
        max_cluster_list.append(cs["max_cluster_size"])

    axes[0].scatter(ratios, n_clusters_list, s=30, alpha=0.6, c="#2c3e50")
    axes[0].set_xlabel("|F|/|R|")
    axes[0].set_ylabel("Number of clusters")
    axes[0].set_title("Clusters vs Feasibility", fontweight="bold")

    axes[1].scatter(ratios, frag_list, s=30, alpha=0.6, c="#8e44ad")
    axes[1].set_xlabel("|F|/|R|")
    axes[1].set_ylabel("Fragmentation (1 - max_cluster/F)")
    axes[1].set_title("Fragmentation vs Feasibility", fontweight="bold")

    axes[2].scatter(ratios, max_cluster_list, s=30, alpha=0.6, c="#16a085")
    axes[2].set_xlabel("|F|/|R|")
    axes[2].set_ylabel("Largest cluster size")
    axes[2].set_title("Largest Cluster vs Feasibility", fontweight="bold")

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")
    return fig


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze exhaustive F characterization data")
    parser.add_argument("--data-dir", required=True, help="Directory with *_results.pkl files")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for plots")
    parser.add_argument("--scene-plots", type=int, default=3,
                        help="Number of scene+F plots per difficulty level")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    instances = load_all_instances(args.data_dir)
    print(f"Loaded {len(instances)} unique (xml, object, region) instances")
    print(f"|F|/|R| range: {min(i['ratio'] for i in instances):.1%} – {max(i['ratio'] for i in instances):.1%}")
    print(f"Median |F|/|R|: {np.median([i['ratio'] for i in instances]):.1%}")

    # ── Print tables ──
    print_difficulty_distribution(instances)
    print_clustering_analysis(instances)
    print_direction_analysis(instances)
    print_wall_collision_analysis(instances)

    # ── Generate summary plots ──
    print("\nGenerating plots...")
    plot_ratio_histogram(instances, out / "histogram_FR_ratio.png")
    plot_scatter_F_vs_R(instances, out / "scatter_F_vs_R.png")
    plot_clustering_vs_difficulty(instances, out / "clustering_vs_difficulty.png")
    plot_difficulty_spectrum(instances, out / "difficulty_spectrum_2row.png")

    # ── Scene + F heatmap plots per difficulty ──
    buckets = defaultdict(list)
    for inst in instances:
        buckets[classify_difficulty(inst["ratio"])].append(inst)

    order = ["very_hard", "hard", "medium", "easy", "very_easy"]
    for cat in order:
        insts = buckets.get(cat, [])
        if not insts:
            continue
        # Sort by ratio, pick evenly spaced samples
        insts_sorted = sorted(insts, key=lambda x: x["ratio"])
        n = min(args.scene_plots, len(insts_sorted))
        indices = np.linspace(0, len(insts_sorted) - 1, n, dtype=int)
        for idx in indices:
            inst = insts_sorted[idx]
            fname = f"scene_F_{cat}_{inst['env_name']}_{inst['object_id']}.png"
            plot_scene_and_F(inst, out / fname)

    print(f"\nDone! All plots saved to {out}/")


if __name__ == "__main__":
    main()
