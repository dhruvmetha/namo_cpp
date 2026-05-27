#!/usr/bin/env python3
"""Render each NPZ as a single composite overlay PNG.

Layers (bottom → top):
    background           white
    static walls         dark gray (solid)
    robot_region         blue tint  (semi-transparent)
    goal_sample_region   green tint (semi-transparent)
    other movables       light gray
    target_object        blue        (current pose of the obstacle to push)
    goal_mask_a1         red         (where to push it — action 1)
    goal_mask_a2         orange      (where to push it next — action 2, often empty)

Skipped: local_robot, local_goal (the dots), local_target_goal (alias for a1).

Usage:
    python scripts/visualize_overlay.py --input-dir <npz_dir> --output-dir <png_dir>
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_overlay(npz_path: Path, output_path: Path) -> None:
    data = np.load(npz_path)
    keys = set(data.keys())

    static     = data["local_static"]            if "local_static" in keys           else None
    movables   = data["local_movable"]           if "local_movable" in keys          else None
    tgt_obj    = data["local_target_object"]     if "local_target_object" in keys    else None
    robot_reg  = data["local_robot_region"]      if "local_robot_region" in keys     else None
    goal_reg   = data["local_goal_sample_region"] if "local_goal_sample_region" in keys else None
    mask_a1    = data["local_goal_mask_a1"]      if "local_goal_mask_a1" in keys     else None
    mask_a2    = data["local_goal_mask_a2"]      if "local_goal_mask_a2" in keys     else None

    if static is None:
        print(f"  skip (no local_static): {npz_path}")
        return

    H, W = static.shape
    # Start with white background
    rgb = np.ones((H, W, 3), dtype=np.float32)

    def paint_tint(rgb, mask, color, alpha):
        """Semi-transparent fill: rgb' = (1-alpha)*rgb + alpha*color where mask>0.5."""
        if mask is None: return rgb
        m = (mask > 0.5).astype(np.float32)[..., None]
        col = np.array(color, dtype=np.float32)[None, None, :]
        return rgb * (1 - alpha * m) + alpha * m * col

    def paint_solid(rgb, mask, color):
        """Fully replace pixels where mask>0.5 with `color`."""
        if mask is None: return rgb
        m = (mask > 0.5).astype(np.float32)[..., None]
        col = np.array(color, dtype=np.float32)[None, None, :]
        return rgb * (1 - m) + m * col

    # Region tints first (under objects)
    rgb = paint_tint(rgb, robot_reg, (0.4, 0.65, 0.95), 0.30)   # light blue
    rgb = paint_tint(rgb, goal_reg,  (0.45, 0.85, 0.45), 0.30)  # light green

    # Walls + other movables (solid grays)
    rgb = paint_solid(rgb, static,    (0.20, 0.20, 0.20))       # dark gray
    if movables is not None and tgt_obj is not None:
        non_target_movables = np.clip(movables - tgt_obj, 0.0, 1.0)
    else:
        non_target_movables = movables
    rgb = paint_solid(rgb, non_target_movables, (0.65, 0.65, 0.65))  # light gray

    # Target object current pose
    rgb = paint_solid(rgb, tgt_obj, (0.20, 0.40, 0.95))         # blue
    # Action targets
    rgb = paint_solid(rgb, mask_a1, (0.95, 0.20, 0.20))         # red — action 1
    if mask_a2 is not None and int((mask_a2 > 0.5).sum()) > 0:
        rgb = paint_solid(rgb, mask_a2, (0.98, 0.55, 0.10))     # orange — action 2

    # Metadata
    sd = int(data["solution_depth"][0]) if "solution_depth" in keys else -1
    nh = int(data["num_goal_horizons"][0]) if "num_goal_horizons" in keys else -1
    ep = str(data["episode_id"][0]) if "episode_id" in keys else "?"
    xml = str(data["xml_file"][0]) if "xml_file" in keys else ""
    crop = float(data["local_crop_size_meters"][0]) if "local_crop_size_meters" in keys else -1
    title = (f"{ep}\nsolution_depth={sd}  horizons={nh}  crop={crop:.2f} m\n"
             f"{Path(xml).name if xml else ''}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb, origin="lower")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Legend
    legend_items = [
        ("static walls",         (0.20, 0.20, 0.20)),
        ("other movables",       (0.65, 0.65, 0.65)),
        ("target obj (start)",   (0.20, 0.40, 0.95)),
        ("push target — a1",     (0.95, 0.20, 0.20)),
        ("push target — a2",     (0.98, 0.55, 0.10)),
        ("robot region (tint)",  (0.70, 0.83, 0.97)),
        ("goal region (tint)",   (0.72, 0.92, 0.72)),
    ]
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.3, label=lbl)
               for lbl, c in legend_items]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, fontsize=7, frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    npzs = sorted(in_dir.glob("**/*.npz"))
    if args.limit > 0:
        npzs = npzs[: args.limit]
    print(f"Rendering {len(npzs)} overlays")
    for i, p in enumerate(npzs, 1):
        rel = p.relative_to(in_dir)
        out_png = out_dir / rel.with_suffix(".overlay.png")
        try:
            render_overlay(p, out_png)
            print(f"  [{i}/{len(npzs)}] {rel} → {out_png.name}")
        except Exception as e:
            print(f"  [{i}/{len(npzs)}] FAIL {rel}: {e}")
    print(f"Done. Output dir: {out_dir}")


if __name__ == "__main__":
    main()
