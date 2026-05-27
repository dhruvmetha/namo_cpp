#!/usr/bin/env python3
"""Render NPZ mask files to PNG previews for visual inspection.

For each NPZ, produces one PNG showing all relevant channel masks side-by-side:
- Inputs to the model: local_static, local_movable, local_target_object,
  local_robot_region, local_goal_sample_region
- Targets (what the model predicts): local_goal_mask_a1, local_goal_mask_a2
- Context: local_robot, local_goal, local_target_goal

Usage:
    python scripts/visualize_masks.py --input-dir /scratch/.../shard_0 \\
        --output-dir /scratch/.../previews
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PANELS = [
    ("local_static",             "static walls"),
    ("local_movable",            "all movables"),
    ("local_target_object",      "target obj (input pose)"),
    ("local_robot_region",       "robot region"),
    ("local_goal_sample_region", "goal region"),
    ("local_robot",              "robot circle"),
    ("local_goal",               "goal site"),
    ("local_target_goal",        "target obj @ next-action target"),
    ("local_goal_mask_a1",       "PREDICTION TARGET — action 1"),
    ("local_goal_mask_a2",       "PREDICTION TARGET — action 2"),
]


def render_one(npz_path: Path, output_path: Path) -> None:
    data = np.load(npz_path)
    keys = set(data.keys())

    # Gather available panels
    panels = [(k, label) for (k, label) in PANELS if k in keys]
    cols = 5
    rows = (len(panels) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Metadata header
    sd = int(data["solution_depth"][0]) if "solution_depth" in keys else -1
    nh = int(data["num_goal_horizons"][0]) if "num_goal_horizons" in keys else -1
    ep_id = str(data["episode_id"][0]) if "episode_id" in keys else "?"
    xml = str(data["xml_file"][0]) if "xml_file" in keys else ""
    title = (f"{ep_id}\n"
             f"solution_depth={sd}  num_goal_horizons={nh}\n"
             f"{Path(xml).name if xml else ''}")
    fig.suptitle(title, fontsize=10, y=0.995)

    for ax, (key, label) in zip(axes, panels):
        m = data[key]
        if m.ndim == 3 and m.shape[2] in (1, 3):
            ax.imshow(m, origin="lower")
        else:
            ax.imshow(m, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(panels):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="Directory containing NPZ files (searches recursively)")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write PNG previews")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max NPZs to render (0 = all)")
    ap.add_argument("--pattern", default="**/*.npz",
                    help="Glob pattern under input-dir (default: **/*.npz)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    npzs = sorted(input_dir.glob(args.pattern))
    if args.limit > 0:
        npzs = npzs[: args.limit]
    print(f"Rendering {len(npzs)} NPZs from {input_dir}")
    for i, p in enumerate(npzs, 1):
        rel = p.relative_to(input_dir)
        out_png = output_dir / rel.with_suffix(".png")
        try:
            render_one(p, out_png)
            print(f"  [{i}/{len(npzs)}] {rel} → {out_png}")
        except Exception as e:
            print(f"  [{i}/{len(npzs)}] FAIL {rel}: {e}")
    print(f"Done. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
