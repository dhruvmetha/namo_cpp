"""One-shot push video for inspecting pre-settle / push / post-settle on a single primitive.

Defaults to: wide object, +y face center edge (idx 22), push_steps=3.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_car_primitives import OBJECT_CONFIGS  # noqa: E402
from render_push_videos import render_push  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="wide", choices=["square", "wide", "tall"])
    ap.add_argument("--edge", type=int, default=22, help="Edge index (0..59 for points_per_face=15)")
    ap.add_argument("--depth", type=int, default=3, help="push_steps")
    ap.add_argument("--exit-ramp-ticks", type=int, default=0,
                    help="Linear ctrl ramp-down from push_speed to 0 over N ticks at end of push.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    obj = next(c for c in OBJECT_CONFIGS if c.name == args.shape)
    out_dir = PROJECT_ROOT / "artifacts" / "push_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (
        out_dir / f"one_{obj.name}_edge{args.edge}_depth{args.depth}.mp4"
    )

    print(f"Rendering: {obj.description}, edge={args.edge}, depth={args.depth}, "
          f"exit_ramp_ticks={args.exit_ramp_ticks}")
    render_push(obj, args.edge, args.depth, out_path,
                exit_ramp_ticks=args.exit_ramp_ticks)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
