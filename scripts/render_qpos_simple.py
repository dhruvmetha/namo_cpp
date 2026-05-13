"""Simple top-down MuJoCo renderer from a qpos dump file.

Faster than render_nav_video.py because it doesn't draw the matplotlib trajectory panel.
Use --frame-skip to subsample (default 25 → ~20 fps from a 500Hz dump).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np


def parse_qpos(path):
    """Each line: 'phase nq q0 q1 ...'. Returns list of (phase, q)."""
    frames = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            phase = int(parts[0])
            nq = int(parts[1])
            q = np.asarray([float(x) for x in parts[2:2 + nq]])
            frames.append((phase, q))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("qpos_dump")
    ap.add_argument("output_mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frame-skip", type=int, default=25,
                    help="Render every Nth qpos frame (25 = ~20fps from a 500Hz dump)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--cam-dist", type=float, default=1.5)
    ap.add_argument("--cam-azimuth", type=float, default=90.0)
    ap.add_argument("--cam-elevation", type=float, default=-90.0)
    args = ap.parse_args()

    frames = parse_qpos(args.qpos_dump)
    if not frames:
        print("empty qpos dump", file=sys.stderr)
        sys.exit(1)

    # Subsample
    frames = frames[::args.frame_skip]
    print(f"qpos frames after skip={args.frame_skip}: {len(frames)}")

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.03]
    cam.distance = args.cam_dist
    cam.azimuth = args.cam_azimuth
    cam.elevation = args.cam_elevation

    writer = imageio.get_writer(args.output_mp4, fps=args.fps,
                                codec="libx264", quality=8, format="FFMPEG")
    for phase, q in frames:
        data.qpos[:len(q)] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())
    writer.close()
    renderer.close()
    print(f"wrote {args.output_mp4}")


if __name__ == "__main__":
    main()
