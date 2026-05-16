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
    ap.add_argument("--cam-dist", type=float, default=None,
                    help="Camera distance (m). If omitted, auto-fits to wall extents "
                         "(longest wall span × cam-fit-multiplier).")
    ap.add_argument("--cam-fit-multiplier", type=float, default=1.2,
                    help="When --cam-dist is auto, multiply the longest wall span by this. "
                         "1.2 leaves a small margin; bump up for more zoom-out.")
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

    # Compute auto cam_dist from wall extents if not user-supplied
    cam_dist = args.cam_dist
    cam_lookat = np.array([0.0, 0.0, 0.03])
    if cam_dist is None:
        wall_xs, wall_ys = [], []
        for gi in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            if not name.startswith("wall_"):
                continue
            px, py = float(model.geom_pos[gi, 0]), float(model.geom_pos[gi, 1])
            sx, sy = float(model.geom_size[gi, 0]), float(model.geom_size[gi, 1])
            wall_xs += [px - sx, px + sx]
            wall_ys += [py - sy, py + sy]
        if wall_xs and wall_ys:
            xmin, xmax = min(wall_xs), max(wall_xs)
            ymin, ymax = min(wall_ys), max(wall_ys)
            cam_lookat = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.03])
            cam_dist = max(xmax - xmin, ymax - ymin) * args.cam_fit_multiplier
        else:
            cam_dist = 1.5
        print(f"auto cam_dist={cam_dist:.3f} lookat=({cam_lookat[0]:.3f},{cam_lookat[1]:.3f})")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = cam_lookat
    cam.distance = cam_dist
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
