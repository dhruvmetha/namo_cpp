"""Render qpos dump as a clean MuJoCo top-down video — no matplotlib panel.

Faster than render_nav_video.py because it skips the side-by-side trajectory plot.
Render every Nth qpos tick (default every 1, i.e. all ticks).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image


def parse_qpos(path: Path):
    """Parse the C++ qpos dump.

    Each non-empty line is `phase nq q0 q1 ... q(nq-1)` (space-separated, no colon).
    Some legacy lines may use 'phase: q0 q1 ...' so we handle both.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                phase_str, qstr = line.split(":", 1)
                phase = int(phase_str)
                qs = np.fromstring(qstr, sep=" ", dtype=float)
            else:
                vals = np.fromstring(line, sep=" ", dtype=float)
                if vals.size < 2:
                    continue
                phase = int(vals[0])
                nq = int(vals[1])
                qs = vals[2:2 + nq]
            rows.append((phase, qs))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("qpos_dump")
    ap.add_argument("output_mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cam-dist", type=float, default=1.6)
    ap.add_argument("--cam-elevation", type=float, default=-89.0)
    ap.add_argument("--cam-azimuth", type=float, default=90.0)
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="Render every Nth qpos tick (1 = all ticks).")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    rows = parse_qpos(Path(args.qpos_dump))
    print(f"Loaded {len(rows)} qpos rows")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.05]
    cam.distance = args.cam_dist
    cam.azimuth = args.cam_azimuth
    cam.elevation = args.cam_elevation

    out = Path(args.output_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        idx = 0
        for i, (phase, qs) in enumerate(rows):
            if i % args.frame_skip != 0:
                continue
            n = min(len(qs), model.nq)
            data.qpos[:n] = qs[:n]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            Image.fromarray(img).save(tmp / f"f{idx:06d}.png")
            idx += 1
            if idx % 100 == 0:
                print(f"  rendered {idx}")

        print(f"Rendered {idx} frames; encoding to {out}...")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp / "f%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
        ], check=True, capture_output=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
