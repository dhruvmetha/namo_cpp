#!/usr/bin/env python3
"""Record a navigation + push video for the diff-drive car.

Usage:
    python record_nav_video.py <xml_path> <config_path> <output_mp4>
           [--object NAME] [--edge IDX] [--depth D] [--cam DIST]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import namo_rl
import numpy as np
from PIL import Image


def render_frames(env, action, width=640, height=480, fps=50, cam_distance=1.0):
    """Run one push action while recording video frames."""
    env.start_recording(width, height, capture_frequency=100, max_frames=20000)
    env.set_camera_distance(cam_distance) if hasattr(env, "set_camera_distance") else None
    env.set_camera_position(cam_distance, 90, -75)

    result = env.step(action)
    env.stop_recording()
    frames = env.get_frames()
    return result, frames


def save_mp4(frames, path, fps=30):
    """Encode frames (bytes) to MP4 with ffmpeg."""
    import subprocess
    import tempfile

    if not frames:
        print("No frames captured — cannot write video.")
        return

    # Figure out dimensions from first frame
    first = np.frombuffer(frames[0], dtype=np.uint8)
    # Frames are stored raw; we need the recording dims from env
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, f in enumerate(frames):
            arr = np.frombuffer(f, dtype=np.uint8)
            n = arr.size
            # Try guessing: assume 3 channels. find h,w so h*w*3 = n
            if n % 3 != 0:
                continue
            px = n // 3
            # Prefer width as 640, 800, etc.
            for w in (640, 800, 1024, 1280):
                if px % w == 0:
                    h = px // w
                    img = arr.reshape(h, w, 3)
                    Image.fromarray(img).save(tmp / f"f{i:06d}.png")
                    break
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(tmp / "f%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)
        ], check=True, capture_output=True)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xml")
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--object", default=None)
    parser.add_argument("--edge", type=int, default=0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--cam", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    env = namo_rl.RLEnvironment(args.xml, args.config, False)

    obj = args.object
    if obj is None:
        reachable = env.get_reachable_objects()
        if not reachable:
            print("No reachable objects")
            sys.exit(1)
        obj = reachable[0]
    print(f"Target object: {obj}")
    print(f"Edge: {args.edge}, depth: {args.depth}")

    action = namo_rl.Action()
    action.object_id = obj
    action.edge_idx = args.edge
    action.depth = args.depth

    ob_before = env.get_observation()[f"{obj}_pose"]
    result, frames = render_frames(env, action, args.width, args.height, cam_distance=args.cam)
    ob_after = env.get_observation()[f"{obj}_pose"]
    dist = ((ob_after[0]-ob_before[0])**2 + (ob_after[1]-ob_before[1])**2)**0.5
    print(f"Frames: {len(frames)}, object moved {dist*1000:.1f}mm, done={result.done}")
    print(f"Info: {dict(result.info)}")

    save_mp4(frames, args.output, fps=30)


if __name__ == "__main__":
    main()
