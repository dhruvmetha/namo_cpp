#!/usr/bin/env python3
"""Render a deep-keyhole solution to a top-down MP4 from an eval_policy leaf row.

The existing render_episode_video.py replays a PKL whose actions already carry a
`target` pose. eval_policy's JSONL rows carry only (object_id, edge_idx, depth), so
this resolves each one back to a goal pose by re-enumerating the primitive goals at
that state and matching the pair, then hands the qpos dump to the same renderer.

    python scripts/render_npush_replay.py \\
        --xml  .../v4_snap/b3_b/rb_00198/env.xml \\
        --actions '[{"object_id":"obstacle_0_movable","edge_idx":41,"depth":2}, ...]' \\
        --out  $NAMO_SCRATCH/videos/L3_rb_00198.mp4

Config defaults to margin_5mm, which is what the deep-keyhole campaign ran under. A
scene certified at 5 mm will not necessarily replay at 1 mm.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Must precede any mujoco import. EGL renders fine on the CS boxes; what fails is the
# PyOpenGL teardown in mujoco.egl.GLContext.free, which raises from __del__ and made
# this look like a rendering failure. main() ends with os._exit to skip that teardown.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "build_python"))
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sandbox"))

DEFAULT_CFG = REPO / "config/margin_5mm/namo_config_complete_skill15_car_1x.yaml"


def resolve_and_replay(xml: str, cfg: str, actions: list[dict], qpos_path: Path) -> int:
    """Step each stored action, resolving its goal pose from the live state.

    Returns the number of pushes executed. Raises if an action cannot be matched,
    which means the state diverged and the video would not show the real solution.
    """
    os.environ["NAMO_QPOS_DUMP"] = str(qpos_path)
    qpos_path.unlink(missing_ok=True)

    import namo_rl
    from namo.rl_loop.policy import _UniformPlanner   # .prim only, no checkpoint

    prim = _UniformPlanner().prim
    env = namo_rl.RLEnvironment(xml, cfg, visualize=False)
    env.reset()

    for i, spec in enumerate(actions):
        obj, edge, depth = spec["object_id"], int(spec["edge_idx"]), int(spec["depth"])
        state = env.get_full_state()
        env.get_reachable_objects()                  # warms the wavefront
        match = None
        for edge_goals in prim.generate_goals(obj, state, env, max_goals=0):
            for g in edge_goals:
                if g is None:
                    continue
                if int(getattr(g, "edge_idx", -1)) == edge and int(getattr(g, "depth", -1)) == depth:
                    match = g
                    break
            if match:
                break
        if match is None:
            raise RuntimeError(
                f"push {i}: no goal with edge_idx={edge} depth={depth} on {obj}. "
                "The state diverged from the run that produced these actions.")
        a = namo_rl.Action()
        a.object_id = obj
        a.x, a.y, a.theta = float(match.x), float(match.y), float(match.theta)
        a.edge_idx, a.depth = edge, depth
        env.step(a)
    return len(actions)


def render_topdown(xml_path: Path, frames: list[list[float]], out: Path,
                   width: int = 1280, height: int = 1280, fps: int = 30,
                   margin: float = 1.18) -> None:
    """Top-down MP4, framed to the arena.

    render_episode_video.render_qpos_to_mp4 hardcodes lookat=(0,0,0) and distance=3.5.
    These rooms sit off the origin and span well under a metre, so that leaves the
    arena about 5% of the frame. Here the camera is fitted to the static geometry.
    """
    import xml.etree.ElementTree as ET
    import cv2
    import mujoco

    root = ET.fromstring(xml_path.read_text())
    visual = root.find("visual") or ET.SubElement(root, "visual")
    glob = visual.find("global") or ET.SubElement(visual, "global")
    glob.set("offwidth", str(width)); glob.set("offheight", str(height))
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=xml_path.parent,
                                     delete=False) as fh:
        fh.write(ET.tostring(root, encoding="unicode"))
        tmp_xml = Path(fh.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        # fit to the walls: every geom except the unbounded floor plane
        pts = [model.geom_pos[g][:2] for g in range(model.ngeom)
               if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
        pts = np.asarray(pts) if pts else np.zeros((1, 2))
        lo, hi = pts.min(0), pts.max(0)
        cx, cy = (lo + hi) / 2.0
        span = float((hi - lo).max()) if len(pts) > 1 else 1.0

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [cx, cy, 0.0]
        cam.distance = max(span * margin, 0.35)
        cam.elevation = -89.9
        cam.azimuth = 90.0

        renderer = mujoco.Renderer(model, height=height, width=width)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (width, height))
        step = max(1, int(round((1.0 / fps) / 0.002)))
        for i, q in enumerate(frames):
            if i % step:
                continue
            data.qpos[:] = q[: model.nq]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        writer.release()
        # OpenCV writes FMP4, which browsers and most players refuse. Re-encode to
        # H.264 yuv420p with faststart so the file plays anywhere without a download.
        if shutil.which("ffmpeg"):
            tmp_mp4 = out.with_suffix(".h264.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out),
                            "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(tmp_mp4)], check=True)
            tmp_mp4.replace(out)
        else:
            print("   WARNING: ffmpeg absent, leaving FMP4 which many players reject")
        print(f"   camera: centre ({cx:.2f}, {cy:.2f}) span {span:.2f} m")
    finally:
        tmp_xml.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--actions", required=True, help="JSON list, or @path to a JSON file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--namo-config", default=str(DEFAULT_CFG))
    args = ap.parse_args()

    spec = args.actions
    actions = json.loads(Path(spec[1:]).read_text() if spec.startswith("@") else spec)

    from render_episode_video import _read_qpos_frames

    td = Path(tempfile.mkdtemp())            # cleaned by hand: the GL teardown below
    qpos = td / "qpos.txt"                   # would otherwise fire inside the context
    n = resolve_and_replay(args.xml, args.namo_config, actions, qpos)
    frames = _read_qpos_frames(qpos)
    if not frames:
        raise SystemExit("no qpos frames captured; NAMO_QPOS_DUMP produced nothing")
    render_topdown(Path(args.xml), frames, Path(args.out))
    shutil.rmtree(td, ignore_errors=True)   # the C++ dump leaves siblings behind
    size = Path(args.out).stat().st_size
    print(f"{args.out}  {n} pushes, {len(frames)} ticks, {size/1e6:.1f} MB")
    sys.stdout.flush()
    os._exit(0)                              # skip mujoco.egl's raising __del__


if __name__ == "__main__":
    main()
