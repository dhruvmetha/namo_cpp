#!/usr/bin/env python3
"""Render a single successful region-opening episode to an MP4.

Replays the action_sequence from a PKL on the original env XML, captures
per-tick qpos via NAMO_QPOS_DUMP, then renders offscreen with MuJoCo.

Usage:
    NAMO_QPOS_DUMP unused — handled by this script.
    python scripts/render_episode_video.py \\
        --xml /scratch/dm1487/datasets/car_envs/.../env_0084_pair_001.xml \\
        --pkl /path/to/<host>_env_NNNNNN_results.pkl \\
        --episode-idx 0 \\
        --output /scratch/dm1487/videos/env_0084_pair_001.mp4 \\
        --namo-config config/namo_config_complete_skill15_car_1x.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path

import cv2
import mujoco

WIDTH = 1280
HEIGHT = 720
FPS = 30
SIM_TIMESTEP_S = 0.002
TICKS_PER_FRAME = max(1, int(round((1.0 / FPS) / SIM_TIMESTEP_S)))  # ~17

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "build_python"))
sys.path.insert(0, str(REPO / "python"))


def _read_qpos_frames(path: Path) -> list[list[float]]:
    """Parse 'phase nq q0 q1 ... q(nq-1)' lines."""
    frames = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                nq = int(parts[1])
                q = [float(x) for x in parts[2:2 + nq]]
            except ValueError:
                continue
            if q:
                frames.append(q)
    return frames


def replay_and_dump(xml: str, namo_config: str, action_sequence: list, qpos_path: Path) -> None:
    """Replay the action_sequence on env, dumping qpos to qpos_path."""
    os.environ["NAMO_QPOS_DUMP"] = str(qpos_path)
    # Touch the file so the C++ side opens it fresh
    qpos_path.unlink(missing_ok=True)

    import namo_rl

    env = namo_rl.RLEnvironment(xml, namo_config, visualize=False)
    env.reset()
    for action_dict in action_sequence:
        action = namo_rl.Action()
        action.object_id = action_dict["object_id"]
        target = action_dict["target"]
        action.x = float(target[0])
        action.y = float(target[1])
        action.theta = float(target[2])
        action.edge_idx = int(action_dict.get("edge_idx", -1))
        action.depth = int(action_dict.get("depth", -1))
        env.step(action)


def render_qpos_to_mp4(xml_path: Path, qpos_frames: list[list[float]], output_mp4: Path) -> None:
    """Encode qpos frames into an MP4 via MuJoCo offscreen rendering."""
    # Inject offscreen size into a temp XML so <include>s still resolve
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_path.read_text())
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", str(WIDTH))
    glob.set("offheight", str(HEIGHT))

    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=xml_path.parent, delete=False) as f:
        f.write(ET.tostring(root, encoding="unicode"))
        tmp_xml = Path(f.name)

    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_xml))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

        # Top-down camera: position above arena center, looking straight down
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.0]
        cam.distance = 3.5
        cam.elevation = -89.0  # nearly straight down
        cam.azimuth = 90.0

        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_mp4), fourcc, FPS, (WIDTH, HEIGHT))

        for i, q in enumerate(qpos_frames):
            if i % TICKS_PER_FRAME != 0:
                continue
            data.qpos[:] = q[: model.nq]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
    finally:
        tmp_xml.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--episode-idx", type=int, default=0,
                    help="Which episode (0-indexed) within the PKL to render")
    ap.add_argument("--output", required=True)
    ap.add_argument("--namo-config",
                    default="config/namo_config_complete_skill15_car_1x.yaml")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    eps = d.get("episode_results") or []
    if args.episode_idx >= len(eps):
        print(f"ERROR: only {len(eps)} episodes in {pkl_path}", file=sys.stderr)
        return 1
    ep = eps[args.episode_idx]
    if not ep.get("success"):
        print(f"WARNING: episode {args.episode_idx} did NOT succeed", file=sys.stderr)
    actions = ep.get("action_sequence") or []
    if not actions:
        print(f"ERROR: episode has empty action_sequence", file=sys.stderr)
        return 1

    print(f"  XML:     {args.xml}", file=sys.stderr)
    print(f"  Actions: {len(actions)} ({', '.join(a['object_id'] for a in actions)})", file=sys.stderr)
    print(f"  Output:  {args.output}", file=sys.stderr)

    with tempfile.NamedTemporaryFile(suffix=".qpos", delete=False) as f:
        qpos_path = Path(f.name)
    try:
        replay_and_dump(args.xml, args.namo_config, actions, qpos_path)
        if not qpos_path.exists():
            print(f"ERROR: no qpos dump produced at {qpos_path}", file=sys.stderr)
            return 1
        frames = _read_qpos_frames(qpos_path)
        print(f"  qpos frames captured: {len(frames)}", file=sys.stderr)
        if not frames:
            print(f"ERROR: empty qpos dump", file=sys.stderr)
            return 1
        render_qpos_to_mp4(Path(args.xml), frames, Path(args.output))
        print(f"  ✓ MP4 written: {args.output}", file=sys.stderr)
    finally:
        qpos_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
