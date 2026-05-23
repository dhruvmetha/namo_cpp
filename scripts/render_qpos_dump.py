"""Render a NAMO_QPOS_DUMP file to one MP4 per scene at 1× real-time.

Takes the per-tick qpos dump that ``generate_motion_primitives_db`` writes
when ``NAMO_QPOS_DUMP=<path>`` is set, splits it by scene (the binary
processes scenes in the order they appear in its source: square → wide →
tall), and writes one offscreen-rendered MP4 per segment.

Frames are emitted at 30 FPS in sim time (= 1/30 s ÷ 0.002 s timestep = 1
frame per 16-17 mj_step ticks), so the resulting video plays at real-time
speed even though the offscreen render runs much faster than wall clock.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco


WIDTH = 1280
HEIGHT = 720
FPS = 30
SIM_TIMESTEP_S = 0.002
TICKS_PER_FRAME = int(round((1.0 / FPS) / SIM_TIMESTEP_S))  # = ~17
CAMERA_DISTANCE_FACTOR = 1.4

# Scene order MUST match generate_motion_primitives_db.cpp's `scenes` vector
# (tools/generate_motion_primitives_db.cpp:378-382): square, wide, tall.
SCENES = [
    ("square", "data/nominal_primitive_scene_square_1x_car.xml"),
    ("wide",   "data/nominal_primitive_scene_wide_1x_car.xml"),
    ("tall",   "data/nominal_primitive_scene_tall_1x_car.xml"),
]


def _inject_offscreen_size(xml_path: Path, width: int, height: int) -> str:
    root = ET.fromstring(xml_path.read_text())
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", str(width))
    glob.set("offheight", str(height))
    return ET.tostring(root, encoding="unicode")


def _read_qpos_frames(path: Path) -> list[list[float]]:
    """Parse 'phase nq q0 q1 ... q(nq-1)' lines into a list of qpos arrays."""
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


def render_segment(
    xml_path: Path,
    qpos_frames: list[list[float]],
    output_mp4: Path,
    scene_name: str,
) -> None:
    """Render one scene's qpos slice to an MP4."""
    # Load model with offscreen size injected (write next to original so
    # relative <include> paths still resolve).
    xml_text = _inject_offscreen_size(xml_path, WIDTH, HEIGHT)
    sibling = xml_path.parent / f".__render_{xml_path.stem}.xml"
    sibling.write_text(xml_text)
    try:
        model = mujoco.MjModel.from_xml_path(str(sibling))
    finally:
        try:
            sibling.unlink()
        except OSError:
            pass

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    # Top-down camera framed on the workspace.
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.0]
    extent_m = 0.7  # workspace is ~1.67 m wide; use a tighter framing around origin
    camera.distance = extent_m * CAMERA_DISTANCE_FACTOR
    camera.azimuth = 90.0
    camera.elevation = -90.0

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"VideoWriter failed to open at {output_mp4}")

    n_frames = 0
    for i in range(0, len(qpos_frames), TICKS_PER_FRAME):
        q = qpos_frames[i]
        nq_use = min(len(q), model.nq)
        if nq_use == 0:
            continue
        data.qpos[:nq_use] = q[:nq_use]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        rgb = renderer.render()
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        n_frames += 1

    writer.release()
    renderer.close()
    duration_s = n_frames / FPS
    print(
        f"[render] {scene_name}: wrote {output_mp4} "
        f"({n_frames} frames, {duration_s:.2f} s at {FPS} fps)"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--qpos-dump", required=True,
        help="Path to NAMO_QPOS_DUMP output file.",
    )
    p.add_argument(
        "--out-dir", default="/tmp",
        help="Output directory for MP4 files (default: /tmp).",
    )
    p.add_argument(
        "--prefix", default="primitive",
        help="Filename prefix (default: 'primitive').",
    )
    args = p.parse_args()

    qpos_path = Path(args.qpos_dump).resolve()
    if not qpos_path.exists():
        print(f"[render] qpos dump not found: {qpos_path}", file=sys.stderr)
        return 2

    frames = _read_qpos_frames(qpos_path)
    total = len(frames)
    n_scenes = len(SCENES)
    if total == 0:
        print("[render] qpos dump is empty", file=sys.stderr)
        return 1
    if total % n_scenes != 0:
        print(
            f"[render] WARNING: {total} frames doesn't divide evenly into "
            f"{n_scenes} scenes — last scene's segment may be short or "
            f"include leftover ticks. Assuming square→wide→tall order."
        )
    seg = total // n_scenes
    out_dir = Path(args.out_dir).resolve()

    for i, (scene_name, scene_xml_rel) in enumerate(SCENES):
        start = i * seg
        end = (i + 1) * seg if i + 1 < n_scenes else total
        segment = frames[start:end]
        xml_path = Path(scene_xml_rel).resolve()
        if not xml_path.exists():
            print(f"[render] {scene_name}: XML not found, skipping → {xml_path}")
            continue
        output_mp4 = out_dir / f"{args.prefix}_{scene_name}_edge0_steps10.mp4"
        render_segment(xml_path, segment, output_mp4, scene_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
