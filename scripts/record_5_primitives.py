"""Generate 5 push-primitive MP4s at depth 10 for the wide_1x_car scene.

For each of 5 representative edges, runs ``generate_motion_primitives_db``
with ``--single-edge N --min-push-steps 10`` (depth-10 only), captures the
per-tick qpos via NAMO_QPOS_DUMP, slices out the wide-scene segment from
the dump (binary runs all 3 scenes per invocation), and renders to MP4 at
30 FPS sim-time (= 1× real-time playback).

Output: /tmp/primitive_videos/wide_edge{N}_steps10.mp4 (5 files).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco


# ── Constants ───────────────────────────────────────────────────────────
NAMO_CPP = Path(__file__).resolve().parent.parent
BINARY = NAMO_CPP / "build" / "generate_motion_primitives_db"
CONFIG = NAMO_CPP / "config" / "namo_config_complete_skill15_car_1x.yaml"
WIDE_XML = NAMO_CPP / "data" / "nominal_primitive_scene_wide_1x_car.xml"

# Order generate_motion_primitives_db processes scenes in
# (tools/generate_motion_primitives_db.cpp:378-382): square, wide, tall.
N_SCENES = 3
WIDE_SCENE_IDX = 1   # 0 = square, 1 = wide, 2 = tall

# 5 edges spanning corner/mid/cross-face for variety on the wide object
# (0.15 m × 0.0834 m). Indices 0..29 are top/bottom face, 30..59 are
# right/left face. Edge 0 = top-face corner, 14 = top-face mid, etc.
EDGES = [0, 7, 14, 30, 44]

WIDTH = 1280
HEIGHT = 720
FPS = 30
SIM_TIMESTEP_S = 0.002
TICKS_PER_FRAME = int(round((1.0 / FPS) / SIM_TIMESTEP_S))  # = 17
CAMERA_DISTANCE_FACTOR = 1.4

OUTPUT_DIR = Path("/tmp/primitive_videos")


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


def _run_primitive_gen(edge: int, qpos_out: Path) -> None:
    """Run the C++ binary for a single edge, dumping qpos to ``qpos_out``."""
    if qpos_out.exists():
        qpos_out.unlink()
    env = os.environ.copy()
    env["NAMO_QPOS_DUMP"] = str(qpos_out)
    env["LD_LIBRARY_PATH"] = f"{env.get('MJ_PATH', '')}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    args = [
        str(BINARY),
        "--config", str(CONFIG),
        "--scenes-suffix", "_1x_car",
        "--single-edge", str(edge),
        "--min-push-steps", "10",
    ]
    proc = subprocess.run(args, env=env, cwd=str(NAMO_CPP),
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print(proc.stderr.decode(), file=sys.stderr)
        raise RuntimeError(f"primitive gen exited {proc.returncode} for edge {edge}")


def _render(xml_path: Path, qpos_frames: list[list[float]], output_mp4: Path,
            label: str) -> None:
    xml_text = _inject_offscreen_size(xml_path, WIDTH, HEIGHT)
    sibling = xml_path.parent / f".__render_{xml_path.stem}_{os.getpid()}.xml"
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

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.0]
    camera.distance = 0.7 * CAMERA_DISTANCE_FACTOR
    camera.azimuth = 90.0
    camera.elevation = -90.0

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_mp4),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"VideoWriter failed at {output_mp4}")

    n = 0
    for i in range(0, len(qpos_frames), TICKS_PER_FRAME):
        q = qpos_frames[i]
        nq_use = min(len(q), model.nq)
        data.qpos[:nq_use] = q[:nq_use]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        rgb = renderer.render()
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        n += 1

    writer.release()
    renderer.close()
    print(f"[render] {label}: wrote {output_mp4.name} ({n} frames, "
          f"{n / FPS:.2f} s @ {FPS} fps)")


def main() -> int:
    qpos_tmp = Path("/tmp/_record5_qpos.txt")
    for edge in EDGES:
        print(f"\n── edge {edge} ── running primitive gen …")
        _run_primitive_gen(edge, qpos_tmp)
        frames = _read_qpos_frames(qpos_tmp)
        if not frames:
            print(f"[edge {edge}] qpos dump empty — skipping")
            continue
        per_scene = len(frames) // N_SCENES
        start = WIDE_SCENE_IDX * per_scene
        end = start + per_scene
        wide_segment = frames[start:end]
        if not wide_segment:
            print(f"[edge {edge}] wide segment empty — skipping")
            continue
        out_mp4 = OUTPUT_DIR / f"wide_edge{edge:02d}_steps10.mp4"
        _render(WIDE_XML, wide_segment, out_mp4, f"edge {edge}")

    try:
        qpos_tmp.unlink()
    except OSError:
        pass

    print(f"\nAll MP4s under {OUTPUT_DIR}/")
    for mp4 in sorted(OUTPUT_DIR.glob("wide_edge*_steps10.mp4")):
        print(f"  {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
