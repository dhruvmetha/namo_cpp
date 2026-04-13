from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from car_model.generate_model import generate_all
from car_model.square_eval import evaluate_square_path

PYTHON = "/home/shanoriel/miniforge3/envs/leworldmodel/bin/python"
ASSET_ROOT = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "square_path"
VIDEO_PATH = OUTPUT_DIR / "little_car_square_path.mp4"
TRAJECTORY_PATH = OUTPUT_DIR / "little_car_square_path_xy.png"
RESULT_JSON_PATH = OUTPUT_DIR / "little_car_square_path_metrics.json"
FRAME_DIR = OUTPUT_DIR / "frames"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    generate_all(ASSET_ROOT)
    model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "mjcf" / "little_car_scene.xml"))
    renderer = mujoco.Renderer(model, height=720, width=960)

    frame_every = 10
    fps = int(round(1.0 / (model.opt.timestep * frame_every)))
    xy_samples: list[tuple[float, float]] = []
    captured_frame_paths: list[Path] = []
    step_index = 0
    video_writer = imageio.get_writer(VIDEO_PATH, fps=fps, codec="libx264", quality=8, format="FFMPEG")

    def capture_frame(sim_model: mujoco.MjModel, sim_data: mujoco.MjData) -> None:
        nonlocal step_index
        if step_index % frame_every == 0:
            renderer.update_scene(sim_data, camera="square_path_capture")
            frame = renderer.render()
            video_writer.append_data(frame)
            if len(captured_frame_paths) < 4:
                frame_path = FRAME_DIR / f"frame_{len(captured_frame_paths):02d}.png"
                imageio.imwrite(frame_path, frame)
                captured_frame_paths.append(frame_path)
        car_body_id = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_BODY, "car")
        xy_samples.append((float(sim_data.xpos[car_body_id][0]), float(sim_data.xpos[car_body_id][1])))
        step_index += 1

    try:
        result = evaluate_square_path(ASSET_ROOT, step_callback=capture_frame)
    finally:
        video_writer.close()
        renderer.close()

    RESULT_JSON_PATH.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    xy = np.asarray(xy_samples)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=140)
    ax.plot(xy[:, 0], xy[:, 1], linewidth=2.0, color="#1f77b4")
    ax.scatter([xy[0, 0]], [xy[0, 1]], color="green", label="start", zorder=3)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], color="red", label="end", zorder=3)
    ax.set_title("Little car square-path trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(TRAJECTORY_PATH)
    plt.close(fig)

    print(f"Using Python: {PYTHON}")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Trajectory plot: {TRAJECTORY_PATH}")
    print(f"Metrics JSON: {RESULT_JSON_PATH}")
    print(f"Saved still frames: {[str(path) for path in captured_frame_paths]}")
    print(json.dumps({
        "fps": fps,
        "frame_every": frame_every,
        "render_resolution": [960, 720],
        "captured_samples": len(xy_samples),
        "result": result.to_dict(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
