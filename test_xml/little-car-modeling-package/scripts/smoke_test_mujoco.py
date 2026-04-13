from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from car_model.generate_model import generate_all
from car_model.parameters import default_parameters


if __name__ == "__main__":
    params = default_parameters()
    output = generate_all(PROJECT_ROOT / "assets")

    urdf_model = mujoco.MjModel.from_xml_path(str(output["urdf"]))
    urdf_data = mujoco.MjData(urdf_model)
    for _ in range(10):
        mujoco.mj_step(urdf_model, urdf_data)

    scene_model = mujoco.MjModel.from_xml_path(str(output["mjcf_scene"]))
    scene_data = mujoco.MjData(scene_model)
    free_joint_qpos = scene_model.jnt_qposadr[0]
    scene_data.qpos[free_joint_qpos : free_joint_qpos + 3] = np.array([0.0, 0.0, params.scene_spawn_height_m])
    scene_data.qpos[free_joint_qpos + 3 : free_joint_qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(scene_model, scene_data)
    for _ in range(4000):
        mujoco.mj_step(scene_model, scene_data)

    car_body_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, "car")
    up_dot = float(scene_data.xmat[car_body_id][8])

    print(
        {
            "urdf": str(output["urdf"]),
            "mjcf_scene": str(output["mjcf_scene"]),
            "urdf_nbody": urdf_model.nbody,
            "urdf_njnt": urdf_model.njnt,
            "scene_nbody": scene_model.nbody,
            "scene_njnt": scene_model.njnt,
            "stable_up_dot": up_dot,
            "stable_height": float(scene_data.xpos[car_body_id][2]),
            "free_speed_norm": float(np.linalg.norm(scene_data.qvel[:6])),
        }
    )
