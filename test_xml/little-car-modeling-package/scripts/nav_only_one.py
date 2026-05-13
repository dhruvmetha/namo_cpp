"""Run a single navigate_to call with NAMO_QPOS_DUMP set so we can render
a video of just that one nav attempt.

Usage:
    NAMO_QPOS_DUMP=tmp/x.qpos python nav_only_one.py X Y THETA_DEG
"""
import math
import os
import sys
from pathlib import Path

REPO = Path("/common/home/dm1487/robotics_research/ktamp/namo")
HOST = os.uname().nodename.split('.')[0]
sys.path.insert(0, str(REPO / f"build_python_mjxrl_{HOST}"))
sys.path.insert(0, str(REPO / "python"))

import namo_rl

xml = str(REPO / "test_xml/little-car-modeling-package/artifacts/nav_env_3000e.xml")
cfg = str(REPO / "config/namo_config_car.yaml")

if len(sys.argv) < 4:
    print("Usage: nav_only_one.py X Y THETA_DEG", file=sys.stderr)
    sys.exit(1)
gx = float(sys.argv[1]); gy = float(sys.argv[2]); gt = math.radians(float(sys.argv[3]))

env = namo_rl.RLEnvironment(xml, cfg, False)
env.reset()
obs = env.get_observation()
rp = obs.get("robot_pose", [0, 0, 0])
print(f"start: ({rp[0]:+.3f},{rp[1]:+.3f},{math.degrees(rp[2]):+.2f}°)", flush=True)
print(f"goal:  ({gx:+.3f},{gy:+.3f},{math.degrees(gt):+.2f}°)", flush=True)

r = env.navigate_to(gx, gy, gt)
print(f"success={r.success} reason={r.failure_reason} col={r.collision_object}")
print(f"  end ({r.final_x:+.4f},{r.final_y:+.4f}) yaw={math.degrees(r.final_theta):+.2f}°  "
      f"pos_err={r.pos_error_m*1000:.1f}mm  yaw_err={math.degrees(r.yaw_error_rad):.2f}°  "
      f"steps={r.steps_used}")
