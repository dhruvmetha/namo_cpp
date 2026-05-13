"""Scan generated envs for car-in-obstacle bugs using the WAVEFRONT (authoritative).
Reports cars whose grid cell is occupied (region_map == 0) or unlabeled."""
import os, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, "/common/home/dm1487/robotics_research/ktamp/namo/build_python_mjxrl_" + os.uname().nodename.split('.')[0])
sys.path.insert(0, "/common/home/dm1487/robotics_research/ktamp/namo/python")
sys.path.insert(0, "/common/home/dm1487/robotics_research/ktamp/mujoco_env_creator")

import namo_rl
from wavefront_snapshot import WavefrontSnapshotExporter
import yaml

NAMO_CFG = "/common/home/dm1487/robotics_research/ktamp/namo/config/namo_config_car.yaml"
ROBOT_SIZE = (0.052, 0.052)


def parse_car_pos(xml_path):
    tree = ET.parse(xml_path); root = tree.getroot()
    for body in root.iter("body"):
        if body.get("name") == "car":
            pos = [float(v) for v in body.get("pos", "0 0 0").split()]
            return (pos[0], pos[1])
    return None


def parse_goal_pos(xml_path):
    tree = ET.parse(xml_path); root = tree.getroot()
    for site in root.iter("site"):
        if site.get("name") == "goal":
            pos = [float(v) for v in site.get("pos", "0 0 0").split()]
            return (pos[0], pos[1])
    return None


def check_one(xml_path):
    car = parse_car_pos(xml_path)
    goal = parse_goal_pos(xml_path)
    if car is None or goal is None:
        return None

    env = namo_rl.RLEnvironment(str(xml_path), NAMO_CFG, visualize=False)
    exporter = WavefrontSnapshotExporter(env, resolution=0.01,
                                          robot_half_extent_override=ROBOT_SIZE)
    rng = np.random.default_rng(0)
    snap = exporter.build_snapshot(xml_path=str(xml_path), config_path=NAMO_CFG,
                                   goal_radius=0.05, goals_per_region=0, rng=rng)
    rm = np.asarray(snap.region_map)
    W, H = rm.shape
    xmin, xmax, ymin, ymax = exporter.bounds

    def to_grid(px, py):
        gx = int((px - xmin) / (xmax - xmin) * W)
        gy = int((py - ymin) / (ymax - ymin) * H)
        return max(0, min(W - 1, gx)), max(0, min(H - 1, gy))

    cgx, cgy = to_grid(*car)
    ggx, ggy = to_grid(*goal)
    car_val = int(rm[cgx, cgy])
    goal_val = int(rm[ggx, ggy])
    return {"car_val": car_val, "goal_val": goal_val,
            "car_in_region": car_val > 0, "goal_in_region": goal_val > 0,
            "same_region": car_val > 0 and goal_val > 0 and car_val == goal_val}


def main():
    root = Path("/common/home/dm1487/scratch_namo/generated_car_envs")
    samples = []
    for tdir in sorted(root.glob("*/benchmark_*")):
        # 5 envs per template
        xmls = sorted(tdir.rglob("env_*_pair_*.xml"))[:5]
        for x in xmls:
            samples.append(x)

    car_in_obstacle = 0
    goal_in_obstacle = 0
    same_region = 0
    total = 0

    for x in samples:
        try:
            r = check_one(x)
        except Exception as e:
            print(f"  SKIP {x.name}: {e}")
            continue
        if r is None:
            continue
        total += 1
        if not r["car_in_region"]:
            car_in_obstacle += 1
            print(f"  CAR IN OBSTACLE: {x} (region_map cell = {r['car_val']})")
        if not r["goal_in_region"]:
            goal_in_obstacle += 1
            print(f"  GOAL IN OBSTACLE: {x} (region_map cell = {r['goal_val']})")
        if r["same_region"]:
            same_region += 1
            print(f"  SAME REGION: {x} (region {r['car_val']})")

    print()
    print(f"Total: {total}")
    print(f"  car in obstacle:  {car_in_obstacle}")
    print(f"  goal in obstacle: {goal_in_obstacle}")
    print(f"  same region:      {same_region}")


if __name__ == "__main__":
    main()
