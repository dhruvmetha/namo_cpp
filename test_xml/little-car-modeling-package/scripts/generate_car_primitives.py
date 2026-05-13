"""Generate motion primitives for the tiny car robot.

For each object shape (square, wide, tall), generates push primitives by:
1. Placing the car at each edge approach point, facing the object
2. Driving forward for N push steps
3. Recording the object displacement (delta_x, delta_y, delta_theta)

Output: binary .dat files matching the existing primitive format (14 bytes per primitive).
"""
from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAMO_ROOT = PROJECT_ROOT.parents[1]  # namo/ directory
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from car_model.parameters import default_parameters


# ── Object configurations (scaled for tiny car) ──────────────────────────────

@dataclass
class ObjectConfig:
    name: str
    half_size_x: float  # MuJoCo box half-size x
    half_size_y: float  # MuJoCo box half-size y
    half_size_z: float  # MuJoCo box half-size z
    mass: float
    description: str


# Scale factor: car is 7cm, old robot was 30cm diameter → ratio ≈ 0.233
# Keep same object/robot size ratios as original
# Original half-sizes: square 0.35x0.35, wide 0.45x0.25, tall 0.25x0.45
# Object height matches car height (7cm → half-height 3.5cm)
SCALE = 0.07 / 0.30

OBJECT_CONFIGS = [
    ObjectConfig("square", 0.35 * SCALE, 0.35 * SCALE, 0.035, 0.1, f"Square ({0.70*SCALE*100:.1f}x{0.70*SCALE*100:.1f}cm)"),
    ObjectConfig("wide",   0.45 * SCALE, 0.25 * SCALE, 0.035, 0.1, f"Wide ({0.90*SCALE*100:.1f}x{0.50*SCALE*100:.1f}cm)"),
    ObjectConfig("tall",   0.25 * SCALE, 0.45 * SCALE, 0.035, 0.1, f"Tall ({0.50*SCALE*100:.1f}x{0.90*SCALE*100:.1f}cm)"),
]


# ── Scene XML generation ─────────────────────────────────────────────────────

def generate_scene_xml(obj_config: ObjectConfig, car_params) -> str:
    """Generate a MuJoCo XML scene with the car robot and a single pushable object."""

    car_xml_path = str(PROJECT_ROOT / "assets" / "mjcf" / "little_car.xml")

    # Car starts 20cm away from object (in -x direction), facing +x
    car_start_x = -(obj_config.half_size_x + 0.20)
    car_spawn_z = car_params.scene_spawn_height_m

    # Object at origin
    obj_z = obj_config.half_size_z  # bottom at z=0

    # Physics matched to nav_env_3000e.xml (cone="elliptic", floor friction 0.5/0.005/0.001,
    # object friction 0.0/0.005/0.001, mass 0.1, geom density 1).
    return f"""\
<mujoco model="car_primitive_gen_{obj_config.name}">
  <compiler angle="radian"/>
  <include file="{car_xml_path}"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" iterations="100" gravity="0 0 -9.81"/>
  <default>
    <geom density="1"/>
  </default>

  <worldbody>
    <light name="sun" pos="0 0 2" dir="0 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.85 0.85 0.85 1"
          condim="4" friction="0.5 0.005 0.001"/>

    <!-- Pushable object at origin (mass/friction matched to 3000e movable obstacles) -->
    <body name="obstacle_1_movable" pos="0 0 {obj_z:.6f}">
      <joint type="free"/>
      <geom name="obstacle_1_movable" type="box"
            size="{obj_config.half_size_x:.6f} {obj_config.half_size_y:.6f} {obj_config.half_size_z:.6f}"
            mass="{obj_config.mass:.6f}"
            friction="0.0 0.005 0.001"
            rgba="1 0.3 0.3 1" condim="4"/>
    </body>
  </worldbody>
</mujoco>
"""


# ── Helper functions ─────────────────────────────────────────────────────────

def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert yaw angle to quaternion [w, x, y, z]."""
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def generate_edge_points(half_sx: float, half_sy: float, points_per_face: int = 15,
                         robot_half_length: float = 0.035, clearance: float = 0.005):
    """Generate approach points around the object perimeter.

    Returns list of (edge_x, edge_y, approach_heading) where:
    - edge_x, edge_y: point where robot front face contacts object
    - approach_heading: yaw angle the robot should face when pushing

    Uses same 4-face structure as C++ (points_per_face points per face).
    """
    edges = []
    offset = robot_half_length + clearance  # distance from object surface to robot center

    # Face 0: +x face (robot approaches from +x, faces -x)
    for i in range(points_per_face):
        t = -1.0 + 2.0 * (i + 0.5) / points_per_face  # -1 to +1
        edges.append((half_sx + offset, t * half_sy, math.pi))  # facing -x

    # Face 1: +y face (robot approaches from +y, faces -y)
    for i in range(points_per_face):
        t = -1.0 + 2.0 * (i + 0.5) / points_per_face
        edges.append((t * half_sx, half_sy + offset, -math.pi / 2))  # facing -y

    # Face 2: -x face (robot approaches from -x, faces +x)
    for i in range(points_per_face):
        t = -1.0 + 2.0 * (i + 0.5) / points_per_face
        edges.append((-(half_sx + offset), t * half_sy, 0.0))  # facing +x

    # Face 3: -y face (robot approaches from -y, faces +y)
    for i in range(points_per_face):
        t = -1.0 + 2.0 * (i + 0.5) / points_per_face
        edges.append((t * half_sx, -(half_sy + offset), math.pi / 2))  # facing +y

    return edges


# ── Primitive generation ─────────────────────────────────────────────────────

def generate_primitives_for_object(
    obj_config: ObjectConfig,
    points_per_face: int = 15,
    max_push_steps: int = 10,
    push_step_duration_s: float = 0.5,
    push_speed: float = 10.0,
    settle_steps: int = 500,
    verbose: bool = False,
) -> list[tuple[float, float, float, int, int]]:
    """Generate motion primitives for one object shape.

    Returns list of (delta_x, delta_y, delta_theta, edge_idx, push_steps).
    """
    car_params = default_parameters()

    # Generate scene XML
    scene_xml = generate_scene_xml(obj_config, car_params)
    scene_path = PROJECT_ROOT / "assets" / "mjcf" / f"primitive_gen_{obj_config.name}.xml"
    scene_path.write_text(scene_xml, encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    steps_per_push = int(push_step_duration_s / dt)

    # Find joint/actuator/body IDs
    car_freejoint_qpos = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")]
    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_drive")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_drive")

    # Find object free joint (the one that's NOT the car's)
    obj_joint_name = None
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "car_freejoint" and "caster" not in name and "wheel" not in name:
            obj_joint_name = name
            break
    if obj_joint_name is None:
        # Free joints are unnamed — find by body
        obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_1_movable")
        obj_joint_qpos = model.jnt_qposadr[model.body_jntadr[obj_body_id]]
    else:
        obj_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, obj_joint_name)
        obj_joint_qpos = model.jnt_qposadr[obj_joint_id]

    # Generate edge approach points
    edges = generate_edge_points(
        obj_config.half_size_x, obj_config.half_size_y,
        points_per_face=points_per_face,
        robot_half_length=car_params.body_half_length_m,
        clearance=0.005,
    )

    num_edges = len(edges)
    print(f"  {obj_config.name}: {num_edges} edges x {max_push_steps} depths = {num_edges * max_push_steps} primitives")

    primitives = []

    for edge_idx, (edge_x, edge_y, heading) in enumerate(edges):
        if verbose and edge_idx % 15 == 0:
            print(f"    Edge {edge_idx}/{num_edges} (face {edge_idx // points_per_face})")

        for push_steps in range(1, max_push_steps + 1):
            # Reset simulation
            mujoco.mj_resetData(model, data)

            # Place object at origin with identity orientation
            data.qpos[obj_joint_qpos : obj_joint_qpos + 3] = [0, 0, obj_config.half_size_z]
            data.qpos[obj_joint_qpos + 3 : obj_joint_qpos + 7] = [1, 0, 0, 0]

            # Place car at edge approach point with correct heading
            car_z = car_params.wheel_radius_m + car_params.scene_spawn_height_m
            data.qpos[car_freejoint_qpos : car_freejoint_qpos + 3] = [edge_x, edge_y, car_z]
            data.qpos[car_freejoint_qpos + 3 : car_freejoint_qpos + 7] = yaw_to_quat(heading)

            mujoco.mj_forward(model, data)

            # Settle (let everything stabilize)
            data.ctrl[left_act] = 0
            data.ctrl[right_act] = 0
            for _ in range(settle_steps):
                mujoco.mj_step(model, data)

            # Record initial object pose
            obj_pos_before = data.qpos[obj_joint_qpos : obj_joint_qpos + 3].copy()
            obj_quat_before = data.qpos[obj_joint_qpos + 3 : obj_joint_qpos + 7].copy()
            yaw_before = quat_to_yaw(obj_quat_before)

            # Drive forward (push) for push_steps * steps_per_push simulation steps
            data.ctrl[left_act] = push_speed
            data.ctrl[right_act] = push_speed
            for _ in range(push_steps * steps_per_push):
                mujoco.mj_step(model, data)

            # Stop and settle
            data.ctrl[left_act] = 0
            data.ctrl[right_act] = 0
            for _ in range(settle_steps):
                mujoco.mj_step(model, data)

            # Record final object pose
            obj_pos_after = data.qpos[obj_joint_qpos : obj_joint_qpos + 3].copy()
            obj_quat_after = data.qpos[obj_joint_qpos + 3 : obj_joint_qpos + 7].copy()
            yaw_after = quat_to_yaw(obj_quat_after)

            # Compute displacement in object's local frame (relative to initial pose)
            delta_x = obj_pos_after[0] - obj_pos_before[0]
            delta_y = obj_pos_after[1] - obj_pos_before[1]
            delta_theta = yaw_after - yaw_before

            # Normalize delta_theta to [-pi, pi]
            while delta_theta > math.pi:
                delta_theta -= 2 * math.pi
            while delta_theta < -math.pi:
                delta_theta += 2 * math.pi

            primitives.append((delta_x, delta_y, delta_theta, edge_idx, push_steps))

    return primitives


# ── Binary I/O (matches C++ format) ─────────────────────────────────────────

def save_primitives(path: Path, primitives: list[tuple[float, float, float, int, int]]):
    """Save primitives in binary format matching C++ NominalPrimitive struct.

    Format: uint32 count, then count * (float delta_x, float delta_y, float delta_theta, uint8 edge_idx, uint8 push_steps)
    """
    with open(path, "wb") as f:
        f.write(struct.pack("I", len(primitives)))
        for dx, dy, dtheta, edge_idx, push_steps in primitives:
            f.write(struct.pack("fffBB", dx, dy, dtheta, edge_idx, push_steps))

    size_bytes = path.stat().st_size
    print(f"  Saved {len(primitives)} primitives to {path} ({size_bytes} bytes)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate motion primitives for tiny car robot")
    parser.add_argument("--points-per-face", type=int, default=15, help="Edge points per object face")
    parser.add_argument("--max-push-steps", type=int, default=10, help="Max push duration levels")
    parser.add_argument("--push-speed", type=float, default=10.0, help="Wheel velocity during push (rad/s)")
    parser.add_argument("--push-step-duration", type=float, default=0.5, help="Duration per push step (seconds)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: namo/data/)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--objects", nargs="*", default=None, help="Object types to generate (square, wide, tall)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else NAMO_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Regenerate car model first
    from car_model.generate_model import generate_all
    generate_all(PROJECT_ROOT / "assets")

    # Filter objects if specified
    configs = OBJECT_CONFIGS
    if args.objects:
        configs = [c for c in configs if c.name in args.objects]

    print(f"=== Tiny Car Motion Primitive Generator ===")
    print(f"Points per face: {args.points_per_face}")
    print(f"Max push steps:  {args.max_push_steps}")
    print(f"Push speed:      {args.push_speed} rad/s")
    print(f"Step duration:   {args.push_step_duration}s")
    print(f"Output dir:      {output_dir}")
    print()

    car_params = default_parameters()
    print(f"Car: {car_params.body_size_m*100:.0f}x{car_params.body_size_m*100:.0f}cm, "
          f"wheel r={car_params.wheel_radius_m*100:.1f}cm")
    print()

    for obj in configs:
        print(f"Object: {obj.description}")
        print(f"  Half-sizes: {obj.half_size_x*100:.2f} x {obj.half_size_y*100:.2f} x {obj.half_size_z*100:.2f} cm")
        print(f"  Full size:  {obj.half_size_x*200:.1f} x {obj.half_size_y*200:.1f} x {obj.half_size_z*200:.1f} cm")
        print(f"  Mass: {obj.mass} kg")

        primitives = generate_primitives_for_object(
            obj,
            points_per_face=args.points_per_face,
            max_push_steps=args.max_push_steps,
            push_speed=args.push_speed,
            push_step_duration_s=args.push_step_duration,
            verbose=args.verbose,
        )

        # Save with naming convention matching existing: motion_primitives_15_{shape}.dat
        output_path = output_dir / f"car_motion_primitives_{args.points_per_face}_{obj.name}.dat"
        save_primitives(output_path, primitives)

        # Print some statistics
        displacements = [math.sqrt(dx**2 + dy**2) for dx, dy, _, _, _ in primitives]
        nonzero = [d for d in displacements if d > 0.001]
        if nonzero:
            print(f"  Displacement stats: min={min(nonzero)*1000:.1f}mm, max={max(nonzero)*1000:.1f}mm, "
                  f"mean={sum(nonzero)/len(nonzero)*1000:.1f}mm")
            print(f"  Zero-displacement: {len(displacements) - len(nonzero)}/{len(displacements)}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
