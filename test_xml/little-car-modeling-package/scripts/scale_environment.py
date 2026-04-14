#!/usr/bin/env python3
"""Scale a NAMO environment XML for the tiny car robot.

Takes an existing environment XML (sized for 30cm point robot) and:
  1. Scales all x,y coordinates by SCALE (0.233 = 7cm car / 30cm point robot)
  2. Fixes object heights to match car primitive objects (7cm tall)
  3. Replaces the point robot body with the diff-drive car
  4. Updates physics settings for stable diff-drive simulation

Usage:
    # Single file
    python scale_environment.py path/to/env.xml path/to/output.xml

    # Batch: copy N random XMLs from a directory tree
    python scale_environment.py --batch /path/to/env_dir /path/to/output_dir --count 5
"""

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

SCALE = 0.233           # spatial scale factor
OBJ_HALF_HEIGHT = 0.035 # 3.5cm half-height for obstacles (matches car primitives)
WALL_HALF_HEIGHT = 0.08 # 8cm half-height for walls (taller than 7cm car)
CAR_SPAWN_Z = 0.010     # 1cm drop for car wheels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _s(v: float) -> str:
    return f"{v:.6f}"


def scale_xy(pos_str: str, scale: float, z_override: float | None = None) -> str:
    """Scale x,y of a space-separated pos/euler string; optionally override z."""
    parts = pos_str.split()
    x = float(parts[0]) * scale
    y = float(parts[1]) * scale
    if len(parts) >= 3:
        z = z_override if z_override is not None else float(parts[2]) * scale
        return f"{_s(x)} {_s(y)} {_s(z)}"
    return f"{_s(x)} {_s(y)}"


def scale_box_size(size_str: str, scale: float, z_override: float | None = None) -> str:
    """Scale hx,hy of a box size; optionally override hz."""
    parts = size_str.split()
    hx = float(parts[0]) * scale
    hy = float(parts[1]) * scale
    if len(parts) >= 3:
        hz = z_override if z_override is not None else float(parts[2]) * scale
        return f"{_s(hx)} {_s(hy)} {_s(hz)}"
    return f"{_s(hx)} {_s(hy)}"


def scale_sphere_size(size_str: str, scale: float) -> str:
    r = float(size_str.split()[0]) * scale
    return _s(r)


def get_robot_start_xy(root: ET.Element) -> tuple[float, float]:
    """Extract robot starting x,y from the point robot sphere geom."""
    robot_body = root.find(".//body[@name='robot']")
    if robot_body is None:
        return 0.0, 0.0
    geom = robot_body.find("geom[@name='robot']")
    if geom is None:
        return 0.0, 0.0
    parts = geom.get("pos", "0 0 0").split()
    return float(parts[0]), float(parts[1])


def car_body_xml(start_x: float, start_y: float) -> str:
    """Inline diff-drive car body at the given start position (euler in degrees)."""
    return f"""\
    <body name="car" pos="{_s(start_x)} {_s(start_y)} {_s(CAR_SPAWN_Z)}">
      <freejoint name="car_freejoint"/>
      <inertial pos="0.000000 0 0.037500" mass="0.350000"
                diaginertia="0.000266 0.000266 0.000286"/>
      <geom name="front_chassis_collision" type="box"
            pos="0.017500 0 0.037500" size="0.017500 0.035000 0.032500"
            rgba="0.3 0.3 0.7 1"/>
      <geom name="rear_chassis_collision" type="box"
            pos="-0.017500 0 0.037500" size="0.017500 0.035000 0.032500"
            rgba="0.25 0.25 0.6 1"/>
      <geom name="front_marker" type="box"
            pos="0.034000 0 0.050500" size="0.002000 0.015000 0.010000"
            rgba="1.0 0.2 0.2 1" contype="0" conaffinity="0"/>
      <body name="rear_support_body" pos="-0.030000 0.000000 0.002500">
        <joint name="rear_caster_joint" type="ball" damping="0.0001"/>
        <inertial pos="0 0 0" mass="0.025000"
                  diaginertia="0.000001 0.000001 0.000001"/>
        <geom name="rear_support" type="sphere" pos="0 0 0" size="0.002500"
              friction="0.000000 0.000000 0.000000" rgba="0.7 0.1 0.1 1"/>
      </body>
      <body name="front_support_body" pos="0.030000 0.000000 0.002500">
        <joint name="front_caster_joint" type="ball" damping="0.0001"/>
        <inertial pos="0 0 0" mass="0.025000"
                  diaginertia="0.000001 0.000001 0.000001"/>
        <geom name="front_support" type="sphere" pos="0 0 0" size="0.002500"
              friction="0.000000 0.000000 0.000000" rgba="0.1 0.7 0.1 1"/>
      </body>
      <body name="left_wheel" pos="0.000000 0.037500 0.015000">
        <inertial pos="0 0 0" mass="0.050000"
                  diaginertia="0.000003 0.000006 0.000003"/>
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0" damping="0.0001"/>
        <geom name="left_wheel_collision" type="cylinder"
              size="0.015000 0.000500" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="right_wheel" pos="0.000000 -0.037500 0.015000">
        <inertial pos="0 0 0" mass="0.050000"
                  diaginertia="0.000003 0.000006 0.000003"/>
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0" damping="0.0001"/>
        <geom name="right_wheel_collision" type="cylinder"
              size="0.015000 0.000500" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
      </body>
    </body>"""


CAR_ACTUATORS = """\
  <actuator>
    <motor name="left_wheel_drive" joint="left_wheel_joint"
           gear="1" ctrlrange="-25.000000 25.000000"
           forcerange="-0.300000 0.300000"/>
    <motor name="right_wheel_drive" joint="right_wheel_joint"
           gear="1" ctrlrange="-25.000000 25.000000"
           forcerange="-0.300000 0.300000"/>
  </actuator>"""


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------

def scale_xml(src_path: Path, dst_path: Path, scale: float = SCALE) -> None:
    ET.register_namespace("", "")
    tree = ET.parse(src_path)
    root = tree.getroot()

    # --- physics settings ---
    option = root.find("option")
    if option is not None:
        option.set("timestep", "0.002")
        option.set("integrator", "implicitfast")
        option.set("iterations", "100")
        option.set("gravity", "0 0 -9.81")

    # --- extract robot start position before removing it ---
    rx, ry = get_robot_start_xy(root)
    sx, sy = rx * scale, ry * scale

    worldbody = root.find("worldbody")

    # --- scale walls ---
    walls_body = worldbody.find("body[@name='walls']")
    if walls_body is not None:
        for geom in walls_body.findall("geom"):
            name = geom.get("name", "")
            pos = geom.get("pos")
            size = geom.get("size")
            if pos:
                geom.set("pos", scale_xy(pos, scale, z_override=WALL_HALF_HEIGHT))
            if size:
                geom.set("size", scale_box_size(size, scale, z_override=WALL_HALF_HEIGHT))

    # --- scale goal site ---
    goal_site = worldbody.find("site[@name='goal']")
    if goal_site is not None:
        pos = goal_site.get("pos")
        if pos:
            goal_site.set("pos", scale_xy(pos, scale, z_override=0.0))
        sz = goal_site.get("size", "0.3")
        goal_site.set("size", scale_sphere_size(sz, scale))

    # --- scale origin marker sites (keep as-is, just scale) ---
    for site in worldbody.findall("site"):
        pos = site.get("pos")
        if pos:
            site.set("pos", scale_xy(pos, scale))
        sz = site.get("size")
        if sz:
            site.set("size", scale_sphere_size(sz, scale))

    # --- scale movable obstacles ---
    for body in worldbody.findall("body"):
        name = body.get("name", "")
        if not name.endswith("_movable"):
            continue
        geom = body.find("geom")
        if geom is None:
            continue
        pos = geom.get("pos")
        size = geom.get("size")
        if pos:
            geom.set("pos", scale_xy(pos, scale, z_override=OBJ_HALF_HEIGHT))
        if size:
            geom.set("size", scale_box_size(size, scale, z_override=OBJ_HALF_HEIGHT))

    # --- remove point robot body ---
    robot_body = worldbody.find("body[@name='robot']")
    if robot_body is not None:
        worldbody.remove(robot_body)

    # --- remove old actuators ---
    old_actuator = root.find("actuator")
    if old_actuator is not None:
        root.remove(old_actuator)

    # --- serialize, inject car body + actuators as raw XML strings ---
    # ElementTree can't insert raw XML fragments easily, so we patch as strings.
    xml_str = ET.tostring(root, encoding="unicode")

    # Insert car body before </worldbody>
    car_xml = car_body_xml(sx, sy)
    xml_str = xml_str.replace("</worldbody>", f"{car_xml}\n  </worldbody>")

    # Append actuators before </mujoco>
    xml_str = xml_str.replace("</mujoco>", f"{CAR_ACTUATORS}\n</mujoco>")

    # Pretty header
    xml_str = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_str

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(xml_str, encoding="utf-8")
    print(f"  {src_path.name} → {dst_path}")


def find_all_xmls(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob("*.xml"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="Source XML file or source directory (with --batch)")
    parser.add_argument("dst", help="Destination XML file or destination directory (with --batch)")
    parser.add_argument("--batch", action="store_true",
                        help="Process multiple files from src dir to dst dir")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of random XMLs to sample in batch mode (default: 5)")
    parser.add_argument("--scale", type=float, default=SCALE,
                        help=f"Spatial scale factor (default: {SCALE})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for batch sampling (default: 42)")
    args = parser.parse_args()

    if args.batch:
        src_dir = Path(args.src)
        dst_dir = Path(args.dst)
        all_xmls = find_all_xmls(src_dir)
        if not all_xmls:
            print(f"No XML files found in {src_dir}")
            return
        random.seed(args.seed)
        chosen = random.sample(all_xmls, min(args.count, len(all_xmls)))
        print(f"Scaling {len(chosen)} environments (scale={args.scale}):")
        for src_xml in chosen:
            rel = src_xml.relative_to(src_dir)
            dst_xml = dst_dir / rel
            scale_xml(src_xml, dst_xml, scale=args.scale)
    else:
        scale_xml(Path(args.src), Path(args.dst), scale=args.scale)

    print("Done.")


if __name__ == "__main__":
    main()
