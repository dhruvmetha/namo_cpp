#!/usr/bin/env python3
"""Materialize a live simulator state back into a standalone MuJoCo scene XML.

Why this exists. `region_opening._explore_from_state` only ever sweeps `adjacency[robot_label]`,
so a boundary between two NON-robot regions is never swept. In a two-hop scene the second keyhole
is exactly such a boundary at t=0 (its blocker is unreachable at t=0 in 813/827 measured cases), so
keyhole 2 cannot be labelled from the original XML at all. Materializing the post-keyhole-1 state as
a fresh scene turns keyhole 2 into an ordinary robot-adjacent boundary, which the existing
collection machinery already handles.

What is written. Every movable object's SE(2) pose and the robot's SE(2) pose, into a copy of the
source scene. Everything else (walls, sizes, materials, actuators, the goal site) is copied verbatim
from the template.

What is deliberately NOT written. Out-of-plane state: object roll/pitch, exact settled z, wheel and
caster joint angles, and all velocities. The emitted scene is a clean at-rest scene at the same SE(2)
configuration. The region graph is built from SE(2) pose + half-extents, so this flattening cannot
change region identity — `verify_roundtrip` in scripts/pipeline/materialize_keyhole2.py checks that
claim per scene rather than assuming it.

The robot orientation is the piece that had no writer before this module: both existing writers
(external_executor/xml_builder.py) set position only, so a naive round-trip silently dropped the
car's yaw and produced a scene whose reachability differed from the state it was supposed to encode.
Here the robot body gets an explicit `quat`, which is convention-independent (MuJoCo quaternions are
always w x y z regardless of `<compiler angle=...>`).

    from namo.core.state_to_xml import write_state_xml
    write_state_xml(env.get_observation(), template_xml, out_xml)
"""

import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Sequence, Tuple

MOVABLE_SUFFIX = "_movable"
POSE_SUFFIX = "_pose"
GOAL_SITE_NAME = "goal"

# Body names that mean "the robot" across the point and car scene generators. The car scenes name the
# robot body "car" (its geoms are front_chassis_collision / rear_chassis_collision), the point scenes
# name both body and geom "robot".
ROBOT_BODY_NAMES = ("car", "robot")

_ORIENTATION_ATTRS = ("quat", "euler", "axisangle", "xyaxes", "zaxis")


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Yaw in radians -> MuJoCo quaternion (w, x, y, z)."""
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _uses_radians(root: ET.Element) -> bool:
    compiler = root.find("compiler")
    if compiler is None:
        return False
    return compiler.get("angle", "degree").lower() == "radian"


def _keep_z(elem: ET.Element, default_z: float) -> float:
    parts = (elem.get("pos") or "").split()
    return float(parts[2]) if len(parts) >= 3 else default_z


def _set_yaw_quat(elem: ET.Element, yaw: float) -> None:
    """Set orientation to a pure yaw, clearing any competing specifier.

    MuJoCo rejects an element carrying two orientation attributes, so the others must go.
    """
    for attr in _ORIENTATION_ATTRS:
        if attr in elem.attrib:
            del elem.attrib[attr]
    w, x, y, z = yaw_to_quat(yaw)
    elem.set("quat", f"{w:.17g} {x:.17g} {y:.17g} {z:.17g}")


def _find_robot_body(worldbody: ET.Element) -> ET.Element:
    for name in ROBOT_BODY_NAMES:
        body = worldbody.find(f"./body[@name='{name}']")
        if body is not None:
            return body
    raise ValueError(
        f"no robot body found under worldbody; looked for {ROBOT_BODY_NAMES}, "
        f"saw {[b.get('name') for b in worldbody.findall('./body')]}"
    )


def movable_names_from_observation(observation: Dict[str, Sequence[float]]) -> Tuple[str, ...]:
    """The movable object ids present in an `env.get_observation()` dict, sorted."""
    return tuple(sorted(
        k[: -len(POSE_SUFFIX)] for k in observation
        if k.endswith(POSE_SUFFIX) and k[: -len(POSE_SUFFIX)].endswith(MOVABLE_SUFFIX)
    ))


def write_state_xml(
    observation: Dict[str, Sequence[float]],
    template_xml: str,
    out_xml: str,
    goal_override: Optional[Tuple[float, float, float]] = None,
) -> str:
    """Emit `out_xml`: the `template_xml` scene with every movable and the robot at `observation`.

    Args:
        observation: `env.get_observation()` — `"<obj>_pose"` and `"robot_pose"` -> [x, y, theta_rad].
        template_xml: the scene the state came from (or any scene with the same bodies).
        out_xml: destination path; parent directories are created.
        goal_override: optional (x, y, z) for the goal site. Left untouched when None, which is
            what keyhole materialization wants — the task goal does not move when an object does.

    Returns:
        `out_xml`.

    Raises:
        ValueError: the template has no worldbody, no robot body, or a body layout this writer
            cannot express faithfully (an obstacle body carrying its own pose). Failing loudly beats
            emitting a scene that silently disagrees with the state it claims to encode.
    """
    tree = ET.parse(template_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{template_xml}: no <worldbody>")
    to_template_angle = (lambda r: r) if _uses_radians(root) else math.degrees

    # ---- movable objects: pose lives on the geom inside a pose-less body ----
    for name in movable_names_from_observation(observation):
        pose = observation[name + POSE_SUFFIX]
        body = worldbody.find(f"./body[@name='{name}']")
        if body is None:
            raise ValueError(f"{template_xml}: observation has '{name}' but the scene has no such body")
        # The generated scenes put the whole pose on the geom and leave the body at the origin. If a
        # body ever carries its own pose the object's world pose is body ∘ geom, and writing the geom
        # alone would be wrong by exactly that offset.
        for attr in ("pos",) + _ORIENTATION_ATTRS:
            if attr in body.attrib:
                raise ValueError(
                    f"{template_xml}: body '{name}' carries {attr}='{body.get(attr)}'; this writer "
                    f"assumes obstacle bodies are pose-less and the geom holds the full pose"
                )
        geom = body.find(f"./geom[@name='{name}']")
        if geom is None:
            raise ValueError(f"{template_xml}: body '{name}' has no geom named '{name}'")

        geom.set("pos", f"{float(pose[0]):.17g} {float(pose[1]):.17g} {_keep_z(geom, 0.05):.17g}")
        # Objects are boxes resting on the floor: roll/pitch are template constants, only yaw moves.
        euler_parts = (geom.get("euler") or "0 0 0").split()
        roll = float(euler_parts[0]) if len(euler_parts) >= 1 else 0.0
        pitch = float(euler_parts[1]) if len(euler_parts) >= 2 else 0.0
        if "quat" in geom.attrib:
            del geom.attrib["quat"]
        geom.set("euler", f"{roll:.17g} {pitch:.17g} {to_template_angle(float(pose[2])):.17g}")

    # ---- robot: pose lives on the body (free-jointed), and its YAW is the piece nothing wrote ----
    robot_pose = observation.get("robot" + POSE_SUFFIX)
    if robot_pose is None:
        raise ValueError("observation has no 'robot_pose'")
    robot_body = _find_robot_body(worldbody)
    robot_geom = robot_body.find("./geom[@name='robot']")
    if robot_geom is not None and "pos" in robot_geom.attrib and robot_body.get("pos") is None:
        # point-robot layout: a single geom named 'robot' holds the position, body has none
        robot_geom.set("pos", f"{float(robot_pose[0]):.17g} {float(robot_pose[1]):.17g} {_keep_z(robot_geom, 0.15):.17g}")
        _set_yaw_quat(robot_geom, float(robot_pose[2]))
    else:
        # car layout: <body name="car" pos="..."><freejoint/> ... </body>
        robot_body.set("pos", f"{float(robot_pose[0]):.17g} {float(robot_pose[1]):.17g} {_keep_z(robot_body, 0.01):.17g}")
        _set_yaw_quat(robot_body, float(robot_pose[2]))

    if goal_override is not None:
        site = worldbody.find(f".//site[@name='{GOAL_SITE_NAME}']")
        if site is None:
            raise ValueError(f"{template_xml}: no <site name='{GOAL_SITE_NAME}'>")
        site.set("pos", f"{goal_override[0]:.17g} {goal_override[1]:.17g} {goal_override[2]:.17g}")

    os.makedirs(os.path.dirname(os.path.abspath(out_xml)), exist_ok=True)
    tree.write(out_xml, encoding="unicode", xml_declaration=True)
    return out_xml
