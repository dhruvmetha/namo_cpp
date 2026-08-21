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


def _find_robot_body_or_none(worldbody: ET.Element):
    """The robot body, or None when the scene does not declare one itself.

    Car scenes captured from the real arena pull the car in with
    `<include file=".../little_car.xml"/>`, so the body exists at load time but
    not in the file this writer edits. Those callers set the pose after loading
    instead; see `robot_pose_set_by_caller`.
    """
    for name in ROBOT_BODY_NAMES:
        body = worldbody.find(f"./body[@name='{name}']")
        if body is not None:
            return body
    return None


def _find_robot_body(worldbody: ET.Element) -> ET.Element:
    body = _find_robot_body_or_none(worldbody)
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
    robot_pose_set_by_caller: bool = False,
) -> str:
    """Emit `out_xml`: the `template_xml` scene with every movable and the robot at `observation`.

    Args:
        observation: `env.get_observation()` — `"<obj>_pose"` and `"robot_pose"` -> [x, y, theta_rad].
        template_xml: the scene the state came from (or any scene with the same bodies).
        out_xml: destination path; parent directories are created.
        goal_override: optional (x, y, z) for the goal site. Left untouched when None, which is
            what keyhole materialization wants — the task goal does not move when an object does.
        robot_pose_set_by_caller: allow a scene whose robot arrives through an `<include>`, where
            this writer cannot reach the body. The emitted scene keeps the include's spawn pose and
            the caller is responsible for calling `set_robot_pose` (or passing
            `starting_robot_pose`) after loading it. Off by default: silently emitting a scene whose
            robot sits somewhere else is exactly the divergence this module exists to prevent.

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
        # Objects are boxes resting on the floor, so only yaw moves and roll/pitch
        # get flattened, which this module documents as deliberate. Written as a
        # quaternion for the same reason the robot is: MuJoCo quats are always
        # w x y z whatever <compiler angle=...> says, and euler is not. A car
        # scene inherits angle="radian" through its <include> of little_car.xml
        # while declaring no compiler tag itself, so a writer reading only the
        # parent file assumed degrees and emitted -179.99 for -3.1414 rad. MuJoCo
        # then read that as radians: -179.99 + 29*2pi = +2.2216, a 53 degree error
        # present before any physics ran.
        _set_yaw_quat(geom, float(pose[2]))

    # ---- robot: pose lives on the body (free-jointed), and its YAW is the piece nothing wrote ----
    robot_pose = observation.get("robot" + POSE_SUFFIX)
    if robot_pose is None:
        raise ValueError("observation has no 'robot_pose'")
    robot_body = _find_robot_body_or_none(worldbody)
    if robot_body is None and not robot_pose_set_by_caller:
        _find_robot_body(worldbody)  # raises, listing what the scene actually has
    # When the include owns the car there is nothing here to write, and the goal
    # site below still needs handling, so only the robot is skipped.
    if robot_body is not None:
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
