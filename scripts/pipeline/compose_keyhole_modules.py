#!/usr/bin/env python3
"""Compose canonical keyhole episodes into controlled multi-keyhole scenes.

The unit is an episode ``(xml, object_id, region)``, never an XML.  Each output starts from the
first donor XML, removes every movable object, then inserts only the selected donor blockers.  The
first donor supplies the robot pose and the last donor supplies the XML goal.  Static validation
uses ``probe_static_topology`` and requires the intended blockers to appear in path order.

``fixed_template`` preserves the original blocker-only pilot.  ``same_template`` keeps one exact
wall layout and transplants only two blockers from episodes of that template.
``same_template_clutter`` additionally retains one non-boundary movable object from the host room.
``room_stitch`` is the retired whole-room stress-test mode that joins complete donor rooms through
controlled portals.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "build_python", REPO / "python", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import namo_rl  # noqa: E402
from namo import eval_sets  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_from_xml  # noqa: E402
from namo.paths import resolve  # noqa: E402
from namo.planners import get_region_snapshot  # noqa: E402
from namo.planners.opening.region_opening import CANONICAL_MIN_REACHABLE_FRACTION  # noqa: E402
from namo.runtime_profile import CANONICAL_NUM_DEPTHS  # noqa: E402
from namo.visualization.wavefront_snapshot import WavefrontSnapshotExporter  # noqa: E402
from probe_static_topology import probe_one, shortest_region_path  # noqa: E402
from verify_geom_disjoint import geom_sig  # noqa: E402


TEMPLATE_RE = re.compile(r"/aug9_car/(set[12]/benchmark_[1-5])/")
MOVABLE_RE = re.compile(r"^obstacle_.*_movable$")
TIERS = ("easy", "medium", "hard")
HORIZONS = ("1push", "2push")
COMPOSITION_MODES = (
    "fixed_template",
    "same_template",
    "same_template_clutter",
    "room_stitch",
)
PORTAL_SIDES = ("east", "north", "south", "west")
INDEPENDENT_POSITION_TOLERANCE_M = 0.002
INDEPENDENT_ANGLE_TOLERANCE_RAD = math.radians(1.0)
CONTACT_HALF_SIZE_RANGE_M = (0.03, 0.055)
CONTACT_MIN_TARGET_TRANSLATION_M = 0.015


class CompositionRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class Donor:
    xml_path: str
    object_id: str
    region: str
    object_center: tuple[float, float]
    object_theta: float
    tier: str
    horizon: str
    template: str
    valid_root: tuple[tuple[int, int], ...]

    @property
    def episode_key(self) -> tuple[str, str, str]:
        return (os.path.realpath(self.xml_path), self.object_id, self.region)


@dataclass(frozen=True)
class PortalInterface:
    side: str
    center: float
    target_label: str
    free_run_m: float


@dataclass(frozen=True)
class RigidTransform:
    rotation_deg: int
    tx: float
    ty: float


def _rows(path: Path) -> Iterator[tuple[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for raw_xml, episodes in data.items():
        xml = os.path.realpath(str(resolve(raw_xml)))
        for episode in episodes:
            yield xml, episode


def _division_path(horizon: str) -> Path:
    if horizon == "1push":
        return eval_sets.ONEPUSH.parent / "onepush_divisions_v3.json"
    return eval_sets.DIVISIONS


def _manifest_path(horizon: str) -> Path:
    return eval_sets.ONEPUSH if horizon == "1push" else eval_sets.PURE2PUSH


def load_donors(horizon: str, tier: str, template: str) -> list[Donor]:
    divisions = {
        (xml, row["object_id"], row.get("region", "goal")): row["division"]
        for xml, row in _rows(_division_path(horizon))
    }
    donors: list[Donor] = []
    for xml, row in _rows(_manifest_path(horizon)):
        match = TEMPLATE_RE.search(xml)
        if match is None or (template != "any" and match.group(1) != template):
            continue
        region = row.get("region", "goal")
        key = (xml, row["object_id"], region)
        if divisions.get(key) != tier:
            continue
        raw_valid = row.get("valid", ()) if horizon == "1push" else row.get("valid_first_push", ())
        donors.append(
            Donor(
                xml_path=xml,
                object_id=row["object_id"],
                region=region,
                object_center=(float(row["object_center"][0]), float(row["object_center"][1])),
                object_theta=float(row.get("object_theta", 0.0)),
                tier=tier,
                horizon=horizon,
                template=match.group(1),
                valid_root=tuple((int(edge), int(depth)) for edge, depth in raw_valid),
            )
        )
    return donors


def _worldbody(root: ET.Element) -> ET.Element:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("XML has no worldbody")
    return worldbody


def _movable_body(root: ET.Element, object_id: str) -> ET.Element:
    for body in _worldbody(root).findall("body"):
        if body.get("name") == object_id:
            return body
    raise KeyError(f"movable body {object_id!r} not found")


def _renamed_blocker(xml_path: str, object_id: str, new_id: str) -> ET.Element:
    source = ET.parse(xml_path).getroot()
    body = copy.deepcopy(_movable_body(source, object_id))
    body.set("name", new_id)
    for geom in body.findall(".//geom"):
        if geom.get("name") == object_id:
            geom.set("name", new_id)
    return body


def _goal_site(root: ET.Element) -> ET.Element:
    site = root.find(".//site[@name='goal']")
    if site is None:
        raise ValueError("XML has no goal site")
    return site


def compose_xml(donors: Sequence[Donor], output: Path) -> None:
    tree = ET.parse(donors[0].xml_path)
    root = tree.getroot()
    worldbody = _worldbody(root)
    for body in list(worldbody.findall("body")):
        if MOVABLE_RE.match(body.get("name") or ""):
            worldbody.remove(body)

    last_root = ET.parse(donors[-1].xml_path).getroot()
    _goal_site(root).set("pos", _goal_site(last_root).get("pos") or "0 0 0")
    for index, donor in enumerate(donors):
        worldbody.append(
            _renamed_blocker(donor.xml_path, donor.object_id, f"obstacle_{index}_movable")
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def compose_same_template_xml(donors: Sequence[Donor], output: Path) -> dict:
    metadata = _same_template_metadata(donors)
    compose_xml(donors, output)
    return metadata


def _same_template_metadata(donors: Sequence[Donor]) -> dict:
    if len(donors) != 2 or any(donor.horizon != "1push" for donor in donors):
        raise ValueError("same-template composition requires exactly two 1push donors")
    templates = {donor.template for donor in donors}
    if len(templates) != 1:
        raise CompositionRejected("donor_template_mismatch")
    wall_signatures = {geom_sig(donor.xml_path)[1] for donor in donors}
    if None in wall_signatures:
        raise CompositionRejected("donor_wall_signature_failed")
    if len(wall_signatures) != 1:
        raise CompositionRejected("donor_wall_signature_mismatch")
    return {
        "mode": "same_template",
        "template": donors[0].template,
        "wall_signature": next(iter(wall_signatures)),
        "host_xml": os.path.realpath(donors[0].xml_path),
        "transplanted": "blockers_only",
    }


@functools.lru_cache(maxsize=None)
def _movable_object_ids(xml_path: str) -> tuple[str, ...]:
    root = ET.parse(xml_path).getroot()
    return tuple(
        body.get("name") or ""
        for body in _worldbody(root).findall("body")
        if MOVABLE_RE.match(body.get("name") or "")
    )


def host_clutter_ids(donor: Donor) -> tuple[str, ...]:
    return tuple(object_id for object_id in _movable_object_ids(donor.xml_path) if object_id != donor.object_id)


def compose_same_template_clutter_xml(
    donors: Sequence[Donor], host_clutter_id: str, output: Path
) -> dict:
    metadata = _same_template_metadata(donors)
    if host_clutter_id not in host_clutter_ids(donors[0]):
        raise CompositionRejected("host_clutter_not_available")

    compose_xml(donors, output)
    tree = ET.parse(output)
    root = tree.getroot()
    clutter_object_id = _intended_blockers(3)[-1]
    _worldbody(root).append(
        _renamed_blocker(donors[0].xml_path, host_clutter_id, clutter_object_id)
    )
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {
        **metadata,
        "mode": "same_template_clutter",
        "transplanted": "blockers_plus_one_host_clutter",
        "clutter_object_ids": [clutter_object_id],
        "host_clutter_source_id": host_clutter_id,
    }


def _box_half_size(root: ET.Element, object_id: str) -> tuple[float, float, float]:
    geom = _movable_body(root, object_id).find(".//geom[@type='box']")
    if geom is None:
        raise CompositionRejected("contact_target_not_box")
    size = _numbers(geom.get("size"))
    return float(size[0]), float(size[1]), float(size[2])


def _box_support(half_size: Sequence[float], theta: float, direction: Sequence[float]) -> float:
    ux = (math.cos(theta), math.sin(theta))
    uy = (-math.sin(theta), math.cos(theta))
    return abs(direction[0] * ux[0] + direction[1] * ux[1]) * half_size[0] + abs(
        direction[0] * uy[0] + direction[1] * uy[1]
    ) * half_size[1]


def sampled_contact_placement(source_row: dict, hop: int, variant: int, seed: int) -> dict:
    target_id = _intended_blockers(2)[hop]
    poses = source_row["replay"]["object_pose_trace"]
    before = poses[hop][target_id]
    after = poses[hop + 1][target_id]
    dx = float(after[0]) - float(before[0])
    dy = float(after[1]) - float(before[1])
    travel = math.hypot(dx, dy)
    if travel < CONTACT_MIN_TARGET_TRANSLATION_M:
        raise CompositionRejected("contact_target_motion_too_small")
    direction = (dx / travel, dy / travel)
    perpendicular = (-direction[1], direction[0])
    identity = source_row.get("geometry_identity", {}).get("full") or source_row["xml_path"]
    digest = hashlib.sha256(f"{seed}|{identity}|{hop}|{variant}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    half_size = (
        rng.uniform(*CONTACT_HALF_SIZE_RANGE_M),
        rng.uniform(*CONTACT_HALF_SIZE_RANGE_M),
    )
    theta = rng.uniform(-math.pi, math.pi)
    gap = travel * rng.uniform(0.05, 0.80)
    lateral = rng.uniform(-0.30, 0.30) * min(half_size)
    root = ET.parse(source_row["xml_path"]).getroot()
    target_half_size = _box_half_size(root, target_id)
    separation = (
        _box_support(target_half_size, float(before[2]), direction)
        + _box_support(half_size, theta, direction)
        + gap
    )
    center = (
        float(before[0]) + direction[0] * separation + perpendicular[0] * lateral,
        float(before[1]) + direction[1] * separation + perpendicular[1] * lateral,
    )
    return {
        "target_hop": hop + 1,
        "target_object_id": target_id,
        "variant": variant,
        "center": [center[0], center[1]],
        "half_size": [half_size[0], half_size[1]],
        "theta": theta,
        "gap_m": gap,
        "lateral_offset_m": lateral,
        "target_clean_translation_m": travel,
    }


def compose_sampled_contact_xml(source_xml: str, placement: dict, output: Path) -> dict:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    target_id = placement["target_object_id"]
    context_id = _intended_blockers(3)[-1]
    body = copy.deepcopy(_movable_body(root, target_id))
    body.set("name", context_id)
    body.set("pos", "0 0 0")
    geom = body.find(".//geom[@type='box']")
    if geom is None:
        raise CompositionRejected("contact_target_not_box")
    source_size = _numbers(geom.get("size"))
    center = placement["center"]
    half_size = placement["half_size"]
    _set_numbers(geom, "pos", (center[0], center[1], source_size[2]))
    _set_numbers(geom, "size", (half_size[0], half_size[1], source_size[2]))
    _set_numbers(geom, "euler", (0.0, 0.0, math.degrees(placement["theta"])))
    geom.set("name", context_id)
    geom.set("rgba", "0 0.7 1 1")
    _worldbody(root).append(body)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {
        "mode": "same_template_contact",
        "source_xml": os.path.realpath(source_xml),
        "clutter_object_ids": [context_id],
        "contact_placement": placement,
    }


def _box_corners(
    center: Sequence[float], half_size: Sequence[float], theta: float
) -> list[tuple[float, float]]:
    ux = (math.cos(theta), math.sin(theta))
    uy = (-math.sin(theta), math.cos(theta))
    return [
        (
            center[0] + sx * half_size[0] * ux[0] + sy * half_size[1] * uy[0],
            center[1] + sx * half_size[0] * ux[1] + sy * half_size[1] * uy[1],
        )
        for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))
    ]


def _rectangles_overlap(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> bool:
    axes = []
    for corners in (left, right):
        for index in (0, 1):
            edge = (
                corners[index + 1][0] - corners[index][0],
                corners[index + 1][1] - corners[index][1],
            )
            norm = math.hypot(*edge)
            axes.append((-edge[1] / norm, edge[0] / norm))
    for axis in axes:
        left_projection = [point[0] * axis[0] + point[1] * axis[1] for point in left]
        right_projection = [point[0] * axis[0] + point[1] * axis[1] for point in right]
        if max(left_projection) <= min(right_projection) or max(right_projection) <= min(
            left_projection
        ):
            return False
    return True


def _geom_rectangle(geom: ET.Element, padding: float = 0.0) -> list[tuple[float, float]]:
    pos = _numbers(geom.get("pos"))
    size = _numbers(geom.get("size"))
    euler = _numbers(geom.get("euler")) or [0.0, 0.0, 0.0]
    return _box_corners(
        pos[:2],
        (size[0] + padding, size[1] + padding),
        math.radians(euler[2]),
    )


def sampled_contact_geometry_failure(xml_path: str) -> str | None:
    root = ET.parse(xml_path).getroot()
    context_id = _intended_blockers(3)[-1]
    context_geom = _movable_body(root, context_id).find(".//geom[@type='box']")
    if context_geom is None:
        return "contact_target_not_box"
    context_corners = _geom_rectangle(context_geom, padding=0.001)

    west = _numbers(_outer_wall(root, "west").get("pos"))[0] + _numbers(
        _outer_wall(root, "west").get("size")
    )[0]
    east = _numbers(_outer_wall(root, "east").get("pos"))[0] - _numbers(
        _outer_wall(root, "east").get("size")
    )[0]
    south = _numbers(_outer_wall(root, "south").get("pos"))[1] + _numbers(
        _outer_wall(root, "south").get("size")
    )[1]
    north = _numbers(_outer_wall(root, "north").get("pos"))[1] - _numbers(
        _outer_wall(root, "north").get("size")
    )[1]
    if any(not (west <= x <= east and south <= y <= north) for x, y in context_corners):
        return "contact_outside_room"
    if any(_rectangles_overlap(context_corners, _geom_rectangle(geom)) for geom in _wall_geoms(root)):
        return "contact_overlaps_wall"
    for body in _worldbody(root).findall("body"):
        object_id = body.get("name") or ""
        if not MOVABLE_RE.match(object_id) or object_id == context_id:
            continue
        geom = body.find(".//geom[@type='box']")
        if geom is not None and _rectangles_overlap(context_corners, _geom_rectangle(geom)):
            return "contact_overlaps_blocker"
    return None


def _numbers(value: str | None) -> list[float]:
    return [float(item) for item in (value or "").split()]


def _set_numbers(element: ET.Element, key: str, values: Sequence[float]) -> None:
    element.set(key, " ".join(f"{float(value):.9f}" for value in values))


def _wall_body(root: ET.Element) -> ET.Element:
    body = _worldbody(root).find("body[@name='walls']")
    if body is None:
        raise CompositionRejected("module_has_no_wall_body")
    return body


def _wall_geoms(root: ET.Element) -> list[ET.Element]:
    geoms = [geom for geom in _wall_body(root).findall("geom") if geom.get("type") == "box"]
    if not geoms:
        raise CompositionRejected("module_has_no_wall_geometries")
    return geoms


def _wall_bounds(root: ET.Element) -> tuple[float, float, float, float]:
    boxes = []
    for geom in _wall_geoms(root):
        pos = _numbers(geom.get("pos"))
        size = _numbers(geom.get("size"))
        boxes.append((pos[0] - size[0], pos[0] + size[0], pos[1] - size[1], pos[1] + size[1]))
    return (
        min(box[0] for box in boxes),
        max(box[1] for box in boxes),
        min(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _outer_wall(root: ET.Element, side: str) -> ET.Element:
    xmin, xmax, ymin, ymax = _wall_bounds(root)
    ranked = []
    for geom in _wall_geoms(root):
        pos = _numbers(geom.get("pos"))
        size = _numbers(geom.get("size"))
        if side == "west":
            error, length = abs((pos[0] - size[0]) - xmin), 2.0 * size[1]
        elif side == "east":
            error, length = abs((pos[0] + size[0]) - xmax), 2.0 * size[1]
        elif side == "south":
            error, length = abs((pos[1] - size[1]) - ymin), 2.0 * size[0]
        elif side == "north":
            error, length = abs((pos[1] + size[1]) - ymax), 2.0 * size[0]
        else:
            raise ValueError(f"unknown portal side {side!r}")
        ranked.append((error, -length, geom))
    error, _negative_length, wall = min(ranked, key=lambda row: (row[0], row[1], row[2].get("name") or ""))
    if error > 1e-6:
        raise CompositionRejected("outer_wall_not_found")
    return wall


def _wall_span(root: ET.Element, side: str) -> tuple[float, float]:
    wall = _outer_wall(root, side)
    pos = _numbers(wall.get("pos"))
    size = _numbers(wall.get("size"))
    if side in {"west", "east"}:
        return pos[1] - size[1], pos[1] + size[1]
    return pos[0] - size[0], pos[0] + size[0]


def _portal_point(root: ET.Element, interface: PortalInterface) -> tuple[float, float]:
    wall = _outer_wall(root, interface.side)
    pos = _numbers(wall.get("pos"))
    if interface.side in {"west", "east"}:
        return pos[0], interface.center
    return interface.center, pos[1]


def _split_outer_wall(root: ET.Element, interface: PortalInterface, width: float) -> None:
    wall_body = _wall_body(root)
    wall = _outer_wall(root, interface.side)
    pos = _numbers(wall.get("pos"))
    size = _numbers(wall.get("size"))
    axis = 1 if interface.side in {"west", "east"} else 0
    lower = pos[axis] - size[axis]
    upper = pos[axis] + size[axis]
    gap_lower = interface.center - 0.5 * width
    gap_upper = interface.center + 0.5 * width
    if gap_lower <= lower or gap_upper >= upper:
        raise CompositionRejected("portal_does_not_fit_outer_wall")

    segments = ((lower, gap_lower), (gap_upper, upper))
    original_name = wall.get("name") or "outer_wall"
    insertion = list(wall_body).index(wall)
    wall_body.remove(wall)
    for index, (start, end) in enumerate(segments):
        segment = copy.deepcopy(wall)
        segment_pos = list(pos)
        segment_size = list(size)
        segment_pos[axis] = 0.5 * (start + end)
        segment_size[axis] = 0.5 * (end - start)
        segment.set("name", f"{original_name}_portal_{index}")
        _set_numbers(segment, "pos", segment_pos)
        _set_numbers(segment, "size", segment_size)
        wall_body.insert(insertion + index, segment)


def _side_angle_deg(side: str) -> int:
    return {"east": 0, "north": 90, "west": 180, "south": 270}[side]


def _rotation_between_sides(source: str, target: str) -> int:
    return (_side_angle_deg(target) - _side_angle_deg(source)) % 360


def _rotate_xy(x: float, y: float, rotation_deg: float) -> tuple[float, float]:
    theta = math.radians(rotation_deg)
    cosine, sine = math.cos(theta), math.sin(theta)
    return cosine * x - sine * y, sine * x + cosine * y


def _transform_xy(x: float, y: float, transform: RigidTransform) -> tuple[float, float]:
    rotated_x, rotated_y = _rotate_xy(x, y, transform.rotation_deg)
    return rotated_x + transform.tx, rotated_y + transform.ty


def _add_yaw_deg(element: ET.Element, rotation_deg: float) -> None:
    euler = _numbers(element.get("euler"))
    if not euler:
        euler = [0.0, 0.0, 0.0]
    if len(euler) != 3:
        raise CompositionRejected("unsupported_euler_shape")
    euler[2] += float(rotation_deg)
    _set_numbers(element, "euler", euler)


def _transform_position(element: ET.Element, transform: RigidTransform) -> None:
    pos = _numbers(element.get("pos"))
    if len(pos) < 2:
        raise CompositionRejected("module_element_missing_xy_position")
    pos[0], pos[1] = _transform_xy(pos[0], pos[1], transform)
    _set_numbers(element, "pos", pos)


def _transform_module(
    root: ET.Element,
    blocker_name: str,
    transform: RigidTransform,
    *,
    transform_robot: bool,
) -> None:
    for wall in _wall_geoms(root):
        _transform_position(wall, transform)
        _add_yaw_deg(wall, transform.rotation_deg)

    blocker = _movable_body(root, blocker_name)
    blocker_geom = blocker.find(f".//geom[@name='{blocker_name}']")
    if blocker_geom is None:
        raise CompositionRejected("module_blocker_geom_missing")
    _transform_position(blocker_geom, transform)
    _add_yaw_deg(blocker_geom, transform.rotation_deg)

    if transform_robot:
        robot = _worldbody(root).find("body[@name='car']")
        if robot is None:
            raise CompositionRejected("module_robot_missing")
        _transform_position(robot, transform)
        _add_yaw_deg(robot, transform.rotation_deg)

    goal = _goal_site(root)
    _transform_position(goal, transform)


def _module_root(donor: Donor, blocker_name: str) -> ET.Element:
    root = ET.parse(donor.xml_path).getroot()
    worldbody = _worldbody(root)
    for body in list(worldbody.findall("body")):
        if MOVABLE_RE.match(body.get("name") or ""):
            worldbody.remove(body)
    worldbody.append(_renamed_blocker(donor.xml_path, donor.object_id, blocker_name))
    return root


def _contiguous_runs(values: np.ndarray, resolution: float) -> list[tuple[float, float]]:
    if len(values) == 0:
        return []
    ordered = np.unique(np.round(values.astype(float), 6))
    runs = []
    start = previous = float(ordered[0])
    for value in ordered[1:]:
        value = float(value)
        if value - previous > 1.5 * resolution:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


@functools.lru_cache(maxsize=None)
def _module_interface(
    donor: Donor,
    role: str,
    config: str,
    portal_width: float,
) -> PortalInterface:
    with tempfile.TemporaryDirectory(prefix="keyhole_module_interface_") as temp_dir:
        single_xml = Path(temp_dir) / "single.xml"
        compose_xml([donor], single_xml)
        env = namo_rl.RLEnvironment(str(single_xml), config, False)
        graph = get_region_snapshot(
            env,
            goals_per_region=0,
            local_info_only=False,
            seed=42,
            use_cpp_unified=True,
            use_xml_goal=True,
        )
        robot_label = graph.get("robot_label")
        goal_label = graph.get("goal_label")
        if not robot_label or not goal_label:
            raise CompositionRejected(f"module_{role}_region_missing")
        if robot_label == goal_label:
            raise CompositionRejected("module_not_keyhole_after_stripping")
        target_label = goal_label if role == "exit" else robot_label

        exporter = WavefrontSnapshotExporter(env, resolution=0.01)
        raster = exporter.build_snapshot(
            xml_path=str(single_xml),
            config_path=config,
            goal_radius=None,
            goals_per_region=0,
            rng=np.random.default_rng(0),
        )
        target_ids = [int(region_id) for region_id, label in raster.region_labels.items() if label == target_label]
        if not target_ids:
            raise CompositionRejected(f"module_{role}_raster_region_missing")
        cells = np.argwhere(np.isin(raster.region_map, target_ids))
        if len(cells) == 0:
            raise CompositionRejected(f"module_{role}_raster_region_empty")

        xmin, xmax, ymin, ymax = raster.bounds
        width, height = raster.region_map.shape
        world_x = xmin + (cells[:, 0] + 0.5) / width * (xmax - xmin)
        world_y = ymin + (cells[:, 1] + 0.5) / height * (ymax - ymin)
        resolution = float(raster.resolution)
        inflation = max(abs(float(value)) for value in raster.robot_half_extent) + max(
            0.0, float(raster.tier1_inflation_margin_m)
        )
        distances = {
            "west": world_x - xmin,
            "east": xmax - world_x,
            "south": world_y - ymin,
            "north": ymax - world_y,
        }
        orthogonal = {
            "west": world_y,
            "east": world_y,
            "south": world_x,
            "north": world_x,
        }
        single_root = ET.parse(single_xml).getroot()
        candidates = []
        for side in PORTAL_SIDES:
            side_distances = distances[side]
            minimum = float(side_distances.min())
            if minimum > 2.0 * inflation + resolution:
                continue
            band = orthogonal[side][side_distances <= minimum + 1.5 * resolution]
            wall_lower, wall_upper = _wall_span(single_root, side)
            for run_lower, run_upper in _contiguous_runs(band, resolution):
                physical_lower = run_lower - inflation
                physical_upper = run_upper + inflation
                feasible_lower = max(physical_lower + 0.5 * portal_width, wall_lower + 0.5 * portal_width)
                feasible_upper = min(physical_upper - 0.5 * portal_width, wall_upper - 0.5 * portal_width)
                if feasible_lower > feasible_upper:
                    continue
                raw_center = 0.5 * (run_lower + run_upper)
                center = min(max(raw_center, feasible_lower), feasible_upper)
                free_run = run_upper - run_lower + resolution
                candidates.append((free_run, side, center))
        if not candidates:
            raise CompositionRejected(f"module_{role}_has_no_outer_portal")
        free_run, side, center = min(
            candidates,
            key=lambda row: (-row[0], PORTAL_SIDES.index(row[1]), row[2]),
        )
        return PortalInterface(
            side=side,
            center=round(float(center), 6),
            target_label=str(target_label),
            free_run_m=round(float(free_run), 6),
        )


def _prefix_wall_names(root: ET.Element, prefix: str) -> ET.Element:
    wall_body = _wall_body(root)
    wall_body.set("name", f"{prefix}walls")
    for geom in wall_body.findall("geom"):
        geom.set("name", f"{prefix}{geom.get('name') or 'wall'}")
    return wall_body


def _add_connector_walls(
    root: ET.Element,
    left_portal: tuple[float, float],
    right_portal: tuple[float, float],
    portal_width: float,
) -> None:
    if abs(left_portal[1] - right_portal[1]) > 1e-6 or right_portal[0] <= left_portal[0]:
        raise CompositionRejected("connector_portals_not_horizontal")
    half_length = 0.5 * (right_portal[0] - left_portal[0])
    center_x = 0.5 * (left_portal[0] + right_portal[0])
    half_thickness = 0.01
    wall_body = ET.Element("body", {"name": "connector_walls"})
    for index, sign in enumerate((-1.0, 1.0)):
        center_y = left_portal[1] + sign * (0.5 * portal_width + half_thickness)
        geom = ET.SubElement(
            wall_body,
            "geom",
            {
                "name": f"connector_wall_{index}",
                "condim": "4",
                "type": "box",
                "rgba": "0.8 0.8 0.8 1",
            },
        )
        _set_numbers(geom, "pos", (center_x, center_y, 0.08))
        _set_numbers(geom, "size", (half_length, half_thickness, 0.08))
    _worldbody(root).append(wall_body)


def _geom_xy_bounds(geom: ET.Element) -> tuple[float, float, float, float]:
    pos = _numbers(geom.get("pos"))
    size = _numbers(geom.get("size"))
    euler = _numbers(geom.get("euler"))
    yaw = math.radians(euler[2] if len(euler) == 3 else 0.0)
    half_x = abs(math.cos(yaw)) * size[0] + abs(math.sin(yaw)) * size[1]
    half_y = abs(math.sin(yaw)) * size[0] + abs(math.cos(yaw)) * size[1]
    return pos[0] - half_x, pos[0] + half_x, pos[1] - half_y, pos[1] + half_y


def _add_global_boundary_walls(root: ET.Element) -> None:
    wall_geoms = [
        geom
        for body in _worldbody(root).findall("body")
        if "wall" in (body.get("name") or "")
        for geom in body.findall("geom")
        if geom.get("type") == "box"
    ]
    if not wall_geoms:
        raise CompositionRejected("assembled_scene_has_no_walls")
    bounds = [_geom_xy_bounds(geom) for geom in wall_geoms]
    xmin = min(bound[0] for bound in bounds)
    xmax = max(bound[1] for bound in bounds)
    ymin = min(bound[2] for bound in bounds)
    ymax = max(bound[3] for bound in bounds)
    half_thickness = 0.01
    half_width = 0.5 * (xmax - xmin) + half_thickness
    half_height = 0.5 * (ymax - ymin) + half_thickness
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    wall_body = ET.Element("body", {"name": "global_boundary_walls"})
    specs = (
        ("wall_1", (xmin - half_thickness, center_y, 0.08), (half_thickness, half_height, 0.08)),
        ("wall_2", (xmax + half_thickness, center_y, 0.08), (half_thickness, half_height, 0.08)),
        ("wall_3", (center_x, ymin - half_thickness, 0.08), (half_width, half_thickness, 0.08)),
        ("wall_4", (center_x, ymax + half_thickness, 0.08), (half_width, half_thickness, 0.08)),
    )
    for name, pos, size in specs:
        geom = ET.SubElement(
            wall_body,
            "geom",
            {
                "name": name,
                "condim": "4",
                "type": "box",
                "rgba": "0.8 0.8 0.8 1",
            },
        )
        _set_numbers(geom, "pos", pos)
        _set_numbers(geom, "size", size)
    _worldbody(root).append(wall_body)


def compose_room_stitch_xml(
    donors: Sequence[Donor],
    output: Path,
    config: str,
    *,
    portal_width: float,
    connector_length: float,
) -> dict:
    if len(donors) != 2 or any(donor.horizon != "1push" for donor in donors):
        raise ValueError("room stitch requires exactly two 1push donors")

    interfaces = (
        _module_interface(donors[0], "exit", config, portal_width),
        _module_interface(donors[1], "entry", config, portal_width),
    )
    roots = (
        _module_root(donors[0], "obstacle_0_movable"),
        _module_root(donors[1], "obstacle_1_movable"),
    )
    for root, interface in zip(roots, interfaces):
        _split_outer_wall(root, interface, portal_width)

    first_rotation = _rotation_between_sides(interfaces[0].side, "east")
    second_rotation = _rotation_between_sides(interfaces[1].side, "west")
    first_transform = RigidTransform(first_rotation, 0.0, 0.0)
    first_portal_local = _portal_point(roots[0], interfaces[0])
    second_portal_local = _portal_point(roots[1], interfaces[1])
    first_portal = _transform_xy(*first_portal_local, first_transform)
    second_portal_rotated = _rotate_xy(*second_portal_local, second_rotation)
    second_transform = RigidTransform(
        second_rotation,
        first_portal[0] + connector_length - second_portal_rotated[0],
        first_portal[1] - second_portal_rotated[1],
    )
    second_portal = _transform_xy(*second_portal_local, second_transform)

    _transform_module(
        roots[0],
        "obstacle_0_movable",
        first_transform,
        transform_robot=True,
    )
    _transform_module(
        roots[1],
        "obstacle_1_movable",
        second_transform,
        transform_robot=False,
    )
    _prefix_wall_names(roots[0], "module_1_")
    second_wall_body = _prefix_wall_names(roots[1], "module_2_")

    output_root = roots[0]
    output_worldbody = _worldbody(output_root)
    output_worldbody.append(copy.deepcopy(second_wall_body))
    output_worldbody.append(copy.deepcopy(_movable_body(roots[1], "obstacle_1_movable")))
    _goal_site(output_root).set("pos", _goal_site(roots[1]).get("pos") or "0 0 0")
    _add_connector_walls(output_root, first_portal, second_portal, portal_width)
    _add_global_boundary_walls(output_root)
    output_root.set("model", "stitched_two_keyhole_environment")

    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(output_root)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {
        "mode": "room_stitch",
        "portal_width_m": float(portal_width),
        "connector_length_m": float(connector_length),
        "modules": [
            {
                "role": role,
                "source_template": donor.template,
                "interface": asdict(interface),
                "transform": asdict(transform),
            }
            for role, donor, interface, transform in zip(
                ("k1", "k2"), donors, interfaces, (first_transform, second_transform)
            )
        ],
    }


def _intended_blockers(hops: int) -> list[str]:
    return [f"obstacle_{index}_movable" for index in range(hops)]


def static_acceptance(row: dict, hops: int) -> tuple[bool, str]:
    if row.get("error"):
        return False, "static_error"
    if row.get("goal_in_free_space") is False:
        return False, "goal_not_in_free_space"
    if row.get("no_path"):
        return False, "no_component_path"
    if row.get("hop_mismatch"):
        return False, "wrong_hop_count"
    if row.get("no_blocking_objects"):
        return False, "k1_boundary_has_no_blocker"
    if row.get("no_reachable_blocker"):
        return False, "k1_not_reachable"
    if row.get("no_pushable_blocker"):
        return False, "k1_no_push_edges"
    boundaries = row.get("boundaries") or []
    if len(boundaries) != hops:
        return False, "wrong_boundary_count"
    actual = [boundary.get("objects") or [] for boundary in boundaries]
    expected = [[name] for name in _intended_blockers(hops)]
    if actual != expected:
        return False, "wrong_blocker_order"
    return True, "accepted"


def _current_path(env: namo_rl.RLEnvironment) -> tuple[list[str] | None, list[list[str]]]:
    snapshot = get_region_snapshot(
        env,
        goals_per_region=0,
        local_info_only=False,
        seed=42,
        use_cpp_unified=True,
        use_xml_goal=True,
    )
    path = shortest_region_path(
        snapshot["adjacency"], snapshot.get("robot_label") or "", snapshot.get("goal_label") or ""
    )
    boundaries: list[list[str]] = []
    for source, target in zip(path or [], (path or [])[1:]):
        forward = snapshot["edge_objects"].get(source, {}).get(target)
        reverse = snapshot["edge_objects"].get(target, {}).get(source)
        boundaries.append(sorted(set(forward if forward is not None else reverse or [])))
    return path, boundaries


def _action(object_id: str, edge: int, depth: int) -> namo_rl.Action:
    action = namo_rl.Action()
    action.object_id = object_id
    action.edge_idx = int(edge)
    action.depth = int(depth)
    action.x = action.y = action.theta = 0.0
    return action


def _intended_reachability_state(
    env: namo_rl.RLEnvironment, object_ids: Sequence[str]
) -> dict:
    reachable = set(env.get_reachable_objects())
    return {
        "goal_reachable": bool(env.is_robot_goal_reachable()),
        "reachable_objects": sorted(reachable),
        "reachable_edges": {
            object_id: sorted(int(edge) for edge in env.get_reachable_edges(object_id))
            if object_id in reachable
            else []
            for object_id in object_ids
        },
    }


def _initial_two_keyhole_failure(state: dict) -> str | None:
    if state["goal_reachable"]:
        return "goal_reachable_at_t0"
    if "obstacle_0_movable" not in state["reachable_objects"]:
        return "k1_not_reachable"
    if not state["reachable_edges"]["obstacle_0_movable"]:
        return "k1_no_push_edges"
    if "obstacle_1_movable" in state["reachable_objects"]:
        return "k2_reachable_at_t0"
    return None


def _post_k1_two_keyhole_failure(push_done: bool, state: dict) -> str | None:
    if not push_done:
        return "k1_push_failed"
    if state["goal_reachable"]:
        return "k1_reached_goal"
    if "obstacle_1_movable" not in state["reachable_objects"]:
        return "k1_did_not_expose_k2"
    if not state["reachable_edges"]["obstacle_1_movable"]:
        return "k2_no_push_edges_after_k1"
    return None


def _post_k2_two_keyhole_failure(push_done: bool, state: dict) -> str | None:
    if not push_done:
        return "k2_push_failed"
    if not state["goal_reachable"]:
        return "final_goal_unreachable"
    return None


def _target_point_counts(
    env: namo_rl.RLEnvironment, target_points: Sequence[Sequence[tuple[float, float]]]
) -> list[int]:
    return [int(env.count_reachable_points(points)[0]) for points in target_points]


def _intended_object_poses(env: namo_rl.RLEnvironment, object_ids: Sequence[str]) -> dict:
    observation = env.get_observation()
    poses = {}
    for object_id in object_ids:
        pose = observation.get(f"{object_id}_pose")
        poses[object_id] = (
            [float(pose[0]), float(pose[1]), float(pose[2])]
            if pose is not None
            else None
        )
    return poses


def _path_boundaries(snapshot: dict, path: Sequence[str]) -> list[list[str]]:
    boundaries = []
    for source, target in zip(path, path[1:]):
        forward = snapshot["edge_objects"].get(source, {}).get(target)
        reverse = snapshot["edge_objects"].get(target, {}).get(source)
        boundaries.append(sorted(set(forward if forward is not None else reverse or [])))
    return boundaries


def _failure_example(
    actions: list[list[list[int]]],
    states: Sequence[dict],
    point_counts: Sequence[Sequence[int]],
    poses: Sequence[dict],
    action_infos: Sequence[dict],
) -> dict:
    return {
        "actions": actions,
        "reachability_trace": list(states),
        "target_point_trace": [list(counts) for counts in point_counts],
        "object_pose_trace": list(poses),
        "action_info_trace": list(action_infos),
    }


def _action_info(result) -> dict:
    info = getattr(result, "info", {})
    return {
        key: str(info.get(key, ""))
        for key in ("movable_collisions", "collision_object", "wall_collision")
    }


def replay_two_keyhole_goal_chain(
    xml_path: str,
    config: str,
    donors: Sequence[Donor],
    *,
    max_attempts: int | None = None,
    monitored_object_ids: Sequence[str] | None = None,
) -> dict:
    """Find known donor actions that expose K2 and then make the XML goal reachable.

    Pinned component-point counts remain in the returned trace for diagnosis. They do not accept or
    reject a complete two-keyhole scene.
    """
    if len(donors) != 2 or any(donor.horizon != "1push" for donor in donors):
        raise ValueError("goal-centric replay requires exactly two 1push donors")

    env = namo_rl.RLEnvironment(xml_path, config, False)
    env.set_robot_goal(*extract_goal_from_xml(xml_path))
    initial = env.get_full_state()
    object_ids = _intended_blockers(2)
    pose_object_ids = list(monitored_object_ids or object_ids)
    snapshot = get_region_snapshot(
        env,
        goals_per_region=100,
        local_info_only=False,
        seed=42,
        use_cpp_unified=True,
        use_xml_goal=True,
    )
    path = shortest_region_path(
        snapshot["adjacency"],
        snapshot.get("robot_label") or "",
        snapshot.get("goal_label") or "",
    )
    base = {
        "component_path": path,
        "boundary_objects": _path_boundaries(snapshot, path or []),
        "goal_in_free_space": bool(snapshot.get("goal_in_free_space", False)),
    }
    if not base["goal_in_free_space"]:
        return {
            **base,
            "status": "goal_not_in_free_space",
            "failure_reason": "goal_not_in_free_space",
            "attempts": 0,
            "actions": None,
        }
    if path is None or len(path) != 3:
        return {
            **base,
            "status": "initial_hop_mismatch",
            "failure_reason": "initial_hop_mismatch",
            "attempts": 0,
            "actions": None,
        }
    expected_boundaries = [[object_id] for object_id in object_ids]
    if base["boundary_objects"] != expected_boundaries:
        return {
            **base,
            "status": "initial_blocker_order_mismatch",
            "failure_reason": "initial_blocker_order_mismatch",
            "attempts": 0,
            "actions": None,
        }

    target_points = [
        [(goal.x, goal.y) for goal in snapshot["region_goals"][label].goals]
        for label in path[1:]
    ]
    thresholds = [
        max(1, math.ceil(CANONICAL_MIN_REACHABLE_FRACTION * len(points)))
        for points in target_points
    ]
    initial_state = _intended_reachability_state(env, object_ids)
    initial_failure = _initial_two_keyhole_failure(initial_state)
    initial_counts = _target_point_counts(env, target_points)
    initial_poses = _intended_object_poses(env, pose_object_ids)
    if initial_failure is not None:
        return {
            **base,
            "status": initial_failure,
            "failure_reason": initial_failure,
            "attempts": 0,
            "actions": None,
            "reachability_trace": [initial_state],
            "target_point_trace": [initial_counts],
            "target_point_thresholds": thresholds,
            "object_pose_trace": [initial_poses],
        }

    attempts = 0
    candidate_rejections: Counter[str] = Counter()
    failure_examples: dict[str, dict] = {}
    post_k1_candidates = 0
    attempt_cap_reached = False
    for k1_edge, k1_depth in donors[0].valid_root:
        if max_attempts is not None and attempts >= max_attempts:
            attempt_cap_reached = True
            break
        env.set_full_state(initial)
        attempts += 1
        k1_result = env.step(_action(object_ids[0], k1_edge, k1_depth))
        k1_info = _action_info(k1_result)
        post_k1_state = _intended_reachability_state(env, object_ids)
        post_k1_counts = _target_point_counts(env, target_points)
        post_k1_poses = _intended_object_poses(env, pose_object_ids)
        k1_failure = _post_k1_two_keyhole_failure(bool(k1_result.done), post_k1_state)
        if k1_failure is not None:
            candidate_rejections[k1_failure] += 1
            failure_examples.setdefault(
                k1_failure,
                _failure_example(
                    [[[int(k1_edge), int(k1_depth)]]],
                    [initial_state, post_k1_state],
                    [initial_counts, post_k1_counts],
                    [initial_poses, post_k1_poses],
                    [k1_info],
                ),
            )
            continue
        post_k1_candidates += 1
        post_k1_full_state = env.get_full_state()

        for k2_edge, k2_depth in donors[1].valid_root:
            if max_attempts is not None and attempts >= max_attempts:
                attempt_cap_reached = True
                break
            env.set_full_state(post_k1_full_state)
            attempts += 1
            k2_result = env.step(_action(object_ids[1], k2_edge, k2_depth))
            k2_info = _action_info(k2_result)
            post_k2_state = _intended_reachability_state(env, object_ids)
            post_k2_counts = _target_point_counts(env, target_points)
            post_k2_poses = _intended_object_poses(env, pose_object_ids)
            k2_failure = _post_k2_two_keyhole_failure(bool(k2_result.done), post_k2_state)
            if k2_failure is not None:
                candidate_rejections[k2_failure] += 1
                failure_examples.setdefault(
                    k2_failure,
                    _failure_example(
                        [
                            [[int(k1_edge), int(k1_depth)]],
                            [[int(k2_edge), int(k2_depth)]],
                        ],
                        [initial_state, post_k1_state, post_k2_state],
                        [initial_counts, post_k1_counts, post_k2_counts],
                        [initial_poses, post_k1_poses, post_k2_poses],
                        [k1_info, k2_info],
                    ),
                )
                continue
            return {
                **base,
                "status": "solved",
                "failure_reason": None,
                "attempts": attempts,
                "actions": [[[int(k1_edge), int(k1_depth)]], [[int(k2_edge), int(k2_depth)]]],
                "reachability_trace": [initial_state, post_k1_state, post_k2_state],
                "target_point_trace": [
                    initial_counts,
                    post_k1_counts,
                    post_k2_counts,
                ],
                "target_point_thresholds": thresholds,
                "object_pose_trace": [
                    initial_poses,
                    post_k1_poses,
                    post_k2_poses,
                ],
                "action_info_trace": [k1_info, k2_info],
                "final_object_poses": post_k2_poses,
                "candidate_rejections": dict(sorted(candidate_rejections.items())),
            }
        if attempt_cap_reached:
            break

    if attempt_cap_reached:
        failure_reason = "replay_attempt_cap"
    elif not donors[0].valid_root:
        failure_reason = "k1_no_known_valid_actions"
    elif post_k1_candidates and not donors[1].valid_root:
        failure_reason = "k2_no_known_valid_actions"
    elif post_k1_candidates:
        eligible = {
            reason: count
            for reason, count in candidate_rejections.items()
            if reason in {"k2_push_failed", "final_goal_unreachable"}
        }
        failure_reason = (
            sorted(eligible, key=lambda reason: (-eligible[reason], reason))[0]
            if eligible
            else "no_goal_chain"
        )
    else:
        eligible = dict(candidate_rejections)
        failure_reason = (
            sorted(eligible, key=lambda reason: (-eligible[reason], reason))[0]
            if eligible
            else "no_goal_chain"
        )
    result = {
        **base,
        "status": "no_goal_chain",
        "failure_reason": failure_reason,
        "attempts": attempts,
        "actions": None,
        "reachability_trace": [initial_state],
        "target_point_trace": [initial_counts],
        "target_point_thresholds": thresholds,
        "object_pose_trace": [initial_poses],
        "candidate_rejections": dict(sorted(candidate_rejections.items())),
    }
    if failure_reason in failure_examples:
        result.update(failure_examples[failure_reason])
        result["target_point_thresholds"] = thresholds
        result["final_object_poses"] = result["object_pose_trace"][-1]
    return result


def replay_component_chain(xml_path: str, config: str, donors: Sequence[Donor]) -> dict:
    env = namo_rl.RLEnvironment(xml_path, config, False)
    initial = env.get_full_state()
    attempts = 0
    snapshot = get_region_snapshot(
        env,
        goals_per_region=100,
        local_info_only=False,
        seed=42,
        use_cpp_unified=True,
        use_xml_goal=True,
    )
    path = shortest_region_path(
        snapshot["adjacency"],
        snapshot.get("robot_label") or "",
        snapshot.get("goal_label") or "",
    )
    if path is None or len(path) - 1 != len(donors):
        return {"status": "initial_hop_mismatch", "attempts": 0, "actions": None}
    target_points = [
        [(goal.x, goal.y) for goal in snapshot["region_goals"][label].goals]
        for label in path[1:]
    ]
    thresholds = [
        max(1, math.ceil(CANONICAL_MIN_REACHABLE_FRACTION * len(points)))
        for points in target_points
    ]

    def state_matches(start_hop: int) -> bool:
        counts = [env.count_reachable_points(points)[0] for points in target_points]
        return all(
            count >= threshold if index < start_hop else count < threshold
            for index, (count, threshold) in enumerate(zip(counts, thresholds))
        )

    def advance(
        hop: int, state, prefix: list[list[list[int]]], candidates: Iterable[tuple[int, int]]
    ) -> list[list[list[int]]] | None:
        nonlocal attempts
        object_id = f"obstacle_{hop}_movable"
        for edge, depth in candidates:
            env.set_full_state(state)
            attempts += 1
            result = env.step(_action(object_id, edge, depth))
            if not result.done or not state_matches(hop + 1):
                continue
            solved = search(hop + 1, env.get_full_state(), prefix + [[[edge, depth]]])
            if solved is not None:
                return solved
        return None

    def search(hop: int, state, actions: list[list[list[int]]]) -> list[list[list[int]]] | None:
        nonlocal attempts
        if hop == len(donors):
            return actions if state_matches(hop) else None
        donor = donors[hop]
        if donor.horizon == "1push":
            return advance(hop, state, actions, donor.valid_root)

        object_id = f"obstacle_{hop}_movable"
        for setup_edge, setup_depth in donor.valid_root:
            env.set_full_state(state)
            attempts += 1
            result = env.step(_action(object_id, setup_edge, setup_depth))
            if not result.done or not state_matches(hop):
                continue
            setup_state = env.get_full_state()
            finish_actions = itertools.product(
                sorted(int(edge) for edge in env.get_reachable_edges(object_id)),
                range(CANONICAL_NUM_DEPTHS),
            )
            for finish_edge, finish_depth in finish_actions:
                env.set_full_state(setup_state)
                attempts += 1
                finish = env.step(_action(object_id, finish_edge, finish_depth))
                if not finish.done or not state_matches(hop + 1):
                    continue
                solved = search(
                    hop + 1,
                    env.get_full_state(),
                    actions + [[[setup_edge, setup_depth], [finish_edge, finish_depth]]],
                )
                if solved is not None:
                    return solved
        return None

    solution = search(0, initial, [])
    final_counts = [env.count_reachable_points(points)[0] for points in target_points]
    return {
        "status": "solved" if solution is not None else "no_donor_action_chain",
        "attempts": attempts,
        "actions": solution,
        "target_point_counts": final_counts,
        "target_point_thresholds": thresholds,
    }


def replay_donor_chain(
    xml_path: str,
    config: str,
    donors: Sequence[Donor],
    *,
    max_attempts: int | None = None,
    monitored_object_ids: Sequence[str] | None = None,
) -> dict:
    if len(donors) == 2 and all(donor.horizon == "1push" for donor in donors):
        return replay_two_keyhole_goal_chain(
            xml_path,
            config,
            donors,
            max_attempts=max_attempts,
            monitored_object_ids=monitored_object_ids,
        )
    return replay_component_chain(xml_path, config, donors)


def _pose_changed(before: Sequence[float] | None, after: Sequence[float] | None) -> bool:
    if before is None or after is None:
        return True
    position_delta = math.dist(before[:2], after[:2])
    angle_delta = abs(math.atan2(math.sin(after[2] - before[2]), math.cos(after[2] - before[2])))
    return (
        position_delta > INDEPENDENT_POSITION_TOLERANCE_M
        or angle_delta > INDEPENDENT_ANGLE_TOLERANCE_RAD
    )


def mechanical_independence_failure(replay: dict) -> str | None:
    poses = replay.get("object_pose_trace") or []
    if len(poses) != 3:
        return "independence_trace_missing"
    k1, k2 = _intended_blockers(2)
    if _pose_changed(poses[0].get(k2), poses[1].get(k2)):
        return "k1_moved_k2"
    if _pose_changed(poses[1].get(k1), poses[2].get(k1)):
        return "k2_moved_k1"
    return None


def passive_clutter_motion_failure(replay: dict, clutter_object_ids: Sequence[str]) -> str | None:
    poses = replay.get("object_pose_trace") or []
    if len(poses) != 3:
        return "passive_clutter_trace_missing"
    for object_id in clutter_object_ids:
        initial = poses[0].get(object_id)
        if any(_pose_changed(initial, state.get(object_id)) for state in poses[1:]):
            return "passive_clutter_moved"
    return None


def clean_counterfactual_decision_edges(
    xml_path: str,
    config: str,
    replay_actions: Sequence[Sequence[Sequence[int]]],
) -> dict:
    """Replay the accepted K1 action without clutter and record ranker decision edges."""
    if len(replay_actions) != 2 or any(len(actions) != 1 for actions in replay_actions):
        return {"status": "clean_counterfactual_action_shape", "decision_edges": None}
    k1, k2 = _intended_blockers(2)
    env = namo_rl.RLEnvironment(xml_path, config, False)
    env.set_robot_goal(*extract_goal_from_xml(xml_path))
    initial = _intended_reachability_state(env, (k1, k2))
    edge, depth = replay_actions[0][0]
    result = env.step(_action(k1, edge, depth))
    post_k1 = _intended_reachability_state(env, (k1, k2))
    if not result.done:
        return {"status": "clean_counterfactual_k1_failed", "decision_edges": None}
    if k2 not in post_k1["reachable_objects"]:
        return {"status": "clean_counterfactual_k1_did_not_expose_k2", "decision_edges": None}
    return {
        "status": "ok",
        "decision_edges": {
            "k1_t0": initial["reachable_edges"][k1],
            "k2_t1": post_k1["reachable_edges"][k2],
        },
    }


def passive_clutter_edge_effect(
    replay: dict, clean_counterfactual: dict
) -> tuple[str | None, dict | None]:
    if clean_counterfactual.get("status") != "ok":
        return clean_counterfactual.get("status") or "clean_counterfactual_failed", None
    states = replay.get("reachability_trace") or []
    if len(states) != 3:
        return "passive_clutter_reachability_trace_missing", None
    k1, k2 = _intended_blockers(2)
    clutter_edges = {
        "k1_t0": states[0]["reachable_edges"][k1],
        "k2_t1": states[1]["reachable_edges"][k2],
    }
    clean_edges = clean_counterfactual["decision_edges"]
    changed = [key for key in ("k1_t0", "k2_t1") if clutter_edges[key] != clean_edges[key]]
    effect = {
        "changed_decisions": changed,
        "clean_edges": clean_edges,
        "clutter_edges": clutter_edges,
    }
    if not changed:
        return "passive_clutter_no_edge_effect", effect
    return None, effect


def _movable_collision_names(info: dict) -> set[str]:
    return {name for name in info.get("movable_collisions", "").split(",") if name}


def sampled_contact_effect(
    replay: dict, clutter_object_id: str, intended_hop: int
) -> tuple[str | None, dict | None]:
    poses = replay.get("object_pose_trace") or []
    infos = replay.get("action_info_trace") or []
    if len(poses) != 3 or len(infos) != 2:
        return "contact_trace_missing", None
    motion_hops = []
    collision_hops = []
    for hop in range(2):
        if _pose_changed(poses[hop].get(clutter_object_id), poses[hop + 1].get(clutter_object_id)):
            motion_hops.append(hop + 1)
        if clutter_object_id in _movable_collision_names(infos[hop]):
            collision_hops.append(hop + 1)
    effect = {
        "intended_hop": intended_hop,
        "motion_hops": motion_hops,
        "collision_hops": collision_hops,
        "action_info_trace": infos,
    }
    if intended_hop not in collision_hops:
        return "contact_not_reported_on_intended_hop", effect
    if intended_hop not in motion_hops:
        return "contact_object_not_moved_on_intended_hop", effect
    return None, effect


def sampled_contact_initialization_failure(replay: dict, placement: dict) -> str | None:
    poses = replay.get("object_pose_trace") or []
    if not poses:
        return "contact_trace_missing"
    context_id = _intended_blockers(3)[-1]
    planned = [*placement["center"], placement["theta"]]
    if _pose_changed(planned, poses[0].get(context_id)):
        return "contact_initial_pose_shifted"
    return None


def augment_contact_manifest(
    source: Path,
    output_dir: Path,
    config: str,
    *,
    limit: int,
    max_attempts: int,
    max_replay_attempts_per_pair: int | None,
    variants_per_hop: int,
    seed: int,
) -> dict:
    source_rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    candidates = [
        (row, hop, variant)
        for row in source_rows
        for hop in range(2)
        for variant in range(variants_per_hop)
    ]
    random.Random(seed).shuffle(candidates)
    rows = []
    rejections: Counter[str] = Counter()
    accepted_geometry: set[str] = set()
    attempts = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    while candidates and attempts < max_attempts and len(rows) < limit:
        source_row, hop, variant = candidates.pop()
        attempts += 1
        try:
            placement = sampled_contact_placement(source_row, hop, variant, seed)
        except CompositionRejected as error:
            rejections[str(error)] += 1
            continue
        output = output_dir / f"contact_{len(rows):04d}.xml"
        composition = compose_sampled_contact_xml(source_row["xml_path"], placement, output)
        reason = sampled_contact_geometry_failure(str(output))
        if reason is not None:
            rejections[reason] += 1
            output.unlink()
            continue
        probe = probe_one((str(output), config, 2))
        ok, reason = static_acceptance(probe, 2)
        if not ok:
            rejections[reason] += 1
            output.unlink()
            continue
        donors = tuple(_donor_from_json(row) for row in source_row["donors"])
        context_id = composition["clutter_object_ids"][0]
        replay = replay_donor_chain(
            str(output),
            config,
            donors,
            max_attempts=max_replay_attempts_per_pair,
            monitored_object_ids=_intended_blockers(2) + [context_id],
        )
        if replay["status"] != "solved":
            rejections[replay.get("failure_reason") or replay["status"]] += 1
            output.unlink()
            continue
        reason = mechanical_independence_failure(replay)
        if reason is None:
            reason = sampled_contact_initialization_failure(replay, placement)
        interaction_effect = None
        if reason is None:
            reason, interaction_effect = sampled_contact_effect(replay, context_id, hop + 1)
        if reason is not None:
            rejections[reason] += 1
            output.unlink()
            continue
        composition["interaction_effect"] = interaction_effect
        full_geometry, wall_geometry = geom_sig(output)
        if full_geometry is None:
            rejections["geometry_identity_failed"] += 1
            output.unlink()
            continue
        if full_geometry in accepted_geometry:
            rejections["duplicate_geometry"] += 1
            output.unlink()
            continue
        rows.append(
            {
                "xml_path": str(output.resolve()),
                "source_manifest": str(source.resolve()),
                "template": source_row["template"],
                "composition": composition,
                "hops": 2,
                "donors": source_row["donors"],
                "geometry_identity": {"full": full_geometry, "walls": wall_geometry},
                "probe": probe,
                "replay": replay,
            }
        )
        accepted_geometry.add(full_geometry)

    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (output_dir / "xmls.txt").write_text(
        "".join(row["xml_path"] + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "mode": "augment_contact_manifest",
        "source_manifest": str(source.resolve()),
        "source_rows": len(source_rows),
        "candidate_variants": len(source_rows) * 2 * variants_per_hop,
        "variants_per_hop": variants_per_hop,
        "max_replay_attempts_per_pair": max_replay_attempts_per_pair,
        "attempted": attempts,
        "accepted": len(rows),
        "rejections": dict(sorted(rejections.items())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def donor_sequences(
    horizons: Sequence[str],
    tiers: Sequence[str],
    template: str,
    min_separation: float,
    seed: int,
    *,
    enforce_min_separation: bool = True,
    pools: Sequence[Sequence[Donor]] | None = None,
) -> Iterable[tuple[Donor, ...]]:
    if pools is None:
        pools = [load_donors(horizon, tier, template) for horizon, tier in zip(horizons, tiers)]
    if any(not pool for pool in pools):
        return []
    candidates = []
    for sequence in itertools.product(*pools):
        if len({donor.xml_path for donor in sequence}) != len(sequence):
            continue
        if enforce_min_separation and any(
            math.dist(left.object_center, right.object_center) < min_separation
            for left, right in itertools.combinations(sequence, 2)
        ):
            continue
        candidates.append(sequence)
    random.Random(seed).shuffle(candidates)
    return candidates


def _donor_json(donor: Donor) -> dict:
    row = asdict(donor)
    row["episode_key"] = list(donor.episode_key)
    return row


def _donor_from_json(row: dict) -> Donor:
    return Donor(
        xml_path=os.path.realpath(str(resolve(row["xml_path"]))),
        object_id=row["object_id"],
        region=row.get("region", "goal"),
        object_center=(float(row["object_center"][0]), float(row["object_center"][1])),
        object_theta=float(row.get("object_theta", 0.0)),
        tier=row["tier"],
        horizon=row["horizon"],
        template=row["template"],
        valid_root=tuple((int(edge), int(depth)) for edge, depth in row["valid_root"]),
    )


def revalidate_manifest(source: Path, output_dir: Path, config: str) -> dict:
    source_rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    accepted_rows = []
    rejected_rows = []
    rejections: Counter[str] = Counter()
    geometry_seen: set[str] = set()
    for source_row in source_rows:
        xml_path = os.path.realpath(str(resolve(source_row["xml_path"])))
        donors = tuple(_donor_from_json(row) for row in source_row["donors"])
        composition = copy.deepcopy(source_row.get("composition") or {})
        composition_mode = composition.get("mode")
        clutter_object_ids = composition.get("clutter_object_ids", [])
        reason = (
            sampled_contact_geometry_failure(xml_path)
            if composition_mode == "same_template_contact"
            else None
        )
        probe = probe_one((xml_path, config, len(donors))) if reason is None else None
        ok = reason is None
        if ok:
            ok, reason = static_acceptance(probe, len(donors))
        replay = None
        if ok:
            replay = replay_donor_chain(
                xml_path,
                config,
                donors,
                monitored_object_ids=_intended_blockers(len(donors)) + clutter_object_ids,
            )
            if replay["status"] != "solved":
                ok = False
                reason = replay.get("failure_reason") or replay["status"]
        if ok and composition_mode in (
            "same_template",
            "same_template_clutter",
            "same_template_contact",
        ):
            reason = mechanical_independence_failure(replay)
            ok = reason is None
        if ok and composition_mode == "same_template_clutter":
            reason = passive_clutter_motion_failure(replay, clutter_object_ids)
            ok = reason is None
        if ok and composition_mode == "same_template_clutter":
            with tempfile.TemporaryDirectory(prefix="keyhole_clean_counterfactual_") as temp_dir:
                clean_xml = Path(temp_dir) / "clean.xml"
                compose_same_template_xml(donors, clean_xml)
                counterfactual = clean_counterfactual_decision_edges(
                    str(clean_xml), config, replay["actions"]
                )
            reason, interaction_effect = passive_clutter_edge_effect(replay, counterfactual)
            ok = reason is None
            if ok:
                composition["interaction_effect"] = interaction_effect
        if ok and composition_mode == "same_template_contact":
            placement = composition["contact_placement"]
            reason = sampled_contact_initialization_failure(replay, placement)
            ok = reason is None
        if ok and composition_mode == "same_template_contact":
            reason, interaction_effect = sampled_contact_effect(
                replay,
                clutter_object_ids[0],
                composition["contact_placement"]["target_hop"],
            )
            ok = reason is None
            if ok:
                composition["interaction_effect"] = interaction_effect
        full_geometry, wall_geometry = geom_sig(xml_path)
        if ok and full_geometry is None:
            ok = False
            reason = "geometry_identity_failed"
        if ok and full_geometry in geometry_seen:
            ok = False
            reason = "duplicate_geometry"
        row = {
            "xml_path": xml_path,
            "source_manifest": str(source.resolve()),
            "template": source_row.get("template") or donors[0].template,
            "composition": composition,
            "hops": len(donors),
            "donors": [_donor_json(donor) for donor in donors],
            "geometry_identity": {"full": full_geometry, "walls": wall_geometry},
            "probe": probe,
            "replay": replay,
        }
        if ok:
            accepted_rows.append(row)
            geometry_seen.add(full_geometry)
        else:
            row["failure_reason"] = reason
            rejected_rows.append(row)
            rejections[reason] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted_rows),
        encoding="utf-8",
    )
    (output_dir / "xmls.txt").write_text(
        "".join(row["xml_path"] + "\n" for row in accepted_rows), encoding="utf-8"
    )
    (output_dir / "rejected.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected_rows),
        encoding="utf-8",
    )
    summary = {
        "mode": "revalidate",
        "source_manifest": str(source.resolve()),
        "attempted": len(source_rows),
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),
        "rejections": dict(sorted(rejections.items())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", choices=HORIZONS)
    parser.add_argument("--tiers", nargs="+", choices=TIERS)
    parser.add_argument("--template", default="set2/benchmark_5")
    parser.add_argument("--composition-mode", choices=COMPOSITION_MODES, default="fixed_template")
    parser.add_argument("--config", default=str(REPO / "config/namo_config_complete_skill15_car_1x.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--revalidate-manifest",
        type=Path,
        help="Revalidate existing composed XMLs and write fresh artifacts without modifying the source.",
    )
    parser.add_argument(
        "--augment-contact-manifest",
        type=Path,
        help="Add sampled movable-object contact interactions to an accepted same-template manifest.",
    )
    parser.add_argument("--contact-variants-per-hop", type=int, default=32)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--min-separation", type=float, default=0.30)
    parser.add_argument("--portal-width", type=float, default=0.10)
    parser.add_argument("--connector-length", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-donor-reuse-early",
        action="store_true",
        help="Use shuffled pair order directly instead of exhausting unused donor episodes first.",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Return success after exhausting candidates even when fewer than --limit are accepted.",
    )
    parser.add_argument(
        "--replay-donor-actions",
        action="store_true",
        help="Require a forward solve using known donor openers; enumerate the second push for 2push donors.",
    )
    parser.add_argument(
        "--max-replay-attempts-per-pair",
        type=int,
        help="Skip a donor pair after this many action simulations; accepted chains remain exact.",
    )
    args = parser.parse_args()
    if args.revalidate_manifest is not None and args.augment_contact_manifest is not None:
        parser.error("choose only one manifest operation")
    if args.revalidate_manifest is not None:
        summary = revalidate_manifest(args.revalidate_manifest, args.out_dir, args.config)
        print(json.dumps(summary, indent=2))
        return 0 if summary["rejected"] == 0 else 2
    if args.augment_contact_manifest is not None:
        summary = augment_contact_manifest(
            args.augment_contact_manifest,
            args.out_dir,
            args.config,
            limit=args.limit,
            max_attempts=args.max_attempts,
            max_replay_attempts_per_pair=args.max_replay_attempts_per_pair,
            variants_per_hop=args.contact_variants_per_hop,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
        return 0 if args.allow_shortfall or summary["accepted"] == args.limit else 2
    if not args.horizons or not args.tiers:
        parser.error("--horizons and --tiers are required when composing new scenes")
    if len(args.horizons) != len(args.tiers):
        parser.error("--horizons and --tiers must have the same length")
    if args.composition_mode == "room_stitch" and (
        len(args.horizons) != 2 or any(horizon != "1push" for horizon in args.horizons)
    ):
        parser.error("--composition-mode room_stitch requires exactly two 1push horizons")
    if args.composition_mode in ("same_template", "same_template_clutter"):
        if len(args.horizons) != 2 or any(horizon != "1push" for horizon in args.horizons):
            parser.error(
                f"--composition-mode {args.composition_mode} requires exactly two 1push horizons"
            )
        if args.template == "any":
            parser.error(
                f"--composition-mode {args.composition_mode} requires one explicit --template"
            )
        if not args.replay_donor_actions:
            parser.error(
                f"--composition-mode {args.composition_mode} requires --replay-donor-actions"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    attempts = 0
    accepted = 0
    rejections: Counter[str] = Counter()
    accepted_geometry: set[str] = set()
    used_donor_episodes: set[tuple[str, str, str]] = set()
    donor_reuse_slots = 0
    module_pools = []
    filtered_pools = None
    if args.composition_mode == "room_stitch":
        filtered_pools = []
        for role, horizon, tier in zip(("exit", "entry"), args.horizons, args.tiers):
            raw_pool = load_donors(horizon, tier, args.template)
            eligible = []
            failures: Counter[str] = Counter()
            for donor in raw_pool:
                try:
                    _module_interface(donor, role, args.config, args.portal_width)
                except CompositionRejected as error:
                    failures[str(error)] += 1
                    continue
                eligible.append(donor)
            filtered_pools.append(eligible)
            module_pools.append(
                {
                    "role": role,
                    "tier": tier,
                    "total": len(raw_pool),
                    "eligible": len(eligible),
                    "rejections": dict(sorted(failures.items())),
                    "eligible_templates": dict(
                        sorted(Counter(donor.template for donor in eligible).items())
                    ),
                }
            )
    sequences = list(
        donor_sequences(
            args.horizons,
            args.tiers,
            args.template,
            args.min_separation,
            args.seed,
            enforce_min_separation=args.composition_mode
            in ("fixed_template", "same_template", "same_template_clutter"),
            pools=filtered_pools,
        )
    )
    candidates: list[tuple[tuple[Donor, ...], str | None]]
    if args.composition_mode == "same_template_clutter":
        candidates = [
            (sequence, clutter_id)
            for sequence in sequences
            for clutter_id in host_clutter_ids(sequence[0])
        ]
        random.Random(args.seed).shuffle(candidates)
    else:
        candidates = [(sequence, None) for sequence in sequences]

    def compose_candidate(
        donors: Sequence[Donor], host_clutter_id: str | None, path: Path
    ) -> dict:
        if args.composition_mode == "room_stitch":
            return compose_room_stitch_xml(
                donors,
                path,
                args.config,
                portal_width=args.portal_width,
                connector_length=args.connector_length,
            )
        if args.composition_mode == "same_template":
            return compose_same_template_xml(donors, path)
        if args.composition_mode == "same_template_clutter":
            if host_clutter_id is None:
                raise CompositionRejected("host_clutter_not_available")
            return compose_same_template_clutter_xml(donors, host_clutter_id, path)
        compose_xml(donors, path)
        return {"mode": "fixed_template"}

    with tempfile.TemporaryDirectory(prefix="keyhole_modules_") as temp_dir:
        while candidates:
            if attempts >= args.max_attempts or accepted >= args.limit:
                break
            preferred = 0
            if not args.allow_donor_reuse_early:
                preferred = next(
                    (
                        index
                        for index, (sequence, _clutter_id) in enumerate(candidates)
                        if all(donor.episode_key not in used_donor_episodes for donor in sequence)
                    ),
                    0,
                )
            donors, host_clutter_id = candidates.pop(preferred)
            attempts += 1
            temp_xml = Path(temp_dir) / f"candidate_{attempts:05d}.xml"
            try:
                composition = compose_candidate(donors, host_clutter_id, temp_xml)
            except CompositionRejected as error:
                rejections[str(error)] += 1
                continue
            probe = probe_one((str(temp_xml), args.config, len(donors)))
            ok, reason = static_acceptance(probe, len(donors))
            if not ok:
                rejections[reason] += 1
                continue
            output = args.out_dir / f"composed_{accepted:04d}.xml"
            try:
                composition = compose_candidate(donors, host_clutter_id, output)
            except CompositionRejected as error:
                rejections[f"final_{error}"] += 1
                continue
            replay = None
            if args.replay_donor_actions:
                monitored_object_ids = _intended_blockers(2) + composition.get(
                    "clutter_object_ids", []
                )
                replay = replay_donor_chain(
                    str(output),
                    args.config,
                    donors,
                    max_attempts=args.max_replay_attempts_per_pair,
                    monitored_object_ids=monitored_object_ids,
                )
                if replay["status"] != "solved":
                    rejections[replay.get("failure_reason") or replay["status"]] += 1
                    output.unlink()
                    continue
                if args.composition_mode in ("same_template", "same_template_clutter"):
                    independence_failure = mechanical_independence_failure(replay)
                    if independence_failure is not None:
                        rejections[independence_failure] += 1
                        output.unlink()
                        continue
                if args.composition_mode == "same_template_clutter":
                    clutter_ids = composition["clutter_object_ids"]
                    clutter_motion_failure = passive_clutter_motion_failure(replay, clutter_ids)
                    if clutter_motion_failure is not None:
                        rejections[clutter_motion_failure] += 1
                        output.unlink()
                        continue
                    clean_xml = Path(temp_dir) / f"clean_counterfactual_{attempts:05d}.xml"
                    compose_same_template_xml(donors, clean_xml)
                    clean_counterfactual = clean_counterfactual_decision_edges(
                        str(clean_xml), args.config, replay["actions"]
                    )
                    interaction_failure, interaction_effect = passive_clutter_edge_effect(
                        replay, clean_counterfactual
                    )
                    if interaction_failure is not None:
                        rejections[interaction_failure] += 1
                        output.unlink()
                        continue
                    composition["interaction_effect"] = interaction_effect
            probe = probe_one((str(output), args.config, len(donors)))
            ok, reason = static_acceptance(probe, len(donors))
            if not ok:
                rejections[f"final_{reason}"] += 1
                output.unlink()
                continue
            full_geometry, wall_geometry = geom_sig(output)
            if full_geometry is None:
                rejections["geometry_identity_failed"] += 1
                output.unlink()
                continue
            if full_geometry in accepted_geometry:
                rejections["duplicate_geometry"] += 1
                output.unlink()
                continue
            row = {
                "xml_path": str(output.resolve()),
                "template": args.template,
                "composition": composition,
                "hops": len(donors),
                "donors": [_donor_json(donor) for donor in donors],
                "geometry_identity": {
                    "full": full_geometry,
                    "walls": wall_geometry,
                },
                "probe": probe,
                "replay": replay,
            }
            rows.append(row)
            accepted_geometry.add(full_geometry)
            donor_reuse_slots += sum(
                donor.episode_key in used_donor_episodes for donor in donors
            )
            used_donor_episodes.update(donor.episode_key for donor in donors)
            accepted += 1

    manifest = args.out_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (args.out_dir / "xmls.txt").write_text(
        "".join(row["xml_path"] + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "attempted": attempts,
        "accepted": accepted,
        "horizons": args.horizons,
        "tiers": args.tiers,
        "template": args.template,
        "composition_mode": args.composition_mode,
        "candidate_variants": len(candidates) + attempts,
        "min_separation": args.min_separation,
        "portal_width_m": args.portal_width if args.composition_mode == "room_stitch" else None,
        "connector_length_m": (
            args.connector_length if args.composition_mode == "room_stitch" else None
        ),
        "module_pools": module_pools,
        "replay_donor_actions": bool(args.replay_donor_actions),
        "max_replay_attempts_per_pair": args.max_replay_attempts_per_pair,
        "allow_donor_reuse_early": bool(args.allow_donor_reuse_early),
        "unique_donor_episodes": len(used_donor_episodes),
        "accepted_donor_slots": accepted * len(args.horizons),
        "reused_donor_slots": donor_reuse_slots,
        "rejections": dict(sorted(rejections.items())),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if args.allow_shortfall or accepted == args.limit else 2


if __name__ == "__main__":
    raise SystemExit(main())
