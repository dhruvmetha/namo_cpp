#!/usr/bin/env python3
"""Compose canonical keyhole episodes inside one fixed Aug9 wall template.

The unit is an episode ``(xml, object_id, region)``, never an XML.  Each output starts from the
first donor XML, removes every movable object, then inserts only the selected donor blockers.  The
first donor supplies the robot pose and the last donor supplies the XML goal.  Static validation
uses ``probe_static_topology`` and requires the intended blockers to appear in path order.

This is deliberately a pilot composer.  It tests whether blocker-only keyhole modules are portable
before adding donor context objects or scaling collection.
"""

from __future__ import annotations

import argparse
import copy
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
from probe_static_topology import probe_one, shortest_region_path  # noqa: E402
from verify_geom_disjoint import geom_sig  # noqa: E402


TEMPLATE_RE = re.compile(r"/aug9_car/(set[12]/benchmark_[1-5])/")
MOVABLE_RE = re.compile(r"^obstacle_.*_movable$")
TIERS = ("easy", "medium", "hard")
HORIZONS = ("1push", "2push")


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
        if match is None or match.group(1) != template:
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
                template=template,
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
) -> dict:
    return {
        "actions": actions,
        "reachability_trace": list(states),
        "target_point_trace": [list(counts) for counts in point_counts],
        "object_pose_trace": list(poses),
    }


def replay_two_keyhole_goal_chain(
    xml_path: str, config: str, donors: Sequence[Donor]
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
    initial_poses = _intended_object_poses(env, object_ids)
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
    for k1_edge, k1_depth in donors[0].valid_root:
        env.set_full_state(initial)
        attempts += 1
        k1_result = env.step(_action(object_ids[0], k1_edge, k1_depth))
        post_k1_state = _intended_reachability_state(env, object_ids)
        post_k1_counts = _target_point_counts(env, target_points)
        post_k1_poses = _intended_object_poses(env, object_ids)
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
                ),
            )
            continue
        post_k1_candidates += 1
        post_k1_full_state = env.get_full_state()

        for k2_edge, k2_depth in donors[1].valid_root:
            env.set_full_state(post_k1_full_state)
            attempts += 1
            k2_result = env.step(_action(object_ids[1], k2_edge, k2_depth))
            post_k2_state = _intended_reachability_state(env, object_ids)
            post_k2_counts = _target_point_counts(env, target_points)
            post_k2_poses = _intended_object_poses(env, object_ids)
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
                "final_object_poses": post_k2_poses,
                "candidate_rejections": dict(sorted(candidate_rejections.items())),
            }

    if not donors[0].valid_root:
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


def replay_donor_chain(xml_path: str, config: str, donors: Sequence[Donor]) -> dict:
    if len(donors) == 2 and all(donor.horizon == "1push" for donor in donors):
        return replay_two_keyhole_goal_chain(xml_path, config, donors)
    return replay_component_chain(xml_path, config, donors)


def donor_sequences(
    horizons: Sequence[str], tiers: Sequence[str], template: str, min_separation: float, seed: int
) -> Iterable[tuple[Donor, ...]]:
    pools = [load_donors(horizon, tier, template) for horizon, tier in zip(horizons, tiers)]
    if any(not pool for pool in pools):
        return []
    candidates = []
    for sequence in itertools.product(*pools):
        if len({donor.xml_path for donor in sequence}) != len(sequence):
            continue
        if any(
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
        probe = probe_one((xml_path, config, len(donors)))
        ok, reason = static_acceptance(probe, len(donors))
        replay = None
        if ok:
            replay = replay_donor_chain(xml_path, config, donors)
            if replay["status"] != "solved":
                ok = False
                reason = replay.get("failure_reason") or replay["status"]
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
    parser.add_argument("--config", default=str(REPO / "config/namo_config_complete_skill15_car_1x.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--revalidate-manifest",
        type=Path,
        help="Revalidate existing composed XMLs and write fresh artifacts without modifying the source.",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--min-separation", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
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
    args = parser.parse_args()
    if args.revalidate_manifest is not None:
        summary = revalidate_manifest(args.revalidate_manifest, args.out_dir, args.config)
        print(json.dumps(summary, indent=2))
        return 0 if summary["rejected"] == 0 else 2
    if not args.horizons or not args.tiers:
        parser.error("--horizons and --tiers are required when composing new scenes")
    if len(args.horizons) != len(args.tiers):
        parser.error("--horizons and --tiers must have the same length")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    attempts = 0
    accepted = 0
    rejections: Counter[str] = Counter()
    accepted_geometry: set[str] = set()
    used_donor_episodes: set[tuple[str, str, str]] = set()
    donor_reuse_slots = 0
    sequences = list(
        donor_sequences(args.horizons, args.tiers, args.template, args.min_separation, args.seed)
    )
    with tempfile.TemporaryDirectory(prefix="keyhole_modules_") as temp_dir:
        while sequences:
            if attempts >= args.max_attempts or accepted >= args.limit:
                break
            preferred = next(
                (
                    index
                    for index, sequence in enumerate(sequences)
                    if all(donor.episode_key not in used_donor_episodes for donor in sequence)
                ),
                0,
            )
            donors = sequences.pop(preferred)
            attempts += 1
            temp_xml = Path(temp_dir) / f"candidate_{attempts:05d}.xml"
            compose_xml(donors, temp_xml)
            probe = probe_one((str(temp_xml), args.config, len(donors)))
            ok, reason = static_acceptance(probe, len(donors))
            if not ok:
                rejections[reason] += 1
                continue
            output = args.out_dir / f"composed_{accepted:04d}.xml"
            compose_xml(donors, output)
            replay = None
            if args.replay_donor_actions:
                replay = replay_donor_chain(str(output), args.config, donors)
                if replay["status"] != "solved":
                    rejections[replay.get("failure_reason") or replay["status"]] += 1
                    output.unlink()
                    continue
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
        "min_separation": args.min_separation,
        "replay_donor_actions": bool(args.replay_donor_actions),
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
