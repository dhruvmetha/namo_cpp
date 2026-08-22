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
from namo.paths import resolve  # noqa: E402
from namo.planners import get_region_snapshot  # noqa: E402
from namo.runtime_profile import CANONICAL_NUM_DEPTHS  # noqa: E402
from probe_static_topology import is_junk, probe_one, shortest_region_path  # noqa: E402


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
    if is_junk(row):
        return False, "static_junk"
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


def replay_donor_chain(xml_path: str, config: str, donors: Sequence[Donor]) -> dict:
    env = namo_rl.RLEnvironment(xml_path, config, False)
    initial = env.get_full_state()
    attempts = 0

    def state_matches(start_hop: int) -> bool:
        remaining = len(donors) - start_hop
        if remaining == 0:
            return env.is_robot_goal_reachable()
        path, boundaries = _current_path(env)
        if path is None or len(path) - 1 != remaining:
            return False
        expected = [[f"obstacle_{index}_movable"] for index in range(start_hop, len(donors))]
        return boundaries == expected

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
            return actions if env.is_robot_goal_reachable() else None
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
    return {
        "status": "solved" if solution is not None else "no_donor_action_chain",
        "attempts": attempts,
        "actions": solution,
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", choices=HORIZONS, required=True)
    parser.add_argument("--tiers", nargs="+", choices=TIERS, required=True)
    parser.add_argument("--template", default="set2/benchmark_5")
    parser.add_argument("--config", default=str(REPO / "config/namo_config_complete_skill15_car_1x.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--min-separation", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--replay-donor-actions",
        action="store_true",
        help="Require a forward solve using known donor openers; enumerate the second push for 2push donors.",
    )
    args = parser.parse_args()
    if len(args.horizons) != len(args.tiers):
        parser.error("--horizons and --tiers must have the same length")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    attempts = 0
    accepted = 0
    rejections: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="keyhole_modules_") as temp_dir:
        for donors in donor_sequences(
            args.horizons, args.tiers, args.template, args.min_separation, args.seed
        ):
            if attempts >= args.max_attempts or accepted >= args.limit:
                break
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
                    rejections[replay["status"]] += 1
                    output.unlink()
                    continue
            probe = probe_one((str(output), args.config, len(donors)))
            ok, reason = static_acceptance(probe, len(donors))
            if not ok:
                rejections[f"final_{reason}"] += 1
                output.unlink()
                continue
            row = {
                "xml_path": str(output.resolve()),
                "template": args.template,
                "hops": len(donors),
                "donors": [_donor_json(donor) for donor in donors],
                "probe": probe,
                "replay": replay,
            }
            rows.append(row)
            accepted += 1

    manifest = args.out_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "attempted": attempts,
        "accepted": accepted,
        "horizons": args.horizons,
        "tiers": args.tiers,
        "template": args.template,
        "min_separation": args.min_separation,
        "replay_donor_actions": bool(args.replay_donor_actions),
        "rejections": dict(sorted(rejections.items())),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if accepted == args.limit else 2


if __name__ == "__main__":
    raise SystemExit(main())
