#!/usr/bin/env python3
"""Certify independent keyhole horizons and publish genuine-two-push scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import namo_rl

import compose_keyhole_modules as composer


CANONICAL_DEPTHS = 5
CANONICAL_MIN_REACHABLE_FRACTION = 0.2
BLOCKER_IDS = ("obstacle_0_movable", "obstacle_1_movable")
TYPED_PATTERNS = ("12", "21", "22")
PARTIAL_PATTERNS = ("2?", "?2")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSON objects from a JSONL file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write deterministic JSONL, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _geometry_id(row: dict[str, Any]) -> str:
    value = str((row.get("geometry_identity") or {}).get("full") or "")
    if not value:
        raise ValueError("scene has no full geometry identity")
    return value


def _goal_from_xml(path: Path) -> tuple[float, float, float]:
    site = ET.parse(path).getroot().find(".//site[@name='goal']")
    if site is None:
        raise ValueError(f"{path} has no goal site")
    values = [float(value) for value in (site.get("pos") or "").split()]
    if len(values) < 2:
        raise ValueError(f"{path} has an invalid goal position")
    return values[0], values[1], 0.0


def _action(object_id: str, edge: int, depth: int) -> Any:
    action = namo_rl.Action()
    action.object_id = object_id
    action.edge_idx = int(edge)
    action.depth = int(depth)
    action.x = action.y = action.theta = 0.0
    return action


def _step_record(result: Any, point_count: int, threshold: int) -> dict[str, Any]:
    info = dict(result.info or {})
    done = bool(result.done)
    return {
        "done": done,
        "opened": done and point_count >= threshold,
        "reachable_point_count": int(point_count),
        "failure_type": info.get("failure_type", ""),
        "failure_reason": info.get("failure_reason", ""),
        "collision_object": info.get("collision_object", ""),
        "wall_collision": str(info.get("wall_collision", "false")).lower() == "true",
    }


def certify_gate(
    env: Any,
    object_id: str,
    target_points: Sequence[Sequence[float]],
    witness_actions: Sequence[Sequence[int]],
    *,
    discover_witness: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Exhaust one-push cells, then replay a supplied or newly searched witness."""
    threshold = max(
        1,
        math.ceil(CANONICAL_MIN_REACHABLE_FRACTION * len(target_points)),
    )
    initial_state = env.get_full_state()
    initial_point_count = int(env.count_reachable_points(target_points)[0])
    target_reachable = object_id in set(env.get_reachable_objects())
    reachable_edges = (
        sorted(int(edge) for edge in env.get_reachable_edges(object_id))
        if target_reachable
        else []
    )
    trials: list[dict[str, Any]] = []
    valid_onepush: list[list[int]] = []
    for edge in reachable_edges:
        for depth in range(CANONICAL_DEPTHS):
            env.set_full_state(initial_state)
            result = env.step(_action(object_id, edge, depth))
            point_count = int(env.count_reachable_points(target_points)[0])
            record = {
                "edge": edge,
                "depth": depth,
                **_step_record(result, point_count, threshold),
            }
            trials.append(record)
            if record["opened"]:
                valid_onepush.append([edge, depth])

    env.set_full_state(initial_state)
    witness_search = None
    if (not valid_onepush and initial_point_count < threshold and reachable_edges
            and not witness_actions and discover_witness is not None):
        witness_search = discover_witness()
        witness_actions = witness_search["actions"]
        env.set_full_state(initial_state)
    witness_steps: list[dict[str, Any]] = []
    for edge, depth in witness_actions:
        result = env.step(_action(object_id, int(edge), int(depth)))
        point_count = int(env.count_reachable_points(target_points)[0])
        witness_steps.append(
            {
                "edge": int(edge),
                "depth": int(depth),
                **_step_record(result, point_count, threshold),
            }
        )

    initial_closed = initial_point_count < threshold
    witness_done = [step["done"] for step in witness_steps]
    witness_opened = [step["opened"] for step in witness_steps]
    if valid_onepush:
        label = "1"
        classification = "onepush_exists"
    elif (
        initial_closed
        and target_reachable
        and bool(reachable_edges)
        and len(witness_steps) == 2
        and witness_done == [True, True]
        and witness_opened == [False, True]
    ):
        label = "2"
        classification = "genuine_2push"
    else:
        label = "?"
        classification = "two_push_witness_failed"

    return {
        "object_id": object_id,
        "label": label,
        "classification": classification,
        "target_point_count": len(target_points),
        "target_point_threshold": threshold,
        "initial_reachable_point_count": initial_point_count,
        "initial_closed": initial_closed,
        "target_reachable": target_reachable,
        "reachable_edges": reachable_edges,
        "reachable_edge_count": len(reachable_edges),
        "tried_onepush_count": len(trials),
        "valid_onepush": valid_onepush,
        "valid_onepush_count": len(valid_onepush),
        "onepush_trials": trials,
        "witness_search": witness_search,
        "witness": {
            "actions": [[int(edge), int(depth)] for edge, depth in witness_actions],
            "done": witness_done,
            "opened": witness_opened,
            "steps": witness_steps,
        },
    }


def classify_scene(k1_label: str, k2_label: str) -> str:
    """Return a retained pattern or ``reject`` when neither gate is certified two-push."""
    pattern = f"{k1_label}{k2_label}"
    if "2" not in pattern:
        return "reject"
    return pattern if pattern in TYPED_PATTERNS + PARTIAL_PATTERNS else "reject"


def boundary_label(object_certificates: Sequence[dict[str, Any]]) -> str:
    """Combine complete per-object sweeps; any alternative one-push opener wins."""
    labels = {row["label"] for row in object_certificates}
    return "1" if "1" in labels else "2" if "2" in labels else "?"


def generated_group(pattern: str) -> str | None:
    """Keep 11 and any independently certified two-push, including partial labels."""
    if pattern == "11":
        return "11"
    return "has_2push" if pattern in TYPED_PATTERNS + PARTIAL_PATTERNS else None


def write_removed_blockers_xml(source: Path, blockers: Sequence[str], output: Path) -> None:
    """Remove only the initial K1 blockers in a separate K2 certification XML."""
    tree = ET.parse(source)
    world = tree.getroot().find("worldbody")
    if world is None:
        raise ValueError(f"{source} has no worldbody")
    bodies = {body.get("name"): body for body in world.findall("body")}
    for object_id in blockers:
        world.remove(bodies[object_id])
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def replay_complete_witness(
    env: Any,
    blocker_ids: Sequence[str],
    actions_by_hop: Sequence[Sequence[Sequence[int]]],
) -> dict[str, Any]:
    """Replay the known full-scene chain; success is end-goal reachability only."""
    initial_goal_reachable = bool(env.is_robot_goal_reachable())
    action_records: list[dict[str, Any]] = []
    for object_id, hop_actions in zip(blocker_ids, actions_by_hop):
        for edge, depth in hop_actions:
            result = env.step(_action(object_id, int(edge), int(depth)))
            info = dict(result.info or {})
            record = {
                "object_id": object_id,
                "edge": int(edge),
                "depth": int(depth),
                "done": bool(result.done),
                "failure_type": info.get("failure_type", ""),
                "failure_reason": info.get("failure_reason", ""),
                "collision_object": info.get("collision_object", ""),
                "wall_collision": str(info.get("wall_collision", "false")).lower()
                == "true",
            }
            action_records.append(record)
            if not record["done"]:
                return {
                    "success": False,
                    "initial_goal_reachable": initial_goal_reachable,
                    "end_goal_reachable": bool(env.is_robot_goal_reachable()),
                    "actions": action_records,
                    "failure_reason": "full_witness_action_failed",
                }
    end_goal_reachable = bool(env.is_robot_goal_reachable())
    success = not initial_goal_reachable and end_goal_reachable
    return {
        "success": success,
        "initial_goal_reachable": initial_goal_reachable,
        "end_goal_reachable": end_goal_reachable,
        "actions": action_records,
        "failure_reason": None if success else "full_witness_end_goal_unreachable",
    }


def write_independent_xml(source: Path, hop: int, output: Path) -> None:
    """Write one gate with the other intended blocker removed."""
    if hop not in (1, 2):
        raise ValueError(f"hop must be 1 or 2, got {hop}")
    target_id = BLOCKER_IDS[hop - 1]
    other_id = BLOCKER_IDS[2 - hop]
    tree = ET.parse(source)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{source} has no worldbody")
    bodies = {body.get("name"): body for body in worldbody.findall("body")}
    if target_id not in bodies or other_id not in bodies:
        raise ValueError(f"{source} does not contain both intended blockers")
    worldbody.remove(bodies[other_id])
    root.set("model", f"independent_k{hop}_{root.get('model', 'scene')}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def _target_points(env: Any) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    snapshot = composer.get_region_snapshot(
        env,
        goals_per_region=100,
        local_info_only=False,
        seed=42,
        use_cpp_unified=True,
        use_xml_goal=True,
    )
    robot_label = str(snapshot.get("robot_label") or "")
    goal_label = str(snapshot.get("goal_label") or "")
    path = composer.shortest_region_path(snapshot["adjacency"], robot_label, goal_label)
    if path is None or len(path) != 2:
        raise ValueError(f"independent gate does not have exactly one boundary: {path}")
    points = [
        (float(goal.x), float(goal.y))
        for goal in snapshot["region_goals"][goal_label].goals
    ]
    if not points:
        raise ValueError("independent target region has no sampled points")
    return points, {"robot_label": robot_label, "goal_label": goal_label, "path": path}


def build_tasks(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate nominal-22 input rows and create one task per unique scene."""
    tasks = []
    seen = set()
    for scene_index, row in enumerate(rows):
        geometry_id = _geometry_id(row)
        if geometry_id in seen:
            raise ValueError(f"duplicate full geometry identity: {geometry_id}")
        seen.add(geometry_id)
        actions = (row.get("replay") or {}).get("actions") or []
        pattern = str((row.get("pilot") or {}).get("horizon_pattern") or "")
        if pattern != "22" or len(actions) != 2 or [len(hop) for hop in actions] != [2, 2]:
            raise ValueError(f"scene {scene_index} is not a nominal-22 witness")
        if (row.get("replay") or {}).get("status") != "solved":
            raise ValueError(f"scene {scene_index} has no solved full witness")
        tasks.append(
            {
                "task_index": scene_index,
                "task_id": f"{scene_index:04d}_{geometry_id[:16]}",
                "source_xml": str(Path(row["xml_path"]).resolve()),
                "geometry_id": geometry_id,
                "tier_pair": (row.get("pilot") or {}).get("tier_pair"),
                "source_cell": (row.get("pilot") or {}).get("cell"),
                "witness_actions": [
                    [[int(edge), int(depth)] for edge, depth in hop]
                    for hop in actions
                ],
            }
        )
    return tasks


def _runtime() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "namo_rl": str(Path(namo_rl.__file__).resolve()),
    }


def certify_scene(
    task: dict[str, Any],
    output_dir: Path,
    config: str,
) -> dict[str, Any]:
    """Certify both isolated gates and the known complete-scene witness."""
    started = time.perf_counter()
    source_xml = Path(task["source_xml"])
    goal = _goal_from_xml(source_xml)
    gates = []
    for hop, object_id in enumerate(BLOCKER_IDS, start=1):
        independent_xml = output_dir / "independent_xml" / f"{task['task_id']}_k{hop}.xml"
        write_independent_xml(source_xml, hop, independent_xml)
        env = namo_rl.RLEnvironment(str(independent_xml), config, False)
        env.reset()
        env.set_robot_goal_silent(*goal)
        points, topology = _target_points(env)
        gate = certify_gate(
            env,
            object_id,
            points,
            task["witness_actions"][hop - 1],
        )
        gate.update(
            {
                "hop": hop,
                "independent_xml": str(independent_xml.resolve()),
                "topology": topology,
            }
        )
        gates.append(gate)

    full_env = namo_rl.RLEnvironment(str(source_xml), config, False)
    full_env.reset()
    full_env.set_robot_goal_silent(*goal)
    full_witness = replay_complete_witness(
        full_env,
        BLOCKER_IDS,
        task["witness_actions"],
    )
    pattern = classify_scene(gates[0]["label"], gates[1]["label"])
    return {
        **task,
        "status": "ok",
        "measured_pattern": pattern,
        "retained": full_witness["success"] and pattern != "reject",
        "gates": gates,
        "full_witness": full_witness,
        "runtime": _runtime(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def prepare(source: Path, output: Path) -> None:
    """Prepare one deterministic certification task per source scene."""
    if output.exists():
        raise FileExistsError(output)
    rows = read_jsonl(source)
    tasks = build_tasks(rows)
    write_jsonl(output, tasks)
    print(json.dumps({"source_scenes": len(rows), "tasks": len(tasks)}, sort_keys=True))


def prepare_generated(source: Path, output: Path, witness_budget: int) -> None:
    """Make one task per room geometry, selecting one deterministic start/goal pair."""
    if output.exists():
        raise FileExistsError(output)
    if witness_budget <= 0:
        raise ValueError("witness budget must be positive")
    tasks, seen = [], set()
    paths = sorted(source.rglob("*.xml"))
    for xml in paths:
        geometry, walls = composer.geom_sig(str(xml))
        if geometry is None:
            raise ValueError(f"cannot identify geometry: {xml}")
        if geometry in seen:
            continue
        seen.add(geometry)
        relative = xml.relative_to(source)
        # The generator writes setN/benchmark_M/run_XXXX/env_XXXX_pair_YYY.xml.
        template = "/".join(relative.parts[:2])
        index = len(tasks)
        tasks.append({
            "task_index": index, "task_id": f"{index:05d}_{geometry}",
            "source_kind": "generated", "source_xml": str(xml.resolve()),
            "geometry_id": geometry, "wall_geometry_id": walls,
            "template": template, "witness_budget": witness_budget,
        })
    write_jsonl(output, tasks)
    print(json.dumps({"xmls": len(paths), "unique_rooms": len(tasks),
                      "by_template": dict(Counter(t["template"] for t in tasks))}))


class _CompletedPushEnv:
    """Keep failed setup states out of certification-only witness discovery."""

    def __init__(self, inner):
        self.inner = inner
        self.completed = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self, action):
        state = self.inner.get_full_state()
        result = self.inner.step(action)
        self.completed = bool(result.done)
        if not self.completed:
            self.inner.set_full_state(state)
        return result


def _discover_witness(env: Any, object_id: str, points: Sequence[Sequence[float]],
                      xml: Path, budget: int) -> dict[str, Any]:
    """Use the unchanged uniform best-first search to find a same-object witness."""
    from types import SimpleNamespace
    import numpy as np
    from namo.strategies import PrimitiveGoalStrategy
    from namo.planners.opening.best_first_search import solve_scene

    planner = SimpleNamespace(prim=PrimitiveGoalStrategy(
        data_dir=str(composer.REPO / "data"), primitive_prefix="1x_car_d5_",
        max_push_steps=CANONICAL_DEPTHS), scorer=None)
    solution = {}
    threshold = math.ceil(CANONICAL_MIN_REACHABLE_FRACTION * len(points))
    solved, simulations, _, _, end = solve_scene(
        planner, _CompletedPushEnv(env), _goal_from_xml(xml), str(xml),
        env.get_full_state(), 2, budget, "uniform", "mean5", "q",
        np.random.default_rng(42), restrict_obj=object_id,
        is_open=lambda current: current.completed and
            int(current.count_reachable_points(points)[0]) >= threshold,
        raw=True, discount="off", dedupe_noop=True, prune_jam_depth=True,
        region_samples=[(float(x), float(y), 0.0) for x, y in points],
        solution_out=solution,
    )
    return {"actions": [[int(goal.edge_idx), int(goal.depth)]
                        for _, goal in solution.get("plan", [])] if solved else [],
            "simulations": int(simulations), "end": end, "seed": 42,
            "budget": budget}


def _generated_snapshot(env: Any) -> tuple[dict[str, Any], list[str] | None]:
    snapshot = composer.get_region_snapshot(
        env, goals_per_region=100, local_info_only=False, seed=42,
        use_cpp_unified=True, use_xml_goal=True)
    path = composer.shortest_region_path(
        snapshot["adjacency"], str(snapshot.get("robot_label") or ""),
        str(snapshot.get("goal_label") or ""))
    return snapshot, path


def _certify_generated_boundary(env: Any, xml: Path, snapshot: dict[str, Any],
                                source: str, target: str, budget: int) -> dict[str, Any]:
    from probe_static_topology import _boundary_objects

    objects, error = _boundary_objects(snapshot["edge_objects"], source, target)
    if error:
        raise ValueError(error)
    points = [(float(goal.x), float(goal.y))
              for goal in snapshot["region_goals"][target].goals]
    if not points:
        return {"label": "?", "reason": "empty_target", "objects": objects,
                "object_certificates": []}
    baseline = env.get_full_state()
    certificates = []
    for object_id in objects:
        env.set_full_state(baseline)
        certificate = certify_gate(
            env, object_id, points, (),
            discover_witness=lambda: _discover_witness(env, object_id, points, xml, budget))
        certificates.append(certificate)
    env.set_full_state(baseline)
    initial_closed = int(env.count_reachable_points(points)[0]) < math.ceil(
        CANONICAL_MIN_REACHABLE_FRACTION * len(points))
    return {"label": boundary_label(certificates) if initial_closed else "?",
            "initial_closed": initial_closed, "source_region": source,
            "target_region": target, "objects": objects, "target_points": points,
            "object_certificates": certificates}


def certify_generated_scene(task: dict[str, Any], output_dir: Path, config: str) -> dict[str, Any]:
    """Certify K1 in the complete scene, and K2 with only initial K1 removed."""
    started = time.perf_counter()
    xml = Path(task["source_xml"])
    env = namo_rl.RLEnvironment(str(xml), config, False)
    env.reset()
    env.set_robot_goal_silent(*_goal_from_xml(xml))
    snapshot, path = _generated_snapshot(env)
    base = {**task, "runtime": _runtime(), "initial_path": path,
            "full_solution_verified": False}
    if not path or len(path) != 3 or env.is_robot_goal_reachable():
        return {**base, "status": "invalid_topology", "retained": False}
    k1 = _certify_generated_boundary(env, xml, snapshot, path[0], path[1], task["witness_budget"])
    k1.update(hop=1, certification_state="original_complete_scene")
    independent = output_dir / "independent_xml" / f"{task['task_id']}_k2.xml"
    write_removed_blockers_xml(xml, k1["objects"], independent)
    k2_env = namo_rl.RLEnvironment(str(independent), config, False)
    k2_env.reset()
    k2_env.set_robot_goal_silent(*_goal_from_xml(independent))
    k2_snapshot, k2_path = _generated_snapshot(k2_env)
    if k2_path and len(k2_path) == 2:
        k2 = _certify_generated_boundary(
            k2_env, independent, k2_snapshot, k2_path[0], k2_path[1], task["witness_budget"])
    else:
        k2 = {"label": "?", "reason": "k1_removed_not_one_hop",
              "object_certificates": [], "objects": []}
    k2.update(hop=2, certification_state="initial_k1_removed", path=k2_path,
              removed_objects=k1["objects"], independent_xml=str(independent.resolve()))
    pattern = k1["label"] + k2["label"]
    group = generated_group(pattern)
    return {**base, "status": "ok", "gates": [k1, k2],
            "measured_pattern": pattern, "group": group, "retained": group is not None,
            "elapsed_seconds": round(time.perf_counter() - started, 6)}


def aggregate_generated(tasks_path: Path, results_dir: Path, output_dir: Path) -> None:
    """Publish the two independent-label groups without conditioning on full solves."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    tasks = read_jsonl(tasks_path)
    results = [json.loads(p.read_text()) for p in sorted((results_dir / "results").glob("task_*.json"))]
    by_index = {row["task_index"]: row for row in results}
    if len(results) != len(tasks) or set(by_index) != set(range(len(tasks))):
        raise RuntimeError("incomplete or duplicate generated-scene certificates")
    retained, errors = [], []
    for task in tasks:
        row = by_index[task["task_index"]]
        if row["task_id"] != task["task_id"]:
            raise RuntimeError(f"mismatched task {task['task_id']}")
        if row["status"] == "error":
            errors.append(row)
        if not row.get("retained"):
            continue
        for gate in row["gates"]:
            for certificate in gate["object_certificates"]:
                if certificate["tried_onepush_count"] != certificate["reachable_edge_count"] * CANONICAL_DEPTHS:
                    raise RuntimeError(f"incomplete one-push sweep in {row['task_id']}")
        retained.append({
            "xml_path": row["source_xml"], "geometry_id": row["geometry_id"],
            "template": row["template"], "horizon_pattern": row["measured_pattern"],
            "group": row["group"], "category": row["group"], "scene_type": "sampled",
            "testbed": {"category": row["group"], "scene_type": "sampled"},
            "tier_pair": None, "cohort": "aug9_sampled_independent",
            "certificate_file": str((results_dir / "results" / f"task_{row['task_index']:04d}.json").resolve()),
            "full_solution_verified": False,
        })
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "certification_results.jsonl", [by_index[i] for i in range(len(tasks))])
    write_jsonl(output_dir / "errors.jsonl", errors)
    write_jsonl(output_dir / "manifest.jsonl", retained)
    (output_dir / "xmls.txt").write_text("".join(row["xml_path"] + "\n" for row in retained))
    for group in ("11", "has_2push"):
        write_jsonl(output_dir / group / "manifest.jsonl", [r for r in retained if r["group"] == group])
    summary = {"tasks": len(tasks), "retained": len(retained), "technical_errors": len(errors),
               "status_counts": dict(Counter(r["status"] for r in results)),
               "patterns": dict(Counter(r.get("measured_pattern", "invalid") for r in results)),
               "by_template": dict(Counter(r["template"] for r in retained)),
               "groups": dict(Counter(r["group"] for r in retained)),
               "full_solvability": "not required for independent labels; measured separately by Full NAMO"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if errors:
        raise RuntimeError(f"{len(errors)} technical errors; preserved in errors.jsonl")


def run_tasks(
    tasks_path: Path,
    task_index: int,
    task_count: int,
    output_dir: Path,
    config: str,
) -> None:
    """Process the strided task subset assigned to one Slurm array worker."""
    tasks = read_jsonl(tasks_path)
    if task_count <= 0 or task_index < 0 or task_index >= task_count:
        raise ValueError(f"invalid task shard {task_index}/{task_count}")
    for index in range(task_index, len(tasks), task_count):
        task = tasks[index]
        result_path = output_dir / "results" / f"task_{index:04d}.json"
        if result_path.exists():
            raise FileExistsError(result_path)
        try:
            certify = certify_generated_scene if task.get("source_kind") == "generated" else certify_scene
            result = certify(task, output_dir, config)
        except Exception as error:
            result = {
                **task,
                "status": "error",
                "retained": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "runtime": _runtime(),
            }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(result_path)
        print(
            json.dumps(
                {
                    "task_index": index,
                    "task_id": task["task_id"],
                    "status": result["status"],
                    "pattern": result.get("measured_pattern"),
                },
                sort_keys=True,
            )
        )


def _compact_certification(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": result["task_id"],
        "measured_pattern": result["measured_pattern"],
        "full_witness": result["full_witness"],
        "gates": [
            {
                key: gate[key]
                for key in (
                    "hop",
                    "object_id",
                    "label",
                    "classification",
                    "target_point_count",
                    "target_point_threshold",
                    "initial_reachable_point_count",
                    "reachable_edge_count",
                    "tried_onepush_count",
                    "valid_onepush_count",
                    "valid_onepush",
                    "witness",
                )
            }
            for gate in result["gates"]
        ],
    }


def _write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output_dir)}\n")
    (output_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def aggregate(
    source: Path,
    tasks_path: Path,
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Join complete results and publish typed and partial genuine-two-push manifests."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_rows = read_jsonl(source)
    tasks = read_jsonl(tasks_path)
    result_paths = sorted((results_dir / "results").glob("task_*.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    by_index = {int(row["task_index"]): row for row in results}
    expected = set(range(len(tasks)))
    if len(results) != len(by_index) or set(by_index) != expected:
        raise RuntimeError(
            f"result coverage mismatch: missing={sorted(expected - set(by_index))[:10]} "
            f"extra={sorted(set(by_index) - expected)[:10]}"
        )
    source_by_geometry = {_geometry_id(row): row for row in source_rows}
    if len(source_by_geometry) != len(source_rows):
        raise RuntimeError("source manifest contains duplicate geometries")

    retained: dict[str, list[dict[str, Any]]] = {
        pattern: [] for pattern in TYPED_PATTERNS + PARTIAL_PATTERNS
    }
    rejected = []
    ordered_results = [by_index[index] for index in range(len(tasks))]
    for task, result in zip(tasks, ordered_results):
        if result["task_id"] != task["task_id"]:
            raise RuntimeError(f"task/result mismatch at {task['task_index']}")
        if result["status"] == "ok":
            for gate in result["gates"]:
                expected_trials = gate["reachable_edge_count"] * CANONICAL_DEPTHS
                if gate["tried_onepush_count"] != expected_trials:
                    raise RuntimeError(f"incomplete primitive sweep in {task['task_id']}")
        pattern = result.get("measured_pattern", "reject")
        if result.get("retained") and pattern in retained:
            source_row = dict(source_by_geometry[result["geometry_id"]])
            source_row["horizon_certification"] = _compact_certification(result)
            retained[pattern].append(source_row)
        else:
            rejected.append(result)

    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "certification_results.jsonl", ordered_results)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    for pattern, rows in retained.items():
        write_jsonl(output_dir / "patterns" / pattern / "manifest.jsonl", rows)
    typed_rows = [row for pattern in TYPED_PATTERNS for row in retained[pattern]]
    partial_rows = [row for pattern in PARTIAL_PATTERNS for row in retained[pattern]]
    write_jsonl(output_dir / "manifest.jsonl", typed_rows)
    write_jsonl(output_dir / "partial_manifest.jsonl", partial_rows)
    population = {
        "name": "genuine-two-push-two-keyhole-v2",
        "scenes": [
            {
                "xml_path": row["xml_path"],
                "cluster_id": f"geometry:{_geometry_id(row)}",
                "horizon_pattern": row["horizon_certification"]["measured_pattern"],
            }
            for row in typed_rows
        ],
    }
    (output_dir / "population.json").write_text(
        json.dumps(population, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pattern_counts = {pattern: len(rows) for pattern, rows in retained.items()}
    status_counts = Counter(row["status"] for row in ordered_results)
    rejection_counts = Counter(
        row.get("error")
        or (row.get("full_witness") or {}).get("failure_reason")
        or "no_certified_two_push"
        for row in rejected
    )
    summary = {
        "source_scenes": len(source_rows),
        "task_count": len(tasks),
        "result_count": len(ordered_results),
        "status_counts": dict(sorted(status_counts.items())),
        "typed_scenes": len(typed_rows),
        "partial_scenes": len(partial_rows),
        "rejected_scenes": len(rejected),
        "pattern_counts": pattern_counts,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "unique_typed_geometries": len({_geometry_id(row) for row in typed_rows}),
        "total_onepush_simulations": sum(
            gate["tried_onepush_count"]
            for row in ordered_results
            if row["status"] == "ok"
            for gate in row["gates"]
        ),
        "contract": {
            "local_opening": "at least 20 percent of 100 fixed target-region points",
            "genuine_2push": "zero completed one-push openers and a completed [closed, open] witness",
            "full_namo_success": "complete-scene end goal reachable",
            "retention": "at least one independently certified genuine-2push gate",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_checksums(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    generated = commands.add_parser("prepare-generated")
    generated.add_argument("--source", type=Path, required=True)
    generated.add_argument("--output", type=Path, required=True)
    generated.add_argument("--witness-budget", type=int, default=4000)
    run = commands.add_parser("run")
    run.add_argument("--tasks", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--task-count", type=int, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--config", required=True)
    agg = commands.add_parser("aggregate")
    agg.add_argument("--source", type=Path, required=True)
    agg.add_argument("--tasks", type=Path, required=True)
    agg.add_argument("--results-dir", type=Path, required=True)
    agg.add_argument("--output-dir", type=Path, required=True)
    gen_agg = commands.add_parser("aggregate-generated")
    gen_agg.add_argument("--tasks", type=Path, required=True)
    gen_agg.add_argument("--results-dir", type=Path, required=True)
    gen_agg.add_argument("--output-dir", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args.source, args.output)
    elif args.command == "prepare-generated":
        prepare_generated(args.source, args.output, args.witness_budget)
    elif args.command == "aggregate-generated":
        aggregate_generated(args.tasks, args.results_dir, args.output_dir)
    elif args.command == "run":
        run_tasks(
            args.tasks,
            args.task_index,
            args.task_count,
            args.output_dir,
            args.config,
        )
    else:
        aggregate(
            args.source,
            args.tasks,
            args.results_dir,
            args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
