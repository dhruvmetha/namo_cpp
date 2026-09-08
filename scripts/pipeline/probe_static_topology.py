#!/usr/bin/env python3
"""Static per-scene topology + blocker-reachability probe. ONE region snapshot, ZERO simulated pushes.

Purpose: cheaply filter known-junk scenes out of a generated pool BEFORE spending compute labelling
them. Every quantity here is read off the t=0 state, so a scene can be rejected without a single
second of physics.

Per XML it records:
  goal_in_free_space  the C++ snapshot flag (goal point landed in a labelled free region)
  robot_label / goal_label / region_path / hop_count
  per boundary on that path: the blocking objects from edge_objects, which of them the robot can
    reach at t=0 (`get_reachable_objects`), and how many push edges each reachable one exposes
    (`get_reachable_edges`)
  derived flags: no_blocking_objects, no_reachable_blocker, no_pushable_blocker, hop_mismatch

Boundary object lists name the objects on the boundary. A multi-name list alone does *not* say
whether either object opens the boundary: `multi_object_edges` is the authoritative marker for a
boundary that needs the whole multi-object plug. The optional census mode below preserves that
distinction rather than treating every multi-name boundary as an OR.

Which boundary matters. The deploy planner (full_namo_planner.search) always opens `path[1]` — the
FIRST hop off the robot region — so only boundary 0 is a *static* defect. Boundaries further along
the path sit behind the first one and are unreachable at t=0 almost by construction; their flags are
recorded for diagnosis but the scene-level `*_first` flags are what should gate the pool.

Reuses: namo.planners.get_region_snapshot (the same authoritative C++ snapshot the eval selection in
namo.environment_selection.analyze_environment_path_length uses) and the opener's own boundary-object
resolution rule (best_first_region_opening._boundary_objects), so the probe and the planner cannot
disagree about what blocks a boundary.

  python scripts/pipeline/probe_static_topology.py --manifest M.txt --out probe.jsonl \
      --config config/namo_config_complete_skill15_car_1x.yaml --workers 32
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict, deque
from multiprocessing import Pool

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(REPO, "build_python"), os.path.join(REPO, "python"),
           os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import namo_rl  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_from_xml  # noqa: E402
from namo import eval_sets  # noqa: E402
from namo.paths import resolve as resolve_namo_path  # noqa: E402
from namo.planners import get_region_snapshot  # noqa: E402
from eval_common import bin_of  # noqa: E402


def _boundary_objects(edge_objects, source, target):
    """The opener's own rule for "which objects sit on this boundary", byte-for-byte.

    Mirrors BestFirstRegionOpeningPlanner._boundary_objects — inlined rather than imported because
    that module pulls in the scorer/torch stack, which this probe has no use for. Keep in sync.
    """
    forward = edge_objects.get(source, {}).get(target)
    reverse = edge_objects.get(target, {}).get(source)
    if forward is not None and reverse is not None and set(forward) != set(reverse):
        return [], "boundary_object_map_inconsistent"
    return sorted(set(forward if forward is not None else reverse or [])), None


class NoStepEnv:
    """Forwards every RLEnvironment call except step(), which is a hard error.

    The probe is DEFINED as zero simulated pushes; this turns that from a promise into a runtime
    guarantee. If a future edit reaches for the simulator, the run dies instead of quietly
    costing a second of physics per scene.
    """

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        if name == "step":
            raise AssertionError(
                "probe_static_topology performs ZERO simulated pushes; env.step() is forbidden"
            )
        return getattr(self._env, name)


def shortest_region_path(adjacency, src, tgt):
    """One deterministic shortest region path src->tgt, or None if disconnected."""
    if src == tgt:
        return [src]
    if src not in adjacency or not tgt:
        return None
    parent = {src: None}
    frontier = deque([src])
    while frontier:
        node = frontier.popleft()
        for nb in sorted(adjacency.get(node, ())):
            if nb in parent:
                continue
            parent[nb] = node
            if nb == tgt:
                path = [tgt]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            frontier.append(nb)
    return None


def goal_clearance_m(object_info, observation, goal_xy):
    """Distance from the goal point to the nearest movable object's footprint (rough, axis-aligned).

    Diagnostic only: a small clearance means a push can drop the object ON the goal point, which is
    how a scene that is fine at t=0 turns into `goal_region_invalid` mid-episode. Sizes come from
    get_object_info (half-extents, no positions); positions from get_observation ('<name>_pose').
    Uses the larger half-extent as the footprint radius, so it under-reports clearance for
    elongated boxes.
    """
    best = float("inf")
    for name, info in object_info.items():
        if not (name.startswith("obstacle_") and name.endswith("_movable")):
            continue
        pose = observation.get(f"{name}_pose")
        if pose is None or not {"size_x", "size_y"} <= set(info):
            continue
        d = math.dist((float(pose[0]), float(pose[1])), goal_xy)
        best = min(best, d - max(float(info["size_x"]), float(info["size_y"])))
    return None if best == float("inf") else round(best, 4)


def probe_one(args):
    xml, config, expect_hop = args
    row = {"xml_path": xml}
    t0 = time.time()
    try:
        env = NoStepEnv(namo_rl.RLEnvironment(xml, config, False))
        snap = get_region_snapshot(
            env,
            goals_per_region=0,
            local_info_only=False,
            seed=42,
            use_cpp_unified=True,
            use_xml_goal=True,
        )
        robot_label = snap.get("robot_label") or ""
        goal_label = snap.get("goal_label") or ""
        adjacency = snap["adjacency"]
        edge_objects = snap["edge_objects"]

        row["goal_in_free_space"] = bool(snap.get("goal_in_free_space", False))
        row["goal_reachable_at_t0"] = bool(snap.get("goal_reachable", False))
        row["robot_label"] = robot_label
        row["goal_label"] = goal_label
        row["n_regions"] = len(set(snap["region_labels"].values()))

        # Reachability at t=0. set_robot_goal mirrors what the planner does at the top of every
        # iteration; get_reachable_objects then builds the wavefront from the robot's pose.
        goal = extract_goal_from_xml(xml)
        env.set_robot_goal(*goal)
        reachable = set(env.get_reachable_objects())
        row["n_reachable_objects"] = len(reachable)
        row["goal_clearance_m"] = goal_clearance_m(
            env.get_object_info(), env.get_observation(), (goal[0], goal[1])
        )

        path = shortest_region_path(adjacency, robot_label, goal_label) if (robot_label and goal_label) else None
        row["region_path"] = path
        row["hop_count"] = (len(path) - 1) if path else -1
        row["hop_mismatch"] = row["hop_count"] != expect_hop

        boundaries = []
        for src, tgt in zip(path or [], (path or [])[1:]):
            objs, err = _boundary_objects(edge_objects, src, tgt)
            reach = sorted(o for o in objs if o in reachable)
            edges = {o: len(env.get_reachable_edges(o)) for o in reach}
            boundaries.append({
                "source_region": src,
                "target_region": tgt,
                "objects": objs,
                "boundary_error": err,
                "reachable_objects": reach,
                "reachable_edges": edges,
                "no_blocking_objects": not objs,
                "no_reachable_blocker": bool(objs) and not reach,
                "no_pushable_blocker": bool(objs) and not any(edges.values()),
            })
        row["boundaries"] = boundaries

        # Scene-level flags. `*_first` gates the pool (the planner only ever opens boundary 0);
        # `*_any` is kept for diagnosis.
        first = boundaries[0] if boundaries else None
        row["no_blocking_objects"] = bool(first and first["no_blocking_objects"])
        row["no_reachable_blocker"] = bool(first and first["no_reachable_blocker"])
        row["no_pushable_blocker"] = bool(first and first["no_pushable_blocker"])
        row["no_reachable_blocker_any"] = any(b["no_reachable_blocker"] for b in boundaries)
        row["no_pushable_blocker_any"] = any(b["no_pushable_blocker"] for b in boundaries)
        row["no_blocking_objects_any"] = any(b["no_blocking_objects"] for b in boundaries)
        row["no_path"] = path is None
        row["error"] = None
    except Exception as exc:  # one bad XML must not kill a shard
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["t_probe_s"] = round(time.time() - t0, 3)
    return row


# Junk = any static defect that makes the scene unusable as a two-hop region-opening problem.
# goal_in_free_space is inverted here (False is the defect). Note the ~0.6% cost: the deploy planner
# retries a different boundary when the shortest path's first one fails, so a `no_reachable_blocker`
# scene can still be solvable by a re-route (1 of 159 on the aug9 pool).
DROP_RULES = ("error", "no_path", "hop_mismatch", "no_blocking_objects",
              "no_reachable_blocker", "no_pushable_blocker")


def is_junk(row):
    return bool(any(row.get(r) for r in DROP_RULES) or row.get("goal_in_free_space") is False)


def summarize(probe_jsonl, out_dir):
    """Flag census + the surviving / dropped XML lists that feed the next phase."""
    rows = [json.loads(ln) for ln in open(probe_jsonl) if ln.strip()]
    print(f"rows {len(rows)}  errors {sum(1 for r in rows if r.get('error'))}")
    for f in ("goal_in_free_space", "goal_reachable_at_t0", "no_path", "hop_mismatch",
              "no_blocking_objects", "no_reachable_blocker", "no_pushable_blocker",
              "no_blocking_objects_any", "no_reachable_blocker_any", "no_pushable_blocker_any"):
        print(f"  {f:28s} {sum(1 for r in rows if r.get(f) is True):5d}")
    hops = {}
    for r in rows:
        hops[r.get("hop_count")] = hops.get(r.get("hop_count"), 0) + 1
    print("  hop_count:", dict(sorted(hops.items(), key=lambda kv: -kv[1])))
    os.makedirs(out_dir, exist_ok=True)
    keep = sorted(r["xml_path"] for r in rows if not is_junk(r))
    drop = sorted(r["xml_path"] for r in rows if is_junk(r))
    for name, lst in (("surviving_xmls.txt", keep), ("dropped_xmls.txt", drop)):
        with open(os.path.join(out_dir, name), "w") as f:
            f.write("".join(x + "\n" for x in lst))
        print(f"  wrote {os.path.join(out_dir, name)}  ({len(lst)})")


CENSUS_SCHEMA = "boundary_group_census_v1"


def load_two_push_divisions(path):
    """Canonical (realpath, object, region) -> division map; no setup-density re-binning here."""
    divisions = json.load(open(path))
    out = {}
    for xml_path, episodes in divisions.items():
        room = os.path.realpath(str(resolve_namo_path(xml_path)))
        for episode in episodes:
            key = (room, str(episode["object_id"]), episode.get("region"))
            if key in out:
                raise ValueError(f"duplicate canonical 2push division key {key}")
            out[key] = str(episode["division"])
    return out


def load_manifest_sources(onepush_manifest, pure2push_manifest, two_push_divisions=None):
    """Read source episode identities without ever collapsing sibling episodes by XML."""
    two_push_divisions = load_two_push_divisions(two_push_divisions or str(eval_sets.DIVISIONS))
    sources_by_room = defaultdict(list)
    for leg, path in (("1push", onepush_manifest), ("2push", pure2push_manifest)):
        manifest = json.load(open(path))
        if not isinstance(manifest, dict):
            raise ValueError(f"{leg} manifest must be {{xml: [episodes]}}, got {type(manifest).__name__}")
        for xml_path, episodes in manifest.items():
            if not isinstance(episodes, list):
                raise ValueError(f"{leg} manifest {xml_path!r} has non-list episodes")
            resolved = os.path.realpath(str(resolve_namo_path(xml_path)))
            for episode in episodes:
                if not isinstance(episode, dict) or not episode.get("object_id"):
                    raise ValueError(f"{leg} manifest {xml_path!r} has episode without object_id")
                region = episode.get("region")
                if leg == "1push":
                    tier = bin_of(float(episode["solve_rate"]))
                else:
                    division_key = (resolved, str(episode["object_id"]), region)
                    if division_key not in two_push_divisions:
                        raise ValueError(f"2push episode missing canonical division {division_key}")
                    tier = two_push_divisions[division_key]
                sources_by_room[resolved].append({
                    "leg": leg,
                    "xml_path": str(xml_path),
                    "room_realpath": resolved,
                    "object_id": str(episode["object_id"]),
                    "region": region,
                    "tier": tier,
                    "object_center": episode.get("object_center"),
                })
    for room in sources_by_room:
        sources_by_room[room].sort(key=_source_sort_key)
    return dict(sources_by_room)


def _source_sort_key(source):
    return (source["leg"], source["object_id"], str(source.get("tier")),
            str(source.get("region")), json.dumps(source.get("object_center")), source["xml_path"])


def _group_id(room_realpath, robot_label, goal_label, boundary_objects):
    key = [room_realpath, robot_label, goal_label, sorted(boundary_objects)]
    return "bg_" + hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()[:16]


def _stratum(source):
    """Input provenance, deliberately not a difficulty label for a deduped group."""
    return f"{source['leg']}:{source.get('tier') or 'unknown'}"


def group_kind_from_snapshot(snapshot, robot_label, goal_label, boundary_objects):
    """Classify from the explicit marker; object count must never imply joint blockage."""
    assert "multi_object_edges" in snapshot, (
        "snapshot dropped multi_object_edges; refusing to infer joint blockage from object count"
    )
    multi = snapshot["multi_object_edges"]
    marked_forward = goal_label in multi.get(robot_label, set())
    marked_reverse = robot_label in multi.get(goal_label, set())
    if marked_forward != marked_reverse:
        raise ValueError("multi_object_edges is asymmetric for the robot-goal boundary")
    if marked_forward:
        return "joint_blockage"
    return "singleton" if len(boundary_objects) == 1 else "alternatives"


def _json_pose(value):
    return [round(float(x), 6) for x in value] if value is not None else None


def _base_census_record(room, sources, snapshot=None):
    return {
        "schema": CENSUS_SCHEMA,
        "room_realpath": room,
        "source_episodes": sorted(sources, key=_source_sort_key),
        "source_strata": sorted({_stratum(s) for s in sources}),
        "robot_label": (snapshot or {}).get("robot_label") or "",
        "goal_label": (snapshot or {}).get("goal_label") or "",
        "boundary_objects": [],
        "group_kind": None,
        "target_points": [],
        "reachable_objects": [],
        "reachable_edges": {},
        "pushable_boundary_objects": [],
        "initial_target_reachable_count": 0,
        "initial_target_reachable_fraction": None,
        "eligibility_failures": [],
        "initial_robot_pose": None,
        "initial_object_poses": {},
        "eligible": False,
        "exclusion_reason": None,
        "error": None,
    }


def _excluded_record(room, sources, reason, snapshot=None, error=None):
    row = _base_census_record(room, sources, snapshot)
    row["group_id"] = "excluded_" + hashlib.sha256(
        json.dumps([room, reason, [_source_sort_key(s) for s in row["source_episodes"]]],
                   separators=(",", ":")).encode()).hexdigest()[:16]
    row["exclusion_reason"] = reason
    row["error"] = error
    return row


def _target_points(snapshot, goal_label):
    bundle = snapshot.get("region_goals", {}).get(goal_label)
    goals = getattr(bundle, "goals", None) if bundle is not None else None
    return [[float(goal.x), float(goal.y)] for goal in goals or []]


def census_room(task):
    """Census all manifest episodes sharing one resolved room with exactly one static env."""
    room, sources, config = task
    try:
        env = NoStepEnv(namo_rl.RLEnvironment(room, config, False))
        goal = extract_goal_from_xml(room)
        # Match the canonical evaluator's root setup before it samples the fixed target region.
        env.set_robot_goal(*goal)
        reachable = sorted(env.get_reachable_objects())
        snapshot = get_region_snapshot(
            env, goals_per_region=100, local_info_only=False, seed=42,
            use_cpp_unified=True, use_xml_goal=True,
        )
        robot_label = snapshot.get("robot_label") or ""
        goal_label = snapshot.get("goal_label") or ""
        if snapshot.get("goal_reachable", False):
            return [_excluded_record(room, sources, "already_open", snapshot)]
        if not _target_points(snapshot, goal_label):
            return [_excluded_record(room, sources, "empty_goal_samples", snapshot)]
        if not robot_label or not goal_label or goal_label not in snapshot["adjacency"].get(robot_label, set()):
            return [_excluded_record(room, sources, "not_adjacent", snapshot)]
        objects, boundary_error = _boundary_objects(snapshot["edge_objects"], robot_label, goal_label)
        if boundary_error:
            return [_excluded_record(room, sources, boundary_error, snapshot)]
        kind = group_kind_from_snapshot(snapshot, robot_label, goal_label, objects)
        source_on_boundary = [s for s in sources if s["object_id"] in objects]
        rows = []
        for source in sources:
            if source["object_id"] in objects:
                continue
            excluded = _excluded_record(room, [source], "source_object_not_on_boundary", snapshot)
            excluded.update({"boundary_objects": objects, "group_kind": kind,
                             "target_points": _target_points(snapshot, goal_label)})
            rows.append(excluded)
        if not source_on_boundary:
            return rows

        reachable_edges = {obj: len(env.get_reachable_edges(obj)) for obj in reachable}
        target_points = _target_points(snapshot, goal_label)
        target_reachable_count = int(env.count_reachable_points(target_points)[0])
        target_reachable_fraction = target_reachable_count / len(target_points)
        pushable_boundary = [obj for obj in objects
                             if obj in reachable and reachable_edges.get(obj, 0) > 0]
        observation = env.get_observation()
        row = _base_census_record(room, source_on_boundary, snapshot)
        row.update({
            "group_id": _group_id(room, robot_label, goal_label, objects),
            "boundary_objects": objects,
            "group_kind": kind,
            "target_points": target_points,
            "reachable_objects": reachable,
            "reachable_edges": reachable_edges,
            "pushable_boundary_objects": pushable_boundary,
            "initial_target_reachable_count": target_reachable_count,
            "initial_target_reachable_fraction": target_reachable_fraction,
            "initial_robot_pose": _json_pose(observation.get("robot_pose")),
            "initial_object_poses": {
                obj: _json_pose(observation.get(f"{obj}_pose")) for obj in objects
            },
        })
        # The frozen policy-group cohort is multi-object by construction. Singleton rows remain
        # visible in the census, but cannot quietly leak into a group-policy evaluation.
        failures = []
        if len(objects) == 1:
            failures.append("singleton_boundary")
        if not pushable_boundary:
            failures.append("no_reachable_pushable_boundary_object")
        if target_reachable_fraction >= 0.2:
            failures.append("initial_target_fraction_at_least_0_2")
        row["eligibility_failures"] = failures
        row["eligible"] = not failures
        row["exclusion_reason"] = failures[0] if failures else None
        rows.append(row)
        return rows
    except Exception as exc:  # malformed room/config must be an auditable exclusion, never a success
        return [_excluded_record(room, sources, "error", error=f"{type(exc).__name__}: {exc}")]


def select_pilot_groups(rows, limit=24):
    """Round-robin across boundary kind and source leg/tier; no outcomes participate."""
    candidates = sorted((r for r in rows if r.get("eligible")), key=lambda r: r["group_id"])
    by_stratum = defaultdict(list)
    for row in candidates:
        for stratum in row.get("source_strata", []):
            by_stratum[(row.get("group_kind") or "unclassified", stratum)].append(row["group_id"])
    selected = []
    while len(selected) < limit:
        progressed = False
        for stratum in sorted(by_stratum):
            while by_stratum[stratum] and by_stratum[stratum][0] in selected:
                by_stratum[stratum].pop(0)
            if by_stratum[stratum] and len(selected) < limit:
                selected.append(by_stratum[stratum].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def summarize_census(rows, onepush_manifest, pure2push_manifest, two_push_divisions):
    eligible = [r for r in rows if r.get("eligible")]
    exclusion_counts = Counter(r.get("exclusion_reason") for r in rows if not r.get("eligible"))
    source_strata = Counter(
        stratum for row in eligible for stratum in row.get("source_strata", [])
    )
    return {
        "schema": CENSUS_SCHEMA,
        "onepush_manifest": os.path.abspath(onepush_manifest),
        "pure2push_manifest": os.path.abspath(pure2push_manifest),
        "two_push_divisions": os.path.abspath(two_push_divisions),
        "n_records": len(rows),
        "n_eligible_multi_object_groups": len(eligible),
        "eligible_group_ids": [r["group_id"] for r in sorted(eligible, key=lambda r: r["group_id"])],
        "group_kind_counts": dict(sorted(Counter(r.get("group_kind") or "unclassified" for r in rows).items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "eligible_source_strata_counts": dict(sorted(source_strata.items())),
        "pilot_group_ids": select_pilot_groups(rows),
        "pilot_selection": "deterministic round-robin over boundary kind and source leg:tier strata; source strata are provenance, not group difficulty; no outcome fields were selected on",
    }


def run_group_census(onepush_manifest, pure2push_manifest, out_path, config, start, end, workers,
                     two_push_divisions):
    if not config:
        raise ValueError("--config is required with --episode-manifests")
    _require_census_margin(config)
    sources_by_room = load_manifest_sources(onepush_manifest, pure2push_manifest, two_push_divisions)
    rooms = sorted(sources_by_room)
    rooms = rooms[start:(end if end is not None else len(rooms))]
    tasks = [(room, sources_by_room[room], config) for room in rooms]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    t0 = time.time()
    rows = []
    if workers > 1:
        with Pool(workers) as pool:
            for room_rows in pool.imap_unordered(census_room, tasks, chunksize=1):
                rows.extend(room_rows)
    else:
        for room_rows in map(census_room, tasks):
            rows.extend(room_rows)
    with open(out_path, "w") as f:
        # Keep the JSONL stable even when static rooms were probed concurrently.
        for row in sorted(rows, key=lambda r: r["group_id"]):
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize_census(rows, onepush_manifest, pure2push_manifest, two_push_divisions)
    summary_path = out_path + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    pilot_path = out_path + ".pilot.jsonl"
    chosen = set(summary["pilot_group_ids"])
    with open(pilot_path, "w") as f:
        for row in sorted((r for r in rows if r.get("group_id") in chosen), key=lambda r: r["group_id"]):
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"done {len(rows)} census records from {len(rooms)} rooms -> {out_path} in {time.time()-t0:.0f}s", flush=True)
    print(f"summary -> {summary_path}", flush=True)
    print(f"pilot -> {pilot_path}", flush=True)


def _require_census_margin(config):
    """The 5 mm cohort is config-bound; never silently census it under the later 1 mm geometry."""
    config_path = os.path.abspath(config)
    margin_path = os.path.join(os.path.dirname(config_path), "wavefront_inflation.yaml")
    if not os.path.isfile(margin_path):
        raise ValueError(f"census config {config_path} has no sibling wavefront_inflation.yaml")
    margin = yaml.safe_load(open(margin_path)).get("tier1", {}).get("base_inflation_margin_m")
    if float(margin) != 0.005:
        raise ValueError(
            f"census requires 5 mm tier1 margin, got {margin!r} from {margin_path}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", metavar="PROBE_JSONL",
                    help="skip probing; reduce an existing probe JSONL into the flag census + "
                         "surviving/dropped XML lists (written next to --out)")
    ap.add_argument("--manifest", help="file of XML paths, one per line")
    ap.add_argument("--episode-manifests", nargs=2, metavar=("ONEPUSH_JSON", "PURE2PUSH_JSON"),
                    help="run the bounded group census from explicit canonical 1-push and pure-2-push "
                         "episode manifests; writes --out, --out.summary.json, and --out.pilot.jsonl")
    ap.add_argument("--two-push-divisions", default=str(eval_sets.DIVISIONS),
                    help="canonical pure-2-push division JSON; joined by realpath, object, and region")
    ap.add_argument("--out", required=True, help="output JSONL, one row per XML")
    ap.add_argument("--config", help="namo config YAML")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--expect-hop", type=int, default=2, help="hop_mismatch = recomputed hop != this")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    if a.summarize:
        summarize(a.summarize, os.path.dirname(os.path.abspath(a.out)))
        return

    if a.episode_manifests:
        if a.manifest:
            ap.error("--manifest and --episode-manifests are mutually exclusive")
        run_group_census(*a.episode_manifests, a.out, a.config, a.start, a.end, a.workers,
                         a.two_push_divisions)
        return

    if not a.manifest:
        ap.error("--manifest is required unless --summarize or --episode-manifests is used")

    xmls = [ln.strip() for ln in open(a.manifest) if ln.strip()]
    xmls = xmls[a.start:(a.end if a.end is not None else len(xmls))]
    tasks = [(x, a.config, a.expect_hop) for x in xmls]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    t0 = time.time()
    with open(a.out, "w") as f:
        if a.workers > 1:
            with Pool(a.workers) as pool:
                for i, row in enumerate(pool.imap_unordered(probe_one, tasks, chunksize=4), 1):
                    f.write(json.dumps(row) + "\n")
                    if i % 100 == 0:
                        f.flush()
                        print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
        else:
            for i, t in enumerate(tasks, 1):
                f.write(json.dumps(probe_one(t)) + "\n")
                if i % 25 == 0:
                    f.flush()
                    print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(tasks)} rows -> {a.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
