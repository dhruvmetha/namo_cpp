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

Boundary object lists are COUNTERFACTUAL CERTIFICATES (see mujoco_env_creator/generate_envs.py
::_runtime_topology): the wavefront removes each listed object independently while every other
object stays, so a two-name list is an OR boundary — either object alone opens it. That is why
`no_reachable_blocker` is "ALL of them are unreachable", not "any of them is".

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
import json
import math
import os
import sys
import time
from collections import deque
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(REPO, "build_python"), os.path.join(REPO, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import namo_rl  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_from_xml  # noqa: E402
from namo.planners import get_region_snapshot  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", metavar="PROBE_JSONL",
                    help="skip probing; reduce an existing probe JSONL into the flag census + "
                         "surviving/dropped XML lists (written next to --out)")
    ap.add_argument("--manifest", help="file of XML paths, one per line")
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
