#!/usr/bin/env python3
"""Materialize the state after one keyhole is opened, as a standalone XML, so the NEXT keyhole can be labelled.

One round of an N-hop chain. Run it once per keyhole: round k reads the XMLs round k-1 emitted, opens
that scene's current first boundary, and writes the resulting state out as a fresh scene. A 2-hop scene
needs 1 round, a 3-hop scene 2, a 4-hop scene 3.

The problem this solves. `region_opening._explore_from_state` only ever sweeps `adjacency[robot_label]`,
so a boundary between two NON-robot regions is never swept. Every keyhole past the first is exactly such
a boundary, and its blocker is unreachable at t=0 in the overwhelming majority of scenes. So keyhole k>1
has no label computable from the original XML. Opening keyhole k-1 merges the robot region with the next
one along the path, which makes keyhole k an ordinary robot-adjacent boundary — and a scene written out
at that state is an ordinary region-opening problem the existing collection already handles.

Which opener. A scene has many valid openers and each leaves a DIFFERENT state, so the next keyhole is
undefined until one opener is fixed. Convention, applied independently at every round: take the
lexicographically smallest `(n_pushes, object_id, (edge_idx, depth) per push)` — shortest chain first,
then lexicographic. Deterministic, reproducible, model-free and seed-free; these labels must not depend
on the ranker being evaluated. `object_id` is in the key because a boundary is an OR over several
blocking objects, so `(edge, depth)` alone does not identify a push.

Two facts force a refinement, both measured (uniform 300-scene sample of the 2-hop pool):
  * Candidates come from the exhaustive `primitive_trial_log`, NOT from the planner's recorded
    solutions. The planner filters those to MINIMUM push cost (region_opening.py:2578), which on one
    scene left 9 of 57 valid openers visible — the convention would silently have meant "cheapest".
  * An opener that passes the 20%-of-100-points test does not always ADVANCE the scene: the pushed
    object can land inside the region it just opened and split it, leaving the goal as far away as
    before (167 of 499 openers) or disconnecting it (54 of 499). The canonical opener is therefore the
    first candidate in the order above whose emitted XML verifies with the hop count reduced by one.

No replay. Every candidate's post-push state comes from the sweep itself — `resulting_state` on the
depth-1 trial-log entries, or `AttemptResult.resulting_state` for chains — so no push is ever
re-executed and the known replay divergence on collision pushes cannot occur.

Collision checking. The sweep config sets `region_allow_collisions: true`, so the planner calls
`env.set_collision_checking(False)` for the whole sweep. This script never steps the sim outside the
planner, so it inherits that setting rather than re-deriving it.

Verification, per scene, recorded in the output row — nothing is assumed:
  * SE(2) round trip: reload the emitted XML in a fresh env and compare every movable AND the robot to
    the state it was written from. A writer that drops the car's yaw is the exact silent failure mode.
  * region graph: the emitted scene's shortest robot->goal region path must be one hop shorter, and its
    next boundary's blocking-object set is compared with the input scene's.

  python scripts/pipeline/materialize_keyhole2.py \
      --manifest surviving_xmls_cspaths.txt --out-root <root>/round1 --workers 96 \
      --kh1-chain-depth 2 --kh1-timeout 1800
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque
from multiprocessing import Pool

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(REPO, "build_python"), os.path.join(REPO, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import namo_rl  # noqa: E402
from namo.core.base_planner import PlannerConfig  # noqa: E402
from namo.core.state_to_xml import write_state_xml  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_from_xml  # noqa: E402
from namo.planners import get_region_snapshot  # noqa: E402
from namo.planners.opening.region_opening import RegionOpeningPlanner  # noqa: E402
from namo.rl_loop.build_train_h5 import _rlstate  # noqa: E402  (qpos/qvel lists -> namo_rl.RLState)

# Algorithm-params keys the planner reads, and the YAML key each comes from. Mirrors
# modular_parallel_collection.main()'s algorithm_params dict for the keys this sweep uses; anything not
# listed keeps the planner's own default.
_ALGO_KEYS = (
    "region_allow_collisions", "region_max_chain_depth", "region_max_solutions_per_neighbor",
    "region_max_recorded_solutions_per_neighbor", "region_chain_link_cost",
    "region_min_reachable_fraction", "region_frontier_beam_width", "region_ml_ignore_blacklist",
    "region_selection_strategy", "region_exhaustive_mode", "region_label_mode", "region_sample_k",
    "region_sample_restarts", "region_timeout_per_neighbour_sec", "primitive_prefix",
    "target_goal_region", "shuffle_edges", "shuffle_seed",
)


def shortest_region_path(adjacency, src, tgt):
    """One deterministic shortest region path src->tgt, or None. Same rule as probe_static_topology."""
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


def boundary_objects(edge_objects, source, target):
    """The opener's own rule for "which objects sit on this boundary" (see probe_static_topology)."""
    forward = edge_objects.get(source, {}).get(target)
    reverse = edge_objects.get(target, {}).get(source)
    if forward is not None and reverse is not None and set(forward) != set(reverse):
        return []
    return sorted(set(forward if forward is not None else reverse or []))


def _dtheta(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def build_planner_config(algo_yaml, chain_depth, seed=42, timeout_per_neighbour=None):
    cfg = yaml.safe_load(open(algo_yaml))
    params = {k: cfg[k] for k in _ALGO_KEYS if k in cfg}
    params["primitive_data_dir"] = cfg.get("primitive_data_dir", "data")
    params["region_max_chain_depth"] = chain_depth
    if timeout_per_neighbour is not None:
        params["region_timeout_per_neighbour_sec"] = float(timeout_per_neighbour)
    return PlannerConfig(
        max_depth=cfg.get("max_depth", 5),
        max_goals_per_object=cfg.get("max_goals_per_object", 5),
        max_terminal_checks=cfg.get("max_terminal_checks", 50000),
        max_search_time_seconds=cfg.get("search_timeout", 1800.0),
        goals_per_region=cfg.get("goals_per_region", 100),
        random_seed=seed,
        verbose=False,
        algorithm_params=params,
    ), cfg


def opener_key(attempt):
    """Sort key of a successful AttemptResult: (n_pushes, object_id, (edge,depth) per push).

    Sorting on this gives SHORTEST chain first, then lexicographic — so a 1-push opener always beats a
    2-push one, and among equals the smallest (object_id, edge_idx, depth) wins. A region-opening chain
    pushes ONE object, so object_id is a scalar. Returns None for an attempt with no executed push
    (`already_accessible`), which is not an opener.
    """
    goals = attempt.goal_chain or []
    if not goals or attempt.chosen_object_id is None:
        return None
    cells = []
    for g in goals:
        edge, depth = getattr(g, "edge_idx", None), getattr(g, "depth", None)
        if edge is None or depth is None:
            return None
        cells.append((int(edge), int(depth)))
    return (len(cells), attempt.chosen_object_id, tuple(cells))


def _jkey(k):
    """JSON-safe form of an opener key: [n_pushes, object_id, [[edge, depth], ...]]."""
    return [k[0], k[1], [list(c) for c in k[2]]]


def trial_log_openers(attempts):
    """Ordered [(key, RLState)] for EVERY depth-1 push the exhaustive sweep found to open the boundary.

    Straight off `primitive_trial_log`, which is the complete sweep record and carries the post-push
    `resulting_state` on depth-1 entries when max_chain_depth >= 2 (region_opening.py:3103). The
    per-object log is identical across that object's AttemptResults, hence the dedup by object.
    """
    out, seen = [], set()
    for a in attempts:
        obj = a.chosen_object_id
        if obj is None or obj in seen:
            continue
        seen.add(obj)
        for t in a.primitive_trial_log or []:
            rs = t.get("resulting_state")
            if t.get("chain_depth") != 1 or not t.get("success") or not rs:
                continue
            key = (1, obj, ((int(t["edge_idx"]), int(t["depth"])),))
            out.append((key, _rlstate(rs["qpos"], rs["qvel"])))
    out.sort(key=lambda kv: kv[0])
    return out


def keyhole1_key(attempts, max_chain_depth):
    """Per-object exhaustive answer key from the trial logs, on the canonical scale.

      solve_rate_1push       = depth-1 cells that OPEN / depth-1 cells TRIED           (F)
      solve_rate_first_push  = distinct first-pushes ENABLING a depth-2 solve / expanded  (F1')

    Same derivation as build_2push_validset.py, so these join to the project's hard/med/easy bins.
    """
    key = {}
    for a in attempts:
        obj = a.chosen_object_id
        if obj is None or obj in key:
            continue                       # the trial log is object-level; identical on every attempt
        log = a.primitive_trial_log or []
        tried = {(t["edge_idx"], t["depth"]) for t in log if t.get("chain_depth") == 1}
        valid = {(t["edge_idx"], t["depth"]) for t in log if t.get("chain_depth") == 1 and t.get("success")}
        if not tried:
            continue
        rec = {"tried": len(tried), "valid": len(valid), "solve_rate_1push": len(valid) / len(tried),
               "timed_out": bool(getattr(a, "neighbour_timed_out", False))}
        if max_chain_depth >= 2:
            tried_fp = {(t["parent_edge"], t["parent_depth"]) for t in log
                        if t.get("chain_depth") == 2 and t.get("parent_edge") is not None}
            valid_fp = {(t["parent_edge"], t["parent_depth"]) for t in log
                        if t.get("chain_depth") == 2 and t.get("parent_edge") is not None and t.get("success")}
            rec.update(tried_first_push=len(tried_fp), valid_first_push=len(valid_fp),
                       solve_rate_first_push=(len(valid_fp) / len(tried_fp)) if tried_fp else None)
        key[obj] = rec
    return key


def process_one(args):
    xml, cfg_file, algo_yaml, out_root, seed, kh1_chain_depth, kh1_timeout = args
    row = {"xml_path": xml}
    t0 = time.time()
    try:
        env = namo_rl.RLEnvironment(xml, cfg_file, False)
        goal = extract_goal_from_xml(xml)
        env.set_robot_goal(*goal)

        # goal_radius stays at the collection default (None). Passing a number moves the goal into the
        # robot's own region in ~11% of rooms and silently breaks the manifest join.
        snap = get_region_snapshot(env, goals_per_region=0, goal_radius=None, local_info_only=False,
                                   seed=seed, use_cpp_unified=True, use_xml_goal=True)
        robot_label, goal_label = snap.get("robot_label") or "", snap.get("goal_label") or ""
        path = shortest_region_path(snap["adjacency"], robot_label, goal_label)
        row["region_path"] = path
        row["hop_count"] = (len(path) - 1) if path else -1
        if not path or len(path) < 2:
            row["status"] = "no_region_path"
            return _done(row, t0)
        hop_in = len(path) - 1
        kh1_target = path[1]
        row["kh1_boundary_objects"] = boundary_objects(snap["edge_objects"], robot_label, kh1_target)
        row["next_boundary_objects"] = (boundary_objects(snap["edge_objects"], path[1], path[2])
                                        if len(path) >= 3 else [])

        # ---- keyhole 1: exhaustive sweep of the FIRST boundary only ----
        pcfg, _ = build_planner_config(algo_yaml, chain_depth=kh1_chain_depth, seed=seed,
                                       timeout_per_neighbour=kh1_timeout)
        planner = RegionOpeningPlanner(env, pcfg)
        result = planner.search(goal, target_neighbor=kh1_target)
        attempts = (result.algorithm_stats or {}).get("attempt_results") or []
        row["kh1_key"] = keyhole1_key(attempts, kh1_chain_depth)
        row["kh1_pushes"] = int((result.algorithm_stats or {}).get("total_primitives_attempted", 0))
        row["kh1_timed_out"] = any(getattr(a, "neighbour_timed_out", False) for a in attempts)
        row["kh1_failure_reasons"] = sorted({a.failure_reason for a in attempts if a.failure_reason})

        # Ordered candidate openers, shortest chain first then lexicographic. Two sources:
        #  (A) every depth-1 cell the exhaustive sweep found to OPEN, read straight off the trial log —
        #      the COMPLETE set. The planner's own recorded solutions are filtered to MINIMUM push cost
        #      (region_opening.py:2578: `min_cost_chains`), so choosing from them would silently narrow
        #      the convention to "cheapest opener" (9 of 57 valid cells on one measured scene). The trial
        #      log carries `resulting_state` on every depth-1 push whenever max_chain_depth >= 2, so the
        #      opener is never re-executed and replay divergence cannot occur.
        #  (B) recorded multi-push chains, used only when (A) is empty — which is most scenes, since
        #      ~70% have no 1-push keyhole-1 opener at all. These come from the min-cost set.
        cands = trial_log_openers(attempts) if kh1_chain_depth >= 2 else []
        row["kh1_openers_1push"] = len(cands)
        if not cands:
            cands = [(k, a.resulting_state) for k, a in
                     sorted((k, a) for k, a in ((opener_key(a), a) for a in attempts if a.success)
                            if k is not None)
                     if a.resulting_state is not None]
        row["kh1_candidate_openers"] = len(cands)
        if not cands:
            row["status"] = "no_kh1_opener"
            return _done(row, t0)
        row["lex_min_opener"] = _jkey(cands[0][0])

        # An opener that passes the 20%-of-100-points test does NOT always leave the goal one hop away:
        # the pushed object can end up inside the middle region and split it, so the piece touching the
        # goal becomes a NEW region and the scene stays two-hop (measured: 167 of 499 openers leave 2
        # hops, 54 disconnect the goal entirely). Keyhole 2 only exists at a one-hop post-opener state,
        # so the canonical opener is the FIRST candidate in the order above whose emitted XML verifies
        # as one hop. The emitted XML is the authority, not the live state — they disagreed on 1 of 47
        # scenes, where a 32 um difference flipped a wavefront cell.
        out_xml = os.path.join(out_root, "xmls", _scene_id(xml) + ".xml")
        rejected = []
        accepted = False
        fallback = None                      # first candidate overall, materialized only if none advances
        for k, st in cands:
            env.set_full_state(st)
            s = get_region_snapshot(env, goals_per_region=0, goal_radius=None, local_info_only=False,
                                    seed=seed, use_cpp_unified=True, use_xml_goal=True)
            p = shortest_region_path(s["adjacency"], s.get("robot_label") or "", s.get("goal_label") or "")
            hop = (len(p) - 1) if p else -1
            if hop != hop_in - 1:
                rejected.append([_jkey(k), hop, "live"])
                if fallback is None and hop >= 1:
                    fallback = (k, st, hop)   # opened a boundary but did not shorten the path
                continue

            src_obs = {n: [float(v[0]), float(v[1]), float(v[2])] for n, v in env.get_observation().items()}
            write_state_xml(src_obs, xml, out_xml)
            env2 = namo_rl.RLEnvironment(out_xml, cfg_file, False)
            env2.set_robot_goal(*goal)
            obs2 = env2.get_observation()
            snap2 = get_region_snapshot(env2, goals_per_region=0, goal_radius=None, local_info_only=False,
                                        seed=seed, use_cpp_unified=True, use_xml_goal=True)
            rl2, gl2 = snap2.get("robot_label") or "", snap2.get("goal_label") or ""
            path2 = shortest_region_path(snap2["adjacency"], rl2, gl2)
            hop2 = (len(path2) - 1) if path2 else -1
            if hop2 != hop_in - 1:
                rejected.append([_jkey(k), hop2, "xml"])
                continue

            dxy = {n: math.dist(src_obs[n][:2], obs2[n][:2]) for n in src_obs if n in obs2}
            dth = {n: _dtheta(src_obs[n][2], obs2[n][2]) for n in src_obs if n in obs2}
            objs2 = boundary_objects(snap2["edge_objects"], rl2, path2[1])
            reach2 = set(env2.get_reachable_objects())
            row.update(
                canonical_opener=_jkey(k),
                canonical_is_lex_min=(_jkey(k) == row["lex_min_opener"]),
                out_xml=out_xml,
                missing_bodies=sorted(set(src_obs) - set(obs2)),
                max_dxy_mm=round(1000.0 * max(dxy.values()), 4),
                max_dtheta_deg=round(math.degrees(max(dth.values())), 4),
                robot_dxy_mm=round(1000.0 * dxy["robot_pose"], 4),
                robot_dtheta_deg=round(math.degrees(dth["robot_pose"]), 4),
                post_region_path=path2,
                post_hop_count=hop2,
                post_goal_in_free_space=bool(snap2.get("goal_in_free_space", False)),
                post_n_regions=len(set(snap2["region_labels"].values())),
                post_next_boundary_objects=objs2,
                next_boundary_matches=(objs2 == row["next_boundary_objects"]),
                post_next_reachable_objects=sorted(o for o in objs2 if o in reach2),
                status="ok",
            )
            accepted = True
            break
        row["rejected_openers"] = rejected
        if not accepted:
            if os.path.exists(out_xml):
                os.remove(out_xml)          # the last write was a REJECTED state; never leave it behind
            row["status"] = "no_opener_decrements_hop"
            # Also emit the non-advancing opener under a SEPARATE root. Whether "the scene still has as
            # many hops to go" should end the chain or merely count as one more keyhole is a definition
            # call, and materializing both here means that call can be made without re-running the sweep.
            if fallback is not None:
                k, st, hop = fallback
                env.set_full_state(st)
                alt_xml = os.path.join(out_root, "xmls_nodecrement", _scene_id(xml) + ".xml")
                src_obs = {n: [float(v[0]), float(v[1]), float(v[2])]
                           for n, v in env.get_observation().items()}
                write_state_xml(src_obs, xml, alt_xml)
                env2 = namo_rl.RLEnvironment(alt_xml, cfg_file, False)
                env2.set_robot_goal(*goal)
                snap2 = get_region_snapshot(env2, goals_per_region=0, goal_radius=None,
                                            local_info_only=False, seed=seed, use_cpp_unified=True,
                                            use_xml_goal=True)
                p2 = shortest_region_path(snap2["adjacency"], snap2.get("robot_label") or "",
                                          snap2.get("goal_label") or "")
                row.update(nodecrement_opener=_jkey(k), nodecrement_out_xml=alt_xml,
                           nodecrement_hop_count=(len(p2) - 1) if p2 else -1)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return _done(row, t0)


def _done(row, t0):
    row["t_s"] = round(time.time() - t0, 2)
    return row


def _scene_id(xml):
    """Stable, collision-free id. Round 1 flattens the pool-relative path; later rounds inherit the id
    that is already the emitted file's basename, so a scene keeps ONE identity down the whole chain."""
    p = os.path.realpath(xml)
    parts = p.split(os.sep)
    if len(parts) >= 2 and parts[-2] == "xmls":
        return parts[-1][:-4] if parts[-1].endswith(".xml") else parts[-1]
    tail = parts[-4:] if len(parts) >= 4 else parts
    return "__".join(tail).replace(".xml", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="file of scene XML paths, one per line")
    ap.add_argument("--out-root", required=True, help="root for emitted XMLs + rows.jsonl")
    ap.add_argument("--algo-yaml",
                    default=os.path.join(REPO, "python/namo/data_collection/"
                                               "region_opening_exhaustive_2push_multihop_car.yaml"))
    ap.add_argument("--config", default=os.path.join(REPO, "config/namo_config_complete_skill15_car_1x.yaml"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", default="rows.jsonl")
    ap.add_argument("--kh1-chain-depth", type=int, default=1,
                    help="1 = only 1-push keyhole-1 openers (cheap); 2 = also exhaust 2-push chains "
                         "(needed for the ~70%% of scenes with no 1-push opener)")
    ap.add_argument("--kh1-timeout", type=float, default=None,
                    help="override region_timeout_per_neighbour_sec for the keyhole-1 sweep")
    a = ap.parse_args()

    xmls = [ln.strip() for ln in open(a.manifest) if ln.strip()]
    xmls = xmls[a.start:(a.end if a.end is not None else len(xmls))]
    os.makedirs(a.out_root, exist_ok=True)
    out_path = os.path.join(a.out_root, a.out_name)
    tasks = [(x, a.config, a.algo_yaml, a.out_root, a.seed, a.kh1_chain_depth, a.kh1_timeout)
             for x in xmls]

    t0 = time.time()
    with open(out_path, "w") as f:
        if a.workers > 1:
            with Pool(a.workers) as pool:
                for i, row in enumerate(pool.imap_unordered(process_one, tasks, chunksize=1), 1):
                    f.write(json.dumps(row) + "\n")
                    if i % 20 == 0:
                        f.flush()
                        print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
        else:
            for i, t in enumerate(tasks, 1):
                f.write(json.dumps(process_one(t)) + "\n")
                f.flush()
                print(f"{i}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(tasks)} rows -> {out_path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
