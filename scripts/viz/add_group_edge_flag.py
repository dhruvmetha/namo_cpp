#!/usr/bin/env python3
"""Mark every card whose robot-to-goal doorway needs BOTH movables, and whether a way around exists.

Run AFTER build_scene_cards.py and before --index-only; a card rebuild wipes these fields.

    python scripts/viz/add_group_edge_flag.py --out $NAMO_SCRATCH/viz/real_2mov [--shard i --nshards n]

`build_region_connectivity_graph` writes an edge for a doorway no single object opens, and records
that pair in `multi_object_edges` as well as in `adjacency` (wavefront_grid.cpp:920-927). So the
single-object graph is adjacency minus those pairs, and a goal unreachable in it has no route that
avoids a two-block door.

Measured on the 2220-card pool: 376 rooms have such a boundary between robot and goal, and 240 of
them have no way around it. Running best-first over both, budget 900, those 240 exhaust the budget
13.8% of the time against 2.9% for the 136 that do have an alternative route, a 5x gap on scenes
that are otherwise the same shape. Worth seeing on the card rather than discovering in an aggregate.
"""
import argparse
import collections
import glob
import json
import os


def reachable(graph, start, goal):
    seen, q = {start}, collections.deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            return True
        for v in graph.get(u, ()):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="gallery data root holding cards/")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--config", default="config/namo_config_complete_skill15_car_1x.yaml")
    a = ap.parse_args()

    import namo_rl

    rooms = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(a.out, "cards", "*.json"))):
        rooms[json.load(open(p))["meta"]["xml"]].append(p)

    done = 0
    for n, (xml, paths) in enumerate(sorted(rooms.items())):
        if n % a.nshards != a.shard:
            continue
        try:
            env = namo_rl.RLEnvironment(xml, a.config, False)
            env.get_reachable_objects()
            snap = env.get_region_snapshot(100, -1.0, False, 42, True)
        except Exception:
            continue
        adj = {k: set(v) for k, v in dict(snap.get("adjacency", {})).items()}
        grp = {k: set(v) for k, v in dict(snap.get("multi_object_edges", {})).items()}
        rl, gl = snap.get("robot_label"), snap.get("goal_label")
        if not rl or not gl:
            continue
        solo = {k: (v - grp.get(k, set())) for k, v in adj.items()}
        fields = {
            "door_needs_both_blocks": gl in grp.get(rl, set()),
            "has_route_around": reachable(solo, rl, gl),
        }
        for p in paths:
            card = json.load(open(p))
            card["meta"].update(fields)
            with open(p, "w") as fh:
                json.dump(card, fh, separators=(",", ":"))
        done += len(paths)
    print(f"shard {a.shard}: DONE {done} cards", flush=True)


if __name__ == "__main__":
    main()
