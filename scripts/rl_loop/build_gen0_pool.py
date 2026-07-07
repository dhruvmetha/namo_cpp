#!/usr/bin/env python3
"""Build the gen-0 rollout pool from a v4_hq h1 validset manifest + run the disjointness gate.

The pool is a mixed-difficulty SUBSET of an `episodes_deadends*.json` manifest, keyed by the
full per-episode xml path — exactly the validset format `namo.rl_loop.episodes.load_pool` reads
(object_id / object_center / solve_rate / valid / tried). Difficulty is per-episode via bin_of
(hard<0.05, med<0.30, else easy), NEVER a file label (per the pipeline invariant).

Rooms: the manifest keys are PER-PAIR xmls (run_NNNN_env_NNNN_pair_*.xml); many pairs share one
STATIC room (run_NNNN_env_NNNN). We tag each episode with its base room so the downstream split
(build_split_grouped.py) holds out by ROOM, not per-pair — otherwise the same static geometry
leaks across train/dev/test.

Disjointness gate (hard): drop any pool base room that overlaps a namo_testset_v1 room by
  (a) base-room NAME, or
  (b) translation-invariant static-wall geometry hash (catches the same room re-exported/renamed).
Both counts are logged to the gate report. Basenames can collide across generation trees, so the
geometry hash is the source of truth; the name match is a conservative backstop.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from eval_common import bin_of                 # noqa: E402  hard<0.05, med<0.30, else easy


_RUN_ENV = re.compile(r"^(run_\d+_env_\d+)")
_ENV = re.compile(r"^(env_\d+)")


def base_room(xml_path: str) -> str:
    """Static-room id for holdout grouping. Handles the several on-disk naming schemes:
      run_NNNN_env_NNNN_pair_*.xml     -> run_NNNN_env_NNNN
      env_NNNN_pair_*.xml              -> env_NNNN
      .../run_NNNN/env_NNNN_pair_*.xml -> run_NNNN (parent dir carries the room id)
    """
    b = os.path.basename(xml_path)
    m = _RUN_ENV.match(b)
    if m:
        return m.group(1)
    parent = os.path.basename(os.path.dirname(xml_path))
    if _RUN_ENV.match(parent) or _ENV.match(parent) or parent.startswith("run_"):
        # dir names the room; combine with the env id in the file for uniqueness
        m2 = _ENV.match(b)
        return f"{parent}/{m2.group(1)}" if m2 else parent
    m3 = _ENV.match(b)
    if m3:
        return m3.group(1)
    return b


def geom_hash(xml_path: str, resolver) -> str:
    """Translation-invariant hash of the static wall layout. None if unreadable."""
    p = str(resolver(xml_path))
    if not os.path.exists(p):
        return None
    try:
        root = ET.parse(p).getroot()
    except Exception:
        return None
    walls = None
    for body in root.iter("body"):
        if body.get("name") == "walls":
            walls = body
            break
    if walls is None:
        return None
    geoms = []
    xs, ys = [], []
    for g in walls.findall("geom"):
        pos = [round(float(v), 4) for v in (g.get("pos", "0 0 0")).split()]
        size = tuple(round(float(v), 4) for v in (g.get("size", "0 0 0")).split())
        eul = g.get("euler", g.get("quat", "0 0 0"))
        eul = tuple(round(float(v), 3) for v in eul.split())
        gtype = g.get("type", "box")
        geoms.append((gtype, tuple(pos), size, eul))
        if len(pos) >= 2:
            xs.append(pos[0]); ys.append(pos[1])
    if not geoms:
        return None
    # translation-invariant: subtract the wall centroid before hashing positions
    cx = round(sum(xs) / len(xs), 4) if xs else 0.0
    cy = round(sum(ys) / len(ys), 4) if ys else 0.0
    canon = []
    for gtype, pos, size, eul in geoms:
        pp = list(pos)
        if len(pp) >= 2:
            pp[0] = round(pp[0] - cx, 4); pp[1] = round(pp[1] - cy, 4)
        canon.append((gtype, tuple(pp), size, eul))
    canon.sort()
    return hashlib.sha1(json.dumps(canon).encode()).hexdigest()


def testset_rooms(labels_dir: str, resolver):
    """Collect (base-room names, geom hashes) across every label json in namo_testset_v1/labels."""
    names, hashes = set(), set()
    rep_by_room = {}   # base_room -> one xml key (for hashing, once per room)
    n_keys = 0
    for jf in sorted(Path(labels_dir).glob("*.json")):
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for k in d.keys():
            if not (isinstance(k, str) and k.endswith(".xml")):
                continue
            n_keys += 1
            br = base_room(k)
            names.add(br)
            rep_by_room.setdefault(br, k)
    n_missing = 0
    for br, k in rep_by_room.items():
        h = geom_hash(k, resolver)
        if h:
            hashes.add(h)
        else:
            n_missing += 1
    return names, hashes, {"label_files": [str(p.name) for p in sorted(Path(labels_dir).glob("*.json"))],
                           "n_keys": n_keys, "n_base_rooms": len(names),
                           "n_geom_hashes": len(hashes), "n_unhashable_rooms": n_missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="episodes_deadends*.json (h1 validset)")
    ap.add_argument("--testset-labels-dir", required=True, help="namo_testset_v1/labels")
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--out-gate-report", required=True)
    ap.add_argument("--per-room-per-diff", type=int, default=2, help="max episodes per (room, difficulty)")
    ap.add_argument("--target", type=int, default=5000, help="cap on total pool episodes")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    # resolver: map legacy /scratch prefix onto this box (no-op on Amarel)
    from namo.paths import resolve as _resolve

    rng = random.Random(a.seed)
    manifest = json.load(open(a.manifest))
    print(f"[pool] manifest keys (per-pair xmls): {len(manifest)}", flush=True)

    # --- testset rooms for the gate ---
    ts_names, ts_hashes, ts_info = testset_rooms(a.testset_labels_dir, _resolve)
    print(f"[gate] testset: {ts_info}", flush=True)

    # --- group pool episodes by base room; tag difficulty ---
    by_room = defaultdict(list)   # base_room -> [(xml_key, record, difficulty)]
    for xml_key, recs in manifest.items():
        br = base_room(xml_key)
        for r in recs:
            sr = r.get("solve_rate")
            diff = bin_of(sr) if sr is not None else "med"
            by_room[br].append((xml_key, r, diff))
    print(f"[pool] distinct base rooms in manifest: {len(by_room)}", flush=True)

    # --- gate: drop rooms overlapping the testset by name OR geometry hash ---
    dropped_name, dropped_geom, kept_rooms = [], [], []
    for br in by_room:
        if br in ts_names:
            dropped_name.append(br); continue
        rep = by_room[br][0][0]
        h = geom_hash(rep, _resolve)
        if h is not None and h in ts_hashes:
            dropped_geom.append(br); continue
        kept_rooms.append(br)
    print(f"[gate] pool rooms dropped by NAME overlap: {len(dropped_name)}", flush=True)
    print(f"[gate] pool rooms dropped by GEOMETRY overlap: {len(dropped_geom)}", flush=True)
    print(f"[gate] pool rooms kept: {len(kept_rooms)}", flush=True)

    # --- mixed-difficulty stratified sample from kept rooms ---
    pool = defaultdict(list)      # xml_key -> [records]  (validset format)
    counts = defaultdict(int)
    room_used = set()
    selected = []
    for br in kept_rooms:
        buckets = defaultdict(list)
        for xml_key, r, diff in by_room[br]:
            buckets[diff].append((xml_key, r, diff))
        for diff, items in buckets.items():
            rng.shuffle(items)
            selected.extend(items[:a.per_room_per_diff])
    rng.shuffle(selected)
    if a.target and len(selected) > a.target:
        selected = selected[:a.target]
    for xml_key, r, diff in selected:
        pool[xml_key].append(r)
        counts[diff] += 1
        room_used.add(base_room(xml_key))

    with open(a.out_pool, "w") as f:
        json.dump(pool, f)
    n_eps = sum(len(v) for v in pool.values())
    report = {
        "manifest": a.manifest, "testset_labels_dir": a.testset_labels_dir,
        "testset": ts_info,
        "gate": {"dropped_by_name": len(dropped_name), "dropped_by_geometry": len(dropped_geom),
                 "dropped_name_sample": sorted(dropped_name)[:20],
                 "dropped_geom_sample": sorted(dropped_geom)[:20],
                 "rooms_kept_after_gate": len(kept_rooms)},
        "pool": {"n_episodes": n_eps, "n_pair_xmls": len(pool), "n_base_rooms": len(room_used),
                 "difficulty": dict(counts),
                 "per_room_per_diff": a.per_room_per_diff, "target": a.target, "seed": a.seed},
    }
    with open(a.out_gate_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[pool] WROTE {a.out_pool}: {n_eps} episodes / {len(pool)} pair-xmls / "
          f"{len(room_used)} base rooms  difficulty={dict(counts)}", flush=True)
    print(f"[gate] report -> {a.out_gate_report}", flush=True)


if __name__ == "__main__":
    main()
