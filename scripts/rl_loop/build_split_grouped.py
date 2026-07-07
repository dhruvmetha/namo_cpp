#!/usr/bin/env python3
"""Freeze an 80/10/10 split that holds out by BASE ROOM, not per-pair xml.

The stock build_split.py groups by the pool key (here a per-pair xml path), which would scatter
pairs of one static room across train/dev/test -> room-geometry leakage. This groups the pool's
per-pair xml keys by base room (run_NNNN_env_NNNN via build_gen0_pool.base_room), splits the
DISTINCT base rooms 80/10/10, and writes the split as full-xml-path lists so the unchanged
load_split/episodes_in consume it directly. Asserts 0% base-room leakage across the three sets.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"), str(REPO / "scripts/rl_loop")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from build_gen0_pool import base_room            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fracs", default="0.8,0.1,0.1")
    a = ap.parse_args()
    fr = tuple(float(x) for x in a.fracs.split(","))

    pool = json.load(open(a.pool_key))
    by_room = defaultdict(list)          # base_room -> [full xml keys]
    for xml_key in pool:
        by_room[base_room(xml_key)].append(xml_key)
    rooms = sorted(by_room)
    rng = random.Random(a.seed)
    rng.shuffle(rooms)
    n = len(rooms)
    n_tr = int(round(fr[0] * n)); n_dev = int(round(fr[1] * n))
    tr_rooms = rooms[:n_tr]; dev_rooms = rooms[n_tr:n_tr + n_dev]; te_rooms = rooms[n_tr + n_dev:]
    # zero base-room leakage
    assert not (set(tr_rooms) & set(dev_rooms)), "room leak tr/dev"
    assert not (set(tr_rooms) & set(te_rooms)), "room leak tr/te"
    assert not (set(dev_rooms) & set(te_rooms)), "room leak dev/te"

    def flatten(rlist):
        out = []
        for r in rlist:
            out.extend(by_room[r])
        return sorted(out)

    train = flatten(tr_rooms); dev = flatten(dev_rooms); test = flatten(te_rooms)
    # zero per-pair-xml leakage (implied, but assert)
    assert not (set(train) & set(dev)) and not (set(train) & set(test)) and not (set(dev) & set(test))
    with open(a.out, "w") as f:
        json.dump({"seed": a.seed, "pool_key": a.pool_key, "fracs": list(fr),
                   "grouping": "base_room", "n_rooms": n,
                   "n_rooms_train": len(tr_rooms), "n_rooms_dev": len(dev_rooms),
                   "n_rooms_test": len(te_rooms),
                   "train": train, "dev": dev, "test": test}, f)
    print(f"rooms: train={len(tr_rooms)} dev={len(dev_rooms)} test={len(te_rooms)} "
          f"(base-room grouped) | episodes: train={len(train)} dev={len(dev)} test={len(test)}  ->  {a.out}")


if __name__ == "__main__":
    main()
