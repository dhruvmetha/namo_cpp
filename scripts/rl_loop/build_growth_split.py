#!/usr/bin/env python3
"""Cumulative growth pool + room-held-out split for EXP-2026-07-08-rl-growth-arms.

Pool_G = gen0_pool  UNION  batch_gen1 ... batch_genG   (each batch a disjoint SUBSET of its source).
Split_G (rooms held out by BASE ROOM, dev/test FROZEN from gen0):
  train = gen0_split.train  +  batch xmls whose base room is NOT in that batch's dev slice
  dev   = gen0_split.dev    +  batch xmls whose base room IS  in that batch's dev slice  (new-batch dev)
  test  = gen0_split.test                                                                (frozen)
Asserts 0 base-room leakage across train/dev/test and that every pool xml is covered by the split.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"), str(REPO / "scripts/rl_loop")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from build_gen0_pool import base_room             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen0-pool", required=True)
    ap.add_argument("--gen0-split", required=True)
    ap.add_argument("--batches", nargs="+", required=True)
    ap.add_argument("--batch-devrooms", nargs="+", required=True,
                    help="parallel to --batches: devrooms json {dev_rooms:[...]} per batch")
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--out-split", required=True)
    a = ap.parse_args()
    assert len(a.batches) == len(a.batch_devrooms), "batches and batch-devrooms must be parallel"

    pool = dict(json.load(open(a.gen0_pool)))          # copy
    g0 = json.load(open(a.gen0_split))
    train = list(g0["train"]); dev = list(g0["dev"]); test = list(g0["test"])

    for bpath, dpath in zip(a.batches, a.batch_devrooms):
        batch = json.load(open(bpath))
        dev_rooms = set(json.load(open(dpath))["dev_rooms"])
        for xml_key, recs in batch.items():
            assert xml_key not in pool, f"batch xml already in pool: {xml_key}"
            pool[xml_key] = recs
            (dev if base_room(xml_key) in dev_rooms else train).append(xml_key)

    tr, dv, te = set(train), set(dev), set(test)
    # room-level disjointness (the real invariant)
    rtr = {base_room(k) for k in tr}; rdv = {base_room(k) for k in dv}; rte = {base_room(k) for k in te}
    assert not (rtr & rdv), f"room leak train/dev: {sorted(rtr & rdv)[:5]}"
    assert not (rtr & rte), f"room leak train/test: {sorted(rtr & rte)[:5]}"
    assert not (rdv & rte), f"room leak dev/test: {sorted(rdv & rte)[:5]}"
    # every pool xml covered exactly once
    covered = tr | dv | te
    missing = set(pool.keys()) - covered
    assert not missing, f"{len(missing)} pool xmls not in split, e.g. {sorted(missing)[:3]}"

    n_eps = sum(len(v) for v in pool.values())
    with open(a.out_pool, "w") as f:
        json.dump(pool, f)
    with open(a.out_split, "w") as f:
        json.dump({"grouping": "base_room", "gen0_split": a.gen0_split, "batches": a.batches,
                   "n_rooms_train": len(rtr), "n_rooms_dev": len(rdv), "n_rooms_test": len(rte),
                   "train": sorted(tr), "dev": sorted(dv), "test": sorted(te)}, f)
    print(f"[pool] {a.out_pool}: {len(pool)} xmls / {n_eps} eps", flush=True)
    print(f"[split] {a.out_split}: rooms train={len(rtr)} dev={len(rdv)} test={len(rte)} | "
          f"episodes train={len(tr)} dev={len(dv)} test={len(te)}", flush=True)


if __name__ == "__main__":
    main()
