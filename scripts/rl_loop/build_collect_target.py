#!/usr/bin/env python3
"""Per-generation COLLECTION target for a growth arm (EXP-2026-07-08-rl-growth-arms).

The generation's rollout budget goes to NEW episodes + a refresh sample of OLD ones, not the whole
growing pool. This writes a thin collect-pool + collect-split that the unchanged collect_shard.py
reads (its train_specs == everything here):
  collect = (this-gen batch xmls on the TRAIN side, 100%)  UNION  (refresh_frac sample of OLD train)
  old train = cumulative_split.train  MINUS  this batch's xmls
The persistent buffer (seeded from armA/buffer.pkl, copied per arm) still carries every prior solve,
so refreshing only a sample of old episodes does not lose coverage.
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cumulative-pool", required=True)
    ap.add_argument("--cumulative-split", required=True)
    ap.add_argument("--this-batch", required=True)
    ap.add_argument("--refresh-frac", type=float, default=0.25)
    ap.add_argument("--out-collect-pool", required=True)
    ap.add_argument("--out-collect-split", required=True)
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()

    pool = json.load(open(a.cumulative_pool))
    split = json.load(open(a.cumulative_split))
    train_keys = set(split["train"])
    batch_keys = set(json.load(open(a.this_batch)).keys())

    new_train = sorted(batch_keys & train_keys)          # new-batch episodes on the train side
    old_train = sorted(train_keys - batch_keys)          # everything already in the pool
    rng = random.Random(a.seed)
    n_refresh = int(round(a.refresh_frac * len(old_train)))
    refresh = rng.sample(old_train, n_refresh) if n_refresh else []

    collect_keys = sorted(set(new_train) | set(refresh))
    collect_pool = {k: pool[k] for k in collect_keys}
    with open(a.out_collect_pool, "w") as f:
        json.dump(collect_pool, f)
    with open(a.out_collect_split, "w") as f:
        json.dump({"grouping": "collect", "train": collect_keys, "dev": [], "test": []}, f)
    n_eps = sum(len(v) for v in collect_pool.values())
    print(f"[collect] new_train={len(new_train)} old_train={len(old_train)} "
          f"refresh({a.refresh_frac})={len(refresh)}  ->  collect xmls={len(collect_keys)} eps={n_eps}", flush=True)
    print(f"[collect] pool -> {a.out_collect_pool}", flush=True)
    print(f"[collect] split -> {a.out_collect_split}", flush=True)


if __name__ == "__main__":
    main()
