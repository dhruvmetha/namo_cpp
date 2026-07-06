#!/usr/bin/env python3
"""One SLURM-array collection shard: roll out a slice of TRAIN episodes -> one pkl.

The array launcher (collect.slurm) runs N of these, one per node; run_generation.py then
harvests the whole directory (--pre-collected-dir). Arm A (uniform pi0) needs no ckpt and no
GPU; arm B passes --ckpt and scores with the model (CPU on a compute node, GPU if present).
"""
import argparse
import os
import pickle
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop.config import LoopConfig                 # noqa: E402
from namo.rl_loop.episodes import load_pool                 # noqa: E402
from namo.rl_loop.splits import load_split, episodes_in     # noqa: E402
from namo.rl_loop.buffer import SolveBuffer                 # noqa: E402
from namo.rl_loop.policy import Policy                      # noqa: E402
from namo.rl_loop.collector import collect_episode          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="LoopConfig json")
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--buffer", default="", help="persistent buffer.pkl (for forced-sweep trigger)")
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()

    cfg = LoopConfig.from_json(a.config)
    specs = load_pool(cfg.pool_key)
    split = load_split(cfg.split_file)
    train_specs = episodes_in(specs, split, "train")
    shard = train_specs[a.shard_idx::a.n_shards]

    buf_first = SolveBuffer.load(a.buffer).first_actions_by_episode() if a.buffer else {}
    policy = Policy(ckpt=cfg.ckpt, score_h=cfg.score_h)
    rng = random.Random(a.seed + a.shard_idx)

    out = []
    for ep in shard:
        try:
            recs = collect_episode(ep, cfg, policy, set(buf_first.get(ep.key, set())), rng)
        except Exception as e:
            print(f"  skip {ep.xml_key} {ep.object_id}: {e}", flush=True)
            continue
        out.extend(r.to_dict() for r in recs)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"rollouts_shard{a.shard_idx:04d}.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"shard {a.shard_idx}: {len(shard)} episodes -> {len(out)} rollouts -> {path}", flush=True)


if __name__ == "__main__":
    main()
