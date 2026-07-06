#!/usr/bin/env python3
"""Run ONE RL self-imitation generation end-to-end (collect -> buffer -> train -> dev eval).

  gen 0 (arm A):  --arm A --generation 0                       (uniform pi0, no ckpt)
  gen N (arm B):  --arm B --generation N --ckpt <prev pi.ckpt>  (policy-conditioned)

Collection is in-process (--n-workers) unless --pre-collected-dir points at a SLURM fan-out
(collect.slurm -> collect_shard.py). --fast-smoke does a tiny end-to-end pass for verification.
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop.config import LoopConfig                 # noqa: E402
from namo.rl_loop.run_generation import run_generation      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A", choices=["A", "B"])
    ap.add_argument("--generation", type=int, default=0)
    ap.add_argument("--pool-key", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="policy ckpt: arm B pretrain at gen 0, or EITHER arm's own prev-gen pi ckpt at gen>0")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--rollouts-per-episode", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.10)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--revalidate-fraction", type=float, default=0.1)
    ap.add_argument("--pre-collected-dir", default="")
    ap.add_argument("--expected-shards", type=int, default=0,
                    help="required with --pre-collected-dir: SLURM NSHARDS; harvest hard-fails on a count mismatch")
    ap.add_argument("--fast-smoke", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--collect-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()

    cfg = LoopConfig(
        arm=a.arm, generation=a.generation, run_root=a.out_root,
        ckpt=a.ckpt,
        pool_key=a.pool_key, split_file=a.split_file,
        rollouts_per_episode=a.rollouts_per_episode, temperature=a.temperature, epsilon=a.epsilon,
        max_depth=a.max_depth, gamma=a.gamma, max_epochs=a.max_epochs,
        revalidate_fraction=a.revalidate_fraction,
    )
    run_generation(cfg, a.out_root, n_workers=a.n_workers, fast_smoke=a.fast_smoke,
                   eval_limit=a.eval_limit, collect_limit=a.collect_limit, seed=a.seed,
                   pre_collected_dir=a.pre_collected_dir, expected_shards=a.expected_shards)


if __name__ == "__main__":
    main()
