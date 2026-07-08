#!/usr/bin/env python3
"""Author a LoopConfig json for a growth-arm collection shard (EXP-2026-07-08-rl-growth-arms).

collect_shard.py reads LoopConfig.from_json; this writes it with the growth-collection knobs
(R=16, T=0.1, eps=0.10, forced sweeps on, object-restricted) and Amarel-native pool/split/ckpt
paths. Run on either box (paths are stored as /scratch/dm1487 keys, box-agnostic)."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from namo.rl_loop.config import LoopConfig       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--pool-key", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--ckpt", default=None, help="collection policy ckpt (Amarel path); omit for uniform")
    ap.add_argument("--rollouts-per-episode", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = LoopConfig(
        arm=a.arm, generation=a.generation, ckpt=(a.ckpt or None),
        pool_key=a.pool_key, split_file=a.split_file,
        rollouts_per_episode=a.rollouts_per_episode, temperature=a.temperature, epsilon=a.epsilon,
    )
    cfg.to_json(a.out)
    print(f"[config] arm={a.arm} gen={a.generation} R={a.rollouts_per_episode} T={a.temperature} "
          f"eps={a.epsilon} ckpt={'yes' if a.ckpt else 'uniform'} -> {a.out}")


if __name__ == "__main__":
    main()
