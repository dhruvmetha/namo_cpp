#!/usr/bin/env python3
"""Build + freeze the 80/10/10 room-held-out split for the RL loop (run ONCE before gen-0)."""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop.splits import build_split       # noqa: E402
from namo.paths import DATASETS                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-key", required=True, help="per-episode key json defining the pool")
    ap.add_argument("--out", required=True, help="output frozen split json")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    sp = build_split(a.pool_key, a.out, seed=a.seed)
    print(f"rooms: train={len(sp.train)} dev={len(sp.dev)} test={len(sp.test)}  ->  {a.out}")


if __name__ == "__main__":
    main()
