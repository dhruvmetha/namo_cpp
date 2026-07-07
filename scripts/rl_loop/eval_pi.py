#!/usr/bin/env python3
"""Standalone dev eval of a trained pi ckpt: greedy open@1/2/5/10 + setup-hit@1/2/4/8, stratified
by difficulty x horizon. Reuses eval_gen (same protocol as run_generation's in-loop eval), so a
gen's pi head can be scored without re-running the whole generation — e.g. when the V head hung
but pi is complete.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from namo.rl_loop._bootstrap import ensure_paths          # noqa: E402
ensure_paths()
from namo.rl_loop.config import LoopConfig                 # noqa: E402
from namo.rl_loop.episodes import load_pool                # noqa: E402
from namo.rl_loop.splits import load_split, episodes_in    # noqa: E402
from namo.rl_loop.eval_gen import greedy_open_ks, setup_ranking  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-ckpt", required=True)
    ap.add_argument("--pool-key", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--arm", default="?")
    ap.add_argument("--pi-info", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cfg = LoopConfig(pool_key=a.pool_key, split_file=a.split_file)
    specs = load_pool(a.pool_key)
    split = load_split(a.split_file)
    dev = episodes_in(specs, split, "dev")
    print(f"[eval_pi] arm={a.arm} dev_episodes={len(dev)} pi={a.pi_ckpt}", flush=True)
    greedy = greedy_open_ks(a.pi_ckpt, dev, cfg, limit=a.limit)
    print("[eval_pi] greedy done, running setup-ranking...", flush=True)
    setup = setup_ranking(a.pi_ckpt, dev, cfg, limit=a.limit)
    out = {"arm": a.arm, "pi_ckpt": a.pi_ckpt, "pi_info": a.pi_info,
           "n_dev": len(dev),
           "greedy_open_by_tier_horizon": greedy,
           "setup_ranking_by_tier_horizon": setup}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"=== DEV EVAL {a.arm} ({a.pi_info}) ===")
    print("GREEDY open@k (horizon/difficulty):")
    for k in sorted(greedy):
        print("  ", k, greedy[k])
    print("SETUP-hit@k (horizon/difficulty):")
    for k in sorted(setup):
        print("  ", k, setup[k])
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
