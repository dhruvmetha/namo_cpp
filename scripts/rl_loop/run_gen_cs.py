#!/usr/bin/env python3
"""CS-side generation driver: harvest Amarel-collected pkls -> PARALLEL H5 render -> train -> eval.

Same flow as namo.rl_loop.run_generation.run_generation, with two changes needed to run at scale
on the CS estate every generation:
  1. PARALLEL H5 render — build_train_h5.build_h5 renders every kept (state,action) row serially
     (~64 ms/row on CS) and re-renders the whole accumulating buffer each generation; at ~10^5 rows
     that is hours. We render across `--render-workers` processes (byte-identical rows, just sharded)
     and merge the shard H5s. Verified equal to the serial builder on the smoke buffer.
  2. Config knobs the stock CLI doesn't expose: --vhead-fail-keep-frac (bound the V-only rows),
     --revalidate-fraction (default 0 on CS — the collection sim already validated solves; the
     Amarel-3.2.7 vs CS-3.2.8 replay would wrongly drop valid near-threshold solves), --render-workers,
     --num-workers (train dataloader), --max-epochs.

Everything else (buffer retention, weights, targets, eval, kill signals, report) reuses the stock
modules unchanged, so the numbers are identical to run_generation's.
"""
import argparse
import glob
import multiprocessing as mp
import os
import pickle
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop._bootstrap import ensure_paths           # noqa: E402
ensure_paths()
from namo.rl_loop.config import LoopConfig                  # noqa: E402
from namo.rl_loop.episodes import load_pool                 # noqa: E402
from namo.rl_loop.splits import load_split, episodes_in     # noqa: E402
from namo.rl_loop.buffer import SolveBuffer, _keystr        # noqa: E402
from namo.rl_loop.build_train_h5 import _write_h5           # noqa: E402
from namo.rl_loop.train_gen import train_generation         # noqa: E402
from namo.rl_loop.eval_gen import greedy_open_ks, setup_ranking  # noqa: E402
from namo.rl_loop.report import (check_kill_signals, build_report_row,  # noqa: E402
                                 print_report_row, save_report_row)


# parallel H5 render lives in namo.rl_loop.parallel_h5 (SPAWN-based pool — fork deadlocks after
# the torch/sage threadpool imports; hit as a hard hang on the first gen-0 attempt).
from namo.rl_loop.parallel_h5 import build_h5_parallel   # noqa: E402


# ---------------- generation flow ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A", choices=["A", "B"])
    ap.add_argument("--generation", type=int, default=0)
    ap.add_argument("--pool-key", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pre-collected-dir", required=True)
    ap.add_argument("--expected-shards", type=int, required=True)
    ap.add_argument("--rollouts-per-episode", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--vhead-fail-keep-frac", type=float, default=0.1)
    ap.add_argument("--revalidate-fraction", type=float, default=0.0)
    ap.add_argument("--render-workers", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--train-v", action="store_true",
                    help="train the V head too. OFF by default: the hl_gauss V-head training HANGS "
                         "(both arms, both gens — see card V-head status); leaving it off keeps it from "
                         "ever blocking the pi eval again. Re-enable once the hang is fixed.")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()

    cfg = LoopConfig(
        arm=a.arm, generation=a.generation, run_root=a.out_root, ckpt=a.ckpt,
        pool_key=a.pool_key, split_file=a.split_file,
        rollouts_per_episode=a.rollouts_per_episode, temperature=a.temperature, gamma=a.gamma,
        vhead_fail_keep_frac=a.vhead_fail_keep_frac, revalidate_fraction=a.revalidate_fraction,
        max_epochs=a.max_epochs, num_workers=a.num_workers, train_v=a.train_v,
    )
    gen_dir = os.path.join(a.out_root, f"gen{a.generation}")
    os.makedirs(gen_dir, exist_ok=True)
    cfg.to_json(os.path.join(gen_dir, "config_cs.json"))

    specs = load_pool(a.pool_key)
    split = load_split(a.split_file)
    train_specs = episodes_in(specs, split, "train")
    dev_specs = episodes_in(specs, split, "dev")

    buffer_path = os.path.join(a.out_root, "buffer.pkl")
    buf = SolveBuffer.load(buffer_path)

    pkls = sorted(glob.glob(os.path.join(a.pre_collected_dir, "*.pkl")))
    if len(pkls) != a.expected_shards:
        raise RuntimeError(f"shard harvest incomplete: found {len(pkls)}, expected {a.expected_shards} "
                           f"in {a.pre_collected_dir}")
    rollout_dicts = []
    for p in pkls:
        with open(p, "rb") as f:
            rollout_dicts.extend(pickle.load(f))

    buf.ingest(rollout_dicts, cfg, rng=random.Random(a.seed))
    if cfg.revalidate_fraction > 0:
        dropped = buf.revalidate(cfg, cfg.revalidate_fraction, random.Random(a.seed + 1))
        print(f"[revalidate] dropped {dropped} broken solves", flush=True)
    buf.save(buffer_path)

    hard_keys = {_keystr(ep.key) for ep in train_specs if ep.difficulty == "hard"}
    stats = buf.stats(hard_keys)
    stats["rollouts_this_gen"] = len(rollout_dicts)
    stats["solved_this_gen"] = int(sum(r["solved"] for r in rollout_dicts))
    print(f"[KILL-SIGNAL-1] hard-episode positive coverage = {stats['hard_positive_coverage']} "
          f"(threshold 0.50)  unique_solves={stats['unique_solves_by_tier']}", flush=True)

    h5_path = os.path.join(gen_dir, "train_data", "data.h5")
    h5info = build_h5_parallel(buf, cfg, h5_path, cfg.car_config, a.render_workers)
    print(f"[h5] {h5info}", flush=True)

    ckpts = {}
    if h5info["n_bc_rows"] > 0:
        ckpts = train_generation(h5_path, cfg, os.path.join(gen_dir, "ckpts"), fast_smoke=False)

    eval_report = setup_report = {}
    if ckpts.get("pi_ckpt"):
        eval_report = greedy_open_ks(ckpts["pi_ckpt"], dev_specs, cfg, limit=a.eval_limit)
        setup_report = setup_ranking(ckpts["pi_ckpt"], dev_specs, cfg, limit=a.eval_limit)

    prev = _prev_hard_unique(a.out_root, a.generation)
    kills = check_kill_signals(a.generation, stats, eval_report, cfg.eval_max_pushes, prev)
    row = build_report_row(a.generation, a.arm, stats, eval_report, kills,
                           extra={"setup_ranking_by_tier_horizon": setup_report,
                                  "h5": h5info, "ckpts": ckpts})
    print_report_row(row)
    save_report_row(row, os.path.join(gen_dir, "report.json"))


def _prev_hard_unique(out_root, generation):
    if generation <= 0:
        return None
    prev = os.path.join(out_root, f"gen{generation - 1}", "report.json")
    if not os.path.exists(prev):
        return None
    import json
    return json.load(open(prev)).get("buffer", {}).get("unique_solves_by_tier", {}).get("hard")


if __name__ == "__main__":
    main()
