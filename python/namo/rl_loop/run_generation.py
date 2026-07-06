"""One generation, end to end: collect -> harvest+filter -> build H5 -> train -> dev eval
-> report row + kill-signal check.

  gen 0 (arm A): cfg.ckpt=None (uniform pi0).  gen>0 / arm B: cfg.ckpt = the previous
  generation's pi checkpoint. Everything else is driven by LoopConfig. Collection runs on
  TRAIN rooms; greedy-open eval runs on DEV rooms (both from the frozen split).
"""
import os
import pickle
import random
from typing import List, Optional

from ._bootstrap import ensure_paths
ensure_paths()
from .config import LoopConfig
from .episodes import EpisodeSpec, load_pool
from .splits import load_split, episodes_in
from .buffer import SolveBuffer, _keystr
from .collector import run_collection
from .build_train_h5 import build_h5
from .train_gen import train_generation
from .eval_gen import greedy_open_ks, setup_ranking
from .report import check_kill_signals, build_report_row, print_report_row, save_report_row


def _load_rollout_pkls(pkls: List[str]) -> List[dict]:
    out = []
    for p in pkls:
        with open(p, "rb") as f:
            out.extend(pickle.load(f))
    return out


def run_generation(cfg: LoopConfig, out_root: str, n_workers: int = 1,
                   fast_smoke: bool = False, eval_limit: int = 0,
                   collect_limit: int = 0, seed: int = 7000,
                   pre_collected_dir: str = "") -> dict:
    """pre_collected_dir: harvest rollout pkls produced by a SLURM collect fan-out
    (scripts/rl_loop/collect_shard.py) instead of collecting in-process."""
    gen_dir = os.path.join(out_root, f"gen{cfg.generation}")
    os.makedirs(gen_dir, exist_ok=True)
    cfg.to_json(os.path.join(gen_dir, "config.json"))

    specs = load_pool(cfg.pool_key)
    split = load_split(cfg.split_file)
    train_specs = episodes_in(specs, split, "train")
    dev_specs = episodes_in(specs, split, "dev")
    if collect_limit:
        train_specs = train_specs[:collect_limit]

    # --- persistent buffer across generations ---
    buffer_path = os.path.join(out_root, "buffer.pkl")
    buf = SolveBuffer.load(buffer_path)

    # --- 1. collect on TRAIN rooms (in-process) or harvest a SLURM fan-out ---
    if pre_collected_dir:
        import glob
        pkls = sorted(glob.glob(os.path.join(pre_collected_dir, "*.pkl")))
    else:
        buf_first = buf.first_actions_by_episode()
        pkls = run_collection(train_specs, cfg, buf_first,
                              os.path.join(gen_dir, "collect"), n_workers=n_workers, seed=seed)
    rollout_dicts = _load_rollout_pkls(pkls)

    # --- 2. harvest + filter into the buffer ---
    buf.ingest(rollout_dicts, cfg, rng=random.Random(seed))
    if cfg.revalidate_fraction > 0:
        dropped = buf.revalidate(cfg, cfg.revalidate_fraction, random.Random(seed + 1))
        print(f"[revalidate] dropped {dropped} broken solves", flush=True)
    buf.save(buffer_path)

    hard_keys = {_keystr(ep.key) for ep in train_specs if ep.difficulty == "hard"}
    stats = buf.stats(hard_keys)
    n_solved = sum(r["solved"] for r in rollout_dicts)
    stats["rollouts_this_gen"] = len(rollout_dicts)
    stats["solved_this_gen"] = int(n_solved)

    # --- 3. build training H5 ---
    h5_path = os.path.join(gen_dir, "train_data", "data.h5")
    h5info = build_h5(buf, cfg, h5_path, render_config=cfg.car_config)

    # --- 4. train both heads ---
    ckpts = {}
    if h5info["n_bc_rows"] > 0:
        ckpts = train_generation(h5_path, cfg, os.path.join(gen_dir, "ckpts"), fast_smoke=fast_smoke)

    # --- 5. dev eval: greedy-open + setup-ranking, both stratified by difficulty x horizon ---
    eval_report = {}
    setup_report = {}
    if ckpts.get("pi_ckpt"):
        eval_report = greedy_open_ks(ckpts["pi_ckpt"], dev_specs, cfg, limit=eval_limit)
        setup_report = setup_ranking(ckpts["pi_ckpt"], dev_specs, cfg, limit=eval_limit)

    # --- 6. kill signals + report row ---
    prev = _prev_hard_unique(out_root, cfg.generation)
    kills = check_kill_signals(cfg.generation, stats, eval_report, cfg.eval_max_pushes, prev)
    row = build_report_row(cfg.generation, cfg.arm, stats, eval_report, kills,
                           extra={"setup_ranking_by_tier_horizon": setup_report,
                                  "h5": h5info, "ckpts": ckpts})
    print_report_row(row)
    save_report_row(row, os.path.join(gen_dir, "report.json"))
    return row


def _prev_hard_unique(out_root: str, generation: int) -> Optional[int]:
    if generation <= 0:
        return None
    prev = os.path.join(out_root, f"gen{generation - 1}", "report.json")
    if not os.path.exists(prev):
        return None
    import json
    r = json.load(open(prev))
    return r.get("buffer", {}).get("unique_solves_by_tier", {}).get("hard")
