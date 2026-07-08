#!/usr/bin/env python3
"""CS-side growth-generation driver (EXP-2026-07-08-rl-growth-arms).

Harvest Amarel-collected rollout pkls -> PARALLEL H5 render -> WARM-START train pi -> dev eval ->
report + kill signals. Identical machinery to the predecessor's run_gen_cs.py (spawn render,
--mem=220G at the SLURM layer, uncompressed ctx, --expected-shards harvest assert) with ONE change:
pi training WARM-STARTS from --init-ckpt (the arm's previous-generation pi; gen-1 = the seed
armA/gen1 pi). The predecessor trained pi from SCRATCH on the cumulative buffer each generation;
this experiment switches BOTH arms to warm-start (identical treatment, so the arm comparison still
isolates the data diet). V head stays OFF (hl_gauss hang, open bug).

pool-key / split-file = the CUMULATIVE growth pool/split (coverage over the full train, dev eval on
the frozen gen0 dev + this arm's new-batch dev). Collection was targeted separately
(build_collect_target -> collect_shard on Amarel), so this driver only harvests + trains + evals.

revalidate default 0.0: solves were collected+validated on Amarel (MuJoCo 3.2.7); re-executing them
on CS (3.2.8) would wrongly drop valid near-threshold solves (documented in run_gen_cs.py).
"""
import argparse
import glob
import os
import pickle
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop._bootstrap import ensure_paths                          # noqa: E402
ensure_paths()
from namo.rl_loop.config import LoopConfig, NUM_DEPTHS                     # noqa: E402
from namo.rl_loop.episodes import load_pool                               # noqa: E402
from namo.rl_loop.splits import load_split, episodes_in                   # noqa: E402
from namo.rl_loop.buffer import SolveBuffer, _keystr                      # noqa: E402
from namo.rl_loop.eval_gen import greedy_open_ks, setup_ranking           # noqa: E402
from namo.rl_loop.report import (check_kill_signals, build_report_row,    # noqa: E402
                                 print_report_row, save_report_row)
from namo.rl_loop.parallel_h5 import build_h5_parallel                    # noqa: E402


def train_pi_warm(h5_path: str, cfg: LoopConfig, out_dir: str, init_ckpt: str) -> str:
    """Filtered-BC pi head (softmax_ce), WARM-STARTED from init_ckpt via state_dict (cross-sage safe:
    the ckpt is 138 `network.*` tensors, identical arch to the fresh pi net). Returns best-val ckpt."""
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from namo.rl_loop.sage_ext._sage import EdgeCrossAttn
    from namo.rl_loop.sage_ext.weighted_module import WeightedClassifierModule
    from namo.rl_loop.sage_ext.rl_dataset import RLDataModule

    net = EdgeCrossAttn(img_size=64, patch=4, in_channels=5, dim=192, scene_depth=4, edge_depth=4,
                        heads=6, num_depths=NUM_DEPTHS, num_edges=60, use_local=True,
                        pos_fourier=True, use_edge_embed=True, budget_cond=False, value_bins=0)
    module = WeightedClassifierModule(
        network=net, base_lr=cfg.base_lr, weight_decay=0.01,
        warmup_steps=200, decay_steps=100000, end_lr=1e-6,
        head_mode="softmax_ce", value_vmin=0.0, value_vmax=1.0, dice_weight=0.0)
    warm = "from-scratch"
    if init_ckpt:
        sd = torch.load(init_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = module.load_state_dict(sd, strict=False)
        warm = f"warm[{os.path.basename(init_ckpt)}] missing={len(missing)} unexpected={len(unexpected)}"
        if len(missing) > 4 or len(unexpected) > 4:
            raise RuntimeError(f"warm-start state_dict mismatch too large: {warm} "
                               f"missing={missing[:6]} unexpected={unexpected[:6]}")
    print(f"[train pi] init = {warm}", flush=True)

    dm = RLDataModule(h5_path, mode="pi", batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    ckpt_dir = os.path.join(out_dir, "pi", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min", save_top_k=1,
                              save_last=True, filename="epoch{epoch:03d}-val_loss{val_loss:.4f}",
                              auto_insert_metric_name=False)
    es = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    trainer = pl.Trainer(max_epochs=cfg.max_epochs, accelerator="auto", devices=1, precision="16-mixed",
                         callbacks=[ckpt_cb, es], logger=False, enable_progress_bar=False,
                         num_sanity_val_steps=0)
    trainer.fit(module, dm)
    return ckpt_cb.best_model_path or os.path.join(ckpt_dir, "last.ckpt")


def _prev_hard_unique(out_root, generation):
    if generation <= 1:
        return None
    prev = os.path.join(out_root, f"gen{generation - 1}", "report.json")
    if not os.path.exists(prev):
        return None
    import json
    return json.load(open(prev)).get("buffer", {}).get("unique_solves_by_tier", {}).get("hard")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="N or C (label)")
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--pool-key", required=True, help="CUMULATIVE growth pool")
    ap.add_argument("--split-file", required=True, help="CUMULATIVE growth split")
    ap.add_argument("--out-root", required=True, help="growth_{N,C} run dir")
    ap.add_argument("--init-ckpt", default=None, help="warm-start pi from this ckpt (prev gen / seed)")
    ap.add_argument("--pre-collected-dir", required=True)
    ap.add_argument("--expected-shards", type=int, required=True)
    ap.add_argument("--rollouts-per-episode", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--vhead-fail-keep-frac", type=float, default=0.1)
    ap.add_argument("--revalidate-fraction", type=float, default=0.0)
    ap.add_argument("--render-workers", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()

    cfg = LoopConfig(
        arm=a.arm, generation=a.generation, run_root=a.out_root, ckpt=a.init_ckpt,
        pool_key=a.pool_key, split_file=a.split_file,
        rollouts_per_episode=a.rollouts_per_episode, temperature=a.temperature, gamma=a.gamma,
        vhead_fail_keep_frac=a.vhead_fail_keep_frac, revalidate_fraction=a.revalidate_fraction,
        max_epochs=a.max_epochs, num_workers=a.num_workers, train_v=False,
    )
    gen_dir = os.path.join(a.out_root, f"gen{a.generation}")
    os.makedirs(gen_dir, exist_ok=True)
    cfg.to_json(os.path.join(gen_dir, "config_cs.json"))

    specs = load_pool(a.pool_key)
    split = load_split(a.split_file)
    train_specs = episodes_in(specs, split, "train")
    dev_specs = episodes_in(specs, split, "dev")
    print(f"[growth] arm={a.arm} gen={a.generation} pool_eps={len(specs)} "
          f"train={len(train_specs)} dev={len(dev_specs)}", flush=True)

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
        ckpts["pi_ckpt"] = train_pi_warm(h5_path, cfg, os.path.join(gen_dir, "ckpts"), a.init_ckpt)

    eval_report = setup_report = {}
    if ckpts.get("pi_ckpt"):
        eval_report = greedy_open_ks(ckpts["pi_ckpt"], dev_specs, cfg, limit=a.eval_limit)
        setup_report = setup_ranking(ckpts["pi_ckpt"], dev_specs, cfg, limit=a.eval_limit)

    prev = _prev_hard_unique(a.out_root, a.generation)
    kills = check_kill_signals(a.generation, stats, eval_report, cfg.eval_max_pushes, prev)
    row = build_report_row(a.generation, a.arm, stats, eval_report, kills,
                           extra={"setup_ranking_by_tier_horizon": setup_report,
                                  "h5": h5info, "ckpts": ckpts, "init_ckpt": a.init_ckpt})
    print_report_row(row)
    save_report_row(row, os.path.join(gen_dir, "report.json"))


if __name__ == "__main__":
    main()
