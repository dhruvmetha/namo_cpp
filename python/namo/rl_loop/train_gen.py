"""Train the two heads for one generation on the loop's training H5.

Thin driver over the reused sage stack: EdgeCrossAttn (network) + WeightedClassifierModule
(ClassifierModule subclass) + RLDataModule + Lightning Trainer, mirroring
sage/src/train_classifier.py's wiring (instantiate -> fit) but with the weighted module/data.

  pi : softmax_ce head (value_bins=0) — filtered BC over the taken action, per-sample weighted.
  V  : hl_gauss head (value_bins=51, [0,1]) — MC-return regression on the taken action only.

Both use the sharp/e4 encoder recipe (pos_fourier + edge_embed + local gather), NOT budget-
conditioned (single ranker, per the card), so LiveScorer/eval_scorer auto-detect and load them.
Returns the best-val checkpoint path per head.
"""
import os
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from ._bootstrap import ensure_paths
ensure_paths()
from .config import LoopConfig, NUM_DEPTHS
from .action_motion import action_motion_feature_dim, configured_action_motion_encoding
from .sage_ext._sage import EdgeCrossAttn
from .sage_ext.weighted_module import WeightedClassifierModule
from .sage_ext.rl_dataset import RLDataModule


def _make_network(value_bins: int) -> EdgeCrossAttn:
    encoding = configured_action_motion_encoding()
    sharp_motion = os.environ.get("NAMO_ACTION_MOTION_SHARP", "0") == "1"
    net = EdgeCrossAttn(
        img_size=64, patch=4, in_channels=5, dim=192, scene_depth=4, edge_depth=4, heads=6,
        num_depths=NUM_DEPTHS, num_edges=60, use_local=True,
        pos_fourier=True, use_edge_embed=True,          # sharp/e4 identity recipe
        budget_cond=False, value_bins=value_bins,       # single ranker (no horizon conditioning)
        action_motion_dim=action_motion_feature_dim(encoding),
        action_motion_fourier=sharp_motion, action_motion_fourier_L=8,
        action_depth_embed=sharp_motion,
    )
    net.action_motion_encoding = encoding
    return net


def _train_one(h5_path: str, mode: str, cfg: LoopConfig, out_dir: str,
               max_epochs: Optional[int] = None, fast_smoke: bool = False) -> str:
    head_mode = "softmax_ce" if mode == "pi" else "hl_gauss"
    value_bins = 0 if mode == "pi" else 51
    net = _make_network(value_bins)
    module = WeightedClassifierModule(
        network=net, base_lr=cfg.base_lr, weight_decay=0.01,
        warmup_steps=200, decay_steps=100000, end_lr=1e-6,
        head_mode=head_mode, value_vmin=0.0, value_vmax=1.0, dice_weight=0.0,
    )
    dm = RLDataModule(h5_path, mode=mode, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    ckpt_dir = os.path.join(out_dir, mode, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min", save_top_k=1,
                              save_last=True, filename="epoch{epoch:03d}-val_loss{val_loss:.4f}",
                              auto_insert_metric_name=False)
    cbs = [ckpt_cb]
    if not fast_smoke:
        cbs.append(EarlyStopping(monitor="val_loss", mode="min", patience=8))
    trainer = pl.Trainer(
        max_epochs=(max_epochs if max_epochs is not None else cfg.max_epochs),
        accelerator="auto", devices=1, precision="16-mixed",
        callbacks=cbs, logger=False, enable_progress_bar=False,
        limit_train_batches=(3 if fast_smoke else 1.0),
        limit_val_batches=(2 if fast_smoke else 1.0),
        num_sanity_val_steps=0,
    )
    trainer.fit(module, dm)
    return ckpt_cb.best_model_path or os.path.join(ckpt_dir, "last.ckpt")


def train_generation(h5_path: str, cfg: LoopConfig, out_dir: str,
                     fast_smoke: bool = False) -> dict:
    out = {}
    if cfg.train_pi:
        out["pi_ckpt"] = _train_one(h5_path, "pi", cfg, out_dir,
                                    max_epochs=(1 if fast_smoke else None), fast_smoke=fast_smoke)
    if cfg.train_v:
        out["v_ckpt"] = _train_one(h5_path, "v", cfg, out_dir,
                                   max_epochs=(1 if fast_smoke else None), fast_smoke=fast_smoke)
    return out
