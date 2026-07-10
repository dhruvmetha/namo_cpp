#!/usr/bin/env python3
"""Train Q1 — the rung-1 dense opener classifier ("does one shove open the goal now?").

REACHABLE-ONLY arm: per-cell BCE(+Dice) over f_grid, masked to value_mask*r_mask (the TRIED cells).
Reuses the e4 EdgeCrossAttn (sharp recipe, value_bins=0 -> (B,60,5) sigmoid head) + the repo's
WeightedClassifierModule (a sage ClassifierModule subclass; head_mode=sigmoid_bce,
bce_reachable_only=True, uniform weight so the weighting is a no-op) + a ROOM-grouped datamodule on
the rung-1 H5 (python/namo/rl_loop/sage_ext/rung1_dataset.py).

Base trainer adapted: python/namo/rl_loop/train_gen.py (same _make_network + module family),
retargeted from the RL-loop chosen-action schema to the dense rung-1 schema.

__main__-guarded so the spawn-context dataloader workers (the V-head-hang fix) re-import cleanly.

Usage (smoke, arrakis GPU3):
  CUDA_VISIBLE_DEVICES=3 python scripts/rl_loop/train_q1.py \
    --h5 /common/users/dm1487/scratch_namo/exit/rung1_smoke50/rung1_smoke50.h5 \
    --out-dir <run_dir> --epochs 40 --batch-size 32 --num-workers 2
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "python"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop._bootstrap import ensure_paths                 # noqa: E402
ensure_paths()

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import lightning.pytorch as pl                                   # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback  # noqa: E402

from namo.rl_loop.train_gen import _make_network                 # noqa: E402  (e4 EdgeCrossAttn builder)
from namo.rl_loop.sage_ext.weighted_module import WeightedClassifierModule  # noqa: E402
from namo.rl_loop.sage_ext.rung1_dataset import Rung1DataModule  # noqa: E402


def build_module(base_lr: float, dice_weight: float, pos_weight: float,
                 warmup_steps: int, decay_steps: int) -> WeightedClassifierModule:
    net = _make_network(value_bins=0)            # (B,60,5) sigmoid head, sharp/e4 recipe
    return WeightedClassifierModule(
        network=net, base_lr=base_lr, weight_decay=0.01,
        warmup_steps=warmup_steps, decay_steps=decay_steps, end_lr=1e-6,
        head_mode="sigmoid_bce", bce_reachable_only=True,        # BCE(+Dice) on loss_mask (tried) only
        dice_weight=dice_weight, pos_weight=pos_weight,
    )


class EpochLoss(Callback):
    """Print per-epoch train/val loss so the smoke can SEE the loss decrease."""
    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        tr = m.get("train_loss_epoch", m.get("train_loss"))
        vl = m.get("val_loss")
        tr = float(tr) if tr is not None else float("nan")
        vl = float(vl) if vl is not None else float("nan")
        print(f"[epoch {trainer.current_epoch:03d}] train_loss={tr:.4f} val_loss={vl:.4f}", flush=True)


# ---------------------------------------------------------------------------
# opener diagnostics on the held-out room split (over TRIED cells)
# ---------------------------------------------------------------------------
def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y); p = np.asarray(p)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average-rank tie correction
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = np.empty(len(counts))
    lo = 0
    for j, c in enumerate(counts):
        avg[j] = (lo + 1 + lo + c) / 2.0
        lo += c
    ranks = avg[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(module, ds, device, tag="val"):
    module.eval().to(device)
    probs, ys = [], []
    top1_hits = top1_rows = 0
    for k in range(len(ds)):
        b = ds[k]
        ctx = b["context"].unsqueeze(0).to(device)
        cpx = b.get("contact_px")
        cpx = cpx.unsqueeze(0).to(device) if cpx is not None else None
        logits = module(ctx, cpx)                       # (1,60,5)
        p = torch.sigmoid(logits.float())[0].cpu()      # (60,5)
        f = b["f_labels"]; lm = b["loss_mask"]; rm = b["r_mask"]
        tried = lm > 0.5
        probs.append(p[tried].numpy()); ys.append(f[tried].numpy())
        if float(f[tried].sum()) > 0:                   # row has >=1 opener among tried cells
            top1_rows += 1
            reach = rm > 0.5
            ps = p.clone(); ps[~reach] = -1.0           # rank only reachable cells (deploy-realistic)
            idx = int(ps.reshape(-1).argmax())
            top1_hits += int(float(f.reshape(-1)[idx]) == 1.0)
    y = np.concatenate(ys) if ys else np.zeros(0)
    p = np.concatenate(probs) if probs else np.zeros(0)
    auc = _auc(y, p)
    base = float((y == 1).mean()) if len(y) else float("nan")
    pred = p >= 0.5
    tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    top1 = top1_hits / top1_rows if top1_rows else float("nan")
    print(f"\n===== Q1 opener diagnostics ({tag}, over TRIED cells) =====")
    print(f"  tried cells       : {len(y)}  (opener base rate {base:.3f})")
    print(f"  AUC (opener)      : {auc:.4f}")
    print(f"  precision@0.5     : {prec:.4f}   recall@0.5: {rec:.4f}   (tp={tp} fp={fp} fn={fn})")
    print(f"  top-1 opener hit  : {top1:.4f}  over {top1_rows} rows with an opener "
          f"(random ~ base rate {base:.3f})")
    print("=========================================================\n", flush=True)
    return dict(auc=auc, precision=prec, recall=rec, top1=top1, base=base,
                n_tried=len(y), top1_rows=top1_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--decay-steps", type=int, default=100000)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--train-split", type=float, default=0.9)
    ap.add_argument("--patience", type=int, default=0, help="EarlyStopping patience (0=off)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pl.seed_everything(a.seed, workers=True)
    ckpt_dir = os.path.join(a.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    module = build_module(a.lr, a.dice_weight, a.pos_weight, a.warmup_steps, a.decay_steps)
    dm = Rung1DataModule(a.h5, batch_size=a.batch_size, num_workers=a.num_workers,
                         train_split=a.train_split, split_seed=a.split_seed)

    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min", save_top_k=1,
                              save_last=True, filename="epoch{epoch:03d}-val_loss{val_loss:.4f}",
                              auto_insert_metric_name=False)
    cbs = [ckpt_cb, EpochLoss()]
    if a.patience > 0:
        cbs.append(EarlyStopping(monitor="val_loss", mode="min", patience=a.patience))

    trainer = pl.Trainer(
        max_epochs=a.epochs, accelerator="auto", devices=1, precision="16-mixed",
        callbacks=cbs, logger=False, enable_progress_bar=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, dm)
    best = ckpt_cb.best_model_path or os.path.join(ckpt_dir, "last.ckpt")
    print(f"\n[train_q1] best ckpt: {best}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dm.setup()

    # --- save+reload integrity ---
    # (a) two independent reloads of the SAME ckpt must be bit-identical (deterministic reload);
    # (b) the reloaded model must REPRODUCE the monitored best val_loss (the ckpt is the right model).
    # NB: we do NOT compare vs the in-memory `module` — ModelCheckpoint saves the BEST epoch, not the
    # final one, so those legitimately differ.
    ck = torch.load(best, map_location="cpu", weights_only=False)  # Lightning ckpt: numpy hparams

    def _reload():
        m = build_module(a.lr, a.dice_weight, a.pos_weight, a.warmup_steps, a.decay_steps)
        m.load_state_dict(ck["state_dict"])
        return m.eval().to(device)

    m1, m2 = _reload(), _reload()
    b = dm.val_dataset[0]
    ctx = b["context"].unsqueeze(0).to(device)
    cpx = b.get("contact_px")
    cpx = cpx.unsqueeze(0).to(device) if cpx is not None else None
    with torch.no_grad():
        d = float((m1(ctx, cpx).float() - m2(ctx, cpx).float()).abs().max())
    print(f"[reload check] two-reload max|Δlogit| = {d:.3e} "
          f"({'OK identical' if d == 0.0 else 'NONDETERMINISTIC'})", flush=True)

    # reproduce monitored best val_loss with the reloaded model (val = one batch of all val rows)
    from torch.utils.data import DataLoader
    vb = next(iter(DataLoader(dm.val_dataset, batch_size=len(dm.val_dataset))))
    vb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vb.items()}
    with torch.no_grad():
        vlogits = m1(vb["context"], vb.get("contact_px"))
        vloss = float(m1._compute_masked_loss(vlogits, vb["f_labels"], vb["loss_mask"]))
    monitored = float(ck.get("callbacks", {}) and getattr(ckpt_cb, "best_model_score", None) or 0.0)
    print(f"[reload check] reloaded val_loss = {vloss:.4f}  (monitored best = {monitored:.4f}, "
          f"Δ={abs(vloss - monitored):.4f})", flush=True)

    # --- opener diagnostics: train (can it fit?) then held-out room split (does it generalize?) ---
    evaluate(m1, dm.train_dataset, device, tag="TRAIN (reloaded best ckpt)")
    evaluate(m1, dm.val_dataset, device, tag="VAL held-out rooms (reloaded best ckpt)")


if __name__ == "__main__":
    main()
