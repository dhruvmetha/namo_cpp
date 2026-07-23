#!/usr/bin/env python3
"""Train Q2 — the pooled depth-<=2 dense VALUE field (regression over gamma^k, REACHABLE-ONLY).

Q2 reuses Q1's EXACT plumbing (e4 EdgeCrossAttn, WeightedClassifierModule, room-grouped datamodule,
spawn-context dataloader) and only swaps the OBJECTIVE: BCE opener -> distributional VALUE regression.

  head  : hl_gauss value head (value_bins=51, range [0,1]) -> net emits (B,60,5,51); inference
          value = E[bin] in [0,1]  (Imani & White "Stop Regressing" 2403.03950).
  target: value_target (opener/finisher=1, setup=gamma=0.9, dead=0) on the REACHABLE-AND-TRIED cells
          (loss_mask = value_mask * r_mask). The -1 unreachable band and untried MASK carry no
          gradient (the reachable-only arm; -1 fold-in deferred). Loss = masked CE to the Gaussian-
          smoothed target histogram (== value regression; the repo's canonical value head).

Why hl_gauss over plain MSE: it is the repo's registered value head, already fully wired (loss / val
monitor / metrics) with ZERO sage edits, and eval_scorer auto-detects it (value_bins = head_out //
num_depths) and applies HLGauss.value -> (60,5). MSE would need a new head_mode AND would break the
BCE val-monitor + eval_scorer's sigmoid path. hl_gauss is strictly cleaner and eval-compatible.

__main__-guarded so the spawn-context dataloader workers (the V-head-hang fix) re-import cleanly.

Usage (smoke, arrakis GPU4):
  CUDA_VISIBLE_DEVICES=4 python scripts/rl_loop/train_q2.py \
    --h5 <pooled_smoke.h5> --out-dir <run_dir> --epochs 60 --batch-size 64 --num-workers 2
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
from namo.rl_loop.sage_ext.q2_dataset import Q2DataModule        # noqa: E402
from namo.rl_loop.sage_ext._sage import EdgeCrossAttn, HLGauss, ClassifierModule  # noqa: E402

VALUE_BINS = 51
GAMMA = 0.9


def build_module(base_lr: float, warmup_steps: int, decay_steps: int) -> WeightedClassifierModule:
    net = _make_network(value_bins=VALUE_BINS)   # (B,60,5,51) hl_gauss head, sharp/e4 recipe
    return WeightedClassifierModule(
        network=net, base_lr=base_lr, weight_decay=0.01,
        warmup_steps=warmup_steps, decay_steps=decay_steps, end_lr=1e-6,
        head_mode="hl_gauss", value_vmin=0.0, value_vmax=1.0, dice_weight=0.0,  # value regression
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
# value-ranking diagnostics on the held-out room split (over reachable-and-tried cells)
# ---------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")

    def _rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(1, len(x) + 1)
        # average-rank tie correction
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        avg = np.empty(len(counts)); lo = 0
        for j, c in enumerate(counts):
            avg[j] = (lo + 1 + lo + c) / 2.0; lo += c
        return avg[inv]
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


@torch.no_grad()
def evaluate(module, ds, device, tag="val"):
    module.eval().to(device)
    hg = HLGauss(num_bins=VALUE_BINS, vmin=0.0, vmax=1.0)
    preds, tgts = [], []
    # reachable top-1: over rows that HAVE a win among tried cells, is the highest-value reachable
    # cell a win (target==1) / a win-or-setup (target>=0.9)?
    r_top1_win = r_top1_winsetup = r_top1_rows = 0
    for k in range(len(ds)):
        b = ds[k]
        ctx = b["context"].unsqueeze(0).to(device)
        cpx = b.get("contact_px")
        cpx = cpx.unsqueeze(0).to(device) if cpx is not None else None
        am = b.get("action_motion")
        am = am.unsqueeze(0).to(device) if am is not None else None
        logits = module(ctx, cpx, action_motion=am)       # (1,60,5,51)
        val = hg.value(logits.float())[0].cpu()          # (60,5) predicted value in [0,1]
        f = b["f_labels"]; lm = b["loss_mask"]; rm = b["r_mask"]
        tried = lm > 0.5
        preds.append(val[tried].numpy()); tgts.append(f[tried].numpy())
        if float((f[tried] >= 0.999).sum()) > 0:         # row has >=1 win (target==1) among tried
            r_top1_rows += 1
            reach = rm > 0.5
            vv = val.clone(); vv[~reach] = -1.0          # rank only reachable cells (deploy-realistic)
            idx = int(vv.reshape(-1).argmax())
            t = float(f.reshape(-1)[idx])
            r_top1_win += int(t >= 0.999)
            r_top1_winsetup += int(t >= GAMMA - 0.05)
    p = np.concatenate(preds) if preds else np.zeros(0)
    y = np.concatenate(tgts) if tgts else np.zeros(0)

    def _bucket(lo, hi):
        m = (y >= lo) & (y < hi)
        return int(m.sum()), (float(p[m].mean()) if m.any() else float("nan"))
    n_dead, mp_dead = _bucket(-0.05, 0.05)               # target == 0
    n_setup, mp_setup = _bucket(GAMMA - 0.05, GAMMA + 0.05)  # target == 0.9
    n_win, mp_win = _bucket(0.95, 1.05)                  # target == 1
    rho = _spearman(p, y)
    r_top1 = r_top1_win / r_top1_rows if r_top1_rows else float("nan")
    r_top1ws = r_top1_winsetup / r_top1_rows if r_top1_rows else float("nan")

    print(f"\n===== Q2 value-ranking diagnostics ({tag}, over reachable-and-tried cells) =====")
    print(f"  trained cells        : {len(y)}")
    print(f"  Spearman rho(pred,tgt): {rho:.4f}   (1.0 = perfect value ordering)")
    print(f"  mean predicted value by target bucket (want dead < setup < win):")
    print(f"      dead  (tgt=0.0) : mean_pred={mp_dead:.4f}   n={n_dead}")
    print(f"      setup (tgt=0.9) : mean_pred={mp_setup:.4f}   n={n_setup}")
    print(f"      win   (tgt=1.0) : mean_pred={mp_win:.4f}   n={n_win}")
    ok = (not np.isnan(mp_dead) and not np.isnan(mp_win) and mp_win > mp_dead and
          (np.isnan(mp_setup) or (mp_setup > mp_dead and mp_win >= mp_setup - 1e-6)))
    print(f"      ordering monotone: {'YES' if ok else 'NO'}")
    print(f"  reachable top-1 (over {r_top1_rows} rows with a win among tried):")
    print(f"      P(top-1 is a win, tgt=1)      : {r_top1:.4f}")
    print(f"      P(top-1 is win-or-setup >=0.9): {r_top1ws:.4f}")
    print("=============================================================================\n", flush=True)
    return dict(spearman=rho, mp_dead=mp_dead, mp_setup=mp_setup, mp_win=mp_win,
                r_top1_win=r_top1, r_top1_winsetup=r_top1ws, n_dead=n_dead, n_setup=n_setup,
                n_win=n_win, n_tried=len(y))


def _build_net_like_eval_scorer(ck, num_depths):
    """Replicate eval_scorer.load_scorer's edge_crossattn arch auto-detect (the value_bins branch is
    what we must confirm) so we KNOW the ckpt loads the same way eval_scorer loads it."""
    sd = ck["state_dict"]
    dim = sd["network.edge_norm.weight"].shape[0]
    sdep = sum(1 for k in sd if k.startswith("network.scene_blocks.") and k.endswith(".n1.weight"))
    edep = sum(1 for k in sd if k.startswith("network.edge_blocks.") and k.endswith(".n1.weight"))
    patch = 64 // int(round(sd["network.scene_pos"].shape[1] ** 0.5))
    kw = dict(img_size=64, patch=patch, in_channels=5, num_depths=num_depths,
              dim=dim, scene_depth=sdep, edge_depth=edep, heads=dim // 32)
    if "network.local_proj.weight" not in sd:
        kw["use_local"] = False
    pin = sd["network.edge_pos.0.weight"].shape[1]
    if pin != 2:
        kw.update(pos_fourier=True, fourier_L=pin // 4)
    if "network.edge_embed.weight" in sd:
        kw["use_edge_embed"] = True
    if "network.action_motion_proj.0.weight" in sd:
        from namo.rl_loop.action_motion import action_motion_feature_dim
        motion_proj_in = sd["network.action_motion_proj.0.weight"].shape[1]
        motion_tag = ck.get("action_motion_encoding")
        motion_dim = action_motion_feature_dim(motion_tag) if motion_tag else motion_proj_in
        kw["action_motion_dim"] = motion_dim
        if motion_proj_in != motion_dim:
            kw.update(action_motion_fourier=True,
                      action_motion_fourier_L=motion_proj_in // (2 * motion_dim))
        if "network.action_depth_embed.weight" in sd:
            kw["action_depth_embed"] = True
    head_out = sd["network.head.2.weight"].shape[0]
    value_bins = None
    if head_out != num_depths:
        value_bins = head_out if kw.get("action_motion_dim", 0) else head_out // num_depths
        kw["value_bins"] = value_bins
    net = EdgeCrossAttn(**kw)
    return net, value_bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--decay-steps", type=int, default=100000)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--train-split", type=float, default=0.9)
    ap.add_argument("--patience", type=int, default=0, help="EarlyStopping patience (0=off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--postcheck-limit", type=int, default=0,
                    help="limit each post-training reload/diagnostic split; 0 keeps the full split")
    a = ap.parse_args()

    pl.seed_everything(a.seed, workers=True)
    ckpt_dir = os.path.join(a.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    module = build_module(a.lr, a.warmup_steps, a.decay_steps)
    dm = Q2DataModule(a.h5, batch_size=a.batch_size, num_workers=a.num_workers,
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
    print(f"\n[train_q2] best ckpt: {best}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dm.setup()
    from torch.utils.data import DataLoader, Subset
    train_post = dm.train_dataset
    val_post = dm.val_dataset
    if a.postcheck_limit > 0:
        train_post = Subset(train_post, range(min(a.postcheck_limit, len(train_post))))
        val_post = Subset(val_post, range(min(a.postcheck_limit, len(val_post))))

    # --- save+reload integrity (two independent reloads must be bit-identical) ---
    ck = torch.load(best, map_location="cpu", weights_only=False)

    def _reload():
        m = build_module(a.lr, a.warmup_steps, a.decay_steps)
        m.load_state_dict(ck["state_dict"])
        return m.eval().to(device)

    m1, m2 = _reload(), _reload()
    b = val_post[0]
    ctx = b["context"].unsqueeze(0).to(device)
    cpx = b.get("contact_px")
    cpx = cpx.unsqueeze(0).to(device) if cpx is not None else None
    with torch.no_grad():
        am = b.get("action_motion")
        am = am.unsqueeze(0).to(device) if am is not None else None
        d = float((m1(ctx, cpx, action_motion=am).float() -
                   m2(ctx, cpx, action_motion=am).float()).abs().max())
    print(f"[reload check] two-reload max|Δlogit| = {d:.3e} "
          f"({'OK identical' if d == 0.0 else 'NONDETERMINISTIC'})", flush=True)

    # reproduce monitored best val_loss (hl_gauss CE) with the reloaded model (one big val batch)
    vb = next(iter(DataLoader(val_post, batch_size=len(val_post))))
    vb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vb.items()}
    with torch.no_grad():
        vlogits = m1(vb["context"], vb.get("contact_px"), action_motion=vb.get("action_motion"))
        vloss = float(m1._compute_masked_loss(vlogits, vb["f_labels"], vb["loss_mask"]))
    monitored = float(getattr(ckpt_cb, "best_model_score", None) or 0.0)
    print(f"[reload check] reloaded val_loss = {vloss:.4f}  (monitored best = {monitored:.4f}, "
          f"Δ={abs(vloss - monitored):.4f}; small Δ expected — batch-vs-pooled mask normalization)",
          flush=True)

    # --- eval_scorer-loadable check: rebuild the net via eval_scorer's EXACT arch auto-detect,
    #     load the full state_dict through a stock ClassifierModule, forward, confirm (60,5) value ---
    sd = ck["state_dict"]
    net_es, vb_det = _build_net_like_eval_scorer(ck, num_depths=5)
    es_model = ClassifierModule(network=net_es, head_mode="hl_gauss",
                                value_vmin=0.0, value_vmax=1.0)
    es_model.load_state_dict(sd)   # must succeed -> arch matches
    es_model.eval().to(device)
    with torch.no_grad():
        es_logits = es_model(ctx, cpx, action_motion=am)  # (1,60,5,51)
        es_val = HLGauss(num_bins=VALUE_BINS).value(es_logits.float())  # (1,60,5)
    print(f"[eval_scorer-load check] detected value_bins={vb_det}  "
          f"logits shape={tuple(es_logits.shape)}  value shape={tuple(es_val.shape)}  "
          f"value range=[{float(es_val.min()):.3f},{float(es_val.max()):.3f}]  "
          f"({'OK (60,5) value head' if tuple(es_val.shape) == (1, 60, 5) else 'BAD SHAPE'})",
          flush=True)

    # --- value-ranking diagnostics: train (can it fit?) then held-out rooms (does it generalize?) ---
    evaluate(m1, train_post, device, tag="TRAIN (reloaded best ckpt)")
    evaluate(m1, val_post, device, tag="VAL held-out rooms (reloaded best ckpt)")


if __name__ == "__main__":
    main()
