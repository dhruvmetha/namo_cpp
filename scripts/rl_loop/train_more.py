#!/usr/bin/env python3
"""Train MORE's push network (PushNet, a dense ResNet-34 FPN) on MORE's own search trees.

Reference: lifelong_trainer.py and dataset.py at arc-l/more 70402c0; paper section IV-C.

One sample is ONE row of a MORE tree (dataset.py:150-193): a simulated action with its label
max(child.q) and weight min(visits, 50). The state's picture is rotated so THAT push points right
(dataset.py:69-87), the label is painted into a 3x3 patch at the rotated contact pixel, centre at
full weight and the 8 neighbours at half (dataset.py:414-422), and the loss is SmoothL1(beta=0.8)
times the weight map, summed and divided by the batch (lifelong_trainer.py:356-357). Pixels
nobody labelled carry weight 0. Actions the tree never simulated have no row and no loss: MORE's
collection tree has no such children (mcts/nodes.py:47-63).

Coordinates. contact_px is (column, row), as eval_scorer.contact_px writes it and EdgeCrossAttn
reads it. Our 60 edges are 4 faces x 15 points and push along the face normal toward the mate
edge e ^ 1 (namo_push_controller.cpp:178-210, :258-260), which is the angle we rotate by.

Validation. The rooms verify_more_net.py scores (heldout_rooms, 10% at seed 0) are excluded here.

Open choices, PROVISIONAL defaults (python/namo/rl_loop/more_training.py):
  --invalid ours|more     unreachable-cell negatives
  --schedule ours|more    the pilot's recipe, or MORE's 51-epoch run plus the 21-epoch fine-tune

  python train_more.py --h5 more_train.h5 --out runs/more_s1 [--schedule more] [--invalid more]
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python",):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo.rl_loop.more_net import MorePushNet  # noqa: E402
from namo.rl_loop.more_training import (  # noqa: E402
    INVALID_MODES, LOSS_BETA, MORE_INVALID_WEIGHT, NUM_DEPTHS, SCHEDULES, Stepper, decode,
    heldout_rooms, invalid_cells, make_optimizer, push_angle, rotate_batch, rotate_pixel, stages)

PATCH = 1              # dataset.py:414-418 paints best_loc-1 : best_loc+2, a 3x3 patch


class MoreActionDataset(Dataset):
    """One item per tree row (dataset.py:150-193), plus "ours" negatives when asked for.

    Rows are the H5 cells with evidence 1 and weight > 0. A legal action the tree never simulated
    (evidence 0) is never an item. `exclude_rooms` drops whole rooms (the verify split).
    """

    def __init__(self, h5_path, invalid="ours", invalid_cap=30, seed=0, exclude_rooms=()):
        if invalid not in INVALID_MODES:
            raise ValueError(f"invalid must be one of {INVALID_MODES}")
        self.path, self.invalid = h5_path, invalid
        self._f = None
        with h5py.File(h5_path, "r") as f:
            evidence = f["evidence"][:]
            reach = f["reach_mask"][:]
            weight = f["weight"][:]
            xml = decode(f["xml"][:])
        self.n_states = evidence.shape[0]
        rows = (evidence == 1) & (weight > 0)
        neg = invalid_cells(reach, weight, invalid, invalid_cap, np.random.default_rng(seed))
        if exclude_rooms:
            out = np.isin(xml, list(exclude_rooms))
            rows[out] = False
            neg[out] = False
        self.index = np.concatenate([np.argwhere(rows), np.argwhere(neg)]).astype(np.int32)
        self.is_neg = np.concatenate([np.zeros(int(rows.sum()), bool), np.ones(int(neg.sum()), bool)])

    def __len__(self):
        return len(self.index)

    def _file(self):
        if self._f is None:                                   # workers open their own handle
            self._f = h5py.File(self.path, "r")
        return self._f

    def __getitem__(self, i):
        s, e, d = (int(v) for v in self.index[i])
        f = self._file()
        board = f["contact_px"][s].astype(np.float32)
        if self.is_neg[i]:
            label, weight = -1.0, 1.0                         # "ours" negative (PROVISIONAL)
        else:
            label, weight = float(f["label"][s][e, d]), float(f["weight"][s][e, d])
        unreachable = (f["reach_mask"][s] == 0).astype(np.float32) if self.invalid == "more" \
            else np.zeros((60, NUM_DEPTHS), np.float32)
        return (torch.from_numpy(f["ctx"][s].astype(np.float32)), torch.from_numpy(board),
                torch.tensor(push_angle(board, e)), torch.tensor(e), torch.tensor(d),
                torch.tensor(label), torch.tensor(weight), torch.from_numpy(unreachable))


def paint(target, weight_map, rpx, depth, label, weight):
    """dataset.py:414-422: the label on a 3x3 patch, weight w at the centre and w/2 on the 8
    neighbours. MORE paints `label if label > 0 else 0`; tree labels are max(child.q) >= 0, so
    only an "ours" negative (-1, PROVISIONAL) is ever below zero, and it is kept."""
    b, _d, h, w = target.shape
    rows = rpx[:, 1].round().long().clamp(PATCH, h - 1 - PATCH)     # rpx is (column, row)
    cols = rpx[:, 0].round().long().clamp(PATCH, w - 1 - PATCH)
    for i in range(b):
        r, c, d = rows[i], cols[i], depth[i]
        target[i, d, r - PATCH:r + PATCH + 1, c - PATCH:c + PATCH + 1] = label[i]
        weight_map[i, d, r - PATCH:r + PATCH + 1, c - PATCH:c + PATCH + 1] = weight[i] * 0.5
        weight_map[i, d, r, c] = weight[i]


def paint_invalid(target, weight_map, rboard, unreachable):
    """dataset.py:423-430 for our library: every unreachable cell of the state -> label 0 at 0.001.

    rboard (B, 60, 2) rotated contact pixels, unreachable (B, 60, D). MORE applies its invalid masks
    after the action patch, so they win where they overlap; ours follows that order.
    """
    b, _d, h, w = target.shape
    idx = unreachable.nonzero(as_tuple=False)                 # (n, 3): sample, edge, depth
    if idx.numel() == 0:
        return
    px = rboard[idx[:, 0], idx[:, 1]]
    rows = px[:, 1].round().long().clamp(PATCH, h - 1 - PATCH)
    cols = px[:, 0].round().long().clamp(PATCH, w - 1 - PATCH)
    for dr in range(-PATCH, PATCH + 1):
        for dc in range(-PATCH, PATCH + 1):
            target[idx[:, 0], idx[:, 2], rows + dr, cols + dc] = 0.0
            weight_map[idx[:, 0], idx[:, 2], rows + dr, cols + dc] = MORE_INVALID_WEIGHT


def batch_loss(net, batch, dev, size):
    ctx, board, angle, edge, depth, label, weight, unreachable = (t.to(dev) for t in batch)
    rot = rotate_batch(ctx, angle)
    rboard = rotate_pixel(board, angle[:, None], size)                    # (B, 60, 2)
    rpx = rboard[torch.arange(len(edge), device=dev), edge.long()]
    target = torch.zeros(ctx.shape[0], NUM_DEPTHS, size, size, device=dev)
    wmap = torch.zeros_like(target)
    paint(target, wmap, rpx, depth.long(), label, weight)
    paint_invalid(target, wmap, rboard, unreachable)
    pred = net(rot)
    loss = F.smooth_l1_loss(pred, target, beta=LOSS_BETA, reduction="none") * wmap
    return loss.sum() / target.shape[0]                       # lifelong_trainer.py:356-357


def train_stage(net, ds, stage, args, dev, size, out, history, tag):
    opt, sched = make_optimizer(net.parameters(), stage)
    loader = DataLoader(ds, batch_size=stage.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=(dev == "cuda"))
    stepper = Stepper(opt, sched, stage, len(loader))
    for epoch in range(stage.epochs):
        net.train()
        total, seen, recent, t0 = 0.0, 0, deque(maxlen=200), time.time()
        for step, batch in enumerate(loader):
            loss = batch_loss(net, batch, dev, size)
            opt.zero_grad()
            loss.backward()
            if stage.clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), stage.clip)
            opt.step()
            stepper.after_iteration(epoch, step)
            n = batch[0].shape[0]
            total += loss.item() * n
            seen += n
            recent.append(loss.item())
            if step % 200 == 0:
                print(f"  {tag} e{epoch} step {step}/{len(loader)} recent {sum(recent)/len(recent):.5f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {seen/max(time.time()-t0, 1e-9):.0f} samples/s",
                      flush=True)
                recent.clear()
        stepper.after_epoch(epoch)
        history.append({"stage": tag, "epoch": epoch, "loss": total / max(seen, 1), "samples": seen})
        print(f"{tag} epoch {epoch} loss {total/max(seen, 1):.5f} over {seen} samples", flush=True)
        torch.save({"model": net.state_dict(), "stage": tag, "epoch": epoch, "args": vars(args)},
                   out / f"{tag}_epoch{epoch:03d}.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schedule", choices=sorted(SCHEDULES), default="ours",
                    help="PROVISIONAL default 'ours' (the pilot's recipe); 'more' = lifelong_trainer.py + fine-tune")
    ap.add_argument("--invalid", choices=["ours", "more"], default="ours",
                    help="PROVISIONAL default 'ours' (label -1, weight 1, capped); 'more' = dataset.py:423-430")
    ap.add_argument("--invalid-cap", type=int, default=30, help="'ours' negatives kept per state; 0 = all")
    ap.add_argument("--epochs", type=int, default=None, help="override the first stage's epochs")
    ap.add_argument("--batch", type=int, default=None, help="override the first stage's batch")
    ap.add_argument("--lr", type=float, default=None, help="override the first stage's learning rate")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--holdout-frac", type=float, default=0.1,
                    help="rooms withheld for verify_more_net.py (same rule, seed --holdout-seed)")
    ap.add_argument("--holdout-seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    with h5py.File(args.h5, "r") as f:
        held = heldout_rooms(decode(f["xml"][:]), args.holdout_frac, args.holdout_seed)
        in_ch, size = f["ctx"].shape[1], f["ctx"].shape[-1]
    ds = MoreActionDataset(args.h5, invalid=args.invalid, invalid_cap=args.invalid_cap,
                           seed=args.seed, exclude_rooms=held)
    print(f"states {ds.n_states}  samples {len(ds)} (negatives {int(ds.is_neg.sum())})  "
          f"held-out rooms {len(held)}  ctx {in_ch}x{size}x{size}  schedule {args.schedule}  "
          f"invalid {args.invalid}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MorePushNet(in_channels=in_ch, num_depths=NUM_DEPTHS).to(dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for stage in stages(args.schedule, args.epochs, args.batch, args.lr):
        train_stage(net, ds, stage, args, dev, size, out, history, stage.name)
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "heldout_rooms.json").write_text(json.dumps(sorted(held)))
    print("done", flush=True)


if __name__ == "__main__":
    main()
