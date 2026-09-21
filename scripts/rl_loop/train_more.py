#!/usr/bin/env python3
"""Train MORE's push network on its own search returns.

`lifelong_trainer.py` regresses a dense value map with SmoothL1(beta=0.8), multiplies
the per-pixel loss by a weight map, sums it and divides by the batch. The weight is the
action's visit count capped at 50 (`dataset.py:191`), so an action the search barely
examined barely moves the network.

One sample is ONE action, as in their dataset. The image is rotated so that push points
in a canonical direction and the label is painted into a 3x3 patch at the rotated
contact pixel (`dataset.py:376`); everything else in the weight map is zero, so only
that patch produces gradient. Rotating per action is how a position-only head encodes
direction, and their deploy path does the same thing (`mcts_utils.py:911` rotates by
each action's own angle and batches).

Scale note. MORE had 65,384 logged actions and trained ~50 epochs, so about 3.3M
samples in total. Our action library is far wider, so one pass over a campaign's logged
actions is already comparable to their entire run. Epoch counts here are not comparable
to theirs and the card should say so.

  python train_more.py --h5 more_train.h5 --out runs/more_s1 --epochs 4
"""
import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

import sys
REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python",):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from namo.rl_loop.more_net import MorePushNet  # noqa: E402

LOSS_BETA = 0.8        # lifelong_trainer.py:242 SmoothL1Loss(beta=...)
PATCH = 1              # dataset.py paints best_loc-1 : best_loc+2, a 3x3 patch
NUM_EDGES, NUM_DEPTHS = 60, 5


class MoreActionDataset(Dataset):
    """One item per logged action, mirroring LifelongDataset.

    The index is built once over every cell carrying evidence. `include_unsampled`
    decides whether actions nobody tried are part of it: MORE writes them with label 0
    and weight 1, which reads as "this push fails" when it only means "nobody looked".
    Keeping them is faithful, dropping them is what this project's own labelling does,
    and the flag exists so the card can report both rather than argue about it.
    """

    def __init__(self, h5_path, include_unsampled=True, include_unreachable=True,
                 unreachable_cap=30, seed=0):
        self.path = h5_path
        self._f = None
        with h5py.File(h5_path, "r") as f:
            evidence = f["evidence"][:]
            reach = f["reach_mask"][:]
            weight = f["weight"][:]
            self.n_states = evidence.shape[0]
        keep = np.zeros_like(weight, dtype=bool)
        keep |= evidence == 1
        if include_unsampled:
            keep |= (reach > 0) & (evidence == 0)
        if include_unreachable:
            # Cap the free geometric negatives per state. There are 233 of them against
            # 16.5 cells backed by real physics, so uncapped they are 78% of the corpus.
            # HY5U reads all 233 off one board for one forward pass; MORE pays a pass per
            # sample, so leaving them uncapped spends its whole training budget relearning
            # the same highly redundant geometry. The cap keeps the signal and the cost
            # proportionate, and is a named deviation like the rest.
            rng = np.random.default_rng(seed)
            un = (reach == 0) & (weight > 0)
            for s_i in range(un.shape[0]):
                cells = np.argwhere(un[s_i])
                if len(cells) > unreachable_cap:
                    drop = rng.choice(len(cells), len(cells) - unreachable_cap, replace=False)
                    un[s_i][tuple(cells[drop].T)] = False
            keep |= un
        self.index = np.argwhere(keep).astype(np.int32)       # (n, 3) state, edge, depth

    def __len__(self):
        return len(self.index)

    def _file(self):
        if self._f is None:                                   # workers open their own handle
            self._f = h5py.File(self.path, "r")
        return self._f

    def __getitem__(self, i):
        s, e, d = (int(v) for v in self.index[i])
        f = self._file()
        ctx = torch.from_numpy(f["ctx"][s].astype(np.float32))
        px = f["contact_px"][s][e]
        label = float(f["label"][s][e, d])
        weight = float(f["weight"][s][e, d])
        # The push direction is the contact's outward normal, which for our library is
        # fixed by the edge index: 60 edges evenly around the object.
        angle = 2.0 * math.pi * e / NUM_EDGES
        return ctx, torch.tensor(px, dtype=torch.float32), torch.tensor(angle), \
            torch.tensor(label), torch.tensor(weight), torch.tensor(d)


def rotate_batch(ctx, angle):
    """Rotate each image by its own angle, as utils.rotate does before the forward pass."""
    b = ctx.shape[0]
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(b, 2, 3, device=ctx.device, dtype=ctx.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    grid = F.affine_grid(theta, ctx.shape, align_corners=True)
    return F.grid_sample(ctx, grid, mode="bilinear", align_corners=True)


def rotate_pixel(px, angle, size):
    """Where a pixel lands after the same rotation, so the patch is painted correctly."""
    centre = (size - 1) / 2.0
    y, x = px[:, 0] - centre, px[:, 1] - centre
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack([y * cos - x * sin + centre, y * sin + x * cos + centre], dim=1)


def paint(target, weight_map, rpx, depth, label, weight):
    """Write the label into a 3x3 patch at the rotated pixel; zero weight elsewhere."""
    b, _d, h, w = target.shape
    rows = rpx[:, 0].round().long().clamp(PATCH, h - 1 - PATCH)
    cols = rpx[:, 1].round().long().clamp(PATCH, w - 1 - PATCH)
    for i in range(b):
        r0, r1 = rows[i] - PATCH, rows[i] + PATCH + 1
        c0, c1 = cols[i] - PATCH, cols[i] + PATCH + 1
        target[i, depth[i], r0:r1, c0:c1] = label[i]
        weight_map[i, depth[i], r0:r1, c0:c1] = weight[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--drop-unsampled", action="store_true",
                    help="exclude actions nobody tried; MORE keeps them at label 0 weight 1")
    ap.add_argument("--drop-unreachable", action="store_true",
                    help="exclude the free geometric negatives")
    ap.add_argument("--unreachable-cap", type=int, default=30,
                    help="free geometric negatives kept per state (233 exist); 0 keeps all")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ds = MoreActionDataset(args.h5, include_unsampled=not args.drop_unsampled,
                           include_unreachable=not args.drop_unreachable,
                           unreachable_cap=(args.unreachable_cap or 10**9), seed=args.seed)
    with h5py.File(args.h5, "r") as f:
        in_ch, size = f["ctx"].shape[1], f["ctx"].shape[-1]
    print(f"states {ds.n_states}  samples {len(ds)}  ctx {in_ch}x{size}x{size}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MorePushNet(in_channels=in_ch, num_depths=NUM_DEPTHS).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=(dev == "cuda"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        net.train()
        total, seen = 0.0, 0
        for ctx, px, angle, label, weight, depth in loader:
            ctx, angle = ctx.to(dev), angle.to(dev)
            px, label, weight, depth = px.to(dev), label.to(dev), weight.to(dev), depth.to(dev)
            rot = rotate_batch(ctx, angle)
            rpx = rotate_pixel(px, angle, size)
            target = torch.zeros(ctx.shape[0], NUM_DEPTHS, size, size, device=dev)
            wmap = torch.zeros_like(target)
            paint(target, wmap, rpx, depth, label, weight)

            pred = net(rot)
            # lifelong_trainer.py:356 -- elementwise SmoothL1 times the weight map,
            # summed and divided by the batch, NOT averaged over pixels. Zero-weight
            # pixels contribute nothing, which is how only the painted patch trains.
            loss = F.smooth_l1_loss(pred, target, beta=LOSS_BETA, reduction="none") * wmap
            loss = loss.sum() / target.shape[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * target.shape[0]
            seen += target.shape[0]
        mean = total / max(seen, 1)
        history.append({"epoch": epoch, "loss": mean, "samples": seen})
        print(f"epoch {epoch} loss {mean:.5f} over {seen} samples", flush=True)
        torch.save({"model": net.state_dict(), "epoch": epoch, "args": vars(args)},
                   out / f"more_epoch{epoch:03d}.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
