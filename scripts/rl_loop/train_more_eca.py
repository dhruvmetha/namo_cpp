#!/usr/bin/env python3
"""Train our EdgeCrossAttn on MORE's own labels and loss, so its output is MORE's prior.

The network is the Q2/HY5U architecture (namo.rl_loop.train_gen._make_network) with a scalar
head: one value per (edge, depth), no HL-Gauss bins, no sigmoid. The targets and the loss are
MORE's (lifelong_trainer.py and dataset.py at arc-l/more 70402c0, paper IV-C):

  target  max(child.q) of each tree row, in push_result units (0 to 1.2)
  weight  min(visits, 50) (dataset.py:191); 0 for every cell without a row
  loss    SmoothL1(beta=0.8) x weight, summed, divided by the batch (lifelong_trainer.py:356-357)

What does not carry over, and why. MORE's PushNet is a dense pixel map, so each sample is one
action, the image is rotated so that push points right, and the label is painted on a 3x3
patch. EdgeCrossAttn already emits one value per (edge, depth) from a contact token, so a sample
here is one state with all of its rows, and there is no rotation or painting. The per-cell
weighting is MORE's; what a batch holds is not (8 actions in MORE, B states with all their rows
here).

Input: the more_build_h5.py H5 rendered at --render-size 64 (EdgeCrossAttn's input size).
Deploy: `--prior more_eca --planner more` in eval_bestfirst.py reads the checkpoint through
more_scorer.MoreEcaScorer, which returns the raw output (LiveScorer would apply a sigmoid).

Open choices, PROVISIONAL defaults (python/namo/rl_loop/more_training.py):
  --invalid ours|more     unreachable-cell negatives
  --schedule ours|more    our EdgeCrossAttn recipe (AdamW, Q2's numbers), or MORE's SGD schedule

  python train_more_eca.py --h5 more_trees_64.h5 --out runs/more_eca_s1 [--schedule more]
"""
import argparse
import json
import math
import sys
import time
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

from namo.rl_loop.more_training import (  # noqa: E402
    INVALID_MODES, LOSS_BETA, MORE_INVALID_WEIGHT, SCHEDULES, Stepper, decode, heldout_rooms,
    invalid_cells, make_optimizer, stages)

# PROVISIONAL default for this trainer: train_q2.py's numbers for the same network
# (AdamW, lr 3e-4, weight decay 0.01, 200 warmup steps, batch 256, 40 epochs).
ECA_OURS = dict(epochs=40, batch=256, lr=3e-4, weight_decay=0.01, warmup=200, end_lr=1e-6)


class MoreBoardDataset(Dataset):
    """One item per saved tree state: its picture, contact pixels, and MORE's per-cell target."""

    def __init__(self, h5_path, invalid="ours", invalid_cap=30, seed=0, exclude_rooms=()):
        if invalid not in INVALID_MODES:
            raise ValueError(f"invalid must be one of {INVALID_MODES}")
        self.path, self.invalid, self._f = h5_path, invalid, None
        with h5py.File(h5_path, "r") as f:
            evidence = f["evidence"][:]
            reach = f["reach_mask"][:]
            weight = f["weight"][:]
            xml = decode(f["xml"][:])
        rows = (evidence == 1) & (weight > 0)
        self.neg = invalid_cells(reach, weight, invalid, invalid_cap, np.random.default_rng(seed))
        keep = rows.any(axis=(1, 2))
        if exclude_rooms:
            keep &= ~np.isin(xml, list(exclude_rooms))
        self.states = np.flatnonzero(keep).astype(np.int64)
        self.n_rows = int(rows[self.states].sum())

    def __len__(self):
        return len(self.states)

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def __getitem__(self, i):
        s = int(self.states[i])
        f = self._file()
        rows = (f["evidence"][s] == 1) & (f["weight"][s] > 0)
        target = np.where(rows, f["label"][s], 0.0).astype(np.float32)
        weight = np.where(rows, f["weight"][s], 0.0).astype(np.float32)
        if self.invalid == "more":                         # dataset.py:423-430, all invalid cells
            un = f["reach_mask"][s] == 0
            target[un], weight[un] = 0.0, MORE_INVALID_WEIGHT
        else:                                              # PROVISIONAL "ours" negatives
            target[self.neg[s]], weight[self.neg[s]] = -1.0, 1.0
        return (torch.from_numpy(f["ctx"][s].astype(np.float32)),
                torch.from_numpy(f["contact_px"][s].astype(np.float32)),
                torch.from_numpy(target), torch.from_numpy(weight))


def more_loss(pred, target, weight):
    """lifelong_trainer.py:356-357 over cells instead of pixels."""
    return (F.smooth_l1_loss(pred, target, beta=LOSS_BETA, reduction="none") * weight).sum() / pred.shape[0]


def make_net():
    from namo.rl_loop.train_gen import _make_network
    return _make_network(value_bins=0)


def save_ckpt(net, path, meta):
    """A checkpoint eval_scorer.load_scorer reads: ClassifierModule's own state_dict layout."""
    from namo.rl_loop.sage_ext._sage import ClassifierModule
    module = ClassifierModule(network=net)
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": {},
                "action_motion_encoding": getattr(net, "action_motion_encoding", None),
                "more_eca": meta}, path)


def _adamw(net, n_iters, cfg):
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    def lam(step):
        if step < cfg["warmup"]:
            return (step + 1) / cfg["warmup"]
        p = min(1.0, (step - cfg["warmup"]) / max(1, n_iters - cfg["warmup"]))
        floor = cfg["end_lr"] / cfg["lr"]
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lam)


def run(net, ds, args, dev, out, history):
    if args.schedule == "ours":
        cfg = dict(ECA_OURS, **{k: v for k, v in (("epochs", args.epochs), ("batch", args.batch),
                                                    ("lr", args.lr)) if v is not None})
        loader = DataLoader(ds, batch_size=cfg["batch"], shuffle=True, num_workers=args.workers, drop_last=False)
        opt, sched = _adamw(net, cfg["epochs"] * len(loader), cfg)
        plan = [("ours", cfg["epochs"], loader, opt, lambda e, s: sched.step(), lambda e: None)]
    else:
        plan = []
        for st in stages("more", args.epochs, args.batch, args.lr):
            loader = DataLoader(ds, batch_size=st.batch, shuffle=True, num_workers=args.workers, drop_last=False)
            opt, sched = make_optimizer(net.parameters(), st)
            stepper = Stepper(opt, sched, st, len(loader))
            plan.append((st.name, st.epochs, loader, opt, stepper.after_iteration, stepper.after_epoch))
    for tag, epochs, loader, opt, after_iter, after_epoch in plan:
        for epoch in range(epochs):
            net.train()
            total, seen, t0 = 0.0, 0, time.time()
            for step, (ctx, cpx, target, weight) in enumerate(loader):
                ctx, cpx, target, weight = (t.to(dev) for t in (ctx, cpx, target, weight))
                loss = more_loss(net(ctx, cpx), target, weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                after_iter(epoch, step)
                total += loss.item() * ctx.shape[0]
                seen += ctx.shape[0]
            after_epoch(epoch)
            history.append({"stage": tag, "epoch": epoch, "loss": total / max(seen, 1), "states": seen,
                            "seconds": round(time.time() - t0, 2)})
            print(f"{tag} epoch {epoch} loss {total/max(seen, 1):.5f} states {seen} "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
            save_ckpt(net, out / f"{tag}_epoch{epoch:03d}.ckpt",
                      {"stage": tag, "epoch": epoch, "schedule": args.schedule, "invalid": args.invalid})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schedule", choices=sorted(SCHEDULES), default="ours",
                    help="PROVISIONAL default 'ours' (train_q2.py's AdamW recipe); 'more' = lifelong_trainer.py + fine-tune")
    ap.add_argument("--invalid", choices=list(INVALID_MODES), default="ours",
                    help="PROVISIONAL default 'ours' (label -1, weight 1, capped); 'more' = label 0 at 0.001")
    ap.add_argument("--invalid-cap", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--holdout-seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    with h5py.File(args.h5, "r") as f:
        size = int(f["ctx"].shape[-1])
        held = heldout_rooms(decode(f["xml"][:]), args.holdout_frac, args.holdout_seed)
    if size != 64:
        raise SystemExit(f"EdgeCrossAttn reads 64 px crops; this H5 is {size} px (more_build_h5.py --render-size 64)")
    ds = MoreBoardDataset(args.h5, args.invalid, args.invalid_cap, args.seed, exclude_rooms=held)
    print(f"states {len(ds)}  rows {ds.n_rows}  held-out rooms {len(held)}  schedule {args.schedule}  "
          f"invalid {args.invalid}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = make_net().to(dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    run(net, ds, args, dev, out, history)
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "heldout_rooms.json").write_text(json.dumps(sorted(held)))
    print("done", flush=True)


if __name__ == "__main__":
    main()
