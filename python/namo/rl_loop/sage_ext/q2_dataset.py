"""Dataset + datamodule for the Q2 depth-<=2 dense VALUE field (regression, reachable-only).

Reads the POOLED Q2 H5 (scripts/pipeline/pool_q2_h5.py = rung-1 + filtered rung-2) — one row per
episode/tree-node with a DENSE 60x5 value field — and feeds the reused sage ClassifierModule the
dict it consumes for the REACHABLE-ONLY VALUE objective ("how good is each shove, in gamma^k?"):

  f_labels  = value_target          (real-valued target: opener/finisher=1, setup=gamma=0.9, dead=0)
  loss_mask = value_mask * r_mask   (TRIED reachable cells; excludes the -1 unreachable band AND the
              reachable-but-untried MASK cells -> on loss_mask the target is in {0, 0.9, 1})
  r_mask    = r_mask                (legal/reachable cells; carried for the module's API)

Trained with head_mode="hl_gauss" (value_bins=51, range [0,1]) so the (B,60,5,bins) head regresses
the gamma^k target as a distributional value (Imani & White "Stop Regressing"); the loss is a masked
cross-entropy restricted to loss_mask, so the -1 band and untried cells carry NO gradient (the
reachable-only arm; -1 fold-in deferred). This mirrors Q1's plumbing EXACTLY (same net, same room-
grouped split, same spawn-context dataloader) and only swaps the OBJECTIVE (BCE opener -> value MSE/CE).

Train/val split is ROOM-grouped by the H5 `xml` field (a scene's rows never straddle). Dataloader
reuses the spawn-context + persistent_workers + timeout fix (the V-head-hang fix, commit 9191960);
requires a __main__-guarded entry script (scripts/rl_loop/train_q2.py is).
"""
import multiprocessing as mp
import random
from typing import List, Optional

import os
import h5py
import numpy as np
import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader

NUM_DEPTHS = 5
Q2_POS_WEIGHT = float(os.environ.get("Q2_POS_WEIGHT", "1.0"))   # up-weight positive (opener/setup) cells vs dead(0)


class Q2ValueDataset(Dataset):
    def __init__(self, h5_path: str, indices: List[int]):
        self.h5_path = h5_path
        self.indices = indices
        self._h5 = None

    def __len__(self):
        return len(self.indices)

    def _f(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, k):
        i = self.indices[k]
        f = self._f()
        ctx = torch.from_numpy(f["ctx"][i].astype(np.float32))             # (5,64,64)
        r_mask = torch.from_numpy(f["r_mask"][i].astype(np.float32))        # (60,5)
        v_tgt = torch.from_numpy(f["value_target"][i].astype(np.float32))   # (60,5) {-1,0,0.9,1}
        v_mask = torch.from_numpy(f["value_mask"][i].astype(np.float32))    # (60,5) {0,1}
        loss_mask = v_mask * r_mask                                         # reachable-and-tried cells
        out = {
            "context": ctx,
            "f_labels": v_tgt,                                             # REGRESSION target (gamma^k)
            "r_mask": r_mask,
            "loss_mask": loss_mask,
            "ratio": 0.0,                                                  # metrics binning placeholder
            "weight": 1.0 + (Q2_POS_WEIGHT - 1.0) * (v_tgt > 0).float(),   # up-weight opener/setup cells vs dead(0)
        }
        if "contact_px" in f:
            out["contact_px"] = torch.from_numpy(f["contact_px"][i].astype(np.float32))
        return out


class Q2DataModule(pl.LightningDataModule):
    def __init__(self, h5_path: str, batch_size: int = 256, num_workers: int = 8,
                 train_split: float = 0.9, pin_memory: bool = True, split_seed: int = 0, **_):
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.pin_memory = pin_memory
        self.split_seed = split_seed
        self.train_dataset = None
        self.val_dataset = None
        self.train_idx: List[int] = []
        self.val_idx: List[int] = []

    def setup(self, stage: Optional[str] = None):
        with h5py.File(self.h5_path, "r") as h5:
            n = int(h5.attrs.get("n_samples", h5["ctx"].shape[0]))
            xml = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["xml"][:]]
        groups = {}
        for i in range(n):
            groups.setdefault(xml[i], []).append(i)
        keys = sorted(groups)
        random.Random(self.split_seed).shuffle(keys)
        target = int(n * self.train_split)
        train_idx, val_idx, cum = [], [], 0
        for k in keys:                          # whole rooms to one side (no episode straddles a split)
            if cum < target:
                train_idx += groups[k]; cum += len(groups[k])
            else:
                val_idx += groups[k]
        if not val_idx:                         # tiny corpora (smoke): borrow a slice for a monitor
            val_idx = train_idx[-max(1, len(train_idx) // 10):]
        print(f"[q2 setup] rows={n} rooms={len(groups)} "
              f"train={len(train_idx)} val={len(val_idx)} "
              f"(val rooms={len(set(xml[i] for i in val_idx))})", flush=True)
        self.train_idx, self.val_idx = train_idx, val_idx
        self.train_dataset = Q2ValueDataset(self.h5_path, train_idx)
        self.val_dataset = Q2ValueDataset(self.h5_path, val_idx)

    def _kw(self):
        kw = dict(num_workers=self.num_workers, pin_memory=self.pin_memory)
        if self.num_workers > 0:
            # SPAWN (not fork) after torch/CUDA are live + persistent_workers + timeout: the
            # V-head-hang fix (commit 9191960). Requires a __main__-guarded entry script.
            kw.update(prefetch_factor=4,
                      multiprocessing_context=mp.get_context("spawn"),
                      persistent_workers=True,
                      timeout=300)
        return kw

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          drop_last=False, **self._kw())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, **self._kw())
