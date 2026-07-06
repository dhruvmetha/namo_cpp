"""Dataset + datamodule that feed the RL-loop training H5 into the sage ClassifierModule.

Bespoke (not ScorerH5Dataset) because the labels are genuinely new — a CHOSEN action + a
Monte-Carlo return + per-sample weights, not an exhaustive f_grid. It emits EXACTLY the dict
the (reused, unmodified) ClassifierModule consumes, plus a `weight` field:

  mode="pi"  (filtered BC, solved rows only): f_labels = one-hot(chosen), loss_mask = r_mask
             (legal cells) -> masked softmax-CE onto the taken action; weight = pi_weight.
  mode="v"   (all rows incl. failures): f_labels = value_target at the chosen cell,
             loss_mask = one-hot(chosen) -> HL-Gauss regresses ONLY the taken action to its
             MC return (no bootstrap); weight = v_weight (recency).

Train/val split is room-grouped (by the H5 `xml` field) exactly like ScorerDataModule — a
scene's episodes never straddle. This internal val is just the checkpoint monitor; the real
signal is the held-out dev greedy-open eval (eval_gen.py).
"""
import random
from typing import List, Optional

import h5py
import numpy as np
import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader

NUM_DEPTHS = 5


class RLScorerDataset(Dataset):
    def __init__(self, h5_path: str, indices: List[int], mode: str):
        assert mode in ("pi", "v"), mode
        self.h5_path = h5_path
        self.indices = indices
        self.mode = mode
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
        ctx = torch.from_numpy(f["ctx"][i].astype(np.float32))          # (5,64,64)
        r_mask = torch.from_numpy(f["r_mask"][i].astype(np.float32))     # (60,5)
        e = int(f["chosen_edge"][i]); d = int(f["chosen_depth"][i])
        f_labels = torch.zeros((60, NUM_DEPTHS), dtype=torch.float32)
        loss_mask = torch.zeros((60, NUM_DEPTHS), dtype=torch.float32)
        if self.mode == "pi":
            f_labels[e, d] = 1.0
            loss_mask = r_mask
            weight = float(f["pi_weight"][i])
        else:
            f_labels[e, d] = float(f["value_target"][i])
            loss_mask[e, d] = 1.0
            weight = float(f["v_weight"][i])
        cp = torch.zeros_like(r_mask)
        cp[(r_mask.sum(dim=1) > 0)] = 1.0
        out = {
            "context": ctx, "f_labels": f_labels, "r_mask": r_mask,
            "loss_mask": loss_mask, "cp_reachable": cp, "ratio": 0.0,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }
        if "contact_px" in f:
            out["contact_px"] = torch.from_numpy(f["contact_px"][i].astype(np.float32))
        return out


class RLDataModule(pl.LightningDataModule):
    def __init__(self, h5_path: str, mode: str, batch_size: int = 128, num_workers: int = 8,
                 train_split: float = 0.9, pin_memory: bool = True, **_):
        super().__init__()
        self.h5_path = h5_path
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        with h5py.File(self.h5_path, "r") as h5:
            n = int(h5.attrs.get("n_samples", h5["chosen_edge"].shape[0]))
            xml = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["xml"][:]]
            is_solved = h5["is_solved"][:].astype(int)
        rows = range(n) if self.mode == "v" else [i for i in range(n) if is_solved[i] == 1]
        groups = {}
        for i in rows:
            groups.setdefault(xml[i], []).append(i)
        keys = sorted(groups)
        random.Random(0).shuffle(keys)
        n_rows = sum(len(groups[k]) for k in keys)
        target = int(n_rows * self.train_split)
        train_idx, val_idx, cum = [], [], 0
        for k in keys:
            if cum < target:
                train_idx += groups[k]; cum += len(groups[k])
            else:
                val_idx += groups[k]
        if not val_idx:                      # tiny corpora (smoke): borrow a slice for a monitor
            val_idx = train_idx[-max(1, len(train_idx) // 10):]
        print(f"[rl {self.mode} setup] rows={n_rows} rooms={len(groups)} "
              f"train={len(train_idx)} val={len(val_idx)}", flush=True)
        self.train_dataset = RLScorerDataset(self.h5_path, train_idx, self.mode)
        self.val_dataset = RLScorerDataset(self.h5_path, val_idx, self.mode)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
