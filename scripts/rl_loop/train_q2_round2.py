#!/usr/bin/env python3
"""Round-2 cross-board arms: margin-vs-MAX and episode-grouped lists. [EXP-2026-08-09 round 2]

Round-1 measured (see card): batch-flat softmax lifts AVERAGE dead-cell suppression to a record
(RP V4 0.915) while the dead-board MAX stays afloat (V5 flat) — because (1) the listwise CE stalls
at its floor ~= log(#positives) so the tallest rival keeps only a sliver of gradient, and (2) the
pair V5 actually grades (setup vs its OWN episode's dead boards) almost never co-occurs in a random
batch (only 21% of rows even have stored siblings). One arm per mechanism:

  MM_LAMBDA   > 0: batch-flat margin-vs-max — per tier, every positive must beat the TALLEST
                   strictly-lower cell in the whole batch by MM_MARGIN. Cannot stall: any rival
                   above the gap gets full gradient. Targets mechanism (1).
  EG_LAMBDA   > 0: episode-grouped softmax — the round-1 certain-order CE, but the list is the
                   EPISODE FAMILY (root + every stored child board), co-located in the batch by a
                   grouped sampler (NAMO_GROUP_EPISODES=1). Targets mechanism (2).
  EGMM_LAMBDA > 0: margin-vs-max WITHIN the family — "your setup must beat the tallest cell of
                   every dead board in the episode (children of failed pushes, dead-end children)
                   and the junk of live children" [USER phrasing] — the deploy objective verbatim.

All terms ADD to the AJ2 base loss (regression + censored + per-board aux) via RankAuxModule.
Margin default 0.2: bounded [0,1] head, measured live-dead board-max gap ~=0.2 (autopsy in card).
Family terms fire only on rows whose episode has >=2 boards in the H5 (21%) — logged per-term so a
too-small dose is visible, not silent.

Usage (one arm = one env knob set):
  MM:   MM_LAMBDA=0.1 [MM_MARGIN=0.2]
  EG:   EG_LAMBDA=0.1 NAMO_GROUP_EPISODES=1
  EGMM: EGMM_LAMBDA=0.1 [MM_MARGIN=0.2] NAMO_GROUP_EPISODES=1
"""
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop._bootstrap import ensure_paths  # noqa: E402
ensure_paths()

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_q2_rankaux", str(REPO / "scripts/rl_loop/train_q2_rankaux.py"))
rank = importlib.util.module_from_spec(_spec)
sys.modules["train_q2_rankaux"] = rank
_spec.loader.exec_module(rank)
tq2 = rank.tq2
from namo.rl_loop.sage_ext.q2_dataset import Q2DataModule, Q2ValueDataset  # noqa: E402

MM_LAMBDA = float(os.environ.get("MM_LAMBDA", "0.0"))
MM_MARGIN = float(os.environ.get("MM_MARGIN", "0.2"))
EG_LAMBDA = float(os.environ.get("EG_LAMBDA", "0.0"))
EGMM_LAMBDA = float(os.environ.get("EGMM_LAMBDA", "0.0"))
GROUP_EPISODES = os.environ.get("NAMO_GROUP_EPISODES", "0") == "1"


def margin_vs_max_losses(value, labels, mask, ceiling, margin):
    """Per row-list and per certain tier: relu(max_rival - positive + margin), mean over positives.

    Same (B, ...) list contract and (total, opener, setup) return as the softmax loss. Unlike the
    CE it has no floor: it is exactly zero when every positive clears the tallest rival by the
    margin, and any rival above the gap gets FULL gradient regardless of how many positives exist.
    """
    B = value.shape[0]
    vf = value.reshape(B, -1)
    mf = mask.reshape(B, -1)
    labf = labels.reshape(B, -1)
    cf = torch.zeros_like(mf) if ceiling is None else ceiling.reshape(B, -1)
    exact = (mf > 0) & (cf <= 0)
    zero = value.sum() * 0.0
    if not exact.any():
        return zero, zero, zero

    row_sum = torch.zeros(B, device=value.device, dtype=value.dtype)
    row_terms = torch.zeros(B, device=value.device, dtype=value.dtype)
    opener_sum = torch.zeros_like(row_sum); opener_terms = torch.zeros_like(row_terms)
    setup_sum = torch.zeros_like(row_sum); setup_terms = torch.zeros_like(row_terms)
    for level in torch.unique(labf[exact]).detach():
        pos = exact & ((labf - level).abs() <= 1e-5)
        lower = (mf > 0) & (labf < level - 1e-5)
        valid = pos.any(dim=1) & lower.any(dim=1)
        if not valid.any():
            continue
        valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
        rival_max = vf[valid].masked_fill(~lower[valid], float("-inf")).max(dim=1).values  # (V,)
        pos_v = pos[valid]
        viol = torch.relu(rival_max.unsqueeze(1) - vf[valid] + margin) * pos_v.float()
        per_row = viol.sum(dim=1) / pos_v.float().sum(dim=1)
        ones = torch.ones_like(per_row)
        row_sum = row_sum.index_add(0, valid_idx, per_row)
        row_terms = row_terms.index_add(0, valid_idx, ones)
        if float(level) >= 0.999:
            opener_sum = opener_sum.index_add(0, valid_idx, per_row)
            opener_terms = opener_terms.index_add(0, valid_idx, ones)
        else:
            setup_sum = setup_sum.index_add(0, valid_idx, per_row)
            setup_terms = setup_terms.index_add(0, valid_idx, ones)

    valid_rows = row_terms > 0
    if not valid_rows.any():
        return zero, zero, zero
    total = (row_sum[valid_rows] / row_terms[valid_rows]).mean()
    op_rows = opener_terms > 0; se_rows = setup_terms > 0
    opener = (opener_sum[op_rows] / opener_terms[op_rows]).mean() if op_rows.any() else zero
    setup = (setup_sum[se_rows] / setup_terms[se_rows]).mean() if se_rows.any() else zero
    return total, opener, setup


def _family_lists(t, ep_id):
    """Yield (1,-1) views of each >=2-row episode family in the batch."""
    for e in torch.unique(ep_id):
        rows = (ep_id == e).nonzero(as_tuple=False).squeeze(1)
        if rows.numel() >= 2:
            yield tuple(x[rows].reshape(1, -1) if x is not None else None for x in t)


class Round2Module(rank.RankAuxModule):
    def _weighted_loss(self, logits, labels, mask, weight):
        base = super()._weighted_loss(logits, labels, mask, weight)   # AJ2 base (+XB if set)
        val = self._hl_gauss.value(logits.float())
        lm = getattr(self, "_rank_list_mask", None)
        lm = lm if lm is not None else mask
        ceil = getattr(self, "_rank_ceiling_mask", None)
        extra = base * 0.0

        if MM_LAMBDA > 0.0:
            def _flat(t):
                return None if t is None else t.reshape(1, -1)
            _, mm_o, mm_s = margin_vs_max_losses(
                _flat(val), _flat(labels), _flat(lm), _flat(ceil), MM_MARGIN)
            extra = extra + MM_LAMBDA * (mm_o + mm_s)
            self.log("mm_opener", mm_o, on_step=False, on_epoch=True)
            self.log("mm_setup", mm_s, on_step=False, on_epoch=True)

        ep_id = getattr(self, "_batch_ep_id", None)
        if ep_id is not None and (EG_LAMBDA > 0.0 or EGMM_LAMBDA > 0.0):
            eg = egmm = extra * 0.0
            n_fam = 0
            for v_, l_, m_, c_ in _family_lists((val, labels, lm, ceil), ep_id):
                n_fam += 1
                if EG_LAMBDA > 0.0:
                    _, o, s = rank.certain_order_rank_aux_losses(v_, l_, m_, c_, self.rank_temp)
                    eg = eg + o + s
                if EGMM_LAMBDA > 0.0:
                    _, o, s = margin_vs_max_losses(v_, l_, m_, c_, MM_MARGIN)
                    egmm = egmm + o + s
            if n_fam:
                extra = extra + (EG_LAMBDA * eg + EGMM_LAMBDA * egmm) / n_fam
            self.log("n_families", float(n_fam), on_step=False, on_epoch=True)
            if EG_LAMBDA > 0.0:
                self.log("eg_term", eg / max(n_fam, 1), on_step=False, on_epoch=True)
            if EGMM_LAMBDA > 0.0:
                self.log("egmm_term", egmm / max(n_fam, 1), on_step=False, on_epoch=True)
        return base + extra

    def training_step(self, batch, batch_idx):
        self._batch_ep_id = batch.get("ep_id")
        try:
            return super().training_step(batch, batch_idx)
        finally:
            self._batch_ep_id = None


class GroupedQ2Dataset(Q2ValueDataset):
    """Q2ValueDataset + a stable per-row episode id (precomputed in the datamodule)."""
    def __init__(self, h5_path, indices, ep_ids):
        super().__init__(h5_path, indices)
        self.ep_ids = ep_ids                       # aligned with `indices`

    def __getitem__(self, k):
        out = super().__getitem__(k)
        out["ep_id"] = int(self.ep_ids[k])
        return out


class GroupedQ2DataModule(Q2DataModule):
    """Room-grouped split (unchanged) + family-packed TRAIN batches when NAMO_GROUP_EPISODES=1.

    Batch = whole (xml, object_id) families greedy-packed to batch_size, shuffled each epoch, so
    the family loss terms actually see their lists. Val loader stays plain (monitor unchanged).
    """
    def setup(self, stage=None):
        super().setup(stage)
        with h5py.File(self.h5_path, "r") as h5:
            xml = h5["xml"][:]; obj = h5["object_id"][:]
        keys = {}
        self._row_ep = np.empty(len(xml), dtype=np.int64)
        for i, (x, o) in enumerate(zip(xml, obj)):
            self._row_ep[i] = keys.setdefault((x, o), len(keys))
        self.train_dataset = GroupedQ2Dataset(
            self.h5_path, self.train_idx, self._row_ep[self.train_idx])
        self.val_dataset = GroupedQ2Dataset(
            self.h5_path, self.val_idx, self._row_ep[self.val_idx])
        fams = {}
        for pos, i in enumerate(self.train_idx):
            fams.setdefault(self._row_ep[i], []).append(pos)   # positions within train_dataset
        self._families = list(fams.values())
        n_multi = sum(1 for f in self._families if len(f) >= 2)
        print(f"[q2 grouped] families={len(self._families)} multi-board={n_multi}", flush=True)

    def train_dataloader(self):
        if not GROUP_EPISODES:
            return super().train_dataloader()

        families, bs = self._families, self.batch_size

        class _FamilyBatches:
            def __init__(self):
                self._epoch = 0

            def __iter__(self):
                import random as _r
                order = list(range(len(families)))
                _r.Random(1000 + self._epoch).shuffle(order)
                self._epoch += 1
                batch = []
                for fi in order:
                    fam = families[fi]
                    if len(batch) + len(fam) > bs and batch:
                        yield batch; batch = []
                    batch.extend(fam[:bs])                      # a >bs family is truncated, not split
                if batch:
                    yield batch

            def __len__(self):
                return max(1, sum(len(f) for f in families) // bs)

        return DataLoader(self.train_dataset, batch_sampler=_FamilyBatches(), **self._kw())


def build_module(base_lr, warmup_steps, decay_steps):
    net = tq2._make_network(value_bins=tq2.VALUE_BINS)
    return Round2Module(
        network=net, base_lr=base_lr, weight_decay=0.01,
        warmup_steps=warmup_steps, decay_steps=decay_steps, end_lr=1e-6,
        head_mode="hl_gauss", value_vmin=0.0, value_vmax=1.0, dice_weight=0.0,
    )


if __name__ == "__main__":
    print(f"[round2] MM_LAMBDA={MM_LAMBDA} MM_MARGIN={MM_MARGIN} EG_LAMBDA={EG_LAMBDA} "
          f"EGMM_LAMBDA={EGMM_LAMBDA} GROUP_EPISODES={int(GROUP_EPISODES)} "
          f"XB_LAMBDA={rank.XB_LAMBDA}", flush=True)
    tq2.build_module = build_module
    tq2.Q2DataModule = GroupedQ2DataModule
    tq2.main()
