#!/usr/bin/env python3
"""HY5U rank-only arm: drop the exact-cell and censored regression, keep every ordering term.

The mirror of the `regression only` arm in EXP-2026-08-31. That arm zeroed `RANK_LAMBDA`,
`LOWER_RANK_LAMBDA` and `EGMM_LAMBDA` and kept regression; this one keeps all three ordering terms
and removes regression instead. Everything else stays HY5U: same H5, grouped family batches,
edge self-attention, 51-bin HL-Gauss head, `NAMO_UNREACH_W=0.1`.

What survives, and why the unreachable floor is one of them. The floor is NOT a separate term in
the loss: `GroupedQ2Dataset` zeroes `f_labels` on unreachable cells and folds them into `loss_mask`
at weight `UNREACH_W`, so HY5U regresses them inside the same HL-Gauss call as the exact cells.
"Remove the exact-cell regression, keep the floor" therefore means splitting that one call, which
is what `_split_loss` below does.

The denominator is the part that is easy to get wrong. HL-Gauss reduces by group mean, so the
floor's share of HY5U's regression is `sum(ce * floor) / sum(exact_mask)`, where the denominator
counts the exact cells too (~68 tried cells at weight 1.0 against ~230 unreachable at 0.1, so
~91 vs ~23). Masking the exact cells out of both numerator and denominator would quadruple the
surviving term and confound the ablation with a weight change. `_floor_scale` restores HY5U's
denominator so the one term that stays keeps exactly the magnitude it had.

Checkpoint monitor: `val_loss` is REDEFINED as floor + per-board rank on the val split. The stock
monitor measures the regression this arm does not train, so selecting on it would select on noise.
Consequence, same as train_q2_rankpure.py: `val_loss` is NOT comparable to any other registry row,
and train_q2's post-training "[reload check] reloaded val_loss vs monitored" line recomputes the
REGRESSION formula, so a large delta there is expected. Ignore that one line.

Card: docs/experiments/log/EXP-2026-09-12-hy5u-rank-only.md.

Usage (matches the HY5U grid):
  EGMM_LAMBDA=0.1 RANK_LAMBDA=0.1 LOWER_RANK_LAMBDA=0.05 NAMO_UNREACH_W=0.1 \
  NAMO_GROUP_EPISODES=1 NAMO_GAMMA=0.5 NAMO_EDGE_SELF_ATTN=1 \
  python scripts/rl_loop/train_q2_rankonly.py --h5 <hybrid_train_v1.h5> --out-dir <run> \
      --epochs 12 --batch-size 256 --lr 3e-4
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

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_q2_round2", str(REPO / "scripts/rl_loop/train_q2_round2.py"))
r2 = importlib.util.module_from_spec(_spec)
sys.modules["train_q2_round2"] = r2
_spec.loader.exec_module(r2)
rank = r2.rank
tq2 = r2.tq2


def _wmask(mask, weight):
    """Reproduce WeightedClassifierModule's per-cell weighting of a loss mask."""
    if weight is None:
        return mask
    if weight.dim() == mask.dim():
        return mask * weight
    return mask * weight.view(-1, *([1] * (mask.dim() - 1)))


class RankOnlyModule(r2.Round2Module):
    def training_step(self, batch, batch_idx):
        self._batch_r_mask = batch["r_mask"]
        try:
            return super().training_step(batch, batch_idx)
        finally:
            self._batch_r_mask = None

    def _split_loss(self, logits, f_labels, loss_mask, ceiling, weight):
        """Ordering terms at full strength; regression only on the folded unreachable cells.

        Mask wiring is byte-identical to the HY5U path: the rank list is `loss_mask` (exact +
        ceiling + folded unreachable, the -b fix) and the ceiling mask marks one-sided cells, so
        the opener/setup/family lists see exactly what they see under the full loss.
        """
        hl = self._hl(logits)                                  # censored-capable helper, as in HY5U
        r = self._batch_r_mask
        exact_mask = loss_mask * (1.0 - ceiling)               # HY5U's regression mask
        floor_mask = exact_mask * (1.0 - r)                    # the folded unreachable cells only

        self._rank_ceiling_mask = ceiling
        self._rank_list_mask = loss_mask
        try:
            # zero regression mask -> Round2Module contributes ONLY the rank aux and family terms
            loss = self._weighted_loss(logits, f_labels, torch.zeros_like(loss_mask), weight)
        finally:
            self._rank_list_mask = None
            self._rank_ceiling_mask = None

        floor = hl.loss(logits, f_labels, _wmask(floor_mask, weight)) * self._floor_scale(
            floor_mask, exact_mask, weight)
        self.log("unreach_floor", floor, on_step=False, on_epoch=True, prog_bar=False)
        return loss + floor

    @staticmethod
    def _floor_scale(floor_mask, exact_mask, weight):
        """Restore HY5U's group-mean denominator so the surviving term keeps its original weight."""
        wf = _wmask(floor_mask, weight).sum()
        we = _wmask(exact_mask, weight).sum()
        return (wf / we.clamp_min(1.0)).detach()

    def validation_step(self, batch, batch_idx):
        context = batch["context"]; f_labels = batch["f_labels"]; r_mask = batch["r_mask"]
        loss_mask = batch.get("loss_mask", r_mask)
        logits = self(context, batch.get("contact_px"), batch.get("context_zoom"),
                      batch.get("contact_px_zoom"), H=batch.get("H"),
                      reach_edges=batch.get("reach_edges"),
                      action_motion=batch.get("action_motion"))
        ceiling = batch.get("ceiling_mask")
        if ceiling is None:
            ceiling = torch.zeros_like(loss_mask)
        hl = self._hl(logits)
        val = hl.value(logits.float())
        _, opener, setup = rank.certain_order_rank_aux_losses(
            val, f_labels, loss_mask, ceiling, self.rank_temp)
        loss = rank.weighted_rank_aux(opener, setup, self.rank_lambda, self.lower_rank_lambda)
        exact_mask = loss_mask * (1.0 - ceiling)
        floor_mask = exact_mask * (1.0 - r_mask)
        loss = loss + hl.loss(logits, f_labels, floor_mask) * self._floor_scale(
            floor_mask, exact_mask, None)
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        return loss


def build_module(base_lr, warmup_steps, decay_steps):
    net = tq2._make_network(value_bins=tq2.VALUE_BINS)
    if os.environ.get("NAMO_COMPILE", "0") == "1":
        net.compile()
        print("[compile] torch in-place compile ENABLED")
    return RankOnlyModule(
        network=net, base_lr=base_lr, weight_decay=0.01,
        warmup_steps=warmup_steps, decay_steps=decay_steps, end_lr=1e-6,
        head_mode="hl_gauss", value_vmin=0.0, value_vmax=1.0, dice_weight=0.0,
    )


if __name__ == "__main__":
    print(f"[rankonly] RANK_LAMBDA={rank.RANK_LAMBDA} LOWER_RANK_LAMBDA={rank.LOWER_RANK_LAMBDA} "
          f"EGMM_LAMBDA={r2.EGMM_LAMBDA} MM_MARGIN={r2.MM_MARGIN} UNREACH_W={r2.UNREACH_W} "
          f"(val_loss monitor = floor + per-board rank; exact/censored regression ABSENT)", flush=True)
    tq2.build_module = build_module
    tq2.Q2DataModule = r2.GroupedQ2DataModule       # carries the UNREACH_W folding, as in round 2
    tq2.main()
