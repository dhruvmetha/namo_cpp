#!/usr/bin/env python3
"""Rank-pure trainer: ranking is the ENTIRE loss — no regression, no censored fence. [EXP-2026-08-09]

Loss = RP_WB * (per-board opener CE + per-board setup CE)
     + RP_XB * (batch-flat opener CE + batch-flat setup CE)

Weights default 1.0/1.0: with no other term in the sum, absolute loss scale is meaningless (AdamW
normalizes per-parameter gradient magnitude away) — only the WB:XB ratio is a real knob. The
per-board term is technically a subset of the batch-flat list but is kept as concentrated gradient
on within-board order (the F2-critical comparisons would otherwise be ~300 voices among ~77k).

Deploy consumes only ORDER (canonical GBFS pops argmax; raw==sigmoid verified byte-identical under
combine=q), so no calibration is owed. Head stays the 51-bin HL-Gauss and the score stays E[bin] in
[0,1] -> ckpt is byte-compatible with eval_scorer/time_bestfirst; the bounded range doubles as the
anti-drift brake (watch item: score pile-up at the endpoints — see the card's H2).

Checkpoint monitor: val_loss is REDEFINED here as the same rank loss on the val split (unweighted).
The stock monitor measures regression, which this module does not train — selecting checkpoints by
it would be selecting on noise. Consequence: val_loss is NOT comparable to any other registry row,
and train_q2's post-training "[reload check] reloaded val_loss vs monitored" line recomputes the
REGRESSION formula, so its delta is expected to be large for RP — ignore that one line.

Card: docs/experiments/log/EXP-2026-08-09-crossboard-ranking.md.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/rl_loop/train_q2_rankpure.py --h5 <arjuna0v2_train.h5> \
      --out-dir <run> --epochs 12 --batch-size 256 --num-workers 8 --lr 3e-4
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

# load train_q2_rankaux by file path (it loads train_q2 the same way and owns the loss function)
_spec = importlib.util.spec_from_file_location(
    "train_q2_rankaux", str(REPO / "scripts/rl_loop/train_q2_rankaux.py"))
rank = importlib.util.module_from_spec(_spec)
sys.modules["train_q2_rankaux"] = rank
_spec.loader.exec_module(rank)
tq2 = rank.tq2

RP_WB = float(os.environ.get("RP_WB", "1.0"))
RP_XB = float(os.environ.get("RP_XB", "1.0"))
# Round-2 [card § Round 2]: small regression BRAKE. Round-1 autopsy: rank-only inflated the whole
# scale (spread 0.45->0.67, dead maxima 0.68) — regression's real job was keeping the scale tight,
# not teaching order (F2 held at 0.889 without it). RP_BRAKE adds back exact-cell HL-Gauss at a
# fraction of its old weight purely as the anti-stretch anchor.
RP_BRAKE = float(os.environ.get("RP_BRAKE", "0.0"))


class RankPureModule(rank.RankAuxModule):
    def _rank_only(self, logits, labels, mask, ceiling):
        hl = self._hl(logits)                       # ensures the censored-capable helper exists
        val = hl.value(logits.float())              # (B,60,5) differentiable E[bin]
        _, wb_opener, wb_setup = rank.certain_order_rank_aux_losses(
            val, labels, mask, ceiling, self.rank_temp)

        def _flat(t):
            return None if t is None else t.reshape(1, -1)
        _, xb_opener, xb_setup = rank.certain_order_rank_aux_losses(
            _flat(val), _flat(labels), _flat(mask), _flat(ceiling), self.rank_temp)
        loss = RP_WB * (wb_opener + wb_setup) + RP_XB * (xb_opener + xb_setup)
        for name, v in (("rank_wb_opener", wb_opener), ("rank_wb_setup", wb_setup),
                        ("rank_xb_opener", xb_opener), ("rank_xb_setup", xb_setup)):
            self.log(name, v, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    # training path A (ceiling_mask present in the batch): competition list = loss_mask (exact +
    # ceiling cells), exactly the -b wiring; regression and censored terms deliberately absent
    # (except the optional RP_BRAKE fraction of the exact-cell term — the anti-stretch anchor).
    def _split_loss(self, logits, f_labels, loss_mask, ceiling, weight):
        loss = self._rank_only(logits, f_labels, loss_mask, ceiling)
        if RP_BRAKE > 0.0:
            hl = self._hl(logits)
            exact_mask = loss_mask * (1.0 - ceiling)
            loss = loss + RP_BRAKE * hl.loss(logits, f_labels, exact_mask)
        return loss

    # training path B (no ceiling_mask in the batch)
    def _weighted_loss(self, logits, labels, mask, weight):
        return self._rank_only(logits, labels, mask, None)

    def validation_step(self, batch, batch_idx):
        context = batch["context"]; f_labels = batch["f_labels"]; r_mask = batch["r_mask"]
        loss_mask = batch.get("loss_mask", r_mask)
        logits = self(context, batch.get("contact_px"), batch.get("context_zoom"),
                      batch.get("contact_px_zoom"), H=batch.get("H"),
                      reach_edges=batch.get("reach_edges"),
                      action_motion=batch.get("action_motion"))
        loss = self._rank_only(logits, f_labels, loss_mask, batch.get("ceiling_mask"))
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        return loss


def build_module(base_lr, warmup_steps, decay_steps):
    net = tq2._make_network(value_bins=tq2.VALUE_BINS)
    if os.environ.get("NAMO_COMPILE", "0") == "1":
        net.compile()
        print("[compile] torch in-place compile ENABLED")
    return RankPureModule(
        network=net, base_lr=base_lr, weight_decay=0.01,
        warmup_steps=warmup_steps, decay_steps=decay_steps, end_lr=1e-6,
        head_mode="hl_gauss", value_vmin=0.0, value_vmax=1.0, dice_weight=0.0,
    )


if __name__ == "__main__":
    print(f"[rankpure] RP_WB={RP_WB}  RP_XB={RP_XB}  RANK_TEMP={rank.RANK_TEMP}  "
          f"(val_loss monitor = RANK loss; regression/censored terms ABSENT)", flush=True)
    tq2.build_module = build_module
    tq2.main()
