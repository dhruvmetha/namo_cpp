#!/usr/bin/env python3
"""train_q2 + a listwise softmax-CE RANKING aux (one-off sharpening test on model_1a_rs).

Total loss  L = hl_gauss(value)  +  lambda * softmax_CE( value/T over tried-reachable cells, opener=+ ).

Rationale: hl_gauss alone is a per-cell REGRESSION (calibration), order-blind. Hard openers sit at
median rank ~6 among reachable cells, so best-first burns sims before reaching them. The aux makes the
tried-reachable cells COMPETE (a listwise margin): it pushes the opener's predicted value above the
reachable-dead cells, which is exactly what changes best-first order (solve@1 / sims-to-solve).

Subclass ONLY — no edit to the shared weighted_module.py. Reuses ALL of train_q2's plumbing (dataloader,
callbacks, reload + eval_scorer-load checks, the hang marker) by monkeypatching build_module, so the
emitted ckpt is byte-for-byte eval_scorer/time_bestfirst-compatible. Validation stays PURE hl_gauss
(the checkpoint monitor is unchanged -> apples-to-apples ckpt selection vs model_1a_rs).

Knobs (env or the flags train_q2 already parses are reused; these two are env-only):
  RANK_LAMBDA (default 0.5)   aux weight
  RANK_TEMP   (default 0.15)  softmax temperature over value in [0,1]

Usage:
  RANK_LAMBDA=0.5 RANK_TEMP=0.15 CUDA_VISIBLE_DEVICES=3 TMPDIR=/tmp \
    python scripts/rl_loop/train_q2_rankaux.py --h5 <model_1a_rs_train.h5> --out-dir <run> \
      --epochs 40 --batch-size 256 --num-workers 8 --lr 3e-4
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop._bootstrap import ensure_paths  # noqa: E402
ensure_paths()

import importlib.util  # noqa: E402
import torch  # noqa: E402
from namo.rl_loop.sage_ext.weighted_module import WeightedClassifierModule  # noqa: E402
from namo.rl_loop.train_gen import _make_network  # noqa: E402

# load train_q2 by file path (robust to the scripts/ package-name ambiguity across envs)
_spec = importlib.util.spec_from_file_location("train_q2", str(REPO / "scripts/rl_loop/train_q2.py"))
tq2 = importlib.util.module_from_spec(_spec)
sys.modules["train_q2"] = tq2
_spec.loader.exec_module(tq2)

RANK_LAMBDA = float(os.environ.get("RANK_LAMBDA", "0.5"))
RANK_TEMP = float(os.environ.get("RANK_TEMP", "0.15"))


def rank_aux_loss(value, labels, mask, temp):
    """Listwise softmax-CE over the TRIED-REACHABLE cells (mask>0): push openers (label>=0.999) above
    the reachable-dead cells. value/labels/mask: (B,60,5); value in [0,1]. Rows with no tried opener
    are skipped. Returns a scalar (0 = openers already dominate the softmax; large = openers buried)."""
    B = value.shape[0]
    vf = value.reshape(B, -1)
    mf = mask.reshape(B, -1)
    labf = labels.reshape(B, -1)
    scores = (vf / temp).masked_fill(mf <= 0, float("-inf"))
    logp = torch.log_softmax(scores, dim=1)
    pos = ((labf >= 0.999) & (mf > 0)).float()
    ps = pos.sum(dim=1)
    valid = ps > 0
    if not valid.any():
        return value.sum() * 0.0
    p = pos[valid] / ps[valid].unsqueeze(1)
    ce = -(p * logp[valid].clamp(min=-30.0)).sum(dim=1)
    return ce.mean()


class RankAuxModule(WeightedClassifierModule):
    rank_lambda = RANK_LAMBDA
    rank_temp = RANK_TEMP

    def _weighted_loss(self, logits, labels, mask, weight):
        base = super()._weighted_loss(logits, labels, mask, weight)   # also guarantees self._hl_gauss
        val = self._hl_gauss.value(logits.float())                   # (B,60,5) differentiable E[bin]
        aux = rank_aux_loss(val, labels, mask, self.rank_temp)
        self.log("rank_aux", aux, on_step=False, on_epoch=True, prog_bar=False)
        return base + self.rank_lambda * aux


def build_module(base_lr, warmup_steps, decay_steps):
    net = _make_network(value_bins=tq2.VALUE_BINS)
    return RankAuxModule(
        network=net, base_lr=base_lr, weight_decay=0.01,
        warmup_steps=warmup_steps, decay_steps=decay_steps, end_lr=1e-6,
        head_mode="hl_gauss", value_vmin=0.0, value_vmax=1.0, dice_weight=0.0,
    )


if __name__ == "__main__":
    print(f"[rankaux] RANK_LAMBDA={RANK_LAMBDA}  RANK_TEMP={RANK_TEMP}", flush=True)
    tq2.build_module = build_module   # main() + reload-checks resolve build_module at call time
    tq2.main()
