#!/usr/bin/env python3
"""train_q2 + a certain-order listwise softmax-CE ranking auxiliary.

For each exact-value tier on a board, the auxiliary ranks those actions above trained actions whose
exact value or ceiling is strictly lower. Equal ceilings and unknown cells are not compared. The
exact-1.0 opener term retains its original coefficient, while all lower exact tiers share a separate
bounded coefficient so adding a tier never dilutes opener supervision.

Total loss = exact HL-Gauss + censored ceiling + unreachable floor + opener rank + lower-exact rank.

Subclass ONLY — no edit to the shared weighted_module.py. Reuses ALL of train_q2's plumbing (dataloader,
callbacks, reload + eval_scorer-load checks, the hang marker) by monkeypatching build_module, so the
emitted ckpt is byte-for-byte eval_scorer/time_bestfirst-compatible. Validation stays PURE hl_gauss
(the checkpoint monitor is unchanged -> apples-to-apples ckpt selection vs model_1a_rs).

Knobs (env or the flags train_q2 already parses are reused; these are env-only):
  RANK_LAMBDA       (default 0.1)   exact-1.0 opener auxiliary weight
  LOWER_RANK_LAMBDA (default 0.05)  shared lower-exact auxiliary weight
  RANK_TEMP         (default 0.15)  softmax temperature over value in [0,1]
  NAMO_RANK_EXCLUDE_GUESS (default 0) bootstrapped cells may NOT be positives (still competitors)

NAMO_RANK_EXCLUDE_GUESS=1 -- why. The aux loops over torch.unique(labels[exact]) and treats each
distinct value as a CERTAIN tier whose exact order it enforces. That was written for discrete data:
theta0 has exactly two values, 1.0 and 0.9, so the loop ran twice per batch. Bootstrapped targets
min(cap, 0.9*V-hat) are continuous, so nearly every guessed cell is its own float -- measured 593
unique levels per 256-row batch on aquaman0_train_Bfix vs 27 on arm A and 2 on theta0. Two costs:
(1) it asserts a KNOWN ordering between values differing in the 4th decimal, which is model noise
    graded as ground truth (and the guess half-weight does not reach the aux -- it only weights the
    regression term), and
(2) ~300x the loop iterations, each a masked softmax+CE over the whole batch -- the measured 4x
    epoch-time gap between Bfix (~20 min) and arm A (~5 min) on the SAME a4000.
With the flag, only facts may be the group pushed to the top; guessed cells stay in `lower` (built
from the loss mask, not from `exact`), so every bit of their downward pressure is preserved.
Guess detection reuses the dataset's weight column (guesses are the only 0.5-weighted cells), which
is unambiguous ONLY at Q2_POS_WEIGHT=1.0 -- asserted at import.

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

RANK_LAMBDA = float(os.environ.get("RANK_LAMBDA", "0.1"))   # 0.1 = the bracket winner -> loop default
LOWER_RANK_LAMBDA = float(os.environ.get("LOWER_RANK_LAMBDA", "0.05"))
RANK_TEMP = float(os.environ.get("RANK_TEMP", "0.15"))
EXCLUDE_GUESS = os.environ.get("NAMO_RANK_EXCLUDE_GUESS", "0") == "1"
if EXCLUDE_GUESS:
    from namo.rl_loop.sage_ext.q2_dataset import Q2_POS_WEIGHT
    assert Q2_POS_WEIGHT == 1.0, (
        "NAMO_RANK_EXCLUDE_GUESS recovers the guess mask from the weight column (guesses are the "
        f"only 0.5-weighted cells). At Q2_POS_WEIGHT={Q2_POS_WEIGHT} that is ambiguous: a guessed "
        "positive-target cell and a plain negative-target cell would both weigh 1.0.")


def certain_order_rank_aux_losses(value, labels, mask, ceiling, temp, guess=None):
    """Rank each exact tier above actions with a strictly lower known upper bound.

    value/labels/mask/ceiling: (B,60,5). ``mask`` is tried-and-reachable. ``ceiling`` marks
    one-sided targets, so only mask & ~ceiling cells can be positives. Each board is averaged once
    even if it contains multiple exact tiers. Returns total, opener-only, and sub-1 setup losses.

    guess (B,60,5, optional): bootstrapped cells. Barred from being POSITIVES -- a positive is an
    assertion that this cell is KNOWN to outrank the list, and a model-authored float is not known.
    They stay in ``lower`` (built from ``mf``), so their downward pressure is untouched.
    """
    B = value.shape[0]
    vf = value.reshape(B, -1)
    mf = mask.reshape(B, -1)
    labf = labels.reshape(B, -1)
    cf = torch.zeros_like(mf) if ceiling is None else ceiling.reshape(B, -1)
    exact = (mf > 0) & (cf <= 0)
    if guess is not None:
        exact = exact & (guess.reshape(B, -1) <= 0)
    zero = value.sum() * 0.0
    if not exact.any():
        return zero, zero, zero

    row_sum = torch.zeros(B, device=value.device, dtype=value.dtype)
    row_terms = torch.zeros(B, device=value.device, dtype=value.dtype)
    opener_sum = torch.zeros_like(row_sum)
    opener_terms = torch.zeros_like(row_terms)
    setup_sum = torch.zeros_like(row_sum)
    setup_terms = torch.zeros_like(row_terms)

    for level in torch.unique(labf[exact]).detach():
        pos = exact & ((labf - level).abs() <= 1e-5)
        lower = (mf > 0) & (labf < level - 1e-5)
        valid = pos.any(dim=1) & lower.any(dim=1)
        if not valid.any():
            continue

        valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
        competitors = (pos | lower)[valid]
        scores = (vf[valid] / temp).masked_fill(~competitors, float("-inf"))
        logp = torch.log_softmax(scores, dim=1)
        valid_pos = pos[valid]
        target = valid_pos.float() / valid_pos.sum(dim=1, keepdim=True)
        ce = -(target * logp.clamp(min=-30.0)).sum(dim=1)
        ones = torch.ones_like(ce)
        row_sum = row_sum.index_add(0, valid_idx, ce)
        row_terms = row_terms.index_add(0, valid_idx, ones)
        if float(level) >= 0.999:
            opener_sum = opener_sum.index_add(0, valid_idx, ce)
            opener_terms = opener_terms.index_add(0, valid_idx, ones)
        else:
            setup_sum = setup_sum.index_add(0, valid_idx, ce)
            setup_terms = setup_terms.index_add(0, valid_idx, ones)

    valid_rows = row_terms > 0
    if not valid_rows.any():
        return zero, zero, zero

    total = (row_sum[valid_rows] / row_terms[valid_rows]).mean()
    opener_rows = opener_terms > 0
    setup_rows = setup_terms > 0
    opener = ((opener_sum[opener_rows] / opener_terms[opener_rows]).mean()
              if opener_rows.any() else zero)
    setup = ((setup_sum[setup_rows] / setup_terms[setup_rows]).mean()
             if setup_rows.any() else zero)
    return total, opener, setup


def rank_aux_loss(value, labels, mask, temp, ceiling=None):
    """Backward-compatible scalar wrapper used by focused tests and older callers."""
    return certain_order_rank_aux_losses(value, labels, mask, ceiling, temp)[0]


def weighted_rank_aux(opener_aux, lower_aux, opener_weight=RANK_LAMBDA,
                      lower_weight=LOWER_RANK_LAMBDA):
    """Keep the opener anchor full-strength and add a fixed lower-tier pool."""
    return opener_weight * opener_aux + lower_weight * lower_aux


class RankAuxModule(WeightedClassifierModule):
    rank_lambda = RANK_LAMBDA
    lower_rank_lambda = LOWER_RANK_LAMBDA
    rank_temp = RANK_TEMP

    def _split_loss(self, logits, f_labels, loss_mask, ceiling, weight):
        self._rank_ceiling_mask = ceiling
        try:
            return super()._split_loss(logits, f_labels, loss_mask, ceiling, weight)
        finally:
            self._rank_ceiling_mask = None

    def _weighted_loss(self, logits, labels, mask, weight):
        base = super()._weighted_loss(logits, labels, mask, weight)   # also guarantees self._hl_gauss
        val = self._hl_gauss.value(logits.float())                   # (B,60,5) differentiable E[bin]
        rank_mask = getattr(self, "_rank_list_mask", None)
        rank_ceiling = getattr(self, "_rank_ceiling_mask", None)
        # guesses are the only 0.5-weighted cells (asserted Q2_POS_WEIGHT==1.0 at import)
        guess = (weight < 0.75).float() if (EXCLUDE_GUESS and weight is not None) else None
        _, opener_aux, setup_aux = certain_order_rank_aux_losses(
            val, labels, rank_mask if rank_mask is not None else mask, rank_ceiling, self.rank_temp,
            guess)
        aux = weighted_rank_aux(
            opener_aux, setup_aux, self.rank_lambda, self.lower_rank_lambda)
        self.log("rank_aux", aux, on_step=False, on_epoch=True, prog_bar=False)
        self.log("rank_aux_opener", opener_aux, on_step=False, on_epoch=True, prog_bar=False)
        self.log("rank_aux_setup", setup_aux, on_step=False, on_epoch=True, prog_bar=False)
        return base + aux


def build_module(base_lr, warmup_steps, decay_steps):
    net = _make_network(value_bins=tq2.VALUE_BINS)
    if os.environ.get("NAMO_COMPILE", "0") == "1":
        net.compile()          # in-place: ckpt keys unchanged (never torch.compile(net) — renames keys)
        print("[compile] torch in-place compile ENABLED")
    return RankAuxModule(
        network=net, base_lr=base_lr, weight_decay=0.01,
        warmup_steps=warmup_steps, decay_steps=decay_steps, end_lr=1e-6,
        head_mode="hl_gauss", value_vmin=0.0, value_vmax=1.0, dice_weight=0.0,
    )


if __name__ == "__main__":
    # EXCLUDE_GUESS is printed because it silently changes what the aux ranks, and on 2026-08-08 it
    # could only be confirmed after the fact by noticing v4 and v3 had identical wall times (the
    # 386-tier pathology costs ~4x/epoch, so its absence was the only evidence the flag took).
    print(f"[rankaux] RANK_LAMBDA={RANK_LAMBDA}  LOWER_RANK_LAMBDA={LOWER_RANK_LAMBDA}  "
          f"RANK_TEMP={RANK_TEMP}  EXCLUDE_GUESS={int(EXCLUDE_GUESS)}", flush=True)
    tq2.build_module = build_module   # main() + reload-checks resolve build_module at call time
    tq2.main()
