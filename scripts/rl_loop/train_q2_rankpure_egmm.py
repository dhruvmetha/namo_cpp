#!/usr/bin/env python3
"""Rank-pure EGMM: the fully value-free stack carrying the one V5-moving term. [EXP-2026-08-09 r3]

Loss = per-board softmax + batch-flat softmax (RP's two terms, weight 1 each)
     + RPE_FAM * family margin-vs-max (EGMM's deploy duel)
No regression, no censored fence, categorical labels only. Checkpoint monitor = RP's val rank loss
(family term excluded from val — the val loader is not family-packed, so keeping the monitor to
RP's terms makes it comparable to the RP row).

Watch items, pre-registered on the card: score stretch (RP's, no brake) compounding with the V6
board-scramble (EGMM's). Histogram + V6 are the meters.

Usage: TRAIN_SCRIPT=scripts/rl_loop/train_q2_rankpure_egmm.py (grouping is forced on here).
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ["NAMO_GROUP_EPISODES"] = "1"          # family batching is definitional for this arm

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop._bootstrap import ensure_paths  # noqa: E402
ensure_paths()


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(REPO / rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rp = _load("train_q2_rankpure", "scripts/rl_loop/train_q2_rankpure.py")
r2 = _load("train_q2_round2", "scripts/rl_loop/train_q2_round2.py")
tq2 = rp.tq2

RPE_FAM = float(os.environ.get("RPE_FAM", "1.0"))


class RankPureEGMM(rp.RankPureModule):
    def _rank_only(self, logits, labels, mask, ceiling):
        loss = super()._rank_only(logits, labels, mask, ceiling)
        ep_id = getattr(self, "_batch_ep_id", None)
        if ep_id is not None:
            fam = loss * 0.0
            n_fam = 0
            val = self._hl_gauss.value(logits.float())
            for v_, l_, m_, c_ in r2._family_lists((val, labels, mask, ceiling), ep_id):
                n_fam += 1
                _, o, s = r2.margin_vs_max_losses(v_, l_, m_, c_, r2.MM_MARGIN)
                fam = fam + o + s
            if n_fam:
                loss = loss + RPE_FAM * fam / n_fam
            self.log("n_families", float(n_fam), on_step=False, on_epoch=True)
            self.log("fam_margin", fam / max(n_fam, 1), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        self._batch_ep_id = batch.get("ep_id")
        try:
            return super().training_step(batch, batch_idx)
        finally:
            self._batch_ep_id = None


def build_module(base_lr, warmup_steps, decay_steps):
    net = tq2._make_network(value_bins=tq2.VALUE_BINS)
    return RankPureEGMM(
        network=net, base_lr=base_lr, weight_decay=0.01,
        warmup_steps=warmup_steps, decay_steps=decay_steps, end_lr=1e-6,
        head_mode="hl_gauss", value_vmin=0.0, value_vmax=1.0, dice_weight=0.0,
    )


if __name__ == "__main__":
    print(f"[rankpure-egmm] RPE_FAM={RPE_FAM}  MM_MARGIN={r2.MM_MARGIN}  RANK_TEMP={rp.rank.RANK_TEMP}  "
          f"(no regression; family margin ON; monitor = RP rank loss)", flush=True)
    tq2.build_module = build_module
    tq2.Q2DataModule = r2.GroupedQ2DataModule
    tq2.main()
