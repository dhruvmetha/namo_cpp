#!/usr/bin/env python3
"""RPL: rank-pure + family margin on a TRULY UNBOUNDED linear head. [EXP-2026-08-09 round 3, USER]

The head is one unactivated neuron per cell — (B,60,5) raw scores, no bins, no softmax squash, no
span. HL-Gauss is deleted, not widened: under rank-pure it only ever supplied E[bin], a scalar we
now emit directly (the Chrestien configuration; our deltas are hinge-on-MAX for the family duel and
an explicit leash). Margin = 1.0, the scale-free convention (with a free scale, margin and leash
share one degree of freedom; 1.0 is the unit the scores organize around — Kim/RankSVM lineage).

  loss = per-board softmax + batch-flat softmax          (RP's terms, on raw scores)
       + RPE_FAM * family margin-vs-max (margin 1.0)     (the deploy duel)
       + RPL_LEASH * mean(score^2)                       (anti-inflation; the ONLY anchor)

Plumbing notes, each deliberate:
  * tq2.HLGauss is shimmed scalar-tolerant so train_q2's post-training checks (eval_scorer-load,
    diagnostics) pass a (1,60,5) head instead of crashing before the completion marker.
  * head_mode="bce" — the 5-out head's native module mode; all loss paths it gates are overridden.
  * Deploy: eval loaders auto-detect head_out==num_depths (the legacy branch) and canonical runs
    --raw by default, which consumes unbounded scores natively. One canonical shard is sanity-run
    before any fleet (see card).
Monitor = RP's val rank loss. Score histogram + fam_margin are the pre-registered meters.
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ["NAMO_GROUP_EPISODES"] = "1"
os.environ.setdefault("RPE_MARGIN", "1.0")

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.rl_loop._bootstrap import ensure_paths  # noqa: E402
ensure_paths()

_spec = importlib.util.spec_from_file_location(
    "train_q2_rankpure_egmm", str(REPO / "scripts/rl_loop/train_q2_rankpure_egmm.py"))
rpe = importlib.util.module_from_spec(_spec)
sys.modules["train_q2_rankpure_egmm"] = rpe
_spec.loader.exec_module(rpe)
rp, r2, tq2 = rpe.rp, rpe.r2, rpe.tq2
rank = rp.rank

RPL_LEASH = float(os.environ.get("RPL_LEASH", "1e-3"))
MARGIN = float(os.environ.get("RPE_MARGIN", "1.0"))
RPE_FAM = rpe.RPE_FAM


class RankPureLinear(rpe.RankPureEGMM):
    def _rank_only(self, logits, labels, mask, ceiling):
        val = logits.float()                              # (B,60,5) raw scores — THE head output
        _, wb_o, wb_s = rank.certain_order_rank_aux_losses(val, labels, mask, ceiling, self.rank_temp)

        def _flat(t):
            return None if t is None else t.reshape(1, -1)
        _, xb_o, xb_s = rank.certain_order_rank_aux_losses(
            _flat(val), _flat(labels), _flat(mask), _flat(ceiling), self.rank_temp)
        loss = (wb_o + wb_s) + (xb_o + xb_s)

        ep_id = getattr(self, "_batch_ep_id", None)
        if ep_id is not None:
            fam = loss * 0.0
            n_fam = 0
            for v_, l_, m_, c_ in r2._family_lists((val, labels, mask, ceiling), ep_id):
                n_fam += 1
                _, o, s = r2.margin_vs_max_losses(v_, l_, m_, c_, MARGIN)
                fam = fam + o + s
            if n_fam:
                loss = loss + RPE_FAM * fam / n_fam
            self.log("n_families", float(n_fam), on_step=False, on_epoch=True)
            self.log("fam_margin", fam / max(n_fam, 1), on_step=False, on_epoch=True)

        leash = RPL_LEASH * val.pow(2).mean()
        self.log("leash", leash, on_step=False, on_epoch=True)
        self.log("score_spread", (val.max() - val.min()).detach(), on_step=False, on_epoch=True)
        return loss + leash


class _ScalarOKHLGauss(tq2.HLGauss):
    """tq2's post-training checks call HLGauss.value on head output; a linear head has no bin axis."""
    def value(self, logits):
        if logits.shape[-1] != self.num_bins:
            return logits
        return super().value(logits)


def build_module(base_lr, warmup_steps, decay_steps):
    net = tq2._make_network(value_bins=0)                 # 5-out linear head — the original shape
    return RankPureLinear(
        network=net, base_lr=base_lr, weight_decay=0.01,
        warmup_steps=warmup_steps, decay_steps=decay_steps, end_lr=1e-6,
        head_mode="bce", dice_weight=0.0,
    )


if __name__ == "__main__":
    print(f"[rankpure-linear] MARGIN={MARGIN}  RPL_LEASH={RPL_LEASH}  RPE_FAM={RPE_FAM}  "
          f"RANK_TEMP={rank.RANK_TEMP}  head=LINEAR (no bins, unbounded)", flush=True)
    tq2.HLGauss = _ScalarOKHLGauss
    tq2.build_module = build_module
    tq2.Q2DataModule = r2.GroupedQ2DataModule
    tq2.main()
