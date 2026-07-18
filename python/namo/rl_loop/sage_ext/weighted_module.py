"""WeightedClassifierModule — sage ClassifierModule + per-sample loss weights.

The only thing the RL loop needs beyond the stock scorer trainer is a per-sample weight
(BC weighting: uniform within a trajectory, 2^-(T-T_min) across an episode's trajectories;
V weighting: recency). We subclass — NO edit to the sage file — and fold the weight into the
existing masked reductions:

  hl_gauss  : mask -> mask * weight  (the reduction is (ce*mask).sum()/mask.sum(), so scaling
              the single supervised cell by the sample weight yields a weighted average).
  softmax_ce: reuse the module's per-sample CE, then weight it: (ce*w).sum()/w.sum().

Validation stays UNWEIGHTED and uses loss_mask (so the V monitor supervises only the chosen
cell, matching training) — it is just the checkpoint monitor.
"""
import os

import torch

from ._sage import ClassifierModule, HLGauss
from .hl_gauss_censored import CensoredHLGauss

# ceiling cells (censored observations, beast-0a): loss weight for the censored group-mean term.
CENS_WEIGHT = float(os.environ.get("NAMO_CENS_WEIGHT", "1.0"))


class WeightedClassifierModule(ClassifierModule):
    def _hl(self, logits):
        """Ensure the (censored-capable, endpoint-fixed) HL-Gauss helper exists and matches the head."""
        if not isinstance(self._hl_gauss, CensoredHLGauss) or self._hl_gauss.num_bins != logits.shape[-1]:
            self._hl_gauss = CensoredHLGauss(num_bins=logits.shape[-1],
                                             vmin=self.value_vmin, vmax=self.value_vmax)
        return self._hl_gauss

    def _split_loss(self, logits, f_labels, loss_mask, ceiling, weight):
        """Exact cells -> HL-Gauss CE (weighted); ceiling cells -> censored NLL. Group-mean each."""
        hl = self._hl(logits)
        exact_mask = loss_mask * (1.0 - ceiling)
        cens_mask = loss_mask * ceiling
        loss = self._weighted_loss(logits, f_labels, exact_mask, weight)
        if cens_mask.any():
            loss = loss + CENS_WEIGHT * hl.censored_loss(logits, f_labels, cens_mask)
        return loss
    def _weighted_loss(self, logits, labels, mask, weight):
        if weight is None:
            return self._compute_masked_loss(logits, labels, mask)
        if self.head_mode == "hl_gauss":
            if self._hl_gauss is None or self._hl_gauss.num_bins != logits.shape[-1]:
                self._hl_gauss = HLGauss(num_bins=logits.shape[-1],
                                         vmin=self.value_vmin, vmax=self.value_vmax)
            wmask = mask * (weight if weight.dim() == mask.dim()          # per-cell (60x5) weight
                            else weight.view(-1, *([1] * (mask.dim() - 1))))  # per-sample scalar
            return self._hl_gauss.loss(logits, labels, wmask)
        if self.head_mode == "softmax_ce":
            B = logits.shape[0]
            lf = logits.reshape(B, -1); labf = labels.reshape(B, -1); mf = mask.reshape(B, -1)
            masked = lf.masked_fill(mf <= 0, float("-inf"))
            logp = torch.log_softmax(masked, dim=1)
            tgt = labf * mf
            ts = tgt.sum(dim=1, keepdim=True)
            valid = (ts.squeeze(1) > 0)
            if not valid.any():
                return lf.sum() * 0.0
            p = tgt[valid] / ts[valid]
            ce = -(p * logp[valid].clamp(min=-30.0)).sum(dim=1)
            w = weight[valid]
            return (ce * w).sum() / w.sum().clamp_min(1e-6)
        return self._compute_masked_loss(logits, labels, mask)

    def training_step(self, batch, batch_idx):
        context = batch["context"]; f_labels = batch["f_labels"]; r_mask = batch["r_mask"]
        loss_mask = batch.get("loss_mask", r_mask)
        logits = self(context, batch.get("contact_px"), batch.get("context_zoom"),
                      batch.get("contact_px_zoom"), H=batch.get("H"), reach_edges=batch.get("reach_edges"))
        ceiling = batch.get("ceiling_mask")
        if ceiling is not None:
            loss = self._split_loss(logits, f_labels, loss_mask, ceiling, batch.get("weight"))
        else:
            loss = self._weighted_loss(logits, f_labels, loss_mask, batch.get("weight"))
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        context = batch["context"]; f_labels = batch["f_labels"]; r_mask = batch["r_mask"]
        loss_mask = batch.get("loss_mask", r_mask)
        logits = self(context, batch.get("contact_px"), batch.get("context_zoom"),
                      batch.get("contact_px_zoom"), H=batch.get("H"), reach_edges=batch.get("reach_edges"))
        ceiling = batch.get("ceiling_mask")
        if ceiling is not None:
            # val stays PURE (no rank-aux, unweighted): exact-cell regression + censored fence only.
            # Bypass _weighted_loss so RankAuxModule's override can't leak the aux into the monitor.
            hl = self._hl(logits)
            exact_mask = loss_mask * (1.0 - ceiling)
            loss = self._compute_masked_loss(logits, f_labels, exact_mask)
            cens_mask = loss_mask * ceiling
            if cens_mask.any():
                loss = loss + CENS_WEIGHT * hl.censored_loss(logits, f_labels, cens_mask)
        else:
            loss = self._compute_masked_loss(logits, f_labels, loss_mask)
        self.val_loss(loss)
        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        return loss
