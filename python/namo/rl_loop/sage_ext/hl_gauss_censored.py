"""CensoredHLGauss — HL-Gauss head + one-sided ("ceiling") observations. [beast-0a]

Search is a censoring instrument: a depth-k search that finds an opening yields an EXACT value
(gamma^(d-1)); one that doesn't yields only a CEILING (V <= gamma^k). Exact cells keep the HL-Gauss
soft-target CE. Ceiling cells use the discrete-survival censored likelihood: -log P(V <= c) — punish
ONLY predicted mass above the ceiling; below it the model carries no gradient (the data doesn't know).
Do NOT regress a (truncated) Gaussian at the ceiling — that invents "V ~= c" information.

Endpoint fix: an exact target at vmax (opener=1.0) is ONE-HOT at the top bin. The parent smears a
Gaussian that gets chopped at the range edge and renormalised (target mean ~0.985) — a systematic
shave on the ranker's anchor label.

Ceilings (0.9, 0.81 = gamma^k) do not land on the 51-bin grid: the straddled bin contributes the
FRACTION of its width below the ceiling (interpolated CDF), so the constraint is exact, not snapped.
"""
import torch

from ._sage import HLGauss


class CensoredHLGauss(HLGauss):
    _END_EPS = 1e-6

    def target(self, y: torch.Tensor) -> torch.Tensor:
        t = super().target(y)
        hi = (y >= self.vmax - self._END_EPS)
        lo = (y <= self.vmin + self._END_EPS)
        if hi.any() or lo.any():
            t = t.clone()
            if hi.any():
                t[hi] = 0.0
                t[hi.unsqueeze(-1).expand_as(t) & (torch.arange(t.shape[-1], device=t.device) == t.shape[-1] - 1)] = 1.0
            if lo.any():
                t[lo] = 0.0
                t[lo.unsqueeze(-1).expand_as(t) & (torch.arange(t.shape[-1], device=t.device) == 0)] = 1.0
        return t

    def censored_loss(self, logits: torch.Tensor, ceilings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """-log P(V <= ceiling) on masked cells; mean over the mask (group-mean reduction).

        logits: (..., num_bins); ceilings: (...) upper bounds in [vmin, vmax]; mask: (...) {0,1}.
        Per-bin weight toward the cumulative = clip((c - lower_edge)/width, 0, 1): full bins below
        the ceiling count 1, the straddled bin counts its sub-ceiling fraction, bins above count 0.
        """
        p = torch.softmax(logits, dim=-1)
        edges = self.bin_edges.to(logits.device)
        width = (edges[1] - edges[0])
        lower = edges[:-1]                                            # (num_bins,)
        w = ((ceilings.unsqueeze(-1) - lower) / width).clamp(0.0, 1.0)
        P = (p * w).sum(-1)                                           # (...) mass at-or-below ceiling
        nll = -torch.log(P.clamp_min(1e-8))
        mask = mask.float()
        return (nll * mask).sum() / mask.sum().clamp_min(1.0)
