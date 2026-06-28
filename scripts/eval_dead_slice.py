#!/usr/bin/env python3
"""Dead-slice probe (M2b gate, H0b): does the model VALUE hopeless states low?

For N dead + N solvable rows from a scorer H5, compute each state's value V = top-k-mean of the
predicted (60,5) map (the deploy-time pool; NOT raw max — fluke-dominated) and report the
separation: mean V per class + AUC(dead vs solvable). A model that never saw dead-ends (M2a) is
the CONTROL — H0b predicts it values dead states ~like solvable ones; M2b passing = V_dead <<
V_solvable with high AUC.

Pools over ALL 300 cells by default (deploy-realistic: the model doesn't know reachability) and
also reports the candidate-set pool (cells with r_mask>0) — if all-cells fails while candidate
pool separates, the untried-cell optimism leak is the cause (see journal).

  python scripts/eval_dead_slice.py --ckpt <.ckpt> --h5 $NAMO_SCRATCH/h5/v4_hq_m2b_scorer/data.h5 \
      --n 500 --h 1 --out $NAMO_SCRATCH/eval/dead_slice_<name>.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/scripts", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
from eval_scorer import load_scorer  # noqa: E402
from namo.paths import H5  # noqa: E402


def auc(pos, neg):
    """P(score_pos > score_neg) by rank — pos should be HIGH (solvable)."""
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    rp = r[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", default=str(H5 / "v4_hq_m2b_scorer/data.h5"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--h", type=int, default=1, help="budget H for conditioned models")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model = load_scorer(a.ckpt, 5, "cpu", "edge_crossattn")
    budget = getattr(model.network, "budget_cond", False)
    bins = getattr(model.network, "value_bins", 0)

    f = h5py.File(a.h5, "r")
    dead = f["dead"][:]
    rng = np.random.default_rng(0)
    idx_d = rng.choice(np.where(dead == 1)[0], size=min(a.n, int((dead == 1).sum())), replace=False)
    idx_s = rng.choice(np.where(dead == 0)[0], size=min(a.n, int((dead == 0).sum())), replace=False)

    def values(idxs):
        v_all, v_cand = [], []
        for i in sorted(idxs.tolist()):
            ctx = torch.from_numpy(f["ctx"][i][None]).float()
            cpx = torch.from_numpy(f["contact_px"][i][None]).float()
            kw = {"H": torch.full((1,), a.h, dtype=torch.long)} if budget else {}
            if getattr(model.network, "reach_flag_input", False):
                rm = f["r_mask"][i]
                kw["reach_edges"] = torch.from_numpy((rm.sum(axis=1) > 0).astype("int64"))[None]
            with torch.no_grad():
                t = model(ctx, cpx, **kw)[0]
            if t.dim() == 3:
                from src.model.hl_gauss import HLGauss
                t = HLGauss(num_bins=t.shape[-1]).value(t)
            else:
                t = torch.sigmoid(t)
            s = t.numpy().reshape(-1)
            v_all.append(float(np.sort(s)[-a.topk:].mean()))
            cand = s[f["r_mask"][i].reshape(-1) > 0]
            v_cand.append(float(np.sort(cand)[-min(a.topk, len(cand)):].mean()) if len(cand) else 0.0)
        return np.array(v_all), np.array(v_cand)

    va_d, vc_d = values(idx_d)
    va_s, vc_s = values(idx_s)
    res = {
        "ckpt": a.ckpt, "h5": a.h5, "n_dead": len(idx_d), "n_solv": len(idx_s),
        "H": a.h, "topk": a.topk, "budget_cond": bool(budget), "value_bins": int(bins),
        "all_cells": {"V_dead": float(va_d.mean()), "V_solv": float(va_s.mean()),
                      "auc": float(auc(va_s, va_d))},
        "candidate_pool": {"V_dead": float(vc_d.mean()), "V_solv": float(vc_s.mean()),
                           "auc": float(auc(vc_s, vc_d))},
    }
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
