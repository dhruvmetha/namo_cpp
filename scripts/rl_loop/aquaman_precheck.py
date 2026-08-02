#!/usr/bin/env python3
"""Aquaman H0 precheck — can the deploy model's view of an UNSWEPT remainder detect life?

EXP-2026-08-02-bootstrap-value-loop, round 0 step 1. Zero sims; reuses eval_auc.score_h5
(cached ckpt x H5 forward passes).

The quiz (counterfactual sweep-stop): child boards in d20_plus_setup_only.h5 are near-
exhaustively swept (median 70 tried cells, ~52% contain a verified winner). Per board:
order TRIED cells by the model's own score, pretend the sweep stopped after the top-K,
and grade the model's V-hat = top5-mean over the REMAINDER (outcomes known):
  class LIVE = remainder contains a winner cell (value_target 1.0)
  class DEAD = remainder all failed (ceiling cells)
Report Mann-Whitney AUC(LIVE vs DEAD) at K in {10,20,30} + the bootstrap-target preview
min(0.81, 0.9*V-hat) histogram on DEAD remainders (what aquaman-0 would write there).
Also: V-hat over the genuinely-untried cells (no truth; real target preview for those).

H0 bar (pre-registered in the card): AUC >= 0.75 proceed; ~0.5 stop.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "python", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_auc import score_h5  # noqa: E402  cached (ckpt, h5) -> (N,60,5) model values
from eval_common import mw_auc  # noqa: E402

TOP_M = 5  # V-hat aggregator = mean of top-5 remainder scores (deploy V uses mean5)


def vhat(scores):
    top = np.sort(scores)[::-1][:TOP_M]
    return float(top.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--min-remainder", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    values = score_h5(args.ckpt, args.h5, args.device)  # (N,60,5)
    out = {"ckpt": args.ckpt, "h5": args.h5, "ks": {}}
    with h5py.File(args.h5, "r") as f:
        is_root = f["is_root"][:].astype(bool)
        child_idx = np.where(~is_root)[0]
        vm = f["value_mask"]
        rm = f["r_mask"]
        vt = f["value_target"]

        per_k = {k: {"live": [], "dead": []} for k in args.ks}
        untried_vhats = []
        for i in child_idx:
            tried = (vm[i] > 0.5) & (rm[i] > 0.5)
            if not tried.any():
                continue
            s = values[i]
            win = tried & (vt[i] > 0.95)
            t_flat = np.where(tried.ravel())[0]
            order = t_flat[np.argsort(s.ravel()[t_flat])[::-1]]  # tried cells, model-score desc
            win_flat = set(np.where(win.ravel())[0])
            for k in args.ks:
                if len(order) < k + args.min_remainder:
                    continue
                rem = order[k:]
                v = vhat(s.ravel()[rem])
                (per_k[k]["live"] if any(c in win_flat for c in rem) else per_k[k]["dead"]).append(v)
            untried = (rm[i] > 0.5) & ~(vm[i] > 0.5)
            if untried.sum() >= 1:
                untried_vhats.append(vhat(s.ravel()[np.where(untried.ravel())[0]]))

    hist_edges = np.linspace(0.0, 0.81, 28)
    for k in args.ks:
        live, dead = per_k[k]["live"], per_k[k]["dead"]
        auc = mw_auc(np.array(live), np.array(dead)) if live and dead else None
        tgt_dead = np.minimum(0.81, 0.9 * np.array(dead)) if dead else np.array([])
        out["ks"][k] = {
            "n_live": len(live), "n_dead": len(dead), "auc_live_vs_dead": auc,
            "vhat_live_median": float(np.median(live)) if live else None,
            "vhat_dead_median": float(np.median(dead)) if dead else None,
            "target_dead_hist": np.histogram(tgt_dead, bins=hist_edges)[0].tolist(),
            "target_dead_quartiles": [float(q) for q in np.percentile(tgt_dead, [25, 50, 75])] if len(tgt_dead) else None,
        }
        print(f"K={k}: n_live={len(live)} n_dead={len(dead)} AUC={auc if auc is None else round(auc,3)} "
              f"V-hat median live={out['ks'][k]['vhat_live_median']} dead={out['ks'][k]['vhat_dead_median']}")
    if untried_vhats:
        u = np.array(untried_vhats)
        tgt = np.minimum(0.81, 0.9 * u)
        out["untried"] = {"n_boards": len(u),
                         "target_quartiles": [float(q) for q in np.percentile(tgt, [25, 50, 75])],
                         "target_hist": np.histogram(tgt, bins=hist_edges)[0].tolist(),
                         "clip_frac_at_081": float((0.9 * u >= 0.81).mean())}
        print(f"true-untried boards={len(u)} target quartiles={out['untried']['target_quartiles']} "
              f"clip@0.81={out['untried']['clip_frac_at_081']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
