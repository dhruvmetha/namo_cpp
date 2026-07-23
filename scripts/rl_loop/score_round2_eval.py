#!/usr/bin/env python3
"""Score exact setup/opener ordering on the exhaustive round2 held-out H5."""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for path in (REPO / "build_python", REPO / "python", REPO / "scripts", REPO / "scripts/sandbox"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if os.environ.get("SAGE_REPO"):
    sys.path.insert(0, os.environ["SAGE_REPO"])


def mw_auc(positive, negative):
    if not len(positive) or not len(negative):
        return float("nan")
    values = np.concatenate([positive, negative])
    ranks = values.argsort().argsort().astype(np.float64) + 1
    npos = len(positive)
    return float((ranks[:npos].sum() - npos * (npos + 1) / 2) / (npos * len(negative)))


def first_positive_rank(scores, positive, candidates):
    order = np.argsort(-scores[candidates], kind="stable")
    ranked_positive = positive[candidates][order]
    hits = np.flatnonzero(ranked_positive)
    return int(hits[0] + 1) if hits.size else None


def summarize_ranks(ranks):
    values = np.asarray(ranks)
    return {
        "n_boards": int(values.size),
        "hit_at_1_pct": round(100.0 * np.count_nonzero(values <= 1) / values.size, 1),
        "hit_at_5_pct": round(100.0 * np.count_nonzero(values <= 5) / values.size, 1),
        "median_first_rank": float(np.median(values)),
        "mean_first_rank": round(float(values.mean()), 2),
        "p90_first_rank": float(np.percentile(values, 90)),
    }


def score_checkpoint(checkpoint, h5_path, batch_size, limit):
    from live_scorer import LiveScorer
    from src.model.hl_gauss import HLGauss

    scorer = LiveScorer(ckpt=checkpoint)
    model = scorer.model.eval()
    device = scorer.device
    setup_scores, root_dead_scores = [], []
    opener_scores, finish_dead_scores = [], []
    setup_ranks, opener_ranks = [], []
    recall20_hit = recall20_total = 0

    with h5py.File(h5_path, "r") as data:
        total = min(limit or len(data["ctx"]), len(data["ctx"]))
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            context = torch.from_numpy(data["ctx"][start:end].astype(np.float32)).to(device)
            contact = torch.from_numpy(data["contact_px"][start:end].astype(np.float32)).to(device)
            with torch.no_grad():
                logits = model(context, contact)
                values = HLGauss(num_bins=logits.shape[-1]).value(logits.float()).cpu().numpy()
            labels = data["value_target"][start:end]
            known = (data["value_mask"][start:end] > 0.5) & (data["r_mask"][start:end] > 0.5)
            reachable = data["r_mask"][start:end] > 0.5
            kinds = [x.decode() if isinstance(x, bytes) else str(x) for x in data["node_kind"][start:end]]

            for index, kind in enumerate(kinds):
                score = values[index]
                label = labels[index]
                exact = known[index]
                if kind == "root":
                    positive = exact & np.isclose(label, 0.9)
                    negative = exact & np.isclose(label, 0.0)
                    setup_scores.append(score[positive])
                    root_dead_scores.append(score[negative])
                    if positive.any() and negative.any():
                        setup_ranks.append(first_positive_rank(score, positive, positive | negative))
                elif kind == "depth2":
                    positive = exact & np.isclose(label, 1.0)
                    negative = exact & np.isclose(label, 0.0)
                    opener_scores.append(score[positive])
                    finish_dead_scores.append(score[negative])
                    if positive.any() and negative.any():
                        opener_ranks.append(first_positive_rank(score, positive, positive | negative))
                    if positive.any() and int(negative.sum()) >= 20:
                        order = np.argsort(-score[reachable[index]], kind="stable")
                        recall20_total += 1
                        recall20_hit += int(positive[reachable[index]][order[:20]].any())
            print(f"{end}/{total}", flush=True)

    setup_scores = np.concatenate(setup_scores)
    root_dead_scores = np.concatenate(root_dead_scores)
    opener_scores = np.concatenate(opener_scores)
    finish_dead_scores = np.concatenate(finish_dead_scores)
    return {
        "checkpoint": checkpoint,
        "root_setup_ordering": {
            "setup_vs_dead_auc": round(mw_auc(setup_scores, root_dead_scores), 4),
            "n_setup_cells": int(setup_scores.size),
            "n_dead_cells": int(root_dead_scores.size),
            "setup_median_score": round(float(np.median(setup_scores)), 4),
            "dead_median_score": round(float(np.median(root_dead_scores)), 4),
            **summarize_ranks(setup_ranks),
        },
        "finish_opener_guard": {
            "opener_vs_dead_auc": round(mw_auc(opener_scores, finish_dead_scores), 4),
            "n_opener_cells": int(opener_scores.size),
            "n_dead_cells": int(finish_dead_scores.size),
            "opener_median_score": round(float(np.median(opener_scores)), 4),
            "dead_median_score": round(float(np.median(finish_dead_scores)), 4),
            **summarize_ranks(opener_ranks),
            "recall_at_20_pct": round(100.0 * recall20_hit / recall20_total, 1),
            "recall_at_20_n": recall20_total,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    results = [score_checkpoint(path, args.h5, args.batch_size, args.limit) for path in args.ckpt]
    output = {"h5": args.h5, "models": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
