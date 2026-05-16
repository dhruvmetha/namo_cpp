#!/usr/bin/env python3
"""Score ML predictions against ground-truth F.

Inputs:
    --gt-dir: directory of *_results.pkl files (F characterization GT)
    --ml-preds: pkl file written by ml_prediction_offline.py
    --out-dir: where to write figures + per-instance CSV

Metrics computed per (xml, neighbour, object) instance:
    hit@K       — at least one ML top-K slot is in F
    precision@K — |topK ∩ F| / |topK|
    recall@K    — |topK ∩ F| / |F|
    coverage@K  — |topK ∩ R| / |topK|  (R = reachable primitives)

Baseline: random-from-R same K, averaged over many draws (sanity floor).

Stratification:
    difficulty bucket: |F|/|R| → very_easy / easy / medium / hard / very_hard
                       (same bins as analyze_F.py)
    chain depth: 1 (F = F1) or 2 (F = F1', uses parent_edge_idx)
"""
from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIFFICULTY_BINS = [
    ("very_hard", 0.00, 0.05),
    ("hard",      0.05, 0.15),
    ("medium",    0.15, 0.40),
    ("easy",      0.40, 0.70),
    ("very_easy", 0.70, 1.01),
]


def bucket_for_ratio(r: float) -> str:
    for name, lo, hi in DIFFICULTY_BINS:
        if lo <= r < hi:
            return name
    return "very_easy"


def _instance_key(ep) -> Optional[Tuple[str, str, str]]:
    stats = ep.get("algorithm_stats") or {}
    if not isinstance(stats, dict):
        return None
    return (ep.get("xml_file"), stats.get("neighbour_region_label"),
            stats.get("chosen_object_id"))


def load_gt(gt_dir: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Return {key: {F: set, R: set, F1prime: set, trial_log: list}} deduped."""
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for pkl in sorted(Path(gt_dir).glob("*_results.pkl")):
        try:
            d = pickle.load(open(pkl, "rb"))
        except Exception:
            continue
        for ep in d.get("episode_results") or []:
            k = _instance_key(ep)
            if k is None or k[0] is None or k in out:
                continue
            stats = ep["algorithm_stats"]
            tlog = stats.get("primitive_trial_log")
            if not tlog:
                continue
            R = set()      # reachable primitives (chain_depth==1)
            F = set()      # feasible primitives (immediate opening, chain_depth==1)
            F1_prime = set()  # push-1 (edge,depth) that enabled a chain==2 success
            for t in tlog:
                cd = t.get("chain_depth", 1)
                e, dpt = t["edge_idx"], t["depth"]
                if cd == 1:
                    R.add((e, dpt))
                    if t["success"]:
                        F.add((e, dpt))
                elif cd == 2 and t["success"]:
                    pe, pd = t.get("parent_edge_idx"), t.get("parent_depth")
                    if pe is not None and pe >= 0 and pd is not None:
                        F1_prime.add((int(pe), int(pd)))
            out[k] = {
                "F": F, "R": R, "F1_prime": F1_prime, "trial_log": tlog,
                "n_trials": len(tlog),
            }
    return out


def load_ml(ml_pkl: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    raw = pickle.load(open(ml_pkl, "rb"))
    out = {}
    for r in raw["results"]:
        if not r.get("ok"):
            continue
        out[tuple(r["key"])] = r
    return out, raw


def topk_set(aligned: List[Dict[str, Any]], K: int) -> set:
    by_votes = sorted(aligned, key=lambda x: -x.get("votes", 0))
    return {(int(p["edge_idx"]), int(p["depth_idx"]))
            for p in by_votes[:K] if p.get("edge_idx") is not None}


def metrics_for(topk: set, F: set, R: set) -> Dict[str, float]:
    if not topk:
        return {"hit": 0.0, "prec": 0.0, "rec": 0.0, "cov": 0.0,
                "topk_size": 0, "F_size": len(F)}
    return {
        "hit": float(len(topk & F) > 0),
        "prec": len(topk & F) / len(topk),
        "rec": len(topk & F) / max(1, len(F)),
        "cov": len(topk & R) / len(topk),
        "topk_size": len(topk),
        "F_size": len(F),
    }


def random_baseline(R: set, F: set, K: int, draws: int = 200,
                    rng: np.random.Generator = None) -> Dict[str, float]:
    """Random-from-R baseline, averaged."""
    if not R or K <= 0:
        return {"hit": 0.0, "prec": 0.0, "rec": 0.0}
    if rng is None:
        rng = np.random.default_rng(42)
    R_list = list(R)
    K_eff = min(K, len(R_list))
    hits = precs = recs = 0.0
    F_set = F
    for _ in range(draws):
        sample = set(map(tuple, rng.choice(np.array(R_list), size=K_eff, replace=False)))
        inter = len(sample & F_set)
        hits += float(inter > 0)
        precs += inter / max(1, len(sample))
        recs += inter / max(1, len(F_set))
    return {"hit": hits / draws, "prec": precs / draws, "rec": recs / draws}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--ml-preds", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--horizon", choices=["1push", "2push"], default="1push",
                   help="Whether to score against F1 (1push) or F1' (2push)")
    p.add_argument("--ks", default="1,3,5,10,20,32",
                   help="Comma-separated K values to evaluate")
    p.add_argument("--random-draws", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Ks = [int(x) for x in args.ks.split(",")]

    print(f"Loading GT from {args.gt_dir}", flush=True)
    gt = load_gt(args.gt_dir)
    print(f"  {len(gt)} GT instances", flush=True)

    print(f"Loading ML preds from {args.ml_preds}", flush=True)
    ml_preds, meta = load_ml(args.ml_preds)
    print(f"  {len(ml_preds)} ML predictions", flush=True)

    # Join
    rows = []
    keys = sorted(set(gt.keys()) & set(ml_preds.keys()))
    print(f"Joined instances: {len(keys)}", flush=True)
    if not keys:
        print("ERROR: no overlapping instances between GT and ML preds.")
        return 2

    rng = np.random.default_rng(42)
    for k in keys:
        g = gt[k]
        m = ml_preds[k]
        F_target = g["F1_prime"] if args.horizon == "2push" else g["F"]
        R_set = g["R"]
        F_size = len(F_target)
        R_size = len(R_set)
        ratio = F_size / R_size if R_size else 0.0
        bucket = bucket_for_ratio(ratio)
        gated = F_size > 0
        for K in Ks:
            tk = topk_set(m["ml_aligned"], K)
            mtr = metrics_for(tk, F_target, R_set)
            base = random_baseline(R_set, F_target, K, draws=args.random_draws, rng=rng)
            rows.append({
                "xml": k[0], "region": k[1], "object": k[2],
                "horizon": args.horizon,
                "K": K,
                "F_size": F_size, "R_size": R_size, "F_over_R": ratio,
                "bucket": bucket, "gated": gated,
                "ml_hit": mtr["hit"], "ml_prec": mtr["prec"],
                "ml_rec": mtr["rec"], "ml_cov": mtr["cov"],
                "ml_topk_size": mtr["topk_size"],
                "rand_hit": base["hit"], "rand_prec": base["prec"],
                "rand_rec": base["rec"],
            })

    # Write CSV
    csv_path = out_dir / f"ml_vs_F_{args.horizon}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)", flush=True)

    # Aggregate by bucket
    print(f"\n=== {args.horizon.upper()} aggregate (gated to instances with F>0) ===")
    print(f"  meta: ML model = {meta['ml_model']}")
    print(f"        samples={meta['samples']} sampler={meta['sampler']} steps={meta['num_steps']}"
          f" pos_tol={meta['pos_tol']} ang_tol={meta['ang_tol']} k_nearest={meta['k_nearest']}")
    by = defaultdict(list)
    for r in rows:
        if not r["gated"]:
            continue
        by[(r["bucket"], r["K"])].append(r)
    overall = defaultdict(list)
    for r in rows:
        if not r["gated"]:
            continue
        overall[r["K"]].append(r)

    bucket_order = [b[0] for b in DIFFICULTY_BINS]
    print(f"\n{'bucket':>11s} {'K':>4s} {'n':>5s} "
          f"{'ml_hit':>8s} {'rand_hit':>10s} {'lift':>7s} "
          f"{'ml_prec':>9s} {'ml_rec':>8s} {'ml_cov':>8s} "
          f"{'avg_|F|':>8s} {'avg_|R|':>8s}")
    for b in bucket_order:
        for K in Ks:
            sub = by.get((b, K), [])
            if not sub:
                continue
            n = len(sub)
            mh = np.mean([r["ml_hit"] for r in sub])
            rh = np.mean([r["rand_hit"] for r in sub])
            mp = np.mean([r["ml_prec"] for r in sub])
            mr = np.mean([r["ml_rec"] for r in sub])
            mc = np.mean([r["ml_cov"] for r in sub])
            avgF = np.mean([r["F_size"] for r in sub])
            avgR = np.mean([r["R_size"] for r in sub])
            print(f"{b:>11s} {K:>4d} {n:>5d} "
                  f"{mh:>8.3f} {rh:>10.3f} {mh-rh:>+7.3f} "
                  f"{mp:>9.3f} {mr:>8.3f} {mc:>8.3f} "
                  f"{avgF:>8.1f} {avgR:>8.1f}")
    print(f"\n{'OVERALL':>11s} {'K':>4s} {'n':>5s} "
          f"{'ml_hit':>8s} {'rand_hit':>10s} {'lift':>7s} ml_rec ml_prec")
    for K in Ks:
        sub = overall.get(K, [])
        if not sub: continue
        mh = np.mean([r["ml_hit"] for r in sub])
        rh = np.mean([r["rand_hit"] for r in sub])
        mp = np.mean([r["ml_prec"] for r in sub])
        mr = np.mean([r["ml_rec"] for r in sub])
        print(f"{'':>11s} {K:>4d} {len(sub):>5d} {mh:>8.3f} {rh:>10.3f} {mh-rh:>+7.3f} {mr:>6.3f} {mp:>7.3f}")

    # ─── Plots ─────────────────────────────────────────────────────────────
    # Plot 1: hit@K vs K, per bucket
    fig, ax = plt.subplots(figsize=(7, 5))
    for b in bucket_order:
        ys = []
        ys_rand = []
        for K in Ks:
            sub = by.get((b, K), [])
            if not sub:
                ys.append(np.nan); ys_rand.append(np.nan); continue
            ys.append(np.mean([r["ml_hit"] for r in sub]))
            ys_rand.append(np.mean([r["rand_hit"] for r in sub]))
        if all(np.isnan(ys)):
            continue
        line, = ax.plot(Ks, ys, marker="o", label=f"{b} (ml)")
        ax.plot(Ks, ys_rand, marker="x", linestyle="--",
                color=line.get_color(), alpha=0.5, label=f"{b} (rand)")
    ax.set_xlabel("K (top-K aligned primitive slots)")
    ax.set_ylabel("hit@K  (P[≥1 ML slot in F])")
    ax.set_title(f"ML vs random Top-K hit-rate vs ground-truth F  ({args.horizon})")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"hit_at_K_{args.horizon}.png", dpi=140)
    plt.close(fig)

    # Plot 2: recall@K
    fig, ax = plt.subplots(figsize=(7, 5))
    for b in bucket_order:
        ys = []
        for K in Ks:
            sub = by.get((b, K), [])
            ys.append(np.mean([r["ml_rec"] for r in sub]) if sub else np.nan)
        if all(np.isnan(ys)): continue
        ax.plot(Ks, ys, marker="o", label=b)
    ax.set_xlabel("K"); ax.set_ylabel("recall@K (|topK ∩ F| / |F|)")
    ax.set_title(f"ML recall vs F  ({args.horizon})")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"recall_at_K_{args.horizon}.png", dpi=140)
    plt.close(fig)

    # Plot 3: coverage (sanity)
    fig, ax = plt.subplots(figsize=(7, 5))
    for b in bucket_order:
        ys = []
        for K in Ks:
            sub = by.get((b, K), [])
            ys.append(np.mean([r["ml_cov"] for r in sub]) if sub else np.nan)
        if all(np.isnan(ys)): continue
        ax.plot(Ks, ys, marker="o", label=b)
    ax.set_xlabel("K"); ax.set_ylabel("coverage@K (|topK ∩ R| / |topK|)")
    ax.set_title(f"ML prediction reachability sanity ({args.horizon})")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"coverage_at_K_{args.horizon}.png", dpi=140)
    plt.close(fig)

    print(f"\nFigures + CSV written under {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
