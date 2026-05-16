#!/usr/bin/env python3
"""Direction-hypothesis test.

Hypothesis: the ML model has learned scene → direction (face / contact point)
but has a collapsed depth marginal. If true, ML predictions should hit
F at the face level (4-way) and contact-point level (60-way, ignoring depth)
substantially better than uniform-random-from-R, even on hard problems where
the joint (edge, depth) hit-rate is below random.

Accept H_direction: face-hit lift > 0 on hard / very_hard at K=1..3.
Reject H_direction: face-hit lift <= 0 on hard buckets.

Usage:
    python test_direction_hypothesis.py \\
        --gt-dir /common/users/dm1487/scratch_namo/f_char_2push_test_300_chain1/modular_data_westeros \\
        --ml-preds /common/users/dm1487/scratch_namo/ml_preds_2push_test_300_chain1.pkl
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


POINTS_PER_FACE = 15  # matches points_per_face in the YAML configs
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


def edge_to_face(edge_idx: int) -> int:
    return int(edge_idx) // POINTS_PER_FACE


def project(slots: set, level: str) -> set:
    """Project a set of (edge_idx, depth) onto coarser axis."""
    if level == "joint":
        return slots
    if level == "contact":
        return {int(e) for (e, d) in slots}
    if level == "face":
        return {edge_to_face(e) for (e, d) in slots}
    raise ValueError(level)


def _instance_key(ep) -> Optional[Tuple[str, str, str]]:
    stats = ep.get("algorithm_stats") or {}
    if not isinstance(stats, dict):
        return None
    return (ep.get("xml_file"), stats.get("neighbour_region_label"),
            stats.get("chosen_object_id"))


def load_gt(gt_dir: str) -> Dict[Tuple[str, str, str], Dict[str, set]]:
    out = {}
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
            R = set(); F = set()
            for t in tlog:
                if t.get("chain_depth", 1) != 1:
                    continue
                slot = (t["edge_idx"], t["depth"])
                R.add(slot)
                if t["success"]:
                    F.add(slot)
            out[k] = {"R": R, "F": F}
    return out


def load_ml(ml_pkl: str) -> Tuple[Dict[Tuple[str, str, str], Dict], Dict]:
    raw = pickle.load(open(ml_pkl, "rb"))
    out = {}
    for r in raw["results"]:
        if not r.get("ok"):
            continue
        out[tuple(r["key"])] = r
    return out, raw


def topk_aligned(aligned: List[Dict[str, Any]], K: int,
                 filter_to: Optional[set] = None) -> List[Tuple[int, int]]:
    """Return top-K (edge, depth) slots by vote count, optionally filtered to R."""
    by_votes = sorted(aligned, key=lambda x: -x.get("votes", 0))
    out = []
    for p in by_votes:
        if p.get("edge_idx") is None or p.get("depth_idx") is None:
            continue
        s = (int(p["edge_idx"]), int(p["depth_idx"]))
        if filter_to is not None and s not in filter_to:
            continue
        out.append(s)
        if len(out) >= K:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--ml-preds", required=True)
    p.add_argument("--ks", default="1,3,5,10,20,32")
    p.add_argument("--random-draws", type=int, default=200)
    p.add_argument("--csv-out", default=None)
    args = p.parse_args()

    Ks = [int(x) for x in args.ks.split(",")]
    rng = np.random.default_rng(42)

    gt = load_gt(args.gt_dir)
    ml, _meta = load_ml(args.ml_preds)
    keys = sorted(set(gt) & set(ml))
    print(f"GT instances: {len(gt)}  ML preds: {len(ml)}  joined: {len(keys)}")

    # Aggregator: (level, bucket, K) -> list of (ml_hit, rand_hit) per instance
    agg = defaultdict(list)

    rows = []  # for CSV
    for k in keys:
        g = gt[k]
        F = g["F"]; R = g["R"]
        if len(F) == 0 or len(R) == 0:
            continue
        ratio = len(F) / len(R)
        bucket = bucket_for_ratio(ratio)
        m = ml[k]

        # Pre-project F and R per level so we can do all K in one pass
        F_lv = {lv: project(F, lv) for lv in ("joint", "contact", "face")}
        R_lv = {lv: project(R, lv) for lv in ("joint", "contact", "face")}

        for K in Ks:
            # ML reachable-filtered top-K at the joint (edge,depth) level
            tk = topk_aligned(m["ml_aligned"], K, filter_to=R)
            for lv in ("joint", "contact", "face"):
                tk_lv = project(set(tk), lv)
                ml_hit = float(len(tk_lv & F_lv[lv]) > 0)
                # Random-from-R baseline, projected to same level
                R_list = list(R)
                K_eff = min(K, len(R_list))
                hits = 0
                for _ in range(args.random_draws):
                    sample_joint = set(map(tuple, rng.choice(
                        np.array(R_list), size=K_eff, replace=False)))
                    sample_lv = project(sample_joint, lv)
                    if sample_lv & F_lv[lv]:
                        hits += 1
                rand_hit = hits / args.random_draws
                agg[(lv, bucket, K)].append((ml_hit, rand_hit))
                rows.append({
                    "xml": k[0], "region": k[1], "object": k[2],
                    "bucket": bucket, "K": K, "level": lv,
                    "F_size_lv": len(F_lv[lv]),
                    "R_size_lv": len(R_lv[lv]),
                    "ml_hit": ml_hit, "rand_hit": rand_hit,
                })

    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote CSV: {args.csv_out}")

    bucket_order = [b[0] for b in DIFFICULTY_BINS]
    # Print one block per level
    for lv in ("face", "contact", "joint"):
        print(f"\n=== Level: {lv.upper()}  (project F and ML to this granularity, then Top-K hit-rate) ===")
        print(f"{'bucket':>11s} {'K':>3s} {'n':>4s}  "
              f"{'ml_hit':>7s} {'rand':>6s} {'lift':>7s}  ml-rand verdict")
        for b in bucket_order:
            for K in Ks:
                data = agg.get((lv, b, K), [])
                if not data:
                    continue
                n = len(data)
                mh = float(np.mean([x[0] for x in data]))
                rh = float(np.mean([x[1] for x in data]))
                lift = mh - rh
                # Quick verdict
                if lift > 0.05:
                    v = "ml-better"
                elif lift < -0.05:
                    v = "ml-worse"
                else:
                    v = "tie"
                print(f"{b:>11s} {K:>3d} {n:>4d}  {mh:>7.3f} {rh:>6.3f} {lift:>+7.3f}  {v}")

    # Headline summary
    print("\n=== DECISION RULE ===")
    print("H_direction: model learned scene->direction.")
    print("Accept if face-hit lift > 0 on HARD buckets at K=1..3.")
    print(f"\n{'bucket':>11s} {'K':>3s}  {'face_ml':>8s} {'face_rnd':>9s} {'lift':>7s}  "
          f"{'contact_ml':>10s} {'contact_rnd':>11s} {'lift':>7s}  "
          f"{'joint_ml':>9s} {'joint_rnd':>10s} {'lift':>7s}")
    for b in ("very_hard", "hard"):
        for K in (1, 3, 5):
            face = agg.get(("face", b, K), [])
            cont = agg.get(("contact", b, K), [])
            jnt = agg.get(("joint", b, K), [])
            if not face: continue
            fm = np.mean([x[0] for x in face]); fr = np.mean([x[1] for x in face])
            cm = np.mean([x[0] for x in cont]); cr = np.mean([x[1] for x in cont])
            jm = np.mean([x[0] for x in jnt]);  jr = np.mean([x[1] for x in jnt])
            print(f"{b:>11s} {K:>3d}  {fm:>8.3f} {fr:>9.3f} {fm-fr:>+7.3f}  "
                  f"{cm:>10.3f} {cr:>11.3f} {cm-cr:>+7.3f}  "
                  f"{jm:>9.3f} {jr:>10.3f} {jm-jr:>+7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
