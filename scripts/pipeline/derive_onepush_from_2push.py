#!/usr/bin/env python3
"""Derive the eval-compatible 1-push answer key from the unified depth-2 collection's labels.

The new test set is collected in ONE depth-2 exhaustive pass (build_2push_validset.py → twopush.json),
which carries per episode both `valid_1push` (F, chain-depth-1) and `valid_first_push` (F1', chain-depth-2)
under the new success bar. This script projects the depth-1 part into the schema `eval_scorer.py --episodes`
expects ({xml: [{object_id, object_center, object_theta, region, valid, tried, solve_rate}]}), keeping only
1-push-solvable episodes (valid_1push non-empty) — that's the 1-push tier.

KEY COMPATIBILITY: eval_scorer matches an H5 sample by its `xml` field, which historically equals the
`outputs/test_*_phase1/...` symlink path (the v3_test_episodes.json keys). twopush.json is keyed by realpath,
so we re-key each scene to the ORIGINAL eval key when --eval-key-src maps realpath→that key; otherwise realpath.
"""
import argparse
import json
import os

RP = os.path.realpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twopush", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-key-src", help="old v3_test_episodes.json — inherit its xml keys (realpath-matched) "
                    "so the new key is drop-in for eval_scorer.py. Optional; default = realpath keys.")
    a = ap.parse_args()

    two = json.load(open(a.twopush))

    # realpath -> original eval key (symlink path), to preserve eval-time lookup
    real2key = {}
    if a.eval_key_src and os.path.exists(a.eval_key_src):
        src = json.load(open(a.eval_key_src))
        for k in src:
            real2key[RP(k)] = k

    out = {}
    n_scene = n_ep = 0
    for real, eps in two.items():
        key = real2key.get(real, real)
        rows = []
        for e in eps:
            v = e.get("valid_1push") or []
            t = e.get("tried_1push") or []
            if not v:                              # 1-push tier = solvable in one push under the new bar
                continue
            rows.append({
                "object_id": e.get("object_id"),
                "object_center": e.get("object_center"),
                "object_theta": e.get("object_theta"),
                "region": e.get("region"),
                "valid": v,
                "tried": t,
                "solve_rate": (len(v) / len(t)) if t else 0.0,
            })
        if rows:
            out[key] = rows
            n_scene += 1
            n_ep += len(rows)

    json.dump(out, open(a.out, "w"))
    print(f"1-push key: {n_scene} scenes, {n_ep} episodes (1-push-solvable under new bar) -> {a.out}")
    if real2key:
        n_inherited = sum(1 for real in two if real in real2key)
        print(f"  inherited {n_inherited} eval keys from {os.path.basename(a.eval_key_src)} "
              f"(rest fall back to realpath)")


if __name__ == "__main__":
    main()
