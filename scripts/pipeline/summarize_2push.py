#!/usr/bin/env python3
"""Summarize the 2-push answer key (F1') + light CROSS-CHECK vs the 1-push key.

Reads labels/twopush.json (from build_2push_validset.py) and labels/onepush.json. Produces
stats/twopush_stats.json:
  - coverage + true MIN-DEPTH partition (1push-solvable | genuine-2push-only | unsolved-within-2push)
  - F1' size + solve_rate_first_push histograms; per-corpus breakdown
  - cross-check on the OVERLAP with the 1-push key: does the depth-2 collection's re-derived depth-1 F
    agree with the canonical 1-push `valid`? (overlap is small — most 2-push-tier scenes are 1-push-fail.)
"""
import argparse
import json
import os
from collections import Counter


def match_oc(eps, oc, tol=0.001):
    best, bd = None, 1e9
    for e in eps:
        c = e.get("object_center")
        if not c:
            continue
        d = ((c[0] - oc[0]) ** 2 + (c[1] - oc[1]) ** 2) ** 0.5
        if d < bd:
            best, bd = e, d
    return (best, bd) if bd <= tol else (None, bd)


def corpus_of(real):
    return "aug9" if "/aug9_car/" in real else ("feb" if "/feb_car/" in real else "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twopush", required=True)
    ap.add_argument("--onepush", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    two = json.load(open(a.twopush))
    one = json.load(open(a.onepush))

    mindepth = Counter(); f1p_sizes = Counter(); srfp = Counter(); corpus = Counter()
    n_eps = 0
    for real, eps in two.items():
        for e in eps:
            n_eps += 1
            corpus[corpus_of(real)] += 1
            is1 = e.get("is_1push_solvable")
            nF1p = len(e.get("valid_first_push") or [])
            if is1:
                mindepth["1push_solvable"] += 1
            elif nF1p > 0:
                mindepth["genuine_2push_only"] += 1
            else:
                mindepth["unsolved_within_2push"] += 1
            f1p_sizes[min(nF1p, 30)] += 1
            srfp[round((e.get("solve_rate_first_push") or 0.0) // 0.05 * 0.05, 2)] += 1

    matched = exact = 0
    for real, eps in two.items():
        okey = one.get(real)
        if not okey:
            continue
        for e in eps:
            oc = e.get("object_center")
            om, d = match_oc(okey, oc) if oc else (None, 9)
            if om is None:
                continue
            matched += 1
            if {tuple(t) for t in (e.get("valid_1push") or [])} == {tuple(t) for t in (om.get("valid") or [])}:
                exact += 1

    stats = {
        "n_scenes": len(two),
        "n_episodes": n_eps,
        "min_depth_partition": dict(mindepth),
        "n_genuine_2push_only": mindepth.get("genuine_2push_only", 0),
        "corpus": dict(corpus),
        "f1prime_size_hist (capped@30)": dict(sorted(f1p_sizes.items())),
        "solve_rate_first_push_hist_0.05": dict(sorted(srfp.items())),
        "cross_check_overlap_with_1push_key": {
            "episodes_matched": matched, "exact_F_agreement": exact,
            "exact_frac": round(exact / matched, 4) if matched else None,
            "note": "overlap is small by design — the 2-push tier is mostly 1-push-FAIL scenes",
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(stats, open(a.out, "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
