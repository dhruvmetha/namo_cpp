#!/usr/bin/env python3
"""Diff two timing/eval jsonl runs (OLD render vs NEW render) on the SAME episodes/seed.
Keys by (xml, object_id, model). Reports: per-episode solved-flag mismatches (MUST be 0 for the
render fix to be solve-preserving) + aggregate solve% per (model, tier) old vs new."""
import sys, json
from collections import defaultdict


def load(fn):
    d = {}
    for line in open(fn):
        r = json.loads(line)
        d[(r["xml"], r["object_id"], r["model"])] = r
    return d


def main():
    old = load(sys.argv[1]); new = load(sys.argv[2]); label = sys.argv[3] if len(sys.argv) > 3 else ""
    keys = sorted(set(old) & set(new))
    mism = [k for k in keys if bool(old[k]["solved"]) != bool(new[k]["solved"])]
    # model-only (render-dependent) mismatches: random doesn't render
    mism_model = [k for k in mism if k[2] != "random"]
    print(f"=== {label}  matched episodes={len(keys)}  (old-only={len(set(old)-set(new))}, new-only={len(set(new)-set(old))})")
    print(f"  SOLVED-FLAG MISMATCHES: {len(mism)} total  |  {len(mism_model)} on the rendering models (Hz/NoHz)")
    if mism_model:
        for k in mism_model[:10]:
            print(f"    DIFF {k}: old.solved={old[k]['solved']} new.solved={new[k]['solved']}")
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # (model,tier) -> [n, old_solved, new_solved]
    for k in keys:
        m, t = k[2], old[k]["tier"]
        agg[m][t][0] += 1; agg[m][t][1] += int(old[k]["solved"]); agg[m][t][2] += int(new[k]["solved"])
    print(f"  {'model':<7}{'tier':<5}{'n':>4}{'old_solve%':>11}{'new_solve%':>11}")
    for m in ["Hz", "NoHz", "random"]:
        for t in ["easy", "med", "hard"]:
            n, o, nw = agg[m][t]
            if n:
                print(f"  {m:<7}{t:<5}{n:>4}{100*o/n:>11.1f}{100*nw/n:>11.1f}")
    print(f"  >>> VERDICT: {'IDENTICAL solve decisions on Hz/NoHz (render fix is solve-preserving)' if not mism_model else 'MISMATCH — render changed solve behavior!'}")


if __name__ == "__main__":
    main()
