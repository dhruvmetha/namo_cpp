#!/usr/bin/env python3
"""Assemble the 2x2 matrix {Horizon,NoHorizon}x{v1,v2}: ranking (hard@1 H=1/H=2) + SOLVE (best-first @900).

Reads, per run:
  $NAMO_SCRATCH/eval/<run>_rank/<tag>_h{1,2}.json   (eval_scorer; divisions.hard.scorer_realistic['@1'])
  $NAMO_SCRATCH/eval/bf900_<run>/shard_*.jsonl       (best-first solve leaves)
Random baseline (model-agnostic): bf900_uniform_s0..4 (5-seed mean).

  python scripts/sandbox/reduce_2x2.py
"""
import os, sys, json, glob, math
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/build_python", f"{REPO}/python"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from namo.paths import SCRATCH  # noqa: E402
EVAL = str(SCRATCH / "eval")
RUNS = {  # label -> run_name (seed 1)
    "Horizon-v1":   "qfull_v4hq_s1",
    "NoHorizon-v1": "qfull_nohz_v4hq_s1",
    "Horizon-v2":   "qfull_v2_v4hq_s1",
    "NoHorizon-v2": "qfull_nohz_v2_v4hq_s1",
}
KS = [2, 10, 50, 100, 900]


def rank_hard_at1(run, h):
    fs = glob.glob(f"{EVAL}/{run}_rank/*_h{h}.json")
    if not fs:
        return None
    d = json.load(open(fs[0]))
    try:
        return d["divisions"]["hard"]["scorer_realistic"]["@1"]
    except Exception:
        return None


def solve_curve(d):
    rows = []
    for f in glob.glob(f"{d}/shard_*.jsonl"):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    if not rows:
        return None
    n = len(rows); solved = [r for r in rows if r.get("solved")]
    at = {k: 100.0 * sum(1 for r in solved if r["sims"] <= k) / n for k in KS}
    avg = (sum(r["sims"] for r in solved) / len(solved)) if solved else 0.0
    return n, at, avg


def random_curve():
    dirs = [f"{EVAL}/bf900_uniform_s{i}" for i in range(5)]
    cs = [solve_curve(d) for d in dirs]
    cs = [c for c in cs if c]
    if not cs:
        return None
    means = {k: sum(c[1][k] for c in cs) / len(cs) for k in KS}
    avg = sum(c[2] for c in cs) / len(cs)
    return means, avg


def main():
    print("\n================ 2x2 MATRIX — RANKING (hard@1) + SOLVE (best-first @900) ================\n")
    hdr = f"{'cell':14s} | {'rankH1':>6s} {'rankH2':>6s} | " + " ".join(f"s@{k:<4d}" for k in KS) + f" {'avgSim':>6s}"
    print(hdr); print("-" * len(hdr))
    for label, run in RUNS.items():
        r1, r2 = rank_hard_at1(run, 1), rank_hard_at1(run, 2)
        sc = solve_curve(f"{EVAL}/bf900_{run}")
        r1s = f"{r1:6.1f}" if r1 is not None else "   -- "
        r2s = f"{r2:6.1f}" if r2 is not None else "   -- "
        if sc:
            n, at, avg = sc
            sols = " ".join(f"{at[k]:5.1f}" for k in KS) + f" {avg:6.1f}"
            tail = f"(n={n})"
        else:
            sols = " ".join(f"{'--':>5s}" for _ in KS) + f" {'--':>6s}"; tail = "(no solve yet)"
        print(f"{label:14s} | {r1s} {r2s} | {sols}  {tail}")
    rc = random_curve()
    if rc:
        means, avg = rc
        sols = " ".join(f"{means[k]:5.1f}" for k in KS) + f" {avg:6.1f}"
        print(f"{'RANDOM(5seed)':14s} | {'  -- ':>6s} {'  -- ':>6s} | {sols}  (shared baseline)")
    print("\nrankH1/H2 = eval_scorer hard@1 at budget 1 / 2 (onepush key). s@K = best-first solve-rate within K sims")
    print("(object-constrained pure2push, cap 900). avgSim = avg sims-to-solve. H4 fix target: rankH2 -> rankH1.\n")


if __name__ == "__main__":
    main()
