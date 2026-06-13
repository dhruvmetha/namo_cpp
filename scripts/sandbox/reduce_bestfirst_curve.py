#!/usr/bin/env python3
"""Solve@K curve from best-first leaf jsonls — capped at 900 sims.

Best-first explores candidates in a budget-INDEPENDENT order (the heap ordering doesn't depend on the
sim budget; the budget only truncates). So a single budget-900 run records, per episode, the exact sim
index at which it solved (or 900 if it never did) — and solve@K for ANY K<=900 = fraction of episodes
with solved & sims<=K. One 900-cap run therefore yields the WHOLE reactive->search curve.

  reduce_bestfirst_curve.py --label model  /scratch/dm1487/eval/bf900_model_ep16
  reduce_bestfirst_curve.py --label random --avg-seeds \
      /scratch/dm1487/eval/bf900_uniform_s0 ... /scratch/dm1487/eval/bf900_uniform_s4
"""
import sys, os, json, glob, argparse, math

KS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 900]


def load_dir(d):
    rows = []
    for f in glob.glob(os.path.join(d, "shard_*.jsonl")):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def curve(rows):
    n = len(rows)
    if not n:
        return None
    solved = [r for r in rows if r.get("solved")]
    at = {k: 100.0 * sum(1 for r in solved if r["sims"] <= k) / n for k in KS}
    return {"n": n, "n_solved": len(solved),
            "solve_at": at,
            "avg_sims_all": sum(r["sims"] for r in rows) / n,
            "avg_sims_to_solve": (sum(r["sims"] for r in solved) / len(solved)) if solved else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--avg-seeds", action="store_true",
                    help="treat each dir as one seed; report mean +/- std of the per-seed curves")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()

    cs = []
    for d in a.dirs:
        c = curve(load_dir(d))
        if c is None:
            print(f"  {os.path.basename(d):28s}: no rows", file=sys.stderr); continue
        cs.append(c)
        print(f"  {os.path.basename(d):28s}: n={c['n']:4d} solved={c['n_solved']:4d} "
              f"@900={c['solve_at'][900]:.1f}% avg_sims_all={c['avg_sims_all']:.1f} "
              f"avg_to_solve={c['avg_sims_to_solve']:.1f}", file=sys.stderr)

    hdr = "  ".join(f"@{k}" for k in KS)
    print(f"\n=== {a.label} : solve@K (cap 900) ===")
    print(f"{'':10s}{hdr}")
    if a.avg_seeds and len(cs) > 1:
        out = {"label": a.label, "n_seeds": len(cs), "per_seed_n": [c["n"] for c in cs]}
        means = {}
        for k in KS:
            vals = [c["solve_at"][k] for c in cs]
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
            means[k] = (m, sd)
        print("mean      " + "  ".join(f"{means[k][0]:.1f}" for k in KS))
        print("std       " + "  ".join(f"{means[k][1]:.1f}" for k in KS))
        out["solve_at_mean"] = {str(k): round(means[k][0], 2) for k in KS}
        out["solve_at_std"] = {str(k): round(means[k][1], 2) for k in KS}
        out["avg_sims_all_mean"] = round(sum(c["avg_sims_all"] for c in cs) / len(cs), 1)
        out["avg_sims_to_solve_mean"] = round(sum(c["avg_sims_to_solve"] for c in cs) / len(cs), 1)
        print(json.dumps(out))
    else:
        c = cs[0]
        print(c["label"] if False else "value     " + "  ".join(f"{c['solve_at'][k]:.1f}" for k in KS))
        out = {"label": a.label, "n": c["n"], "n_solved": c["n_solved"],
               "solve_at": {str(k): round(c["solve_at"][k], 2) for k in KS},
               "avg_sims_all": round(c["avg_sims_all"], 1),
               "avg_sims_to_solve": round(c["avg_sims_to_solve"], 1)}
        print(json.dumps(out))


if __name__ == "__main__":
    main()
