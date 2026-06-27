#!/usr/bin/env python3
"""DIAGNOSTIC (informed-2push H1/H2/H3): leaf recall@s1 + which first-push-value SCALAR best
separates good (leads-to-solvable-s1) from dead-end first pushes. Reuses BeamPlanner; no fork.

Per scene (collisions OFF = train-match, target-region-goal):
  sweep first pushes a1 (capped) -> sim s0->s1.  if s1 already opens -> a1 is 1-push, skip.
  else score s1 (_candidates -> 2nd pushes ordered by P); verify in scorer order up to top-K,
  stop at first that opens -> rank_succ (None = dead within top-K).
Per leaf we log candidate first-push-value scalars from the s1 map (NO extra sims) + the good/dead
label, so H3a (training-free scalar) and H3b (learned Q(s0,a1)) both read off the same dump.

Outputs: aggregate JSON (--out) + per-leaf JSONL (--leaf-out, for AUC sweep + H3b training seed).
Shard with --start/--end over the manifest for parallel runs.
"""
import sys, os, json, time, argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402
from scorer_beam import (BeamPlanner, make_env, make_action, read_manifest,  # noqa: E402
                         MANIFEST_2PUSH, FALLBACK_GOAL)
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import SCRATCH, MANIFESTS  # noqa: E402

PURE2PUSH = str(MANIFESTS / "test_pure2push_combined.txt")


def auc(pos, neg):
    """Mann-Whitney AUC = P(score(pos) > score(neg)); 0.5 = no separation."""
    if not pos or not neg:
        return float("nan")
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    # rank-based (ties = 0.5) but vectorized-ish; sizes are small (~hundreds)
    c = 0.0
    for a in pos:
        c += np.sum(a > neg) + 0.5 * np.sum(a == neg)
    return float(c / (len(pos) * len(neg)))


def leaf_scalars(pool2):
    """Candidate first-push-value scalars from the s1 scorer map (no sims)."""
    ps = np.array([float(p) for (_o, _g, p) in pool2], float)
    s = np.sort(ps)[::-1]
    return {
        "maxP": float(s[0]),
        "mean_top5": float(s[:5].mean()),
        "frac_ge_099": float((ps >= 0.99).mean()),
        "margin_top1_2": float(s[0] - (s[1] if len(s) > 1 else 0.0)),
        "n_pushes": int(len(ps)),
        "mean_all": float(ps.mean()),
    }


SCALAR_KEYS = ["maxP", "mean_top5", "frac_ge_099", "margin_top1_2", "n_pushes", "mean_all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=PURE2PUSH)
    ap.add_argument("--ckpt", default=None, help="scorer ckpt (default: champion; pass an m2b ckpt for the refreshed baseline)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=25)
    ap.add_argument("--first-cap", type=int, default=20)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--out", default=str(SCRATCH / "eval/diag_fpv.json"))
    ap.add_argument("--leaf-out", default=str(SCRATCH / "eval/diag_fpv_leaves.jsonl"))
    ap.add_argument("--scalars-only", action="store_true",
                    help="skip the beam solvability check (~20 sims/first-push); log only first-push-value "
                         "scalars + edge1/depth1. Use when grading vs the EXHAUSTIVE F1' (label set to -1).")
    a = ap.parse_args()

    planner = BeamPlanner(max_first=a.first_cap) if a.ckpt is None else BeamPlanner(ckpt=a.ckpt, max_first=a.first_cap)
    xmls = read_manifest(a.manifest, None)[a.start:a.end]

    recs = []   # per-leaf dicts
    ranks = []
    n_scenes = n_already = n_onepush = n_leaves = n_solv = n_sims = 0
    t0 = time.time()
    lf = open(a.leaf_out, "w")

    for xi, xml in enumerate(xmls):
        try:
            env = make_env(xml)
            goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal)
            env.get_reachable_objects()
            if env.is_robot_goal_reachable():
                n_already += 1
                continue
            s0 = env.get_full_state()
            budget = planner._first_budget(env, s0)
            onepush = False
            for (obj, g1) in budget[:a.first_cap]:
                env.set_full_state(s0)
                env.step(make_action(obj, g1)); n_sims += 1
                if env.is_robot_goal_reachable():
                    onepush = True
                    continue
                s1 = env.get_full_state()
                pool2 = planner._candidates(env, goal, xml, s1)
                if not pool2:
                    continue
                n_leaves += 1
                sc = leaf_scalars(pool2)
                rank_succ = None
                if a.scalars_only:
                    label = -1                       # graded post-hoc vs exhaustive F1', not the beam
                else:
                    for rank, (o2, g2, _p2) in enumerate(pool2[:a.topk]):
                        env.set_full_state(s1)
                        env.step(make_action(o2, g2)); n_sims += 1
                        if env.is_robot_goal_reachable():
                            rank_succ = rank
                            break
                    label = int(rank_succ is not None)
                    if label:
                        n_solv += 1
                        ranks.append(rank_succ)
                rec = {"xml": xml, "obj": obj, "edge1": int(g1.edge_idx), "depth1": int(g1.depth),
                       "label": label, "rank_succ": rank_succ, **sc}
                recs.append(rec)
                lf.write(json.dumps(rec) + "\n"); lf.flush()
            n_scenes += 1
            n_onepush += int(onepush)
        except Exception as e:
            print(f"[skip] {os.path.basename(xml)}: {e}", flush=True)
            continue
        if (xi + 1) % 5 == 0:
            print(f"[{a.start}+{xi+1}/{len(xmls)}] scenes={n_scenes} leaves={n_leaves} "
                  f"solv={n_solv} sims={n_sims} t={time.time()-t0:.0f}s", flush=True)
    lf.close()

    r = np.array(ranks)
    recall_at = lambda k: float((r < k).mean()) if len(r) else float("nan")
    good = {k: [x[k] for x in recs if x["label"] == 1] for k in SCALAR_KEYS}
    dead = {k: [x[k] for x in recs if x["label"] == 0] for k in SCALAR_KEYS}
    out = {
        "shard": [a.start, a.end], "n_scenes_used": n_scenes, "n_already_open": n_already,
        "n_scenes_first_push_alone_opened": n_onepush,
        "n_leaves": n_leaves, "n_solvable_leaves": n_solv, "n_sims": n_sims,
        "leaf_recall_at_s1": {f"@{k}": recall_at(k) for k in (1, 3, 5, 10, 20)},
        "median_rank_of_success": float(np.median(r)) if len(r) else None,
        "first_push_value_AUC_good_vs_dead": {k: auc(good[k], dead[k]) for k in SCALAR_KEYS},
        "params": {"first_cap": a.first_cap, "topk": a.topk, "collisions": "off",
                   "target_region_goal": True, "manifest": a.manifest},
        "wall_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== RESULT ===")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {a.out} and {a.leaf_out}", flush=True)


if __name__ == "__main__":
    main()
