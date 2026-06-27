#!/usr/bin/env python3
"""DEEP Q-value audit — answers: (1) is H=1 Q calibrated / does it translate to a real opener? (2) does H=2 pick
setups whose downstream H=1 value is high? (3) is the search's needle reachable under the model's ranking?
(4) WHY does NoHorizon beat Horizon reactively? Traces top-H=2 a1 -> sim -> top-H=1 a2 vs the exhaustive (a1,a2)
ground-truth pairmap, for BOTH models on the SAME scenes.

  python scripts/sandbox/analyze_qvalue.py --n 120 --out $NAMO_SCRATCH/eval/qvalue_audit.json
"""
import sys, os, json, pickle, argparse, math
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from scorer_beam import BeamPlanner, make_env, make_action, read_manifest, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import SCRATCH, MANIFESTS, DATASETS  # noqa: E402

PM = pickle.load(open(SCRATCH / "eval/exhaustive_pairmap_pure2.pkl", "rb"))["pairmap"]
CKPTS = {
    "Hz-v2": next(iter(__import__('glob').glob(str(SCRATCH / "sage_outputs/scorer/qfull_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch008-val_loss0.6728.ckpt")))),
    "NoHz-v2": next(iter(__import__('glob').glob(str(SCRATCH / "sage_outputs/scorer/qfull_nohz_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch007-val_loss0.7041.ckpt")))),
}


def ed(g):
    return (int(getattr(g, "edge_idx", -1)), int(getattr(g, "depth", -1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=str(SCRATCH / "eval/qvalue_audit.json"))
    a = ap.parse_args()
    planners = {nm: BeamPlanner(ckpt=ck) for nm, ck in CKPTS.items()}
    xmls = read_manifest(str(MANIFESTS / "test_pure2_fromkey.txt"), None)
    key = json.load(open(str(DATASETS / "namo_testset_v1/labels/pure2push.json")))
    keyrp = {os.path.realpath(k): v for k, v in key.items()}
    agg = {nm: {"calib": [], "a1_setup_top1": [], "a1_setup_rank": [], "a2_open_top1": [],
                "a2_open_rank": [], "react": [], "h2_auc_pos": [], "h2_auc_neg": [],
                "h2_vs_maxh1": []} for nm in planners}
    done = 0
    for xml in xmls:
        if done >= a.n:
            break
        recs = key.get(xml) or keyrp.get(os.path.realpath(xml))
        if not recs:
            continue
        try:
            env = make_env(xml); goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state()
        except Exception:
            continue
        for rec in recs:
            obj = rec["object_id"]; pm = PM.get((os.path.realpath(xml), obj))
            if not pm:
                continue
            setups = {a1: {a2 for a2, ok in m.items() if ok} for a1, m in pm.items()}
            setups = {a1: o for a1, o in setups.items() if o}            # a1 -> GT opening a2 set
            if not setups:
                continue
            done += 1
            for nm, pl in planners.items():
                # --- H=2 first-push ranking at s0 ---
                pool2 = rank_first_pushes_h2(pl, env, goal, xml, s0, 2, restrict_obj=obj)  # [(obj,G,q)] desc
                a1rank = [(ed(g), q) for (_o, g, q) in pool2]
                if not a1rank:
                    continue
                # H=2 discriminates setups? collect Q for setup vs non-setup a1
                for (cell, q) in a1rank:
                    (agg[nm]["h2_auc_pos"] if cell in setups else agg[nm]["h2_auc_neg"]).append(q)
                top_a1 = a1rank[0][0]
                agg[nm]["a1_setup_top1"].append(int(top_a1 in setups))
                agg[nm]["a1_setup_rank"].append(next((i + 1 for i, (c, _) in enumerate(a1rank) if c in setups), 99))
                # --- take the model's top-1 a1, sim it, score second push at H=1 ---
                Gtop = pool2[0][1]
                env.set_full_state(s0); env.step(make_action(obj, Gtop)); s1 = env.get_full_state()
                pool1 = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj)
                a2rank = [(ed(g), q) for (_o, g, q) in pool1]
                gt_open = setups.get(top_a1, set())                       # GT openers IF top_a1 is a real setup
                # H=2 of top_a1 vs max H=1 over a2 in s1 (handoff coherence)
                if a2rank:
                    agg[nm]["h2_vs_maxh1"].append((a1rank[0][1], max(q for _, q in a2rank)))
                if gt_open and a2rank:                                    # only meaningful if top_a1 is a setup
                    agg[nm]["a2_open_top1"].append(int(a2rank[0][0] in gt_open))
                    agg[nm]["a2_open_rank"].append(next((i + 1 for i, (c, _) in enumerate(a2rank) if c in gt_open), 99))
                    for (cell, q) in a2rank:
                        agg[nm]["calib"].append((q, int(cell in gt_open)))
                # reactive 2-push success (model's greedy top a1 -> top a2), GT-graded
                agg[nm]["react"].append(int(top_a1 in setups and a2rank and a2rank[0][0] in setups.get(top_a1, set())))
        if done % 20 == 0:
            print(f"  [{done}/{a.n}]", file=sys.stderr, flush=True)

    out = {}
    for nm, d in agg.items():
        def m(x): return round(float(np.mean(x)), 3) if x else None
        # H=2 AUC (does H=2 Q rank setups above non-setups?)
        pos, neg = d["h2_auc_pos"], d["h2_auc_neg"]
        auc = None
        if pos and neg:
            auc = np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in pos[:300] for n in neg[:300]])
        # calibration bins
        calib = d["calib"]; bins = {}
        if calib:
            qs = np.array([c[0] for c in calib]); ys = np.array([c[1] for c in calib])
            for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
                msk = (qs >= lo) & (qs < hi)
                if msk.sum():
                    bins[f"{lo:.1f}-{hi:.1f}"] = [round(float(ys[msk].mean()), 2), int(msk.sum())]
        h2m = d["h2_vs_maxh1"]
        corr = None
        if len(h2m) > 5:
            xa = np.array([x for x, _ in h2m]); ya = np.array([y for _, y in h2m])
            if xa.std() > 0 and ya.std() > 0:
                corr = round(float(np.corrcoef(xa, ya)[0, 1]), 3)
        out[nm] = {
            "n": len(d["a1_setup_top1"]),
            "H2_picks_setup@1": m(d["a1_setup_top1"]),
            "H2_first_setup_rank_med": float(np.median(d["a1_setup_rank"])) if d["a1_setup_rank"] else None,
            "H2_setup_vs_nonsetup_AUC": round(float(auc), 3) if auc is not None else None,
            "H1_top1_a2_opens": m(d["a2_open_top1"]),
            "H1_first_opener_rank_med": float(np.median(d["a2_open_rank"])) if d["a2_open_rank"] else None,
            "H1_calibration(Qbin->openrate)": bins,
            "H2a1_vs_maxH1s1_corr": corr,
            "reactive_2push_success": m(d["react"]),
        }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
