#!/usr/bin/env python3
"""[USER idea] Query H=2 (not H=1) on the post-push state s1 for the FINISH. The aug made H=2 value a 1-push win at
1.0, and at s1 the finish IS a 1-push win — so H=2 might rank it sharper than the mushy H=1 (which learned s1 finishes
from the skewed postpush set). FREE if it works (just change the budget token, no retrain).
For N scenes: model's top setup -> s1, exhaustive-a2 label, score finish RAW with h=1 AND h=2; compare opener-vs-non
separation AND top-1-finish-opens (the reactive metric: is the model's #1 reachable finish a real opener?)."""
import os, sys, glob, argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", f"{REPO}/scripts/pipeline", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np, json  # noqa: E402
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points  # noqa: E402
from exit_collect import exhaustive_a2  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import DATASETS, SCRATCH  # noqa: E402
KEY = str(DATASETS / "v4_hq_h2/labels_exhaustive_pure2push.json")
CK = glob.glob(f"{SCRATCH}/sage_outputs/scorer/qfull_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch008-val_loss0.6728.ckpt")[0]


def top1_opens(raw, f_grid, r_mask):
    """Among REACHABLE finishes, is the highest-scored one a real opener?"""
    tried = r_mask >= 0.5
    if not tried.any():
        return None
    masked = np.where(tried, raw, -1e9)
    e, dp = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return bool(f_grid[e, dp] >= 0.999)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--start", type=int, default=0, help="xml-index shard start (full test set = shard across CPUs)")
    ap.add_argument("--end", type=int, default=0, help="xml-index shard end (0 = to the end)")
    ap.add_argument("--ckpt", default=CK, help="model to score the finish with (default Hz-v2)")
    ap.add_argument("--key", default=KEY, help="scenes to step into s1 + live-label a2 (TEST key = the gate on novel s1)")
    ap.add_argument("--out", default=str(SCRATCH / "eval/check_h2_finish.json"))
    ap.add_argument("--fixed-setup", action="store_true",
                    help="step to a GT-valid (model-independent) setup's s1 so v2/v3 are scored on the SAME finish "
                         "problem — isolates finish quality from setup changes (the clean gate)")
    a = ap.parse_args()
    pl = BeamPlanner(ckpt=a.ckpt)
    d = json.load(open(a.key))
    p1, n1, p2, n2 = [], [], [], []
    t1_hit = t2_hit = t_tot = 0; n = 0
    xmls = list(d); xmls = xmls[a.start:(a.end if a.end else len(xmls))]
    for xml in xmls:
        if n >= a.n:
            break
        for rec in d[xml]:
            if n >= a.n:
                break
            obj = rec["object_id"]
            try:
                env = make_env(xml); goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
                env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state()
                gp = sample_goal_points(env)
            except Exception:
                continue
            if not gp:
                continue
            pool = rank_first_pushes_h2(pl, env, goal, xml, s0, 2, restrict_obj=obj)
            if not pool:
                continue
            if a.fixed_setup:   # GT-valid, model-INDEPENDENT setup -> both models step to the SAME s1 (isolates finish)
                gt = {tuple(t) for t in rec.get("valid_first_push", [])}
                cand = sorted([g for (_o, g, _q) in pool if (int(g.edge_idx), int(g.depth)) in gt],
                              key=lambda g: (int(g.edge_idx), int(g.depth)))
                if not cand:
                    continue
                setup_goal = cand[0]
            else:
                setup_goal = pool[0][1]   # model's own top setup (deployment-style)
            env.set_full_state(s0); env.step(make_action(obj, setup_goal)); s1 = env.get_full_state()
            f_grid, r_mask, n_open, n_tried = exhaustive_a2(pl, env, goal, xml, s1, gp, obj)
            if n_open == 0 or n_tried == 0:
                continue
            env.set_full_state(s1); raw1 = pl.scorer.score_state(env, obj, goal, xml, h=1, raw=True)
            env.set_full_state(s1); raw2 = pl.scorer.score_state(env, obj, goal, xml, h=2, raw=True)
            tried = r_mask >= 0.5; op = (f_grid >= 0.999) & tried; nn = (~(f_grid >= 0.999)) & tried
            p1 += raw1[op].tolist(); n1 += raw1[nn].tolist(); p2 += raw2[op].tolist(); n2 += raw2[nn].tolist()
            h1 = top1_opens(raw1, f_grid, r_mask); h2 = top1_opens(raw2, f_grid, r_mask)
            if h1 is not None:
                t_tot += 1; t1_hit += int(h1); t2_hit += int(h2)
            n += 1
            if n % 20 == 0:
                print(f"  [{n}] sepH1={np.mean(p1)-np.mean(n1):.3f} sepH2={np.mean(p2)-np.mean(n2):.3f} "
                      f"top1H1={t1_hit/max(t_tot,1):.2f} top1H2={t2_hit/max(t_tot,1):.2f}", file=sys.stderr, flush=True)
    out = {"n_states": n,
           # raw sums+counts so sharded runs aggregate EXACTLY (sep = (Σp/np) - (Σn/nn); top1 = Σhit/Σtot)
           "agg": {"sum_p1": float(np.sum(p1)), "n_p1": len(p1), "sum_n1": float(np.sum(n1)), "n_n1": len(n1),
                   "sum_p2": float(np.sum(p2)), "n_p2": len(p2), "sum_n2": float(np.sum(n2)), "n_n2": len(n2),
                   "t1_hit": t1_hit, "t2_hit": t2_hit, "t_tot": t_tot},
           "FINISH scored at H=1 (current)": {"openers": round(float(np.mean(p1)), 3), "non": round(float(np.mean(n1)), 3),
                                              "separation": round(float(np.mean(p1) - np.mean(n1)), 3),
                                              "top1_finish_opens": round(t1_hit / max(t_tot, 1), 3)},
           "FINISH scored at H=2 (the idea)": {"openers": round(float(np.mean(p2)), 3), "non": round(float(np.mean(n2)), 3),
                                              "separation": round(float(np.mean(p2) - np.mean(n2)), 3),
                                              "top1_finish_opens": round(t2_hit / max(t_tot, 1), 3)},
           "n_top1": t_tot,
           "verdict": "H=2 sep/top1 >> H=1 -> the idea WORKS (free finish fix); <= -> H=2 has the same/worse gap"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
