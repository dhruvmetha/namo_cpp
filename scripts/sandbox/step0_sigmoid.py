#!/usr/bin/env python3
"""STEP 0: is the 'mushy finish' a SIGMOID double-squash (free fix) or a real training problem?
Score the H=1 finishing pushes on post-setup states two ways — RAW E[bin] vs the deployed sigmoid(E[bin]) — and
compare separation between GT openers and non-openers. If RAW is sharp/separated (openers~0.9, non~0.1) and the
sigmoid mushes it to [0.5,0.73], the fix is one line. If RAW is ALSO mushy, it's a training problem."""
import sys, os, json, pickle, glob
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from scorer_beam import BeamPlanner, make_env, make_action, read_manifest, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import SCRATCH, DATASETS, MANIFESTS  # noqa: E402

PM = pickle.load(open(str(SCRATCH / "eval/exhaustive_pairmap_pure2.pkl"), "rb"))["pairmap"]
CK = glob.glob(f"{SCRATCH}/sage_outputs/scorer/qfull_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch008-val_loss0.6728.ckpt")[0]
ed = lambda g: (int(getattr(g, "edge_idx", -1)), int(getattr(g, "depth", -1)))


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    pl = BeamPlanner(ckpt=CK); sc = pl.scorer
    xmls = read_manifest(str(MANIFESTS / "test_pure2_fromkey.txt"), None)
    key = json.load(open(str(DATASETS / "namo_testset_v1/labels/pure2push.json")))
    keyrp = {os.path.realpath(k): v for k, v in key.items()}
    raw_op, raw_non, sig_op, sig_non = [], [], [], []
    done = 0
    for xml in xmls:
        if done >= n_target:
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
            setups = {a1: o for a1, o in setups.items() if o}
            if not setups:
                continue
            pool0 = rank_first_pushes_h2(pl, env, goal, xml, s0, 2, restrict_obj=obj)
            if not pool0:
                continue
            top_a1 = ed(pool0[0][1])
            if top_a1 not in setups:
                continue                                      # only score finishes on a REAL setup
            done += 1
            env.set_full_state(s0); env.step(make_action(obj, pool0[0][1])); s1 = env.get_full_state()
            # reachable 2nd-push cells (sigmoid pool) + RAW grid at s1
            pool1 = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj)
            env.set_full_state(s1)
            ctx, _ = sc.render_ctx(env, obj, goal, xml); cpx = sc.contact_px_live(env, obj)
            raw = sc.score_ctx(ctx, cpx, h=1, raw=True)        # (60,5) E[bin]
            sig = sc.score_ctx(ctx, cpx, h=1, raw=False)       # (60,5) sigmoid(E[bin]) = deployed
            gt = setups[top_a1]
            for (_o, g, _q) in pool1:                           # over reachable 2nd pushes only
                e, d = ed(g)
                if not (0 <= e < 60 and 0 <= d < 5):
                    continue
                (raw_op if (e, d) in gt else raw_non).append(float(raw[e, d]))
                (sig_op if (e, d) in gt else sig_non).append(float(sig[e, d]))
        if done % 30 == 0:
            print(f"  [{done}/{n_target}]", file=sys.stderr, flush=True)

    def stats(x):
        a = np.array(x); return dict(mean=round(a.mean(), 3), p10=round(np.percentile(a, 10), 3),
                                     p90=round(np.percentile(a, 90), 3), n=len(a))
    out = {"n_setups": done,
           "RAW E[bin]":   {"openers": stats(raw_op), "non_openers": stats(raw_non),
                            "separation": round(np.mean(raw_op) - np.mean(raw_non), 3)},
           "SIGMOID (deployed)": {"openers": stats(sig_op), "non_openers": stats(sig_non),
                                  "separation": round(np.mean(sig_op) - np.mean(sig_non), 3)}}
    json.dump(out, open(str(SCRATCH / "eval/step0_sigmoid.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
