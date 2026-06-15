#!/usr/bin/env python3
"""ORACLE HEADROOM — what does fixing the setup vs the finish actually buy reactively (@2)?
For each pure-2 scene, decompose reactive @2 into 4 conditions using the exhaustive (a1,a2)->opens GT:
  model/model      : model's top setup, then model's top finish from it           (= today)
  oracle-finish    : model's top setup; if it's a REAL setup, a perfect finish opens it
  oracle-setup     : guaranteed a REAL setup (the model's best real one), then model's top finish
  oracle/oracle    : a real setup + a perfect finish                               (= ~100%)
Tells us: fixing finish alone is capped by setup-top-1; fixing setup alone capped by finish-top-1."""
import sys, json, pickle, glob, os
REPO = "/cache/home/dm1487/projects/namo/namo_cpp"; SAGE = "/cache/home/dm1487/projects/namo/sage_learning"
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from scorer_beam import BeamPlanner, make_env, make_action, read_manifest, FALLBACK_GOAL
from eval_m3 import rank_first_pushes_h2
from namo.core.xml_goal_parser import extract_goal_with_fallback
PM = pickle.load(open("/scratch/dm1487/eval/exhaustive_pairmap_pure2.pkl", "rb"))["pairmap"]
CK = glob.glob("/scratch/dm1487/sage_outputs/scorer/qfull_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch008-val_loss0.6728.ckpt")[0]
ed = lambda g: (int(getattr(g, "edge_idx", -1)), int(getattr(g, "depth", -1)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 130
    pl = BeamPlanner(ckpt=CK)
    xmls = read_manifest("/scratch/dm1487/manifests/test_pure2_fromkey.txt", None)
    key = json.load(open("/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push.json"))
    keyrp = {os.path.realpath(k): v for k, v in key.items()}
    mm = ofin = ose = oo = tot = 0
    for xml in xmls:
        if tot >= n:
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
            tot += 1
            ranked = [(ed(g), g) for (_o, g, _q) in pool0]
            top_a1, top_g = ranked[0]
            best_real = next(((c, g) for (c, g) in ranked if c in setups), None)   # model's top REAL setup
            oo += 1                                                                 # oracle/oracle always solves (pure2)
            ofin += int(top_a1 in setups)                                           # oracle finish: solves iff top setup real
            # model finish from the MODEL's top setup (model/model)
            env.set_full_state(s0); env.step(make_action(obj, top_g)); s1 = env.get_full_state()
            p1 = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj)
            if top_a1 in setups and p1 and ed(p1[0][1]) in setups[top_a1]:
                mm += 1
            # model finish from the ORACLE's real setup (oracle-setup)
            if best_real is not None:
                env.set_full_state(s0); env.step(make_action(obj, best_real[1])); s1o = env.get_full_state()
                p1o = rank_first_pushes_h2(pl, env, goal, xml, s1o, 1, restrict_obj=obj)
                if p1o and ed(p1o[0][1]) in setups[best_real[0]]:
                    ose += 1
        if tot % 25 == 0:
            print(f"  [{tot}/{n}]", file=sys.stderr, flush=True)
    out = {"n": tot,
           "reactive@2  model/model": round(100 * mm / tot, 1),
           "reactive@2  oracle-FINISH (model setup)": round(100 * ofin / tot, 1),
           "reactive@2  oracle-SETUP (model finish)": round(100 * ose / tot, 1),
           "reactive@2  oracle/oracle": round(100 * oo / tot, 1),
           "note": "fix-finish capped by setup-top1; fix-setup capped by finish-top1; need BOTH for ~100%"}
    json.dump(out, open("/scratch/dm1487/eval/oracle_headroom.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
