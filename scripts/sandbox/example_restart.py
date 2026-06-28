#!/usr/bin/env python3
"""Find + print a concrete example of the planner RESTARTING: it plays a valid setup (sim1), a finishing push that
ACTUALLY opens the path sits right there, but the blend priority sends it to a fresh first-push instead (restart)."""
import sys, os, json, pickle, glob
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402
from scorer_beam import BeamPlanner, make_env, make_action, read_manifest, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import SCRATCH, MANIFESTS, DATASETS  # noqa: E402

PM = pickle.load(open(SCRATCH / "eval/exhaustive_pairmap_pure2.pkl", "rb"))["pairmap"]
CK = glob.glob(str(SCRATCH / "sage_outputs/scorer/qfull_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch008-val_loss0.6728.ckpt"))[0]
ed = lambda g: (int(getattr(g, "edge_idx", -1)), int(getattr(g, "depth", -1)))
v5 = lambda pool: (lambda qs: sum(qs[:5]) / min(5, len(qs)) if qs else 0.0)(sorted((q for _o, _g, q in pool), reverse=True))


def main():
    pl = BeamPlanner(ckpt=CK)
    xmls = read_manifest(str(MANIFESTS / "test_pure2_fromkey.txt"), None)
    key = json.load(open(str(DATASETS / "namo_testset_v1/labels/pure2push.json")))
    keyrp = {os.path.realpath(k): v for k, v in key.items()}
    found = 0
    for xml in xmls:
        if found >= 2:
            break
        recs = key.get(xml) or keyrp.get(os.path.realpath(xml))
        if not recs:
            continue
        try:
            env = make_env(xml); goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state(); pts0 = sample_goal_points(env)
        except Exception:
            continue
        if not pts0:
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
            if len(pool0) < 2:
                continue
            V0 = v5(pool0); top_a1 = ed(pool0[0][1]); q_a1 = pool0[0][2]
            q_a1_2nd = pool0[1][2]; a1_2nd = ed(pool0[1][1])
            if top_a1 not in setups:                 # want the planner's 1st pick to be a REAL setup
                continue
            env.set_full_state(s0); env.step(make_action(obj, pool0[0][1])); s1 = env.get_full_state()
            pool1 = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj)
            if not pool1:
                continue
            Vs1 = v5(pool1); top_child = ed(pool1[0][1]); q_child = pool1[0][2]
            gt_open = setups[top_a1]                 # GT finishing pushes for this setup
            # rank of the first GT opener in the model's H=1 ranking
            opener_rank = next((i + 1 for i, (_o, g, _q) in enumerate(pool1) if ed(g) in gt_open), None)
            pri_child = 0.5 * q_child + 0.5 * Vs1
            pri_restart = 0.5 * q_a1_2nd + 0.5 * V0
            restart = pri_restart > pri_child
            if restart and opener_rank is not None:   # the failure: restart while a real opener is available
                found += 1
                # does the TOP child open? (what diving would have given)
                env.set_full_state(s1); env.step(make_action(obj, pool1[0][1])); top_child_opens = goal_open_pts(env, pts0)
                print(f"\n===== EXAMPLE {found}: {os.path.basename(xml)}  object={obj} =====")
                print(f" sim1: planner pushes edge {top_a1[0]} depth {top_a1[1]}  (H=2 setup-score {q_a1:.2f}) — a REAL setup")
                print(f"       -> path NOT open yet (this is a 2-push scene)")
                print(f" at the new state, the finishing pushes:")
                print(f"       best 2nd push by the model: edge {top_child[0]} depth {top_child[1]}  (H=1 finish-score {q_child:.2f}); opens={top_child_opens}")
                print(f"       a GT push that DOES open the path sits at rank {opener_rank} of the model's list ({len(pool1)} candidates)")
                print(f" the planner's priority choice for the NEXT sim:")
                print(f"       FINISH this setup (dive):     0.5*{q_child:.2f} + 0.5*Vs1({Vs1:.2f}) = {pri_child:.3f}")
                print(f"       RESTART w/ a new 1st push:    0.5*{q_a1_2nd:.2f} + 0.5*V0({V0:.2f})  = {pri_restart:.3f}   (edge {a1_2nd[0]} d{a1_2nd[1]})")
                print(f"   ==> RESTART wins by {pri_restart-pri_child:+.3f}  -> planner abandons the setup to try a fresh push")
                print(f"       (that fresh push CANNOT open the path alone; the finishing push was right there.)")
                print(f"   ROOT: V0(setup-mode, {V0:.2f}) >= Vs1(finish-mode, {Vs1:.2f}) -> the two modes' scales make 'restart' look better.")
            if found >= 2:
                break
    if not found:
        print("no clear restart-with-available-opener example in the scanned scenes")


if __name__ == "__main__":
    main()
