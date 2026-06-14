#!/usr/bin/env python3
"""Trace WHY NoHorizon beats Horizon at reactive @2 despite Horizon's better setup->opener handoff.
Best-first @2: sim1=top first-push (won't open on pure-2), sim2=highest-priority among {2nd+ first-pushes} U
{top a1's children}. @2 solves IFF sim2 is a CHILD that opens. priority=0.5*Q + 0.5*V (V=mean_top5 of that
state's Q). HYP: Horizon's H2 first-push values (V0) > H1 child values (V_s1) -> blend keeps preferring fresh
first-pushes -> sim2 is a non-opening first-push -> @2 low. NoHz single head -> V0~=V_s1 -> dives into children."""
import sys, os, json, pickle, argparse, glob
REPO = "/cache/home/dm1487/projects/namo/namo_cpp"; SAGE = "/cache/home/dm1487/projects/namo/sage_learning"
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
from scorer_beam import BeamPlanner, make_env, make_action, read_manifest, FALLBACK_GOAL
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts
from namo.core.xml_goal_parser import extract_goal_with_fallback

CK = {"Hz-v2": glob.glob("/scratch/dm1487/sage_outputs/scorer/qfull_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch008-val_loss0.6728.ckpt")[0],
      "NoHz-v2": glob.glob("/scratch/dm1487/sage_outputs/scorer/qfull_nohz_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch007-val_loss0.7041.ckpt")[0]}


def v_top5(pool):
    qs = sorted((q for (_o, _g, q) in pool), reverse=True)
    return sum(qs[:5]) / min(5, len(qs)) if qs else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="/scratch/dm1487/eval/trace_reactive.json"); a = ap.parse_args()
    pls = {nm: BeamPlanner(ckpt=c) for nm, c in CK.items()}
    xmls = read_manifest("/scratch/dm1487/manifests/test_pure2_fromkey.txt", None)
    key = json.load(open("/scratch/dm1487/datasets/namo_testset_v1/labels/pure2push.json"))
    keyrp = {os.path.realpath(k): v for k, v in key.items()}
    agg = {nm: {"V0": [], "Vs1": [], "sim2_child": [], "sim2_child_opens": [], "at2": []} for nm in pls}
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
            pts0 = sample_goal_points(env)
        except Exception:
            continue
        if not pts0:
            continue
        for rec in recs:
            obj = rec["object_id"]; done += 1
            for nm, pl in pls.items():
                h0 = 2                                  # both query at remaining budget 2 for the first push
                pool0 = rank_first_pushes_h2(pl, env, goal, xml, s0, h0, restrict_obj=obj)
                if len(pool0) < 2:
                    continue
                V0 = v_top5(pool0)
                q_a1_2nd = pool0[1][2]                  # 2nd-best first-push Q
                Gtop = pool0[0][1]
                env.set_full_state(s0); env.step(make_action(obj, Gtop)); s1 = env.get_full_state()
                pool1 = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj)
                if not pool1:
                    continue
                Vs1 = v_top5(pool1); q_a2_top = pool1[0][2]; Ga2 = pool1[0][1]
                # blend priorities (0.5 Q + 0.5 V), same as solve_scene
                pri_child = 0.5 * q_a2_top + 0.5 * Vs1
                pri_firstpush = 0.5 * q_a1_2nd + 0.5 * V0
                sim2_child = pri_child >= pri_firstpush
                agg[nm]["V0"].append(V0); agg[nm]["Vs1"].append(Vs1)
                agg[nm]["sim2_child"].append(int(sim2_child))
                opens = False
                if sim2_child:                          # sim2 = the top child; does it open?
                    env.set_full_state(s1); env.step(make_action(obj, Ga2)); opens = goal_open_pts(env, pts0)
                    agg[nm]["sim2_child_opens"].append(int(opens))
                agg[nm]["at2"].append(int(sim2_child and opens))
        if done % 25 == 0:
            print(f"  [{done}/{a.n}]", file=sys.stderr, flush=True)
    out = {}
    for nm, d in agg.items():
        f = lambda x: round(float(np.mean(x)), 3) if x else None
        out[nm] = {"n": len(d["V0"]), "V0_mean(H2 firstpush)": f(d["V0"]), "Vs1_mean(H1 child)": f(d["Vs1"]),
                   "V0_minus_Vs1": round((np.mean(d["V0"]) - np.mean(d["Vs1"])), 3) if d["V0"] else None,
                   "sim2_is_child_frac": f(d["sim2_child"]),
                   "sim2_child_opens_frac": f(d["sim2_child_opens"]),
                   "at2_solve(child&opens)": f(d["at2"])}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
