#!/usr/bin/env python3
"""RETIRED: the GT-valid-setup arms below (ARM (i), ARM (ii), and the wrong_setup taxonomy leg) are scored against
`pure2push.json`'s `valid_first_push` list. That list is badly incomplete — an exhaustive sweep over `testset_gt.h5`
finds a median of 11 good first pushes per episode where the manifest records only 4, because the manifest's
finish search was budget-limited. These numbers are therefore a lower bound that understates the model and
distorts the easy/medium/hard tier comparison — consistent with, just stronger than, the "conservative lower
bound" this module already calls the key-based numbers below.
Canonical replacement: `scripts/eval_auc.py` over `testset_gt.h5` (see docs/experiments/auc_metrics_reconciliation.md).
The BASELINE arm (fully-learned greedy@2, sim-grounded via goal_open_pts) is unaffected by this issue.

Phase-0 GATE (EXP-2026-07-06-rl-only-self-imitation): oracle reactive decomposition on the pure-2push set,
GREEDY protocol (forced-dive reactive, NOT best-first search). Car robot, region criterion, object-constrained,
same protocol as eval_reactive_argmax (H=2 setup / H=1 finish, restrict_obj=labeled object). One pass per episode
computes all three arms + the fully-learned baseline, so open@2 reproduces the reactive-MPC 40.7 anchor.

Per episode (pure2push key -> valid_first_push = GT-correct setups):
  * BASELINE (fully-learned greedy@2): g1=argmax setup@H2 -> exec -> g2=argmax finish@H1 -> exec -> open?  (== 40.7 anchor)
  * ARM (ii) recoverable: is the model's greedy g1 a GT-valid setup? (g1 in valid_first_push u valid_1push)
  * ARM (i) oracle-setup + learned finish: for each GT-valid setup v (ordered by model score, cap MAXSETUPS),
        exec v -> model greedy finish -> open? Report any / modelpref(top-scored valid setup) / mean over setups.
  * ARM (iii) miss taxonomy (only when baseline FAILS):
        wrong_setup         : g1 not a GT-valid setup.
        failed_finish       : g1 IS a GT-valid setup, but NO opening 2nd push exists at the model-executed s1
                              (the valid setup didn't 'take' -> control drift / near-threshold).
        aliasing_or_control : g1 valid AND an opening finish EXISTS at s1, but the model's greedy g2 missed it
                              (scorer mis-ranks the finish / crop aliasing, or control failed on the ranked finish).

Leaf jsonl (one line/episode) -> aggregate with agg_phase0.py, split by pure2push_divisions.json `division`."""
import sys, os, json, argparse, random
from pathlib import Path

MAIN = "/common/home/dm1487/robotics_research/ktamp/namo"      # shared-FS main checkout (compiled bindings live here)
SAGE = os.environ.get("SAGE_REPO", "/common/home/dm1487/robotics_research/ktamp/sage_learning")
for _p in (f"{MAIN}/build_python", f"{MAIN}/python", f"{MAIN}/scripts", f"{MAIN}/scripts/sandbox",
           f"{MAIN}/scripts/pipeline", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL          # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts          # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback                     # noqa: E402
from namo.paths import DATASETS, resolve                                             # noqa: E402
from namo import eval_sets                                                            # noqa: E402


def greedy_finish(pl, env, goal, xml, s1, obj, gp):
    """Model's greedy 2nd push at live state s1 (H=1, object-constrained). Returns (opened:0/1, fpool)."""
    fpool = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj, score=True)
    if not fpool:
        return 0, fpool
    gf = fpool[0][1]
    env.set_full_state(s1); env.step(make_action(obj, gf))
    return int(goal_open_pts(env, gp)), fpool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", default=str(eval_sets.PURE2PUSH))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = to end (xml-index shard)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaf-out", required=True)
    ap.add_argument("--h", type=int, default=2, help="setup query budget (2=foresight); finish always H=1")
    ap.add_argument("--max-setups", type=int, default=12, help="arm(i): cap GT-valid setups tried per episode (score-ordered)")
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()
    MAXS = a.max_setups
    pl = BeamPlanner(ckpt=a.ckpt)
    key = json.load(open(a.key))
    xmls = list(key); xmls = xmls[a.start:(a.end if a.end else len(xmls))]
    n = 0; skip = 0; leaf = []
    g1_in_tried_hits = 0; capped = 0
    for xi, xml in enumerate(xmls):
        for rec in key[xml]:
            obj = rec["object_id"]; reg = rec.get("region")
            valid = {tuple(t) for t in rec.get("valid_first_push", [])} | {tuple(t) for t in rec.get("valid_1push", [])}
            tried = {tuple(t) for t in rec.get("tried_first_push", [])}
            try:
                xmlp = str(resolve(xml)); env = make_env(xmlp); goal = extract_goal_with_fallback(xmlp, FALLBACK_GOAL)
                env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state()
                gp = sample_goal_points(env)
            except Exception:
                skip += 1; continue
            if not gp or goal_open_pts(env, gp):
                skip += 1; continue
            pool0 = rank_first_pushes_h2(pl, env, goal, xml, s0, a.h, restrict_obj=obj, score=True)
            if not pool0:
                skip += 1; continue
            n += 1
            cand = {}                                                       # (e,d) -> (Goal, model_score)
            for (_o, g, v) in pool0:
                cand[(int(g.edge_idx), int(g.depth))] = (g, float(v))
            g1 = pool0[0][1]; g1_ed = (int(g1.edge_idx), int(g1.depth))
            g1_in_valid = g1_ed in valid                                    # ARM (ii): recoverable
            g1_in_tried_hits += int(g1_ed in tried)

            # ---------- BASELINE fully-learned greedy@2 + ARM (iii) miss taxonomy (SIM-GROUNDED) ----------
            # valid_first_push is a conservative LOWER bound (GT's 2nd-push verify budget was limited), so the miss
            # taxonomy is grounded by SIMULATION at the model-executed s1, not by the key: does ANY 2nd push open there?
            #   finish_exists      -> aliasing_or_control  (landed openable, model's greedy finish missed it)
            #   no finish + g1 valid setup -> failed_finish (GT-good setup, but executed into a dead state: drift/near-thresh)
            #   no finish + g1 not valid   -> wrong_setup   (GT and sim agree: dead setup)
            env.set_full_state(s0); env.step(make_action(obj, g1))
            miss = ""; finish_exists = None
            if goal_open_pts(env, gp):                                      # opened on setup alone (rare for pure2push)
                base_open = 1
            else:
                s1 = env.get_full_state()
                base_open, fpool = greedy_finish(pl, env, goal, xml, s1, obj, gp)
                if not base_open:
                    finish_exists = False
                    for (_o2, g2, _v2) in fpool:                            # does ANY 2nd push open at model-executed s1?
                        env.set_full_state(s1); env.step(make_action(obj, g2))
                        if goal_open_pts(env, gp):
                            finish_exists = True; break
                    if finish_exists:
                        miss = "aliasing_or_control"
                    elif g1_in_valid:
                        miss = "failed_finish"
                    else:
                        miss = "wrong_setup"

            # ---------- ARM (i): oracle setup + learned greedy finish ----------
            vs = sorted(((cand[ed][1], ed) for ed in valid if ed in cand), key=lambda x: -x[0])  # valid setups model can execute, by score
            armi_avail = len(vs)
            if armi_avail > MAXS:
                capped += 1
            vs = vs[:MAXS]
            armi_tried = 0; armi_open = 0; armi_any = 0; armi_modelpref = None
            for i, (_sc, ed) in enumerate(vs):
                g_setup = cand[ed][0]
                env.set_full_state(s0); env.step(make_action(obj, g_setup))
                if goal_open_pts(env, gp):                                  # oracle setup alone opened -> finishable trivially
                    opened = 1
                else:
                    s1v = env.get_full_state()
                    opened, _ = greedy_finish(pl, env, goal, xml, s1v, obj, gp)
                armi_tried += 1; armi_open += opened; armi_any = max(armi_any, opened)
                if i == 0:
                    armi_modelpref = opened

            leaf.append({
                "xml": xml, "object_id": obj, "region": reg,
                "n_valid": len(valid), "g1_edge": g1_ed[0], "g1_depth": g1_ed[1],
                "g1_in_tried": int(g1_ed in tried),
                "base_open": base_open, "miss": miss,
                "finish_exists": (int(finish_exists) if finish_exists is not None else None),
                "armii_recoverable": int(g1_in_valid),
                "armi_avail": armi_avail, "armi_tried": armi_tried, "armi_open": armi_open,
                "armi_any": int(armi_any), "armi_modelpref": (int(armi_modelpref) if armi_modelpref is not None else None),
            })
        if xi % 25 == 0:
            print(f"  [{xi}/{len(xmls)}] n={n} base_open={sum(r['base_open'] for r in leaf)} "
                  f"armi_any={sum(r['armi_any'] for r in leaf)}", file=sys.stderr, flush=True)

    summ = {"ckpt": os.path.basename(a.ckpt), "n": n, "skip": skip,
            "g1_in_tried": g1_in_tried_hits, "armi_capped_episodes": capped,
            "base_open": sum(r["base_open"] for r in leaf),
            "armii_recoverable": sum(r["armii_recoverable"] for r in leaf),
            "armi_any": sum(r["armi_any"] for r in leaf),
            "armi_modelpref": sum((r["armi_modelpref"] or 0) for r in leaf)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summ, open(a.out, "w"), indent=1)
    os.makedirs(os.path.dirname(a.leaf_out), exist_ok=True)
    with open(a.leaf_out, "w") as fh:
        for r in leaf:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
