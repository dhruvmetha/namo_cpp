#!/usr/bin/env python3
"""Phase-0 GREY-ZONE diagnostic (EXP-2026-07-06-rl-only-self-imitation): teacher-forced-finish sweep over the
model's SETUP ranks k=1,2,4,8. For each pure2push episode, rank the model's first pushes at s0 (H=2, object-
constrained), take the top-8. For each ranked setup, teacher-force the FINISH (oracle: does ANY 2nd push open at
the setup's resulting state?) -> the setup is 'finishable'. Also flag key-validity (GT valid_first_push).

Answers the grey-zone question: how deep in the model's SETUP ranking must you look to find a good (finishable)
setup? setup-hit@k = a finishable setup appears in the model's top-k.
  setup-hit@1 high    -> greedy setup already good (not our regime; it's 40.7).
  setup-hit@8 >> @1   -> a good setup IS surfaced in the model's top-8, just mis-ranked #1 => LEARNABLE RANKING
                         (exactly what RL self-imitation / a width-8 setup beam fixes).
  setup-hit@8 ~ @1    -> the model can't even surface a good setup in its top-8 => coverage/representation wall.
Reports both sim-grounded (oracle finish, capped at 25 tries/setup) and key-based (conservative LB) hit@k."""
import sys, os, json, argparse
from pathlib import Path

MAIN = "/common/home/dm1487/robotics_research/ktamp/namo"
SAGE = os.environ.get("SAGE_REPO", "/common/home/dm1487/robotics_research/ktamp/sage_learning")
for _p in (f"{MAIN}/build_python", f"{MAIN}/python", f"{MAIN}/scripts", f"{MAIN}/scripts/sandbox",
           f"{MAIN}/scripts/pipeline", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL          # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts          # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback                     # noqa: E402
from namo.paths import DATASETS, resolve                                             # noqa: E402

KRANK = 8          # sweep the model's top-KRANK setups
FIN_CAP = 25       # cap oracle-finish tries per setup


def finishable(pl, env, goal, xml, s0, obj, g_setup, gp):
    """Teacher-forced (oracle) finish: exec setup, does ANY reachable 2nd push open the region?"""
    env.set_full_state(s0); env.step(make_action(obj, g_setup))
    if goal_open_pts(env, gp):
        return 1
    s1 = env.get_full_state()
    fp = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj, score=True)
    for (_o, g2, _v) in fp[:FIN_CAP]:
        env.set_full_state(s1); env.step(make_action(obj, g2))
        if goal_open_pts(env, gp):
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", default=str(DATASETS / "namo_testset_v1/labels/pure2push.json"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaf-out", required=True)
    ap.add_argument("--h", type=int, default=2)
    a = ap.parse_args()
    pl = BeamPlanner(ckpt=a.ckpt)
    key = json.load(open(a.key))
    xmls = list(key); xmls = xmls[a.start:(a.end if a.end else len(xmls))]
    n = 0; skip = 0; leaf = []
    for xi, xml in enumerate(xmls):
        for rec in key[xml]:
            obj = rec["object_id"]; reg = rec.get("region")
            valid = {tuple(t) for t in rec.get("valid_first_push", [])} | {tuple(t) for t in rec.get("valid_1push", [])}
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
            top = pool0[:KRANK]
            key_at = 0; sim_at = 0
            for r, (o, g, _sc) in enumerate(top, 1):
                ed = (int(g.edge_idx), int(g.depth))
                if key_at == 0 and ed in valid:
                    key_at = r
                if sim_at == 0 and finishable(pl, env, goal, xml, s0, obj, g, gp):
                    sim_at = r
                if sim_at and key_at:
                    break                       # both found; deeper ranks irrelevant for hit@k
            leaf.append({"xml": xml, "object_id": obj, "region": reg,
                         "ntop": len(top), "setup_key_at": key_at, "setup_sim_at": sim_at})
        if xi % 25 == 0:
            print(f"  [{xi}/{len(xmls)}] n={n}", file=sys.stderr, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"ckpt": os.path.basename(a.ckpt), "n": n, "skip": skip}, open(a.out, "w"), indent=1)
    os.makedirs(os.path.dirname(a.leaf_out), exist_ok=True)
    with open(a.leaf_out, "w") as fh:
        for r in leaf:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({"n": n, "skip": skip}))


if __name__ == "__main__":
    main()
