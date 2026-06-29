#!/usr/bin/env python3
"""PURE REACTIVE policy = argmax setup -> step -> argmax finish -> step -> opened?  (region criterion, object-constrained).

Zero search, exactly 2 sims/episode. Unlike best-first solve@2, this FORCES the dive (sim 2 is always the model's top
finish at the setup's s1), so it removes the 'won't-dive' confound and measures the model's true reactive ceiling:
  - setup = argmax Q(s0, ., H=2)  (the model's #1 first push, object-constrained)
  - finish = argmax Q(s1, ., H=1)  (the model's #1 second push at the resulting state)
  - graded by goal_open_pts (>=20% of s0-sampled goal-region points reachable) = the canonical region criterion.
Reports open@1 (setup alone opens, ~0 on pure-2) and open@2 (the argmax-argmax reactive number)."""
import sys, os, json, argparse, random
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", f"{REPO}/scripts/pipeline", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import DATASETS, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", default=str(DATASETS / "namo_testset_v1/labels/pure2push.json"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = to end (xml-index shard)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaf-out", default="", help="optional per-episode jsonl: {xml,object_id,region,open2}")
    ap.add_argument("--h", type=int, default=2, help="query budget for the FIRST push (2=foresight; 1=told-you-have-1-push)")
    ap.add_argument("--prior", default="q", choices=["q", "uniform"], help="q=argmax(model); uniform=RANDOM pick, no model")
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pl = BeamPlanner(ckpt=a.ckpt)
    key = json.load(open(a.key))
    xmls = list(key); xmls = xmls[a.start:(a.end if a.end else len(xmls))]
    n = 0; open1 = 0; open2 = 0; skip = 0; leaf = []
    for xi, xml in enumerate(xmls):
        for rec in key[xml]:
            obj = rec["object_id"]; reg = rec.get("region")
            try:
                xmlp = str(resolve(xml)); env = make_env(xmlp); goal = extract_goal_with_fallback(xmlp, FALLBACK_GOAL)
                env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state()
                gp = sample_goal_points(env)
            except Exception:
                skip += 1; continue
            if not gp or goal_open_pts(env, gp):
                skip += 1; continue
            pool0 = rank_first_pushes_h2(pl, env, goal, xml, s0, a.h, restrict_obj=obj, score=(a.prior == "q"))  # first push @ query budget a.h
            if not pool0:
                skip += 1; continue
            n += 1
            _o, g1, _q = pool0[0] if a.prior == "q" else rng.choice(pool0)              # ARGMAX or RANDOM setup
            env.set_full_state(s0); env.step(make_action(obj, g1))
            if goal_open_pts(env, gp):
                open1 += 1; open2 += 1
                leaf.append({"xml": xml, "object_id": obj, "region": reg, "open2": 1}); continue  # opened in 1
            s1 = env.get_full_state()
            pool1 = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj, score=(a.prior == "q"))   # finishes @H=1
            if not pool1:
                leaf.append({"xml": xml, "object_id": obj, "region": reg, "open2": 0}); continue
            _o2, g2, _q2 = pool1[0] if a.prior == "q" else rng.choice(pool1)             # ARGMAX or RANDOM finish
            env.set_full_state(s1); env.step(make_action(obj, g2))
            o2 = 1 if goal_open_pts(env, gp) else 0
            open2 += o2
            leaf.append({"xml": xml, "object_id": obj, "region": reg, "open2": o2})
        if xi % 25 == 0:
            print(f"  [{xi}/{len(xmls)}] n={n} open@2={open2}", file=sys.stderr, flush=True)
    out = {"ckpt": os.path.basename(a.ckpt), "n": n, "skip": skip,
           "open1": open1, "open2": open2,
           "reactive_argmax@1": round(100 * open1 / max(n, 1), 1),
           "reactive_argmax@2": round(100 * open2 / max(n, 1), 1)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    if a.leaf_out:
        os.makedirs(os.path.dirname(a.leaf_out), exist_ok=True)
        with open(a.leaf_out, "w") as fh:
            for r in leaf:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
