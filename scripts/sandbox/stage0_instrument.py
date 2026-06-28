#!/usr/bin/env python3
"""STAGE 0 instrumentation: the REALIZED rank of the true setup (s0) and true finish (observed s1) for the current
model, vs the exhaustive GT pairmap, tiered by difficulty. This is the realizable-headroom measurement that gates the
whole redesign (how far is the current model from the oracle-ranking ceiling the pairmap implies).

Per episode (from exhaustive_pairmap_pure2.pkl, the (a1,a2)->opens GT):
  (a) realized SETUP rank @s0  = position of the FIRST GT-valid setup (a1 with >=1 opener) in the model's H=hsetup ranking
  (b) realized FINISH rank @s1 = sim the best GT setup (most openers) -> s1 -> position of the FIRST GT opener in the
      model's H=1 ranking  [fixed-GT-s1 = isolates the finish ranker from the setup picker]
Also records n_reachable, n_gt_setups, n_openers for context. rank is 0-based (0 = model's #1 == a real answer).
Sharded by xml-index. Read-only on training. Reuses rank_first_pushes_h2 (zero-sim model scoring) + 1 sim/episode."""
import sys, os, json, pickle, argparse, statistics as st
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", f"{REPO}/scripts/pipeline", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import SCRATCH, DATASETS  # noqa: E402


def order_of(pool):
    return [(int(getattr(g, "edge_idx", -1)), int(getattr(g, "depth", -1)), o, g) for (o, g, _q) in pool]


def rank_of(order, targets):
    """0-based position of the first target (edge,depth) in `order`; None if none present."""
    idx = {(e, d): i for i, (e, d, _o, _g) in enumerate(order)}
    rs = [idx[t] for t in targets if t in idx]
    return min(rs) if rs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairmap", default=str(SCRATCH / "eval/exhaustive_pairmap_pure2.pkl"))
    ap.add_argument("--divisions", default=str(DATASETS / "namo_testset_v1/labels/pure2push_divisions.json"))
    ap.add_argument("--hsetup", type=int, default=2, help="budget to query for the SETUP ranking (2=horizon; NoHz ignores)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0=to end (xml-index shard over pairmap episodes)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pm = pickle.load(open(a.pairmap, "rb"))["pairmap"]
    div = {}
    try:
        dd = json.load(open(a.divisions))
        for xml, recs in dd.items():
            for r in recs:
                div[(xml, r.get("object_id") or r.get("object"))] = r.get("division", "?")
    except Exception:
        pass
    pl = BeamPlanner(ckpt=a.ckpt)
    eps = list(pm.items()); eps = eps[a.start:(a.end if a.end else len(eps))]
    rows = []
    for i, ((xml, obj), a1map) in enumerate(eps):
        gt_setups = [a1 for a1, a2m in a1map.items() if any(a2m.values())]
        if not gt_setups:
            continue
        try:
            env = make_env(xml); goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state()
        except Exception:
            continue
        pool0 = rank_first_pushes_h2(pl, env, goal, xml, s0, a.hsetup, restrict_obj=obj, score=True)
        if not pool0:
            continue
        order0 = order_of(pool0)
        setup_rank = rank_of(order0, gt_setups)
        # finish probe on the best GT setup (most openers) -> s1
        best_a1 = max(gt_setups, key=lambda a1: sum(1 for v in a1map[a1].values() if v))
        gt_openers = [a2 for a2, v in a1map[best_a1].items() if v]
        g_best = next((g for (e, d, _o, g) in order0 if (e, d) == best_a1), None)
        finish_rank = None; n_reach1 = None
        if g_best is not None:
            try:
                env.set_full_state(s0); env.step(make_action(obj, g_best)); s1 = env.get_full_state()
                pool1 = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj, score=True)
                order1 = order_of(pool1); n_reach1 = len(order1)
                finish_rank = rank_of(order1, gt_openers)
            except Exception:
                pass
        rows.append({"tier": div.get((xml, obj), "?"), "setup_rank": setup_rank, "finish_rank": finish_rank,
                     "n_reach0": len(order0), "n_gt_setups": len(gt_setups), "n_openers": len(gt_openers),
                     "n_reach1": n_reach1})
        if i % 25 == 0:
            print(f"  [{i}/{len(eps)}] n={len(rows)}", file=sys.stderr, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"ckpt": os.path.basename(a.ckpt), "rows": rows}, open(a.out, "w"))
    # quick console summary
    def summ(key):
        for t in ["easy", "medium", "hard"]:
            rs = [r[key] for r in rows if r["tier"] == t and r[key] is not None]
            tot = sum(1 for r in rows if r["tier"] == t)
            if tot:
                print(f"  {key:11} {t:7}: found {len(rs)}/{tot} ({100*len(rs)/tot:.0f}%), median rank "
                      f"{st.median(rs):.1f}, top1 {100*sum(1 for x in rs if x==0)/max(len(rs),1):.0f}%")
    print(f"n={len(rows)}")
    summ("setup_rank"); summ("finish_rank")


if __name__ == "__main__":
    main()
