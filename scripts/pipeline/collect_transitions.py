#!/usr/bin/env python3
"""STAGE 1 — collect (s0, a1, s1, finish-labels) TRANSITIONS for the bootstrapped value.

Extends the ExIt finish collector (scripts/pipeline/exit_collect.py): it already sets s0 and steps a setup a1 -> s1.
Here we ALSO render the s0 crop, so each row is a full transition:
  - ctx0 / contact_px0  = s0 crop + contacts (for the SETUP value Q(s0, a1): forward(ctx0) -> map -> cell a1)
  - a1_edge / a1_depth  = the setup action taken
  - ctx / contact_px    = s1 crop + contacts (for the FINISH value V(s1) = top-k mean Q(s1, .))
  - f_grid / r_mask      = exhaustive finish labels at s1 (which a2 open; reachable a2)
  - dead                 = (n_open==0): s1 with NO opener -> bootstrap target for a1 ~ 0 (the discriminative signal)
The bootstrap trainer then sets the setup target = [a1 opens]==0 here ? gamma * V(s1) : 1.0, V(s1) from a frozen
finish net on ctx. --setups both -> model's top-K (on-policy, incl DEAD setups) + the labeled valid setups
(opener-bearing) = the full good/dead range the setup value must discriminate. Reuses exit_collect helpers (DRY)."""
import sys, os, json, argparse, math, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", f"{REPO}/scripts/pipeline", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402
import h5py  # noqa: E402
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from exit_collect import exhaustive_a2, obj_moved, ed, iter_records, TRAIN_KEY, OUT  # noqa: E402
from namo.paths import H5, resolve  # noqa: E402


def collect(a):
    pl = BeamPlanner(ckpt=a.ckpt)
    KEYS = ("ctx0", "contact_px0", "ctx", "contact_px", "f_grid", "r_mask", "object_center", "xml", "ratio",
            "a1_edge", "a1_depth", "dead", "n_open", "n_tried")
    buf = {k: [] for k in KEYS}
    n = kept = skip_noeffect = 0; t0 = time.time()
    for xml, rec in iter_records(a.key, a.start, a.end):
        obj = rec["object_id"]
        try:
            xmlp = str(resolve(xml)); env = make_env(xmlp); goal = extract_goal_with_fallback(xmlp, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state()
            goal_pts = sample_goal_points(env)
        except Exception as ex:
            print(f"  skip {os.path.basename(xml)}: {ex}", file=sys.stderr); continue
        if not goal_pts:
            continue
        s0_pose = env.get_observation()[f"{obj}_pose"]
        # render s0 ONCE (same crop for every setup of this episode)
        env.set_full_state(s0)
        try:
            ctx0, _ = pl.scorer.render_ctx(env, obj, goal, xml); cpx0 = pl.scorer.contact_px_live(env, obj)
        except Exception as ex:
            print(f"  s0 render fail {os.path.basename(xml)}: {ex}", file=sys.stderr); continue
        ctx0 = ctx0.astype(np.float32); cpx0 = cpx0.astype(np.float32)
        pool = rank_first_pushes_h2(pl, env, goal, xml, s0, 2, restrict_obj=obj)
        ed2g = {}
        for (_o, g, _q) in pool:
            ed2g.setdefault(ed(g), g)
        model_goals = [g for (_o, g, _q) in pool[:a.topk_setups]] if a.setups in ("model", "both") else []
        valid_goals = []
        if a.setups in ("valid", "both"):
            vfp = [tuple(x) for x in rec.get("valid_first_push", [])]
            valid_goals = [ed2g[c] for c in vfp[:a.topk_setups] if c in ed2g]
        mg_eds = {ed(g) for g in model_goals}
        setup_goals = model_goals + [g for g in valid_goals if ed(g) not in mg_eds]
        seen = set()
        for g1 in setup_goals:
            a1 = ed(g1)
            if a1 in seen:
                continue
            seen.add(a1)
            env.set_full_state(s0); env.step(make_action(obj, g1)); s1 = env.get_full_state()
            dxy, dth = obj_moved(env, obj, s0_pose)
            if dxy < 0.005 and dth < math.radians(3):
                skip_noeffect += 1; continue
            f_grid, r_mask, n_open, n_tried = exhaustive_a2(pl, env, goal, xml, s1, goal_pts, obj)
            if n_tried == 0:
                continue
            env.set_full_state(s1)
            try:
                ctx1, _ = pl.scorer.render_ctx(env, obj, goal, xml); cpx1 = pl.scorer.contact_px_live(env, obj)
            except Exception as ex:
                print(f"  s1 render fail {os.path.basename(xml)} a1={a1}: {ex}", file=sys.stderr); continue
            oc = env.get_observation()[f"{obj}_pose"][:2]
            buf["ctx0"].append(ctx0); buf["contact_px0"].append(cpx0)
            buf["ctx"].append(ctx1.astype(np.float32)); buf["contact_px"].append(cpx1.astype(np.float32))
            buf["f_grid"].append(f_grid); buf["r_mask"].append(r_mask)
            buf["object_center"].append(np.array(oc, np.float32)); buf["xml"].append(xml)
            buf["ratio"].append(np.float32(n_open / max(n_tried, 1)))
            buf["a1_edge"].append(np.int64(a1[0])); buf["a1_depth"].append(np.int64(a1[1]))
            buf["dead"].append(np.int64(int(n_open == 0)))
            buf["n_open"].append(np.int64(n_open)); buf["n_tried"].append(np.int64(n_tried))
            kept += 1
        n += 1
        if n % 25 == 0:
            print(f"  [{n}] kept={kept} noeffect={skip_noeffect} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    M = kept
    os.makedirs(os.path.dirname(a.out_h5), exist_ok=True)
    with h5py.File(a.out_h5, "w") as f:
        for k, shp in (("ctx0", (5, OUT, OUT)), ("ctx", (5, OUT, OUT)), ("contact_px0", (60, 2)),
                       ("contact_px", (60, 2)), ("f_grid", (60, 5)), ("r_mask", (60, 5))):
            f.create_dataset(k, data=np.stack(buf[k]) if M else np.zeros((0, *shp), np.float32), compression="lzf")
        f.create_dataset("object_center", data=np.stack(buf["object_center"]) if M else np.zeros((0, 2), np.float32))
        f.create_dataset("ratio", data=np.array(buf["ratio"], np.float32))
        for k in ("a1_edge", "a1_depth", "dead", "n_open", "n_tried"):
            f.create_dataset(k, data=np.array(buf[k], np.int64))
        f.create_dataset("xml", data=np.array(buf["xml"], dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        f.attrs["n_samples"] = M
    dead = int(np.array(buf["dead"]).sum()) if M else 0
    print(json.dumps({"out": a.out_h5, "episodes": n, "transitions": M, "dead_s1": dead,
                      "frac_dead": round(dead / max(M, 1), 3), "noeffect_skipped": skip_noeffect}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model whose top-K on-policy setups we step (current best, e.g. NoHz-v3)")
    ap.add_argument("--key", default=TRAIN_KEY)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=5076)
    ap.add_argument("--topk-setups", type=int, default=6, help="setups expanded/scene (more = more transitions incl dead)")
    ap.add_argument("--setups", default="both", choices=["model", "valid", "both"])
    ap.add_argument("--out-h5", default=str(H5 / "v4_hq_transitions/shard_0.h5"))
    collect(ap.parse_args())


if __name__ == "__main__":
    main()
