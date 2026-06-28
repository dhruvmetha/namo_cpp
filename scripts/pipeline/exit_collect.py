#!/usr/bin/env python3
"""ExIt / DAgger ON-POLICY FINISH collection (the cure for the finish deploy-shift).

The finish head (H=1 on post-setup state s1) is mushy on TEST (sep 0.27) although sharp on TRAIN (0.75)
because it was trained ONLY on the COLLECTION planner's s1 states but is queried at deploy on the s1 the
MODEL lands in (GEN-GAP DISENTANGLED: ~2/3 deploy distribution shift). Static data can't fix an off-policy
shift; the cure is to collect the s1 the deployed model actually visits, label it exhaustively, retrain.

For each TRAIN pure-2 episode (key = labels_exhaustive_pure2push.json, the genuine 2-push set):
  1. the MODEL ranks the object's first pushes at H=2 (the setup head) and we take its top-K setups a1
     (the on-policy setups — exactly what the deployed model commits to);
  2. step to s1 = the state the model lands in (no-effect pushes filtered: <5mm & <3deg object move);
  3. EXHAUSTIVELY simulate every reachable a2 of the object at s1, label opens via the LABEL criterion
     goal_open_pts (>=20% of 100 s0-sampled goal pts reachable) -> f_grid/r_mask;
  4. render the s1 crop the SAME way deploy scores (LiveScorer.render_ctx, region_samples=None -> goal
     channel = the single robot_goal point, identical to rank_first_pushes_h2 at deploy & to the H5 the
     scorer trained on) + contact_px at the s1 object pose.
Write a scorer H5 shard (ctx/f_grid/r_mask/contact_px/ratio/object_center/xml/H/dead/postpush + onpolicy=1
provenance), byte-compatible with ScorerH5Dataset -> drops straight into the v2 mix in place of postpush.

--validate: NO rendering/output. Pick KNOWN real setups a1 from the labels' valid_first_push, run the
exhaustive-a2 labeling, and compare the opener count to the labels' frac_first_push [e,d,n_open,n_tried].
This gates the sim+success-check correctness against the independent collection before any big run.
"""
import sys, os, json, argparse, math, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402
from scorer_beam import BeamPlanner, make_env, make_action, FALLBACK_GOAL  # noqa: E402
from eval_m3 import rank_first_pushes_h2, sample_goal_points, goal_open_pts  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402
from namo.paths import DATASETS, H5, SCRATCH, resolve  # noqa: E402

TRAIN_KEY = str(DATASETS / "v4_hq_h2/labels_exhaustive_pure2push.json")
OUT = 64
ed = lambda g: (int(getattr(g, "edge_idx", -1)), int(getattr(g, "depth", -1)))


def exhaustive_a2(pl, env, goal, xml, s1, goal_pts, obj):
    """Sim every reachable a2 of `obj` at s1; return (f_grid(60,5), r_mask(60,5), n_open, n_tried).
    r_mask=1 on every reachable (tried) cell; f_grid=1 where the push opens the goal region (>=20%)."""
    f_grid = np.zeros((60, 5), np.float32); r_mask = np.zeros((60, 5), np.float32)
    pool = rank_first_pushes_h2(pl, env, goal, xml, s1, 1, restrict_obj=obj, score=False)  # candidate a2 pool
    n_open = 0
    for (o, g, _q) in pool:
        e, d = ed(g)
        if not (0 <= e < 60 and 0 <= d < 5):
            continue
        r_mask[e, d] = 1.0
        env.set_full_state(s1); env.step(make_action(o, g))
        if goal_open_pts(env, goal_pts):
            f_grid[e, d] = 1.0; n_open += 1
    return f_grid, r_mask, n_open, int(r_mask.sum())


def obj_moved(env, obj, s0_pose):
    p = env.get_observation()[f"{obj}_pose"]
    dxy = math.hypot(p[0] - s0_pose[0], p[1] - s0_pose[1])
    dth = abs((p[2] - s0_pose[2] + math.pi) % (2 * math.pi) - math.pi)
    return dxy, dth


def iter_records(key, start, end):
    d = json.load(open(key))
    keys = list(d.keys())[start:end]
    for xml in keys:
        for rec in d[xml]:
            yield xml, rec


# ---------------------------------------------------------------------------------------------------
def validate(a):
    """Cross-check exhaustive-a2 labeling vs the labels' frac_first_push opener counts (no model, no render)."""
    pl = BeamPlanner(ckpt=a.ckpt)
    n = ok = 0; rows = []
    for xml, rec in iter_records(a.key, a.start, a.end):
        if n >= a.max_scenes:
            break
        obj = rec["object_id"]; vfp = [tuple(x) for x in rec.get("valid_first_push", [])]
        if not vfp:
            continue
        frac = {(int(e), int(dp)): (int(no), int(nt)) for (e, dp, no, nt) in rec.get("frac_first_push", [])}
        try:
            xmlp = str(resolve(xml)); env = make_env(xmlp); goal = extract_goal_with_fallback(xmlp, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects(); s0 = env.get_full_state()
            goal_pts = sample_goal_points(env)
        except Exception as ex:
            print(f"  skip {os.path.basename(xml)}: {ex}", file=sys.stderr); continue
        if not goal_pts:
            continue
        # take up to 2 KNOWN real setups, step, label, compare opener count
        for a1 in vfp[:2]:
            pool = rank_first_pushes_h2(pl, env, goal, xml, s0, 1, restrict_obj=obj, score=False)
            g1 = next((g for (_o, g, _q) in pool if ed(g) == a1), None)
            if g1 is None:
                continue
            env.set_full_state(s0); env.step(make_action(obj, g1)); s1 = env.get_full_state()
            _fg, _rm, n_open, n_tried = exhaustive_a2(pl, env, goal, xml, s1, goal_pts, obj)
            lab_open, lab_tried = frac.get(a1, (None, None))
            match = (lab_open is not None) and (abs(n_open - lab_open) <= max(1, int(0.1 * max(lab_open, 1))))
            n += 1; ok += int(match)
            rows.append({"xml": os.path.basename(xml), "obj": obj, "a1": a1,
                         "mine_open": n_open, "mine_tried": n_tried, "lab_open": lab_open, "lab_tried": lab_tried,
                         "match": match})
            print(f"  a1={a1} mine={n_open}/{n_tried} label={lab_open}/{lab_tried} {'OK' if match else 'MISMATCH'}",
                  file=sys.stderr, flush=True)
    out = {"n": n, "matched": ok, "match_rate": round(100 * ok / max(n, 1), 1), "rows": rows[:40]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


# ---------------------------------------------------------------------------------------------------
def collect(a):
    """Model-driven on-policy finish collection -> scorer H5 shard."""
    import h5py
    pl = BeamPlanner(ckpt=a.ckpt)
    buf = {k: [] for k in ("ctx", "f_grid", "r_mask", "contact_px", "object_center", "xml", "ratio",
                           "H", "dead", "postpush", "onpolicy", "a1_edge", "a1_depth")}
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
        pool = rank_first_pushes_h2(pl, env, goal, xml, s0, 2, restrict_obj=obj)  # ranked candidate first pushes
        ed2g = {}
        for (_o, g, _q) in pool:
            ed2g.setdefault(ed(g), g)
        model_goals = [g for (_o, g, _q) in pool[:a.topk_setups]] if a.setups in ("model", "both") else []
        valid_goals = []
        if a.setups in ("valid", "both"):                       # the LABELED real setups -> opener-bearing s1
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
            if dxy < 0.005 and dth < math.radians(3):     # no-effect push -> not a real s1 (postpush filter)
                skip_noeffect += 1; continue
            f_grid, r_mask, n_open, n_tried = exhaustive_a2(pl, env, goal, xml, s1, goal_pts, obj)
            if n_tried == 0:
                continue
            env.set_full_state(s1)                         # render the crop AT s1 (deploy convention)
            try:
                ctx, _ = pl.scorer.render_ctx(env, obj, goal, xml)          # (5,64,64), region_samples=None
                cpx = pl.scorer.contact_px_live(env, obj)                   # (60,2)
            except Exception as ex:
                print(f"  render fail {os.path.basename(xml)} a1={a1}: {ex}", file=sys.stderr); continue
            oc = env.get_observation()[f"{obj}_pose"][:2]
            buf["ctx"].append(ctx.astype(np.float32)); buf["f_grid"].append(f_grid); buf["r_mask"].append(r_mask)
            buf["contact_px"].append(cpx.astype(np.float32)); buf["object_center"].append(np.array(oc, np.float32))
            buf["xml"].append(xml); buf["ratio"].append(np.float32(n_open / max(n_tried, 1)))
            buf["H"].append(np.int64(1)); buf["dead"].append(np.int64(int(n_open == 0)))
            buf["postpush"].append(np.int64(1)); buf["onpolicy"].append(np.int64(1))
            buf["a1_edge"].append(np.int64(a1[0])); buf["a1_depth"].append(np.int64(a1[1]))
            kept += 1
        n += 1
        if n % 25 == 0:
            print(f"  [{n}] kept={kept} noeffect={skip_noeffect} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    # write H5 shard
    M = kept
    os.makedirs(os.path.dirname(a.out_h5), exist_ok=True)
    with h5py.File(a.out_h5, "w") as f:
        f.create_dataset("ctx", data=np.stack(buf["ctx"]) if M else np.zeros((0, 5, OUT, OUT), np.float32),
                         compression="lzf")
        f.create_dataset("f_grid", data=np.stack(buf["f_grid"]) if M else np.zeros((0, 60, 5), np.float32),
                         compression="lzf")
        f.create_dataset("r_mask", data=np.stack(buf["r_mask"]) if M else np.zeros((0, 60, 5), np.float32),
                         compression="lzf")
        f.create_dataset("contact_px", data=np.stack(buf["contact_px"]) if M else np.zeros((0, 60, 2), np.float32),
                         compression="lzf")
        f.create_dataset("object_center", data=np.stack(buf["object_center"]) if M else np.zeros((0, 2), np.float32))
        f.create_dataset("ratio", data=np.array(buf["ratio"], np.float32))
        f.create_dataset("H", data=np.array(buf["H"], np.int64))
        f.create_dataset("dead", data=np.array(buf["dead"], np.int64))
        f.create_dataset("postpush", data=np.array(buf["postpush"], np.int64))
        f.create_dataset("onpolicy", data=np.array(buf["onpolicy"], np.int64))
        f.create_dataset("a1_edge", data=np.array(buf["a1_edge"], np.int64))
        f.create_dataset("a1_depth", data=np.array(buf["a1_depth"], np.int64))
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("xml", data=np.array(buf["xml"], dtype=object), dtype=dt)
        f.attrs["n_samples"] = M
    open_rows = int(np.array(buf["dead"]).sum()) if M else 0
    print(json.dumps({"out": a.out_h5, "episodes": n, "rows": M, "dead_rows": open_rows,
                      "noeffect_skipped": skip_noeffect,
                      "frac_dead": round(open_rows / max(M, 1), 3)}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the deployed model whose on-policy s1 we collect (Hz-v2)")
    ap.add_argument("--key", default=TRAIN_KEY)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=5076)
    ap.add_argument("--topk-setups", type=int, default=2, help="how many setups to expand/scene")
    ap.add_argument("--setups", default="model", choices=["model", "valid", "both"],
                    help="which setups to step to make finish data: model=model's top-k @H=2 (deploy-matching); "
                         "valid=the LABELED real setups from valid_first_push (opener-rich, DIVERSE finish-skill data "
                         "— the core generalization lever, on-policy barely matters per isolation 0.057); both=union.")
    ap.add_argument("--validate", action="store_true", help="cross-check labeling vs frac_first_push (no render/out)")
    ap.add_argument("--max-scenes", type=int, default=40, help="validate-mode cap")
    ap.add_argument("--out", default=str(SCRATCH / "eval/exit_validate.json"))
    ap.add_argument("--out-h5", default=str(H5 / "v4_hq_exit_finish/shard_0.h5"))
    a = ap.parse_args()
    (validate if a.validate else collect)(a)


if __name__ == "__main__":
    main()
