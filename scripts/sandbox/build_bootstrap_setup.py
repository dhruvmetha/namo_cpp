#!/usr/bin/env python3
"""STAGE 1 (grounded) — build the bootstrap-SETUP scorer H5 CHEAPLY from the exhaustive training labels (NO re-sim).

The labels (labels_exhaustive_pure2push.json) already store `frac_first_push = [[edge,depth,n_open,n_tried],...]`
for EVERY tried setup = the GROUND-TRUTH finish outcome at each s1. So V_GT(s1) is free; we only render s0.

Per episode: render the s0 crop (ctx + contact_px), and build a 60x5 SETUP target map:
  cell (edge,depth) target = [n_open==0] ? 0 : gamma * V_GT,   r_mask = 1 for every tried setup (dense supervision)
  --vsummary depth   : V_GT = 1.0           -> solvable setup target = 0.9      (existence; ~the flat status-quo label)
  --vsummary density : V_GT = n_open/n_tried -> solvable setup target = 0.9*density  (FINDABILITY = the cost-to-go signal)
[--vsummary is the STAGE-3 ablation: change ONLY this.] Output = standard scorer H5; train with budget_cond=false (single
Q, Horizon dropped). Cheap: ~5076 s0 renders, no MuJoCo stepping, no a2 sims. gamma=0.9."""
import sys, os, json, argparse, time
REPO = "/cache/home/dm1487/projects/namo/namo_cpp"; SAGE = "/cache/home/dm1487/projects/namo/sage_learning"
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", f"{REPO}/scripts/pipeline", SAGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np, h5py  # noqa: E402
from scorer_beam import BeamPlanner, make_env, FALLBACK_GOAL  # noqa: E402
from namo.core.xml_goal_parser import extract_goal_with_fallback  # noqa: E402

GAMMA = 0.9
OUT = 64


def episodes(key):
    d = json.load(open(key))
    for xml, recs in d.items():
        for rec in recs:
            yield xml, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="any registered ckpt — used ONLY for its renderer (no scoring)")
    ap.add_argument("--key", default="/scratch/dm1487/datasets/v4_hq_h2/labels_exhaustive_pure2push.json")
    ap.add_argument("--vsummary", default="density", choices=["depth", "density"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0=to end (episode-index shard)")
    ap.add_argument("--out-h5", required=True)
    a = ap.parse_args()
    pl = BeamPlanner(ckpt=a.ckpt)
    eps = list(episodes(a.key)); eps = eps[a.start:(a.end if a.end else len(eps))]
    buf = {k: [] for k in ("ctx", "contact_px", "f_grid", "r_mask", "object_center", "xml", "ratio", "H", "dead")}
    n = kept = nsolv = ndead = 0; t0 = time.time()
    for xml, rec in eps:
        obj = rec["object_id"]; ffp = rec.get("frac_first_push", [])
        if not ffp:
            continue
        try:
            env = make_env(xml); goal = extract_goal_with_fallback(xml, FALLBACK_GOAL)
            env.set_robot_goal(*goal); env.get_reachable_objects()
            ctx, _ = pl.scorer.render_ctx(env, obj, goal, xml); cpx = pl.scorer.contact_px_live(env, obj)
        except Exception as ex:
            print(f"  skip {os.path.basename(xml)}: {ex}", file=sys.stderr); continue
        fg = np.zeros((60, 5), np.float32); rm = np.zeros((60, 5), np.float32); ns = nd = 0
        for row in ffp:
            e, d, n_open, n_tried = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if not (0 <= e < 60 and 0 <= d < 5):
                continue
            rm[e, d] = 1.0
            if n_open == 0:
                fg[e, d] = 0.0; nd += 1
            else:
                V = 1.0 if a.vsummary == "depth" else n_open / max(n_tried, 1)
                fg[e, d] = GAMMA * float(V); ns += 1
        if rm.sum() == 0:
            continue
        buf["ctx"].append(ctx.astype(np.float32)); buf["contact_px"].append(cpx.astype(np.float32))
        buf["f_grid"].append(fg); buf["r_mask"].append(rm)
        oc = rec.get("object_center", [0.0, 0.0])
        buf["object_center"].append(np.array(oc[:2], np.float32)); buf["xml"].append(xml)
        buf["ratio"].append(np.float32(ns / max(ns + nd, 1))); buf["H"].append(np.int64(2))
        buf["dead"].append(np.int64(int(ns == 0)))
        kept += 1; nsolv += ns; ndead += nd; n += 1
        if n % 200 == 0:
            print(f"  [{n}] kept={kept} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    M = kept
    os.makedirs(os.path.dirname(a.out_h5), exist_ok=True)
    with h5py.File(a.out_h5, "w") as f:
        f.create_dataset("ctx", data=np.stack(buf["ctx"]) if M else np.zeros((0, 5, OUT, OUT), np.float32), compression="lzf")
        f.create_dataset("contact_px", data=np.stack(buf["contact_px"]) if M else np.zeros((0, 60, 2), np.float32), compression="lzf")
        f.create_dataset("f_grid", data=np.stack(buf["f_grid"]) if M else np.zeros((0, 60, 5), np.float32), compression="lzf")
        f.create_dataset("r_mask", data=np.stack(buf["r_mask"]) if M else np.zeros((0, 60, 5), np.float32), compression="lzf")
        f.create_dataset("object_center", data=np.stack(buf["object_center"]) if M else np.zeros((0, 2), np.float32))
        f.create_dataset("ratio", data=np.array(buf["ratio"], np.float32))
        f.create_dataset("H", data=np.array(buf["H"], np.int64))
        f.create_dataset("dead", data=np.array(buf["dead"], np.int64))
        f.create_dataset("xml", data=np.array(buf["xml"], dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        f.attrs["n_samples"] = M
    print(json.dumps({"out": a.out_h5, "vsummary": a.vsummary, "rows": M,
                      "solvable_setup_cells": nsolv, "dead_setup_cells": ndead}, indent=1))


if __name__ == "__main__":
    main()
