#!/usr/bin/env python3
"""RUNG-1 trial-log -> training H5 (dense 60x5 value field, {-1,0,1} + mask).

Reads region_opening EXHAUSTIVE+SAMPLED depth-1 pkls (produced by region_opening_rung1_car.yaml) and
emits ONE row per EPISODE (= (scene, chosen object, goal region)). All ~25 sampled shoves are executed
as a SINGLE push from the SAME start state, so ctx is the start-state render, shared by the whole row.

Per row (60x5 = (4 edges x 15 pts) x 5 car-d5 depths):
  ctx          (5,64,64) f16  start-state crop (LiveScorer.render_ctx, region_samples=None ->
                              goal channel = the single robot_goal point, exactly how deploy scores)
  contact_px   (60,2)   f32   edge contact pixels at the start-state object pose
  r_mask       (60,5)   f32   1 on reachable edges (all depths) = legal cells (ScorerH5Dataset convention)
  value_target (60,5)   f32   {-1,0,1}: opener=1 | tried-didn't-open=0 | unreachable edge=-1 |
                              reachable-but-unsampled=0 (MASKED, target is a placeholder)
  value_mask   (60,5)   f32   {0,1}: 1 = in loss (opener / non-opener / unreachable);
                              0 = reachable-but-unsampled (no executed signal -> excluded)
  f_grid       (60,5)   f32   (value_target==1) binary opener layer, for drop-in vanilla-scorer readers
  has_opener   i8             1 if >=1 opener (depth-1 solvable) else 0  <- the rung-2 workload sort
  xml str, object_id str, robot_goal (3,) f32, n_reach_edges/n_sampled/n_open i32, edges_agree i8 (QC)

LABEL SEMANTICS (grounded, not invented):
  - reachable-but-untried -> MASK (NOT 0): forcing untried cells to 0 is the measured "C15 poison"
    (false negatives suppress valid pushes). See docs .../_1push_bottleneck.md.
  - unreachable edge -> trainable -1: the deliberate feasibility band (the value head learns
    reachability). See docs .../_reachability_loss_v3.md ("unreachable_k").
  The "tried" set is the primitive_trial_log (the cells actually executed); the reachable set is the
  reachability_log ROOT node (parent_edge is None) — the exact set the trial candidates were filtered to.

Blacklist caveat: even in exhaustive mode, a physical stuck/collision at depth d skips deeper depths on
that edge, so those cells are reachable-but-not-in-trial-log -> MASKED (conservative; no executed signal).
"""
import sys, os, glob, argparse, pickle
from pathlib import Path

import numpy as np

# --- path bootstrap (portable across CS boxes / worktree) ---
REPO = Path(__file__).resolve().parents[2]
SAGE = os.environ.get("SAGE_REPO", "/common/home/dm1487/robotics_research/ktamp/sage_learning")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import namo_rl                                                  # noqa: E402
from scorer_beam import make_env, FALLBACK_GOAL                 # noqa: E402
from namo.rl_loop.build_train_h5 import _Renderer               # noqa: E402  (model-free LiveScorer)

NUM_DEPTHS = 5
CAR_CFG = f"{REPO}/config/namo_config_complete_skill15_car_1x.yaml"


def _getf(e, k, default=None):
    return e.get(k, default) if isinstance(e, dict) else getattr(e, k, default)


def _root_reachable_edges(reach_log, obj):
    """Root-node reachable edge set (chain_depth==1, parent_edge is None), for `obj` if tagged."""
    if not reach_log:
        return None
    roots = [r for r in reach_log if r.get("parent_edge") is None]
    if obj is not None:
        tagged = [r for r in roots if r.get("object_id") in (None, obj)]
        roots = tagged or roots
    if not roots:
        return None
    # If somehow >1 root (shouldn't at chain_depth 1), union to be safe.
    edges = set()
    for r in roots:
        edges.update(int(x) for x in (r.get("reachable_edges") or []))
    return edges


def _build_labels(trial_log, reachable_edges):
    """Return (value_target(60,5), value_mask(60,5), r_mask(60,5), n_sampled, n_open)."""
    value_target = np.zeros((60, NUM_DEPTHS), np.float32)
    value_mask = np.zeros((60, NUM_DEPTHS), np.float32)

    # tried cells win: {opener=1, else 0}, mask=1. OR across dup entries (restarts).
    tried = {}
    for t in trial_log:
        e, d = int(t["edge_idx"]), int(t["depth"])
        if 0 <= e < 60 and 0 <= d < NUM_DEPTHS:
            tried[(e, d)] = tried.get((e, d), 0) or (1 if t["success"] else 0)
    n_open = sum(1 for v in tried.values() if v == 1)

    reach = set(int(x) for x in (reachable_edges or []))
    for e in range(60):
        for d in range(NUM_DEPTHS):
            if (e, d) in tried:
                value_target[e, d] = float(tried[(e, d)])
                value_mask[e, d] = 1.0
            elif e in reach:
                # reachable but not executed -> MASK (no signal)
                value_mask[e, d] = 0.0
            else:
                # unreachable edge -> trainable feasibility negative
                value_target[e, d] = -1.0
                value_mask[e, d] = 1.0

    r_mask = np.zeros((60, NUM_DEPTHS), np.float32)
    for e in reach:
        if 0 <= e < 60:
            r_mask[e, :] = 1.0
    return value_target, value_mask, r_mask, len(tried), n_open


def _episodes(pkl_paths):
    for p in pkl_paths:
        try:
            d = pickle.load(open(p, "rb"))
        except Exception as ex:
            print(f"  [skip] {p}: {ex}")
            continue
        for e in (_getf(d, "episode_results") or []):
            yield p, e


def build(pkl_glob, out_h5, render_config, limit=None):
    pkls = sorted(glob.glob(pkl_glob))
    print(f"[rung1] {len(pkls)} pkls from {pkl_glob}")
    renderer = _Renderer(render_config)
    env_cache = {}

    rows = []
    stats = dict(episodes=0, skipped_no_tl=0, skipped_no_reach=0, edges_agree=0, edges_disagree=0,
                 ep_with_opener=0, ep_no_opener=0)
    for p, e in _episodes(pkls):
        st = _getf(e, "algorithm_stats") or {}
        tl = st.get("primitive_trial_log")
        obj = st.get("chosen_object_id")
        xml = _getf(e, "xml_file")
        goal = _getf(e, "robot_goal") or FALLBACK_GOAL
        if not tl:
            stats["skipped_no_tl"] += 1
            continue
        tl = [t for t in tl if int(t.get("chain_depth", 1)) == 1]  # rung-1: depth-1 only
        if not tl:
            stats["skipped_no_tl"] += 1
            continue
        reach = _root_reachable_edges(st.get("reachability_log"), obj)
        if not reach:
            stats["skipped_no_reach"] += 1
            continue

        value_target, value_mask, r_mask, n_sampled, n_open = _build_labels(tl, reach)

        # --- render ctx + contact_px at the START (reset) state == the search baseline ---
        if xml not in env_cache:
            env_cache[xml] = make_env(xml)
        env = env_cache[xml]
        env.reset()
        # QC: does the reset state reproduce the search's reachable-edge set?
        try:
            env.get_reachable_objects()  # warm wavefront before per-object edge query
            live_edges = set(int(x) for x in env.get_reachable_edges(obj))
        except Exception:
            live_edges = None
        agree = live_edges is not None and live_edges == reach
        stats["edges_agree" if agree else "edges_disagree"] += 1

        ctx, _ = renderer.render_ctx(env, obj, goal, xml)
        cpx = renderer.contact_px_live(env, obj)

        has_opener = 1 if n_open > 0 else 0
        stats["ep_with_opener" if has_opener else "ep_no_opener"] += 1
        stats["episodes"] += 1

        rows.append(dict(
            ctx=ctx.astype(np.float16), contact_px=cpx.astype(np.float32), r_mask=r_mask,
            value_target=value_target, value_mask=value_mask,
            f_grid=(value_target == 1.0).astype(np.float32),
            has_opener=np.int8(has_opener),
            xml=str(xml), object_id=str(obj),
            robot_goal=np.array(goal[:3], np.float32),
            n_reach_edges=np.int32(len(reach)), n_sampled=np.int32(n_sampled), n_open=np.int32(n_open),
            edges_agree=np.int8(1 if agree else 0),
        ))
        if limit and len(rows) >= limit:
            break

    _write(out_h5, rows)
    return rows, stats


def _write(path, rows):
    import h5py
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["n_samples"] = len(rows)
        f.attrs["num_depths"] = NUM_DEPTHS
        f.attrs["label_scheme"] = ("value_target{-1,0,1}+value_mask: opener=1|tried-no-open=0|"
                                   "unreachable-edge=-1|reachable-unsampled=MASK; row=one episode; ctx=start state")
        f.attrs["generation"] = 0
        if not rows:
            return
        dt = h5py.string_dtype(encoding="utf-8")
        stack = lambda k: np.stack([r[k] for r in rows])
        arr = lambda k, npy: np.array([r[k] for r in rows], npy)
        f.create_dataset("ctx", data=stack("ctx"))                    # no compression (dataloader speed)
        f.create_dataset("contact_px", data=stack("contact_px"))
        f.create_dataset("r_mask", data=stack("r_mask"))
        f.create_dataset("value_target", data=stack("value_target"))
        f.create_dataset("value_mask", data=stack("value_mask"))
        f.create_dataset("f_grid", data=stack("f_grid"))
        f.create_dataset("robot_goal", data=stack("robot_goal"))
        f.create_dataset("has_opener", data=arr("has_opener", np.int8))
        f.create_dataset("n_reach_edges", data=arr("n_reach_edges", np.int32))
        f.create_dataset("n_sampled", data=arr("n_sampled", np.int32))
        f.create_dataset("n_open", data=arr("n_open", np.int32))
        f.create_dataset("edges_agree", data=arr("edges_agree", np.int8))
        f.create_dataset("xml", data=np.array([r["xml"] for r in rows], dtype=object), dtype=dt)
        f.create_dataset("object_id", data=np.array([r["object_id"] for r in rows], dtype=object), dtype=dt)


def _report(rows, stats, out_h5):
    n = len(rows)
    print("\n================ RUNG-1 H5 SUMMARY ================")
    print(f"out_h5 = {out_h5}")
    print(f"episodes(rows) = {n}  | skipped_no_trial_log={stats['skipped_no_tl']} "
          f"skipped_no_reachability={stats['skipped_no_reach']}")
    if n == 0:
        return
    vt = np.stack([r["value_target"] for r in rows]); vm = np.stack([r["value_mask"] for r in rows])
    tot = vt.size
    n_pos = int(((vt == 1) & (vm == 1)).sum())
    n_zero = int(((vt == 0) & (vm == 1)).sum())
    n_neg = int(((vt == -1) & (vm == 1)).sum())
    n_mask = int((vm == 0).sum())
    print(f"cells total = {tot}  (rows x 300)")
    print(f"  target= 1 (opener)     : {n_pos:>8}  ({100*n_pos/tot:.2f}%)")
    print(f"  target= 0 (no-open)    : {n_zero:>8}  ({100*n_zero/tot:.2f}%)")
    print(f"  target=-1 (unreachable): {n_neg:>8}  ({100*n_neg/tot:.2f}%)")
    print(f"  MASKED (reach-unsampl) : {n_mask:>8}  ({100*n_mask/tot:.2f}%)")
    print(f"  trained cells (mask=1) : {n_pos+n_zero+n_neg:>8}  ({100*(n_pos+n_zero+n_neg)/tot:.2f}%)")
    eo, en = stats["ep_with_opener"], stats["ep_no_opener"]
    print(f"episodes with >=1 opener (1-push solvable): {eo}/{n} ({100*eo/n:.1f}%)")
    print(f"episodes with 0 openers  (rung-2 workload): {en}/{n} ({100*en/n:.1f}%)")
    print(f"QC reset-state reachable-edge agreement: {stats['edges_agree']}/{stats['edges_agree']+stats['edges_disagree']} agree")
    ns = np.array([int(r["n_sampled"]) for r in rows]); nr = np.array([int(r["n_reach_edges"]) for r in rows])
    print(f"n_sampled per row: min={ns.min()} med={int(np.median(ns))} max={ns.max()} "
          f"| reachable edges per row: min={nr.min()} med={int(np.median(nr))} max={nr.max()}")
    print(f"ctx shape={rows[0]['ctx'].shape} contact_px={rows[0]['contact_px'].shape} "
          f"r_mask={rows[0]['r_mask'].shape} value_target={rows[0]['value_target'].shape}")
    print("==================================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-glob", required=True, help="glob for *_env_*_results.pkl")
    ap.add_argument("--out", required=True, help="output H5 path")
    ap.add_argument("--render-config", default=CAR_CFG)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows, stats = build(a.pkl_glob, a.out, a.render_config, a.limit)
    _report(rows, stats, a.out)
