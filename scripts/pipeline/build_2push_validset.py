#!/usr/bin/env python3
"""Build the 2-PUSH answer key (F1') from the exhaustive depth-2 collection pkls.

Companion to build_episode_validsets.py (which builds the 1-push key). A depth-2 exhaustive run records
EVERY solution as its own episode_result; a 2-push solution's action_sequence = [first_push, second_push],
each a {edge_idx, depth, target} primitive. We aggregate per episode (pushed object + goal region):

  F1'  (valid_first_push)  = { action_sequence[0].(edge,depth) : success, chain_depth==2 }   <- ENABLING first pushes
  F    (valid_1push)       = { action_sequence[0].(edge,depth) : success, chain_depth==1 }    <- cross-check vs 1-push key
  tried                    = depth-1 reachable cells from primitive_trial_log (the denominator)

Output: {realpath: [{object_id, object_center:[x,y], object_theta, region,
                     valid_1push:[[e,d]...], valid_first_push:[[e,d]...], tried:[[e,d]...],
                     is_2push_solvable, solve_rate_first_push}, ...]}

Match an eval sample to its episode exactly like the 1-push key: nearest object_center (<=0.01 m).
Reuses NOTHING that would fork build_episode_validsets — different label (chain) so it is a sibling, not a copy.
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from glob import glob
from multiprocessing import Pool

RP = os.path.realpath


def _cell(step):
    return (int(step.get("edge_idx", -1)), int(step.get("depth", -1)))


def pkl_episodes(pkl):
    """Returns list of raw (epkey, fields) contributions; aggregation happens in main across pkls."""
    out = []
    try:
        d = pickle.load(open(pkl, "rb"))
    except Exception:
        return out
    for ep in d.get("episode_results") or []:
        st = ep.get("algorithm_stats") or {}
        obj = st.get("chosen_object_id")
        region = st.get("neighbour_region_label")
        xml = ep.get("xml_file") or pkl
        real = RP(xml)
        # pushed object's INITIAL pose (anchors the episode, matches H5 local_tight_object_center)
        pose = None
        for key in ("state_observations", "original_state_observations"):
            so = ep.get(key) or []
            if so and isinstance(so[0], dict) and f"{obj}_pose" in so[0]:
                pose = so[0][f"{obj}_pose"]; break
        if pose is None:
            continue
        oc = (round(float(pose[0]), 4), round(float(pose[1]), 4))
        epkey = (real, obj, oc, region)
        seq = ep.get("action_sequence") or []
        log = st.get("primitive_trial_log") or []
        chain = st.get("chain_depth")
        first = _cell(seq[0]) if seq else None
        out.append({
            "epkey": epkey, "real": real, "obj": obj, "oc": oc, "region": region,
            "theta": float(pose[2]),
            "success": bool(ep.get("success")),
            "chain": int(chain) if chain is not None else len(seq),
            "first": first,
            "tried": [(t["edge_idx"], t["depth"]) for t in log],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls-root", required=True, help="root containing shard_*/pkls/**/*.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    pkls = glob(os.path.join(a.pkls_root, "**", "*.pkl"), recursive=True)
    print(f"reading {len(pkls)} pkls", file=sys.stderr)

    agg = {}   # epkey -> {f1, f1p, tried, theta, ...}
    with Pool(a.workers) as pool:
        for recs in pool.imap_unordered(pkl_episodes, pkls, chunksize=8):
            for r in recs:
                e = agg.setdefault(r["epkey"], {
                    "real": r["real"], "obj": r["obj"], "oc": r["oc"], "region": r["region"],
                    "theta": r["theta"], "f1": set(), "f1p": set(), "tried": set()})
                e["tried"].update(r["tried"])
                if r["success"] and r["first"] is not None:
                    if r["chain"] >= 2:
                        e["f1p"].add(r["first"])
                    elif r["chain"] == 1:
                        e["f1"].add(r["first"])

    by_real = defaultdict(list)
    n_2solvable = 0
    for (real, obj, oc, region), e in agg.items():
        tried = sorted(e["tried"])
        f1 = sorted(e["f1"]); f1p = sorted(e["f1p"])
        solvable = len(f1p) > 0
        n_2solvable += solvable
        by_real[real].append({
            "object_id": obj,
            "object_center": [oc[0], oc[1]],
            "object_theta": e["theta"],
            "region": region,
            "valid_1push": [list(t) for t in f1],
            "valid_first_push": [list(t) for t in f1p],
            "tried": [list(t) for t in tried],
            "is_2push_solvable": solvable,
            "solve_rate_first_push": (len(f1p) / len(tried)) if tried else 0.0,
        })

    json.dump(dict(by_real), open(a.out, "w"))
    print(f"{len(by_real)} scenes, {len(agg)} episodes, {n_2solvable} 2-push-solvable episodes", file=sys.stderr)
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
