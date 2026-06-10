#!/usr/bin/env python3
"""Build the 2-PUSH answer key (F + F1') from the exhaustive depth-2 collection pkls.

Companion to build_episode_validsets.py (1-push key). Derives labels from the EXHAUSTIVE
`primitive_trial_log` (NOT the recorded episode_results, which are only a SAMPLE of solutions). The trial
log is tagged per entry with `chain_depth` + `parent_{edge,depth}` (region_opening.py), so per episode
(pushed object + goal region):

  F   (valid_1push)        = { (edge,depth)            : success, chain_depth==1 }   <- exhaustive 1-push solving cells
  F1' (valid_first_push)   = { (parent_edge,parent_dep): success, chain_depth==2 }   <- first-pushes ENABLING a 2-push solve
  tried_1push              = { (edge,depth)            : chain_depth==1 }            <- reachable depth-1 cells (denominator)
  tried_first_push         = { (parent_edge,parent_dep): chain_depth==2 }            <- first-pushes expanded to depth-2

Output: {realpath: [{object_id, object_center, object_theta, region,
                     valid_1push, valid_first_push, tried_1push, tried_first_push,
                     is_1push_solvable, is_2push_solvable, solve_rate_1push, solve_rate_first_push}, ...]}

Match an eval sample to its episode by nearest object_center (<=1mm), same rule as the 1-push key.
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


def pkl_episodes(pkl):
    """Per pkl -> list of (epkey, pose, trial-entries). The trial log is object-level (same on every
    episode_result for that object), so we dedup by epkey downstream."""
    out = []
    try:
        d = pickle.load(open(pkl, "rb"))
    except Exception:
        return out
    seen_obj = set()
    for ep in d.get("episode_results") or []:
        st = ep.get("algorithm_stats") or {}
        obj = st.get("chosen_object_id")
        region = st.get("neighbour_region_label")
        real = RP(ep.get("xml_file") or pkl)
        pose = None
        for key in ("state_observations", "original_state_observations"):
            so = ep.get(key) or []
            if so and isinstance(so[0], dict) and f"{obj}_pose" in so[0]:
                pose = so[0][f"{obj}_pose"]; break
        if pose is None:
            continue
        oc = (round(float(pose[0]), 4), round(float(pose[1]), 4))
        epkey = (real, obj, oc, region)
        if epkey in seen_obj:                 # object-level trial log is identical across its episode_results
            continue
        seen_obj.add(epkey)
        log = st.get("primitive_trial_log") or []
        out.append({"epkey": epkey, "theta": float(pose[2]), "log": log})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pure-out", help="optional: also write the PURE-2-push view "
                    "(episodes with is_1push_solvable==False & is_2push_solvable==True)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    pkls = glob(os.path.join(a.pkls_root, "**", "*.pkl"), recursive=True)
    pkls = [p for p in pkls if "collection_summary" not in os.path.basename(p)]
    print(f"reading {len(pkls)} pkls", file=sys.stderr)

    agg = {}
    n_untagged = 0
    with Pool(a.workers) as pool:
        for recs in pool.imap_unordered(pkl_episodes, pkls, chunksize=8):
            for r in recs:
                e = agg.setdefault(r["epkey"], {
                    "theta": r["theta"], "f1": set(), "f1p": set(),
                    "tried1": set(), "triedfp": set(), "tagged": False})
                for t in r["log"]:
                    cd = t.get("chain_depth")
                    if cd is None:
                        n_untagged += 1
                        continue
                    e["tagged"] = True
                    if cd == 1:
                        cell = (t["edge_idx"], t["depth"])
                        e["tried1"].add(cell)
                        if t.get("success"):
                            e["f1"].add(cell)
                    elif cd == 2:
                        pe, pd = t.get("parent_edge"), t.get("parent_depth")
                        if pe is None:
                            continue
                        pcell = (pe, pd)
                        e["triedfp"].add(pcell)
                        if t.get("success"):
                            e["f1p"].add(pcell)

    by_real = defaultdict(list)
    n_1, n_2only, n_unsolved = 0, 0, 0
    for (real, obj, oc, region), e in agg.items():
        f1 = sorted(e["f1"]); f1p = sorted(e["f1p"])
        t1 = sorted(e["tried1"]); tfp = sorted(e["triedfp"])
        is1 = len(f1) > 0
        is2 = is1 or len(f1p) > 0
        if is1:
            n_1 += 1
        elif len(f1p) > 0:
            n_2only += 1
        else:
            n_unsolved += 1
        by_real[real].append({
            "object_id": obj, "object_center": [oc[0], oc[1]], "object_theta": e["theta"], "region": region,
            "valid_1push": [list(t) for t in f1],
            "valid_first_push": [list(t) for t in f1p],
            "tried_1push": [list(t) for t in t1],
            "tried_first_push": [list(t) for t in tfp],
            "is_1push_solvable": is1,
            "is_2push_solvable": is2,
            "solve_rate_1push": (len(f1) / len(t1)) if t1 else 0.0,
            "solve_rate_first_push": (len(f1p) / len(tfp)) if tfp else 0.0,
        })

    json.dump(dict(by_real), open(a.out, "w"))
    print(f"{len(by_real)} scenes, {len(agg)} episodes "
          f"(1push-solvable {n_1}, 2push-only {n_2only}, unsolved-<=2 {n_unsolved})", file=sys.stderr)
    if n_untagged:
        print(f"⚠ {n_untagged} UNTAGGED trial entries skipped — re-collect with the tagged region_opening.py",
              file=sys.stderr)
    print(f"wrote {a.out}", file=sys.stderr)

    if a.pure_out:
        # PURE-2-push VIEW: exhaustively no 1-push, but 2-push-solvable. A derived subset of --out, not a new source.
        pure = {}
        n_ep = 0
        for real, eps in by_real.items():
            keep = [e for e in eps if (not e["is_1push_solvable"]) and e["is_2push_solvable"]]
            if keep:
                pure[real] = keep
                n_ep += len(keep)
        json.dump(pure, open(a.pure_out, "w"))
        print(f"pure-2-push view: {len(pure)} scenes, {n_ep} episodes -> {a.pure_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
