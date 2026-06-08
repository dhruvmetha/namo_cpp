#!/usr/bin/env python3
"""Build a PER-EPISODE answer key for the test set, keyed by xml -> [episodes].

A single xml (room) can have multiple planner episodes that push DIFFERENT objects toward
DIFFERENT goal regions, each with its own success set. The old per-xml validsets kept only one
episode, so a test sample from another episode got scored against the wrong key (mis-flagged as
failure). Here every valid episode is recorded with the identity needed to match it to its H5
sample exactly: the pushed object's pose (object_center/theta), the goal region, and valid/tried.

Match rule at eval time: pick the episode whose object_center is nearest the sample's object_center
(should be ~0 mm — the H5 object_center IS the pushed object's pose), then it's the right object;
GT (edge,depth) ∈ that episode's valid confirms the right goal.

Output: <out> = {xml: [{object_id, object_center:[x,y], object_theta, region, solve_rate,
                         valid:[[e,d]...], tried:[[e,d]...]}, ...]}
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool


def pkl_episodes(pkl):
    out = []
    try:
        d = pickle.load(open(pkl, "rb"))
    except Exception:
        return out
    for ep in d.get("episode_results") or []:
        st = ep.get("algorithm_stats") or {}
        log = st.get("primitive_trial_log") or []
        if not log:
            continue
        tried = sorted({(t["edge_idx"], t["depth"]) for t in log})
        valid = sorted({(t["edge_idx"], t["depth"]) for t in log if t.get("success")})
        if not valid:
            continue
        obj = st.get("chosen_object_id")
        # pushed object's INITIAL pose (object_center matches the H5's local_tight_object_center)
        pose = None
        for key in ("state_observations", "original_state_observations"):
            so = ep.get(key) or []
            if so and isinstance(so[0], dict) and f"{obj}_pose" in so[0]:
                pose = so[0][f"{obj}_pose"]; break
        if pose is None:
            continue  # can't anchor it -> skip (rare; only valid episodes are kept)
        out.append({
            "xml": ep.get("xml_file") or pkl,
            "object_id": obj,
            "object_center": [float(pose[0]), float(pose[1])],
            "object_theta": float(pose[2]),
            "region": st.get("neighbour_region_label"),
            "solve_rate": len(valid) / len(log),
            "valid": [list(t) for t in valid],
            "tried": [list(t) for t in tried],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)  # the locked per-division pkl lists
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    pkls = set()
    for m in a.manifests:
        pkls.update(ln.strip() for ln in open(m) if ln.strip())
    pkls = sorted(pkls)
    print(f"reading {len(pkls)} pkls", file=sys.stderr)

    by_xml = {}
    n_eps = 0
    with Pool(a.workers) as pool:
        for recs in pool.imap_unordered(pkl_episodes, pkls, chunksize=16):
            for r in recs:
                by_xml.setdefault(r["xml"], []).append(r)
                n_eps += 1

    multi = sum(1 for v in by_xml.values() if len(v) > 1)
    print(f"{len(by_xml)} xmls, {n_eps} valid episodes, {multi} xmls with >1 episode", file=sys.stderr)
    json.dump(by_xml, open(a.out, "w"))
    print(f"wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
