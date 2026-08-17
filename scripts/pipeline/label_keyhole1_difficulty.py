#!/usr/bin/env python3
"""Assign each two-hop scene's FIRST keyhole a canonical difficulty tier.

Joins the exhaustive depth-2 answer key (scripts/pipeline/build_2push_validset.py) against the
static topology probe (scripts/pipeline/probe_static_topology.py) and keeps only the episodes that
sit on **boundary 0** — the first hop off the robot region, the only boundary the deploy planner
ever opens (full_namo_planner.search). The collection enumerates every neighbour of the robot
region, so this filter is what turns "all neighbours" into "the keyhole".

The tier is `eval_common.bin_of(solve_rate_1push)` — hard < 0.05, med < 0.30, easy >= 0.30 — where
`solve_rate_1push = len(valid_1push)/len(tried_1push)`. That is byte-for-byte the definition
build_episode_validsets.py:79 uses for namo_testset_v1, so these labels land on the SAME scale as
the existing corpus and are directly comparable to it.

⚠ Every join here is on `os.path.realpath(xml)`, NEVER the basename. The aug9 pool has only 800
unique basenames across 2,535 scenes (generated/set{1,2}/benchmark_{1..5}/run_XXXX/ reuse the same
env_XXXX_pair_NNN.xml names), so a basename join silently mislabels ~68% of the pool.

Per-EPISODE is the unit (docs/pipeline/multi_episode_rooms.md): boundary 0 is an OR boundary and
can list more than one blocking object, and each (object, target region) is its own episode with its
own solve rate. Scenes are rolled up separately and both views are reported:
  pooled  sum(valid)/sum(tried) over the keyhole's episodes — the planner ranks pushes across all
          reachable blockers at the boundary, so this is what it actually faces
  best    max solve_rate over those episodes — the easiest single object
For the ~97% of scenes whose boundary 0 lists one object the two coincide.

  python scripts/pipeline/label_keyhole1_difficulty.py \
      --validset 2push_key.json --probe probe.jsonl --surviving surviving_xmls.txt \
      --out-episodes kh1_episodes.jsonl --out-scenes kh1_scenes.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "scripts"))

from eval_common import bin_of  # noqa: E402  the ONE binning rule; never redefine it here

RP = os.path.realpath
TIERS = ("easy", "med", "hard")


def load_probe(path):
    """realpath -> {target_region, objects, reachable}. Boundary 0 only."""
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        b = (r.get("boundaries") or [None])[0]
        path_ = r.get("region_path")
        if not b or not path_ or len(path_) < 2:
            continue
        out[RP(r["xml_path"])] = {
            "target_region": path_[1],
            "objects": list(b["objects"]),
            "reachable": list(b["reachable_objects"]),
            "region_path": path_,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validset", required=True, help="build_2push_validset.py --out JSON")
    ap.add_argument("--probe", required=True, help="probe_static_topology.py JSONL")
    ap.add_argument("--surviving", required=True, help="surviving_xmls.txt (the exact population)")
    ap.add_argument("--out-episodes", required=True, help="one row per keyhole-1 EPISODE")
    ap.add_argument("--out-scenes", required=True, help="one row per scene (rolled up)")
    a = ap.parse_args()

    probe = load_probe(a.probe)
    key = {RP(k): v for k, v in json.load(open(a.validset)).items()}
    pop = [RP(x) for x in open(a.surviving).read().split() if x.strip()]
    print(f"population {len(pop)} scenes | probe {len(probe)} | answer key {len(key)} scenes",
          file=sys.stderr)

    ep_rows, scene_rows = [], []
    miss_probe = miss_key = miss_ep = 0

    for xml in pop:
        pr = probe.get(xml)
        if pr is None:
            miss_probe += 1
            continue
        tgt, objs = pr["target_region"], set(pr["objects"])
        eps = key.get(xml)
        if eps is None:
            miss_key += 1
            scene_rows.append({"xml_path": xml, "target_region": tgt, "keyhole_objects": sorted(objs),
                               "status": "no_collection", "n_episodes": 0})
            continue
        # boundary-0 filter: same target region AND one of that boundary's blocking objects
        kh = [e for e in eps if e.get("region") == tgt and e.get("object_id") in objs]
        if not kh:
            miss_ep += 1
            scene_rows.append({"xml_path": xml, "target_region": tgt, "keyhole_objects": sorted(objs),
                               "status": "no_keyhole1_episode", "n_episodes": 0,
                               "regions_collected": sorted({e.get("region") for e in eps})})
            continue

        nv = nt = 0
        best = -1.0
        for e in kh:
            v, t = len(e["valid_1push"]), len(e["tried_1push"])
            sr = e["solve_rate_1push"]
            nv += v
            nt += t
            best = max(best, sr)
            ep_rows.append({
                "xml_path": xml, "object_id": e["object_id"], "object_center": e["object_center"],
                "target_region": tgt, "region_path": pr["region_path"],
                "solve_rate_1push": sr, "solve_rate_first_push": e["solve_rate_first_push"],
                "n_valid_1push": v, "n_tried_1push": t,
                "n_valid_first_push": len(e["valid_first_push"]),
                "n_tried_first_push": len(e["tried_first_push"]),
                "is_1push_solvable": e["is_1push_solvable"], "is_2push_solvable": e["is_2push_solvable"],
                "depth2_censored": e["depth2_censored"], "is_dead_within_2push": e["is_dead_within_2push"],
                "reachable_at_t0": e["object_id"] in set(pr["reachable"]),
                "tier": bin_of(sr),
            })
        pooled = (nv / nt) if nt else 0.0
        scene_rows.append({
            "xml_path": xml, "target_region": tgt, "keyhole_objects": sorted(objs),
            "status": "ok", "n_episodes": len(kh),
            "objects_collected": sorted(e["object_id"] for e in kh),
            "solve_rate_1push_pooled": pooled, "solve_rate_1push_best": best,
            "n_valid_1push": nv, "n_tried_1push": nt,
            "solve_rate_first_push_best": max(e["solve_rate_first_push"] for e in kh),
            "any_1push_solvable": any(e["is_1push_solvable"] for e in kh),
            "any_2push_solvable": any(e["is_2push_solvable"] for e in kh),
            "tier": bin_of(pooled), "tier_best": bin_of(best),
        })

    for path, rows in ((a.out_episodes, ep_rows), (a.out_scenes, scene_rows)):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    ok = [s for s in scene_rows if s["status"] == "ok"]

    def hist(counter, total):
        return "  ".join(f"{t} {counter.get(t, 0)} ({100.0 * counter.get(t, 0) / max(1, total):.1f}%)"
                         for t in TIERS)

    print(f"\nkeyhole-1 EPISODES: {len(ep_rows)}", file=sys.stderr)
    print("  tier  " + hist(Counter(r["tier"] for r in ep_rows), len(ep_rows)), file=sys.stderr)
    print(f"\nkeyhole-1 SCENES labelled: {len(ok)} of {len(pop)}", file=sys.stderr)
    print("  tier (pooled)  " + hist(Counter(s["tier"] for s in ok), len(ok)), file=sys.stderr)
    print("  tier (best)    " + hist(Counter(s["tier_best"] for s in ok), len(ok)), file=sys.stderr)
    print(f"\n  1push-solvable {sum(1 for s in ok if s['any_1push_solvable'])}  "
          f"2push-solvable {sum(1 for s in ok if s['any_2push_solvable'])}  "
          f"dead-within-2 {sum(1 for s in ok if not s['any_2push_solvable'])}", file=sys.stderr)
    print(f"  episodes per scene: {dict(sorted(Counter(s['n_episodes'] for s in ok).items()))}",
          file=sys.stderr)
    if miss_probe or miss_key or miss_ep:
        print(f"\n⚠ unlabelled: no_probe_row {miss_probe}  no_collection {miss_key}  "
              f"no_keyhole1_episode {miss_ep}", file=sys.stderr)
    print(f"\nwrote {a.out_episodes} ({len(ep_rows)})  {a.out_scenes} ({len(scene_rows)})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
