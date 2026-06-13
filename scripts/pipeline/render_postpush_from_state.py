#!/usr/bin/env python3
"""Render POST-PUSH states s1 = T(s0, a1) from the COLLECTOR-SAVED state (horizon-Q v2, the OOD fix).

This SUPERSEDES replay_postpush.py (⛔ blocked: replay diverges on collisions). The v2 re-collection
(job 56018429) extended region_opening's reachability_log to persist `state_observation` per expanded
node, so every post-push s1 the search VISITED is now on disk — the exact state whose a2-outcomes are in
the trial log. No replay, no env, no MuJoCo step ⇒ the collision-divergence bug CANNOT occur (we render
the collector's own saved state, byte-for-byte).

Per episode (one scene, one chosen object): reachability_log has root nodes (chain_depth==1, the s0) and
post-push nodes (chain_depth==2, each s1 = result of first push a1=(parent_edge,parent_depth)). For each
UNIQUE post-push s1 (dedup by (parent_edge,parent_depth) — region-goal duplicates are byte-identical,
verified Δpose=0), render the scene at s1 and self-carry its label from the trial log:
  • kids[(pe,pd)] = {(a2_edge,a2_depth): success} from primitive_trial_log chain_depth==2 entries.
  • openers = a2 with success  -> f_grid=1     (GOOD post-push: ≥1 opener; DEAD: none).
  • tried   = all a2 in kids    -> loss_mask    (sampled ~k a2; the rest UNKNOWN, masked — no C15 bug).
  • reach   = node reachable_edges -> r_mask (edge-level, expand all depths downstream).
FILTER no-effect s1 (object SE(2) ≈ root: xy<5mm AND |Δθ|<3° ⇒ push engaged nothing, redundant w/ root).

The render is OFFLINE (visualizer reads episode dict only; static_object_info comes from the pkl). Labels
are self-carried (pp_* keys) so the downstream H5 packer needs NO object_center matching.

  python scripts/pipeline/render_postpush_from_state.py --pkl-list shard.txt --output-dir out/ \
      [--max-per-episode 0]   # 0 = all unique post-push s1; >0 = cap for diversity/balance
"""
import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

import namo.visualization.mask_generation.batch_collection as bc
from sage_learning.visualizer import NAMODataVisualizer

CFG = "config/namo_config_complete_skill15_car_1x.yaml"
XY_EPS = 0.005          # 5 mm
TH_EPS = math.radians(3)  # 3 degrees


def _dtheta(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def kids_by_a1(trial_log):
    """(parent_edge,parent_depth) -> {(a2_edge,a2_depth): success_bool} from chain_depth==2 entries."""
    kids = defaultdict(dict)
    for t in trial_log:
        if t.get("chain_depth") == 2 and t.get("parent_edge") is not None:
            kids[(t["parent_edge"], t["parent_depth"])][(t["edge_idx"], t["depth"])] = bool(t.get("success"))
    return kids


def unique_postpush(reach_log):
    """Dedup chain_depth==2 nodes by (parent_edge,parent_depth) (region-goal dups are identical)."""
    seen = {}
    for n in reach_log:
        if n.get("chain_depth") == 2 and n.get("parent_edge") is not None:
            seen.setdefault((n["parent_edge"], n["parent_depth"]), n)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-list", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cfg", default=CFG)
    ap.add_argument("--max-per-episode", type=int, default=0,
                    help="0=all unique post-push s1; >0 caps per episode (stable order) for balance")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import pickle, random
    rng = random.Random(a.seed)

    pkls = [l.strip() for l in open(a.pkl_list) if l.strip()]
    viz = NAMODataVisualizer(figsize=(10, 8), namo_config_path=a.cfg)
    n_render = n_good = n_dead = n_noeff = n_skip = 0

    for pi, pkl in enumerate(pkls):
        try:
            d = pickle.load(open(pkl, "rb"))
        except Exception:
            continue
        for er in d.get("episode_results") or []:
            st = er.get("algorithm_stats") or {}
            obj = st.get("chosen_object_id")
            xml = er.get("xml_file")
            rl = st.get("reachability_log") or []
            tl = st.get("primitive_trial_log") or []
            soi = er.get("static_object_info") or {}
            if not obj or not xml or not rl or not soi:
                continue
            roots = [n for n in rl if n.get("chain_depth") == 1]
            if not roots:
                continue
            pose_key = f"{obj}_pose"
            try:
                root_pose = np.asarray(roots[0]["state_observation"][pose_key], dtype=float)
            except Exception:
                continue
            kids = kids_by_a1(tl)
            nodes = unique_postpush(rl)
            items = sorted(nodes.items())  # stable order (pe,pd)
            if a.max_per_episode > 0 and len(items) > a.max_per_episode:
                items = [items[i] for i in sorted(rng.sample(range(len(items)), a.max_per_episode))]
            for (pe, pd), node in items:
                ch = kids.get((pe, pd))
                if not ch:
                    n_skip += 1
                    continue
                so = node.get("state_observation") or {}
                if pose_key not in so:
                    n_skip += 1
                    continue
                p = np.asarray(so[pose_key], dtype=float)
                if np.max(np.abs(p[:2] - root_pose[:2])) < XY_EPS and _dtheta(p[2], root_pose[2]) < TH_EPS:
                    n_noeff += 1
                    continue  # no-effect: s1 ≈ s0, redundant with root
                tried = np.array(sorted(ch.keys()), dtype=np.int16)          # (n,2) loss_mask
                openers = np.array(sorted([c for c, s in ch.items() if s]), dtype=np.int16)  # f_grid=1
                dead = int(len(openers) == 0)
                reach = np.array(sorted(node.get("reachable_edges") or []), dtype=np.int16)   # r_mask edges
                ep = {
                    "episode_id": f"{er.get('episode_id','ep')}_pp_{pe}_{pd}",
                    "algorithm": er.get("algorithm", "region_opening"),
                    "solution_found": (dead == 0),
                    "state_observations": [so],
                    "post_action_state_observations": [],
                    "action_sequence": [{"object_id": obj}],
                    "static_object_info": soi,
                    "robot_goal": er.get("robot_goal", [0, 0, 0]),
                    "xml_file": xml,
                    "reachable_objects_before_action": None,
                    "algorithm_stats": {"chosen_object_id": obj,
                                        "neighbour_region_label": st.get("neighbour_region_label"),
                                        "region_goals_sampled": st.get("region_goals_sampled")},
                }
                try:
                    masks, meta = bc.process_episode(ep, viz, generate_local=True, local_only=False)
                except Exception:
                    n_skip += 1
                    continue
                if not masks or meta is None:
                    n_skip += 1
                    continue
                # self-carry the label IN THE MASKS DICT — save_episode_data only persists a hardcoded
                # whitelist of `metadata` keys (pp_* in meta would be silently dropped), but copies
                # EVERY masks key verbatim (save_dict = dict(masks)). So labels ride in masks.
                masks["pp_postpush"] = np.array([1], dtype=np.int16)
                masks["pp_dead"] = np.array([dead], dtype=np.int16)
                masks["pp_H"] = np.array([1], dtype=np.int16)
                masks["pp_parent_edge"] = np.array([int(pe)], dtype=np.int16)
                masks["pp_parent_depth"] = np.array([int(pd)], dtype=np.int16)
                masks["pp_tried_ed"] = tried[:, 0] if len(tried) else np.zeros(0, np.int16)
                masks["pp_tried_dp"] = tried[:, 1] if len(tried) else np.zeros(0, np.int16)
                masks["pp_open_ed"] = openers[:, 0] if len(openers) else np.zeros(0, np.int16)
                masks["pp_open_dp"] = openers[:, 1] if len(openers) else np.zeros(0, np.int16)
                masks["pp_reach_edges"] = reach if len(reach) else np.zeros(0, np.int16)
                out = os.path.join(a.output_dir, meta.get("task_id") or "t", f"{ep['episode_id']}.npz")
                bc.save_episode_data(masks, meta, out)
                n_render += 1
                n_dead += dead
                n_good += (1 - dead)
        if pi % 200 == 0:
            print(f"  [{pi}/{len(pkls)}] rendered={n_render} good={n_good} dead={n_dead} "
                  f"noeff={n_noeff} skip={n_skip}", file=sys.stderr, flush=True)
    print(f"DONE: rendered={n_render} (good={n_good} dead={n_dead}) noeff_filtered={n_noeff} "
          f"skipped={n_skip} -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
