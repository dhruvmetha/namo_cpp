#!/usr/bin/env python3
"""Replay-render DEAD post-push states (horizon-Q v2, the post-push OOD fix).

⛔ BLOCKED 2026-06-13: replay reproduces FREE-SPACE pushes exactly but DIVERGES on COLLISION pushes
(47/47 diverge vs 298/298 free-space match). Dead s1 with collisions != the collector's labeled s1.
USE COLLECTION-TIME STATE-SAVING + re-collection instead (journal §9). Kept for the free-space-valid
replay+render logic, which is reusable.

The H2 collection VISITED every post-push state s1 = T(s0, a1) but only persisted the SOLUTION-path
states; dead-branch s1's were discarded (only their a2 outcomes survive in the trial log). Those dead
post-push states are exactly what the search-leaf / reactive-second-step OOD needs (the 0.549 dead-leaf
calibration). They're RECOVERABLE by deterministic replay: load XML->s0, env.step(a1) reproduces the
collector's s1 (verified), render it.

Per pkl (one scene's episodes): for each episode, find EXPANDED-DEAD first pushes (a1 that were tried at
depth-2 and whose every a2 child failed), sample `--per-episode` of them, replay each, render s1 as a
dead-end H=1 row anchored at the post-push pose. The npz self-carries the label info (pp_tried_ed/dp =
the ~48 tried a2 cells, all 0 -> the all-zero f_grid), so build_postpush_h5.py packs it with NO matching.

  python scripts/pipeline/replay_postpush.py --pkl-list shard.txt --output-dir out/ --per-episode 1
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

# env + renderer (need MJ_PATH + build_python on PYTHONPATH; the SLURM driver sets them)
import namo_rl
import namo.visualization.mask_generation.batch_collection as bc
from sage_learning.visualizer import NAMODataVisualizer

CFG = "config/namo_config_complete_skill15_car_1x.yaml"


def dead_first_pushes(trial_log):
    """expanded-dead a1 -> list of its tried a2 cells (all failed). Skips a1 with any opening a2."""
    kids = defaultdict(dict)
    for t in trial_log:
        if t.get("chain_depth") == 2:
            pe, pd = t.get("parent_edge"), t.get("parent_depth")
            if pe is None:
                continue
            kids[(pe, pd)][(t["edge_idx"], t["depth"])] = bool(t.get("success"))
    return {a1: list(ch) for a1, ch in kids.items() if ch and not any(ch.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-list", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--per-episode", type=int, default=1, help="dead a1's to replay per episode (diversity)")
    ap.add_argument("--cfg", default=CFG)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import pickle, random
    rng = random.Random(a.seed)

    pkls = [l.strip() for l in open(a.pkl_list) if l.strip()]
    viz = NAMODataVisualizer(figsize=(10, 8), namo_config_path=a.cfg)
    n_render = 0
    for pi, pkl in enumerate(pkls):
        try:
            d = pickle.load(open(pkl, "rb"))
        except Exception:
            continue
        # one env per scene (episodes share the xml); reset + step per a1
        env = None
        for er in d.get("episode_results") or []:
            st = er.get("algorithm_stats") or {}
            obj = st.get("chosen_object_id")
            xml = er.get("xml_file")
            log = st.get("primitive_trial_log") or []
            if not obj or not xml:
                continue
            dead = dead_first_pushes(log)
            if not dead:
                continue
            picks = rng.sample(list(dead), min(a.per_episode, len(dead)))
            rg = st.get("region_goals_sampled") or er.get("region_goals_sampled")
            region = st.get("neighbour_region_label")
            for (e1, d1) in picks:
                try:
                    if env is None:
                        env = namo_rl.RLEnvironment(xml, a.cfg, visualize=False)
                    env.reset()
                    act = namo_rl.Action(); act.object_id = obj; act.edge_idx = int(e1); act.depth = int(d1)
                    env.step(act)
                    obs1 = env.get_observation(); soi = env.get_object_info()
                    ep = {
                        "episode_id": f"{er.get('episode_id','ep')}_ppdead_{e1}_{d1}",
                        "solution_found": False, "state_observations": [obs1],
                        "post_action_state_observations": [], "action_sequence": [{"object_id": obj}],
                        "static_object_info": soi, "robot_goal": er.get("robot_goal", [0, 0, 0]),
                        "xml_file": xml, "reachable_objects_before_action": None,
                        "algorithm_stats": {"chosen_object_id": obj, "neighbour_region_label": region,
                                            "region_goals_sampled": rg,
                                            "primitive_trial_log": [{"edge_idx": e, "depth": dd, "success": False,
                                                                     "chain_depth": 1} for (e, dd) in dead[(e1, d1)]]},
                    }
                    masks, meta = bc.process_episode(ep, viz, generate_local=True, local_only=False)
                    if not masks:
                        continue
                    # self-carry the dead-label: the ~48 tried a2 cells (all 0 -> all-zero f_grid)
                    tried = np.array(dead[(e1, d1)], dtype=np.int16)  # (n,2): edge,depth
                    meta["pp_dead"] = 1
                    meta["pp_tried_ed"] = tried[:, 0] if len(tried) else np.zeros(0, np.int16)
                    meta["pp_tried_dp"] = tried[:, 1] if len(tried) else np.zeros(0, np.int16)
                    out = os.path.join(a.output_dir, meta["task_id"] or "t", f"{ep['episode_id']}.npz")
                    bc.save_episode_data(masks, meta, out)
                    n_render += 1
                except Exception:
                    continue
        if pi % 200 == 0:
            print(f"  [{pi}/{len(pkls)}] rendered={n_render}", file=sys.stderr, flush=True)
    print(f"DONE: {n_render} dead post-push npz -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
