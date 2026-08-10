#!/usr/bin/env python3
"""THE canonical separation + ranking diagnostic for the NAMO ranker (offline, exhaustive GT).

One tool, one set of definitions. Every "setup-vs-dead AUC" / "opener-vs-dead AUC" / setup-rank
number in the cards must come from here. Seven different measurements used to share the name "AUC";
the variant grammar and the retired numbers are in docs/experiments/auc_metrics_reconciliation.md.

Consolidates and replaces four drifted code paths:
  scripts/rl_loop/score_round2_eval.py           (V1 + rank, exact labels)      -> superseded
  scripts/sandbox/setup_value_check.py --agg     (Table 2, valid_first_push)    -> label-artifact, retired
  <scratch>/round1/analysis/compare_aucs.py      (vfp positives, tried-only neg) -> retired, never in git
  <scratch>/round3/eval/gt_model_errors/*.py     (D2/D5 + arch-inferring loader) -> loader promoted here

Runs on any box with a GPU and NO namo_rl / simulator dependency: exhaustive-GT H5s already carry
model-ready `ctx` (5,64,64) and `contact_px` (60,2), so this only needs torch + the checkpoint.

    python scripts/eval_auc.py --ckpt A.ckpt --ckpt B.ckpt --eval-set twopush_gt_h5 --out grid.json

Scores are cached per (checkpoint, eval-set) under $NAMO_SCRATCH/eval/auc_grid/cache/, so re-running
with more variants or more tiers costs nothing.

VARIANTS (all Mann-Whitney AUC; all reported per difficulty tier when the eval set has tiers)
  root separation      V1  root cells POOLED across boards : exact setup (0.9) vs exact dead (0.0)
                       V2  same masks, WITHIN board (mean of per-board AUCs)
  cross-board          V3  root board-max vs dead child board-max        (symmetric order statistic)
                       V4  best setup cell vs all reachable dead child CELLS   (cell vs cell)
                       V5  best setup cell vs dead child board-MAX       (1 draw vs best-of-~70)
                       V5m V5 restricted to moved (non-noop) dead boards
                       V6  live child board-max vs dead child board-max  (board-level live/dead)
  finish separation    F1  child cells POOLED : exact opener (1.0) vs exact dead (0.0)
                       F2  same masks, WITHIN board
RANK metrics (what best-first actually consumes) come with the hypergeometric random floor from
eval_common.floor_no_replacement -- the same floor every other NAMO eval uses.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "python", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_common import floor_no_replacement, mw_auc  # noqa: E402  the shared grading contract
from namo import eval_sets  # noqa: E402
from namo.paths import SCRATCH  # noqa: E402

SETUP, OPENER, DEAD = 0.9, 1.0, 0.0


# --------------------------------------------------------------------------- checkpoint -> network
def load_network(ckpt, device):
    """Load an EdgeCrossAttn scorer with its architecture INFERRED from the state dict.

    The training flags are not stored in a readable form on every checkpoint generation, so the
    shapes are the truth. Promoted from the round-3 GT diagnostic, where it was verified against
    live_scorer on arrakis; keeping it here is what lets this tool run without the simulator.
    """
    sys.path.insert(0, str(Path.home() / "robotics_research/ktamp/sage_learning"))
    from src.model.dit.edge_crossattn import EdgeCrossAttn
    from src.model.hl_gauss import HLGauss

    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    state = checkpoint["state_dict"]
    net_state = {k[len("network."):]: v for k, v in state.items() if k.startswith("network.")}
    dim = state["network.edge_norm.weight"].shape[0]
    pos_in = state["network.edge_pos.0.weight"].shape[1]
    pos_fourier = pos_in != 2
    kwargs = dict(
        img_size=64, patch=64 // int(round(state["network.scene_pos"].shape[1] ** 0.5)),
        in_channels=5, num_depths=5, dim=dim, heads=dim // 32,
        scene_depth=sum(1 for k in state if k.startswith("network.scene_blocks.") and k.endswith(".n1.weight")),
        edge_depth=sum(1 for k in state if k.startswith("network.edge_blocks.") and k.endswith(".n1.weight")),
        use_local="network.local_proj.weight" in state,
        fine_stem="network.fine_conv.weight" in state,
        use_edge_embed="network.edge_embed.weight" in state,
        edge_self_attn="network.edge_blocks.0.slf.in_proj_weight" in state,
    )
    if pos_fourier:
        kwargs.update(pos_fourier=True, fourier_L=pos_in // 4)
    if kwargs["fine_stem"]:
        kwargs["fine_stride"] = state["network.fine_conv.weight"].shape[-1]
    if "network.action_motion_proj.0.weight" in state:
        from namo.rl_loop.action_motion import action_motion_feature_dim
        motion_proj_in = state["network.action_motion_proj.0.weight"].shape[1]
        motion_tag = checkpoint.get("action_motion_encoding")
        motion_dim = action_motion_feature_dim(motion_tag) if motion_tag else motion_proj_in
        kwargs["action_motion_dim"] = motion_dim
        if motion_proj_in != motion_dim:
            kwargs.update(action_motion_fourier=True,
                          action_motion_fourier_L=motion_proj_in // (2 * motion_dim))
        if "network.action_depth_embed.weight" in state:
            kwargs["action_depth_embed"] = True
        if "network.action_depth_attn.attn.in_proj_weight" in state:
            kwargs["action_depth_self_attn"] = True
    head_out = state["network.head.2.weight"].shape[0]
    # head_out == num_depths -> RAW linear head (rank-pure linear, EXP-2026-08-09): scores ARE the
    # logits, no bins. The old inference (head_out//5 = 1 bin) built a 1-bin HL-Gauss whose value is
    # the constant 0.5 — shapes load, output is garbage. (No 1- or 5-bin model has ever existed.)
    if head_out == 5:
        value_bins = 0
    else:
        value_bins = head_out if kwargs.get("action_motion_dim", 0) else head_out // 5
    kwargs["value_bins"] = value_bins
    net = EdgeCrossAttn(**kwargs)
    from namo.rl_loop.action_motion import checkpoint_action_motion_encoding
    net.action_motion_encoding = checkpoint_action_motion_encoding(
        checkpoint, kwargs.get("action_motion_dim", 0))
    net.load_state_dict(net_state)
    if value_bins == 0:
        class _Raw:
            def value(self, x):
                return x
        return net.eval().to(device), _Raw()
    return net.eval().to(device), HLGauss(num_bins=value_bins)


def score_h5(ckpt, h5_path, device, batch=512):
    """Model value per cell, (N,60,5). Cached on disk keyed by (ckpt, h5) content identity."""
    key = hashlib.sha1(f"{Path(ckpt).resolve()}|{Path(h5_path).name}".encode()).hexdigest()[:16]
    cache = SCRATCH / "eval/auc_grid/cache" / f"{key}.npy"
    if cache.exists():
        return np.load(cache)
    net, hl = load_network(ckpt, device)
    with h5py.File(h5_path, "r") as data:
        n = data["ctx"].shape[0]
        values = np.zeros((n, 60, 5), dtype=np.float32)
        for start in range(0, n, batch):
            end = min(start + batch, n)
            ctx = torch.from_numpy(data["ctx"][start:end].astype(np.float32)).to(device)
            cpx = torch.from_numpy(data["contact_px"][start:end].astype(np.float32)).to(device)
            with torch.no_grad():
                action_motion = None
                if net.action_motion_dim > 0:
                    from namo.rl_loop.action_motion import action_motion_from_contact_px
                    action_motion = action_motion_from_contact_px(
                        cpx, encoding=net.action_motion_encoding,
                        feature_dim=net.action_motion_dim)
                values[start:end] = hl.value(
                    net(ctx, cpx, action_motion=action_motion).float()).cpu().numpy()
            print(f"  score {end}/{n}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, values)
    return values


# --------------------------------------------------------------------------- eval-set structure
def decode(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def load_structure(h5_path):
    """Board identity + labels, derived ONLY from fields both exhaustive-GT H5s carry.

    Episode key honours the multi_episode_rooms invariant: (xml, object_id, robot_goal) when the
    goal is recorded, else (xml, object_id) -- the key actually used is reported in the output.
    """
    with h5py.File(h5_path, "r") as data:
        kind = np.array([decode(x) for x in data["node_kind"][:]])
        xml = np.array([decode(x) for x in data["xml"][:]])
        obj = np.array([decode(x) for x in data["object_id"][:]])
        has_goal = "robot_goal" in data
        goal = [tuple(np.round(g, 4)) for g in data["robot_goal"][:]] if has_goal else [()] * len(kind)
        reach = data["r_mask"][:] > 0.5
        exact = (data["value_mask"][:] > 0.5) & reach
        target = data["value_target"][:]
    episodes = collections.defaultdict(list)
    for i, key in enumerate(zip(xml.tolist(), obj.tolist(), goal)):
        episodes[key].append(i)
    return dict(kind=kind, reach=reach, exact=exact, target=target, episodes=episodes,
                episode_key="(xml, object_id, robot_goal)" if has_goal else "(xml, object_id)")


def load_tiers(struct):
    """Per-episode difficulty from the canonical divisions file (the ONLY tier source)."""
    divisions = json.load(open(eval_sets.DIVISIONS))
    lookup = {(x, ep["object_id"]): ep["division"] for x, eps in divisions.items() for ep in eps}
    rename = {"easy": "easy", "medium": "med", "hard": "hard"}
    return {key: rename[lookup[(key[0], key[1])]] for key in struct["episodes"] if (key[0], key[1]) in lookup}


# --------------------------------------------------------------------------- metrics
def first_positive_rank(scores, positive, candidates):
    order = np.argsort(-scores[candidates], kind="stable")
    hits = np.flatnonzero(positive[candidates][order])
    return int(hits[0] + 1) if hits.size else None


def rank_block(ranks, pools):
    """Rank stats + the hypergeometric random floor over the SAME boards (eval_common contract)."""
    ranks = np.asarray([r for r in ranks if r is not None], float)
    if not ranks.size:
        return None
    out = {"n_boards": int(ranks.size),
           "hit_at_1_pct": round(100 * float(np.mean(ranks <= 1)), 1),
           "hit_at_5_pct": round(100 * float(np.mean(ranks <= 5)), 1),
           "median_first_rank": float(np.median(ranks)),
           "mean_first_rank": round(float(ranks.mean()), 2),
           "p90_first_rank": float(np.percentile(ranks, 90))}
    for k in (1, 5):
        out[f"floor_at_{k}_pct"] = round(100 * float(np.mean([floor_no_replacement(f, r, k) for f, r in pools])), 1)
    return out


def analyse(struct, values, episode_keys):
    """All variants over one set of episodes (one tier, or all)."""
    kind, reach, exact, target = struct["kind"], struct["reach"], struct["exact"], struct["target"]
    root_pos, root_neg, root_within = [], [], []
    setup_cell, root_bmax = [], []
    live_bmax, dead_bmax, dead_moved_bmax, dead_cells = [], [], [], []
    open_pos, open_neg, open_within = [], [], []
    setup_ranks, setup_pools, open_ranks, open_pools = [], [], [], []
    n_boards = collections.Counter()

    for key in episode_keys:
        rows = struct["episodes"][key]
        roots = [i for i in rows if kind[i] == "root"]
        children = [i for i in rows if kind[i] != "root" and reach[i].any()]
        for i in roots:
            positive, negative = exact[i] & np.isclose(target[i], SETUP), exact[i] & np.isclose(target[i], DEAD)
            n_boards["root"] += 1
            if positive.any():
                root_pos.append(values[i][positive])
                setup_cell.append(float(values[i][positive].max()))
                root_bmax.append(float(values[i][reach[i]].max()))
            if negative.any():
                root_neg.append(values[i][negative])
            if positive.any() and negative.any():
                root_within.append(mw_auc(values[i][positive], values[i][negative]))
                setup_ranks.append(first_positive_rank(values[i], positive, positive | negative))
                setup_pools.append((int(positive.sum()), int((positive | negative).sum())))
        for i in children:
            positive, negative = exact[i] & np.isclose(target[i], OPENER), exact[i] & np.isclose(target[i], DEAD)
            live = bool(positive.any())
            n_boards["live" if live else "dead"] += 1
            board_max = float(values[i][reach[i]].max())
            if live:
                live_bmax.append(board_max)
                open_pos.append(values[i][positive])
            else:
                dead_bmax.append(board_max)
                dead_cells.append(values[i][reach[i]])
                if kind[i] != "depth2_noop":
                    dead_moved_bmax.append(board_max)
            if negative.any():
                open_neg.append(values[i][negative])
            if live and negative.any():
                open_within.append(mw_auc(values[i][positive], values[i][negative]))
                open_ranks.append(first_positive_rank(values[i], positive, positive | negative))
                open_pools.append((int(positive.sum()), int((positive | negative).sum())))

    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    mean = lambda xs: round(float(np.mean(xs)), 4) if xs else None
    return {
        "n_episodes": len(episode_keys),
        "n_boards": dict(n_boards),
        "separation_root": {"V1_pooled": mw_auc(cat(root_pos), cat(root_neg)), "V2_within_board": mean(root_within)},
        "separation_finish": {"F1_pooled": mw_auc(cat(open_pos), cat(open_neg)), "F2_within_board": mean(open_within)},
        "cross_board": {
            "V3_rootmax_vs_deadmax": mw_auc(root_bmax, dead_bmax),
            "V4_setupcell_vs_deadcells": mw_auc(setup_cell, cat(dead_cells)),
            "V5_setupcell_vs_deadmax": mw_auc(setup_cell, dead_bmax),
            "V5m_setupcell_vs_moved_deadmax": mw_auc(setup_cell, dead_moved_bmax),
            "V6_livemax_vs_deadmax": mw_auc(live_bmax, dead_bmax),
            "median_cells_per_dead_board": int(np.median([len(c) for c in dead_cells])) if dead_cells else None,
        },
        "rank_setup": rank_block(setup_ranks, setup_pools),
        "rank_finish": rank_block(open_ranks, open_pools),
    }


# --------------------------------------------------------------------------- driver
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", required=True, help="repeatable; label:path or path")
    parser.add_argument("--eval-set", default="twopush_gt_h5", help="name in config/eval_sets.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    h5_path = eval_sets.path(args.eval_set)
    struct = load_structure(h5_path)
    tiers = load_tiers(struct)
    groups = {"all": list(struct["episodes"])}
    for tier in ("easy", "med", "hard"):
        keys = [k for k, t in tiers.items() if t == tier]
        if keys:
            groups[tier] = keys
    if len(groups) > 1:
        groups["all_tiered"] = [k for k in struct["episodes"] if k in tiers]

    models = {}
    for spec in args.ckpt:
        label, _, path = spec.rpartition(":") if ":" in spec else ("", "", spec)
        label = label or Path(path).parts[-3]
        print(f"[{label}] {path}", flush=True)
        values = score_h5(path, h5_path, args.device)
        models[label] = {"ckpt": str(path), **{g: analyse(struct, values, keys) for g, keys in groups.items()}}

    result = {"eval_set": args.eval_set, "h5": str(h5_path), "episode_key": struct["episode_key"],
              "n_episodes_total": len(struct["episodes"]), "n_episodes_tiered": len(tiers), "models": models}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
