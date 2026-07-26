"""Recompute every 'setup-vs-dead AUC' variant from ONE cached score file.

Model  : round3/models/d20_plus_setup_only_splitloss/epoch011 (the 2026-07-24 deploy ckpt)
Scores : gt_model_errors/values.npy  (66,456 x 60 x 5)
Eval   : round2/h5/testset_gt.h5     (exhaustive root+finish GT on the canonical testset)

Variants (all Mann-Whitney AUC, all per tier):
  V1  root cell-level, POOLED across boards : setup cells (value_target 0.9, exact) vs root dead cells (0.0, exact)
  V2  root cell-level, WITHIN board (mean of per-board AUCs), same masks as V1
  V3  cross-board, symmetric order stat     : root board-max vs dead post-push board-max
  V4  cross-board, cell vs cell             : best true setup cell vs ALL dead post-push cells (reachable)
  V5  cross-board, as reported (D5)         : best true setup cell vs dead post-push board-MAX
  V5m V5 restricted to moved (non-noop) dead boards  -> the card's 0.583
  V6  board-level live vs dead (D2)         : live post-push board-max vs dead post-push board-max
"""
import json
import collections
from collections import defaultdict

import h5py
import numpy as np

H5 = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/testset_gt.h5"
VALS = "/common/users/dm1487/scratch_namo/curriculum2/beast/round3/eval/gt_model_errors/values.npy"
DIV = "/common/users/dm1487/scratch_namo/datasets/namo_testset_v1/labels/pure2push_divisions.json"

vals = np.load(VALS)
f = h5py.File(H5, "r")
dec = lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
nk = np.array([dec(x) for x in f["node_kind"][:]])
xml = np.array([dec(x) for x in f["xml"][:]])
oid = np.array([dec(x) for x in f["object_id"][:]])
rg = f["robot_goal"][:]
n_win = f["n_win"][:]
cd = f["chain_depth"][:]
pdep = f["parent_depth"][:]
pedge = f["parent_edge"][:]
setup_moved = f["setup_moved"][:]
rmask = f["r_mask"][:] > 0.5
vmask = f["value_mask"][:] > 0.5
vtgt = f["value_target"][:]
f.close()

exact = vmask & rmask

epi_idx = defaultdict(list)
for i, e in enumerate(zip(xml.tolist(), oid.tolist(), [tuple(np.round(r, 4)) for r in rg])):
    epi_idx[e].append(i)

div = json.load(open(DIV))
div_lookup = {(x, ep["object_id"]): ep for x, eps in div.items() for ep in eps}
tiermap = {"easy": "easy", "medium": "med", "hard": "hard"}

EPS = []
for e, idxs in epi_idx.items():
    ridx = [i for i in idxs if nk[i] == "root"][0]
    ep = div_lookup.get((e[0], e[1]))
    if ep is None:
        continue
    kids = [i for i in idxs if i != ridx]
    setups = {(int(pedge[k]), int(pdep[k])): k for k in kids if n_win[k] > 0}
    EPS.append(dict(root=ridx, tier=tiermap.get(ep["division"]), setups=setups, kids=kids))

TIERS = ["easy", "med", "hard", "all"]
in_tier = lambda t, ep: t == "all" or ep["tier"] == t


def mw_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not pos.size or not neg.size:
        return float("nan")
    v = np.concatenate([pos, neg])
    ranks = v.argsort().argsort().astype(float) + 1
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


out = {}
for t in TIERS:
    eps = [ep for ep in EPS if in_tier(t, ep)]
    kids = [k for ep in eps for k in ep["kids"] if cd[k] == 2 and rmask[k].any()]

    # ---- V1 / V2: root cell-level, exact labels (the score_round2_eval.py definition)
    pool_pos, pool_neg, per_board = [], [], []
    for ep in eps:
        r = ep["root"]
        pos = exact[r] & np.isclose(vtgt[r], 0.9)
        neg = exact[r] & np.isclose(vtgt[r], 0.0)
        if pos.any():
            pool_pos.append(vals[r][pos])
        if neg.any():
            pool_neg.append(vals[r][neg])
        if pos.any() and neg.any():
            per_board.append(mw_auc(vals[r][pos], vals[r][neg]))
    v1 = mw_auc(np.concatenate(pool_pos), np.concatenate(pool_neg)) if pool_pos and pool_neg else float("nan")
    v2 = float(np.mean(per_board)) if per_board else float("nan")

    # ---- setup positives used by the cross-board variants
    setup_cell = np.array([max(vals[ep["root"]][o, d] for (o, d) in ep["setups"]) for ep in eps if ep["setups"]])
    root_bmax = np.array([float(vals[ep["root"]][rmask[ep["root"]]].max()) for ep in eps if ep["setups"]])

    dead = [k for k in kids if n_win[k] == 0]
    live = [k for k in kids if n_win[k] > 0]
    dead_bmax = np.array([float(vals[k][rmask[k]].max()) for k in dead])
    live_bmax = np.array([float(vals[k][rmask[k]].max()) for k in live])
    dead_moved_bmax = np.array([float(vals[k][rmask[k]].max()) for k in dead if setup_moved[k] == 1])
    dead_cells = np.concatenate([vals[k][rmask[k]] for k in dead]) if dead else np.array([])

    out[t] = {
        "n_episodes": len(eps),
        "n_dead_boards": len(dead),
        "n_live_boards": len(live),
        "V1_root_cell_pooled": round(v1, 4),
        "V2_root_cell_within_board": round(v2, 4),
        "V3_rootmax_vs_deadmax": round(mw_auc(root_bmax, dead_bmax), 4),
        "V4_setupcell_vs_deadcells": round(mw_auc(setup_cell, dead_cells), 4),
        "V5_setupcell_vs_deadmax": round(mw_auc(setup_cell, dead_bmax), 4),
        "V5m_setupcell_vs_movedddeadmax": round(mw_auc(setup_cell, dead_moved_bmax), 4),
        "V6_livemax_vs_deadmax": round(mw_auc(live_bmax, dead_bmax), 4),
        "median_dead_board_cells": int(np.median([int(rmask[k].sum()) for k in dead])) if dead else None,
    }

print(json.dumps(out, indent=2))
json.dump(out, open("/common/users/dm1487/scratch_namo/tmp/claude-89862/auc_reconcile.json", "w"), indent=2)
