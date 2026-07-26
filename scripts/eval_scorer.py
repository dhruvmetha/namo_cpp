#!/usr/bin/env python3
"""Evaluate the 1-push SCORER on the SAME test episodes as the diffusion baseline — apples-to-apples.

Method (deterministic, exact — no sampling):
  For each test episode (matched per-episode by object_center, re-binned by true solve_rate):
    scores = sigmoid(scorer(5 masks))            # (60,5) P(success) per (edge,depth)
    rank the REACHABLE candidates by score; success@k = any of the top-k is valid.
  Two reachability masks (mirrors eval_classifier):
    - realistic: contact-point level (if an edge is reachable, all its depths are candidates) — what is
      known at deploy; a top-k pick on a depth-blocked cell is a wasted attempt (counts as non-hit).
    - oracle: exact per-(edge,depth) reachability (best case).
  Bin by true difficulty (ratio = |valid|/|reachable|): hard<0.05, med<0.30, else easy.
  Floor = random, WITHOUT replacement (hypergeometric). Compare to the diffusion's corrected
  reachable numbers (hard 5.9/55.2 @1/@20).

Full diagnostic panel per bin (one run, all from the same forward pass — the single source of
"why or why not it works"; supersedes the depth_hist.py / wrong_edge one-offs):
  - success@k, oracle@k, floor@k                  (the WHAT)
  - edge@k          right edge in top-k, any depth (the edge ceiling 2-push search exploits)
  - failure_decomp@1  success / right_edge_wrong_depth / wrong_edge  (not_reachable=0 by construction)
  - wrong_edge       overall % and as-share-of-misses %
  - depth_top1_hist  d0..d4 distribution of the top-1 pick + depth-acc given a right edge
  - rank_first_valid median / %≤3 / %≤10 / %≤20 of the first valid push in the ranked pool
  - score_separation mean(valid)-mean(invalid) margin + % scenes with positive margin
Existing JSON keys (scorer_realistic/scorer_oracle/floor) are unchanged so resolve_robust.sh still parses.
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import MASKS, OUT, match_episode, bin_of, floor_no_replacement, mw_auc  # shared grading contract
from namo.paths import DATASETS, H5  # noqa: E402
from namo import eval_sets  # noqa: E402


def contact_px(edge, hw, hd, theta, crop_m, S=64):
    """60 edge contact-pixel coords in the SxS crop frame (same convention as add_contact_px.py)."""
    n = 15
    sl = lambda a, b, i: a + (b - a) * (i / (n - 1))
    if edge < 30:
        j = edge // 2; lx = sl(-hw, hw, j); ly = hd if edge % 2 == 0 else -hd
    else:
        k = (edge - 30) // 2; lx = hw if edge % 2 == 0 else -hw; ly = sl(-hd, hd, k)
    c, s = math.cos(theta), math.sin(theta)
    wx = c * lx - s * ly; wy = s * lx + c * ly
    res = crop_m / S; cx = S / 2.0
    return cx + wx / res, cx + wy / res


def load_scorer(ckpt, num_depths, device, network="dit_classifier"):
    import inspect
    _sage = os.environ.get("SAGE_REPO", "")
    if _sage and _sage not in sys.path:
        sys.path.insert(0, _sage)
    from src.model.classifier_module import ClassifierModule
    # weights_only=False: our own trusted ckpt; PyTorch 2.6 default rejects the numpy scalar in hparams
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    sd = ck["state_dict"]
    if network == "edge_crossattn":
        from src.model.dit.edge_crossattn import EdgeCrossAttn
        # infer arch from the checkpoint (so any dim/depth variant loads): dim from edge_norm,
        # depths from block counts, heads = dim//32 (our convention: 192->6, 256->8).
        dim = sd["network.edge_norm.weight"].shape[0]
        sdep = sum(1 for k in sd if k.startswith("network.scene_blocks.") and k.endswith(".n1.weight"))
        edep = sum(1 for k in sd if k.startswith("network.edge_blocks.") and k.endswith(".n1.weight"))
        patch = 64 // int(round(sd["network.scene_pos"].shape[1] ** 0.5))  # infer patch from #scene tokens
        kw = dict(img_size=64, patch=patch, in_channels=5, num_depths=num_depths,
                  dim=dim, scene_depth=sdep, edge_depth=edep, heads=dim // 32)
        if "network.zoom_pos" in sd:   # dual-crop (use_zoom) ckpt — infer zoom arch + turn it on
            zgrid = int(round(sd["network.zoom_pos"].shape[1] ** 0.5))
            zpatch = sd["network.zoom_patch.proj.weight"].shape[-1]
            zdep = sum(1 for k in sd if k.startswith("network.zoom_blocks.") and k.endswith(".n1.weight"))
            kw.update(use_zoom=True, zoom_size=zgrid * zpatch, zoom_patch=zpatch, zoom_depth=zdep)
        if "network.local_proj.weight" not in sd:   # NO-GATHER ablation ckpt (coord + cross-attn only)
            kw["use_local"] = False
        pin = sd["network.edge_pos.0.weight"].shape[1]   # 2=raw coord, 4L=Fourier
        if pin != 2:
            kw.update(pos_fourier=True, fourier_L=pin // 4)
        if "network.edge_embed.weight" in sd:            # per-edge embedding
            kw["use_edge_embed"] = True
        if "network.fine_conv.weight" in sd:             # de-aliased fine-stem gather
            kw["fine_stem"] = True
            kw["fine_stride"] = sd["network.fine_conv.weight"].shape[-1]
        if "network.edge_blocks.0.slf.in_proj_weight" not in sd:   # H2 ablation: no inter-edge self-attn
            kw["edge_self_attn"] = False
        if "network.budget_embed.weight" in sd:          # budget-conditioned horizon-Q ckpt
            kw["budget_cond"] = True
            kw["max_budget"] = sd["network.budget_embed.weight"].shape[0] - 1
        if "network.reach_embed.weight" in sd:           # M2d: per-edge reachability input flag
            kw["reach_flag_input"] = True
        if "network.action_motion_proj.0.weight" in sd:
            from namo.rl_loop.action_motion import action_motion_feature_dim
            motion_proj_in = sd["network.action_motion_proj.0.weight"].shape[1]
            motion_tag = ck.get("action_motion_encoding")
            motion_dim = action_motion_feature_dim(motion_tag) if motion_tag else motion_proj_in
            kw["action_motion_dim"] = motion_dim
            if motion_proj_in != motion_dim:
                denom = 2 * motion_dim
                if motion_proj_in % denom:
                    raise ValueError(f"invalid Fourier motion width {motion_proj_in} for dim {motion_dim}")
                kw.update(action_motion_fourier=True, action_motion_fourier_L=motion_proj_in // denom)
            if "network.action_depth_embed.weight" in sd:
                kw["action_depth_embed"] = True
        head_out = sd["network.head.2.weight"].shape[0]
        if head_out != num_depths:                       # HL-Gauss value head
            kw["value_bins"] = head_out if kw.get("action_motion_dim", 0) else head_out // num_depths
        net = EdgeCrossAttn(**kw)
        from namo.rl_loop.action_motion import checkpoint_action_motion_encoding
        net.action_motion_encoding = checkpoint_action_motion_encoding(
            ck, kw.get("action_motion_dim", 0))
    else:
        from src.model.dit.dit_classifier import DiTClassifier
        net = DiTClassifier(img_size=64, in_channels=5, num_depths=num_depths)
    hp = ck.get("hyper_parameters", {})
    sig = set(inspect.signature(ClassifierModule.__init__).parameters) - {"self", "network"}
    model = ClassifierModule(network=net, **{k: v for k, v in hp.items() if k in sig})
    model.load_state_dict(sd)
    model.eval().to(device)
    return model


def topk_hit(scores, valid_cells, cand_idx, ks):
    """scores (60,5); valid_cells set of (e,d); cand_idx list of flat candidate indices (reachable).
    Returns {k: hit bool} = any of top-k ranked candidates is valid."""
    flat = scores.reshape(-1)
    order = sorted(cand_idx, key=lambda j: -flat[j])
    out = {}
    for k in ks:
        top = order[:k]
        out[k] = any((j // 5, j % 5) in valid_cells for j in top)
    return out


KS = [1, 5, 10, 20]


def diagnostic_record(scores, valid, tried, reach_cp=None, identity=None):
    """Grade one score grid against one episode's exact valid/tried action sets.

    ``reach_cp`` is the deployment-realistic candidate pool. Legacy H5 evaluation derives it from
    contact-level reachability; live evaluation passes the exact primitive pool produced by the current
    deployment code. ``tried`` remains the oracle per-cell reachability pool from the answer key.
    """
    valid = {tuple(t) for t in valid}
    tried = {tuple(t) for t in tried}
    if reach_cp is None:
        reach_cp = [ee * 5 + dd for ee in {x for x, _ in tried} for dd in range(5)]
    reach_cp = sorted(set(int(j) for j in reach_cp))
    reach_exact = sorted(ee * 5 + dd for (ee, dd) in tried if dd < 5 and ee * 5 + dd in reach_cp)
    if not reach_cp:
        raise ValueError("episode has no deployment-realistic candidate pushes")

    hit_o = topk_hit(scores, valid, reach_exact, KS)
    hit_r = topk_hit(scores, valid, reach_cp, KS)
    flat = scores.reshape(-1)
    order = sorted(reach_cp, key=lambda j: -flat[j])
    valid_flat = {ee * 5 + dd for (ee, dd) in valid}
    solving_edges = {ee for (ee, _) in valid}
    best = order[0]
    be, bd = best // 5, best % 5
    if best in valid_flat:
        cat = "success"
    elif be in solving_edges:
        cat = "right_edge_wrong_depth"
    else:
        cat = "wrong_edge"
    edge_hit = {k: any((j // 5) in solving_edges for j in order[:k]) for k in KS}
    rank_fv = next((idx + 1 for idx, j in enumerate(order) if j in valid_flat), None)
    depth_right = (bd in {dd for (ee, dd) in valid if ee == be}) if be in solving_edges else None
    sc = np.array([flat[j] for j in reach_cp])
    vm = np.array([1 if j in valid_flat else 0 for j in reach_cp])
    sep = float(sc[vm == 1].mean() - sc[vm == 0].mean()) \
        if (vm.sum() > 0 and (len(sc) - vm.sum()) > 0) else None
    # opener-vs-dead separation at horizon 1 — the 1-push half of the AUC panel that
    # scripts/eval_auc.py reports for 2-push. Same variant grammar (pooled vs within-board):
    # see docs/experiments/auc_metrics_reconciliation.md. Raw score piles kept so the
    # cross-episode POOLED AUC can be formed in aggregate_records.
    pos_scores = sc[vm == 1].tolist()
    neg_scores = sc[vm == 0].tolist()
    candidate_cells = {(j // 5, j % 5) for j in reach_cp}
    rec = {
        "nF": len(valid_flat & set(reach_cp)), "nR": len(reach_cp),
        "hit_r": hit_r, "hit_o": hit_o, "edge_hit": edge_hit, "cat": cat,
        "top1_edge": be, "top1_depth": bd, "top1_score": float(flat[best]),
        "depth_right": depth_right, "rank_fv": rank_fv, "sep": sep,
        "pos_scores": pos_scores, "neg_scores": neg_scores,
        "n_valid_total": len(valid), "n_tried_total": len(tried),
        "n_valid_missing_from_pool": len(valid - candidate_cells),
    }
    if identity:
        rec.update(identity)
    return rec


def aggregate_records(records):
    """Aggregate the shared scorer diagnostic panel by true per-episode difficulty."""
    out = {}
    for b in ("hard", "med", "easy"):
        rows = [r for r in records if bin_of(r["sr"]) == b]
        n = len(rows)
        if not n:
            out[b] = {"n": 0}
            continue
        real = {f"@{k}": round(np.mean([r["hit_r"][k] for r in rows]) * 100, 1) for k in KS}
        orac = {f"@{k}": round(np.mean([r["hit_o"][k] for r in rows]) * 100, 1) for k in KS}
        floor = {f"@{k}": round(np.mean([floor_no_replacement(r["nF"], r["nR"], k)
                                          for r in rows]) * 100, 1) for k in KS}
        edge = {f"@{k}": round(np.mean([r["edge_hit"][k] for r in rows]) * 100, 1) for k in KS}
        decomp = {c: round(np.mean([r["cat"] == c for r in rows]) * 100, 1)
                  for c in ("success", "right_edge_wrong_depth", "wrong_edge")}
        decomp["not_reachable"] = 0.0
        n_miss = sum(1 for r in rows if r["cat"] != "success")
        we_overall = round(np.mean([r["cat"] == "wrong_edge" for r in rows]) * 100, 1)
        we_of_miss = round(sum(r["cat"] == "wrong_edge" for r in rows) / n_miss * 100, 1) if n_miss else 0.0
        dh = [0] * 5
        for r in rows:
            dh[r["top1_depth"]] += 1
        depth_hist = {f"d{i}": round(100 * dh[i] / n, 1) for i in range(5)}
        dr = [r["depth_right"] for r in rows if r["depth_right"] is not None]
        depth_acc = round(np.mean(dr) * 100, 1) if dr else None
        ranks = [r["rank_fv"] for r in rows if r["rank_fv"] is not None]
        rank_stats = {"median": float(np.median(ranks)) if ranks else None,
                      "pct_le3": round(np.mean([x <= 3 for x in ranks]) * 100, 1) if ranks else None,
                      "pct_le10": round(np.mean([x <= 10 for x in ranks]) * 100, 1) if ranks else None,
                      "pct_le20": round(np.mean([x <= 20 for x in ranks]) * 100, 1) if ranks else None,
                      "n_with_valid_in_pool": len(ranks)}
        seps = [r["sep"] for r in rows if r["sep"] is not None]
        sep_stats = {"mean_margin": round(float(np.mean(seps)), 3) if seps else None,
                     "pct_positive": round(np.mean([s > 0 for s in seps]) * 100, 1) if seps else None}
        graded = [r for r in rows if r["pos_scores"] and r["neg_scores"]]
        sep_stats["auc_pooled"] = mw_auc([s for r in graded for s in r["pos_scores"]],
                                         [s for r in graded for s in r["neg_scores"]])
        within = [mw_auc(r["pos_scores"], r["neg_scores"]) for r in graded]
        sep_stats["auc_within_episode"] = round(float(np.mean(within)), 4) if within else None
        sep_stats["auc_n_episodes"] = len(graded)
        out[b] = {
            "n": n, "sr_mean_pct": round(np.mean([r["sr"] for r in rows]) * 100, 2),
            "scorer_realistic": real, "scorer_oracle": orac, "floor": floor,
            "edge_at_k": edge, "failure_decomp_at1_pct": decomp,
            "wrong_edge_overall_pct": we_overall, "wrong_edge_of_misses_pct": we_of_miss,
            "depth_top1_hist_pct": depth_hist, "depth_acc_given_right_edge_pct": depth_acc,
            "rank_first_valid": rank_stats, "score_separation": sep_stats,
            "valid_missing_from_pool": int(sum(r["n_valid_missing_from_pool"] for r in rows)),
        }
    return out


def write_result(ckpt, records, out_path, mode):
    res = {"ckpt": ckpt, "mode": mode, "n_episodes": len(records),
           "divisions": aggregate_records(records)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    for b in ("hard", "med", "easy"):
        r = res["divisions"][b]
        if not r.get("n"):
            continue
        sk = r["scorer_realistic"]; fl = r["floor"]; ek = r["edge_at_k"]
        fd = r["failure_decomp_at1_pct"]; dh = r["depth_top1_hist_pct"]
        rk = r["rank_first_valid"]; ss = r["score_separation"]
        print(f"\n===== [{b}]  n={r['n']}  sr_mean={r['sr_mean_pct']}% =====", file=sys.stderr)
        print(f"  success@k   1/5/10/20 : {sk['@1']}/{sk['@5']}/{sk['@10']}/{sk['@20']}", file=sys.stderr)
        print(f"  edge@k      1/5/10/20 : {ek['@1']}/{ek['@5']}/{ek['@10']}/{ek['@20']}   (right edge, any depth)", file=sys.stderr)
        print(f"  floor@k     1/5/10/20 : {fl['@1']}/{fl['@5']}/{fl['@10']}/{fl['@20']}   (random, no-replacement)", file=sys.stderr)
        print(f"  fail@1   succ/rEwD/wrongE : {fd['success']}/{fd['right_edge_wrong_depth']}/{fd['wrong_edge']}%", file=sys.stderr)
        print(f"  wrong-edge  overall={r['wrong_edge_overall_pct']}%  of-misses={r['wrong_edge_of_misses_pct']}%", file=sys.stderr)
        print(f"  depth top1  d0..d4 : {dh['d0']}/{dh['d1']}/{dh['d2']}/{dh['d3']}/{dh['d4']}%   depth-acc|rightEdge={r['depth_acc_given_right_edge_pct']}%", file=sys.stderr)
        print(f"  rank-1st-valid  median={rk['median']}  <=3={rk['pct_le3']}%  <=10={rk['pct_le10']}%  <=20={rk['pct_le20']}%", file=sys.stderr)
        print(f"  score-sep   margin={ss['mean_margin']}  positive={ss['pct_positive']}%", file=sys.stderr)
        print(f"  valid cells missing from live pool: {r['valid_missing_from_pool']}", file=sys.stderr)
    print(f"\nwrote {out_path}", file=sys.stderr)
    return res


def live_canonical_records(ckpt, episodes_path, start, end, leaf_out):
    """Score canonical one-push episodes through the current live deployment renderer, with zero pushes."""
    from namo.core.xml_goal_parser import extract_goal_with_fallback
    from namo.paths import resolve
    from scorer_beam import BeamPlanner, FALLBACK_GOAL, make_env
    from eval_m3 import rank_first_pushes_h2

    cv2.setNumThreads(1)
    key = json.load(open(episodes_path))
    episodes = [(xml, rec_idx, rec) for xml in sorted(key)
                for rec_idx, rec in enumerate(key[xml])]
    identities = [(xml, rec["object_id"], rec.get("region"),
                   round(float(rec["object_center"][0]), 4), round(float(rec["object_center"][1]), 4))
                  for xml, _rec_idx, rec in episodes]
    if len(set(identities)) != len(identities):
        raise RuntimeError(f"duplicate episode identities: {len(identities) - len(set(identities))}")
    stop = len(episodes) if end < 0 else min(end, len(episodes))
    selected = episodes[start:stop]
    if not selected:
        raise ValueError(f"empty live episode slice [{start}:{stop}] of {len(episodes)}")

    planner = BeamPlanner(ckpt=ckpt)
    records = []
    fout = open(leaf_out, "w") if leaf_out else None
    t0 = time.perf_counter()
    for local_i, (xml, rec_idx, gt) in enumerate(selected):
        global_i = start + local_i
        xp = str(resolve(xml))
        env = make_env(xp)
        goal = extract_goal_with_fallback(xp, FALLBACK_GOAL)
        env.set_robot_goal(*goal)
        env.get_reachable_objects()
        obs = env.get_observation()
        obj = gt["object_id"]
        pose_key = f"{obj}_pose"
        if pose_key not in obs:
            raise RuntimeError(f"episode {global_i}: {obj} absent from live observation for {xp}")
        center = gt["object_center"]
        center_err = math.hypot(float(obs[pose_key][0]) - float(center[0]),
                                float(obs[pose_key][1]) - float(center[1]))
        if center_err > 0.01:
            raise RuntimeError(f"episode {global_i}: object-center mismatch {center_err:.6f} m for {obj}")
        s0 = env.get_full_state()
        pool = rank_first_pushes_h2(planner, env, goal, xp, s0, 1,
                                    restrict_obj=obj, score=True, raw=True)
        scores = np.full((60, 5), -1e9, dtype=np.float32)
        cells = []
        for pool_obj, g, q in pool:
            if pool_obj != obj:
                raise RuntimeError(f"episode {global_i}: candidate escaped object restriction")
            e, d = int(g.edge_idx), int(g.depth)
            if (e, d) in cells:
                raise RuntimeError(f"episode {global_i}: duplicate live action {(e, d)}")
            cells.append((e, d))
            scores[e, d] = float(q)
        identity = {
            "i": global_i, "xml": xml, "object_id": obj, "region": gt.get("region"),
            "object_center": [float(center[0]), float(center[1])], "object_center_error_m": center_err,
            "sr": min(1.0, float(gt.get("solve_rate", len(gt["valid"]) / max(1, len(gt["tried"]))))),
        }
        rec = diagnostic_record(scores, gt["valid"], gt["tried"],
                                reach_cp=[e * 5 + d for e, d in cells], identity=identity)
        if rec["rank_fv"] is None:
            raise RuntimeError(f"episode {global_i}: no valid GT action appears in the live candidate pool")
        records.append(rec)
        if fout:
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
        if (local_i + 1) % 10 == 0 or local_i == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [live {local_i + 1}/{len(selected)}] {elapsed:.1f}s ",
                  f"({elapsed / (local_i + 1):.3f}s/episode)", file=sys.stderr, flush=True)
    if fout:
        fout.close()
    return records, len(episodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    # Canonical 1-push answer key = namo_testset_v1 under the stricter 20% success bar (2026-06-10).
    # (Old key $NAMO_SCRATCH/manifests/v3_test_episodes.json was the "any point reachable" bar.)
    ap.add_argument("--episodes",
                    default=str(eval_sets.ONEPUSH))
    ap.add_argument("--h5-root", default=str(H5))
    ap.add_argument("--divisions", default="hard,med,easy")
    ap.add_argument("--num-depths", type=int, default=5)
    ap.add_argument("--h", type=int, default=1, help="budget to query for budget-Q ckpts (1=1-push panel; "
                    "2=does the H=2 query still rank 1-push openers on this 1-push set). Ignored for non-budget ckpts.")
    ap.add_argument("--network", default="dit_classifier", choices=["dit_classifier", "edge_crossattn"])
    ap.add_argument("--zoom-window", type=float, default=0.24, help="dual-crop zoom window (m), must match the build")
    ap.add_argument("--live-canonical", action="store_true",
                    help="score onepush_episodes.json through the current live renderer; no push simulations")
    ap.add_argument("--start", type=int, default=0, help="live mode: inclusive index in canonical episode order")
    ap.add_argument("--end", type=int, default=-1, help="live mode: exclusive index; -1 means all episodes")
    ap.add_argument("--leaf-out", help="live mode: optional per-episode JSONL for audit/monitoring")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.live_canonical:
        records, total = live_canonical_records(a.ckpt, a.episodes, a.start, a.end, a.leaf_out)
        full_run = a.start == 0 and (a.end < 0 or a.end >= total)
        if full_run:
            counts = {b: sum(bin_of(r["sr"]) == b for r in records) for b in ("easy", "med", "hard")}
            expected = {"easy": 698, "med": 421, "hard": 204}
            if len(records) != 1323 or counts != expected:
                raise RuntimeError(f"canonical count gate failed: n={len(records)} bins={counts}, expected 1323/{expected}")
            missing = sum(r["n_valid_missing_from_pool"] for r in records)
            if missing:
                raise RuntimeError(f"live candidate pool omitted {missing} valid GT cells")
        write_result(a.ckpt, records, a.out, mode="live_canonical")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_scorer(a.ckpt, a.num_depths, device, a.network)
    USE_ZOOM = bool(getattr(model.network, "use_zoom", False))
    ZS = int(getattr(model.network, "zoom_size", 128))
    if USE_ZOOM:
        print(f"  [dual-crop eval] zoom_window={a.zoom_window}m zoom_size={ZS}", file=sys.stderr)
    epf = json.load(open(a.episodes))
    # per-episode records, deduped across division files
    seen = set()
    recs_out = []  # (sr, {hit_real}, {hit_oracle}, cov_real)
    for div in a.divisions.split(","):
        h5 = f"{a.h5_root}/v3_test_{div}_lzf_tight_data/data.h5"
        f = h5py.File(h5, "r")
        N = int(f.attrs["n_samples"])
        xml = [x[0].decode() if isinstance(x[0], bytes) else str(x[0]) for x in f["xml_file"][:]]
        e = f["edge_idx_a1"][:, 0].astype(int); d = f["depth_idx_a1"][:, 0].astype(int)
        oc = f["local_tight_object_center"][:]
        oth = f["local_tight_object_theta"][:, 0]; osz = f["target_object_size"][:]
        cmm = f["local_tight_crop_size_meters"][:, 0]
        for i in range(N):
            gt = (int(e[i]), int(d[i]))
            rec, dm = match_episode(epf.get(xml[i]), oc[i], gt)
            if rec is None or dm > 0.01:
                continue
            key = (xml[i], round(float(oc[i, 0]), 4), round(float(oc[i, 1]), 4))
            if key in seen:
                continue
            seen.add(key)
            valid = {tuple(t) for t in rec["valid"]}; tried = {tuple(t) for t in rec["tried"]}
            sr = min(1.0, float(rec.get("solve_rate", len(valid) / max(1, len(tried)))))
            # build context (wide)
            raw = [f[k][i].astype(np.float32) for k in MASKS]
            chans = [cv2.resize(m, (OUT, OUT), interpolation=cv2.INTER_AREA) for m in raw]
            ctx = torch.from_numpy(np.stack(chans)[None]).float().to(device)
            cpx = np.array([contact_px(ee, float(osz[i, 0]), float(osz[i, 1]), float(oth[i]), float(cmm[i]))
                            for ee in range(60)], dtype=np.float32)
            cpx_t = torch.from_numpy(cpx[None]).float().to(device)
            ztup = (None, None)
            if USE_ZOOM:                       # dual-crop: tight object crop + its contact pixels
                S224 = raw[0].shape[0]; cn = S224 // 2
                half = int(round(a.zoom_window / float(cmm[i]) * S224 / 2))
                zc = [cv2.resize(m[cn - half:cn + half, cn - half:cn + half], (ZS, ZS), interpolation=cv2.INTER_AREA)
                      for m in raw]
                cz = np.array([contact_px(ee, float(osz[i, 0]), float(osz[i, 1]), float(oth[i]), a.zoom_window, ZS)
                               for ee in range(60)], dtype=np.float32)
                ztup = (torch.from_numpy(np.stack(zc)[None]).float().to(device),
                        torch.from_numpy(cz[None]).float().to(device))
            with torch.no_grad():
                # budget-Q ckpt: trained with H always present -> eval at H=`a.h` (default 1 = the 1-push
                # panel; --h 2 tests whether the H=2 query still nails 1-push openers on this 1-push set).
                # HL-Gauss head emits (60,5,bins) -> E[bin] value in [0,1]; the sigmoid below is
                # monotone so all top-k rankings are unchanged.
                hkw = {"H": torch.full((1,), int(a.h), dtype=torch.long, device=device)} \
                    if getattr(model.network, "budget_cond", False) else {}
                if getattr(model.network, "reach_flag_input", False):
                    # per-edge contact-point reachability bit from the episode's tried set (edge-level)
                    rbits = torch.zeros(1, 60, dtype=torch.long, device=device)
                    for (te, _td) in tried:
                        if 0 <= te < 60: rbits[0, te] = 1
                    hkw["reach_edges"] = rbits
                if getattr(model.network, "action_motion_dim", 0) > 0:
                    from namo.rl_loop.action_motion import action_motion_from_contact_px
                    hkw["action_motion"] = action_motion_from_contact_px(
                        cpx_t, encoding=model.network.action_motion_encoding,
                        feature_dim=model.network.action_motion_dim)
                t = model(ctx, cpx_t, ztup[0], ztup[1], **hkw)[0]
                if t.dim() == 3:
                    from src.model.hl_gauss import HLGauss
                    t = HLGauss(num_bins=t.shape[-1]).value(t)
                logits = t.cpu().numpy()  # (60,5)
            scores = 1.0 / (1.0 + np.exp(-logits))
            reach_cp = [ee * 5 + dd for ee in {x for x, _ in tried} for dd in range(5)]  # realistic (contact-pt)
            recs_out.append(diagnostic_record(scores, valid, tried, reach_cp=reach_cp,
                                               identity={"sr": sr}))
        f.close()
        print(f"  scored {div}: total kept={len(recs_out)}", file=sys.stderr, flush=True)

    write_result(a.ckpt, recs_out, a.out, mode="legacy_h5")


if __name__ == "__main__":
    main()
