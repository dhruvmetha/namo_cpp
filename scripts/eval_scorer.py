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

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import MASKS, OUT, match_episode, bin_of, floor_no_replacement  # shared grading contract


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
    sys.path.insert(0, "/cache/home/dm1487/projects/namo/sage_learning")
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
        net = EdgeCrossAttn(**kw)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", default="/scratch/dm1487/manifests/v3_test_episodes.json")
    ap.add_argument("--h5-root", default="/scratch/dm1487/h5")
    ap.add_argument("--divisions", default="hard,med,easy")
    ap.add_argument("--num-depths", type=int, default=5)
    ap.add_argument("--network", default="dit_classifier", choices=["dit_classifier", "edge_crossattn"])
    ap.add_argument("--zoom-window", type=float, default=0.24, help="dual-crop zoom window (m), must match the build")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_scorer(a.ckpt, a.num_depths, device, a.network)
    USE_ZOOM = bool(getattr(model.network, "use_zoom", False))
    ZS = int(getattr(model.network, "zoom_size", 128))
    if USE_ZOOM:
        print(f"  [dual-crop eval] zoom_window={a.zoom_window}m zoom_size={ZS}", file=sys.stderr)
    epf = json.load(open(a.episodes))
    KS = [1, 5, 10, 20]

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
                logits = model(ctx, cpx_t, ztup[0], ztup[1])[0].cpu().numpy()  # (60,5)
            scores = 1.0 / (1.0 + np.exp(-logits))
            # candidate sets (flat indices)
            reach_exact = [ee * 5 + dd for (ee, dd) in tried if dd < 5]                 # oracle
            reach_cp = [ee * 5 + dd for ee in {x for x, _ in tried} for dd in range(5)]  # realistic (contact-pt)
            hit_o = topk_hit(scores, valid, reach_exact, KS)
            hit_r = topk_hit(scores, valid, reach_cp, KS)

            # ----- diagnostics (same forward pass, deployment-realistic reach_cp pool) -----
            flat = scores.reshape(-1)
            order = sorted(reach_cp, key=lambda j: -flat[j])     # reachable candidates, best score first
            valid_flat = {ee * 5 + dd for (ee, dd) in valid}
            solving_edges = {ee for (ee, _) in valid}
            best = order[0]; be, bd = best // 5, best % 5
            # failure category of the top-1 pick (not_reachable is 0 by construction — we rank reachable only)
            if best in valid_flat:
                cat = "success"
            elif be in solving_edges:
                cat = "right_edge_wrong_depth"
            else:
                cat = "wrong_edge"
            edge_hit = {k: any((j // 5) in solving_edges for j in order[:k]) for k in KS}  # right edge in top-k
            rank_fv = next((idx + 1 for idx, j in enumerate(order) if j in valid_flat), None)  # 1-indexed
            depth_right = (bd in {dd for (ee, dd) in valid if ee == be}) if be in solving_edges else None
            sc = np.array([flat[j] for j in reach_cp])
            vm = np.array([1 if j in valid_flat else 0 for j in reach_cp])
            sep = float(sc[vm == 1].mean() - sc[vm == 0].mean()) if (vm.sum() > 0 and (len(sc) - vm.sum()) > 0) else None

            recs_out.append({"sr": sr, "nF": len(valid), "nR": len(tried), "hit_r": hit_r, "hit_o": hit_o,
                             "edge_hit": edge_hit, "cat": cat, "top1_depth": bd, "depth_right": depth_right,
                             "rank_fv": rank_fv, "sep": sep})
        f.close()
        print(f"  scored {div}: total kept={len(recs_out)}", file=sys.stderr, flush=True)

    # aggregate per true-difficulty bin
    def agg(records):
        out = {}
        for b in ("hard", "med", "easy"):
            R = [r for r in records if bin_of(r["sr"]) == b]
            n = len(R)
            if not n:
                out[b] = {"n": 0}; continue
            real = {f"@{k}": round(np.mean([r["hit_r"][k] for r in R]) * 100, 1) for k in KS}
            orac = {f"@{k}": round(np.mean([r["hit_o"][k] for r in R]) * 100, 1) for k in KS}
            # random floor: without replacement (hypergeometric), per-episode F=nF, R=nR
            floor = {f"@{k}": round(np.mean([floor_no_replacement(r["nF"], r["nR"], k) for r in R]) * 100, 1) for k in KS}
            edge = {f"@{k}": round(np.mean([r["edge_hit"][k] for r in R]) * 100, 1) for k in KS}  # right edge, any depth
            # failure decomposition of the top-1 pick (not_reachable=0 by construction for the scorer)
            decomp = {c: round(np.mean([r["cat"] == c for r in R]) * 100, 1)
                      for c in ("success", "right_edge_wrong_depth", "wrong_edge")}
            decomp["not_reachable"] = 0.0
            n_miss = sum(1 for r in R if r["cat"] != "success")
            we_overall = round(np.mean([r["cat"] == "wrong_edge" for r in R]) * 100, 1)
            we_of_miss = round(sum(r["cat"] == "wrong_edge" for r in R) / n_miss * 100, 1) if n_miss else 0.0
            # depth of top-1 pick + depth-accuracy given the edge was a solving edge
            dh = [0] * 5
            for r in R:
                dh[r["top1_depth"]] += 1
            depth_hist = {f"d{i}": round(100 * dh[i] / n, 1) for i in range(5)}
            dr = [r["depth_right"] for r in R if r["depth_right"] is not None]
            depth_acc = round(np.mean(dr) * 100, 1) if dr else None
            # rank of first valid push in the score-sorted reachable pool (1-indexed)
            ranks = [r["rank_fv"] for r in R if r["rank_fv"] is not None]
            rank_stats = {"median": float(np.median(ranks)) if ranks else None,
                          "pct_le3": round(np.mean([x <= 3 for x in ranks]) * 100, 1) if ranks else None,
                          "pct_le10": round(np.mean([x <= 10 for x in ranks]) * 100, 1) if ranks else None,
                          "pct_le20": round(np.mean([x <= 20 for x in ranks]) * 100, 1) if ranks else None,
                          "n_with_valid_in_pool": len(ranks)}
            # score separation: mean(valid) - mean(invalid) over the reachable pool
            seps = [r["sep"] for r in R if r["sep"] is not None]
            sep_stats = {"mean_margin": round(float(np.mean(seps)), 3) if seps else None,
                         "pct_positive": round(np.mean([s > 0 for s in seps]) * 100, 1) if seps else None}
            out[b] = {"n": n, "sr_mean_pct": round(np.mean([r["sr"] for r in R]) * 100, 2),
                      "scorer_realistic": real, "scorer_oracle": orac, "floor": floor,    # existing keys (do not rename)
                      "edge_at_k": edge, "failure_decomp_at1_pct": decomp,
                      "wrong_edge_overall_pct": we_overall, "wrong_edge_of_misses_pct": we_of_miss,
                      "depth_top1_hist_pct": depth_hist, "depth_acc_given_right_edge_pct": depth_acc,
                      "rank_first_valid": rank_stats, "score_separation": sep_stats}
        return out

    res = {"ckpt": a.ckpt, "n_episodes": len(recs_out), "divisions": agg(recs_out)}
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    for b in ("hard", "med", "easy"):
        r = res["divisions"][b]
        if not r.get("n"):
            continue
        sk = r["scorer_realistic"]; fl = r["floor"]; ek = r["edge_at_k"]; fd = r["failure_decomp_at1_pct"]
        dh = r["depth_top1_hist_pct"]; rk = r["rank_first_valid"]; ss = r["score_separation"]
        print(f"\n===== [{b}]  n={r['n']}  sr_mean={r['sr_mean_pct']}% =====", file=sys.stderr)
        print(f"  success@k   1/5/10/20 : {sk['@1']}/{sk['@5']}/{sk['@10']}/{sk['@20']}", file=sys.stderr)
        print(f"  edge@k      1/5/10/20 : {ek['@1']}/{ek['@5']}/{ek['@10']}/{ek['@20']}   (right edge, any depth)", file=sys.stderr)
        print(f"  floor@k     1/5/10/20 : {fl['@1']}/{fl['@5']}/{fl['@10']}/{fl['@20']}   (random, no-replacement)", file=sys.stderr)
        print(f"  fail@1   succ/rEwD/wrongE : {fd['success']}/{fd['right_edge_wrong_depth']}/{fd['wrong_edge']}%", file=sys.stderr)
        print(f"  wrong-edge  overall={r['wrong_edge_overall_pct']}%  of-misses={r['wrong_edge_of_misses_pct']}%", file=sys.stderr)
        print(f"  depth top1  d0..d4 : {dh['d0']}/{dh['d1']}/{dh['d2']}/{dh['d3']}/{dh['d4']}%   depth-acc|rightEdge={r['depth_acc_given_right_edge_pct']}%", file=sys.stderr)
        print(f"  rank-1st-valid  median={rk['median']}  ≤3={rk['pct_le3']}%  ≤10={rk['pct_le10']}%  ≤20={rk['pct_le20']}%", file=sys.stderr)
        print(f"  score-sep   margin={ss['mean_margin']}  positive={ss['pct_positive']}%", file=sys.stderr)
    print(f"\nwrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
