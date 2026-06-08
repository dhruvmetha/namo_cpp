#!/usr/bin/env python3
"""Build the 1-push SCORER (HACMan-critic) training H5 by JOINING:
  - the already-correct car scene masks from the diffusion H5 (local_tight 5 channels), and
  - the exhaustive per-episode labels (f_grid 60x5 + reachable mask) from the per-episode key.

This reuses verified car masks (apples-to-apples with the diffusion baseline) and the per-episode
answer key (docs/pipeline/multi_episode_rooms.md) — NO re-rendering, NO point-robot generator, NO
crop mismatch. Each output row = one (xml, pushed-object) EPISODE:
  ctx (5,64,64)   : static, movable, target_object, robot_region, goal_sample_region  (resized 224->64)
  f_grid (60,5)   : 1=push opens path, 0=tried&failed, NaN=unreachable
  r_mask (60,5)   : 1=reachable/tried, 0=unreachable   (the action mask / candidate set)
  xml, object_center, ratio(F/R)
Split is held out BY ROOM downstream (never by row). Sanity gates are asserted + printed.
"""
import argparse
import json
import math
import sys

import cv2
import h5py
import numpy as np

CHANS = ["static", "movable", "target_object", "robot_region", "goal_sample_region"]
OUT = 64


def match_episode(recs, oci, gt):
    if not recs:
        return None, 1e9
    pool = [r for r in recs if gt in {tuple(t) for t in r["valid"]}] or recs
    rec = min(pool, key=lambda r: (r["object_center"][0] - oci[0]) ** 2 + (r["object_center"][1] - oci[1]) ** 2)
    return rec, math.hypot(rec["object_center"][0] - oci[0], rec["object_center"][1] - oci[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-h5", default="/scratch/dm1487/h5/v3_1push_le10_lzf_tight_data/data.h5")
    ap.add_argument("--episodes", default="/scratch/dm1487/manifests/v3_train_le10_episodes.json")
    ap.add_argument("--out-h5", default="/scratch/dm1487/h5/v3_scorer_1push/data.h5")
    ap.add_argument("--crop", default="tight", choices=["tight", "wide"],
                    help="which crop's masks to read (wide=1.2m FOV, the E5-wide lever)")
    a = ap.parse_args()
    MASKS = [f"local_{a.crop}_{c}" for c in CHANS]

    epf = json.load(open(a.episodes))
    f = h5py.File(a.src_h5, "r")
    N = int(f.attrs["n_samples"])
    xml = [x[0].decode() if isinstance(x[0], bytes) else str(x[0]) for x in f["xml_file"][:]]
    e = f["edge_idx_a1"][:, 0].astype(int); d = f["depth_idx_a1"][:, 0].astype(int)
    oc = f[f"local_{a.crop}_object_center"][:]

    # one row per unique EPISODE (xml, object_center); dedup repeats
    seen = {}; rows = []; bad = 0; gtok = 0
    for i in range(N):
        gt = (int(e[i]), int(d[i]))
        rec, dm = match_episode(epf.get(xml[i]), oc[i], gt)
        if rec is None or dm > 0.01:
            bad += 1; continue
        key = (xml[i], round(float(oc[i, 0]), 4), round(float(oc[i, 1]), 4))
        if key in seen:
            continue
        seen[key] = 1
        valid = {tuple(t) for t in rec["valid"]}; tried = {tuple(t) for t in rec["tried"]}
        gtok += (gt in valid)
        rows.append((i, valid, tried))
    print(f"src={N} unique-episodes={len(rows)} bad_match={bad} gt_in_valid={gtok/len(rows)*100:.2f}%", flush=True)
    assert gtok / len(rows) > 0.99, "per-episode label join failed (gt not in valid)"

    M = len(rows)
    import os
    os.makedirs(os.path.dirname(a.out_h5), exist_ok=True)
    dst = h5py.File(a.out_h5, "w")
    ctx = dst.create_dataset("ctx", (M, 5, OUT, OUT), dtype="float32", compression="lzf", chunks=(32, 5, OUT, OUT))
    fg = dst.create_dataset("f_grid", (M, 60, 5), dtype="float32", compression="lzf")
    rm = dst.create_dataset("r_mask", (M, 60, 5), dtype="float32", compression="lzf")
    ratio = dst.create_dataset("ratio", (M,), dtype="float32")
    ocd = dst.create_dataset("object_center", (M, 2), dtype="float32")
    xmld = dst.create_dataset("xml", (M,), dtype=h5py.string_dtype())

    edge_align_err = 0
    for j, (i, valid, tried) in enumerate(rows):
        if j % 1000 == 0:
            print(f"  [{j}/{M}]", file=sys.stderr, flush=True)
        chans = []
        for k in MASKS:
            m = f[k][i].astype(np.float32)
            chans.append(cv2.resize(m, (OUT, OUT), interpolation=cv2.INTER_AREA))
        ctx[j] = np.stack(chans)
        grid = np.full((60, 5), np.nan, dtype=np.float32)
        for (ee, dd) in tried:
            if 0 <= ee < 60 and 0 <= dd < 5:
                grid[ee, dd] = 0.0
        for (ee, dd) in valid:
            if 0 <= ee < 60 and 0 <= dd < 5:
                grid[ee, dd] = 1.0
        mask = (~np.isnan(grid)).astype(np.float32)
        fg[j] = np.nan_to_num(grid, nan=0.0)   # store 0 for unreachable; r_mask gates the loss
        rm[j] = mask
        R = int(mask.sum()); F = int(np.nansum(grid))
        ratio[j] = F / R if R else 0.0
        ocd[j] = oc[i]; xmld[j] = xml[i]
        # gate: the H5 GT push (e,d) must be a reachable+valid cell of this grid
        if not (mask[int(e[i]), int(d[i])] and grid[int(e[i]), int(d[i])] == 1.0):
            edge_align_err += 1

    dst.attrs["n_samples"] = M
    dst.attrs["source_h5"] = a.src_h5
    dst.attrs["channels"] = ",".join(MASKS)
    f.close(); dst.close()
    print(f"wrote {a.out_h5}  rows={M}  edge_align_err={edge_align_err} (must be 0)", flush=True)
    assert edge_align_err == 0, "edge/depth alignment gate failed"


if __name__ == "__main__":
    main()
