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

KNOWN UNKNOWN ABOUT THE MASKS THIS JOINS, for whoever retrains or audits this H5.
This script re-uses masks; it does not render. They came from batch_collection.py, which calls
sage_learning's generate_all_masks_highres. That function renders its region channels through the
unified wavefront (WavefrontSnapshotExporter.from_geometry, pure geometry), and on failure prints a
warning and falls back to a legacy BFS that its own message says "may use wrong robot size".

Nothing in the mask-generation package recorded whether that fallback fired: no unified_ok, no flag,
no column. So any row rendered under a fallback carries wrong-size region channels and there is NO
FIELD IN THIS H5 to find those rows by.

Measured 2026-08-28: 0 of 14 currently-renderable captured scenes fall back, so the rate is plausibly
low. That is a measurement on today's scenes, not on the corpus this H5 was built from, and it is not
evidence about those rows. Deployment now refuses rather than falling back (namo_cpp 1628d1f); this
H5 predates that.

Renderer parity itself is NOT in question: training and deployment call the same function, so there
is no train-deploy skew in the region channels.
"""
import argparse
import json
import math
import os
import sys

import cv2
import h5py
import numpy as np
from namo.paths import H5, MANIFESTS

CHANS = ["static", "movable", "target_object", "robot_region", "goal_sample_region"]
OUT = 64


def _rec_valid(rec):
    """Record's 'a chain exists' set, format-agnostic: 1-push key has `valid`; the 2-push key
    (build_2push_validset) has valid_1push ∪ valid_first_push (a1 of a 2-push chain lives there)."""
    if "valid" in rec:
        return {tuple(t) for t in rec["valid"]}
    return {tuple(t) for t in rec.get("valid_1push", [])} | {tuple(t) for t in rec.get("valid_first_push", [])}


def match_episode(recs, oci, gt, dead=False):
    """dead=True (npz edge_idx_a1 == -1 sentinel): the row is a DEAD-END — match only among dead-end
    records (no chain at any depth). gt=(-1,-1) would otherwise fall back to nearest-center, which is
    ambiguous when the same object serves several goal regions (could attach a SOLVABLE record's labels)."""
    if not recs:
        return None, 1e9
    if dead:
        pool = [r for r in recs if not _rec_valid(r)]
    else:
        pool = [r for r in recs if gt in _rec_valid(r)] or [r for r in recs if _rec_valid(r)] or recs
    if not pool:
        return None, 1e9
    rec = min(pool, key=lambda r: (r["object_center"][0] - oci[0]) ** 2 + (r["object_center"][1] - oci[1]) ** 2)
    return rec, math.hypot(rec["object_center"][0] - oci[0], rec["object_center"][1] - oci[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-h5", nargs="+", default=[str(H5 / "v3_1push_le10_lzf_tight_data/data.h5")],
                    help="one or MORE mask H5s (parallel pack shards); rows are matched per-file and "
                         "deduped across files by episode identity")
    ap.add_argument("--episodes", default=str(MANIFESTS / "v3_train_le10_episodes.json"))
    ap.add_argument("--out-h5", default=str(H5 / "v3_scorer_1push/data.h5"))
    ap.add_argument("--crop", default="tight", choices=["tight", "wide"],
                    help="which crop's masks to read (wide=1.2m FOV, the E5-wide lever)")
    ap.add_argument("--format", default="onepush", choices=["onepush", "twopush"],
                    help="episodes-JSON schema. onepush = build_episode_validsets (valid/tried; 1 row/episode). "
                         "twopush = build_2push_validset (valid_1push/valid_first_push/tried_*; TWO rows per "
                         "episode sharing one ctx: an H=1 row and a gamma-valued H=2 row + frac datasets).")
    ap.add_argument("--gamma", type=float, default=0.9,
                    help="discount for opens-in-2 cells on H=2 rows (locked decision #2; tunable)")
    a = ap.parse_args()
    MASKS = [f"local_{a.crop}_{c}" for c in CHANS]

    epf = json.load(open(a.episodes))
    fs = [h5py.File(p, "r") for p in a.src_h5]

    # one row per unique EPISODE = matched validset record; dedup by the record's identity
    # (object_center + goal region), NOT by npz center alone — the same object can serve several
    # goal regions (distinct episodes, distinct labels) and dead-end + solvable can share a center.
    # `seen` spans ALL src files, so overlapping packs can't double-emit an episode.
    seen = {}; rows = []; bad = 0; gtok = 0; n_solv = 0; n_dead = 0; _rp_cache = {}
    metas = []   # per-file (xml, e, d, oc) — the write pass indexes rows by (file, row)
    for fi, f in enumerate(fs):
        N = int(f.attrs["n_samples"])
        xml = [x[0].decode() if isinstance(x[0], bytes) else str(x[0]) for x in f["xml_file"][:]]
        e = f["edge_idx_a1"][:, 0].astype(int); d = f["depth_idx_a1"][:, 0].astype(int)
        oc = f[f"local_{a.crop}_object_center"][:]
        metas.append((xml, e, d, oc))
        for i in range(N):
            dead = int(e[i]) < 0   # npz dead-end sentinel (edge_idx_a1 == -1)
            gt = (int(e[i]), int(d[i]))
            # 2-push keys are REALPATHS (build_2push_validset); npz carry shard-symlink paths — fall back.
            recs = epf.get(xml[i])
            if recs is None:
                if xml[i] not in _rp_cache:
                    _rp_cache[xml[i]] = os.path.realpath(xml[i])
                recs = epf.get(_rp_cache[xml[i]])
            rec, dm = match_episode(recs, oc[i], gt, dead=dead)
            if rec is None or dm > 0.01:
                bad += 1; continue
            key = (xml[i], round(float(rec["object_center"][0]), 4), round(float(rec["object_center"][1]), 4),
                   str(rec.get("region")), bool(_rec_valid(rec)))
            if key in seen:
                continue
            seen[key] = 1
            if dead:
                n_dead += 1
            else:
                n_solv += 1; gtok += (gt in _rec_valid(rec))
            rows.append((fi, i, rec, dead))
        print(f"  src[{fi}] {a.src_h5[fi]}: {N} rows scanned", flush=True)
    print(f"src_total={sum(int(f.attrs['n_samples']) for f in fs)} unique-episodes={len(rows)} "
          f"(solvable={n_solv} dead-end={n_dead}) bad_match={bad} "
          f"gt_in_valid={gtok/max(n_solv,1)*100:.2f}%", flush=True)
    assert n_solv == 0 or gtok / n_solv > 0.99, "per-episode label join failed (gt not in valid)"

    def grids_for(rec, dead):
        """Per-record list of (grid60x5 float|nan=unknown, H, frac_pairs|None).

        onepush: one H=1 grid — 1.0 on valid, 0 on tried-failed, nan elsewhere (legacy semantics).
        twopush: TWO grids sharing the record's ctx —
          H=1: 1.0 on valid_1push, 0 on tried_1push-failed.
          H=2: 1.0 on valid_1push (opens in 1 ⇒ within 2), gamma on valid_first_push∖valid_1push,
               0 ONLY on expanded-and-dead cells (tried_first_push minus both valids); a cell tried at
               level 1 but NEVER expanded has UNKNOWN H=2 outcome -> stays nan (masked), never 0.
        """
        if "valid" in rec:   # onepush schema
            g = np.full((60, 5), np.nan, np.float32)
            for (ee, dd) in rec["tried"]:
                if 0 <= ee < 60 and 0 <= dd < 5: g[ee, dd] = 0.0
            for (ee, dd) in rec["valid"]:
                if 0 <= ee < 60 and 0 <= dd < 5: g[ee, dd] = 1.0
            return [(g, 1, None)]
        v1 = {tuple(t) for t in rec["valid_1push"]}
        vfp = {tuple(t) for t in rec["valid_first_push"]}
        g1 = np.full((60, 5), np.nan, np.float32)
        for (ee, dd) in rec["tried_1push"]:
            if 0 <= ee < 60 and 0 <= dd < 5: g1[ee, dd] = 0.0
        for (ee, dd) in v1:
            if 0 <= ee < 60 and 0 <= dd < 5: g1[ee, dd] = 1.0
        g2 = np.full((60, 5), np.nan, np.float32)
        for (ee, dd) in rec["tried_first_push"]:
            if 0 <= ee < 60 and 0 <= dd < 5: g2[ee, dd] = 0.0
        for (ee, dd) in vfp:
            if 0 <= ee < 60 and 0 <= dd < 5: g2[ee, dd] = a.gamma
        for (ee, dd) in v1:
            if 0 <= ee < 60 and 0 <= dd < 5: g2[ee, dd] = 1.0
        return [(g1, 1, None), (g2, 2, rec.get("frac_first_push"))]

    n_rows_per = 2 if a.format == "twopush" else 1
    M = len(rows) * n_rows_per
    os.makedirs(os.path.dirname(a.out_h5), exist_ok=True)
    dst = h5py.File(a.out_h5, "w")
    ctx = dst.create_dataset("ctx", (M, 5, OUT, OUT), dtype="float32", compression="lzf", chunks=(min(32, M), 5, OUT, OUT))
    fg = dst.create_dataset("f_grid", (M, 60, 5), dtype="float32", compression="lzf")
    rm = dst.create_dataset("r_mask", (M, 60, 5), dtype="float32", compression="lzf")
    ratio = dst.create_dataset("ratio", (M,), dtype="float32")
    ocd = dst.create_dataset("object_center", (M, 2), dtype="float32")
    xmld = dst.create_dataset("xml", (M,), dtype=h5py.string_dtype())
    deadd = dst.create_dataset("dead", (M,), dtype="uint8")   # 1 = dead-end episode (no chain, H0b)
    Hd = dst.create_dataset("H", (M,), dtype="int8")          # remaining push budget of the row's labels
    if a.format == "twopush":   # robustness fractions (per first push): succ/tried over unique child cells
        fsuc = dst.create_dataset("frac_succ", (M, 60, 5), dtype="int16", compression="lzf")
        ftry = dst.create_dataset("frac_tried", (M, 60, 5), dtype="int16", compression="lzf")

    edge_align_err = 0
    j = 0
    for ridx, (fi, i, rec, dead) in enumerate(rows):
        if ridx % 1000 == 0:
            print(f"  [{ridx}/{len(rows)}]", file=sys.stderr, flush=True)
        f = fs[fi]; xml, e, d, oc = metas[fi]
        chans = []
        for k in MASKS:
            m = f[k][i].astype(np.float32)
            chans.append(cv2.resize(m, (OUT, OUT), interpolation=cv2.INTER_AREA))
        ctx_row = np.stack(chans)
        gt_checked = False
        for grid, H, fracs in grids_for(rec, dead):
            mask = (~np.isnan(grid)).astype(np.float32)
            ctx[j] = ctx_row
            fg[j] = np.nan_to_num(grid, nan=0.0)   # store 0 for unknown; r_mask gates the loss
            rm[j] = mask
            R = int(mask.sum()); F = float(np.nansum(grid))
            ratio[j] = F / R if R else 0.0
            ocd[j] = oc[i]; xmld[j] = xml[i]; deadd[j] = int(dead); Hd[j] = H
            if fracs is not None:
                fs_g = np.zeros((60, 5), np.int16); ft_g = np.zeros((60, 5), np.int16)
                for pe, pd, ns, nt in fracs:
                    if 0 <= pe < 60 and 0 <= pd < 5:
                        fs_g[pe, pd] = ns; ft_g[pe, pd] = nt
                fsuc[j] = fs_g; ftry[j] = ft_g
            # gate: the npz GT first-push must be a known-positive cell at the DEEPEST grid for this
            # record (H=1 for onepush; H=2 covers 2-push chains whose a1 is a setup push). Dead rows
            # have no gt (sentinel -1).
            if not dead and not gt_checked and H == n_rows_per:
                gt_checked = True
                if not (mask[int(e[i]), int(d[i])] and fg[j][int(e[i]), int(d[i])] > 0.0):
                    edge_align_err += 1
            j += 1

    dst.attrs["n_samples"] = M
    dst.attrs["source_h5"] = ";".join(a.src_h5)
    dst.attrs["channels"] = ",".join(MASKS)
    dst.attrs["format"] = a.format
    dst.attrs["gamma"] = a.gamma
    f.close(); dst.close()
    print(f"wrote {a.out_h5}  rows={M}  edge_align_err={edge_align_err} (must be 0)", flush=True)
    assert edge_align_err == 0, "edge/depth alignment gate failed"


if __name__ == "__main__":
    main()
