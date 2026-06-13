#!/usr/bin/env python3
"""Pack POST-PUSH npz (render_postpush_from_state.py) into a scorer H5 — horizon-Q v2 OOD data.

Unlike build_scorer_dataset.py (which JOINS src-h5 masks + episodes-JSON labels by object_center), the
post-push npz SELF-CARRY their labels (pp_* keys), so this is a direct npz -> scorer-H5 pack: NO match,
NO episodes JSON. Output is byte-compatible with ScorerH5Dataset (ctx/f_grid/r_mask/ratio/H/contact_px)
plus stratification tags (dead, postpush) for the v2 WeightedRandomSampler.

Per row (one post-push state s1):
  ctx (5,64,64) : local_tight_{static,movable,target_object,robot_region,goal_sample_region}, 224->64 INTER_AREA
  f_grid (60,5) : 1 at pp_open (a2 opens), 0 at pp_tried-but-not-open, nan->0 elsewhere (r_mask gates loss)
  r_mask (60,5) : 1 at pp_tried (the ~k SAMPLED a2 = what we KNOW = the loss mask; rest UNKNOWN/masked)
  contact_px (60,2) : per-edge contact pixel at the s1 object pose (add_contact_px.contact_px)
  H=1, dead (1=no opener), postpush=1, object_center, xml, ratio(F/R)

  python scripts/pipeline/build_postpush_h5.py --npz-dir out/ --out-h5 /scratch/.../v4_hq_postpush/data.h5
  python scripts/pipeline/build_postpush_h5.py --npz-list shards.txt --out-h5 ...   # for sharded packs
"""
import argparse
import glob
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_contact_px import contact_px  # noqa: E402

CHANS = ["static", "movable", "target_object", "robot_region", "goal_sample_region"]
OUT = 64


def npz_paths(a):
    if a.npz_list:
        return [l.strip() for l in open(a.npz_list) if l.strip()]
    return sorted(glob.glob(os.path.join(a.npz_dir, "**", "*.npz"), recursive=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default=None, help="recursive glob of post-push npz")
    ap.add_argument("--npz-list", default=None, help="file of npz paths (alternative to --npz-dir)")
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--crop", default="tight", choices=["tight", "wide"])
    a = ap.parse_args()
    assert a.npz_dir or a.npz_list, "need --npz-dir or --npz-list"
    paths = npz_paths(a)
    print(f"found {len(paths)} npz", flush=True)
    MASKS = [f"local_{a.crop}_{c}" for c in CHANS]

    os.makedirs(os.path.dirname(a.out_h5), exist_ok=True)
    M = len(paths)
    dst = h5py.File(a.out_h5, "w")
    cM = max(1, min(32, M))
    ctx = dst.create_dataset("ctx", (M, 5, OUT, OUT), maxshape=(None, 5, OUT, OUT), dtype="float32",
                             compression="lzf", chunks=(cM, 5, OUT, OUT))
    fg = dst.create_dataset("f_grid", (M, 60, 5), maxshape=(None, 60, 5), dtype="float32", compression="lzf")
    rm = dst.create_dataset("r_mask", (M, 60, 5), maxshape=(None, 60, 5), dtype="float32", compression="lzf")
    cpd = dst.create_dataset("contact_px", (M, 60, 2), maxshape=(None, 60, 2), dtype="float32", compression="lzf")
    ratio = dst.create_dataset("ratio", (M,), maxshape=(None,), dtype="float32")
    ocd = dst.create_dataset("object_center", (M, 2), maxshape=(None, 2), dtype="float32")
    xmld = dst.create_dataset("xml", (M,), maxshape=(None,), dtype=h5py.string_dtype())
    Hd = dst.create_dataset("H", (M,), maxshape=(None,), dtype="int8")
    deadd = dst.create_dataset("dead", (M,), maxshape=(None,), dtype="uint8")
    ppd = dst.create_dataset("postpush", (M,), maxshape=(None,), dtype="uint8")  # stratification tag for v2 sampler

    j = 0
    n_good = n_dead = n_bad = 0
    for ip, p in enumerate(paths):
        try:
            d = np.load(p, allow_pickle=True)
            if f"has_local_{a.crop}_masks" in d and not bool(d[f"has_local_{a.crop}_masks"][0]):
                n_bad += 1; continue
            chans = [cv2.resize(d[k].astype(np.float32), (OUT, OUT), interpolation=cv2.INTER_AREA) for k in MASKS]
            grid = np.full((60, 5), np.nan, np.float32)
            te, td = d["pp_tried_ed"], d["pp_tried_dp"]
            for ee, dd in zip(te, td):
                if 0 <= ee < 60 and 0 <= dd < 5:
                    grid[ee, dd] = 0.0
            oe, od = d["pp_open_ed"], d["pp_open_dp"]
            for ee, dd in zip(oe, od):
                if 0 <= ee < 60 and 0 <= dd < 5:
                    grid[ee, dd] = 1.0
            mask = (~np.isnan(grid)).astype(np.float32)
            if mask.sum() == 0:
                n_bad += 1; continue   # nothing tried -> no supervision
            # contact pixels at the s1 object pose
            th = float(d[f"local_{a.crop}_object_theta"][0])
            hw, hd = float(d["target_object_size"][0]), float(d["target_object_size"][1])
            cm = float(d[f"local_{a.crop}_crop_size_meters"][0])
            cpx = np.zeros((60, 2), np.float32)
            for e in range(60):
                cpx[e] = contact_px(e, hw, hd, th, cm)
            ctx[j] = np.stack(chans)
            fg[j] = np.nan_to_num(grid, nan=0.0)
            rm[j] = mask
            cpd[j] = cpx
            R = int(mask.sum()); F = float(np.nansum(grid))
            ratio[j] = F / R if R else 0.0
            ocd[j] = d[f"local_{a.crop}_object_center"][:2]
            xmld[j] = str(d["xml_file"][0])
            Hd[j] = int(d["pp_H"][0])
            dead = int(d["pp_dead"][0]); deadd[j] = dead; ppd[j] = 1
            n_dead += dead; n_good += (1 - dead)
            j += 1
        except Exception as ex:
            n_bad += 1
            if n_bad <= 5:
                print(f"  bad {os.path.basename(p)}: {ex}", file=sys.stderr)
            continue
        if ip % 5000 == 0:
            print(f"  [{ip}/{M}] packed={j} good={n_good} dead={n_dead} bad={n_bad}", file=sys.stderr, flush=True)

    # shrink to the rows actually written
    for name in ("ctx", "f_grid", "r_mask", "contact_px", "ratio", "object_center", "xml", "H", "dead", "postpush"):
        dst[name].resize(j, axis=0)
    dst.attrs["n_samples"] = j
    dst.attrs["channels"] = ",".join(MASKS)
    dst.attrs["state_type"] = "postpush"
    dst.close()
    print(f"wrote {a.out_h5}  rows={j} (good={n_good} dead={n_dead}) bad={n_bad}", flush=True)


if __name__ == "__main__":
    main()
