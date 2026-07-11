#!/usr/bin/env python3
"""Pool rung-1 + rung-2 H5s into ONE Q2 value-training H5 (the depth-<=2 dense value field).

Q2 trains a dense (60,5) value regressor over gamma^k on the REACHABLE-AND-TRIED cells (the same
reachable-only arm as Q1, deferring the -1 fold-in). It needs only the COMMON columns both rungs
share, so this script:

  1. copies rung-1 rows wholesale (depth-1 grids: value_target in {-1,0,1});
  2. copies rung-2 rows (per-tree-node grids: value_target in {-1,0,gamma=0.9,1}) EXCEPT the
     setup_moved==0 no-op nodes (~27%: their ctx == the start state and they carry no wins, so
     they are redundant with the root row and only dilute the pool) -> DROP them;
  3. keeps ONLY the columns the Q2 datamodule reads:
        ctx, contact_px, r_mask, value_target, value_mask, xml
     (rung-2's setup_moved / node_kind / diagnostics are consumed here for the filter, not carried).

ctx is preserved UNCOMPRESSED float16 (as in both sources) so the dataloader stays decompression-free.
xml is written as a variable-length string (the room key the datamodule groups the train/val split by).

Usage:
  POOL smoke:
    python scripts/pipeline/pool_q2_h5.py \
      --rung1 /common/users/dm1487/scratch_namo/exit/rung1_smoke50/rung1_smoke50.h5 \
      --rung2 /common/users/dm1487/scratch_namo/exit/rung2_smoke/rung2_smoke.h5 \
      --out   /common/users/dm1487/scratch_namo/exit/q2_pool_smoke/q2_pool_smoke.h5
  POOL full:
    python scripts/pipeline/pool_q2_h5.py \
      --rung1 /common/users/dm1487/scratch_namo/exit/rung1_full.h5 \
      --rung2 /common/users/dm1487/scratch_namo/exit/rung2_full.h5 \
      --out   /common/users/dm1487/scratch_namo/exit/q2_pool_full/q2_pool_full.h5
"""
import argparse
import os

import h5py
import numpy as np

COMMON = ["ctx", "contact_px", "r_mask", "value_target", "value_mask", "xml"]
NUM_DEPTHS = 5
GAMMA = 0.9


def _copy_block(src, dst, src_idx, dst_off, cols, chunk=512):
    """Copy rows src_idx (sorted 1-D index array) of `src` into `dst` starting at dst_off, chunked."""
    n = len(src_idx)
    written = 0
    # iterate over contiguous source windows spanning src_idx to keep reads sequential
    for lo in range(0, n, chunk):
        idx = src_idx[lo:lo + chunk]
        a, b = int(idx[0]), int(idx[-1]) + 1
        for c in cols:
            block = src[c][a:b]                       # contiguous read
            sel = block[idx - a]                      # gather the wanted rows
            if c == "xml":
                sel = np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                                for x in sel], dtype=object)
            dst[c][dst_off + written: dst_off + written + len(idx)] = sel
        written += len(idx)
    return written


def _report(out_path):
    with h5py.File(out_path, "r") as f:
        n = f["value_target"].shape[0]
        cells = n * 60 * NUM_DEPTHS
        cnt = {"-1": 0, "0": 0, "0.9": 0, "1": 0, "other": 0}
        masked = 0
        tr_cnt = {"0": 0, "0.9": 0, "1": 0, "other": 0}
        tr_tot = 0
        for lo in range(0, n, 2048):
            hi = min(lo + 2048, n)
            vt = f["value_target"][lo:hi]
            vm = f["value_mask"][lo:hi]
            rm = f["r_mask"][lo:hi]
            for key, val in [("-1", -1.0), ("0", 0.0), ("0.9", GAMMA), ("1", 1.0)]:
                cnt[key] += int(np.isclose(vt, val).sum())
            known = (np.isclose(vt, -1.0) | np.isclose(vt, 0.0) |
                     np.isclose(vt, GAMMA) | np.isclose(vt, 1.0))
            cnt["other"] += int((~known).sum())
            lm = (vm > 0.5) & (rm > 0.5)               # reachable-and-tried = the TRAINED cells
            masked += int((~lm).sum())
            tvt = vt[lm]
            tr_tot += int(tvt.size)
            for key, val in [("0", 0.0), ("0.9", GAMMA), ("1", 1.0)]:
                tr_cnt[key] += int(np.isclose(tvt, val).sum())
            tr_known = (np.isclose(tvt, 0.0) | np.isclose(tvt, GAMMA) | np.isclose(tvt, 1.0))
            tr_cnt["other"] += int((~tr_known).sum())
    print(f"\n===== pooled Q2 H5: {out_path} =====")
    print(f"  rows (episodes/nodes) : {n}")
    print(f"  cells (n*60*{NUM_DEPTHS})     : {cells}")
    print(f"  value_target over ALL cells:")
    for key in ["-1", "0", "0.9", "1", "other"]:
        print(f"      == {key:5s} : {cnt[key]:11d}  ({100.0*cnt[key]/cells:6.3f}%)")
    print(f"      masked (loss_mask==0; not trained): {masked:11d}  ({100.0*masked/cells:6.3f}%)")
    print(f"  value_target over TRAINED cells (reachable-and-tried = value_mask*r_mask): {tr_tot} cells")
    for key in ["0", "0.9", "1", "other"]:
        frac = 100.0 * tr_cnt[key] / tr_tot if tr_tot else 0.0
        print(f"      == {key:5s} : {tr_cnt[key]:11d}  ({frac:6.3f}% of trained)")
    print("=========================================================\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung1", required=True)
    ap.add_argument("--rung2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(a.rung1, "r") as h1, h5py.File(a.rung2, "r") as h2:
        n1 = int(h1["value_target"].shape[0])
        sm = h2["setup_moved"][:]
        idx2 = np.where(sm != 0)[0]
        n2 = len(idx2)
        n2_drop = int((sm == 0).sum())
        N = n1 + n2
        print(f"[pool] rung1 rows={n1}  rung2 rows={len(sm)} "
              f"(keep setup_moved!=0 -> {n2}, drop setup_moved==0 -> {n2_drop}, "
              f"{100.0*n2_drop/max(len(sm),1):.1f}%)  ==> pooled N={N}", flush=True)

        with h5py.File(a.out, "w") as fo:
            # allocate output datasets (ctx uncompressed float16, matching sources)
            fo.create_dataset("ctx", shape=(N, 5, 64, 64), dtype=np.float16)
            fo.create_dataset("contact_px", shape=(N, 60, 2), dtype=np.float32)
            fo.create_dataset("r_mask", shape=(N, 60, NUM_DEPTHS), dtype=np.float32)
            fo.create_dataset("value_target", shape=(N, 60, NUM_DEPTHS), dtype=np.float32)
            fo.create_dataset("value_mask", shape=(N, 60, NUM_DEPTHS), dtype=np.float32)
            fo.create_dataset("xml", shape=(N,), dtype=str_dt)

            # rung-1 block (all rows)
            w1 = _copy_block(h1, fo, np.arange(n1), 0, COMMON, chunk=a.chunk)
            # rung-2 block (setup_moved != 0)
            w2 = _copy_block(h2, fo, idx2, w1, COMMON, chunk=a.chunk)
            assert w1 + w2 == N, (w1, w2, N)

            fo.attrs["n_samples"] = N
            fo.attrs["num_depths"] = NUM_DEPTHS
            fo.attrs["gamma"] = GAMMA
            fo.attrs["n_rung1"] = n1
            fo.attrs["n_rung2_kept"] = n2
            fo.attrs["n_rung2_dropped_setup_moved0"] = n2_drop
            fo.attrs["source_rung1"] = a.rung1
            fo.attrs["source_rung2"] = a.rung2
            fo.attrs["label_scheme"] = (
                "Q2 pooled depth-<=2 value field. value_target{-1,0,gamma=0.9,1}+value_mask over "
                "rung-1 (depth-1, one row/episode) + rung-2 (per-tree-node, setup_moved==0 nodes "
                "DROPPED). TRAIN on reachable-and-tried cells (value_mask*r_mask): targets in "
                "{0,0.9,1}; the -1 unreachable band and untried MASK are excluded (reachable-only arm)."
            )
    _report(a.out)


if __name__ == "__main__":
    main()
