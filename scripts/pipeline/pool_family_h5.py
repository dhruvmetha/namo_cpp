#!/usr/bin/env python3
"""Pool per-shard family H5s (from build_rung2_h5.py --family-select) into ONE train H5.

Generic glob-based concat: intersection of columns across all shard files, ND arrays (chunks/lzf
preserved per-row: chunks=(1,)+shape[1:], compression=lzf -- mirrors what build_rung2_h5.py already
writes per-shard and what scripts/rl_loop/aquaman_concat.py uses for other pooled H5s in this repo.
1-D scalar columns are written uncompressed/contiguous (chunks=None), matching family0_train_v2.h5's
own layout. object-dtype columns (xml, node_kind, object_id) become h5py variable-length strings.

Usage:
  python pool_family_h5.py --shard-glob "$NAMO_SCRATCH/family1/h5_shards/family_shard_*.h5" \
      --out "$NAMO_SCRATCH/family1/family1_train_v1.h5"
"""
import argparse
from glob import glob

import h5py
import numpy as np

CH = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    parts = sorted(glob(a.shard_glob))
    if not parts:
        raise SystemExit(f"no shards matched {a.shard_glob}")
    ins = [h5py.File(p, "r") for p in parts]

    cols = set(ins[0].keys())
    for f in ins[1:]:
        cols &= set(f.keys())
    cols = sorted(cols)
    sizes = [f[cols[0]].shape[0] for f in ins]
    N = sum(sizes)
    print(f"shards={len(ins)} sizes_min={min(sizes)} sizes_max={max(sizes)} N={N} cols={cols}", flush=True)

    out = h5py.File(a.out, "w")
    for c in cols:
        src = ins[0][c]
        if h5py.check_string_dtype(src.dtype) or src.dtype == object:
            out.create_dataset(c, shape=(N,), dtype=h5py.string_dtype())
        else:
            out.create_dataset(c, shape=(N,) + src.shape[1:], dtype=src.dtype,
                                compression="lzf" if len(src.shape) > 1 else None,
                                chunks=((1,) + src.shape[1:]) if len(src.shape) > 1 else None)

    off = 0
    for f, sz in zip(ins, sizes):
        for s in range(0, sz, CH):
            e = min(s + CH, sz)
            for c in cols:
                out[c][off + s:off + e] = f[c][s:e]
        off += sz
        print(f"copied {off}/{N}", flush=True)

    out.attrs["n_samples"] = N
    out.close()
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
