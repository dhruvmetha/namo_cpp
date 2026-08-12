#!/usr/bin/env python3
"""Hybrid corpus [EXP-2026-08-09 2x2 follow-up]: old-corpus root rows + family child boards.

The 2x2 localized the 1p deficit to root supervision CONTENT (board count x sweep depth), which
no cell varied. This corpus replaces the family corpus's thin roots (cap-12 sweeps, 188k boards)
with the old corpus's rows (257k rows, d20-deep exhaustive 1p labels) while keeping the family
children that broke the V5 wall. 66% of family child episodes share (xml, object_id) with an
old-corpus row, so the episode-grouped family lists join across sources.

Label harmonization (MANDATORY): old corpus encodes setup=0.5, family encodes setup=0.9. The rank
losses tier by label VALUE, so mixed encodings would rank 0.9-setups above 0.5-setups. Old rows
are remapped 0.5 -> 0.9 here (the family convention EGMMF/R1 trained with).

Columns = intersection needed by Q2ValueDataset (ctx, r_mask, value_target, value_mask,
ceiling_mask, contact_px, xml, object_id) + chain_depth (old rows: from is_root; family: kept)
so NAMO_ROOT_FRAC stays usable. Output per-row chunks + lzf (gzip auto-chunk kills DataLoaders).

Usage:
  python build_hybrid_h5.py --old "$NAMO_SCRATCH/aquaman/round0/arjuna0v2_train.h5" \
      --family "$NAMO_SCRATCH/aquaman/round0/family0_train_v2.h5" \
      --out "$NAMO_SCRATCH/aquaman/round0/hybrid_train_v1.h5"
"""
import argparse

import h5py
import numpy as np

COLS = ["ctx", "r_mask", "value_target", "value_mask", "ceiling_mask", "contact_px",
        "xml", "object_id", "chain_depth"]
BATCH = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    old = h5py.File(a.old, "r")
    fam = h5py.File(a.family, "r")
    fam_sel = np.nonzero(fam["chain_depth"][:] == 2)[0]          # children only
    n_old, n_fam = old["ctx"].shape[0], len(fam_sel)
    n = n_old + n_fam
    print(f"old rows: {n_old}  family child rows: {n_fam}  total: {n}")

    out = h5py.File(a.out, "w")
    str_dt = h5py.string_dtype()
    for c in COLS:
        src = fam[c]                                              # family file has every column
        if src.dtype == object or c in ("xml", "object_id"):
            out.create_dataset(c, shape=(n,), dtype=str_dt)
        else:
            shape = (n,) + src.shape[1:]
            chunks = (1,) + src.shape[1:] if src.ndim > 1 else None
            out.create_dataset(c, shape=shape, dtype=src.dtype,
                               chunks=chunks, compression="lzf" if src.ndim > 1 else None)

    def put(col, dst_off, data):
        out[col][dst_off:dst_off + len(data)] = data

    dec = lambda arr: [x.decode() if isinstance(x, bytes) else str(x) for x in arr]
    # old corpus first
    for s in range(0, n_old, BATCH):
        e = min(s + BATCH, n_old)
        for c in ("ctx", "r_mask", "value_mask", "ceiling_mask", "contact_px"):
            put(c, s, old[c][s:e])
        vt = old["value_target"][s:e].astype(np.float32)
        vt[np.isclose(vt, 0.5)] = 0.9                             # harmonize setup tier
        put("value_target", s, vt)
        put("xml", s, dec(old["xml"][s:e]))
        put("object_id", s, dec(old["object_id"][s:e]))
        put("chain_depth", s, np.where(old["is_root"][s:e] > 0, 1, 2).astype(np.int8))
    # family children
    for s in range(0, n_fam, BATCH):
        idx = fam_sel[s:s + BATCH]
        for c in ("ctx", "r_mask", "value_target", "value_mask", "ceiling_mask",
                  "contact_px", "chain_depth"):
            put(c, n_old + s, fam[c][idx])
        put("xml", n_old + s, dec(fam["xml"][idx]))
        put("object_id", n_old + s, dec(fam["object_id"][idx]))

    out.attrs["n_samples"] = n
    out.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
