#!/usr/bin/env python3
"""Aquaman round-1 final merge: old_refreshed.h5 + assemble part-H5s -> aquaman1_train.h5.

Intersection columns (guess_mask present in all inputs). sample_weight recomputed GLOBALLY
(beast arm-B convention: 50/50 expected exposure root vs post-push over the merged counts).
"""
import sys
from glob import glob

import h5py
import numpy as np

import sys as _sys
ARM = _sys.argv[1] if len(_sys.argv) > 1 else "B1"
OLD = "/common/users/dm1487/scratch_namo/aquaman/round1/old_refreshed.h5"
PARTS = sorted(glob("/common/users/dm1487/scratch_namo/aquaman/round2/assemble_bs1/part_*.h5"))
OUT = f"/common/users/dm1487/scratch_namo/aquaman/round2/aquaman2_train_R{ARM}.h5"
CH = 4096

ins = [h5py.File(OLD, "r")] + [h5py.File(p, "r") for p in PARTS]
cols = set(ins[0].keys())
for f in ins[1:]:
    cols &= set(f.keys())
cols = sorted(cols)
sizes = [f["ctx"].shape[0] for f in ins]
N = sum(sizes)
print(f"inputs={len(ins)} sizes={sizes} N={N} cols={cols}", flush=True)

is_root = np.concatenate([f["is_root"][:] for f in ins]).astype(bool)
n_root = int(is_root.sum())
w_root, w_child = N / (2.0 * n_root), N / (2.0 * max(N - n_root, 1))
print(f"root={n_root} child={N-n_root} w_root={w_root:.3f} w_child={w_child:.3f}", flush=True)

out = h5py.File(OUT, "w")
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
            if c == "sample_weight":
                continue
            blk = f[c][s:e]
            # B2: delete the OLD block's mute caps (first input file only): capped cells leave the loss.
            if ARM == "B2" and f is ins[0] and c in ("value_mask", "ceiling_mask"):
                cm = ins[0]["ceiling_mask"][s:e] > 0.5
                gm = ins[0]["guess_mask"][s:e] > 0.5 if "guess_mask" in ins[0] else np.zeros_like(cm, bool)
                kill = cm & ~gm
                blk = blk.copy(); blk[kill] = 0.0
            out[c][off + s:off + e] = blk
    off += sz
    print(f"copied {off}/{N}", flush=True)
out["sample_weight"][:] = np.where(is_root, w_root, w_child).astype(np.float32)
out.attrs["n_samples"] = N
out.close()
print("wrote", OUT, flush=True)
