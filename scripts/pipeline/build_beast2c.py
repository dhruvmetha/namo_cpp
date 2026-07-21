#!/usr/bin/env python3
"""beast2c_{ceil,hard}.h5 — beast-2c ablation [USER 2026-07-21]: remove ALL fully-dead boards.

From beast2_exh_ceil.h5, keep only rows (boards) with >=1 POSITIVE cell — a verified opener (1.0)
or an exact verified setup (0.9 with ceiling_mask==0). Drops the ~97%-dead exhausted finish boards
AND the fully-dead root boards; dead signal then lives only in dead cells on live boards.
sample_weight recomputed (root/depth2 ratio changes drastically). beast2c_hard derived from
beast2c_ceil by the standard transform (all ceiling cells -> 0.0, ceiling_mask zeroed) — the
identical-rows soft/hard pair, third rung of the label-rule ladder.
"""
import shutil
import time

import h5py
import numpy as np

SRC = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/beast2_exh_ceil.h5"
OUT_C = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/beast2c_ceil.h5"
OUT_H = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/beast2c_hard.h5"
R1_N = 163597
CH = 20000
t0 = time.time()

f = h5py.File(SRC, "r")
N = int(f.attrs["n_samples"])
is_root = f["is_root"][:]

# pass 1: keep mask = board has >=1 exact positive cell (1.0 opener, or 0.9-exact setup)
keep = np.zeros(N, bool)
for s in range(0, N, CH):
    e = min(s + CH, N)
    vt = f["value_target"][s:e]
    vm = f["value_mask"][s:e]
    rm = f["r_mask"][s:e]
    cm = f["ceiling_mask"][s:e]
    pos = (vm == 1) & (rm == 1) & (cm == 0) & (vt >= 0.89)
    keep[s:e] = pos.any(axis=(1, 2))
    if (s // CH) % 10 == 0:
        print(f"  scan {e}/{N} ({time.time()-t0:.0f}s)", flush=True)

r1_drop = int((~keep[:R1_N]).sum())
r2_root_drop = int((~keep[R1_N:] & (is_root[R1_N:] == 1)).sum())
r2_d2_drop = int((~keep[R1_N:] & (is_root[R1_N:] == 0)).sum())
NO = int(keep.sum())
n_root = int(is_root[keep].sum())
n_d2 = NO - n_root
w_root, w_d2 = NO / (2.0 * n_root), NO / (2.0 * n_d2)
print(f"in={N} keep={NO} | dropped: R1 roots {r1_drop:,} · R2 roots {r2_root_drop:,} · "
      f"R2 finish boards {r2_d2_drop:,} | out root={n_root:,} d2={n_d2:,} w={w_root:.3f}/{w_d2:.3f}", flush=True)

cols = list(f.keys())
out = h5py.File(OUT_C, "w")
for c in cols:
    src = f[c]
    if src.dtype == object or h5py.check_string_dtype(src.dtype):
        out.create_dataset(c, shape=(NO,), dtype=h5py.string_dtype())
    else:
        out.create_dataset(c, shape=(NO,) + src.shape[1:], dtype=src.dtype,
                           compression="lzf" if len(src.shape) > 1 else None,
                           chunks=((1,) + src.shape[1:]) if len(src.shape) > 1 else None)
out.attrs["n_samples"] = NO

off = 0
for s in range(0, N, CH):
    e = min(s + CH, N)
    m = keep[s:e]
    k = int(m.sum())
    if k == 0:
        continue
    for c in cols:
        if c == "sample_weight":
            out[c][off:off + k] = np.where(f["is_root"][s:e][m] == 1, w_root, w_d2).astype(np.float32)
        else:
            out[c][off:off + k] = f[c][s:e][m]
    off += k
    if (s // CH) % 10 == 0:
        print(f"  copy {off}/{NO} ({time.time()-t0:.0f}s)", flush=True)
assert off == NO, (off, NO)
out.close()
f.close()
print(f"ceil DONE {OUT_C} rows={NO} ({time.time()-t0:.0f}s)", flush=True)

# hard twin: identical rows, all ceilings -> 0.0
shutil.copyfile(OUT_C, OUT_H)
fh = h5py.File(OUT_H, "r+")
n_zeroed = 0
for s in range(0, NO, CH):
    e = min(s + CH, NO)
    cm = fh["ceiling_mask"][s:e]
    if not (cm == 1).any():
        continue
    vt = fh["value_target"][s:e]
    hit = cm == 1
    vt[hit] = 0.0
    fh["value_target"][s:e] = vt
    fh["ceiling_mask"][s:e] = np.zeros_like(cm)
    n_zeroed += int(hit.sum())
for s in range(0, NO, 200000):
    e = min(s + 200000, NO)
    assert not (fh["ceiling_mask"][s:e] == 1).any()
fh.close()
print(f"hard DONE {OUT_H}: ceilings zeroed {n_zeroed:,} ({time.time()-t0:.0f}s)", flush=True)
