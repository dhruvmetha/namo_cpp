#!/usr/bin/env python3
"""beast2_exh_ceil.h5 — the grammar-correct round-2 training set [USER corrections 2026-07-21].

From beast2_all.h5, three changes to the R2 block (R1 block [0, R1_N) untouched — already c081 grammar):
  1. DROP sparse top-5-hit finish boards (n_win >= 1 AND n_tried <= 5, the early-stop signature):
     only EXHAUSTIVELY-swept finish boards train. Root rows all stay.
  2. CEILING restamp on tried-dead cells — a full sweep proves failure within the 2-push horizon only:
     root dead 0.0 -> 0.81 (= gamma^2: could open at depth 3+), finish dead 0.0 -> 0.9 (= gamma: proven
     not-opener, could be a deeper setup). ceiling_mask=1. Same rule as build_beast0a_h5.py.
  3. sample_weight recomputed for the new root/depth2 counts (arm-B 50/50 expected exposure).

Row alignment: beast2_all's R2 block preserves round2_raw order under the build's keep2 mask; keep2 is
recomputed here identically (node_kind in {root, depth2} AND room not in eval CSV) and count-asserted.
"""
import csv
import os
import time

import h5py
import numpy as np

ALL = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/beast2_all.h5"
RAW = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/round2_raw.h5"
EVAL_CSV = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/identity_eval.csv"
OUT = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/beast2_exh_ceil.h5"
R1_N = 163597
CH = 20000
t0 = time.time()

eval_rooms = set()
with open(EVAL_CSV) as fi:
    for row in csv.DictReader(fi):
        eval_rooms.add(os.path.dirname(row["xml"]))

raw = h5py.File(RAW, "r")
nk = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in raw["node_kind"][:]])
xmlr = [x.decode() if isinstance(x, bytes) else str(x) for x in raw["xml"][:]]
in_eval = np.array([os.path.dirname(x) in eval_rooms for x in xmlr])
keep2 = np.isin(nk, ["root", "depth2"]) & ~in_eval
n_win = raw["n_win"][:][keep2]
n_tried = raw["n_tried"][:][keep2]
nk2 = nk[keep2]
raw.close()

fa = h5py.File(ALL, "r")
NT = int(fa.attrs["n_samples"])
assert NT - R1_N == int(keep2.sum()), (NT - R1_N, int(keep2.sum()))  # alignment guard
is_root_all = fa["is_root"][:]
assert (is_root_all[R1_N:] == (nk2 == "root").astype(np.int8)).all(), "R2 block order mismatch"

sparse_hit = (nk2 == "depth2") & (n_win >= 1) & (n_tried <= 5)
keep_r2 = ~sparse_hit
keep_full = np.concatenate([np.ones(R1_N, bool), keep_r2])
NO = int(keep_full.sum())
n_root = int(is_root_all[keep_full].sum())
n_d2 = NO - n_root
w_root, w_d2 = NO / (2.0 * n_root), NO / (2.0 * n_d2)
print(f"in={NT} dropped_sparse_hit_boards={int(sparse_hit.sum())} out={NO} root={n_root} d2={n_d2} "
      f"w_root={w_root:.3f} w_d2={w_d2:.3f}", flush=True)

cols = list(fa.keys())
out = h5py.File(OUT, "w")
for c in cols:
    src = fa[c]
    if src.dtype == object or h5py.check_string_dtype(src.dtype):
        out.create_dataset(c, shape=(NO,), dtype=h5py.string_dtype())
    else:
        out.create_dataset(c, shape=(NO,) + src.shape[1:], dtype=src.dtype,
                           compression="lzf" if len(src.shape) > 1 else None,
                           chunks=((1,) + src.shape[1:]) if len(src.shape) > 1 else None)
out.attrs["n_samples"] = NO

off = 0
n_root_stamp = n_d2_stamp = 0
for s in range(0, NT, CH):
    e = min(s + CH, NT)
    m = keep_full[s:e]
    k = int(m.sum())
    if k == 0:
        continue
    blk = {c: fa[c][s:e][m] for c in cols}
    r2 = np.arange(s, e)[m] >= R1_N                       # which surviving rows are R2
    if r2.any():
        vt, vm, rm, cm = blk["value_target"], blk["value_mask"], blk["r_mask"], blk["ceiling_mask"]
        dead = (vm == 1) & (rm == 1) & (vt == 0.0) & r2[:, None, None]
        rootrow = blk["is_root"] == 1
        rd = dead & rootrow[:, None, None]
        dd = dead & ~rootrow[:, None, None]
        vt[rd] = 0.81
        vt[dd] = 0.9
        cm[dead] = 1.0
        n_root_stamp += int(rd.sum())
        n_d2_stamp += int(dd.sum())
    blk["sample_weight"] = np.where(blk["is_root"] == 1, w_root, w_d2).astype(np.float32)
    for c in cols:
        out[c][off:off + k] = blk[c]
    off += k
    if (s // CH) % 10 == 0:
        print(f"  {off}/{NO} ({time.time()-t0:.0f}s)", flush=True)
assert off == NO, (off, NO)

# count-assert: no exact-0 tried cell remains anywhere in the R2 region of the output
for s in range(0, NO, 200000):
    e = min(s + 200000, NO)
    vt = out["value_target"][s:e]; vm = out["value_mask"][s:e]; rm = out["r_mask"][s:e]
    cm = out["ceiling_mask"][s:e]
    bad = (vm == 1) & (rm == 1) & (vt == 0.0) & (cm == 0)
    r2_start = max(0, R1_N - s)                            # only the R2 region must be clean
    n_bad = int(bad[r2_start:].sum()) if e > R1_N else 0
    assert n_bad == 0, f"exact-0 uncensored tried cell in R2 region at [{s},{e})"
out.close()
print(f"DONE {OUT}: rows={NO}; root 0->0.81: {n_root_stamp:,}; finish 0->0.9: {n_d2_stamp:,} "
      f"({time.time()-t0:.0f}s)", flush=True)
