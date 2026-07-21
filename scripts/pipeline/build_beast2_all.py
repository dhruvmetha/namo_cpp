#!/usr/bin/env python3
"""beast-2 combined training H5: round-1 clean (censored roots) + round-2 slice (exact roots + post-setup boards).

Sources:
  R1 = beast1_clean.h5 (166,325 root rows, censored c081 grammar, ceiling_mask carried as-is)
  R2 = round2_raw.h5   (build_rung2_h5 node rows from the exhaust-on-miss dead-bank slice; EXACT labels,
                        ceiling_mask=0 by construction — full sweep proves dead, no censoring needed)

Rules (locked with user 2026-07-20):
  - node_kind='depth2_noop' rows DROPPED from training: setup didn't move the object, so ctx duplicates the
    root board with a different partial mask — the duplicate/contradiction class the clean-pair purge removed.
  - eval rooms (identity_eval.csv, room = dirname(xml)) excluded from training -> round2_eval.h5 instead.
  - information dominance on ROOT keys (xml, object_id): an R2 exact root row supersedes any R1 censored row.
  - is_root i8 tag + sample_weight f32 (root vs depth2 expected-exposure ~50/50) for the arm-B
    WeightedRandomSampler; arm A ignores both.
"""
import csv
import os
import time

import h5py
import numpy as np

R1 = "/common/users/dm1487/scratch_namo/curriculum2/beast/round1/beast1_clean.h5"
R2 = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/round2_raw.h5"
EVAL_CSV = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/identity_eval.csv"
OUT_TRAIN = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/beast2_all.h5"
OUT_EVAL = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/round2_eval.h5"
BASE = ["ctx", "contact_px", "r_mask", "value_target", "value_mask", "ceiling_mask", "xml", "object_id"]
CH = 20000
t0 = time.time()

dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]

eval_rooms = set()
with open(EVAL_CSV) as fi:
    for row in csv.DictReader(fi):
        eval_rooms.add(os.path.dirname(row["xml"]))
print(f"eval rooms: {len(eval_rooms)}", flush=True)

h2 = h5py.File(R2, "r")
N2 = h2["ctx"].shape[0]
xml2 = dec(h2["xml"][:])
oid2 = dec(h2["object_id"][:])
nk2 = dec(h2["node_kind"][:])
is_eval2 = np.array([os.path.dirname(x) in eval_rooms for x in xml2])
kind_counts = {k: nk2.count(k) for k in ("root", "depth2", "depth2_noop")}
nk_ok = np.array([nk in ("root", "depth2") for nk in nk2])
keep2 = nk_ok & ~is_eval2
r2_root_keys = {(xml2[i], oid2[i]) for i in np.where(keep2)[0] if nk2[i] == "root"}
print(f"R2 rows={N2} kinds={kind_counts} keep_train={int(keep2.sum())} eval_rows={int(is_eval2.sum())} "
      f"noop_dropped={int((~nk_ok & ~is_eval2).sum())}", flush=True)

h1 = h5py.File(R1, "r")
N1 = h1["ctx"].shape[0]
xml1 = dec(h1["xml"][:])
oid1 = dec(h1["object_id"][:])
dom_drop = np.array([(x, o) in r2_root_keys for x, o in zip(xml1, oid1)])
evalroom_drop = np.array([os.path.dirname(x) in eval_rooms for x in xml1])
keep1 = ~(dom_drop | evalroom_drop)
print(f"R1 rows={N1} dominance_dropped={int(dom_drop.sum())} evalroom_dropped={int((evalroom_drop & ~dom_drop).sum())} "
      f"keep={int(keep1.sum())}", flush=True)

n1k, n2k = int(keep1.sum()), int(keep2.sum())
NT = n1k + n2k
root2 = np.array([nk == "root" for nk in nk2])
n_root = n1k + int((keep2 & root2).sum())
n_d2 = NT - n_root
w_root, w_d2 = NT / (2.0 * n_root), NT / (2.0 * n_d2)
print(f"train rows={NT} (R1 {n1k} + R2 {n2k}) root={n_root} depth2={n_d2} w_root={w_root:.3f} w_d2={w_d2:.3f}", flush=True)


def make_out(path, n, with_meta):
    o = h5py.File(path, "w")
    for c in BASE:
        src = h1["ceiling_mask"] if c == "ceiling_mask" else h2[c]
        if c in ("xml", "object_id"):
            o.create_dataset(c, shape=(n,), dtype=h5py.string_dtype())
        else:
            o.create_dataset(c, shape=(n,) + src.shape[1:], dtype=src.dtype,
                             compression="lzf", chunks=(1,) + src.shape[1:])
    o.create_dataset("is_root", shape=(n,), dtype=np.int8)
    o.create_dataset("sample_weight", shape=(n,), dtype=np.float32)
    if with_meta:
        o.create_dataset("node_kind", shape=(n,), dtype=h5py.string_dtype())
        o.create_dataset("parent_edge", shape=(n,), dtype=np.int16)
        o.create_dataset("parent_depth", shape=(n,), dtype=np.int16)
    o.attrs["n_samples"] = n
    return o


def stream(src, mask, out, off, has_ceiling, is_root_arr, xml_l, oid_l, meta=False):
    n_src = len(mask)
    for s in range(0, n_src, CH):
        e = min(s + CH, n_src)
        m = mask[s:e]
        k = int(m.sum())
        if k == 0:
            continue
        for c in BASE:
            if c == "ceiling_mask" and not has_ceiling:
                out[c][off:off + k] = np.zeros((k, 60, 5), np.float32)
            elif c in ("xml", "object_id"):
                vals = xml_l if c == "xml" else oid_l
                out[c][off:off + k] = [vals[i] for i in range(s, e) if m[i - s]]
            else:
                out[c][off:off + k] = src[c][s:e][m]
        out["is_root"][off:off + k] = is_root_arr[s:e][m]
        out["sample_weight"][off:off + k] = np.where(is_root_arr[s:e][m] == 1, w_root, w_d2).astype(np.float32)
        if meta:
            out["node_kind"][off:off + k] = [nk2[i] for i in range(s, e) if m[i - s]]
            out["parent_edge"][off:off + k] = src["parent_edge"][s:e][m].astype(np.int16)
            out["parent_depth"][off:off + k] = src["parent_depth"][s:e][m].astype(np.int16)
        off += k
        if (s // CH) % 5 == 0:
            print(f"  {os.path.basename(out.filename)} rows={off} ({time.time()-t0:.0f}s)", flush=True)
    return off


ot = make_out(OUT_TRAIN, NT, with_meta=False)
off = stream(h1, keep1, ot, 0, has_ceiling=True, is_root_arr=np.ones(N1, np.int8), xml_l=xml1, oid_l=oid1)
assert off == n1k, (off, n1k)
off = stream(h2, keep2, ot, off, has_ceiling=False, is_root_arr=root2.astype(np.int8), xml_l=xml2, oid_l=oid2)
assert off == NT, (off, NT)
ot.close()

ne = int(is_eval2.sum())
oe = make_out(OUT_EVAL, ne, with_meta=True)
offe = stream(h2, is_eval2, oe, 0, has_ceiling=False, is_root_arr=root2.astype(np.int8), xml_l=xml2, oid_l=oid2, meta=True)
assert offe == ne, (offe, ne)
oe.close()
print(f"DONE train={NT} -> {OUT_TRAIN}; eval={ne} -> {OUT_EVAL} ({time.time()-t0:.0f}s)", flush=True)
