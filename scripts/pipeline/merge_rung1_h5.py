#!/usr/bin/env python3
"""Merge sharded rung-1 H5 parts (from build_rung1_h5.py --shard-*) into one H5 (concat axis 0)."""
import glob
import argparse
import numpy as np
import h5py

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-glob", required=True, help="glob for part_*.h5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = sorted(glob.glob(a.parts_glob))
    print(f"merging {len(parts)} parts")
    keys, buf, total = None, {}, 0
    for p in parts:
        with h5py.File(p, "r") as f:
            if int(f.attrs.get("n_samples", 0)) == 0 or "ctx" not in f:
                print(f"  [empty] {p}")
                continue
            if keys is None:
                keys = list(f.keys())
            for k in keys:
                buf.setdefault(k, []).append(f[k][:])
            total += f["ctx"].shape[0]
    print(f"total rows = {total} across {len(parts)} parts")
    with h5py.File(a.out, "w") as out:
        out.attrs["n_samples"] = total
        out.attrs["num_depths"] = 5
        out.attrs["generation"] = 0
        out.attrs["label_scheme"] = "value_target{-1,0,1}+value_mask; opener=1|tried-no-open=0|unreachable=-1|reach-unsampled=MASK"
        for k in keys:
            cat = np.concatenate(buf[k], axis=0)
            if cat.dtype == object or cat.dtype.kind in ("S", "U"):   # auto-detect string cols (xml, node_kind, ...)
                out.create_dataset(k, data=cat.astype(object), dtype=h5py.string_dtype("utf-8"))
            else:
                out.create_dataset(k, data=cat)
    print(f"wrote {a.out}  ({total} rows)")


if __name__ == "__main__":
    main()
