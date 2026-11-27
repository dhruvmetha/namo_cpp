#!/usr/bin/env python3
"""Write chosen rows of one or more training H5 files into a new H5 (card EXP-2026-11-27-more-spar-equal-rooms).

The sources are read as one table, in the order given: row r of the table is row r - offset of the source
that holds it. --rows is a .npy of table row numbers; they are written in table order (sorted, no repeats),
so rows keep their source order. Every dataset whose leading length is its file's row count is copied, with
its dtype and compression and one row per chunk for arrays (what the trainers' random reads expect; an array
already stored one row per chunk with the same filters is copied as its compressed chunks, without decoding);
strings stay variable-length strings. All sources must have the same per-row datasets. Attributes come
from the first source, minus any whose name starts with a --drop-attr prefix; the output adds n_samples,
subset_sources and the sha256 of the row list.

Usage:
  python subset_h5_rows.py --sources a.h5 b.h5 --rows rows.npy --out subset.h5 [--drop-attr cut_ --drop-attr episodes]
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def filters(d):
    return d.compression, d.compression_opts, d.shuffle, d.fletcher32, d.scaleoffset


def per_row_datasets(f):
    n = int(f.attrs.get("n_samples", f["xml"].shape[0]))
    return n, {k: f[k] for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].shape[:1] == (n,)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--rows", required=True, help=".npy of table row numbers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-attr", action="append", default=[], help="attribute name prefix not to copy")
    ap.add_argument("--block", type=int, default=2048, help="source rows read at a time")
    args = ap.parse_args()

    rows = np.load(args.rows).astype(np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise SystemExit("--rows must be a non-empty 1-D array")
    if np.any(np.diff(rows) <= 0):
        raise SystemExit("--rows must be sorted with no repeats")
    srcs = [h5py.File(p, "r") for p in args.sources]
    tables = [per_row_datasets(f) for f in srcs]
    names = sorted(tables[0][1])
    for p, (_, ds) in zip(args.sources[1:], tables[1:]):
        if sorted(ds) != names:
            raise SystemExit(f"{p} has per-row datasets {sorted(ds)}, first source {names}")
    offsets = np.cumsum([0] + [n for n, _ in tables])
    if rows[0] < 0 or rows[-1] >= offsets[-1]:
        raise SystemExit(f"rows outside the table of {offsets[-1]} rows")
    m = len(rows)
    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with h5py.File(tmp, "w") as g:
        first = tables[0][1]
        for k in names:
            d = first[k]
            if d.dtype.kind in "OSU" or h5py.check_string_dtype(d.dtype):
                g.create_dataset(k, (m,), dtype=h5py.string_dtype())
            else:
                chunks = (1,) + d.shape[1:] if d.ndim > 1 else (True if d.compression else None)
                g.create_dataset(k, (m,) + d.shape[1:], dtype=d.dtype, chunks=chunks,
                                 compression=d.compression, compression_opts=d.compression_opts)
        w = 0
        for s, (n, ds) in enumerate(tables):
            local = rows[(rows >= offsets[s]) & (rows < offsets[s + 1])] - offsets[s]
            # arrays stored one row per chunk with the output's filters: copy the compressed chunks as they are
            direct = [k for k in names if ds[k].ndim > 1 and ds[k].chunks == (1,) + ds[k].shape[1:]
                      and filters(ds[k]) == filters(g[k])]
            for k in direct:
                src, dst, tail = ds[k].id, g[k].id, (0,) * (ds[k].ndim - 1)
                for j, r in enumerate(local):
                    mask, chunk = src.read_direct_chunk((int(r),) + tail)
                    dst.write_direct_chunk((w + j,) + tail, chunk, mask)
            rest = [k for k in names if k not in direct]
            v = w
            for a in range(0, n, args.block):
                pick = local[(local >= a) & (local < a + args.block)]
                if len(pick) == 0:
                    continue
                b = int(pick[-1]) + 1
                for k in rest:
                    block = ds[k][a:b][pick - a]
                    g[k][v:v + len(pick)] = block.astype(object) if block.dtype.kind in "OSU" else block
                v += len(pick)
            w += len(local)
            print(f"source {s} {args.sources[s]}: {len(local)} of {n} rows "
                  f"(chunks copied as stored: {direct})", flush=True)
        assert w == m, (w, m)
        for k, v in srcs[0].attrs.items():
            if not any(k.startswith(p) for p in args.drop_attr + ["n_samples"]):
                g.attrs[k] = v
        g.attrs["n_samples"] = m
        g.attrs["subset_sources"] = json.dumps([str(Path(p).resolve()) for p in args.sources])
        g.attrs["subset_rows_sha256"] = hashlib.sha256(rows.tobytes()).hexdigest()
    tmp.rename(out)
    # spot check: a few written rows equal their source rows in every dataset
    rng = np.random.default_rng(0)
    with h5py.File(out, "r") as g:
        for i in rng.choice(m, min(m, 20), replace=False):
            r = rows[i]; s = int(np.searchsorted(offsets, r, side="right") - 1)
            for k in names:
                a, b = g[k][i], tables[s][1][k][r - offsets[s]]
                if not np.array_equal(np.asarray(a), np.asarray(b)):
                    raise SystemExit(f"row {i} (table row {r}) differs in {k}")
    print(f"wrote {out}: {m} rows, datasets {names}", flush=True)


if __name__ == "__main__":
    main()
