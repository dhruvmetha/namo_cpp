#!/usr/bin/env python3
"""Join more_build_h5.py part files, built over consecutive episode chunks, into one training H5.

Parts are given in episode order. Per-state arrays are copied chunk by chunk without decompressing
(every array is chunked one state per chunk), so the join costs I/O only. The episode ordinals must
run on across parts with no gap or overlap, which proves the parts cover the ordered collection
once each.

With --cuts (more_cut.py's cuts.json), each budget cut's state count is stored as an attribute
cut_<label>_states: the cut's states are the first that many, because the states are ordered by
episode and a cut is a prefix of episodes.

  python more_merge_h5.py --parts p0000.h5 p0001.h5 ... --cuts cuts.json --out more_224.h5
"""
import argparse
import json

import h5py
import numpy as np

PER_STATE = ("ctx", "contact_px", "label", "weight", "reach_mask", "evidence")
SMALL = ("xml", "object_id", "episode")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuts", default="")
    args = ap.parse_args()

    sizes, attrs, next_ep, eps, skipped = [], None, None, 0, 0
    for p in args.parts:
        with h5py.File(p, "r") as f:
            a = {k: f.attrs[k] for k in ("render_size", "visit_cap", "schema")}
            if attrs is None:
                attrs = a
            assert a == attrs, f"{p}: attrs {a} differ from {attrs}"
            off, n_eps = int(f.attrs["episode_offset"]), int(f.attrs["episodes"])
            assert next_ep is None or off == next_ep, f"{p}: episode offset {off}, expected {next_ep}"
            ep = f["episode"][:]
            assert ep.size == 0 or (ep.min() >= off and ep.max() < off + n_eps and np.all(np.diff(ep) >= 0)), p
            next_ep = off + n_eps
            eps += n_eps
            skipped += int(f.attrs["skipped_states"])
            sizes.append(f["ctx"].shape[0])
    total = sum(sizes)
    print(f"{len(args.parts)} parts, {eps} episodes, {total} states, {skipped} skipped states", flush=True)

    with h5py.File(args.parts[0], "r") as f0, h5py.File(args.out, "w") as out:
        for k in PER_STATE:
            d = f0[k]
            out.create_dataset(k, shape=(total,) + d.shape[1:], dtype=d.dtype, compression="lzf",
                               chunks=(1,) + d.shape[1:])
        for k in SMALL:
            d = f0[k]
            out.create_dataset(k, shape=(total,), dtype=d.dtype, chunks=(4096,))
        base = 0
        for p, n in zip(args.parts, sizes):
            with h5py.File(p, "r") as f:
                for k in PER_STATE:
                    src, dst = f[k].id, out[k].id
                    tail = (0,) * (len(f[k].shape) - 1)
                    for i in range(n):
                        mask, chunk = src.read_direct_chunk((i,) + tail)
                        dst.write_direct_chunk((base + i,) + tail, chunk, mask)
                for k in SMALL:
                    out[k][base:base + n] = f[k][:]
            base += n
            print(f"joined {p} ({n} states, {base}/{total})", flush=True)
        for k, v in attrs.items():
            out.attrs[k] = v
        out.attrs["episodes"] = eps
        out.attrs["skipped_states"] = skipped
        out.attrs["parts"] = len(args.parts)
        if args.cuts:
            ep = out["episode"][:]
            for c in json.load(open(args.cuts))["cuts"]:
                if c["episodes"] <= eps:
                    k = int(np.searchsorted(ep, c["episodes"], side="left"))
                    out.attrs[f"cut_{c['cut']}_states"] = k
                    print(f"cut {c['cut']}: {c['episodes']} episodes, {k} states", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
