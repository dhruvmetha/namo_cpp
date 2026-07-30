#!/usr/bin/env python3
"""Append exactly the requested missing episode trees to a rung-2 GT H5."""
import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from namo.paths import resolve


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _target_keys(path, alignment_key):
    alignment = json.load(open(path))
    rows = alignment[alignment_key]
    keys = {(str(resolve(row["xml"])), row["object_id"]) for row in rows}
    if len(keys) != len(rows):
        raise RuntimeError("duplicate target (xml, object_id) keys")
    return keys


def _row_keys(h5):
    return [
        (str(resolve(_text(xml))), _text(object_id))
        for xml, object_id in zip(h5["xml"][:], h5["object_id"][:])
    ]


def _create_like(dst, name, src, size):
    shape = (size,) + src.shape[1:]
    kwargs = {}
    if src.chunks is not None:
        kwargs["chunks"] = (min(src.chunks[0], size),) + src.chunks[1:]
    if src.compression is not None:
        kwargs["compression"] = src.compression
        kwargs["compression_opts"] = src.compression_opts
        kwargs["shuffle"] = src.shuffle
        kwargs["fletcher32"] = src.fletcher32
    return dst.create_dataset(name, shape=shape, dtype=src.dtype, **kwargs)


def _copy_rows(src, dst, dst_start, indices):
    for name in dst:
        dataset = src[name]
        if indices is None:
            for start in range(0, len(dataset), 128):
                end = min(start + 128, len(dataset))
                dst[name][dst_start + start:dst_start + end] = dataset[start:end]
        else:
            values = dataset[np.asarray(indices, dtype=np.int64)]
            dst[name][dst_start:dst_start + len(indices)] = values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--fill-glob", required=True)
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--alignment-key", default="manifest_missing_gt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    targets = _target_keys(args.alignment, args.alignment_key)
    fill_paths = sorted(glob.glob(args.fill_glob, recursive=True))
    if not fill_paths:
        raise RuntimeError(f"no fill H5s matched {args.fill_glob}")

    selections = []
    root_counts = Counter()
    with h5py.File(args.base, "r") as base:
        base_keys = _row_keys(base)
        base_kinds = [_text(value) for value in base["node_kind"][:]]
        base_roots = {key for key, kind in zip(base_keys, base_kinds) if kind == "root"}
        overlap = targets & base_roots
        if overlap:
            raise RuntimeError(f"{len(overlap)} target roots already exist in base H5")
        base_datasets = set(base.keys())
        base_n = len(base["xml"])

    for path in fill_paths:
        with h5py.File(path, "r") as fill:
            missing_datasets = base_datasets - set(fill.keys())
            if missing_datasets:
                raise RuntimeError(f"{path} lacks base datasets: {sorted(missing_datasets)}")
            keys = _row_keys(fill)
            kinds = [_text(value) for value in fill["node_kind"][:]]
            indices = [i for i, key in enumerate(keys) if key in targets]
            for i in indices:
                if kinds[i] == "root":
                    root_counts[keys[i]] += 1
            if indices:
                selections.append((path, indices))

    missing = targets - set(root_counts)
    duplicate = {key: count for key, count in root_counts.items() if count != 1}
    if missing or duplicate:
        raise RuntimeError(f"target root mismatch: missing={len(missing)} duplicate={duplicate}")

    fill_n = sum(len(indices) for _, indices in selections)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with h5py.File(args.base, "r") as base, h5py.File(tmp, "w") as merged:
        for key, value in base.attrs.items():
            merged.attrs[key] = value
        merged.attrs["n_samples"] = base_n + fill_n
        merged.attrs["targeted_gt_roots_added"] = len(targets)
        merged.attrs["targeted_gt_alignment_key"] = args.alignment_key
        for name, dataset in base.items():
            _create_like(merged, name, dataset, base_n + fill_n)
        _copy_rows(base, merged, 0, None)
        cursor = base_n
        for path, indices in selections:
            with h5py.File(path, "r") as fill:
                _copy_rows(fill, merged, cursor, indices)
            cursor += len(indices)
        if cursor != base_n + fill_n:
            raise RuntimeError("merged row count mismatch")
    os.replace(tmp, out)
    print(json.dumps({
        "base_rows": base_n,
        "target_roots_added": len(targets),
        "target_tree_rows_added": fill_n,
        "merged_rows": base_n + fill_n,
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
