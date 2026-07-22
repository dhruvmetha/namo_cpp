#!/usr/bin/env python3
"""Append an exact 200k Colossus-0 experience block to the beast-2c-d20 H5.

The new block is 166,666 positive-bearing mistake rows plus 33,334 negative-only rows. Positive
post-push rows must have winner_rank > 1; rank-1 finish rows have no within-board ranking mistake.
Negative rows contain only verified ceiling cells, split root/post-push as evenly as supply permits.
Unknown cells remain masked. Selection is deterministic under --seed and the output recomputes the
root/post-push sampler weights over the full d20+Colossus stack.
"""
import argparse
import glob
import json
import os
import random

import h5py
import numpy as np


COLS = ["ctx", "contact_px", "r_mask", "value_target", "value_mask", "ceiling_mask", "xml", "object_id"]
CHUNK = 20000


def _dec(values):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def _candidate_pools(paths):
    positive = []
    negative_root = []
    negative_finish = []
    counts = {"rows": 0, "positive": 0, "rank1_finish_excluded": 0, "negative_root": 0,
              "negative_finish": 0, "noop_excluded": 0, "empty": 0}
    for file_idx, path in enumerate(paths):
        with h5py.File(path, "r") as h5:
            n = len(h5["ctx"])
            kinds = _dec(h5["node_kind"][:])
            winner_rank = h5["winner_rank"][:]
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                vt = h5["value_target"][start:end]
                vm = h5["value_mask"][start:end]
                rm = h5["r_mask"][start:end]
                cm = h5["ceiling_mask"][start:end]
                exact_positive = ((vm == 1) & (rm == 1) & (cm == 0) & (vt >= 0.89)).any(axis=(1, 2))
                ceiling_signal = ((vm == 1) & (rm == 1) & (cm == 1)).any(axis=(1, 2))
                bad_zero = ((vm == 1) & (rm == 1) & (cm == 0) & (vt == 0)).any(axis=(1, 2))
                if bad_zero.any():
                    raise AssertionError(f"false exact-zero reachable label in {path} row {start + int(np.where(bad_zero)[0][0])}")
                for local in range(end - start):
                    idx = start + local
                    if kinds[idx] == "depth2_noop":
                        counts["noop_excluded"] += 1
                        continue
                    root = kinds[idx] == "root"
                    if exact_positive[local]:
                        if root or int(winner_rank[idx]) > 1:
                            positive.append((file_idx, idx))
                            counts["positive"] += 1
                        else:
                            counts["rank1_finish_excluded"] += 1
                    elif ceiling_signal[local]:
                        (negative_root if root else negative_finish).append((file_idx, idx))
                        counts["negative_root" if root else "negative_finish"] += 1
                    else:
                        counts["empty"] += 1
            counts["rows"] += n
    return positive, negative_root, negative_finish, counts


def _sample_negative(root, finish, total, rng):
    root_target = total // 2
    finish_target = total - root_target
    take_root = min(root_target, len(root))
    take_finish = min(finish_target, len(finish))
    remaining = total - take_root - take_finish
    if remaining:
        root_spare = len(root) - take_root
        add_root = min(remaining, root_spare)
        take_root += add_root
        remaining -= add_root
    if remaining:
        finish_spare = len(finish) - take_finish
        add_finish = min(remaining, finish_spare)
        take_finish += add_finish
        remaining -= add_finish
    if remaining:
        raise RuntimeError(f"need {total} negative rows, found only {len(root) + len(finish)}")
    return rng.sample(root, take_root) + rng.sample(finish, take_finish), take_root, take_finish


def _make_output(path, base, n):
    out = h5py.File(path, "x")
    for col in COLS:
        src = base[col]
        if col in ("xml", "object_id"):
            out.create_dataset(col, shape=(n,), dtype=h5py.string_dtype())
        else:
            out.create_dataset(col, shape=(n,) + src.shape[1:], dtype=src.dtype,
                               compression="lzf", chunks=(1,) + src.shape[1:])
    out.create_dataset("is_root", shape=(n,), dtype=np.int8)
    out.create_dataset("sample_weight", shape=(n,), dtype=np.float32)
    out.attrs["n_samples"] = n
    return out


def _copy_base(base, out):
    n = len(base["ctx"])
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        for col in COLS:
            out[col][start:end] = base[col][start:end]
        out["is_root"][start:end] = base["is_root"][start:end]
    return n


def _copy_selected(paths, selected, out, offset):
    by_file = {}
    for file_idx, row_idx in selected:
        by_file.setdefault(file_idx, []).append(row_idx)
    for file_idx in sorted(by_file):
        rows = sorted(by_file[file_idx])
        with h5py.File(paths[file_idx], "r") as src:
            for start in range(0, len(rows), 1000):
                batch = rows[start:start + 1000]
                end = offset + len(batch)
                for col in COLS:
                    out[col][offset:end] = src[col][batch]
                kinds = _dec(src["node_kind"][batch])
                out["is_root"][offset:end] = np.array([kind == "root" for kind in kinds], np.int8)
                offset = end
    return offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-h5", required=True)
    parser.add_argument("--new-h5-glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--new-rows", type=int, default=200000)
    parser.add_argument("--negative-rows", type=int, default=33334)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.new_h5_glob))
    if not paths:
        raise RuntimeError(f"no candidate H5s match {args.new_h5_glob}")
    positive_target = args.new_rows - args.negative_rows
    positive, negative_root, negative_finish, counts = _candidate_pools(paths)
    if len(positive) < positive_target:
        raise RuntimeError(f"need {positive_target} positive/mistake rows, found only {len(positive)}")

    rng = random.Random(args.seed)
    selected_positive = rng.sample(positive, positive_target)
    selected_negative, n_neg_root, n_neg_finish = _sample_negative(
        negative_root, negative_finish, args.negative_rows, rng
    )
    selected = selected_positive + selected_negative
    rng.shuffle(selected)

    with h5py.File(args.base_h5, "r") as base:
        base_n = len(base["ctx"])
        base_rooms = set(_dec(base["xml"][:]))
        selected_rooms = set()
        by_file = {}
        for file_idx, row_idx in selected:
            by_file.setdefault(file_idx, []).append(row_idx)
        for file_idx, rows in by_file.items():
            with h5py.File(paths[file_idx], "r") as src:
                selected_rooms.update(_dec(src["xml"][sorted(rows)]))
        exact_room_overlap = base_rooms & selected_rooms
        if exact_room_overlap:
            raise AssertionError(f"{len(exact_room_overlap)} exact XML paths overlap the d20 base")

        total = base_n + len(selected)
        out = _make_output(args.out, base, total)
        offset = _copy_base(base, out)
        offset = _copy_selected(paths, selected, out, offset)
        assert offset == total
        n_root = int(out["is_root"][:].sum())
        n_finish = total - n_root
        if not n_root or not n_finish:
            raise AssertionError((n_root, n_finish))
        out["sample_weight"][:] = np.where(
            out["is_root"][:] == 1,
            total / (2.0 * n_root),
            total / (2.0 * n_finish),
        ).astype(np.float32)
        out.attrs["base_rows"] = base_n
        out.attrs["colossus0_rows"] = len(selected)
        out.attrs["selection_seed"] = args.seed
        out.close()

    report = {
        "base_h5": os.path.realpath(args.base_h5),
        "candidate_h5s": [os.path.realpath(path) for path in paths],
        "candidate_counts": counts,
        "base_rows": base_n,
        "new_rows": len(selected),
        "new_positive_mistake_rows": len(selected_positive),
        "new_negative_root_rows": n_neg_root,
        "new_negative_finish_rows": n_neg_finish,
        "output_rows": base_n + len(selected),
        "output_root_rows": n_root,
        "output_finish_rows": n_finish,
        "selection_seed": args.seed,
        "exact_xml_overlap_with_base": 0,
    }
    with open(args.report, "x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
