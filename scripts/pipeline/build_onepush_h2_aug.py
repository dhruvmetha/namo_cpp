#!/usr/bin/env python3
"""1-PUSH @ H=2 AUGMENTATION — the H2-encompasses-H1 OOD fix [USER 2026-06-13].

PROBLEM: at H=2 the model dilutes on 1-push-solvable scenes (38->14 hard@1) because only ~16% of its H=2
training rows are 1-push-solvable -> the H=2 head is starved of "opener = 1.0 at H=2" examples (H2/H4 in the
ledger). FIX (free, no new render): take the exhaustive 1-push rows (a scorer H5, e.g. v4_hq_m2b_scorer) and
emit an H=2 COPY of 1-push-solvable rows, labeling ONLY the opener cells = 1.0 (valid: opens-in-1 subset
opens-in-2), masking everything else. The 1-push-FAILED cells are UNKNOWN at H=2 (they might be 2-push setups
we never tested) -> masked out, NOT zeroed (avoids the C15 false-negative bug). Sparse-positive H=2 rows.

DESIGN [USER]: emit a DECENT fraction, not the full set (the v2 WeightedRandomSampler does the final balance).
--max-rows caps the augmentation (default 80k); we don't want to double the dataset or over-represent openers.

Speed: TWO-PASS + BATCHED gather-writes. Pass 1 scans only f_grid (cheap) to find solvable global indices;
subsample to --max-rows; Pass 2 reads ctx/oc/xml/cpx for the chosen indices in batches and writes contiguous
slices (single-row writes to an lzf dataset are pathologically slow -- that was the bug).

Output = scorer H5 (same schema + H=2, onepush_h2_aug=1) to ';'-join into the v2 training mix.

  python scripts/pipeline/build_onepush_h2_aug.py --src-h5 /scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5 \
      --out-h5 /scratch/dm1487/h5/v4_hq_onepush_h2_aug/data.h5 --max-rows 80000
"""
import argparse, os
import h5py, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-h5", default="/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5",
                    help="exhaustive 1-push scorer H5 (ctx/f_grid/r_mask/contact_px/object_center/xml)")
    ap.add_argument("--out-h5", default="/scratch/dm1487/h5/v4_hq_onepush_h2_aug/data.h5")
    ap.add_argument("--max-rows", type=int, default=80000, help="cap on emitted aug rows (subsample if more solvable)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan-chunk", type=int, default=8000)
    ap.add_argument("--write-batch", type=int, default=2000)
    a = ap.parse_args()

    s = h5py.File(a.src_h5, "r")
    N = int(s.attrs.get("n_samples", s["f_grid"].shape[0]))
    C = tuple(s["ctx"].shape[1:])                # (5,64,64)
    has_cpx = "contact_px" in s

    # PASS 1: collect 1-push-solvable global indices (scan f_grid only)
    sol = []
    for c0 in range(0, N, a.scan_chunk):
        c1 = min(c0 + a.scan_chunk, N)
        op = s["f_grid"][c0:c1] >= 0.999
        solvable = op.reshape(c1 - c0, -1).any(axis=1)
        sol.append(np.nonzero(solvable)[0] + c0)
    sol = np.concatenate(sol) if sol else np.array([], dtype=np.int64)
    n_sol = len(sol)
    if n_sol > a.max_rows:
        chosen = np.sort(np.random.RandomState(a.seed).choice(sol, size=a.max_rows, replace=False))
    else:
        chosen = sol
    M = len(chosen)
    print(f"src N={N}  1-push-solvable={n_sol}  emitting M={M} (max_rows={a.max_rows})", flush=True)

    os.makedirs(os.path.dirname(a.out_h5), exist_ok=True)
    d = h5py.File(a.out_h5, "w")
    ds = {
        "ctx": d.create_dataset("ctx", (M, *C), dtype="float32", compression="lzf", chunks=(min(32, M), *C)),
        "f_grid": d.create_dataset("f_grid", (M, 60, 5), dtype="float32", compression="lzf"),
        "r_mask": d.create_dataset("r_mask", (M, 60, 5), dtype="float32", compression="lzf"),
        "ratio": d.create_dataset("ratio", (M,), dtype="float32"),
        "object_center": d.create_dataset("object_center", (M, 2), dtype="float32"),
        "xml": d.create_dataset("xml", (M,), dtype=h5py.string_dtype()),
        "H": d.create_dataset("H", (M,), dtype="int8"),
        "dead": d.create_dataset("dead", (M,), dtype="uint8"),
        "onepush_h2_aug": d.create_dataset("onepush_h2_aug", (M,), dtype="uint8"),
    }
    if has_cpx:
        ds["contact_px"] = d.create_dataset("contact_px", (M, 60, 2), dtype="float32", compression="lzf")

    # PASS 2: batched gather-write (chosen is sorted -> valid h5py fancy index)
    for b in range(0, M, a.write_batch):
        idxb = chosen[b:b + a.write_batch].tolist()
        e = b + len(idxb)
        op = (s["f_grid"][idxb] >= 0.999).astype(np.float32)   # (m,60,5) opener mask
        ds["ctx"][b:e] = s["ctx"][idxb]
        ds["f_grid"][b:e] = op                                  # 1.0 at openers, 0 elsewhere (gated by r_mask)
        ds["r_mask"][b:e] = op                                  # loss only on known openers; rest UNKNOWN (masked)
        ds["ratio"][b:e] = 1.0
        ds["object_center"][b:e] = s["object_center"][idxb]
        ds["xml"][b:e] = s["xml"][idxb]
        ds["H"][b:e] = 2; ds["dead"][b:e] = 0; ds["onepush_h2_aug"][b:e] = 1
        if has_cpx:
            ds["contact_px"][b:e] = s["contact_px"][idxb]
        if b % (a.write_batch * 10) == 0:
            print(f"  [{e}/{M}]", flush=True)

    d.attrs["n_samples"] = M
    d.attrs["source"] = a.src_h5
    d.attrs["note"] = "1-push openers relabeled as H=2 sparse-positive (opener=1.0, rest masked); subsampled"
    s.close(); d.close()
    print(f"wrote {a.out_h5}  rows={M} (H=2 sparse-positive 1-push-opener augmentation)", flush=True)


if __name__ == "__main__":
    main()
