#!/usr/bin/env python3
"""1-PUSH @ H=2 AUGMENTATION — the H2-encompasses-H1 OOD fix [USER 2026-06-13].

PROBLEM: at H=2 the model dilutes on 1-push-solvable scenes (38→14 hard@1) because only ~16% of its H=2
training rows are 1-push-solvable → the H=2 head is starved of "opener = 1.0 at H=2" examples (H2/H4 in the
ledger). FIX (free, no new render): take the exhaustive 1-push rows (a scorer H5, e.g. v4_hq_m2b_scorer) and
emit an H=2 COPY of every 1-push-solvable row, labeling ONLY the opener cells = 1.0 (valid: opens-in-1 ⊆
opens-in-2), masking everything else. The 1-push-FAILED cells are UNKNOWN at H=2 (they might be 2-push setups
we never tested) → masked out, NOT zeroed (avoids the C15 false-negative bug). Sparse-positive H=2 rows.

Output is a scorer H5 (same schema + H=2) to ';'-join into the v2 training mix. Tag onepush_h2_aug=1 for the
WeightedRandomSampler. ctx / contact_px / object_center / xml are copied verbatim (same crop, same object).

  python scripts/pipeline/build_onepush_h2_aug.py --src-h5 /scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5 \
      --out-h5 /scratch/dm1487/h5/v4_hq_onepush_h2_aug/data.h5
"""
import argparse, os
import h5py, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-h5", default="/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5",
                    help="exhaustive 1-push scorer H5 (ctx/f_grid/r_mask/contact_px/object_center/xml)")
    ap.add_argument("--out-h5", default="/scratch/dm1487/h5/v4_hq_onepush_h2_aug/data.h5")
    ap.add_argument("--chunk", type=int, default=4000)
    a = ap.parse_args()

    s = h5py.File(a.src_h5, "r")
    N = int(s.attrs.get("n_samples", s["f_grid"].shape[0]))
    C = s["ctx"].shape[1:]                       # (5,64,64)
    os.makedirs(os.path.dirname(a.out_h5), exist_ok=True)
    d = h5py.File(a.out_h5, "w")
    has_cpx = "contact_px" in s
    # over-allocate to N, shrink at end
    ds = {
        "ctx": d.create_dataset("ctx", (N, *C), maxshape=(None, *C), dtype="float32", compression="lzf",
                                chunks=(min(32, N), *C)),
        "f_grid": d.create_dataset("f_grid", (N, 60, 5), maxshape=(None, 60, 5), dtype="float32", compression="lzf"),
        "r_mask": d.create_dataset("r_mask", (N, 60, 5), maxshape=(None, 60, 5), dtype="float32", compression="lzf"),
        "ratio": d.create_dataset("ratio", (N,), maxshape=(None,), dtype="float32"),
        "object_center": d.create_dataset("object_center", (N, 2), maxshape=(None, 2), dtype="float32"),
        "xml": d.create_dataset("xml", (N,), maxshape=(None,), dtype=h5py.string_dtype()),
        "H": d.create_dataset("H", (N,), maxshape=(None,), dtype="int8"),
        "dead": d.create_dataset("dead", (N,), maxshape=(None,), dtype="uint8"),
        "onepush_h2_aug": d.create_dataset("onepush_h2_aug", (N,), maxshape=(None,), dtype="uint8"),
    }
    if has_cpx:
        ds["contact_px"] = d.create_dataset("contact_px", (N, 60, 2), maxshape=(None, 60, 2),
                                            dtype="float32", compression="lzf")
    j = 0
    for c0 in range(0, N, a.chunk):
        c1 = min(c0 + a.chunk, N)
        fg = s["f_grid"][c0:c1]                  # (m,60,5)
        op = fg >= 0.999                          # opener cells (==1.0)
        solvable = op.reshape(c1 - c0, -1).any(axis=1)   # rows with >=1 opener = 1-push-solvable
        idx = np.nonzero(solvable)[0]
        if len(idx) == 0:
            continue
        ctx = s["ctx"][c0:c1]; oc = s["object_center"][c0:c1]
        xml = s["xml"][c0:c1]
        cpx = s["contact_px"][c0:c1] if has_cpx else None
        for k in idx:
            mask = op[k].astype(np.float32)       # loss ONLY on opener cells
            ds["ctx"][j] = ctx[k]
            ds["f_grid"][j] = mask                # 1.0 at openers, 0 elsewhere (gated by r_mask)
            ds["r_mask"][j] = mask                # loss only on the known openers; rest UNKNOWN (masked)
            ds["ratio"][j] = 1.0                  # all unmasked cells are positive
            ds["object_center"][j] = oc[k]; ds["xml"][j] = xml[k]
            ds["H"][j] = 2; ds["dead"][j] = 0; ds["onepush_h2_aug"][j] = 1
            if has_cpx:
                ds["contact_px"][j] = cpx[k]
            j += 1
        if c0 % (a.chunk * 10) == 0:
            print(f"  [{c1}/{N}] emitted={j}", flush=True)
    for name in ds:
        ds[name].resize(j, axis=0)
    d.attrs["n_samples"] = j
    d.attrs["source"] = a.src_h5
    d.attrs["note"] = "1-push openers relabeled as H=2 sparse-positive (opener=1.0, rest masked)"
    s.close(); d.close()
    print(f"wrote {a.out_h5}  rows={j} (H=2 sparse-positive 1-push-opener augmentation)", flush=True)


if __name__ == "__main__":
    main()
