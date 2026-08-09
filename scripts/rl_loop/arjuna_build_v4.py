#!/usr/bin/env python3
"""Arjuna-0 v4 = v3 (hard floor, setup 0.9) + arm B's guesses ON THE CENSORED CHILDREN ONLY.

The 2x2 this completes, all four cells sharing setup=0.9:

                       no floor            floor (bounded -> 0)
    no guesses         base / theta0       v3
    guesses on         Bfix (arm B)        v4   <- this file
    censored children

So v4-vs-v3 isolates the guesses with the floor present, and v4-vs-Bfix isolates the floor with
the guesses present. Arm B's verdict was "labels on the unknowable hurt" (-4 to -5 pts of hard-2p
reach), measured when every other label was bootstrap-flavoured. v4 asks whether that still holds
once everything else is a hard fact.

Guessed VALUES are copied verbatim from aquaman0_train_Bfix.h5 rather than recomputed -- they are
a deterministic function of a fixed checkpoint (min(cap, 0.9 * top-5 mean of theta over the
child's untried cells)), so copying reproduces arm B exactly and costs no GPU.

WHICH cells: exactly those with r_mask=1, value_mask=1 in Bfix, value_mask=0 in v3 -- the ~523k
censored children arm B un-masked. 100% of them carry guess_mask=1. Cells that were ALREADY in
loss keep their v3 label (a bounded cell stays a hard 0), so this changes nothing except the one
population under test.

  python arjuna_build_v4.py --src arjuna0v3_train.h5 --out arjuna0v4_train.h5
"""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

R0 = Path("/common/users/dm1487/scratch_namo/aquaman/round0")
BFIX = R0 / "aquaman0_train_Bfix.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="v3 file to start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()

    print(f"copying {a.src} -> {a.out}", flush=True)
    shutil.copyfile(a.src, a.out)

    cen = dict(rows=0, restored=0, already_inloss_untouched=0, still_masked=0)
    with h5py.File(a.out, "r+") as f, h5py.File(BFIX, "r") as b:
        n = int(f.attrs["n_samples"])
        # The base d20 file carries no guess_mask -- the dataset treats it as optional
        # (`if "guess_mask" in f`), and without it the half-weight path for guessed cells never
        # fires. Create it zeroed so v4's restored cells get arm B's exact half weight.
        if "guess_mask" not in f:
            src = b["guess_mask"]
            f.create_dataset("guess_mask", shape=src.shape, dtype=src.dtype,
                             chunks=src.chunks, compression=src.compression)
            for s in range(0, n, a.chunk):
                e = min(s + a.chunk, n)
                f["guess_mask"][s:e] = np.zeros_like(src[s:e])
            print("created zeroed guess_mask", flush=True)
        for s in range(0, n, a.chunk):
            e = min(s + a.chunk, n)
            rm = f["r_mask"][s:e] > 0.5
            vv = f["value_mask"][s:e] > 0.5
            bv = b["value_mask"][s:e] > 0.5
            add = rm & bv & ~vv                      # the censored children, and only those

            vt = f["value_target"][s:e]
            vm = f["value_mask"][s:e]
            cm = f["ceiling_mask"][s:e]
            gm = f["guess_mask"][s:e]
            vt = np.where(add, b["value_target"][s:e], vt)
            vm = np.where(add, 1.0, vm)
            cm = np.where(add, b["ceiling_mask"][s:e], cm)
            gm = np.where(add, b["guess_mask"][s:e], gm)
            f["value_target"][s:e] = vt
            f["value_mask"][s:e] = vm
            f["ceiling_mask"][s:e] = cm
            f["guess_mask"][s:e] = gm

            cen["rows"] += e - s
            cen["restored"] += int(add.sum())
            cen["already_inloss_untouched"] += int((vv & rm).sum())
            cen["still_masked"] += int((rm & ~vv & ~add).sum())
            if (s // a.chunk) % 25 == 0:
                print(f"  {e}/{n}", flush=True)

    Path(a.out + ".report.json").write_text(json.dumps(cen, indent=1))
    print(json.dumps(cen, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
