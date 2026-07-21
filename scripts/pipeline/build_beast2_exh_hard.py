#!/usr/bin/env python3
"""beast2_exh_hard.h5 — hard-label twin of beast2_exh_ceil.h5 [USER: 'a second version like beast-1-hard'].

IDENTICAL rows to beast2_exh_ceil.h5; the ONLY change is the label rule, mirroring build_beast1_hard:
every ceiling cell (ceiling_mask==1, values 0.81 root / 0.9 finish / R1-inherited ceilings) -> exact 0.0,
ceiling_mask zeroed everywhere. Openers 1.0 / verified setups 0.9-exact / unreachable -1 / masks unchanged.
The controlled soft-vs-hard pair for the dead-heavy regime (the pre-registered clean-pair rerun).
"""
import shutil
import time

import h5py
import numpy as np

SRC = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/beast2_exh_ceil.h5"
OUT = "/common/users/dm1487/scratch_namo/curriculum2/beast/round2/h5/beast2_exh_hard.h5"
CH = 20000
t0 = time.time()

shutil.copyfile(SRC, OUT)
print(f"copied ({time.time()-t0:.0f}s)", flush=True)

f = h5py.File(OUT, "r+")
N = int(f.attrs["n_samples"])
n_zeroed = 0
for s in range(0, N, CH):
    e = min(s + CH, N)
    cm = f["ceiling_mask"][s:e]
    if not (cm == 1).any():
        continue
    vt = f["value_target"][s:e]
    hit = cm == 1
    vt[hit] = 0.0
    f["value_target"][s:e] = vt
    f["ceiling_mask"][s:e] = np.zeros_like(cm)
    n_zeroed += int(hit.sum())
    if (s // CH) % 10 == 0:
        print(f"  {e}/{N} ({time.time()-t0:.0f}s)", flush=True)

for s in range(0, N, 200000):  # count-assert: no ceiling cells and no 0.81/0.9-ceiling values remain
    e = min(s + 200000, N)
    assert not (f["ceiling_mask"][s:e] == 1).any()
f.close()
print(f"DONE {OUT}: ceiling cells zeroed: {n_zeroed:,} ({time.time()-t0:.0f}s)", flush=True)
