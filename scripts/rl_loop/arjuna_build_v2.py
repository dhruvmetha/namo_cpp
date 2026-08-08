#!/usr/bin/env python3
"""Arjuna-0 v2 -- the BIG-DOSE floor test. Every bounded cell becomes an exact 0. Zero sims.

  opener  -> 1.0     (unchanged)
  setup   -> 0.5     (was 0.9; gamma is now 0.5, so finish >> setup >> rest is a wide ladder)
  bounded -> 0.0     exact, two-sided, full weight  <- the whole point
  untried -> masked  (unchanged; we never invent a value for a push nobody tried)

Why v2 exists. v1 converted 114,660 cells = 1.4% of the 8.29M bounded population, because it could
only reach the 26,023 Colossus setup roots that have child boards in the raw shards. Its null result
on V5 therefore tested "does a 1.4% floor fix cross-board comparability", not "does the model need a
floor". v2 needs no linkage at all: a bounded cell is one we simmed and that failed, and under the
deployed hmax=2 semantics that is already the answer. Zeros go from 0.66% of in-loss cells to ~47%.

HONEST ASYMMETRY, recorded before the run [Claude 2026-08-08]:
  * child bounded (<=0.90, 2.84M cells): zeroing is CORRECT. A failed push-2 has no third push, so
    its value to the deployed searcher is exactly 0.
  * root bounded (<=0.81, 5.51M cells): zeroing ASSERTS "not a setup". Some of these are setups we
    never discovered -- the label is wrong for those. This is the same move that regressed
    2026-07-25 (sims-to-solve 46.0 -> 53.8), with one difference: that run also zeroed UNTRIED
    cells, inventing facts about pushes nobody executed. v2 only zeroes cells that were simmed.
  A later arm should split these (child->0, root->masked) to separate the safe half from the risky.

  python arjuna_build_v2.py --out arjuna0v2_train.h5
"""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

R3 = Path("/common/users/dm1487/scratch_namo/curriculum2/beast/round3")
DEPLOY = R3 / "h5/d20_plus_setup_only.h5"
SETUP_NEW = 0.5
OPENER = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()

    print(f"copying {DEPLOY} -> {a.out}", flush=True)
    shutil.copyfile(DEPLOY, a.out)

    cen = dict(rows=0, opener=0, setup=0, bounded_zeroed=0, untouched_masked=0)
    with h5py.File(a.out, "r+") as f:
        n = int(f.attrs["n_samples"])
        for s in range(0, n, a.chunk):
            e = min(s + a.chunk, n)
            vt = f["value_target"][s:e]
            vm = f["value_mask"][s:e] > 0.5
            rm = f["r_mask"][s:e] > 0.5
            cm = f["ceiling_mask"][s:e] > 0.5
            inloss = vm & rm
            exact = inloss & ~cm
            bounded = inloss & cm

            opener = exact & (vt > 0.95)
            setup = exact & (vt > 0.85) & (vt <= 0.95)

            vt = np.where(opener, OPENER, vt)
            vt = np.where(setup, SETUP_NEW, vt)
            vt = np.where(bounded, 0.0, vt)            # every bound becomes a hard zero
            cm_new = np.where(bounded, 0.0, f["ceiling_mask"][s:e])   # two-sided now

            f["value_target"][s:e] = vt
            f["ceiling_mask"][s:e] = cm_new

            cen["rows"] += e - s
            cen["opener"] += int(opener.sum())
            cen["setup"] += int(setup.sum())
            cen["bounded_zeroed"] += int(bounded.sum())
            cen["untouched_masked"] += int((rm & ~vm).sum())
            if (s // a.chunk) % 25 == 0:
                print(f"  {e}/{n}", flush=True)

    tot = cen["opener"] + cen["setup"] + cen["bounded_zeroed"]
    cen["frac_zero_of_inloss"] = round(cen["bounded_zeroed"] / max(tot, 1), 4)
    Path(a.out + ".report.json").write_text(json.dumps(cen, indent=1))
    print(json.dumps(cen, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
