#!/usr/bin/env python3
"""Arjuna-0 -- write FACTS where aquaman wrote guesses. Zero sims, zero GPU.

The one-variable contrast against aquaman arm A. Arm A took root cells that were simmed, failed,
and had a stored child, and wrote min(cap, 0.9*V-hat) -- the model's opinion about whether that
child still hides an opener. The Colossus raw shards already answer that question for a large
fraction of those children:

    n_win > 0                          -> the child HAS an opener  -> parent is a verified SETUP
    n_win == 0 and not censored        -> the child was swept clean -> parent is a verified DEAD push
    n_win == 0 and finish_sweep_censored -> genuinely unknown       -> leave the ceiling alone

So the same cells get 0.9 / 0.0 exact at full weight instead of a half-weight guess.

Why it matters (measured 2026-08-07): 96-100% of every training target sits >= 0.8 -- the
regression is fitting a near-constant, which is why the ranking aux carries ~half the deployed
performance and why cross-board separation is stuck at V5 ~ 0.54. There is not one exact zero in
143,705 exact facts. This is the first label set with a floor.

Semantics: policy-relative. 0.0 means "worthless to the 2-push searcher we deploy", not "worthless
in the world" -- the child was swept at depth 1 and Colossus stored no grandchildren (chain_depth
is only {1,2}), so a deeper value is not merely unknown, it is unobtainable from this data.

  python arjuna_build.py --out arjuna0_train.h5 [--limit-shards N]
"""
import argparse
import json
import sys
from glob import glob
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "python", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

R3 = Path("/common/users/dm1487/scratch_namo/curriculum2/beast/round3")
DEPLOY = R3 / "h5/d20_plus_setup_only.h5"
RAW = Path("/common/users/dm1487/scratch_namo/aquaman/round0/raw")
GAMMA = 0.9


def deploy_root_index():
    """(xml, object_id) -> (row, capped 60x5, masked-but-simmed 60x5). Same join as rebuild_v2."""
    with h5py.File(DEPLOY, "r") as d:
        ir = d["is_root"][:].astype(bool)
        am = d["action_motion_available"][:].astype(bool)
        rows = np.where(ir & am)[0]
        xml, obj = d["xml"][:], d["object_id"][:]
        idx = {}
        for r in rows:
            vm = d["value_mask"][r] > 0.5
            rm = d["r_mask"][r] > 0.5
            idx[(xml[r], obj[r])] = (int(r), vm & rm & (d["ceiling_mask"][r] > 0.5), rm & ~vm)
    return idx


def collect_edits(limit=None):
    """Walk the raw shards; return {(row,e,d): value} plus a census. Facts only."""
    idx = deploy_root_index()
    print(f"deploy roots indexed: {len(idx)}", flush=True)
    edits, cen = {}, dict(rows=0, not_ours=0, noop=0, exact_parent=0, censored=0,
                          live=0, dead=0, collide=0)
    files = sorted(glob(str(RAW / "candidates_*.h5")))
    if limit:
        files = files[:limit]
    for fi, path in enumerate(files):
        with h5py.File(path, "r") as f:
            nk, moved = f["node_kind"][:], f["setup_moved"][:]
            nwin, csr = f["n_win"][:], f["finish_sweep_censored"][:]
            xml, obj = f["xml"][:], f["object_id"][:]
            pe, pd = f["parent_edge"][:], f["parent_depth"][:]
            for i in np.where(nk != b"root")[0]:
                cen["rows"] += 1
                hit = idx.get((xml[i], obj[i]))
                if hit is None:
                    cen["not_ours"] += 1
                    continue
                if moved[i] == 0:
                    cen["noop"] += 1
                    continue
                e, d = int(pe[i]), int(pd[i])
                if not (hit[1][e, d] or hit[2][e, d]):
                    cen["exact_parent"] += 1          # verified opener/setup: NEVER overwrite
                    continue
                if nwin[i] > 0:
                    val = GAMMA                        # child has an opener -> parent is a setup
                    cen["live"] += 1
                elif csr[i] > 0:
                    cen["censored"] += 1               # sweep incomplete -> we do NOT know
                    continue
                else:
                    val = 0.0                          # swept clean -> dead within the deploy horizon
                    cen["dead"] += 1
                k = (hit[0], e, d)
                if k in edits:
                    cen["collide"] += 1
                    edits[k] = max(edits[k], val)      # a live child anywhere wins: existence proof
                else:
                    edits[k] = val
        print(f"  shard {fi+1}/{len(files)} edits={len(edits)}", flush=True)
    return edits, cen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-shards", type=int, default=None)
    a = ap.parse_args()
    import shutil

    edits, cen = collect_edits(a.limit_shards)
    print("census:", json.dumps(cen, indent=1), flush=True)
    print(f"unique cells to write: {len(edits)}", flush=True)

    shutil.copyfile(DEPLOY, a.out)
    by_row = {}
    for (r, e, d), v in edits.items():
        by_row.setdefault(r, []).append((e, d, v))
    with h5py.File(a.out, "r+") as f:
        n = f.attrs["n_samples"]
        fm = f.create_dataset("fact_mask", shape=(n, 60, 5), dtype=np.uint8,
                              compression="lzf", chunks=(1, 60, 5))
        for m, (r, cells) in enumerate(by_row.items()):
            vt, cm, vm = f["value_target"][r], f["ceiling_mask"][r], f["value_mask"][r]
            g = np.zeros((60, 5), np.uint8)
            for e, d, v in cells:
                vt[e, d] = v
                cm[e, d] = 0.0     # two-sided: this is a fact, not a bound
                vm[e, d] = 1.0     # in the loss (class-1 parents were masked)
                g[e, d] = 1
            f["value_target"][r], f["ceiling_mask"][r], f["value_mask"][r] = vt, cm, vm
            fm[r] = g
            if m % 2000 == 0:
                print(f"  write {m}/{len(by_row)}", flush=True)
    vals = np.array(list(edits.values()))
    rep = dict(cells=len(edits), rows=len(by_row), n_zero=int((vals == 0).sum()),
               n_setup=int((vals == GAMMA).sum()), census=cen)
    Path(a.out + ".report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
