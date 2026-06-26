#!/usr/bin/env python3
"""CLEAN one-change ablation: relabel ONLY the h2 H=2 SETUP cells in-place (flat 0.9 -> gamma*V_GT), leaving everything
else identical to NoHz-v3's mix (m2b + h2 + aug + exit_finish_valid). This isolates the bootstrap effect — the first
qboot run confounded it by ALSO dropping aug + swapping the finish data.

Per H=2 row: match to the exhaustive labels by (xml, object_center ~0mm), then for each tried setup in `frac_first_push`
set f_grid[edge,depth] = [n_open==0] ? 0 : gamma*V_GT.  --vsummary density = gamma*(n_open/n_tried) | depth = gamma*1.
H=1 rows (direct openers) and ctx/contact_px/r_mask are UNTOUCHED. Copies the H5 first then rewrites only f_grid
(avoids loading the ~38GB ctx). gamma=0.9. Output is a drop-in replacement for v4_hq_h2_scorer in the NoHz recipe."""
import sys, os, json, argparse, shutil
import numpy as np, h5py

GAMMA = 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-h5", required=True, help="v4_hq_h2_scorer/data.h5 (the setup data NoHz-v3 used)")
    ap.add_argument("--key", default="/scratch/dm1487/datasets/v4_hq_h2/labels_exhaustive_pure2push.json")
    ap.add_argument("--vsummary", default="density", choices=["depth", "density"])
    ap.add_argument("--out-h5", required=True)
    ap.add_argument("--match-tol", type=float, default=1e-4, help="max object_center sq-dist for an episode match (~1cm)")
    a = ap.parse_args()

    d = json.load(open(a.key))
    idx = {xml: [(np.array(r["object_center"][:2], np.float32), r.get("frac_first_push", [])) for r in recs]
           for xml, recs in d.items()}

    os.makedirs(os.path.dirname(a.out_h5), exist_ok=True)
    print(f"copying {a.in_h5} -> {a.out_h5} ...", flush=True)
    shutil.copyfile(a.in_h5, a.out_h5)
    f = h5py.File(a.out_h5, "r+")
    n = int(f.attrs.get("n_samples", f["f_grid"].shape[0]))
    H = f["H"][:]; OC = f["object_center"][:]; XML = f["xml"][:]
    fg = f["f_grid"][:]                                  # (n,60,5) — small, ~373MB
    relabeled = unmatched = skipped_h1 = 0
    for i in range(n):
        if int(H[i]) != 2:
            skipped_h1 += 1; continue
        xml = XML[i].decode() if isinstance(XML[i], (bytes, bytearray)) else XML[i]
        recs = idx.get(xml)
        if not recs:
            unmatched += 1; continue
        oc = OC[i]; best = None; bd = 1e9
        for (roc, ffp) in recs:
            dd = float(np.sum((roc - oc) ** 2))
            if dd < bd:
                bd = dd; best = ffp
        if best is None or bd > a.match_tol:
            unmatched += 1; continue
        for row in best:
            e, dpt, no, nt = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if not (0 <= e < 60 and 0 <= dpt < 5):
                continue
            # RELABEL ONLY existing SETUP cells (==0.9); preserve direct openers (1.0) + dead/unreach (0) untouched.
            if abs(float(fg[i, e, dpt]) - GAMMA) > 0.02:
                continue
            V = 1.0 if a.vsummary == "depth" else (no / max(nt, 1) if no > 0 else 0.0)
            fg[i, e, dpt] = GAMMA * float(V)
        relabeled += 1
        if relabeled % 20000 == 0:
            print(f"  relabeled {relabeled} H=2 rows", flush=True)
    f["f_grid"][:] = fg
    f.attrs["relabeled_setup_vsummary"] = a.vsummary
    f.close()
    print(json.dumps({"out": a.out_h5, "vsummary": a.vsummary, "h2_rows_relabeled": relabeled,
                      "h1_rows_untouched": skipped_h1, "unmatched_h2": unmatched}, indent=1))


if __name__ == "__main__":
    main()
