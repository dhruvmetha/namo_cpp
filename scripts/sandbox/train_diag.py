#!/usr/bin/env python3
"""What works ON TRAIN, per capability — raw E[bin] separation (positive vs negative cells in the loss region),
scored on each source's TRAINING crops at its budget H. Compare to the known TEST numbers to localize every gap."""
import sys, glob, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; SAGE = os.environ.get("SAGE_REPO", "")
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np, h5py, json  # noqa: E402
from live_scorer import LiveScorer  # noqa: E402
from namo.paths import SCRATCH, H5  # noqa: E402
CK = glob.glob(f"{SCRATCH}/sage_outputs/scorer/qfull_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch008-val_loss0.6728.ckpt")[0]
# (label, h5 path, query-H, positive-threshold on f_grid)
JOBS = [("m2b  1-push opener @H=1", str(H5 / "v4_hq_m2b_scorer/data.h5"), 1, 0.999),
        ("h2   setup           @H=2", str(H5 / "v4_hq_h2_scorer/data.h5"), 2, 0.5),
        ("postpush finish      @H=1", str(H5 / "v4_hq_postpush_v2/shard_0.h5"), 1, 0.999)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    sc = LiveScorer(ckpt=CK)
    out = {}
    for label, p, h, thr in JOBS:
        f = h5py.File(p, "r"); N = min(int(f.attrs.get("n_samples", f["ctx"].shape[0])), n)
        pos, neg = [], []
        # H must match the row when budget-conditioned: use the row's H if present else the job H
        hasH = "H" in f
        for i in range(N):
            ctx = f["ctx"][i].astype(np.float32); cpx = f["contact_px"][i].astype(np.float32)
            fg = f["f_grid"][i]; rm = f["r_mask"][i]
            qh = int(f["H"][i]) if hasH else h
            raw = sc.score_ctx(ctx, cpx, h=qh, raw=True)
            tried = rm >= 0.5; p_cells = (fg >= thr) & tried; n_cells = (~(fg >= thr)) & tried
            pos += raw[p_cells].tolist(); neg += raw[n_cells].tolist()
        out[label] = {"openers_mean": round(float(np.mean(pos)), 3) if pos else None,
                      "neg_mean": round(float(np.mean(neg)), 3) if neg else None,
                      "separation": round(float(np.mean(pos) - np.mean(neg)), 3) if pos and neg else None,
                      "pos_p90": round(float(np.percentile(pos, 90)), 3) if pos else None,
                      "n_pos": len(pos), "n_neg": len(neg)}
        f.close()
        print(f"  {label}: sep={out[label]['separation']} (pos {out[label]['openers_mean']} vs neg {out[label]['neg_mean']})", file=sys.stderr, flush=True)
    json.dump(out, open(str(SCRATCH / "eval/train_diag.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
