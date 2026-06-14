#!/usr/bin/env python3
"""TRAIN-vs-TEST: is the mushy finish 'uncertain-but-capable' (data fix works) or a representation wall?
Score the model's RAW H=1 value on the TRAINING finish crops (postpush H5) — opener vs non-opener separation.
If TRAIN separation is SHARP (openers~0.9) but TEST was mushy (0.273) -> it CAN reason, just doesn't generalize ->
DATA fix. If TRAIN is ALSO mushy -> capacity/representation wall (more data won't help)."""
import sys, glob
REPO = "/cache/home/dm1487/projects/namo/namo_cpp"; SAGE = "/cache/home/dm1487/projects/namo/sage_learning"
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np, h5py, json
from live_scorer import LiveScorer

CK = glob.glob("/scratch/dm1487/sage_outputs/scorer/qfull_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch008-val_loss0.6728.ckpt")[0]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    sc = LiveScorer(ckpt=CK)
    f = h5py.File("/scratch/dm1487/h5/v4_hq_postpush_v2/shard_0.h5", "r")   # the FINISH (H=1 on s1) training crops
    N = min(int(f.attrs.get("n_samples", f["ctx"].shape[0])), n)
    op, non = [], []
    for i in range(N):
        ctx = f["ctx"][i].astype(np.float32); cpx = f["contact_px"][i].astype(np.float32)
        fg = f["f_grid"][i]; rm = f["r_mask"][i]
        raw = sc.score_ctx(ctx, cpx, h=1, raw=True)        # (60,5) E[bin], NO sigmoid
        tried = rm >= 0.5; opener = (fg >= 0.999) & tried; nonop = (~(fg >= 0.999)) & tried
        op += raw[opener].tolist(); non += raw[nonop].tolist()
        if i % 100 == 0:
            print(f"  [{i}/{N}]", file=sys.stderr, flush=True)
    op, non = np.array(op), np.array(non)

    def st(a):
        return dict(mean=round(float(a.mean()), 3), p10=round(float(np.percentile(a, 10)), 3),
                    p90=round(float(np.percentile(a, 90)), 3), n=len(a))
    out = {"TRAIN finish (postpush crops), RAW E[bin]":
           {"openers": st(op), "non_openers": st(non), "separation": round(float(op.mean() - non.mean()), 3)},
           "TEST finish separation (from step0_sigmoid)": 0.273,
           "verdict_hint": "TRAIN sep >> TEST 0.273 -> can reason, data fix; TRAIN ~= TEST -> representation wall"}
    json.dump(out, open("/scratch/dm1487/eval/step0_train.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
