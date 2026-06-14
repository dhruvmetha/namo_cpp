#!/usr/bin/env python3
"""CLEAN scene-generalization test for the finish. Replicates the model's EXACT train/val split (room-grouped,
Random(0), 90/10 over the v2 mix), then scores the model's raw H=1 finish value on HELD-OUT (val) postpush crops —
same setups, same render path as training; ONLY the scenes are unseen. Disentangles pure scene-generalization
from the setup-policy / live-render confounds in the earlier train(0.75)-vs-test(0.27) comparison."""
import sys, glob, os, random
REPO = "/cache/home/dm1487/projects/namo/namo_cpp"; SAGE = "/cache/home/dm1487/projects/namo/sage_learning"
for _p in (f"{REPO}/build_python", f"{REPO}/python", f"{REPO}/scripts", f"{REPO}/scripts/sandbox", SAGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np, h5py, json
from live_scorer import LiveScorer
CK = glob.glob("/scratch/dm1487/sage_outputs/scorer/qfull_v2_v4hq_s1/namo-classifier/*/checkpoints/epoch008-val_loss0.6728.ckpt")[0]
H5 = ["/scratch/dm1487/h5/v4_hq_m2b_scorer/data.h5", "/scratch/dm1487/h5/v4_hq_h2_scorer/data.h5",
      "/scratch/dm1487/h5/v4_hq_onepush_h2_aug/data.h5"] + sorted(glob.glob("/scratch/dm1487/h5/v4_hq_postpush_v2/shard_*.h5"),
      key=lambda p: int(p.split("shard_")[1].split(".")[0]))
PP_START = 3  # indices 3.. are postpush


def main():
    # replicate ScorerDataModule.setup() split exactly
    groups = {}; n = 0
    for fi, p in enumerate(H5):
        f = h5py.File(p, "r"); xml = [x.decode() if isinstance(x, bytes) else str(x) for x in f["xml"][:]]
        for i in range(len(xml)):
            groups.setdefault(xml[i], []).append((fi, i))
        n += len(xml); f.close()
    keys = sorted(groups); random.Random(0).shuffle(keys)
    target = int(n * 0.9); cum = 0; val_keys = set()
    for k in keys:
        if cum < target:
            cum += len(groups[k])
        else:
            val_keys.add(k)
    # collect VAL postpush rows (fi >= PP_START and xml in val)
    val_pp = [(fi, i) for k in val_keys for (fi, i) in groups[k] if fi >= PP_START]
    print(f"total rows={n} val_keys={len(val_keys)} val_postpush_rows={len(val_pp)}", file=sys.stderr, flush=True)
    random.Random(1).shuffle(val_pp)
    sc = LiveScorer(ckpt=CK)
    fh = {}
    pos, neg = [], []
    for (fi, i) in val_pp[:600]:
        if fi not in fh:
            fh[fi] = h5py.File(H5[fi], "r")
        f = fh[fi]
        ctx = f["ctx"][i].astype(np.float32); cpx = f["contact_px"][i].astype(np.float32)
        fg = f["f_grid"][i]; rm = f["r_mask"][i]
        raw = sc.score_ctx(ctx, cpx, h=1, raw=True)
        tried = rm >= 0.5; op = (fg >= 0.999) & tried; no = (~(fg >= 0.999)) & tried
        pos += raw[op].tolist(); neg += raw[no].tolist()
    pos, neg = np.array(pos), np.array(neg)
    out = {"VAL postpush (held-out scenes, collection setups, H5 render)":
           {"openers_mean": round(float(pos.mean()), 3), "non_mean": round(float(neg.mean()), 3),
            "separation": round(float(pos.mean() - neg.mean()), 3), "pos_p90": round(float(np.percentile(pos, 90)), 3),
            "n_pos": len(pos), "n_neg": len(neg)},
           "REF: TRAIN postpush sep": 0.75, "REF: TEST live (model-setup s1) sep": 0.273}
    json.dump(out, open("/scratch/dm1487/eval/disentangle_gen.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
