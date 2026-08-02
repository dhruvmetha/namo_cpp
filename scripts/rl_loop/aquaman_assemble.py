#!/usr/bin/env python3
"""Aquaman round-1 build, stage A: score render-NPZs with theta, fill guess targets, emit part-H5.

Streams shard NPZs (never loads the full set): per shard, batch-score row ctxs + grandchild ctxs
with the CURRENT model (raw E[bin]), compute guess targets min(cap, 0.9*top5(theta over untried)),
two-side those cells (ceiling_mask->0, guess_mask->1), append rows to a part-H5.

  python aquaman_assemble.py --part 0 --nparts 4 --gpu 0 \
      --render-dir .../render --ckpt B_s1.ckpt --out part_0.h5
Concat of parts + old_refreshed.h5 happens in aquaman_concat.py.
"""
import argparse
import sys
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (f"{REPO}/python", f"{REPO}/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_auc import load_network  # noqa: E402  arch-inferring loader + HLGauss

GAMMA, TOP_M, BATCH = 0.9, 5, 512


def score(net, hl, dev, ctx, cpx):
    out = np.zeros((len(ctx), 60, 5), np.float32)
    for s in range(0, len(ctx), BATCH):
        c = torch.from_numpy(ctx[s:s + BATCH].astype(np.float32)).to(dev)
        p = torch.from_numpy(cpx[s:s + BATCH].astype(np.float32)).to(dev)
        with torch.no_grad():
            am = None
            if net.action_motion_dim > 0:
                from namo.rl_loop.action_motion import action_motion_from_contact_px
                am = action_motion_from_contact_px(p, encoding=net.action_motion_encoding,
                                                   feature_dim=net.action_motion_dim)
            out[s:s + BATCH] = hl.value(net(c, p, action_motion=am).float()).cpu().numpy()
    return out


def vhat(scores_grid, untried_mask):
    v = scores_grid[untried_mask > 0.5]
    if v.size == 0:
        return None
    return float(np.sort(v)[::-1][:TOP_M].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--nparts", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--render-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    net, hl = load_network(a.ckpt, dev)

    files = sorted(glob(f"{a.render_dir}/shard_*.npz"))[a.part::a.nparts]
    cols = None
    out = h5py.File(a.out, "w")
    n_rows = n_guess = n_noref = 0
    for fi, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        ctx, cpx = z["ctx"], z["cpx"]
        if len(ctx) == 0:
            continue
        rs = score(net, hl, dev, ctx, cpx)
        gcs = score(net, hl, dev, z["gc_ctx"], z["gc_cpx"]) if len(z["gc_ctx"]) else None
        vt, vm, cm, rm = z["vt"].copy(), z["vm"], z["cm"].copy(), z["rmask"]
        gm = np.zeros_like(vt, dtype=np.uint8)
        for ri, e, d, ref, cap in z["guess"]:
            ri, e, d, ref = int(ri), int(e), int(d), int(ref)
            if ref >= 0:
                un = (rm[ref] > 0.5) & ~(vm[ref] > 0.5)
                v = vhat(rs[ref], un)
            else:
                gi = -(ref + 2)
                v = vhat(gcs[gi], z["gc_rmask"][gi]) if gcs is not None else None
            if v is None:
                n_noref += 1
                continue                      # exhausted child: keep mute cap
            vt[ri, e, d] = min(cap, GAMMA * v)
            cm[ri, e, d] = 0.0
            gm[ri, e, d] = 1
            n_guess += 1
        data = {"ctx": ctx, "contact_px": cpx, "r_mask": rm, "value_target": vt,
                "value_mask": vm, "ceiling_mask": cm, "guess_mask": gm,
                "is_root": z["isroot"], "xml": z["xml"], "object_id": z["obj"],
                "sample_weight": np.ones(len(ctx), np.float32)}
        if cols is None:
            cols = {}
            for k, v in data.items():
                if v.dtype == object:
                    cols[k] = out.create_dataset(k, shape=(0,), maxshape=(None,),
                                                 dtype=h5py.string_dtype())
                else:
                    cols[k] = out.create_dataset(k, shape=(0,) + v.shape[1:], maxshape=(None,) + v.shape[1:],
                                                 dtype=v.dtype, compression="lzf" if v.ndim > 1 else None,
                                                 chunks=(1,) + v.shape[1:] if v.ndim > 1 else None)
        n0 = cols["ctx"].shape[0]
        for k, v in data.items():
            cols[k].resize(n0 + len(v), axis=0)
            cols[k][n0:] = v
        n_rows += len(ctx)
        if fi % 20 == 0:
            print(f"part {a.part}: {fi}/{len(files)} rows={n_rows} guesses={n_guess}", flush=True)
    out.attrs["n_samples"] = n_rows
    out.close()
    print(f"part {a.part} DONE rows={n_rows} guesses={n_guess} no_ref={n_noref}", flush=True)


if __name__ == "__main__":
    main()
