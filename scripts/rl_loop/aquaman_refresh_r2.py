#!/usr/bin/env python3
"""Aquaman-0 rebuild v2 — EXACT linkage from colossus raw shards (no pose matching).

EXP-2026-08-02-bootstrap-value-loop round 0 step 2, full dose. Supersedes aquaman_rebuild.py
(v1 pose-match: only 15,869 cells — the curated 200k H5 lacked most capped-cell children).

Raw: 32 shards candidates_NNN.h5 (~3.4M rows) with node_kind/chain_depth/parent_edge/
parent_depth — a depth2 row's (xml, object_id, parent_edge, parent_depth) IS the root cell
that produced it. Deploy H5 groups are unique per (xml, object_id) (census: 0 multi-root).

MAP (one process per shard, GPU round-robin):
  load deploy colossus-root index {(xml,obj) -> row}; for each depth2/depth2_noop shard row
  whose (xml,obj) is a deploy colossus root AND whose (parent_edge,parent_depth) cell is
  CAPPED on that deploy row: V-hat = top5-mean of theta0 raw E[bin] over the child's
  UNTRIED cells (r_mask & ~value_mask). noop children (setup_moved==0) SKIPPED (state
  unchanged -> board == parent, self-referential guess). Emit (deploy_row, e, d, target).
REDUCE (single writer):
  copy deploy H5 -> OUT, apply edits: value_target=min(0.81, 0.9*V-hat), ceiling_mask=0,
  guess_mask=1 (new uint8 dataset). Collisions (same cell from 2 children, e.g. two shards
  splitting an episode): keep LOWEST target (conservative).

  python aquaman_rebuild_v2.py map --shard N --gpu G     # 32x, parallel
  python aquaman_rebuild_v2.py reduce                     # once, after all maps
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "python", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

R3 = Path("/common/users/dm1487/scratch_namo/curriculum2/beast/round3")
DEPLOY = R3 / "h5/d20_plus_setup_only.h5"
CKPT = Path("/common/users/dm1487/scratch_namo/aquaman/round1/models/r1_std_s1/checkpoints/epoch010-val_loss1.8261.ckpt")
RAW = Path("/common/users/dm1487/scratch_namo/aquaman/round0/raw")
WORK = Path("/common/users/dm1487/scratch_namo/aquaman/round2/refresh_work")
OUT = Path("/common/users/dm1487/scratch_namo/aquaman/round2/old_refreshed.h5")
GAMMA, CAP, TOP_M = 0.9, 0.81, 5
BATCH = 512


def deploy_root_index():
    """(xml, object_id) -> (deploy_row, capped 60x5 bool) for the 26k colossus roots."""
    with h5py.File(DEPLOY, "r") as d:
        ir = d["is_root"][:].astype(bool)
        am = d["action_motion_available"][:].astype(bool)
        rows = np.where(ir & am)[0]
        xml = d["xml"][:]
        obj = d["object_id"][:]
        idx = {}
        for r in rows:
            vm = d["value_mask"][r] > 0.5
            rm = d["r_mask"][r] > 0.5
            capped = vm & rm & (d["ceiling_mask"][r] > 0.5)
            maskedr = rm & ~vm            # censored: child existence proves it was simmed
            idx[(xml[r], obj[r])] = (int(r), capped, maskedr)
    return idx


def run_map(shard, gpu):
    import torch
    from eval_auc import load_network

    idx = deploy_root_index()
    print(f"shard {shard}: deploy roots indexed {len(idx)}", flush=True)
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    net, hl = load_network(str(CKPT), dev)

    path = RAW / f"candidates_{shard:03d}.h5"
    edits, skipped_noop, no_untried, not_ours, exact_parent = [], 0, 0, 0, 0
    with h5py.File(path, "r") as f:
        nk = f["node_kind"][:]
        moved = f["setup_moved"][:]
        rows = np.where(nk != b"root")[0]
        xml = f["xml"][:]
        obj = f["object_id"][:]
        pe = f["parent_edge"][:]
        pd = f["parent_depth"][:]
        keep = []
        for i in rows:
            key = (xml[i], obj[i])
            hit = idx.get(key)
            if hit is None:
                not_ours += 1
                continue
            if moved[i] == 0:
                skipped_noop += 1
                continue
            if hit[1][pe[i], pd[i]]:
                keep.append((i, 0))          # class 0: capped parent (cap 0.81)
            elif hit[2][pe[i], pd[i]]:
                keep.append((i, 1))          # class 1: censored/masked parent (cap 0.9)
            else:
                exact_parent += 1            # verified setup/opener cell: never guessed
                continue
        print(f"shard {shard}: children total={len(rows)} keep={len(keep)} "
              f"not_ours={not_ours} noop={skipped_noop} exact_parent={exact_parent}", flush=True)
        keep = np.array(keep, dtype=np.int64).reshape(-1, 2)
        kcls = keep[:, 1]
        keep = keep[:, 0]
        for s in range(0, len(keep), BATCH):
            b = keep[s:s + BATCH]
            ctx = torch.from_numpy(f["ctx"][b].astype(np.float32)).to(dev)
            cpx = torch.from_numpy(f["contact_px"][b].astype(np.float32)).to(dev)
            with torch.no_grad():
                am_feat = None
                if net.action_motion_dim > 0:
                    from namo.rl_loop.action_motion import action_motion_from_contact_px
                    am_feat = action_motion_from_contact_px(
                        cpx, encoding=net.action_motion_encoding, feature_dim=net.action_motion_dim)
                vals = hl.value(net(ctx, cpx, action_motion=am_feat).float()).cpu().numpy()
            vm = f["value_mask"][b]
            rmk = f["r_mask"][b]
            for j, i in enumerate(b):
                untried = (rmk[j] > 0.5) & ~(vm[j] > 0.5)
                if untried.sum() < 1:
                    no_untried += 1
                    continue
                vh = float(np.sort(vals[j][untried])[::-1][:TOP_M].mean())
                drow = idx[(xml[i], obj[i])][0]
                edits.append((drow, int(pe[i]), int(pd[i]), vh, int(kcls[s + j])))
            if s % (BATCH * 20) == 0:
                print(f"  shard {shard}: {s}/{len(keep)}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(WORK / f"edits_{shard:03d}.npy", np.array(edits, dtype=np.float64))
    (WORK / f"edits_{shard:03d}.json").write_text(json.dumps(
        {"keep": int(len(keep)), "noop": skipped_noop, "no_untried": no_untried,
         "not_ours": not_ours, "exact_parent": exact_parent,
         "kept_capped": int((kcls == 0).sum()), "kept_masked": int((kcls == 1).sum())}))
    print(f"shard {shard}: edits={len(edits)} DONE", flush=True)


def run_reduce(arm):
    import shutil
    files = sorted(WORK.glob("edits_*.npy"))
    assert len(files) == 32, f"expected 32 edit files, found {len(files)}"
    all_e = np.concatenate([np.load(f) for f in files if np.load(f).size])
    print(f"total edits {len(all_e)}", flush=True)
    caps = {0: 0.81, 1: 0.9}
    best = {}
    for drow, e, d, vh, cls in all_e:
        if arm == "A" and int(cls) != 0:
            continue                       # arm A: capped parents only
        tgt = min(caps[int(cls)], GAMMA * vh)
        k = (int(drow), int(e), int(d))
        if k not in best or tgt < best[k]:
            best[k] = float(tgt)  # collision -> keep LOWEST (conservative)
    print(f"unique cells {len(best)} (collisions {len(all_e) - len(best)})", flush=True)
    out_path = OUT.with_name(f"aquaman0_train_{arm}.h5")
    shutil.copyfile(DEPLOY, out_path)
    targets = np.array(list(best.values()))
    by_row = {}
    for (r, e, d), t in best.items():
        by_row.setdefault(r, []).append((e, d, t))
    with h5py.File(out_path, "r+") as f:
        n = f.attrs["n_samples"]
        gm = f.create_dataset("guess_mask", shape=(n, 60, 5), dtype=np.uint8,
                              compression="lzf", chunks=(1, 60, 5))
        for m, (r, cells) in enumerate(by_row.items()):
            vt = f["value_target"][r]
            cm = f["ceiling_mask"][r]
            g = np.zeros((60, 5), np.uint8)
            for e, d, t in cells:
                vt[e, d] = t
                cm[e, d] = 0.0
                g[e, d] = 1
            f["value_target"][r] = vt
            f["ceiling_mask"][r] = cm
            gm[r] = g
            if m % 2000 == 0:
                print(f"  write {m}/{len(by_row)}", flush=True)
    rep = {"edits": int(len(all_e)), "unique_cells": len(best), "rows_touched": len(by_row),
           "target_quartiles": [float(x) for x in np.percentile(targets, [25, 50, 75])],
           "target_hist": np.histogram(targets, bins=np.linspace(0, 0.81, 28))[0].tolist(),
           "frac_below_0p1": float((targets < 0.1).mean()),
           "clip_at_cap": float((targets >= CAP - 1e-9).mean())}
    (WORK / f"report_{arm}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "target_hist"}, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["map", "reduce"])
    ap.add_argument("--shard", type=int)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arm", choices=["A", "B"], default="B")
    a = ap.parse_args()
    if a.mode == "map":
        run_map(a.shard, 0)  # CUDA_VISIBLE_DEVICES pins the physical GPU
    else:
        run_reduce(a.arm)
