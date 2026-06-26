#!/usr/bin/env python3
"""STAGE 1 (iteration-0, GROUNDED) + STAGE 3 knob: turn (s0,a1,s1) TRANSITIONS into a bootstrap-SETUP scorer H5.

For each transition: setup target at s0 cell a1 = [dead s1] ? 0 : gamma * V_GT(s1), where V_GT is read from the GROUND
-TRUTH finish labels at s1 (no model -> stable iteration-0 / the Stage-1b seed). The V(s1) SUMMARY is the STAGE-3
ablation (change ONLY this):
  --vsummary depth     : V_GT = 1.0 if solvable      -> target = 0.9 for every solvable setup  (existence; ~status-quo flat label)
  --vsummary density   : V_GT = n_open/n_tried       -> target = 0.9 * density                 (FINDABILITY; the cost-to-go signal)
Rows are grouped by episode (identical s0 crop) so each s0 carries ALL its collected setups' targets in one 60x5 map
(sparse: only collected setup cells are in r_mask/loss_mask). Output = standard scorer H5 (ctx/f_grid/r_mask/contact_px/
object_center/xml/ratio/H/dead), trained by the EXISTING pipeline with budget_cond=false (single Q, Horizon dropped).
Offline H5->H5 (no MuJoCo, no model). gamma=0.9."""
import sys, glob, argparse, hashlib
import numpy as np, h5py

GAMMA = 0.9


def vgt(summary, dead, ratio):
    if dead:
        return 0.0
    return 1.0 if summary == "depth" else float(ratio)   # density = n_open/n_tried (stored as 'ratio')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-glob", required=True, help="transition shards, e.g. /scratch/dm1487/h5/v4_hq_transitions/shard_*.h5")
    ap.add_argument("--vsummary", default="density", choices=["depth", "density"])
    ap.add_argument("--out-h5", required=True)
    a = ap.parse_args()
    shards = sorted(glob.glob(a.in_glob))
    assert shards, f"no shards: {a.in_glob}"
    # group by episode = identical s0 crop (ctx0); merge each episode's setup cells into one 60x5 target map
    epis = {}  # key -> dict(ctx0, contact_px0, xml, oc, fg(60,5), rm(60,5), n_solv, n_dead)
    for p in shards:
        f = h5py.File(p, "r"); n = int(f.attrs.get("n_samples", 0))
        if not n:
            f.close(); continue
        ctx0 = f["ctx0"][:]; cp0 = f["contact_px0"][:]; ae = f["a1_edge"][:]; ad = f["a1_depth"][:]
        dead = f["dead"][:]; ratio = f["ratio"][:]; xml = f["xml"][:]; oc = f["object_center"][:]
        for i in range(n):
            key = hashlib.md5(ctx0[i].tobytes()).hexdigest()
            e = epis.get(key)
            if e is None:
                e = dict(ctx0=ctx0[i], cp0=cp0[i], xml=xml[i], oc=oc[i],
                         fg=np.zeros((60, 5), np.float32), rm=np.zeros((60, 5), np.float32), nsolv=0, ndead=0)
                epis[key] = e
            tgt = GAMMA * vgt(a.vsummary, int(dead[i]), float(ratio[i]))
            e["fg"][int(ae[i]), int(ad[i])] = tgt
            e["rm"][int(ae[i]), int(ad[i])] = 1.0
            e["ndead"] += int(dead[i]); e["nsolv"] += int(dead[i] == 0)
        f.close()
    M = len(epis)
    rows = list(epis.values())
    with h5py.File(a.out_h5, "w") as f:
        f.create_dataset("ctx", data=np.stack([r["ctx0"] for r in rows]) if M else np.zeros((0, 5, 64, 64), np.float32), compression="lzf")
        f.create_dataset("contact_px", data=np.stack([r["cp0"] for r in rows]) if M else np.zeros((0, 60, 2), np.float32), compression="lzf")
        f.create_dataset("f_grid", data=np.stack([r["fg"] for r in rows]) if M else np.zeros((0, 60, 5), np.float32), compression="lzf")
        f.create_dataset("r_mask", data=np.stack([r["rm"] for r in rows]) if M else np.zeros((0, 60, 5), np.float32), compression="lzf")
        f.create_dataset("object_center", data=np.stack([r["oc"] for r in rows]) if M else np.zeros((0, 2), np.float32))
        f.create_dataset("ratio", data=np.array([float(r["nsolv"]) / max(r["nsolv"] + r["ndead"], 1) for r in rows], np.float32))
        f.create_dataset("H", data=np.full(M, 2, np.int64))            # ignored when budget_cond=false
        f.create_dataset("dead", data=np.array([int(r["nsolv"] == 0) for r in rows], np.int64))
        f.create_dataset("xml", data=np.array([r["xml"] for r in rows], dtype=object), dtype=h5py.string_dtype(encoding="utf-8"))
        f.attrs["n_samples"] = M
    tot_cells = sum(int(r["rm"].sum()) for r in rows)
    print(f"OK vsummary={a.vsummary}: {M} episode-rows, {tot_cells} setup cells, "
          f"{sum(r['nsolv'] for r in rows)} solvable / {sum(r['ndead'] for r in rows)} dead setups -> {a.out_h5}")


if __name__ == "__main__":
    main()
