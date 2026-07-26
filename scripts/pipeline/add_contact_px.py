#!/usr/bin/env python3
"""Add `contact_px` (N,60,2) — the 60 edge contact-pixel coords in the 64x64 crop frame — to the
existing scorer H5, so the per-edge (HACMan-style) model can gather/position each edge. Reads the
object pose (theta, half-extents, crop size) from the SOURCE diffusion H5, matched by object_center
(1:1, exact copy). Light: pose math only, no image copy.

Edge layout (verified, src/planning/namo_push_controller.cpp generate_rectangular_edge_points):
  4 faces x 15 points, interleaved. even<30=Top, odd<30=Bottom, even>=30=Right, odd>=30=Left.
"""
import argparse, math
import h5py, numpy as np
from namo.paths import H5


def _local_edge_point(edge, hw, hd):
    n = 15
    sl = lambda a, b, i: a + (b - a) * (i / (n - 1))
    if edge < 30:
        j = edge // 2; lx = sl(-hw, hw, j); ly = hd if edge % 2 == 0 else -hd
    else:
        k = (edge - 30) // 2; lx = hw if edge % 2 == 0 else -hw; ly = sl(-hd, hd, k)
    return lx, ly


def contact_offsets_world(hw, hd, theta):
    """World-frame XY offsets (meters) from the object center for all 60 contact points.

    Edge ordering is 4 faces x 15 points, matching generate_rectangular_edge_points in
    src/skills/namo_push_controller.cpp. This is the single source of that ordering; the
    pixel-space helper below is a thin wrapper over it."""
    out = np.zeros((60, 2), np.float32)
    c, s = math.cos(theta), math.sin(theta)
    for e in range(60):
        lx, ly = _local_edge_point(e, hw, hd)
        out[e, 0] = lx * c - ly * s
        out[e, 1] = lx * s + ly * c
    return out


def contact_px(edge, hw, hd, theta, crop_m, S=64):
    wx, wy = contact_offsets_world(hw, hd, theta)[edge]
    res = crop_m / S; cx = S / 2.0
    return cx + wx / res, cx + wy / res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer-h5", default=str(H5 / "v3_scorer_1push/data.h5"))
    ap.add_argument("--src-h5", nargs="+", default=[str(H5 / "v3_1push_le10_lzf_tight_data/data.h5")],
                    help="one or MORE mask H5s (parallel pack shards / solvable+dead-end packs); the pose "
                         "map is the union — same object_center key in two files carries the same pose")
    ap.add_argument("--crop", default="tight", choices=["tight", "wide"])
    a = ap.parse_args()

    # source pose map: round(object_center) -> (theta, hw, hd, crop_m). crop_m differs tight(0.5)/wide(1.2)
    pose = {}
    for src in a.src_h5:
        s = h5py.File(src, "r")
        soc = s[f"local_{a.crop}_object_center"][:]; sth = s[f"local_{a.crop}_object_theta"][:, 0]
        ssz = s["target_object_size"][:]; scm = s[f"local_{a.crop}_crop_size_meters"][:, 0]
        for i in range(soc.shape[0]):
            key = (round(float(soc[i, 0]), 4), round(float(soc[i, 1]), 4))
            pose[key] = (float(sth[i]), float(ssz[i, 0]), float(ssz[i, 1]), float(scm[i]))
        s.close()

    d = h5py.File(a.scorer_h5, "a")
    oc = d["object_center"][:]; N = oc.shape[0]
    cpx = np.zeros((N, 60, 2), dtype=np.float32); miss = 0
    for i in range(N):
        key = (round(float(oc[i, 0]), 4), round(float(oc[i, 1]), 4))
        if key not in pose:
            miss += 1; cpx[i] = 32.0; continue   # center fallback (should not happen)
        th, hw, hd, cm = pose[key]
        offs = contact_offsets_world(hw, hd, th)  # hoisted: same for all 60 edges of this row
        res = cm / 64; cx = 64 / 2.0
        for e in range(60):
            wx, wy = offs[e]
            cpx[i, e] = (cx + wx / res, cx + wy / res)
    if "contact_px" in d:
        del d["contact_px"]
    d.create_dataset("contact_px", data=cpx, compression="lzf")
    d.close()
    # sanity: contact pixels should sit inside the frame and around the object
    inb = float(np.mean((cpx >= 0) & (cpx <= 64)))
    print(f"added contact_px to {a.scorer_h5}: N={N} miss={miss} in-bounds_frac={inb:.3f} "
          f"px_range=[{cpx.min():.1f},{cpx.max():.1f}] mean={cpx.mean():.1f}")


if __name__ == "__main__":
    main()
