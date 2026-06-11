#!/usr/bin/env python
"""Compare two sets of car motion-primitive .dat files (old vs new geometry).

.dat format: 4-byte LE uint32 count header (=300), then `count` packed records of
14 bytes each = 3 float32 (delta_x, delta_y, delta_theta) + 2 uint8 (edge_idx,
push_steps). The (dx, dy, dtheta) triple is the object-local se2_target.

For each shape, records are aligned by (edge_idx, push_steps) and we report the
per-component mean / max absolute shift plus the fraction of primitives that moved
past 1mm/1deg and 5mm/5deg thresholds. Also flags any non-finite / degenerate-zero
primitive and prints the edges/depths that moved most.

Usage:
  python scripts/compare_primitives_dat.py \
      --new-dir data --old-dir data/_primitive_backup_pre0034 \
      --prefix motion_primitives_1x_car_d5 --shapes square wide tall
"""
import argparse
import struct
import math
import numpy as np

REC_SIZE = 14


def parse_dat(path):
    with open(path, "rb") as f:
        b = f.read()
    n = struct.unpack("<I", b[:4])[0]
    body = b[4:]
    if len(body) != n * REC_SIZE:
        raise ValueError(f"{path}: body {len(body)} != {n}*{REC_SIZE}")
    recs = {}
    for i in range(n):
        off = i * REC_SIZE
        dx, dy, dth = struct.unpack("<fff", body[off:off + 12])
        edge_idx = body[off + 12]
        push_steps = body[off + 13]
        recs[(edge_idx, push_steps)] = (dx, dy, dth)
    return n, recs


def deg(rad):
    return rad * 180.0 / math.pi


def analyze_shape(shape, new_path, old_path):
    n_new, new = parse_dat(new_path)
    n_old, old = parse_dat(old_path)
    keys = sorted(set(new) & set(old))
    only_new = set(new) - set(old)
    only_old = set(old) - set(new)

    diffs = []  # (key, dx_old, dy_old, dth_old, dx_new, dy_new, dth_new)
    nonfinite = []
    degenerate = []  # all-zero se2_target
    for k in keys:
        on = old[k]
        nn = new[k]
        if not all(math.isfinite(v) for v in nn):
            nonfinite.append((k, "new", nn))
        if not all(math.isfinite(v) for v in on):
            nonfinite.append((k, "old", on))
        # degenerate = essentially no displacement at deepest push (a real
        # push at depth>=2 should move a ~7cm object by mm at least)
        if k[1] >= 2 and abs(nn[0]) < 1e-6 and abs(nn[1]) < 1e-6 and abs(nn[2]) < 1e-6:
            degenerate.append((k, nn))
        diffs.append((k, on, nn))

    ddx = np.array([d[2][0] - d[1][0] for d in diffs])
    ddy = np.array([d[2][1] - d[1][1] for d in diffs])
    ddth = np.array([d[2][2] - d[1][2] for d in diffs])
    ddth_deg = np.array([deg(d[2][2] - d[1][2]) for d in diffs])
    keys_arr = [d[0] for d in diffs]

    # positional shift magnitude (xy euclidean) and angular
    dpos = np.hypot(ddx, ddy)

    # fraction past thresholds: position OR angle past bound
    frac_1mm1deg = np.mean((dpos > 0.001) | (np.abs(ddth_deg) > 1.0))
    frac_5mm5deg = np.mean((dpos > 0.005) | (np.abs(ddth_deg) > 5.0))

    res = {
        "shape": shape,
        "n_new": n_new, "n_old": n_old,
        "n_matched": len(keys),
        "only_new": sorted(only_new), "only_old": sorted(only_old),
        "mean_abs_dx_mm": np.mean(np.abs(ddx)) * 1000,
        "mean_abs_dy_mm": np.mean(np.abs(ddy)) * 1000,
        "mean_abs_dth_deg": np.mean(np.abs(ddth_deg)),
        "max_abs_dx_mm": np.max(np.abs(ddx)) * 1000,
        "max_abs_dy_mm": np.max(np.abs(ddy)) * 1000,
        "max_abs_dth_deg": np.max(np.abs(ddth_deg)),
        "mean_pos_shift_mm": np.mean(dpos) * 1000,
        "max_pos_shift_mm": np.max(dpos) * 1000,
        "frac_1mm1deg": frac_1mm1deg,
        "frac_5mm5deg": frac_5mm5deg,
        "nonfinite": nonfinite,
        "degenerate": degenerate,
    }

    # top movers by combined metric (pos shift in mm + angle in deg)
    combined = dpos * 1000 + np.abs(ddth_deg)
    order = np.argsort(-combined)[:5]
    res["top_movers"] = [
        (keys_arr[i], dpos[i] * 1000, ddth_deg[i],
         diffs[i][1], diffs[i][2]) for i in order
    ]
    # also report per-depth mean pos shift
    depth_means = {}
    for d in (1, 2, 3, 4, 5):
        idx = [i for i, k in enumerate(keys_arr) if k[1] == d]
        if idx:
            depth_means[d] = (np.mean(dpos[idx]) * 1000,
                              np.mean(np.abs(ddth_deg[idx])))
    res["depth_means"] = depth_means
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", default="data")
    ap.add_argument("--old-dir", default="data/_primitive_backup_pre0034")
    ap.add_argument("--prefix", default="motion_primitives_1x_car_d5")
    ap.add_argument("--shapes", nargs="+", default=["square", "wide", "tall"])
    args = ap.parse_args()

    print(f"{'shape':8} {'matched':>7} | "
          f"{'mean|dx|':>8} {'mean|dy|':>8} {'mean|dth|':>9} | "
          f"{'max|dx|':>8} {'max|dy|':>8} {'max|dth|':>9} | "
          f"{'meanPos':>8} {'maxPos':>8} | {'>1mm/1deg':>9} {'>5mm/5deg':>9}")
    print(f"{'':8} {'':>7} | {'(mm)':>8} {'(mm)':>8} {'(deg)':>9} | "
          f"{'(mm)':>8} {'(mm)':>8} {'(deg)':>9} | {'(mm)':>8} {'(mm)':>8} | "
          f"{'frac':>9} {'frac':>9}")
    print("-" * 130)
    all_res = []
    for shape in args.shapes:
        new_path = f"{args.new_dir}/{args.prefix}_{shape}.dat"
        old_path = f"{args.old_dir}/{args.prefix}_{shape}.dat"
        r = analyze_shape(shape, new_path, old_path)
        all_res.append(r)
        print(f"{r['shape']:8} {r['n_matched']:>7} | "
              f"{r['mean_abs_dx_mm']:>8.3f} {r['mean_abs_dy_mm']:>8.3f} "
              f"{r['mean_abs_dth_deg']:>9.3f} | "
              f"{r['max_abs_dx_mm']:>8.3f} {r['max_abs_dy_mm']:>8.3f} "
              f"{r['max_abs_dth_deg']:>9.3f} | "
              f"{r['mean_pos_shift_mm']:>8.3f} {r['max_pos_shift_mm']:>8.3f} | "
              f"{r['frac_1mm1deg']:>9.3f} {r['frac_5mm5deg']:>9.3f}")

    print("\n=== stability / sanity ===")
    any_bad = False
    for r in all_res:
        if r["only_new"] or r["only_old"]:
            any_bad = True
            print(f"[{r['shape']}] KEY MISMATCH only_new={r['only_new']} only_old={r['only_old']}")
        if r["nonfinite"]:
            any_bad = True
            print(f"[{r['shape']}] NON-FINITE primitives: {r['nonfinite']}")
        if r["degenerate"]:
            any_bad = True
            print(f"[{r['shape']}] DEGENERATE (zero) primitives at depth>=2: {r['degenerate']}")
    if not any_bad:
        print("OK: all shapes matched key-for-key, all se2_target finite, "
              "no degenerate-zero primitives at depth>=2.")

    print("\n=== top movers per shape (edge,depth): posShift_mm, dth_deg | old(dx,dy,dth) -> new ===")
    for r in all_res:
        print(f"[{r['shape']}]")
        for (k, pos_mm, dth_d, old_v, new_v) in r["top_movers"]:
            print(f"  edge={k[0]:>2} depth={k[1]} : pos {pos_mm:7.2f} mm, "
                  f"dth {dth_d:7.2f} deg | "
                  f"({old_v[0]:+.4f},{old_v[1]:+.4f},{old_v[2]:+.4f}) -> "
                  f"({new_v[0]:+.4f},{new_v[1]:+.4f},{new_v[2]:+.4f})")
        dm = r["depth_means"]
        dm_str = "  ".join(f"d{d}:{v[0]:.2f}mm/{v[1]:.2f}deg" for d, v in dm.items())
        print(f"  per-depth mean shift: {dm_str}")


if __name__ == "__main__":
    main()
