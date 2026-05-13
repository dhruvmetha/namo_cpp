"""Slice a NAMO_QPOS_DUMP by primitive boundaries (detected as large car
xy teleport jumps between consecutive ticks) and report per-primitive
peak chassis pitch.

Outputs per-primitive table + a stacked plot of the first N primitives.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


def quat_pitch(w, x, y, z):
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)


def quat_yaw(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("qpos_dump")
    ap.add_argument("--teleport-threshold-mm", type=float, default=20.0,
                    help="A car xy jump > this between ticks marks a new primitive.")
    ap.add_argument("--plot-first", type=int, default=8,
                    help="Plot the first N primitives stacked.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    car_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_freejoint")
    car_q0 = int(model.jnt_qposadr[car_jnt])

    xs, ys, zs, pitches, yaws = [], [], [], [], []
    with open(args.qpos_dump) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ":" in ln:
                ph, qstr = ln.split(":", 1)
                qs = np.fromstring(qstr, sep=" ", dtype=float)
            else:
                vals = np.fromstring(ln, sep=" ", dtype=float)
                if vals.size < 2:
                    continue
                nq = int(vals[1])
                qs = vals[2:2 + nq]
            if qs.size < car_q0 + 7:
                continue
            xs.append(qs[car_q0]); ys.append(qs[car_q0 + 1]); zs.append(qs[car_q0 + 2] * 1000)
            pitches.append(math.degrees(quat_pitch(qs[car_q0 + 3], qs[car_q0 + 4], qs[car_q0 + 5], qs[car_q0 + 6])))
            yaws.append(math.degrees(quat_yaw(qs[car_q0 + 3], qs[car_q0 + 4], qs[car_q0 + 5], qs[car_q0 + 6])))

    xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
    pitches = np.array(pitches); yaws = np.array(yaws)
    print(f"loaded {len(xs)} ticks ({len(xs)*0.002:.1f}s)")

    # Detect primitive boundaries from car xy teleport jumps
    dx = np.diff(xs); dy = np.diff(ys)
    jump_mm = np.sqrt(dx*dx + dy*dy) * 1000
    boundaries = [0] + (np.where(jump_mm > args.teleport_threshold_mm)[0] + 1).tolist() + [len(xs)]
    print(f"found {len(boundaries)-1} primitive segments "
          f"(teleport threshold = {args.teleport_threshold_mm}mm)")

    # Per-segment peak pitch
    print()
    print(f"{'idx':>4}  {'len':>6}  {'peak |pitch|':>12}  {'peak |z-25|':>11}  {'final-init pitch':>16}")
    print(f"{'---':>4}  {'---':>6}  {'-----------':>12}  {'----------':>11}  {'---------------':>16}")
    seg_data = []
    for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        seg_p = pitches[s:e]
        seg_z = zs[s:e]
        peak_p = float(np.max(np.abs(seg_p)))
        peak_z = float(np.max(np.abs(seg_z - 25.0)))   # rest height ~25mm
        # Where in the segment does the peak occur (% of segment)?
        argmax_pct = 100.0 * np.argmax(np.abs(seg_p)) / max(1, len(seg_p))
        seg_data.append((i, s, e, peak_p, peak_z, argmax_pct, seg_p, seg_z))
        if i < 30:
            print(f"{i:>4}  {e - s:>6}  {peak_p:>10.2f}°  {peak_z:>9.2f}mm  "
                  f"peak@{argmax_pct:>4.0f}% of seg")
    if len(seg_data) > 30:
        print(f"... ({len(seg_data) - 30} more segments)")

    # Aggregate
    peaks = np.array([s[3] for s in seg_data])
    print()
    print(f"Across all {len(seg_data)} primitives:")
    print(f"  mean peak pitch = {peaks.mean():.2f}°  median = {np.median(peaks):.2f}°  "
          f"max = {peaks.max():.2f}°")
    print(f"  fraction with peak > 10°: {(peaks > 10).sum()}/{len(peaks)} "
          f"= {100*(peaks > 10).mean():.0f}%")
    print(f"  fraction with peak > 20°: {(peaks > 20).sum()}/{len(peaks)} "
          f"= {100*(peaks > 20).mean():.0f}%")

    # Plot first N primitives
    n_plot = min(args.plot_first, len(seg_data))
    fig, axes = plt.subplots(n_plot, 1, figsize=(13, 1.5 * n_plot), sharex=False)
    if n_plot == 1:
        axes = [axes]
    for ax, (i, s, e, peak_p, peak_z, argmax_pct, seg_p, seg_z) in zip(axes, seg_data[:n_plot]):
        t = np.arange(len(seg_p)) * 0.002
        ax.plot(t, seg_p, lw=0.7, color="#222")
        ax.axhline(0, color="#888", lw=0.4)
        ax.set_ylabel(f"#{i}\npitch (°)", fontsize=8)
        ax.set_ylim(-45, 45)
        ax.grid(alpha=0.3)
        ax.set_title(f"primitive #{i}  ticks={e-s}  peak |pitch|={peak_p:.1f}°  peak@{argmax_pct:.0f}%",
                     fontsize=9, loc="left")
    axes[-1].set_xlabel("sim time within primitive (s)")
    plt.tight_layout()
    out = Path(args.output) if args.output else Path(args.qpos_dump).with_suffix(".per_prim.png")
    plt.savefig(out, dpi=110)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
